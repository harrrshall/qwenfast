"""Continuous-batching scheduler for the qwenfast runtime.

Pure, single-threaded engine-loop logic: request admission, SSM-slot + KV-page
allocation, chunked-prefill token budget accounting, prefill/decode
interleaving, preemption on page exhaustion, and per-request stop conditions.
Nothing here touches asyncio, HTTP, or tokenization -- :mod:`.engine` is the
thin adapter that drives this against the ``AsyncEngine`` contract.

State machine per request, simplified to three states because
"admitted but still prefilling" and "waiting to be admitted" only differ by
whether ``slot is not None``, which is already tracked:

    WAITING (slot is None)  --admit-->  WAITING (slot allocated, mid-prefill)
                                          --finishes prefill-->  DECODING
    DECODING  --stop condition-->  DONE
    DECODING  --page OOM-->  (preempted) WAITING, at the front of the queue

**Admission**: a request is admitted (allocated an SSM slot + reserved
KV pages) iff a free slot exists and
``free_kv_pages >= ceil(len(prompt)/page_size) + ceil(max_tokens/page_size)``
-- the second term is the "watermark" that reserves room for the request's
own decode tail so it is unlikely to need preemption later.

**Chunked prefill**: one call to :meth:`Scheduler.step` either runs a
*varlen packed* prefill chunk (up to ``max_num_batched_tokens`` tokens,
across as many waiting requests as fit) or one decode step over every
running request -- never both, matching "a step is never mixed".
``RuntimeConfig.prefill_decode_ratio`` (default 4) bounds how many decode
steps run between prefill opportunities, trading TTFT against TPOT.

**Mixed prefill+decode, ``RuntimeConfig.mixed_forward``**: that "never both"
is a design choice, not a law, and at concurrency >= 64 it is the binding constraint
on served throughput. With ``mixed_forward`` on, one :meth:`Scheduler.step`
runs a **single** forward over ``[prefill chunk tokens ‖ one row per running
sequence]`` (:meth:`Scheduler._run_mixed_step`), so the decode rows' GEMMs are
the prefill chunk's GEMMs: at conc 256 the separate-step design is bounded at
~2,120 out tok/s by "42.7 ms decode + 78 ms prefill", while the mixed step is
~97 ms for the same work. The decode-only
graphed step and the speculative step both remain, for the steps where they
are the better shape -- see :meth:`Scheduler._should_mix`.

**Preemption**: "swap, not recompute" -- a hybrid model's SSM state
cannot be cheaply recomputed by replaying the prompt (that would replay the
*whole* prompt, not just the KV window an attention-only model would need).
:meth:`Scheduler._preempt` copies the victim's SSM/conv state rows and
committed KV to pinned-free host tensors, frees its slot and pages, and
requeues it at the *front* of the waiting queue; :meth:`Scheduler
._restore_swapped` reverses this on re-admission without re-running prefill
(the request's ``num_computed_tokens`` already reflects its full prompt +
partial decode, so it goes straight back to ``DECODING``). Preemption policy
is last-admitted-first, tracked via ``Request.admitted_at``.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import torch

from .prefix_cache import PrefixCache
from .fused_model import (
    FusedQwenForCausalLM,
    RuntimeConfig,
    make_mixed_batch,
    make_prefill_batch,
)
from .graphs import GraphedDecoder, sample_tokens
from .mixed_graphs import (
    PAD_TOKEN_ID,
    MixedGraphRunner,
    MixedPadSpec,
    pad_mixed_step,
    reset_scratch_state,
)
from .spec_decode import SpecDecoder

STATUS_WAITING = "WAITING"
STATUS_DECODING = "DECODING"
STATUS_DONE = "DONE"


def context_length_error(n_prompt: int, max_tokens: int, max_context_len: int) -> Optional[str]:
    """``None`` if ``prompt + completion`` fits, else the OpenAI-shaped message.

    One helper, used by both the HTTP layer (-> 400,
    matching what vLLM returns) and the engine adapter, so the limit that is
    *advertised* and the limit that is *enforced* can never drift apart.
    """
    completion = max(int(max_tokens), 0)
    total = int(n_prompt) + completion
    if int(n_prompt) <= max_context_len and total <= max_context_len:
        return None
    return (
        f"This model's maximum context length is {max_context_len} tokens. "
        f"However, you requested {total} tokens ({int(n_prompt)} in the prompt, "
        f"{completion} in the completion). Please reduce the length of the "
        f"prompt or completion."
    )


# =========================================================================== #
# 1. request / params
# =========================================================================== #
@dataclass
class GenParams:
    """The scheduler's own sampling-params shape -- deliberately independent
    of ``server.engine_api.SamplingParams`` so this module has zero import
    dependency on the server package; :mod:`.engine` adapts one to the
    other."""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0  # <= 0 == disabled, matches DeviceBuffers' convention
    max_tokens: int = 16
    ignore_eos: bool = False
    stop_token_ids: Tuple[int, ...] = ()
    eos_token_id: Optional[int] = None


@dataclass
class Request:
    request_id: str
    prompt_token_ids: List[int]
    params: GenParams

    slot: Optional[int] = None
    num_computed_tokens: int = 0  # tokens whose KV/SSM state is committed
    output_token_ids: List[int] = field(default_factory=list)
    last_token: Optional[int] = None
    #: Asynchronous scheduling only.  Tokens whose *sampling
    #: has been launched* on the device but whose value the host has not read
    #: back yet.  ``len(output_token_ids) + pending_tokens`` is therefore the
    #: request's real length as far as admission and ``max_tokens`` are
    #: concerned; ``output_token_ids`` alone is one step behind.  Always 0 on
    #: the synchronous path, which is what makes every gate below a no-op
    #: there.
    pending_tokens: int = 0
    #: ``(generation, row)``: where in the *previous* step's sampled-token
    #: tensor this request's next input token lives.  The decode step feeds it
    #: to the model with a device-side gather -- no D2H on the critical path --
    #: which is the whole point: the sampled token of step N is an *input* of
    #: step N+1, and vLLM's async scheduling avoids the host round trip by
    #: never bringing it to the host in the first place.  ``None`` means
    #: ``last_token`` is authoritative (synchronous path, or the step after a
    #: pipeline drain).
    pending_src: Optional[Tuple[int, int]] = None
    status: str = STATUS_WAITING
    finish_reason: Optional[str] = None
    admitted_at: Optional[float] = None
    created_at: float = field(default_factory=time.monotonic)
    first_token_at: Optional[float] = None

    @property
    def is_finished(self) -> bool:
        return self.status == STATUS_DONE

    @property
    def ttft_s(self) -> Optional[float]:
        if self.first_token_at is None:
            return None
        return self.first_token_at - self.created_at


@dataclass
class StepEvent:
    """One request's delta for a single :meth:`Scheduler.step` call."""

    request: Request
    new_token_ids: List[int]
    finished: bool
    finish_reason: Optional[str]


class _NullEvent:
    """The CPU stand-in for ``torch.cuda.Event`` (asynchronous scheduling).

    A CPU build has no stream, so "the device has caught up" is trivially
    true and both methods are no-ops. Its existence is what lets the
    asynchronous scheduler -- and therefore its one-step-late stop conditions,
    which is where every correctness question lives -- run off a GPU.
    """

    def record(self) -> None:  # noqa: D102
        pass

    def synchronize(self) -> None:  # noqa: D102
        pass


@dataclass
class _PendingStep:
    """One launched-but-not-harvested step (asynchronous scheduling).

    ``rows`` is ``(request, row, is_first_token)`` for every row the step
    sampled, ``row`` indexing both the device token buffer and ``host``.  The
    event is recorded on the stream immediately after the D2H copy, so
    ``event.synchronize()`` waits for *this* step and nothing later -- by the
    time the host reaches it, the next step is already queued behind it and
    the device does not go idle while the host commits.
    """

    rows: List[Tuple["Request", int, bool]]
    event: object
    host: torch.Tensor
    n: int
    gen: int


@dataclass
class _SwapState:
    ssm: torch.Tensor
    conv: torch.Tensor
    kv: List[Tuple[torch.Tensor, torch.Tensor]]
    length: int
    # The MTP head's `h_prev` carry for this slot. A preempted request
    # resumes straight into DECODING (no prompt replay), so the carry has to
    # survive the swap or the first draft after a restore is conditioned on
    # whatever sequence took the slot in the meantime.
    mtp_hidden: Optional[torch.Tensor] = None


@dataclass
class SchedulerStats:
    num_running: int
    num_waiting: int
    ssm_slots_used: int
    ssm_slots_total: int
    kv_pages_used: int
    kv_pages_total: int
    # Speculative decoding telemetry. 0.0 when spec decoding is off.
    spec_accept_length: float = 0.0
    spec_acceptance_rate: float = 0.0
    # Prefix cache telemetry (all 0 when the cache is off).
    prefix_hits: int = 0
    prefix_lookups: int = 0
    prefix_hit_tokens: int = 0
    prefix_prompt_tokens: int = 0
    prefix_entries: int = 0


# =========================================================================== #
# 2. slot allocator
# =========================================================================== #
class SlotManager:
    """Free-list allocator over ``[0, n_slots)`` SSM/KV slot ids.

    This is the *single* allocator that owns ``slot_ids[B]``. The KV pool's
    own free list is disabled by ``fused_model._claim_all_kv_slots`` precisely
    so there is exactly one source of truth
    for "which slot is this request in", shared by the SSM state pool and
    the KV page table.
    """

    def __init__(self, n_slots: int):
        self.n_slots = n_slots
        self._free: Deque[int] = deque(range(n_slots))

    @property
    def num_free(self) -> int:
        return len(self._free)

    def alloc(self) -> int:
        if not self._free:
            raise RuntimeError("SlotManager: no free slots")
        return self._free.popleft()

    def free(self, slot: int) -> None:
        self._free.append(slot)


# =========================================================================== #
# 3. the scheduler
# =========================================================================== #
class Scheduler:
    def __init__(
        self,
        model: FusedQwenForCausalLM,
        decoder: GraphedDecoder,
        rt: RuntimeConfig,
        *,
        max_num_batched_tokens: Optional[int] = None,
        prefill_decode_ratio: Optional[int] = None,
        prefill_chunk_tokens: Optional[int] = None,
        spec: Optional[SpecDecoder] = None,
        spec_max_batch: Optional[int] = None,
        mixed_forward: Optional[bool] = None,
        mixed_graphs: Optional[bool] = None,
        mixed_runner: Optional[MixedGraphRunner] = None,
        overlap: Optional[bool] = None,
        async_scheduling: Optional[bool] = None,
    ):
        self.model = model
        self.decoder = decoder
        # When set, every eligible decode step runs the speculative path
        # instead: k MTP drafts + one windowed verify, emitting 1..k+1 tokens
        # per sequence. `None` gives plain (non-speculative) decoding.
        self.spec = spec
        # The per-step batch-size policy, `--spec-max-batch N`: spec decoding
        # wins at small batch (B<=8) and
        # loses at large batch (the doubled window-kernel pass + the M=B*(k+1)
        # verify GEMM dominate at B>=128), so a step whose live decode batch
        # exceeds this uses the plain graphed step instead, even though `spec`
        # is configured. `None` (the direct-`SpecDecoder`-test default, and
        # every existing caller that does not pass this) means "no cap" --
        # spec runs at any batch size `spec.eligible` allows, the behaviour
        # the direct scheduler tests pin.
        self.spec_max_batch = spec_max_batch
        self.buf = decoder.buf
        self.rt = rt
        self.device = model.device
        self.max_num_batched_tokens = max_num_batched_tokens or rt.max_num_batched_tokens
        self.prefill_decode_ratio = prefill_decode_ratio or rt.prefill_decode_ratio
        # The chunk-size cap is deliberately a *separate*
        # number from `max_num_batched_tokens`: the latter is also the prefill
        # activation-memory bound that `serve.plan_memory` sizes the headroom
        # from, so shrinking it to smooth TPOT would silently
        # shrink the memory plan too and invalidate the printed budget.
        cap = int(prefill_chunk_tokens if prefill_chunk_tokens is not None
                  else rt.prefill_chunk_tokens)
        self.prefill_chunk_tokens = (
            min(cap, self.max_num_batched_tokens) if cap > 0 else self.max_num_batched_tokens
        )
        #: Pinned host staging for the decode step's token harvest. `out[:B]
        #: .to("cpu")` allocates a fresh pageable tensor and copies through a
        #: bounce buffer every step; a pinned destination is a straight DMA
        #: into memory the driver already has registered. Lazily
        #: built because CPU-only test builds have neither pinned memory nor a
        #: reason to want it.
        self._out_host: Optional[torch.Tensor] = None
        #: Seconds spent inside ``_harvest``'s
        #: ``synchronize()`` -- i.e. waiting for the step's GPU work -- split
        #: out from the pinned copy itself, which is ~0.15 ms. Read by
        #: ``profile_serving``'s ledger; nothing in the scheduler reads it.
        self.harvest_gpu_wait_s = 0.0
        self.harvest_calls = 0
        self.page_size = model.kv_pool.cfg.page_size
        # The hard position bound. See
        # `FusedQwenForCausalLM.max_context_len` for why it is the *minimum* of
        # the rotary table and the page-table geometry rather than
        # `rt.max_model_len`. Nothing in this class may ever hand the model a
        # position >= this value: `RotaryTable.lookup` is an unguarded gather,
        # so one out-of-range position is a device-side assert that takes the
        # whole CUDA context (and therefore the engine thread) down.
        self.max_context_len = int(model.max_context_len)

        self.waiting: Deque[Request] = deque()
        self.running: Dict[int, Request] = {}
        self.slots = SlotManager(model.n_slots)
        self._swapped: Dict[str, _SwapState] = {}
        #: Turn to turn prefix cache (``prefix_cache.py``); ``None`` keeps
        #: every code path byte identical to the cache-less scheduler.
        self.prefix_cache: Optional[PrefixCache] = None
        n_prefix = int(getattr(rt, "prefix_cache_entries", 0) or 0)
        if n_prefix > 0:
            self.prefix_cache = PrefixCache(
                model, spec, n_entries=n_prefix,
                min_tokens=int(getattr(rt, "prefix_cache_min_tokens", 512) or 512),
            )
        self._aborted: set = set()
        self._decode_steps_since_prefill = 0
        self._prefill_progressed = False  # set fresh at the top of every _run_prefill_step
        # Chunk telemetry. The shape of the chunk **this** step
        # ran (0 when it was a decode step), so a profiler can label a step
        # without reconstructing the budget arithmetic. Two ints, cleared at
        # the top of `step()`; nothing reads them on the serving path.
        self.last_chunk_tokens = 0
        self.last_chunk_seqs = 0
        self.last_step_progressed = True  # see step()'s docstring
        # `True` when the step that just ran was a *mixed*
        # forward (chunk + decode rows in one pass) rather than one or the
        # other; `last_mixed_decode_rows` is how many decode rows rode along.
        # Cleared at the top of `step()`, exactly like `last_chunk_tokens`, so
        # a profiler can label a step without reconstructing the policy.
        self.last_step_mixed = False
        self.last_mixed_decode_rows = 0
        self._mixed_progressed = False
        #: vLLM-style per-step budget: every step takes up to
        #: `prefill_chunk_tokens` prefill tokens **plus** every running decode
        #: row, in one forward. `None` -> `rt.mixed_forward`. When on,
        #: `prefill_decode_ratio` no longer gates anything (there is nothing to
        #: interleave), and `prefill_chunk_tokens` becomes a pure TPOT/p99 knob
        #: -- see `_run_mixed_step`.
        self.mixed_forward = bool(
            rt.mixed_forward if mixed_forward is None else mixed_forward
        )
        # When on, every mixed step is **padded** to one
        # fixed shape -- `prefill_chunk_tokens` prefill tokens in exactly
        # `mixed_graph_segments` segments, decode rows padded to a graph bucket
        # -- so it can be replayed from a captured (segmented) CUDA graph
        # instead of launched op by op. The padding is applied whether or not a
        # runner is attached: with one it is the graph's shape, without one
        # (CPU, `--no-graphs`) it is the same arithmetic run eagerly, which is
        # what makes "graphed == eager, token for token" testable off a GPU.
        self.last_step_mixed_graphed = False
        self.mixed_runner = mixed_runner
        # `--overlap`: instead of one row-concatenated forward,
        # run the prefill chunk's graph and the decode step's graph
        # **concurrently on two streams**. The two write disjoint slots by
        # construction (a request is either prefilling or decoding, which
        # `_collect_prefill_chunk` enforces structurally), and under
        # `overlap_streams` they also get disjoint graph mempools and disjoint
        # FlashInfer workspaces -- the three things that make "at the same
        # time" mean the same thing as "one after the other".
        self.overlap = bool(overlap if overlap is not None else rt.overlap_streams)
        self.overlap_min_fill = float(getattr(rt, "overlap_min_fill", 0.0) or 0.0)
        self._overlap_stream = None
        if self.overlap and self.device.type == "cuda" and torch.cuda.is_available():
            self._overlap_stream = torch.cuda.Stream(
                priority=int(getattr(rt, "overlap_decode_priority", 0) or 0)
            )
        #: Overlap telemetry, mirroring `last_step_mixed`: the step that just ran
        #: was an overlapped one (two graphs, two streams) rather than a fused
        #: mixed forward.
        self.last_step_overlapped = False
        want_pad = bool(rt.mixed_graphs if mixed_graphs is None else mixed_graphs)
        self.mixed_pad: Optional[MixedPadSpec] = None
        if self.mixed_forward and (want_pad or mixed_runner is not None):
            self.mixed_pad = (
                mixed_runner.spec if mixed_runner is not None
                else MixedPadSpec(
                    chunk_tokens=self.prefill_chunk_tokens,
                    n_segments=int(rt.mixed_graph_segments),
                    buckets=tuple(rt.buckets_for()),
                    scratch_slot=model.scratch_slot,
                    # A pad segment's positions run 0..m-1, so it cannot
                    # be longer than the rotary table.
                    max_pad_len=int(getattr(rt, "max_model_len", 0) or 0),
                )
            )
            if (mixed_runner is not None
                    and mixed_runner.chunk_tokens != self.prefill_chunk_tokens):
                raise ValueError(
                    f"mixed_runner captured chunk_tokens={mixed_runner.chunk_tokens} but this "
                    f"scheduler packs {self.prefill_chunk_tokens}-token chunks -- a graphed "
                    "mixed step's prefill half is a fixed shape, so the two must agree"
                )
            if self.mixed_pad.budget < 1 or self.mixed_pad.max_segments < 1:
                raise ValueError(
                    f"mixed_graphs: prefill_chunk_tokens={self.prefill_chunk_tokens} is too "
                    f"small for mixed_graph_segments={rt.mixed_graph_segments} "
                    "(every padding segment needs at least one token)"
                )
            # A padded query row has to have `kv_len >= q_len` real keys to
            # attend to (FlashInfer's bottom-right causal alignment), so the
            # scratch slot needs pages for the longest padding *segment*. A pad
            # segment is capped at `max_pad_len` and the padding is split across
            # the free plan rows, so it is 172 pages at chunk 8,192 rather than
            # the 512 a single whole-chunk pad segment would need.
            # Reserved once, here, so no step ever allocates pages for padding
            # -- `MixedGraphRunner.prepare` does the same thing and both are
            # idempotent.
            model.kv_pool.ensure_capacity(
                model.scratch_slot,
                min(self.mixed_pad.chunk_tokens, self.mixed_pad.pad_cap),
            )

        # -- asynchronous step scheduling ------------------------------------ #
        # While step N runs on the device, the host schedules, allocates and
        # plans step N+1 and launches it; step N's tokens are harvested one
        # step later through a pinned buffer and a CUDA event.  See `step()`.
        self.async_scheduling = bool(
            rt.async_scheduling if async_scheduling is None else async_scheduling
        )
        #: Deliberately **not** gated on CUDA. On a CPU build there is no
        #: stream to overlap with, so the mode buys nothing -- but every
        #: host-side consequence of it is the same (deferred stop conditions,
        #: the one-step-late EOS trim, the device-side token feed, the
        #: ``pending_tokens`` admission gates), and running it there is what
        #: makes "async output == sync output, token for token" a test that
        #: needs no GPU. The CUDA event becomes a no-op :class:`_NullEvent`.
        self._async_ok = self.async_scheduling
        self._async_active = False
        self._gen = 0
        self._pending: Optional[_PendingStep] = None
        self._launch_rows: List[Tuple[Request, int, bool]] = []
        self._launch_n = 0
        #: ``[cap]`` int32 on the device: the tokens the *last launched* step
        #: sampled, in row order.  Written at the end of every launch and read
        #: by the next launch's gather -- both on the same stream, so the read
        #: is ordered before the write that follows it and one buffer suffices.
        self._tok_dev: Optional[torch.Tensor] = None
        #: Two pinned host buffers and two events, alternating by generation:
        #: the D2H of step N must not land in the buffer step N-1 is still
        #: being read out of.
        self._tok_host: List[torch.Tensor] = []
        self._tok_events: List[object] = []
        self._idx_host: Optional[torch.Tensor] = None
        self._idx_dev: Optional[torch.Tensor] = None
        #: Async telemetry: seconds spent in the deferred harvest's
        #: ``event.synchronize()``.  On a healthy async pipeline this is the
        #: number that goes to ~0 -- the device is already past step N-1 by
        #: the time the host asks.
        self.async_wait_s = 0.0
        self.async_commits = 0
        self.async_steps = 0
        self.sync_steps = 0
        #: Optional :class:`~.step_trace.StepProfiler`.  ``None`` (the default)
        #: makes every ``_mark`` below a single attribute load.
        self.profiler = None

    # -- admission ------------------------------------------------------- #
    def add_request(self, req: Request) -> None:
        self.waiting.append(req)

    def abort(self, request_id: str) -> None:
        self._aborted.add(request_id)

    def has_work(self) -> bool:
        # A launched-but-unharvested step is work even when nothing is
        # running or waiting -- the last request's last token lives in it, and
        # a loop that slept here instead of stepping would never emit it.
        return bool(self.waiting) or bool(self.running) or self._pending is not None

    def context_length_error(self, n_prompt: int, max_tokens: int) -> Optional[str]:
        """``None`` if the request fits this engine's context, else why not.

        The *reject* half of "reject or truncate". The
        HTTP layer calls this before admitting anything, so an over-long
        request is a 400 rather than a dead engine; `_run_prefill_step` and
        `_run_decode_step` below are the truncate half, and exist so that a
        caller that skips this check still cannot make the model index out of
        range.
        """
        return context_length_error(n_prompt, max_tokens, self.max_context_len)

    def stats(self) -> SchedulerStats:
        return SchedulerStats(
            num_running=len(self.running),
            num_waiting=len(self.waiting),
            ssm_slots_used=self.model.n_slots - self.slots.num_free,
            ssm_slots_total=self.model.n_slots,
            kv_pages_used=self.model.kv_pool.cfg.n_pages - self.model.kv_pool.num_free_pages,
            kv_pages_total=self.model.kv_pool.cfg.n_pages,
            spec_accept_length=(self.spec.stats()["spec_accept_length"] if self.spec else 0.0),
            spec_acceptance_rate=(self.spec.stats()["spec_acceptance_rate"] if self.spec else 0.0),
            **self._prefix_stats(),
        )

    def _prefix_stats(self) -> Dict[str, int]:
        if self.prefix_cache is None:
            return {}
        st = self.prefix_cache.snapshot_stats()
        return dict(
            prefix_hits=st.hits, prefix_lookups=st.lookups, prefix_hit_tokens=st.hit_tokens,
            prefix_prompt_tokens=st.prompt_tokens, prefix_entries=st.entries,
        )

    def _watermark_pages(self, remaining_tokens: int) -> int:
        return max(remaining_tokens + self.page_size - 1, 0) // self.page_size

    def _emitted_or_pending(self, req: Request) -> int:
        """How many output tokens this request has *committed to*.

        ``len(output_token_ids)`` on the synchronous path; one more (or as
        many more as are in flight) under asynchronous scheduling, where the
        token exists on the device but the host has not read it. Every
        ``max_tokens``/watermark decision has to use this number, not the
        list's length, or a request would be scheduled for one extra step
        every time the pipeline is full."""
        return len(req.output_token_ids) + req.pending_tokens

    def _can_admit(self, req: Request) -> bool:
        if self.slots.num_free < 1:
            return False
        base = req.num_computed_tokens if req.num_computed_tokens > 0 else len(req.prompt_token_ids)
        remaining = max(req.params.max_tokens - self._emitted_or_pending(req), 0)
        need_pages = self.model.kv_pool.pages_needed(base) + self._watermark_pages(remaining)
        pc = self.prefix_cache
        if pc is not None and self.model.kv_pool.num_free_pages < need_pages:
            # cached prefixes are the first memory to give back: a hit brings
            # its own pages, every other entry may be evicted to make room
            hit = pc.lookup(req.prompt_token_ids, count=False) if req.num_computed_tokens == 0 else None
            if hit is not None:
                need_pages -= len(hit.pages)
            pc.reclaim_pages(need_pages, keep=hit)
        return self.model.kv_pool.num_free_pages >= need_pages

    # -- top-level step ---------------------------------------------------- #
    def step(self) -> List[StepEvent]:
        """One engine step: synchronous, or asynchronously scheduled.

        The synchronous shape is
        :meth:`_step_body`: build, plan, launch, **wait for the device**,
        harvest, do the bookkeeping, return this step's events.  The wait is
        the problem: the loop was measured 99.8 % inside this method with
        `nvidia-smi` at 69.8 %, i.e. ~30 points of wall clock in
        which the host is inside a step and the device is running nothing.
        That idle is the host half of the step -- the FlashInfer plan, the
        input build and upload, the page allocation, the per-request
        bookkeeping -- none of which needs the device, and all of which sits
        *between* two pieces of device work.

        Asynchronous scheduling removes it in the shape vLLM's does:

        1. **Launch step N.**  ``_step_body`` runs exactly as before except
           that its harvest does not synchronise: it publishes the sampled
           tokens into a device buffer, issues one D2H into pinned memory and
           records an event (:meth:`_finish_launch`).  No host work waits for
           the device.
        2. **Commit step N-1.**  Wait on *its* event -- which, because step N
           is already queued behind it, means the device carries straight on
           into step N while the host appends tokens, evaluates stop
           conditions and builds ``StepEvent``s.

        The one thing that makes this legal is the token feed.  The token
        sampled by step N is an *input* of step N+1, so a design that read it
        back on the host would have reintroduced the sync it just removed.
        Instead ``Request.pending_src`` records where the token lives on the
        device and :meth:`_apply_pending_tokens` gathers it there -- the host
        never sees it until one step later.

        **What that costs, precisely.**  Stop conditions are evaluated one
        step late, so a request whose step-N token is EOS has already had a
        step-N+1 row launched for it.  That row's token is computed and
        **discarded** (:meth:`_commit` skips rows whose request is already
        ``DONE``) -- at most one extra token per request, never emitted, which
        is exactly vLLM's accounting.  TTFT and every inter-token gap are
        reported one step later than they occur, and a step's worth of extra
        latency is the price of the throughput.

        **What is not asynchronous.**  A speculative step commits a
        *variable*, device-resident number of tokens per row, so the next
        step's FlashInfer plan (which is built on the host from
        ``num_computed_tokens``) cannot be built before that number is known.
        The pipeline is therefore **drained** before a spec step and refilled
        after it (:meth:`_needs_sync_step`, plus a belt-and-braces fallback to
        the plain path in :meth:`_run_decode_step`).  With
        ``--spec-max-batch 16`` that is the conc 1-16 regime only; every
        mixed, overlapped, prefill and plain-decode step is asynchronous.
        """
        prof = self.profiler
        if prof is not None:
            prof.begin()
        if not self._async_ok:
            self._async_active = False
            self.sync_steps += 1
            events = self._step_body()
            if prof is not None:
                self._profile_end(prof, events)
            return events

        events: List[StepEvent] = []
        if self._needs_sync_step():
            # A spec step is coming: drain, then run it exactly as the
            # synchronous path does.
            events += self.drain()
            self._async_active = False
            self.sync_steps += 1
            events += self._step_body()
            if events and not self.last_step_progressed:
                self.last_step_progressed = True
            if prof is not None:
                self._profile_end(prof, events)
            return events

        self._async_active = True
        self.async_steps += 1
        self._gen += 1
        self._launch_rows = []
        self._launch_n = 0
        body = self._step_body()
        launched = self._finish_launch()
        prev, self._pending = self._pending, launched
        self._mark("c0")
        if prev is not None:
            # Commit events first: they are older than anything `_step_body`
            # just produced, and a request finished here must not have its
            # token appear after its own "finished" event.
            events += self._commit(prev)
        self._mark("c2")
        events += body
        if events and not self.last_step_progressed:
            self.last_step_progressed = True
        if prof is not None:
            self._profile_end(prof, events)
        return events

    # -- profiling hooks (no-ops unless `self.profiler` is set) ------------- #
    def _mark(self, name: str) -> None:
        p = self.profiler
        if p is not None:
            p.mark(name)

    def _gpu_mark(self, name: str) -> None:
        p = self.profiler
        if p is not None:
            p.gpu_mark(name)

    def _profile_end(self, prof, events: List[StepEvent]) -> None:
        from .step_trace import classify_step

        prof.end(
            kind=classify_step(self, events),
            rows=int(self.last_mixed_decode_rows or len(self.running)),
            chunk_tokens=int(self.last_chunk_tokens or 0),
            tokens=sum(len(e.new_token_ids) for e in events),
        )

    # -- the asynchronous pipeline ----------------------------------------- #
    def _needs_sync_step(self) -> bool:
        """Would the coming step commit a device-resident number of tokens?

        Only the speculative step does (1..k+1 per row, decided on the
        device), and the next step's host-side FlashInfer plan needs that
        number.  ``_should_mix`` already yields the step to spec whenever spec
        is eligible, so this is the same predicate ``_run_decode_step`` uses,
        evaluated one level up where a drain is still possible.
        """
        if self.spec is None or not self.running:
            return False
        return self._use_spec(list(self.running.values()))

    def _ensure_async_buffers(self, need: int = 0) -> None:
        """Allocate (or grow) the token/index staging.

        Growth is safe wherever it is reached: the only reader of the *old*
        contents is :meth:`_apply_pending_tokens`, which runs before any
        :meth:`_publish` in the same step, and the pending step's tokens live
        in the pinned host buffer, not here.  It matters because a prefill
        chunk's segment count is bounded only by the token budget -- 8,192
        one-token prompts is a legal, if absurd, step.
        """
        if self._tok_dev is not None and self._tok_dev.numel() >= max(need, 1):
            return
        old_tok = self._tok_dev
        cap = max(int(self.model.n_slots) + 64, int(need) * 2)
        self._tok_host = []
        self._tok_events = []
        dev = self.device
        self._tok_dev = torch.zeros(cap, dtype=torch.int32, device=dev)
        if old_tok is not None:
            # Growth can happen *between* two publishes of the same step (a
            # mixed step publishes its prefill first tokens, then its decode
            # rows), so the rows already written have to survive it.
            self._tok_dev[: old_tok.numel()].copy_(old_tok)
        self._idx_dev = torch.zeros(cap, dtype=torch.int64, device=dev)
        cuda = dev.type == "cuda"
        for _ in range(2):
            try:
                self._tok_host.append(
                    torch.zeros(cap, dtype=torch.int32, pin_memory=True) if cuda
                    else torch.zeros(cap, dtype=torch.int32)
                )
            except (RuntimeError, NotImplementedError):  # pragma: no cover
                self._tok_host.append(torch.zeros(cap, dtype=torch.int32))
            self._tok_events.append(torch.cuda.Event() if cuda else _NullEvent())
        try:
            self._idx_host = (torch.zeros(cap, dtype=torch.int64, pin_memory=True) if cuda
                              else torch.zeros(cap, dtype=torch.int64))
        except (RuntimeError, NotImplementedError):  # pragma: no cover
            self._idx_host = torch.zeros(cap, dtype=torch.int64)

    def _publish(self, toks: torch.Tensor, reqs: Sequence[Request], *, first: bool) -> None:
        """Register ``len(reqs)`` sampled tokens of the step being launched.

        ``toks`` stays on the device: it is copied into ``_tok_dev`` (so the
        next step's gather has one fixed address to read, whatever tensor this
        step happened to sample into) and each request is pointed at its row.
        Nothing is read back here -- that is :meth:`_commit`'s job, one step
        later.
        """
        n = len(reqs)
        if n == 0:
            return
        base = self._launch_n
        self._ensure_async_buffers(base + n)
        self._tok_dev[base : base + n].copy_(toks[:n].to(torch.int32))
        gen = self._gen
        rows = self._launch_rows
        for i, req in enumerate(reqs):
            req.pending_tokens += 1
            req.pending_src = (gen, base + i)
            req.last_token = None
            if not first:
                req.num_computed_tokens += 1
            rows.append((req, base + i, first))
        self._launch_n = base + n

    def _finish_launch(self) -> Optional[_PendingStep]:
        """Close the launch: one D2H into pinned memory, one event, no wait."""
        n = self._launch_n
        if n == 0:
            return None
        parity = self._gen & 1
        host = self._tok_host[parity]
        ev = self._tok_events[parity]
        host[:n].copy_(self._tok_dev[:n], non_blocking=True)
        ev.record()
        return _PendingStep(rows=self._launch_rows, event=ev, host=host, n=n, gen=self._gen)

    def _commit(self, p: _PendingStep) -> List[StepEvent]:
        """Read one launched step's tokens back and do its bookkeeping.

        The only synchronisation on the asynchronous path, and by the time it
        is reached the *next* step is already queued behind the event, so the
        device does not idle through the Python that follows.
        """
        events: List[StepEvent] = []
        t0 = time.perf_counter()
        p.event.synchronize()
        self.async_wait_s += time.perf_counter() - t0
        self._mark("c1")
        self.async_commits += 1
        toks = p.host[: p.n].tolist()
        now = time.monotonic()
        # This loop runs once per decode row per step (256 at conc 256)
        # and is `commit_book` in the step profile, so the bound names are
        # hoisted out of it and the per-row work is kept to the minimum the
        # semantics need.  `pending_src` is compared component-wise rather
        # than against a freshly built `(gen, row)` tuple.
        gen = p.gen
        check_stop = self._check_stop
        finish = self._finish
        append = events.append
        max_ctx = self.max_context_len
        for req, row, first in p.rows:
            req.pending_tokens -= 1
            src = req.pending_src
            if src is not None and src[1] == row and src[0] == gen:
                # Only if no *newer* launch has claimed this request: with a
                # pipeline one step deep that happens whenever the request was
                # scheduled again before its previous token came back.
                req.pending_src = None
            if req.status == STATUS_DONE:
                # EOS / a stop id / an abort resolved one step ago, after this
                # row was already launched.  The token was computed; it is
                # discarded here and never emitted (vLLM does exactly this).
                continue
            tok = toks[row]
            out = req.output_token_ids
            out.append(tok)
            req.last_token = tok
            if first:
                req.first_token_at = now
            params = req.params
            # The common case -- not at the cap, not a stop token -- resolves
            # in two comparisons; `_check_stop` is still the single definition
            # of the rule and is called for everything else.
            if (
                len(out) >= params.max_tokens
                or req.num_computed_tokens >= max_ctx
                or tok == params.eos_token_id
                or tok in params.stop_token_ids
            ):
                finished, reason = check_stop(req, tok)
                if finished:
                    finish(req, reason)
            else:
                finished, reason = False, None
            append(StepEvent(req, [tok], finished, reason))
        return events

    def drain(self) -> List[StepEvent]:
        """Commit any launched-but-unharvested step.  Idempotent."""
        p, self._pending = self._pending, None
        return self._commit(p) if p is not None else []

    def _apply_pending_tokens(self, dst: torch.Tensor, reqs: Sequence[Request]) -> None:
        """Overwrite ``dst[:len(reqs)]`` with the previous step's tokens, on
        the device.

        This is the load-bearing half of asynchronous scheduling: without it
        the host would need step N's sampled tokens to build step N+1's input,
        which is the D2H the whole design exists to move off the critical
        path.  The gather is stream-ordered against both the replay that wrote
        ``_tok_dev`` and the replay that will read ``dst``, so no
        synchronisation is involved -- three small kernels, ~20 us of launch.

        Rows whose ``pending_src`` is not from the immediately previous
        generation (a request that sat out a step, or the first step after a
        drain) keep whatever the host put in ``dst`` -- their ``last_token``,
        which is authoritative in exactly that case.
        """
        n = len(reqs)
        if n == 0 or not self._async_active:
            return
        prev_gen = self._gen - 1
        idx: List[int] = []
        any_pending = False
        for r in reqs:
            s = r.pending_src
            if s is not None and s[0] == prev_gen:
                idx.append(s[1])
                any_pending = True
            else:
                idx.append(-1)
        if not any_pending:
            return
        self._ensure_async_buffers()
        self._idx_host[:n] = torch.tensor(idx, dtype=torch.int64)
        d = self._idx_dev[:n]
        d.copy_(self._idx_host[:n], non_blocking=True)
        gathered = self._tok_dev.index_select(0, d.clamp(min=0)).to(dst.dtype)
        dst[:n] = torch.where(d >= 0, gathered, dst[:n])

    def _step_body(self) -> List[StepEvent]:
        """Run one prefill chunk *or* one decode step against the model.

        Normally exactly one of the two per call ("a step is never mixed",
        unless ``mixed_forward`` is on). The one exception: if a prefill attempt is due
        (``_should_prefill()``) but nothing in ``waiting`` can actually be
        admitted or restored right now (e.g. every free page is reserved by
        already-running requests), that attempt touches the model **not at
        all** -- it is a pure no-op -- so this falls through to a decode
        step in the same call rather than wedging every running request
        behind a queue head that cannot move. ``self._prefill_progressed``
        (set explicitly by ``_run_prefill_step``, not inferred from the
        decode-ratio counter) is the signal that a real chunk ran or a
        swapped request was restored.

        ``self.last_step_progressed`` records whether *anything* happened
        this call (an event, a partial chunk, a restore) -- ``.engine``'s
        background loop uses it to tell "genuinely nothing to do right now"
        (e.g. every waiting request is blocked on page capacity held by
        already-running requests) apart from real work, and sleeps briefly
        in the former case instead of spinning the host CPU at 100% doing
        nothing (``has_work()`` alone cannot distinguish the two: a
        permanently-blocked request keeps ``waiting`` non-empty forever).
        """
        events: List[StepEvent] = []
        # Chunk telemetry: "the chunk *this* step ran, or 0". Reset here rather
        # than in `_run_prefill_step` because a decode step never enters that
        # method, so leaving the previous chunk's value standing would make a
        # profiler count one chunk several times.
        self.last_chunk_tokens = 0
        self.last_chunk_seqs = 0
        self.last_step_mixed = False
        self.last_step_overlapped = False
        self.last_mixed_decode_rows = 0
        self.last_step_mixed_graphed = False
        self._process_aborts(events)
        progressed = bool(events)
        # When `mixed_forward` is on and both halves are
        # non-empty this replaces *both* branches below with one forward. It is
        # tried first and returns early on success; when it declines (nothing
        # admissible to prefill, or spec owns the step) it has committed
        # nothing and the separate prefill/decode path below runs unchanged.
        if self._should_mix():
            events += (self._run_overlap_step() if self.overlap
                       else self._run_mixed_step())
            if self._mixed_progressed:
                self.last_step_progressed = True
                return events
            progressed = progressed or self._prefill_progressed
        if self._should_prefill():
            events += self._run_prefill_step()
            progressed = progressed or self._prefill_progressed
            if self._prefill_progressed:
                self.last_step_progressed = True
                return events  # real progress was made; done for this step
        if self.running:
            before = len(events)
            events += self._run_decode_step()
            progressed = progressed or len(events) > before
        elif self.waiting and not events:
            events += self._run_prefill_step()
            progressed = progressed or self._prefill_progressed
        self.last_step_progressed = progressed
        return events

    def _should_prefill(self) -> bool:
        if not self.waiting:
            return False
        if not self.running:
            return True
        return self._decode_steps_since_prefill >= self.prefill_decode_ratio

    # -- prefill ------------------------------------------------------------ #
    def _collect_prefill_chunk(self, budget: int, events: List[StepEvent],
                               max_segments: Optional[int] = None):
        """Admit/advance waiting requests until ``budget`` tokens are packed.

        Extracted from :meth:`_run_prefill_step` unchanged so :meth:`
        _run_mixed_step` packs its prefill half with *exactly* the same
        admission, watermark, chunking and swap-restore rules -- the mixed step
        must not be a second, subtly different scheduler.

        Mutates: allocates slots, pops from ``waiting``, advances
        ``num_computed_tokens``, moves finished-prompt requests into
        ``running``, sets ``_prefill_progressed`` (which a swap *restore* also
        sets, without contributing to the chunk). Appends "length" events for
        an over-long prompt to ``events``.

        Returns ``(reqs, token_ids, start_positions, slots)``, all empty when
        nothing could be packed -- and when they are empty **nothing has been
        allocated**, so a caller may abandon the step: the loop only leaves the
        `req.slot is None` branch by allocating a slot, and a request holding a
        slot with ``budget > 0`` always contributes at least one token.

        ``max_segments`` (graphed mixed step): stop packing once this many *segments* are in
        the chunk, however much budget is left. A graphed mixed step has a
        fixed number of FlashInfer plan rows, so the segment count is part of
        its shape; ``None`` (every other caller) keeps the old behaviour, where
        only the token budget bounds it.

        The per-segment ``ensure_capacity`` calls are batched into one
        device write by the same ``defer_page_table_writes`` scope
        :meth:`_admit_decode_rows` uses.
        """
        with self.model.kv_pool.defer_page_table_writes():
            return self._collect_prefill_chunk_inner(budget, events, max_segments)

    def _collect_prefill_chunk_inner(self, budget: int, events: List[StepEvent],
                                     max_segments: Optional[int] = None):
        chunk_reqs: List[Request] = []
        chunk_token_ids: List[List[int]] = []
        chunk_start_pos: List[int] = []
        chunk_slots: List[int] = []

        while self.waiting and budget > 0:
            if max_segments is not None and len(chunk_reqs) >= max_segments:
                break
            req = self.waiting[0]
            if len(req.prompt_token_ids) > self.max_context_len:
                # Unreachable through the HTTP layer (`context_length_error`
                # 400s it first), and deliberately still here: prefill would
                # index the rotary table at `len(prompt) - 1`, and one
                # out-of-range gather is a device-side assert, not an
                # exception. Fail the *request*, never the engine.
                self.waiting.popleft()
                req.status = STATUS_DONE
                req.finish_reason = "length"
                events.append(StepEvent(req, [], True, "length"))
                continue
            if req.slot is None:
                if req.request_id in self._swapped:
                    if not self._can_admit(req):
                        break
                    self.waiting.popleft()
                    req.slot = self.slots.alloc()
                    req.admitted_at = time.monotonic()
                    self._restore_swapped(req)
                    req.status = STATUS_DECODING
                    self.running[req.slot] = req
                    self._prefill_progressed = True
                    continue
                if not self._can_admit(req):
                    break
                req.slot = self.slots.alloc()
                req.admitted_at = time.monotonic()
                self.model.reset_slot(req.slot)
                if self.spec is not None:
                    self.spec.reset_slot(req.slot)
                if self.prefix_cache is not None and req.num_computed_tokens == 0:
                    hit = self.prefix_cache.lookup(req.prompt_token_ids)
                    if hit is not None:
                        req.num_computed_tokens = self.prefix_cache.restore(
                            hit, req.slot, self.model.kv_pool
                        )

            remaining_prompt = len(req.prompt_token_ids) - req.num_computed_tokens
            if remaining_prompt <= 0:
                # Nothing left to prefill (shouldn't normally happen while in
                # `waiting`, but stay robust) -- fall through to decode.
                self.waiting.popleft()
                req.status = STATUS_DECODING
                self.running[req.slot] = req
                self._prefill_progressed = True
                continue

            take = min(remaining_prompt, budget)
            start = req.num_computed_tokens
            ids = req.prompt_token_ids[start : start + take]
            self.model.kv_pool.ensure_capacity(req.slot, start + take)

            chunk_reqs.append(req)
            chunk_token_ids.append(ids)
            chunk_start_pos.append(start)
            chunk_slots.append(req.slot)

            req.num_computed_tokens += take
            budget -= take

            if req.num_computed_tokens >= len(req.prompt_token_ids):
                self.waiting.popleft()
                req.status = STATUS_DECODING
                self.running[req.slot] = req
            else:
                break  # this request ate the rest of the budget

        return chunk_reqs, chunk_token_ids, chunk_start_pos, chunk_slots

    def _run_prefill_step(self) -> List[StepEvent]:
        events: List[StepEvent] = []
        self._prefill_progressed = False
        self._mark("admit")
        chunk_reqs, chunk_token_ids, chunk_start_pos, chunk_slots = (
            self._collect_prefill_chunk(self.prefill_chunk_tokens, events)
        )
        self._mark("collect")

        if not chunk_reqs:
            return events

        self._prefill_progressed = True
        self._decode_steps_since_prefill = 0
        self.last_chunk_tokens = sum(len(ids) for ids in chunk_token_ids)
        self.last_chunk_seqs = len(chunk_reqs)
        batch = make_prefill_batch(chunk_token_ids, chunk_start_pos, chunk_slots, self.device)
        self._mark("build")
        self._mark("plan")
        self._gpu_mark("g_launch")
        if self.spec is not None:
            # The MTP head needs (a) its own KV layer populated over the prompt
            # and (b) the chunk's last hidden state as the `h_t` its first draft
            # is conditioned on -- both come out of this one forward, so ask for
            # the hidden states rather than running the stack twice.
            logits, hidden = self.model.prefill_forward(
                batch, all_logits=False, return_hidden=True
            )
            self.spec.on_prefill(batch, hidden)
        else:
            logits = self.model.prefill_forward(batch, all_logits=False)

        finished_reqs, finished_idx = self._prefill_first_token_rows(chunk_reqs, events)
        self._gpu_mark("g_done")
        self._mark("launch")

        if finished_reqs:
            rows = logits[torch.tensor(finished_idx, dtype=torch.long, device=logits.device)]
            self._emit_first_tokens(finished_reqs, rows, events)
        self._mark("book")
        return events

    def _prefill_first_token_rows(
        self, chunk_reqs: Sequence[Request], events: List[StepEvent]
    ) -> Tuple[List[Request], List[int]]:
        """Which of a chunk's segments emit their first token this step.

        A segment mid-prompt has nothing to sample (more chunks to come); a
        ``max_tokens <= 0`` request finishes here with no token at all. Returns
        ``(reqs, indices-into-the-chunk's-segment-order)``, which is the same
        order the logits rows come back in from both ``prefill_forward`` and
        ``mixed_forward``."""
        reqs: List[Request] = []
        idx: List[int] = []
        for i, req in enumerate(chunk_reqs):
            if req.status != STATUS_DECODING:
                continue  # more prefill chunks still needed for this request
            if self.prefix_cache is not None:
                # every caller reaches here after launching the forward that
                # consumed the last prompt token, on the stream that ran it
                self.prefix_cache.snapshot(req.request_id, req.slot, len(req.prompt_token_ids))
            if req.params.max_tokens <= 0:
                self._finish(req, "length")
                events.append(StepEvent(req, [], True, "length"))
                continue
            idx.append(i)
            reqs.append(req)
        return reqs, idx

    def _sample_rows_dev(self, rows: torch.Tensor, reqs: Sequence[Request]) -> torch.Tensor:
        """:func:`sample_tokens` over ``rows``, left **on the device**.

        Split out of :meth:`_sample_rows` for asynchronous scheduling: that path
        wants the sampled ids without the D2H that used to be welded to them.
        """
        dev = self.device
        temps = torch.tensor([r.params.temperature for r in reqs], dtype=torch.float32, device=dev)
        top_ps = torch.tensor([r.params.top_p for r in reqs], dtype=torch.float32, device=dev)
        top_ks = torch.tensor(
            [float(max(r.params.top_k, 0)) for r in reqs], dtype=torch.float32, device=dev
        )
        return sample_tokens(
            rows, temps, top_ps, top_ks, candidates=self.rt.sampler_candidates
        )

    # -- the two commit points, synchronous or deferred --------------------- #
    def _emit_first_tokens(
        self, reqs: Sequence[Request], rows: torch.Tensor, events: List[StepEvent]
    ) -> None:
        """The first token of every prompt that finished its prefill this step.

        ``rows`` is the ``[len(reqs), vocab]`` slice of logits to sample.  On
        the synchronous path this samples, harvests and commits inline, which
        is the original behaviour; on the asynchronous path it
        samples and hands the device tensor to :meth:`_publish`, and the
        commit happens one step later in :meth:`_commit`.
        """
        if not len(reqs):
            return
        toks = self._sample_rows_dev(rows, reqs)
        if self._async_active:
            self._publish(toks, reqs, first=True)
            return
        now = time.monotonic()
        for req, tok in zip(reqs, self._harvest(toks, len(reqs))):
            req.last_token = tok
            req.output_token_ids.append(tok)
            req.first_token_at = now
            finished, reason = self._check_stop(req, tok)
            if finished:
                self._finish(req, reason)
            events.append(StepEvent(req, [tok], finished, reason))

    def _emit_decode_rows(
        self, reqs: Sequence[Request], out: torch.Tensor, events: List[StepEvent]
    ) -> None:
        """One token per running row.  ``out`` is already sampled ids."""
        if not len(reqs):
            return
        if self._async_active:
            self._publish(out, reqs, first=False)
            return
        for req, tok in zip(reqs, self._harvest(out, len(reqs))):
            req.num_computed_tokens += 1
            req.output_token_ids.append(tok)
            req.last_token = tok
            finished, reason = self._check_stop(req, tok)
            if finished:
                self._finish(req, reason)
            events.append(StepEvent(req, [tok], finished, reason))

    def _sample_rows(self, rows: torch.Tensor, reqs: Sequence[Request]) -> List[int]:
        """One :func:`sample_tokens` call over ``rows``, one per request.

        The eager sampler (as opposed to the graphed decode step's, which reads
        the fixed ``DeviceBuffers`` param vectors). ``_harvest`` rather than
        ``.tolist()`` so the D2H goes through pinned memory on CUDA -- on a
        mixed step this is the step's *only* sync, and it covers both the
        prefill segments' first tokens and every decode row."""
        return self._harvest(self._sample_rows_dev(rows, reqs), len(reqs))

    # -- mixed prefill + decode ------------------------------------------------ #
    def _should_mix(self) -> bool:
        """Is this step a mixed one?

        Three conditions, and deliberately no ratio counter: under
        ``mixed_forward`` there is nothing to interleave, so
        ``prefill_decode_ratio`` (which exists only to bound how long decode
        waits behind prefill) has no job. Every step carries the whole running
        set *and* as much prefill as the budget allows.

        1. **Both halves must be non-empty.** With nothing waiting this is a
           decode step (and a graphed one, which is strictly faster than an
           eager mixed step at the same batch); with nothing running it is a
           plain prefill chunk.
        2. **Speculative decoding wins the step if it is eligible.** ``spec``
           is only configured up to ``--spec-max-batch`` (16 on the serving
           path), and at B<=16 a decode step is ~13 ms of mostly-idle GEMMs:
           the mixed step's weight-read amortisation is worth little there,
           while spec's 1..k+1 tokens per sequence is worth a lot. Above the
           cap ``_use_spec`` is False and the mixed step takes over -- which is
           exactly the regime (conc >= 64) it was built for. So the two
           features do not overlap; they partition the batch axis.
        """
        if not self.mixed_forward:
            return False
        if not self.waiting or not self.running:
            return False
        if self.spec is not None and self._use_spec(list(self.running.values())):
            return False
        return True

    def _run_mixed_step(self) -> List[StepEvent]:
        """One forward over ``[prefill chunk ‖ every decode row]``.

        The order of the two collections is "running first, then waiting",
        matching vLLM: the decode rows' page capacity is secured (with
        preemption if it comes to that) *before* any prefill token is admitted,
        so a running sequence is never starved by a fresh prompt.

        Returns ``[]`` with ``_mixed_progressed`` False when the step cannot be
        mixed after all (no decode row survives, or nothing could be prefilled)
        -- in which case nothing has been committed and ``step()`` falls
        through to the ordinary prefill/decode path. That "no chunk => nothing
        allocated" property is what makes the abandon safe; see
        :meth:`_collect_prefill_chunk`.

        **p99 note.** A mixed step's decode rows wait for the whole chunk, so
        the chunk cap is now a direct TPOT floor rather than a
        one-step-in-``prefill_decode_ratio`` stall: at the measured 13,991
        prefill tok/s, an 8,192-token chunk is ~590 ms of TPOT for all 256 rows
        and a 2,048-token one is ~150 ms. Aggregate throughput is (to first
        order) unaffected either way; ``--prefill-chunk-tokens`` is the knob.
        """
        events: List[StepEvent] = []
        self._mixed_progressed = False
        self._prefill_progressed = False

        # -- 1. the decode half: every running row, capacity secured --------- #
        dec_reqs = self._admit_decode_rows(events, window=1)
        self._mark("admit")
        if not dec_reqs:
            return events

        # -- 2. the prefill half, from whatever budget/pages are left -------- #
        # Under `mixed_pad` the budget is `chunk_tokens - n_segments`
        # and the segment count is capped, because a graphed step's prefill
        # half is a fixed number of plan rows every one of which must be
        # non-empty (`MixedPadSpec.budget`).
        pad_spec = self.mixed_pad
        budget = pad_spec.budget if pad_spec is not None else self.prefill_chunk_tokens
        chunk_reqs, chunk_token_ids, chunk_start_pos, chunk_slots = (
            self._collect_prefill_chunk(
                budget, events,
                max_segments=(pad_spec.max_segments if pad_spec is not None else None),
            )
        )
        if not chunk_reqs:
            return events  # nothing to mix with; `step()` runs a decode step

        # A swap-restore inside step 2 puts a request straight back into
        # `running` -- after `dec_reqs` was snapshotted, so it simply does not
        # decode this step. Re-filter for the same reason `_run_decode_step`
        # does: step 1's preemption may have evicted one of the rows it kept.
        dec_reqs = [r for r in dec_reqs if self.running.get(r.slot) is r]
        if not dec_reqs:
            return events

        self._mixed_progressed = True
        self._prefill_progressed = True
        self._decode_steps_since_prefill = 0
        self.last_step_mixed = True
        self.last_chunk_tokens = sum(len(ids) for ids in chunk_token_ids)
        self.last_chunk_seqs = len(chunk_reqs)
        self.last_mixed_decode_rows = len(dec_reqs)

        dec_slots = [r.slot for r in dec_reqs]
        # Async scheduling: 0 where the token is still in flight; `_apply_pending_tokens`
        # overwrites those entries of `batch.token_ids` on the device below.
        dec_tokens = [(0 if r.last_token is None else int(r.last_token)) for r in dec_reqs]
        dec_positions = [r.num_computed_tokens for r in dec_reqs]

        # -- 2b. pad to the graph's fixed shape, if there is one -------------- #
        # `fits` can be False even under `mixed_pad` -- more decode rows than
        # the widest bucket, say -- and then the step simply runs unpadded and
        # eager. Padding is a speed decision; every step still runs.
        padded = None
        if pad_spec is not None and pad_spec.fits(
            sum(len(ids) for ids in chunk_token_ids), len(chunk_reqs), len(dec_reqs)
        ):
            padded = pad_mixed_step(
                chunk_token_ids, chunk_start_pos, chunk_slots,
                dec_slots, dec_tokens, dec_positions, pad_spec,
            )
            # Before the forward: the padding rows thread their recurrence
            # through the scratch slot, and this makes that a fresh zero state
            # every step instead of an accumulating one.
            reset_scratch_state(self.model)

        if padded is not None:
            prefill = make_prefill_batch(
                padded.token_ids, padded.start_positions, padded.slots, self.device
            )
            batch = make_mixed_batch(
                prefill, padded.decode_slots, padded.decode_token_ids,
                padded.decode_positions, self.device,
            )
        else:
            prefill = make_prefill_batch(
                chunk_token_ids, chunk_start_pos, chunk_slots, self.device
            )
            batch = make_mixed_batch(
                prefill, dec_slots, dec_tokens, dec_positions, self.device
            )

        # Async scheduling: the decode rows of a mixed step take their input token from
        # the device too.  `pad_mixed_step` appends its padding *after* the
        # real rows, so the real decode rows are the first `len(dec_reqs)` of
        # the decode block, at `n_prefill_tokens` in the token axis.
        n_pre = int(batch.n_prefill_tokens)
        self._apply_pending_tokens(batch.token_ids[n_pre:], dec_reqs)
        self._mark("build")

        # Before the launch, never after -- see
        # `_spec_plain_hook`. The prefill half's `on_prefill` cannot move (it
        # needs `hidden`), but it is one small layer over the chunk, not a
        # stream synchronisation.
        self._spec_plain_hook(
            dec_slots,
            (batch.token_ids[n_pre : n_pre + len(dec_reqs)] if self._async_active
             else dec_tokens),
            dec_positions,
            len(dec_reqs),
        )

        graphed = (
            padded is not None
            and self.mixed_runner is not None
            and self.mixed_runner.ready(len(dec_reqs))
        )
        self.last_step_mixed_graphed = graphed
        hidden = None
        if graphed:
            p_bucket = self.mixed_runner.prepare_step(batch)
            self._mark("plan")
            self._gpu_mark("g_launch")
            logits, hidden = self.mixed_runner.replay(p_bucket)
        elif self.spec is not None:
            self._mark("plan")
            self._gpu_mark("g_launch")
            logits, hidden = self.model.mixed_forward(batch, return_hidden=True)
        else:
            self._mark("plan")
            self._gpu_mark("g_launch")
            logits = self.model.mixed_forward(batch)
        self._gpu_mark("g_done")
        self._mark("launch")

        if self.spec is not None:
            # Both halves of the MTP bookkeeping the two separate steps would
            # have done: the prefill half over the chunk's own rows (`hidden`'s
            # first `n_prefill_tokens`, which is exactly what a prefill step
            # would have handed it), and `on_plain_step` for the decode rows,
            # which took the plain path here by construction (`_should_mix`
            # yields the step to spec whenever spec is eligible).
            self.spec.on_prefill(batch.prefill, hidden[: batch.n_prefill_tokens])

        # -- 3. one sampler call over [first tokens ‖ decode rows] ------------ #
        # `logits` rows are `[segment last tokens] + [decode rows]`, so the
        # decode rows start at `n_prefill_seqs` (every segment gets a row,
        # whether or not it finished its prompt).
        first_reqs, first_idx = self._prefill_first_token_rows(chunk_reqs, events)
        n_seqs = batch.n_prefill_seqs
        sel = first_idx + list(range(n_seqs, n_seqs + len(dec_reqs)))
        all_reqs = list(first_reqs) + list(dec_reqs)
        sampled_dev = self._sample_rows_dev(
            logits[torch.tensor(sel, dtype=torch.long, device=logits.device)], all_reqs
        )
        n_first = len(first_reqs)
        if self._async_active:
            # One publish per group, so `_publish` can advance
            # `num_computed_tokens` for the decode rows and not for the
            # prefill segments (whose prompt is already counted).
            self._publish(sampled_dev[:n_first], first_reqs, first=True)
            self._publish(sampled_dev[n_first:], dec_reqs, first=False)
            self._mark("book")
            return events

        sampled = self._harvest(sampled_dev, len(all_reqs))
        now = time.monotonic()
        for req, tok in zip(first_reqs, sampled[:n_first]):
            req.last_token = tok
            req.output_token_ids.append(tok)
            req.first_token_at = now
            finished, reason = self._check_stop(req, tok)
            if finished:
                self._finish(req, reason)
            events.append(StepEvent(req, [tok], finished, reason))
        for req, tok in zip(dec_reqs, sampled[n_first:]):
            req.num_computed_tokens += 1
            req.output_token_ids.append(tok)
            req.last_token = tok
            finished, reason = self._check_stop(req, tok)
            if finished:
                self._finish(req, reason)
            events.append(StepEvent(req, [tok], finished, reason))
        self._mark("book")
        return events

    # -- overlapped prefill + decode ------------------------------------------- #
    def _run_overlap_step(self) -> List[StepEvent]:
        """One prefill chunk and one decode step, **concurrently, two streams**.

        Same two collections, in the same order, by the same methods as
        :meth:`_run_mixed_step` -- so admission, watermark, preemption and
        chunk-boundary rules are identical and cannot drift. What differs is
        only how the work reaches the GPU:

        * the prefill half is the graphed mixed step padded to **one** decode
          row (a padding row on the scratch slot; ``MixedGraphRunner`` has no
          zero-row shape), which is exactly a graphed prefill chunk;
        * the decode half is the ordinary graphed decode step, buffers filled
          by :meth:`_fill_decode_buffers` and sampled inside its own graph;
        * both are *planned* first (no host work may sit between two in-flight
          graphs), then the decode graph is launched on ``_overlap_stream``
          and the prefill graph on the current one, and the two are joined.

        Why this is safe to run at the same time rather than merely legal to
        write: the two halves touch **disjoint sequence slots** (a request is
        either prefilling or decoding, which ``_collect_prefill_chunk``
        enforces structurally by only moving a request into ``running`` when
        its prompt completes), so the SSM/conv state rows, the KV pages and
        the ``seq_len`` counters they write are disjoint too. The remaining
        shared state is the two things ``RuntimeConfig.overlap_streams``
        separates at build time -- the graph mempool and FlashInfer's float
        workspace -- and the model weights, which are read-only.

        **The one exception, and why it is benign.** Both halves' *padding*
        rows live on the same slot -- ``model.scratch_slot`` -- so the prefill
        graph's pad segments and the decode graph's pad rows do write the same
        SSM/conv state, the same KV page and the same ``seq_len`` entry at the
        same time. Nothing reads any of it: ``reset_scratch_state`` zeroes the
        slot at the top of every step, both plans are built from *explicit*
        ``kv_lens``/``seq_lens`` rather than from the device counter, and the
        logits rows those padding rows produce (row ``n_segments`` of the
        prefill graph's output, rows ``>= dec_batch`` of ``out_tokens``) are
        never sampled. The race is over bytes with no reader, in a step that
        is a pure function of its real inputs either way.

        With no side stream (CPU, ``--no-graphs``) the two halves run in the
        same order on the same stream. That is the property the CPU tests
        pin: an overlapped step emits, token for token, what a prefill step
        followed by a decode step emits.
        """
        events: List[StepEvent] = []
        self._mixed_progressed = False
        self._prefill_progressed = False

        dec_reqs = self._admit_decode_rows(events, window=1)
        self._mark("admit")
        if not dec_reqs:
            return events

        pad_spec = self.mixed_pad
        budget = pad_spec.budget if pad_spec is not None else self.prefill_chunk_tokens
        chunk_reqs, chunk_token_ids, chunk_start_pos, chunk_slots = (
            self._collect_prefill_chunk(
                budget, events,
                max_segments=(pad_spec.max_segments if pad_spec is not None else None),
            )
        )
        self._mark("collect")
        if not chunk_reqs:
            return events
        dec_reqs = [r for r in dec_reqs if self.running.get(r.slot) is r]
        if not dec_reqs:
            return events

        self._mixed_progressed = True
        self._prefill_progressed = True
        self._decode_steps_since_prefill = 0
        self.last_step_mixed = True
        self.last_step_overlapped = True
        self.last_chunk_tokens = sum(len(ids) for ids in chunk_token_ids)
        self.last_chunk_seqs = len(chunk_reqs)
        self.last_mixed_decode_rows = len(dec_reqs)

        # -- the prefill half ------------------------------------------------ #
        # Graphed: the mixed-step graph padded to **one** decode row -- a
        # padding row on the scratch slot, because `MixedGraphRunner` has no
        # zero-row shape. Eager (CPU, `--no-graphs`): a plain `prefill_forward`
        # over the unpadded chunk, i.e. literally `_run_prefill_step`'s
        # forward, which is what makes "an overlapped step == a prefill step +
        # a decode step" exact rather than approximate off the GPU.
        scratch = self.model.scratch_slot
        graphed = (
            self.mixed_runner is not None
            and self.mixed_runner.ready(1)
            and pad_spec is not None
            and pad_spec.fits(self.last_chunk_tokens, len(chunk_reqs), 1)
            # A graphed chunk always computes `chunk_tokens` tokens; below
            # `overlap_min_fill` of that, most of the step is padding on the
            # scratch slot and an eager chunk over the real tokens is strictly
            # less work. See the field's docstring for the measurement.
            and self.last_chunk_tokens >= self.overlap_min_fill * pad_spec.chunk_tokens
        )
        pre_batch = None
        if graphed:
            padded = pad_mixed_step(
                chunk_token_ids, chunk_start_pos, chunk_slots,
                [scratch], [PAD_TOKEN_ID], [0], pad_spec,
            )
            reset_scratch_state(self.model)
            prefill = make_prefill_batch(
                padded.token_ids, padded.start_positions, padded.slots, self.device
            )
            pre_batch = make_mixed_batch(
                prefill, padded.decode_slots, padded.decode_token_ids,
                padded.decode_positions, self.device,
            )
        else:
            prefill = make_prefill_batch(
                chunk_token_ids, chunk_start_pos, chunk_slots, self.device
            )

        # -- the decode half, by exactly the decode step's own rules --------- #
        dec_batch = len(dec_reqs)
        d_bucket, d_slots, d_input_ids, d_positions, d_seq_lens = (
            self._fill_decode_buffers(dec_reqs)
        )
        self._mark("build")

        # The MTP catch-up goes before both plans and
        # both launches -- it synchronises the stream, and after the launch
        # that means blocking the host for the whole of the step's kernels.
        self._spec_plain_hook(
            d_slots,
            (self.buf.input_ids[:dec_batch] if self._async_active
             else d_input_ids[:dec_batch]),
            d_positions,
            dec_batch,
        )

        # -- plan both, then launch both ------------------------------------- #
        self.last_step_mixed_graphed = graphed
        hidden = None
        if graphed and self._overlap_stream is not None:
            # Both plans first: once two graphs are in flight there is nowhere
            # to put host work, and FlashInfer's plan() does a blocking H2D.
            p_bucket = self.mixed_runner.prepare_step(pre_batch)
            self.decoder.prepare_step(dec_batch, d_slots, seq_lens=d_seq_lens)
            self._mark("plan")
            self._gpu_mark("g_launch")
            main = torch.cuda.current_stream()
            side = self._overlap_stream
            side.wait_stream(main)
            with torch.cuda.stream(side):
                out = self.decoder.replay(d_bucket)
            logits, hidden = self.mixed_runner.replay(p_bucket)
            main.wait_stream(side)
        elif graphed:
            self._mark("plan")
            self._gpu_mark("g_launch")
            logits, hidden = self.mixed_runner.step(pre_batch)
            out = self.decoder.step(dec_batch, d_slots, seq_lens=d_seq_lens)
        elif self.spec is not None:
            self._mark("plan")
            self._gpu_mark("g_launch")
            logits, hidden = self.model.prefill_forward(
                prefill, all_logits=False, return_hidden=True
            )
            out = self.decoder.step(dec_batch, d_slots, seq_lens=d_seq_lens)
        else:
            self._mark("plan")
            self._gpu_mark("g_launch")
            logits = self.model.prefill_forward(prefill, all_logits=False)
            out = self.decoder.step(dec_batch, d_slots, seq_lens=d_seq_lens)
        self._gpu_mark("g_done")
        self._mark("launch")

        if self.spec is not None:
            # The graphed half's `hidden` covers the padded chunk, so the MTP
            # head sees only the real prefill rows; the eager half's is already
            # exactly the chunk, as `_run_prefill_step` hands it over.
            if pre_batch is not None:
                self.spec.on_prefill(pre_batch.prefill,
                                     hidden[: pre_batch.n_prefill_tokens])
            else:
                self.spec.on_prefill(prefill, hidden)

        # -- emit: the chunk's first tokens, then the decode rows ------------ #
        first_reqs, first_idx = self._prefill_first_token_rows(chunk_reqs, events)
        if first_reqs:
            rows = logits[torch.tensor(first_idx, dtype=torch.long, device=logits.device)]
            self._emit_first_tokens(first_reqs, rows, events)
        self._emit_decode_rows(dec_reqs, out, events)
        self._mark("book")
        return events

    # -- decode --------------------------------------------------------------- #
    def _use_spec(self, reqs: Sequence[Request]) -> bool:
        """Is this decode step servable speculatively?

        Greedy-only by default: the acceptance rule "accept while draft ==
        argmax(target)" reproduces greedy decoding exactly, but says nothing
        about a temperature/top-p draw. A batch with any sampling request
        therefore takes the plain decode path *for that step* -- correctness
        first, and the mixed case is rare enough that per-request routing is
        not worth a second graph shape.

        The second veto is ``--spec-max-batch``: a head-to-head measurement
        showed spec decode winning at B<=8 and
        losing at B>=128 (the doubled GDN window pass and the M=B*(k+1) verify
        GEMM dominate), so a step whose *live* decode batch exceeds
        ``self.spec_max_batch`` takes the plain graphed step even though
        ``spec`` is configured and every request in it is greedy. This is a
        per-*step* policy, not a per-request or a build-time one: the same
        scheduler serves both shapes, switching every call as the running set
        crosses the threshold -- see ``spec.on_plain_step`` (called from
        ``_run_decode_step``'s plain branch below) for what keeps that switch
        cheap and correct in both directions.
        """
        if self.spec is None or not reqs:
            return False
        if self.spec_max_batch is not None and len(reqs) > self.spec_max_batch:
            return False
        return self.spec.eligible([r.params.temperature for r in reqs])

    def _admit_decode_rows(self, events: List[StepEvent], *, window: int) -> List[Request]:
        """The running set that can actually be stepped, ``window`` tokens wide.

        Extracted from :meth:`_run_decode_step` so :meth:`_run_mixed_step`
        secures its decode rows by exactly the same rules -- context cap, page
        capacity, preemption, and the "a victim of *this* pass is no longer
        running" re-filter. Requests that cannot proceed are finished (with
        ``"length"`` or ``"abort"``) and their events appended.

        **O(1) device work for the whole pass.**  The loop
        below is unchanged -- it has to be, because preemption is
        order-dependent (a victim chosen for row *i* is no longer running for
        row *i+1*, and the surviving set is the pass's product).  What changed
        is what one iteration *costs*.  ``pages_allocated`` is an O(1) list
        read instead of a CPU reduction over the page-table row, and the
        device half of ``ensure_capacity`` is **staged**: the
        ``defer_page_table_writes`` scope collects every row that crossed a
        page boundary and writes them all in one ``index_copy_`` on the way
        out -- or, in the common step where nobody crossed one, does nothing
        at all.  Writing each such row eagerly would issue a pageable H2D,
        i.e. a ``cudaStreamSynchronize``; with a step's kernels queued ahead
        of it that is a full-step host block, which showed up as ``admit``
        *rising* from 26.3 to 48.6 ms under asynchronous scheduling.
        """
        reqs = list(self.running.values())
        if not reqs:
            return []
        with self.model.kv_pool.defer_page_table_writes():
            return self._admit_decode_rows_inner(reqs, events, window)

    def _admit_decode_rows_inner(
        self, reqs: List[Request], events: List[StepEvent], window: int
    ) -> List[Request]:
        """The body of :meth:`_admit_decode_rows`, inside the deferral scope."""
        ok_reqs: List[Request] = []
        for r in reqs:
            if r.slot is None or self.running.get(r.slot) is not r:
                # already preempted earlier in this same pass, as a side
                # effect of freeing pages for another request in `reqs`
                continue
            # Async scheduling. `_check_stop` finishes a request the step *after* its
            # `max_tokens`-th token, and under asynchronous scheduling that
            # step has not committed yet -- so the request is still in
            # `running` with its last token in flight.  Scheduling another row
            # for it would generate a token past the limit and keep doing so.
            # A no-op synchronously (`pending_tokens == 0`, and a request at
            # its limit has already left `running`).
            if self._emitted_or_pending(r) >= r.params.max_tokens:
                continue
            # Async scheduling. A request with a token in flight is *skipped*, never
            # failed, when it cannot be stepped: the pending commit is about
            # to finish it (context cap) or free pages (a neighbour
            # finishing), and emitting a terminal event here would strand the
            # in-flight token behind its own request's "finished".
            deferred = r.pending_tokens > 0
            if r.num_computed_tokens + window > self.max_context_len:
                # This step would write positions `L .. L+window-1`; the last
                # of them is off the end of the rotary table. `_check_stop`
                # already ends a request the moment it reaches the cap, so on
                # the `window == 1` path this is unreachable -- it is here for
                # the speculative path (`window == k+1`), which writes the
                # whole draft window before it knows how much of it is
                # accepted, and can therefore hit the cap mid-window.
                if deferred:
                    continue
                self._finish(r, "length")
                events.append(StepEvent(r, [], True, "length"))
                continue
            if self._ensure_capacity_with_preemption(r, extra_tokens=window):
                ok_reqs.append(r)
            elif deferred:
                continue
            else:
                # unrecoverable: not enough pages even after preempting
                # everything else. Fail this request rather than wedge the
                # engine loop.
                self._finish(r, "abort")
                events.append(StepEvent(r, [], True, "abort"))
        # a preempted victim disappears from `running`; only keep survivors
        return [r for r in ok_reqs if self.running.get(r.slot) is r]

    def _run_decode_step(self) -> List[StepEvent]:
        events: List[StepEvent] = []
        if not self.running:
            return events

        spec_on = self._use_spec(list(self.running.values()))
        window = self.spec.n if spec_on else 1
        reqs = self._admit_decode_rows(events, window=window)
        self._mark("admit")
        if not reqs:
            self._decode_steps_since_prefill += 1
            return events

        batch = len(reqs)
        if spec_on and self._async_active and any(r.pending_tokens for r in reqs):
            # Belt and braces for `_needs_sync_step`: it predicts the spec
            # branch from the *unfiltered* running set, and `_admit_decode_rows`
            # can shrink that set below `--spec-max-batch` and flip the
            # decision.  A spec step with tokens still in flight would plan the
            # next step over a `num_computed_tokens` it cannot know, so take
            # the plain path for this one step instead (the window it was
            # admitted with is wider than one token, which only over-reserves).
            spec_on = False
        if spec_on:
            self._decode_steps_since_prefill += 1
            return events + self._run_spec_decode_step(reqs)

        bucket, slots, input_ids, positions, seq_lens = self._fill_decode_buffers(reqs)
        self._mark("build")

        # The MTP catch-up runs **before** the launch,
        # not after it. It consumes only this step's *inputs* (slot, input
        # token, position), so it is legal in either place -- and its
        # `plan_draft0` synchronises the stream (FlashInfer's `plan()` needs
        # the indptr on the host), so after the launch it blocks the host for
        # the whole 44 ms of the step's kernels. With the call after the
        # launch, asynchronous scheduling gains nothing: `sync` 48.29 ms
        # synchronously and `book` 48.30 ms asynchronously are the same wait
        # wearing two labels. Before the
        # launch it waits on the *previous* step instead -- which is the whole
        # point of asynchronous scheduling, and free when there is nothing to
        # wait for.
        self._spec_plain_hook(
            slots,
            (self.buf.input_ids[:batch] if self._async_active else input_ids[:batch]),
            positions,
            batch,
        )

        d_bucket = self.decoder.prepare_step(batch, slots, seq_lens=seq_lens)
        self._mark("plan")
        self._gpu_mark("g_launch")
        out = self.decoder.replay(d_bucket)
        self._gpu_mark("g_done")
        self._mark("launch")
        self._decode_steps_since_prefill += 1
        self._emit_decode_rows(reqs, out, events)
        self._mark("book")
        return events

    def _spec_plain_hook(
        self,
        slots: Sequence[int],
        input_ids,
        positions: Sequence[int],
        batch: int,
    ) -> None:
        """``SpecDecoder.on_plain_step`` for a step that took the plain path.

        This step is over ``--spec-max-batch``
        (or a sampling request forced it) while spec decoding is configured
        overall, so the MTP head's own KV layer and ``h_prev`` carry have to be
        advanced by hand or the next re-entry into spec mode drafts off a stale
        carry and a KV cache with holes in it. ``input_ids`` / ``positions``
        are already exactly ``(last_token, num_computed_tokens)`` for
        ``reqs[:batch]`` pre-mutation, so they are reused rather than rebuilt.

        **The call site matters, and it is always before the launch.** ``on_plain_step`` plans a FlashInfer draft
        wrapper, and ``plan()`` needs the indptr on the host -- so it
        synchronises the stream. Called *after* the step's replay, that is a
        host block for the whole of the step's kernel time, and asynchronous
        scheduling then gains nothing: `sync` 48.29 ms synchronously and `book`
        48.30 ms asynchronously are the same wait wearing two labels. Called
        before the launch it waits on the *previous* step, which is precisely
        what asynchronous scheduling is for.

        Under asynchronous scheduling the host does not *have* the input
        tokens -- ``_fill_decode_buffers`` gathered them on the device from the
        previous step's output -- so the device tensor they were gathered into
        is passed instead; only ``F.embedding`` reads them.
        """
        if self.spec is None or batch <= 0:
            return
        self.spec.on_plain_step(slots[:batch], input_ids, positions[:batch])

    def _fill_decode_buffers(self, reqs: Sequence[Request]):
        """Fill the device buffers for a one-token-per-row decode step.

        Extracted from :meth:`_run_decode_step` so :meth:`
        _run_overlap_step` builds its decode half by *exactly* the same rules
        -- bucket padding, scratch rows, and above all the ``seq_lens``
        contract below, which is a correctness precondition rather than an
        optimisation. Returns ``(bucket, slots, input_ids, positions,
        seq_lens)``, every list already padded to ``bucket``.
        """
        batch = len(reqs)
        bucket = self.decoder.bucket_for(batch)
        scratch = self.model.scratch_slot
        pad = bucket - batch

        slots = [r.slot for r in reqs] + [scratch] * pad
        # Async scheduling: `last_token` is `None` for a row whose token is still in
        # flight; 0 is a placeholder that `_apply_pending_tokens` overwrites on
        # the device below, and the value never reaches the model.
        input_ids = [(0 if r.last_token is None else r.last_token) for r in reqs] + [0] * pad
        positions = [r.num_computed_tokens for r in reqs] + [0] * pad
        # Post-step context length, which is what FlashInfer must be planned
        # over: this step writes each sequence's new token at position
        # `num_computed_tokens`, so after the step its committed length is
        # `num_computed_tokens + 1`. Padding rows point at `scratch_slot`,
        # which owns exactly one page and holds seq_len 1.
        #
        # Passing these explicitly is required for correctness, not just to
        # save the D2H read: `plan_decode` runs *before* the graph replay,
        # and `append_kv` (inside the replay) is the only writer that
        # advances the device `seq_len`. Letting `build_flashinfer_indices`
        # read the device counter therefore plans over `num_computed_tokens`
        # keys -- one short -- so every decode step would attend to
        # everything *except* the token it is currently generating from.
        # The bug is GPU-only: `torch_fallback_decode` gathers with
        # `length=None` at *run* time (post-append), so the CPU path the test
        # suite exercises is correct either way.
        #
        # `+ 1` is right *here* because this is the one-token-per-step path;
        # the speculative path is `_run_spec_decode_step`, which spans
        # `L .. L+k` and does its own planning inside `SpecDecoder.step` from
        # the `ctx_lens` it is handed. If the two ever merge, this term
        # becomes `+ tokens_written_this_step`, not `+ 1`.
        seq_lens = [r.num_computed_tokens + 1 for r in reqs] + [1] * pad
        temps = [r.params.temperature for r in reqs] + [1.0] * pad
        top_ps = [r.params.top_p for r in reqs] + [1.0] * pad
        top_ks = [float(max(r.params.top_k, 0)) for r in reqs] + [0.0] * pad

        self.buf.host["input_ids"][:bucket] = torch.tensor(input_ids, dtype=torch.int32)
        self.buf.host["positions"][:bucket] = torch.tensor(positions, dtype=torch.int32)
        self.buf.host["slot_ids"][:bucket] = torch.tensor(slots, dtype=torch.int32)
        self.buf.host["temperature"][:bucket] = torch.tensor(temps, dtype=torch.float32)
        self.buf.host["top_p"][:bucket] = torch.tensor(top_ps, dtype=torch.float32)
        self.buf.host["top_k"][:bucket] = torch.tensor(top_ks, dtype=torch.float32)
        self.buf.upload(["input_ids", "positions", "slot_ids", "temperature", "top_p", "top_k"])
        # Async scheduling: the device-side token feed.  No-op on the synchronous path.
        self._apply_pending_tokens(self.buf.input_ids, reqs)
        return bucket, slots, input_ids, positions, seq_lens

    def _harvest(self, out: torch.Tensor, batch: int) -> List[int]:
        """``out[:batch]`` as a Python list, through pinned memory.

        Still one D2H sync per decode step -- it has to be, the sampled tokens
        are what the next step conditions on -- but a pinned destination skips
        the pageable staging copy the implicit ``.to("cpu")`` does. Falls back
        to the plain path on CPU builds and if pinning is unavailable.
        """
        if out.device.type != "cuda":
            return out[:batch].tolist()
        host = self._out_host
        if host is None or host.numel() < out.shape[0] or host.dtype != out.dtype:
            try:
                host = torch.empty(out.shape[0], dtype=out.dtype, pin_memory=True)
            except (RuntimeError, NotImplementedError):  # pragma: no cover
                self._out_host = None
                return out[:batch].to("cpu").tolist()
            self._out_host = host
        # The two halves are timed separately because a combined "decode
        # harvest (D2H)" time (about 19.7 ms/call at conc 256) is **not** the
        # copy. This is the step's first sync point after the graph replay, so
        # the `synchronize()` drains the whole 64-layer forward that
        # `decoder.step` queued in ~0.09 ms of host time. On a 100%-mixed run,
        # where the host is the bottleneck and the GPU has already caught up by
        # the time this is reached, the same code reads about 0.15 ms.
        # Attributing a 19.5 ms queue drain to a 1 KiB DMA makes this look like
        # an overlappable stall; it is not one, and the split counters keep
        # profiles from making that mistake.
        host[:batch].copy_(out[:batch], non_blocking=True)
        t0 = time.perf_counter()
        torch.cuda.current_stream(out.device).synchronize()
        self.harvest_gpu_wait_s += time.perf_counter() - t0
        self.harvest_calls += 1
        self._mark("sync")
        toks = host[:batch].tolist()
        self._mark("harvest")
        return toks

    def _run_spec_decode_step(self, reqs: List[Request]) -> List[StepEvent]:
        """One speculative step: 1..k+1 tokens per sequence.

        The host side is deliberately identical in shape to
        :meth:`_run_decode_step` -- fill the fixed device buffers, plan, replay,
        harvest -- with two differences:

        * the model is handed each sequence's *committed* context length ``L``
          (``num_computed_tokens``) rather than a position, because the step
          spans ``L .. L+k`` and the planner needs the whole span;
        * the harvest is variable-length: row ``i`` emits ``out_window[i, :m_i]``
          and ``num_computed_tokens`` advances by ``m_i``, which is what rolls
          the KV pointer back over the rejected tail (the next step re-plans
          from ``L + m_i`` and overwrites those positions).

        A stop condition (EOS, a stop id, ``max_tokens``) *inside* the window
        truncates the emitted list at that token and finishes the request: the
        tokens after it were computed but must never be shown, and the state
        they committed is discarded with the slot.
        """
        events: List[StepEvent] = []
        spec = self.spec
        assert spec is not None
        batch = len(reqs)
        slots = [r.slot for r in reqs]
        ctx_lens = [r.num_computed_tokens for r in reqs]
        last_tokens = [int(r.last_token) for r in reqs]

        sampling = (
            [r.params.temperature for r in reqs],
            [r.params.top_p for r in reqs],
            [float(r.params.top_k) for r in reqs],
        )
        out_window, m = spec.step(batch, slots, ctx_lens, last_tokens, sampling=sampling)
        rows = out_window[:batch].to("cpu").tolist()
        ms = m[:batch].to("cpu").tolist()
        spec.note_step(ms)

        for req, row, mi in zip(reqs, rows, ms):
            mi = int(mi)
            req.num_computed_tokens += mi
            new_tokens: List[int] = []
            finished = False
            reason: Optional[str] = None
            for tok in row[:mi]:
                tok = int(tok)
                req.output_token_ids.append(tok)
                req.last_token = tok
                new_tokens.append(tok)
                finished, reason = self._check_stop(req, tok)
                if finished:
                    break
            if finished:
                self._finish(req, reason)
            events.append(StepEvent(req, new_tokens, finished, reason))
        return events

    def _check_stop(self, req: Request, token_id: int) -> Tuple[bool, Optional[str]]:
        if len(req.output_token_ids) >= req.params.max_tokens:
            return True, "length"
        if req.num_computed_tokens >= self.max_context_len:
            # The sequence has filled the context: the *next* decode step would
            # write at position `max_context_len`, which is one past the last
            # row of the rotary table. Stop here, with the token just sampled
            # still emitted (it is a legitimate continuation -- it simply
            # cannot be conditioned on any further).
            return True, "length"
        if not req.params.ignore_eos and req.params.eos_token_id is not None and token_id == req.params.eos_token_id:
            return True, "stop"
        if token_id in req.params.stop_token_ids:
            return True, "stop"
        return False, None

    def _finish(self, req: Request, reason: str) -> None:
        req.status = STATUS_DONE
        req.finish_reason = reason
        self._swapped.pop(req.request_id, None)
        if req.slot is not None:
            # `free_pages`, NOT `free_slot`: slot ids are owned by
            # `SlotManager` (one id indexes the SSM pool *and* the KV page
            # table), and `fused_model._claim_all_kv_slots` marks every KV row
            # permanently live to make that work. `free_slot` would move the
            # row back into the pool's own `_free_slots` and out of
            # `_used_slots`, so the *next* `ensure_capacity` on that recycled
            # slot would raise "slot N is not an allocated sequence slot".
            kept = False
            if self.prefix_cache is not None:
                if req.num_computed_tokens >= len(req.prompt_token_ids):
                    kept = self.prefix_cache.commit(
                        req.request_id, req.prompt_token_ids, req.slot, self.model.kv_pool
                    )
                else:
                    self.prefix_cache.drop_pending(req.request_id)
            if not kept:
                self.model.kv_pool.free_pages(req.slot)
            self.slots.free(req.slot)
            self.running.pop(req.slot, None)
            req.slot = None

    # -- aborts ---------------------------------------------------------------- #
    def _process_aborts(self, events: List[StepEvent]) -> None:
        if not self._aborted:
            return
        handled = set()
        remaining: Deque[Request] = deque()
        while self.waiting:
            r = self.waiting.popleft()
            if r.request_id in self._aborted:
                if self.prefix_cache is not None:
                    self.prefix_cache.drop_pending(r.request_id)
                if r.slot is not None:
                    self.model.kv_pool.free_pages(r.slot)
                    self.slots.free(r.slot)
                    r.slot = None
                self._swapped.pop(r.request_id, None)
                r.status = STATUS_DONE
                r.finish_reason = "abort"
                events.append(StepEvent(r, [], True, "abort"))
                handled.add(r.request_id)
            else:
                remaining.append(r)
        self.waiting = remaining
        for r in list(self.running.values()):
            if r.request_id in self._aborted:
                self._finish(r, "abort")
                events.append(StepEvent(r, [], True, "abort"))
                handled.add(r.request_id)
        self._aborted -= handled

    # -- preemption ------------------------------------------------------------- #
    def _pick_preemption_victim(self, exclude_slot: Optional[int]) -> Optional[Request]:
        candidates = [r for r in self.running.values() if r.slot != exclude_slot]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r.admitted_at or 0.0)

    def _ensure_capacity_with_preemption(self, req: Request, extra_tokens: int = 1) -> bool:
        """``extra_tokens`` is how many *new* positions the coming step may
        write: 1 for a plain decode step, ``k+1`` for a speculative one (the
        whole window is written before the accepted length is known)."""
        pool = self.model.kv_pool
        need_tokens = req.num_computed_tokens + extra_tokens
        need_pages = pool.pages_needed(need_tokens)
        if need_pages > pool.cfg.max_pages_per_seq:
            return False
        have = pool.pages_allocated(req.slot)
        extra = max(need_pages - have, 0)
        if self.prefix_cache is not None and pool.num_free_pages < extra:
            self.prefix_cache.reclaim_pages(extra)
        guard = 0
        max_guard = len(self.running) + 1
        while pool.num_free_pages < extra and guard < max_guard:
            victim = self._pick_preemption_victim(exclude_slot=req.slot)
            if victim is None:
                break
            self._preempt(victim)
            guard += 1
        if pool.num_free_pages < extra:
            return False
        pool.ensure_capacity(req.slot, need_tokens)
        return True

    def _preempt(self, victim: Request) -> None:
        assert victim.slot is not None
        self._swapped[victim.request_id] = self._swap_out(victim)
        self.model.kv_pool.free_pages(victim.slot)
        self.slots.free(victim.slot)
        self.running.pop(victim.slot, None)
        victim.slot = None
        victim.status = STATUS_WAITING
        self.waiting.appendleft(victim)

    def _swap_out(self, req: Request) -> _SwapState:
        slot = req.slot
        ssm = self.model.state_pool[slot].detach().clone().to("cpu")
        conv = self.model.conv_pool[slot].detach().clone().to("cpu")
        n = req.num_computed_tokens
        kv: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer in range(self.model.kv_pool.cfg.n_layers):
            k, v = self.model.kv_pool.gather_dense(layer, slot, n)
            kv.append((k.detach().clone().to("cpu"), v.detach().clone().to("cpu")))
        mtp_hidden = (
            self.spec.h_prev[slot].detach().clone().to("cpu") if self.spec is not None else None
        )
        return _SwapState(ssm=ssm, conv=conv, kv=kv, length=n, mtp_hidden=mtp_hidden)

    def _restore_swapped(self, req: Request) -> None:
        saved = self._swapped.pop(req.request_id)
        slot = req.slot
        self.model.state_pool[slot].copy_(saved.ssm.to(self.device))
        self.model.conv_pool[slot].copy_(saved.conv.to(self.device))
        if self.spec is not None and saved.mtp_hidden is not None:
            self.spec.h_prev[slot].copy_(saved.mtp_hidden.to(self.device))
        n = saved.length
        if n > 0:
            self.model.kv_pool.ensure_capacity(slot, n)
            positions = torch.arange(n, dtype=torch.int32, device=self.device)
            slot_ids = torch.full((n,), slot, dtype=torch.int32, device=self.device)
            for layer, (k, v) in enumerate(saved.kv):
                self.model.kv_pool.append_kv(layer, slot_ids, positions, k.to(self.device), v.to(self.device))


__all__ = [
    "STATUS_WAITING",
    "STATUS_DECODING",
    "STATUS_DONE",
    "GenParams",
    "Request",
    "StepEvent",
    "SchedulerStats",
    "SlotManager",
    "Scheduler",
    "context_length_error",
]
