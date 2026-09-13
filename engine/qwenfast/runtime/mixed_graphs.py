"""CUDA-graphing the mixed prefill+decode step.

What this fixes, in one measurement
-----------------------------------
At concurrency 256, one eager mixed step carrying 1,024 prefill tokens + 120
decode rows takes **153 ms** (p50 146, p99 181). Fitting four chunk sizes (1,024 / 2,042 / 4,068 / 8,139 tokens ->
153 / 221 / 354 / 647 ms) and the line is ``82 ms + T / 14.4k tok/s``: the
marginal token costs what a standalone prefill chunk token costs, and there are
**82 ms of fixed cost per step** on top. Of that, ~11 ms is the decode rows'
marginal GEMM and ~7 ms their SSM state traffic. The remaining ~60 ms is host
launch latency: ~4k eager launches at a measured ~12-15 us/launch host cost,
which the *graphed* decode step hides and the eager mixed step
does not.

That 82 ms is the whole reason small-chunk mixing loses: the ~2,650 out tok/s
throughput bound at that concurrency needs a ~1,100-token chunk on every step, and at that
chunk the fixed cost is bigger than the work.

Why the mixed step "cannot" be captured, and why that is only half true
-----------------------------------------------------------------------
The step's shape is a different varlen packing every step. It is, but the shape that a CUDA graph freezes is the *tensor* shape, not the
segmentation, and only two kernels in the step actually read the segmentation:

* the **fla chunk kernel** (``fla.ops.gated_delta_rule.chunk_gated_delta_rule``)
  and the gather/scatter around it. Its index tensors come from
  ``fla.ops.utils.index.prepare_chunk_indices``, which builds them on the host
  and ends with ``.to(cu_seqlens)`` -- a **pageable H2D copy**, which capture
  forbids outright -- and is memoised by ``tensor_cache`` on argument
  ``is``-identity, so feeding it a reused static ``cu_seqlens`` buffer would
  return a **stale** index tensor for a new step's segmentation. Not "slow to
  capture": silently wrong. Verified against fla-core 0.5.2.
* the **varlen conv** (``kernels_gdn.triton_kernels._conv_prefill_kernel``),
  whose grid is ``(C-tiles, cdiv(max_seqlen, BT), n_seq)``. This one *is*
  capturable at a fixed grid -- it masks with ``if t0 >= n: return`` and
  ``slot < 0`` -- but it sits immediately before the fla call, so keeping it
  eager costs two launches and buys back nothing.

Everything else -- the embedding, both norms per layer, all five GEMMs per
layer, the gate epilogue, the gated RMSNorm, the MLP (tiled), the residual
adds, the KV scatter, FlashInfer's paged prefill ``run()`` (its
``use_cuda_graph=True`` mode exists for exactly this), the conv-update
and GDN decode kernels, the final norm and ``lm_head`` -- has a shape that
depends only on ``(prefill_chunk_tokens, decode_bucket)``.

So: **pad the step to a fixed shape and capture it in segments**, with one
eager *hole* per GDN layer for the two kernels above. 48 holes, 49 graph
segments, ~350 eager launches left of ~4,000. (The default one-graph mode goes
further: ``kernels_gdn.fla_static`` feeds fla caller-supplied index tensors, so
the whole step is a single graph with no holes. See ``MixedGraphRunner``.)

The three padding rules (and what makes them safe)
--------------------------------------------------
1. **Prefill tokens** are padded up to ``chunk_tokens`` by adding whole extra
   *segments* on the model's scratch slot, positions ``0..m-1``. A pad segment
   is a sequence like any other as far as every kernel is concerned; it writes
   the scratch slot's SSM/conv state and the scratch slot's KV pages, which no
   real request ever reads. ``prepare()`` gives the scratch slot
   ``ceil(chunk_tokens / page_size)`` pages so those writes land somewhere real
   (a padded query row must have ``kv_len >= q_len`` under FlashInfer's
   bottom-right causal alignment, so "point it at nothing" is not an option).
2. **Plan rows are a fixed count and every one is non-empty.** FlashInfer's
   graph mode plans over the whole persistent ``qo_indptr`` buffer, so the step
   must always present exactly ``n_segments + bucket`` rows. At most
   ``n_segments - 1`` may be real segments, which leaves at least one pad
   segment; and the scheduler's prefill budget is ``chunk_tokens - n_segments``,
   which leaves at least one token for each pad segment.
3. **Decode rows** are padded to the bucket exactly as ``GraphedDecoder`` pads
   a decode step: scratch slot, position 0, ``kv_len`` 1.

``reset_scratch_state`` zeroes the scratch slot's SSM/conv state before every
step, so the padding rows compute finite garbage rather than an ever-growing
one, and two runs of the same request stream produce the same padding
arithmetic. It is two ``index_fill_`` launches.

What is *not* padded, and does not need to be: the segmentation inside the
chunk. Segment lengths, slots and count vary freely from step to step; in the
segmented mode only the eager hole looks at them, and in the one-graph mode
they are uploaded into static index buffers before each replay.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from ..kernels_gdn import fla_static
from .fused_model import (
    FusedQwenForCausalLM,
    MixedBatch,
    RuntimeConfig,
    StepContext,
)

# --------------------------------------------------------------------------- #
# 1. the padding arithmetic (pure, host-side, CPU-testable)
# --------------------------------------------------------------------------- #
#: The token id every padding row carries. Any in-vocabulary id works -- the
#: rows are discarded -- and 0 keeps the embedding gather off an odd page.
PAD_TOKEN_ID = 0


@dataclass(frozen=True)
class MixedPadSpec:
    """The fixed shape a graphed mixed step is padded to.

    ``chunk_tokens`` is the scheduler's ``prefill_chunk_tokens``;
    ``n_segments`` is the number of prefill *plan rows* (``RuntimeConfig
    .mixed_graph_segments``); ``buckets`` is the decode-row bucket ladder,
    normally ``RuntimeConfig.buckets_for()``.
    """

    chunk_tokens: int
    n_segments: int
    buckets: Tuple[int, ...]
    scratch_slot: int
    #: Longest a **single** padding segment may be. A pad segment lives
    #: on the scratch slot at positions ``0..m-1``, so it indexes the rotary
    #: table at ``m-1`` and needs ``ceil(m / page_size)`` KV pages -- both of
    #: which are sized by ``max_model_len``. Putting all the padding in one
    #: fat segment silently caps a graphed chunk at ``max_model_len``: above
    #: it the rotary lookup runs off the end of the table, which is a
    #: **device-side assert** that surfaces as a cuBLAS failure from whatever
    #: GEMM runs next (e.g. "no backend works for M=4128 N=96 K=5120"), far
    #: from its cause.
    #:
    #: Splitting the padding across the rows instead lifts the cap to
    #: ``n_pad_rows * max_pad_len``, which is what lets an 8,192-token chunk be
    #: graphed inside a 2,752-token context. ``0`` == no limit (a single
    #: fat pad segment, kept so the CPU tests that construct a spec by hand do not
    #: all have to know about it).
    max_pad_len: int = 0

    @property
    def pad_cap(self) -> int:
        """Longest a single padding segment may be. ``chunk_tokens`` uncapped."""
        return self.max_pad_len if self.max_pad_len > 0 else self.chunk_tokens


    @property
    def min_pad_rows(self) -> int:
        """Plan rows that must be left free for padding.

        Enough to absorb the worst case, which is a chunk that is almost all
        padding: ``ceil(chunk_tokens / max_pad_len)``. One, when there is no
        cap: rule 2 exactly.
        """
        cap = self.pad_cap
        return max(1, -(-self.chunk_tokens // cap))

    @property
    def budget(self) -> int:
        """Prefill tokens the scheduler may pack into a graphed step.

        ``chunk_tokens - n_segments``: rule 2 -- every pad segment needs at
        least one token, and there can be up to ``n_segments`` of them.
        """
        return self.chunk_tokens - self.n_segments

    @property
    def max_segments(self) -> int:
        """Real prefill segments allowed.

        ``min_pad_rows`` are always left for padding: one without a length
        cap, more once a single pad segment is length-capped.
        """
        return self.n_segments - self.min_pad_rows

    def bucket_for(self, n_rows: int) -> Optional[int]:
        for b in self.buckets:
            if n_rows <= b:
                return b
        return None

    def pad_sizes(self, n_pad_seg: int, pad_tokens: int) -> Optional[List[int]]:
        """Split ``pad_tokens`` over ``n_pad_seg`` rows, each in ``[1, cap]``.

        Greedy and deterministic: fill each row as full as the cap allows while
        leaving one token for every row after it. ``None`` when it cannot be
        done, which :meth:`fits` reports as "this step is not graphable" and
        the scheduler answers by running it eagerly.
        """
        if n_pad_seg <= 0:
            return None if pad_tokens else []
        cap = self.pad_cap
        if not (n_pad_seg <= pad_tokens <= n_pad_seg * cap):
            return None
        sizes, left = [], pad_tokens
        for i in range(n_pad_seg):
            rows_after = n_pad_seg - i - 1
            take = min(cap, left - rows_after)
            sizes.append(take)
            left -= take
        return sizes

    def fits(self, n_tokens: int, n_real_segments: int, n_rows: int) -> bool:
        return (
            n_real_segments >= 1
            and n_real_segments <= self.max_segments
            and 1 <= n_tokens <= self.budget
            and n_rows >= 1
            and self.bucket_for(n_rows) is not None
            and self.pad_sizes(
                self.n_segments - n_real_segments, self.chunk_tokens - n_tokens
            ) is not None
        )


@dataclass
class PaddedMixedInputs:
    """The six lists ``make_prefill_batch``/``make_mixed_batch`` take, padded."""

    token_ids: List[List[int]]
    start_positions: List[int]
    slots: List[int]
    decode_slots: List[int]
    decode_token_ids: List[int]
    decode_positions: List[int]
    n_real_segments: int
    n_real_rows: int
    bucket: int


def pad_mixed_step(
    chunk_token_ids: Sequence[Sequence[int]],
    chunk_start_pos: Sequence[int],
    chunk_slots: Sequence[int],
    decode_slots: Sequence[int],
    decode_token_ids: Sequence[int],
    decode_positions: Sequence[int],
    spec: MixedPadSpec,
) -> PaddedMixedInputs:
    """Pad one mixed step's inputs to ``spec``'s fixed shape.

    The padding is appended, never interleaved: pad segments follow the real
    segments (so ``last_indices`` / ``logits_indices`` keep the real rows at
    the front and the decode rows start at exactly ``spec.n_segments``), and
    pad decode rows follow the real ones.

    Raises ``ValueError`` if the step does not fit -- the caller is expected to
    have used :meth:`MixedPadSpec.budget` / :attr:`MixedPadSpec.max_segments`
    as its own budget, so this is an assertion, not a control-flow path.
    """
    n_real_seg = len(chunk_token_ids)
    n_tok = sum(len(ids) for ids in chunk_token_ids)
    n_rows = len(decode_slots)
    bucket = spec.bucket_for(n_rows)
    if not spec.fits(n_tok, n_real_seg, n_rows) or bucket is None:
        raise ValueError(
            f"pad_mixed_step: {n_tok} tokens in {n_real_seg} segments + {n_rows} decode "
            f"rows does not fit chunk_tokens={spec.chunk_tokens} "
            f"(budget {spec.budget}), n_segments={spec.n_segments} "
            f"(max {spec.max_segments} real), buckets={spec.buckets}"
        )

    n_pad_seg = spec.n_segments - n_real_seg
    pad_tokens = spec.chunk_tokens - n_tok
    # The split is not arbitrary. It must give every row >= 1
    # token (rule 2) *and* no row more than `spec.max_pad_len`, because a pad
    # segment's positions run 0..m-1 and the rotary table only has
    # `max_model_len` entries. `spec.pad_sizes` is the greedy fill;
    # `spec.fits` above has already established that one exists.
    sizes = spec.pad_sizes(n_pad_seg, pad_tokens)
    if sizes is None:  # pragma: no cover - `fits` is checked immediately above
        raise ValueError(
            f"pad_mixed_step: {pad_tokens} padding tokens do not fit {n_pad_seg} rows "
            f"of at most {spec.pad_cap} each"
        )

    tok = [list(ids) for ids in chunk_token_ids] + [[PAD_TOKEN_ID] * m for m in sizes]
    pos = list(chunk_start_pos) + [0] * n_pad_seg
    slots = list(chunk_slots) + [spec.scratch_slot] * n_pad_seg

    n_pad_rows = bucket - n_rows
    return PaddedMixedInputs(
        token_ids=tok,
        start_positions=pos,
        slots=slots,
        decode_slots=list(decode_slots) + [spec.scratch_slot] * n_pad_rows,
        decode_token_ids=list(decode_token_ids) + [PAD_TOKEN_ID] * n_pad_rows,
        decode_positions=list(decode_positions) + [0] * n_pad_rows,
        n_real_segments=n_real_seg,
        n_real_rows=n_rows,
        bucket=bucket,
    )


def scratch_page_floor(chunk_tokens: int, page_size: int, max_pad_len: int = 0) -> int:
    """Pages the scratch slot needs so a graphed mixed step can be padded.

    Padding rule 1 puts up to ``chunk_tokens`` tokens of padding on the
    scratch slot in a single step, and a padded query row needs
    ``kv_len >= q_len`` real keys under FlashInfer's bottom-right causal
    alignment. So the slot needs ``ceil(chunk_tokens / page_size)`` pages --
    and the KV page table is ``[max_seqs, max_pages_per_seq]``, so
    ``max_pages_per_seq`` has to be at least that too.

    It is not, by default: ``max_pages_per_seq`` is derived from
    ``max_model_len`` (``bench_runtime.serving_pool_sizes``: 173 at
    ``max_model_len=2752``, ``page_size=16``), which is smaller than a
    4,096-token chunk needs; without raising the cap, graphed 4,096- and
    8,192-token chunks fail in ``prepare()`` with
    ``slot 256 needs 256 pages (> max_pages_per_seq=173)``.

    Raising the cap costs ``max_seqs * extra * 4`` bytes of page table (88 KiB
    at 264 slots and chunk 8,192), which is why the callers raise it rather
    than refusing the configuration.
    """
    if max_pad_len:
        chunk_tokens = min(int(chunk_tokens), int(max_pad_len))
    return -(-int(chunk_tokens) // int(page_size))


def reset_scratch_state(model: FusedQwenForCausalLM) -> None:
    """Zero the scratch slot's SSM + conv state (never its KV pages).

    Every padding row and padding segment threads its recurrence through this
    one slot, and nothing ever reads the result. Zeroing it per step is two
    launches and buys two things: the padding rows can never accumulate an
    Inf/NaN state (which would be harmless -- it stays in rows the sampler
    never sees -- but would make a "did padding change anything?" test
    unreproducible), and the padded step is a pure function of its real inputs.
    """
    idx = torch.tensor([model.scratch_slot], dtype=torch.long, device=model.device)
    model.state_pool.index_fill_(0, idx, 0)
    model.conv_pool.index_fill_(0, idx, 0)


# --------------------------------------------------------------------------- #
# 2. one captured shape
# --------------------------------------------------------------------------- #
@dataclass
class _Shape:
    """Everything one ``(chunk_tokens, bucket)`` graph owns."""

    bucket: int
    n_tokens: int  # chunk_tokens + bucket
    ctx: StepContext
    token_ids: torch.Tensor
    positions: torch.Tensor
    slot_ids: torch.Tensor
    decode_slot_ids: torch.Tensor
    logits_indices: torch.Tensor
    graphs: List["torch.cuda.CUDAGraph"] = field(default_factory=list)
    holes: List[Callable[[], torch.Tensor]] = field(default_factory=list)
    logits: Optional[torch.Tensor] = None
    hidden: Optional[torch.Tensor] = None
    captured: bool = False
    #: Real (non-duplicate) rows in the static ``chunk_indices`` buffer
    #: on the last step loaded. Reported by :meth:`MixedGraphRunner.stats`;
    #: nothing in the forward reads it.
    n_real_chunk_rows: int = 0


# --------------------------------------------------------------------------- #
# 3. the runner
# --------------------------------------------------------------------------- #
class MixedGraphRunner:
    """Capture/replay one mixed step per decode bucket, or run it eagerly.

    Mirrors :class:`~.graphs.GraphedDecoder`'s contract one level up: the
    scheduler pads the step, calls :meth:`step`, and gets logits back; the
    host-side plan (``AttentionRunner.plan_mixed_graph``) runs outside the
    graph on every step, and the eager holes run between the segments.

    ``chunk_tokens`` is fixed at construction because the scheduler serves one
    ``--prefill-chunk-tokens`` value: capturing a second one would double both
    the graph pool and the (shared) hole buffer for a shape nothing runs.
    """

    def __init__(
        self,
        model: FusedQwenForCausalLM,
        rt: RuntimeConfig,
        *,
        chunk_tokens: int,
        buckets: Optional[Sequence[int]] = None,
        n_segments: Optional[int] = None,
        pool_handle=None,
        holes: Optional[bool] = None,
        min_bucket: Optional[int] = None,
    ):
        self.model = model
        self.rt = rt
        self.device = model.device
        self.chunk_tokens = int(chunk_tokens)
        self.n_segments = int(n_segments or rt.mixed_graph_segments)
        buckets = tuple(buckets) if buckets is not None else rt.buckets_for()
        # Drop the tiny buckets. Each one is a whole extra capture of a
        # `chunk_tokens + b` step, and a step with 5 decode rows pads up to 32
        # exactly as happily as it pads to 8 -- so the ladder below the floor
        # buys nothing and costs graphs. The largest bucket is always kept,
        # whatever the floor, or `bucket_for` would refuse the widest steps.
        floor = max(1, int(
            min_bucket if min_bucket is not None
            else (getattr(rt, "mixed_graph_min_bucket", 1) or 1)
        ))
        wanted = sorted({int(b) for b in buckets if int(b) > 0})
        kept = [b for b in wanted if b >= floor]
        if wanted and not kept:
            kept = [wanted[-1]]
        self.buckets = tuple(kept)
        self.max_pad_len = int(getattr(rt, "max_model_len", 0) or 0)
        self.spec = MixedPadSpec(
            chunk_tokens=self.chunk_tokens,
            n_segments=self.n_segments,
            buckets=self.buckets,
            scratch_slot=model.scratch_slot,
            max_pad_len=self.max_pad_len,
        )
        if self.spec.max_segments < 1:
            raise ValueError(
                f"MixedGraphRunner: prefill_chunk_tokens={self.chunk_tokens} needs "
                f"{self.spec.min_pad_rows} of the {self.n_segments} prefill plan rows "
                f"for padding (a pad segment cannot exceed max_model_len="
                f"{self.max_pad_len} without indexing past the rotary table), leaving "
                f"none for real segments. Raise --mixed-graph-segments to at least "
                f"{self.spec.min_pad_rows + 1}, lower --prefill-chunk-tokens, or raise "
                f"--max-model-len."
            )
        # A padding segment lives on the scratch slot at positions
        # `0..m-1`, so it indexes the rotary table at `m-1`. The table has
        # `max_model_len` entries, so a pad segment longer than that walks off
        # the end of an `index_select` -- a **device-side assert**, which
        # poisons the CUDA context and then surfaces as
        # `CUBLAS_STATUS_EXECUTION_FAILED` from whatever GEMM runs next.
        # `MixedPadSpec.max_pad_len` caps it and `pad_sizes` splits the padding
        # across the free plan rows instead of putting it all in one; the cap
        # on the *chunk* is then `(n_segments - 1) * max_pad_len`, which is
        # 19,264 at the serving geometry rather than 2,752.
        self._pool = pool_handle
        self._own_pool = pool_handle is None
        self._shapes: Dict[int, _Shape] = {}
        self._hole_buf: Optional[torch.Tensor] = None
        self._cur: Optional[_Shape] = None
        self._cm = None
        self._graph: Optional["torch.cuda.CUDAGraph"] = None
        self._prepared = False
        self.captured = False

        # -- one graph, no holes --------------------------------------------- #
        # `holes=True` is the 49-segment/48-hole capture, kept as a fallback
        # and as a comparison baseline.
        # `holes=False` needs fla driven with caller-supplied index tensors;
        # if that is unavailable the runner says so and degrades to holes
        # rather than capturing something it cannot capture.
        want_holes = bool(rt.mixed_graph_holes if holes is None else holes)
        self.static_index_reason: Optional[str] = None
        if not want_holes and not fla_static.is_available():
            self.static_index_reason = (
                fla_static.unavailable_reason() or "fla static-index path unavailable"
            )
            want_holes = True
        self.holes = want_holes
        self.gdn_chunk_size = int(rt.gdn_chunk_size)
        #: Rows in the static ``chunk_indices`` buffer. Every segmentation of
        #: ``chunk_tokens`` tokens into ``n_segments`` non-empty segments needs
        #: at most this many, and the surplus rows are filled with a copy of
        #: the last real one (``fla_static.build_chunk_meta``).
        self.n_chunk_rows = fla_static.max_chunk_rows(
            self.chunk_tokens, self.n_segments, self.gdn_chunk_size
        )
        # Runner-owned, shared by every bucket: the segmentation is the same
        # tensor for all of them, and it is read *inside* the graph now, so it
        # has to live at a fixed address instead of being a fresh object per
        # step the way the segmented mode needs it to be.
        self._cu_seqlens: Optional[torch.Tensor] = None
        self._seq_slot_ids: Optional[torch.Tensor] = None
        self._chunk_indices: Optional[torch.Tensor] = None
        self._chunk_offsets: Optional[torch.Tensor] = None
        self._host: Dict[str, torch.Tensor] = {}

    # -- capability ---------------------------------------------------------- #
    @property
    def graphs_enabled(self) -> bool:
        return (
            self.device.type == "cuda"
            and torch.cuda.is_available()
            and bool(self.rt.use_cuda_graphs)
            and self.model.attn.backend == "flashinfer"
        )

    def ready(self, n_rows: int) -> bool:
        """Can :meth:`step` replay a graph for ``n_rows`` decode rows?"""
        b = self.spec.bucket_for(n_rows)
        return bool(self.captured and b is not None and self._shapes.get(b, None) is not None
                    and self._shapes[b].captured)

    # -- setup --------------------------------------------------------------- #
    def prepare(self) -> None:
        """Give the scratch slot enough KV pages for a full chunk of padding.

        ``chunk_tokens`` of padding may land on it in one step (rule 1), and a
        padded query row needs ``kv_len >= q_len`` keys to attend to. Idempotent
        (``ensure_capacity`` never shrinks), and paid once at startup so no
        step ever allocates pages for padding.
        """
        if self._prepared:
            return
        # The scratch slot holds one pad segment at a time at positions
        # `0..m-1`, and `m <= max_pad_len`, so it needs pages for the
        # longest *segment*, not for the whole chunk. At chunk 8,192 that is
        # 172 pages instead of 512.
        self.model.kv_pool.ensure_capacity(
            self.model.scratch_slot, min(self.chunk_tokens, self.spec.pad_cap)
        )
        self._alloc_index_buffers()
        self._prepared = True

    # -- the step's segmentation, at fixed addresses ------------------------- #
    def _alloc_index_buffers(self) -> None:
        """Allocate the four static tensors that describe the segmentation.

        ``cu_seqlens`` and ``seq_slot_ids`` are per-step objects in the
        segmented mode because only the eager hole reads them. In the one-graph
        mode they are read *inside* the graph (the conv indexes with ``cu_seqlens``; the state gather/scatter
        indexes with ``seq_slot_ids``), so they have to live at addresses the
        capture can bake in -- and the contents are refreshed by an async copy
        from pinned host memory before every replay.

        Pinned staging on the host is what keeps that copy off the critical
        path: a pageable H2D would stage through a driver bounce buffer and
        block, which is exactly the host cost graphing exists to remove.
        """
        if self._cu_seqlens is not None:
            return
        d = self.device
        n = self.n_segments
        dt = fla_static.INDEX_DTYPE
        self._cu_seqlens = torch.zeros(n + 1, dtype=dt, device=d)
        self._seq_slot_ids = torch.full(
            (n,), self.model.scratch_slot, dtype=torch.int32, device=d
        )
        self._chunk_indices = torch.zeros(self.n_chunk_rows, 2, dtype=dt, device=d)
        self._chunk_offsets = torch.zeros(n + 1, dtype=dt, device=d)
        # Only the two *derived* tensors need a host staging buffer.
        # `cu_seqlens` and `seq_slot_ids` already exist on the device as part
        # of the batch, so refreshing them is a 36-byte D2D copy -- no host
        # round trip at all.
        pin = d.type == "cuda"
        self._host = {
            "chunk_indices": torch.zeros(self.n_chunk_rows, 2, dtype=dt, pin_memory=pin),
            "chunk_offsets": torch.zeros(n + 1, dtype=dt, pin_memory=pin),
        }

    def _upload_segmentation(self, batch: MixedBatch) -> int:
        """Fill the four static buffers from this step's segmentation.

        Pure host arithmetic over ``q_lens`` (which the scheduler built the
        chunk from) plus four async copies out of pinned memory. No device
        read, no ``.item()``, no ``.tolist()`` -- the property that makes the
        GDN prefill half capturable. Returns the number of *real* chunk rows,
        for the stats/tests.
        """
        q_lens = [int(n) for n in batch.prefill.q_lens]
        idx, off, n_real = fla_static.build_chunk_meta(
            q_lens, self.gdn_chunk_size, self.n_chunk_rows
        )
        h = self._host
        h["chunk_indices"].copy_(torch.tensor(idx, dtype=fla_static.INDEX_DTYPE))
        h["chunk_offsets"].copy_(torch.tensor(off, dtype=fla_static.INDEX_DTYPE))
        nb = self.device.type == "cuda"
        # D2D: the batch already holds both of these on the device.
        self._cu_seqlens.copy_(batch.prefill.cu_seqlens, non_blocking=nb)      # type: ignore[union-attr]
        self._seq_slot_ids.copy_(batch.prefill.seq_slot_ids, non_blocking=nb)  # type: ignore[union-attr]
        # H2D out of pinned staging.
        self._chunk_indices.copy_(h["chunk_indices"], non_blocking=nb)  # type: ignore[union-attr]
        self._chunk_offsets.copy_(h["chunk_offsets"], non_blocking=nb)  # type: ignore[union-attr]
        return n_real

    def _shape(self, bucket: int) -> _Shape:
        sh = self._shapes.get(bucket)
        if sh is not None:
            return sh
        self._alloc_index_buffers()
        t = self.chunk_tokens + bucket
        d = self.device
        i32 = dict(dtype=torch.int32, device=d)
        token_ids = torch.zeros(t, **i32)
        positions = torch.zeros(t, **i32)
        slot_ids = torch.full((t,), self.model.scratch_slot, **i32)
        decode_slot_ids = torch.full((bucket,), self.model.scratch_slot, **i32)
        logits_indices = torch.zeros(self.n_segments + bucket, dtype=torch.long, device=d)
        ctx = self.model._context(  # noqa: SLF001 -- the runner *is* the model's step driver
            slot_ids,
            positions,
            seq_slot_ids=self._seq_slot_ids if not self.holes else None,
            cu_seqlens=self._cu_seqlens if not self.holes else None,
            q_lens=None,
            n_prefill_tokens=self.chunk_tokens,
            decode_slot_ids=decode_slot_ids,
            rotary=False,
            # Set only in the one-graph mode; with them the GDN prefill
            # half reads its segmentation entirely off the device and captures.
            chunk_indices=None if self.holes else self._chunk_indices,
            chunk_offsets=None if self.holes else self._chunk_offsets,
            conv_max_seqlen=None if self.holes else self.chunk_tokens,
        )
        ctx.graph_safe_kv = True
        sh = _Shape(
            bucket=bucket,
            n_tokens=t,
            ctx=ctx,
            token_ids=token_ids,
            positions=positions,
            slot_ids=slot_ids,
            decode_slot_ids=decode_slot_ids,
            logits_indices=logits_indices,
        )
        self._shapes[bucket] = sh
        return sh

    # -- the synthetic all-padding step used for warmup and capture ---------- #
    def _synthetic(self, bucket: int) -> MixedBatch:
        from .fused_model import make_mixed_batch, make_prefill_batch

        # `replace(self.spec, ...)`, never a fresh `MixedPadSpec`: a spec built
        # positionally here would silently drop `max_pad_len` and hand the
        # warmup one 8,185-token pad segment on the scratch slot, which is a
        # `kv_len` of 8,185 -- 512 pages against a 173-wide page table, and an
        # `IndexError` inside `build_flashinfer_indices` a long way from here.
        # The synthetic batch must be the *same shape* the real steps are.
        pad = pad_mixed_step([[PAD_TOKEN_ID]], [0], [self.model.scratch_slot],
                             [self.model.scratch_slot], [PAD_TOKEN_ID], [0],
                             replace(self.spec, buckets=(bucket,)))
        prefill = make_prefill_batch(pad.token_ids, pad.start_positions, pad.slots, self.device)
        return make_mixed_batch(prefill, pad.decode_slots, pad.decode_token_ids,
                                pad.decode_positions, self.device)

    def _load(self, sh: _Shape, batch: MixedBatch) -> None:
        """Copy this step's inputs into the shape's static buffers and point
        the persistent ``StepContext`` at this step's (fresh) varlen metadata.

        ``cu_seqlens`` and ``cu_seqlens_cpu`` are deliberately **new tensor
        objects** every step: fla's ``prepare_chunk_indices`` memoises on
        argument identity, so reusing one buffer with new contents would hand
        the chunk kernel the previous step's segmentation. They are only ever
        read in the eager hole, so a fresh object costs nothing.
        """
        if sh.n_tokens != int(batch.token_ids.shape[0]):
            raise ValueError(
                f"MixedGraphRunner: batch has {int(batch.token_ids.shape[0])} rows, "
                f"graph shape is {sh.n_tokens} (chunk {self.chunk_tokens} + bucket {sh.bucket})"
            )
        if int(batch.n_prefill_tokens) != self.chunk_tokens:
            raise ValueError(
                f"MixedGraphRunner: batch has {batch.n_prefill_tokens} prefill tokens, "
                f"graph shape is {self.chunk_tokens} -- the caller must pad (pad_mixed_step)"
            )
        if batch.n_prefill_seqs != self.n_segments:
            raise ValueError(
                f"MixedGraphRunner: batch has {batch.n_prefill_seqs} prefill segments, "
                f"graph shape is exactly {self.n_segments}"
            )
        sh.token_ids.copy_(batch.token_ids, non_blocking=True)
        sh.positions.copy_(batch.positions, non_blocking=True)
        sh.slot_ids.copy_(batch.slot_ids, non_blocking=True)
        sh.decode_slot_ids.copy_(batch.decode_slot_ids, non_blocking=True)
        sh.logits_indices.copy_(batch.logits_indices, non_blocking=True)
        ctx = sh.ctx
        ctx.q_lens = list(batch.prefill.q_lens)
        acc, cu = 0, [0]
        for n in ctx.q_lens:
            acc += int(n)
            cu.append(acc)
        ctx.cu_seqlens_cpu = torch.tensor(cu, dtype=torch.int64, device="cpu")
        if self.holes:
            # Segmented mode. `cu_seqlens`/`cu_seqlens_cpu` are deliberately **new
            # tensor objects** every step here: the hole calls fla's public
            # entry point, whose `prepare_chunk_indices` memoises on argument
            # identity, so reusing one buffer with new contents would hand the
            # chunk kernel the previous step's segmentation. They are only ever
            # read in the eager hole, so a fresh object costs nothing.
            ctx.seq_slot_ids = batch.prefill.seq_slot_ids
            ctx.cu_seqlens = batch.prefill.cu_seqlens
        else:
            # One-graph mode. The exact opposite: both are *persistent* buffers, because
            # they are read inside the captured region. Nothing memoises on
            # them any more -- `fla_static` supplies the index tensors and
            # forbids fla from deriving any.
            sh.n_real_chunk_rows = self._upload_segmentation(batch)

    # -- warmup / capture ----------------------------------------------------- #
    def warmup(self, iters: int = 2) -> None:
        """Run each bucket's shape eagerly, through the real body.

        Same two jobs as ``GraphedDecoder.warmup``: pin every
        ``ResolvedLinear``'s GEMM backend and let Triton/fla/FlashInfer compile
        and autotune, both of which are host-side work that must not happen
        inside a capture. Third job here: build this shape's graph-mode
        FlashInfer prefill wrapper and plan it once.
        """
        self.prepare()
        for b in self.buckets:
            sh = self._shape(b)
            batch = self._synthetic(b)
            for _ in range(max(1, iters)):
                reset_scratch_state(self.model)
                self._load(sh, batch)
                self.model.attn.plan_mixed_graph(
                    batch.plan_slots, batch.plan_q_lens, batch.plan_kv_lens
                )
                self._run_body(sh, graph_break=None)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def _run_body(self, sh: _Shape, graph_break):
        # Recomputed every pass, never carried over: `cos`/`sin` are a gather
        # on *this* step's positions, and the whole point of leaving them None
        # in `_shape` is that the gather lands inside the captured region so a
        # replay re-does it against the refreshed `positions` buffer.
        sh.ctx.cos = None
        sh.ctx.sin = None
        sh.ctx.graph_break = graph_break
        try:
            out = self.model.mixed_forward_body(
                sh.ctx, sh.token_ids, sh.logits_indices, return_hidden=True
            )
        finally:
            sh.ctx.graph_break = None
        return out

    def capture(self, pool_handle=None) -> None:
        """Capture one segmented graph per bucket into a shared mempool.

        No-op unless :attr:`graphs_enabled`. The FlashInfer requirement is the
        same one ``GraphedDecoder.capture`` documents and for the same reason:
        the torch attention fallback syncs inside the region.

        ``pool_handle``: share the decode graphs' mempool (what
        ``QwenFastEngine.start`` passes, the same way it shares it with the
        speculative step). Safe because only one graph ever replays at a time
        and the only tensors read *after* a replay -- ``logits``/``hidden`` --
        are read before the next one starts.
        """
        if not self.graphs_enabled:
            return
        if pool_handle is not None:
            self._pool = pool_handle
            self._own_pool = False
        self.prepare()
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        for b in self.buckets:
            sh = self._shape(b)
            batch = self._synthetic(b)
            reset_scratch_state(self.model)
            self._load(sh, batch)
            self.model.attn.plan_mixed_graph(
                batch.plan_slots, batch.plan_q_lens, batch.plan_kv_lens
            )
            # One more eager pass immediately before capture, on a side stream,
            # for the same reason GraphedDecoder does it: anything lazily
            # allocated or JIT-compiled by this exact shape must happen now.
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self._run_body(sh, graph_break=None)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()

            self._cur = sh
            sh.graphs = []
            sh.holes = []
            try:
                self._begin()
                # `graph_break=None` in the one-graph mode. `_break` is
                # never called, `sh.holes` stays empty, and `sh.graphs` ends up
                # with exactly one entry -- which is what `step` replays.
                logits, hidden = self._run_body(
                    sh, graph_break=(self._break if self.holes else None)
                )
                self._end()
            except BaseException:
                # A capture left half-open poisons every later CUDA call with
                # "operation not permitted when stream is capturing". Close it
                # before the exception leaves this method, then let it out --
                # a mixed step that silently fell back to eager would make a
                # benchmark measure the wrong configuration.
                if self._cm is not None:
                    try:
                        self._cm.__exit__(None, None, None)
                    except BaseException:  # pragma: no cover - best effort
                        pass
                    self._cm = self._graph = None
                self._cur = None
                sh.graphs, sh.holes, sh.captured = [], [], False
                raise
            self._cur = None
            sh.logits, sh.hidden = logits, hidden
            sh.captured = True
        self.captured = True

    # -- segmented capture mechanics ------------------------------------------ #
    def _begin(self) -> None:
        g = torch.cuda.CUDAGraph()
        self._graph = g
        self._cm = torch.cuda.graph(g, pool=self._pool)
        self._cm.__enter__()

    def _end(self) -> None:
        assert self._cm is not None and self._graph is not None and self._cur is not None
        self._cm.__exit__(None, None, None)
        self._cur.graphs.append(self._graph)
        self._cm = None
        self._graph = None

    def _break(self, fn: Callable[[], torch.Tensor]) -> torch.Tensor:
        """End the current graph, run ``fn`` eagerly, start the next one.

        ``fn``'s inputs are tensors the *previous* segment produced, which live
        in the graph's private pool and therefore have addresses that survive
        every replay -- that is the whole reason this works. Its output does
        not (it is a fresh eager allocation each call), so it is copied into
        one shared static buffer that the next segment reads.

        One buffer for all 48 holes, not 48 buffers: the segments run strictly
        in order on one stream, and segment ``i+1`` consumes the hole's output
        in its first few kernels, long before hole ``i+1`` overwrites it. At
        ``chunk_tokens=2048`` that is 25 MiB held instead of 1.2 GiB.
        """
        sh = self._cur
        assert sh is not None
        self._end()
        out = fn()
        buf = self._hole_buffer(out)
        buf.copy_(out)
        sh.holes.append(fn)
        self._begin()
        return buf

    def _hole_buffer(self, like: torch.Tensor) -> torch.Tensor:
        if self._hole_buf is None:
            self._hole_buf = torch.empty(
                like.shape, dtype=like.dtype, device=like.device
            )
        elif tuple(self._hole_buf.shape) != tuple(like.shape) or self._hole_buf.dtype != like.dtype:
            raise RuntimeError(
                "MixedGraphRunner: the GDN prefill hole changed shape/dtype between "
                f"layers ({tuple(self._hole_buf.shape)}/{self._hole_buf.dtype} -> "
                f"{tuple(like.shape)}/{like.dtype}); the shared hole buffer assumes it "
                "cannot (every GDN layer has the same [T_pre, HV, V] core)"
            )
        return self._hole_buf

    # -- the per-step entry point --------------------------------------------- #
    def step(self, batch: MixedBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        """One mixed step for an already-padded ``batch``. Returns
        ``(logits, hidden)``.

        Both are views into the graph's private pool: read them (sample, MTP
        bookkeeping) **before** replaying any other graph, exactly as the
        decode step's ``out_tokens`` must be harvested before the next replay.
        """
        return self.replay(self.prepare_step(batch))

    # -- the same step, with plan and replay split ---------------------------- #
    def prepare_step(self, batch: MixedBatch) -> int:
        """Everything :meth:`step` does on the **host**: load the step's inputs
        into the shape's static buffers and plan FlashInfer. Returns the bucket.

        Split out for ``--overlap``, which has to plan *both*
        halves of a step before it launches either -- once the two graphs are
        in flight on two streams, no host work may sit between them.
        """
        n_rows = batch.n_decode_rows
        bucket = self.spec.bucket_for(n_rows)
        if bucket is None:
            raise ValueError(f"MixedGraphRunner: no bucket for {n_rows} decode rows")
        sh = self._shapes.get(bucket)
        if sh is None or not sh.captured:
            raise RuntimeError(f"MixedGraphRunner: bucket {bucket} was never captured")
        self._load(sh, batch)
        # Host-side, outside the graph -- the mixed step's analogue of
        # `plan_decode`, and the only per-step FlashInfer work.
        self.model.attn.plan_mixed_graph(
            batch.plan_slots, batch.plan_q_lens, batch.plan_kv_lens
        )
        return bucket

    def replay(self, bucket: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """The device half of :meth:`step`: replay ``bucket``'s graph(s).

        Launches on the **current** stream, so a caller that wants this on a
        side stream just wraps it in ``torch.cuda.stream(...)``.
        """
        sh = self._shapes[bucket]
        if not self.holes:
            # One-graph mode: one replay. Everything the step does, including the varlen
            # conv and the fla chunk kernel, is inside it.
            sh.graphs[0].replay()
            return sh.logits, sh.hidden  # type: ignore[return-value]
        holes = sh.holes
        for i, g in enumerate(sh.graphs):
            g.replay()
            if i < len(holes):
                self._hole_buf.copy_(holes[i]())  # type: ignore[union-attr]
        return sh.logits, sh.hidden  # type: ignore[return-value]

    # -- introspection (the profiler and the tests) --------------------------- #
    def stats(self) -> Dict[str, float]:
        return {
            "chunk_tokens": float(self.chunk_tokens),
            "n_segments": float(self.n_segments),
            "holes_mode": float(self.holes),
            "static_indices": float(not self.holes),
            "chunk_index_rows": float(self.n_chunk_rows),
            "real_chunk_rows": float(
                max((s.n_real_chunk_rows for s in self._shapes.values()), default=0)
            ),
            "gdn_chunk_size": float(self.gdn_chunk_size),
            "buckets": float(len(self.buckets)),
            "captured_shapes": float(sum(1 for s in self._shapes.values() if s.captured)),
            "graph_segments": float(
                max((len(s.graphs) for s in self._shapes.values()), default=0)
            ),
            "eager_holes": float(
                max((len(s.holes) for s in self._shapes.values()), default=0)
            ),
            "hole_buffer_mib": float(
                0.0 if self._hole_buf is None
                else self._hole_buf.numel() * self._hole_buf.element_size() / (1024.0 ** 2)
            ),
        }


__all__ = [
    "PAD_TOKEN_ID",
    "scratch_page_floor",
    "MixedPadSpec",
    "PaddedMixedInputs",
    "pad_mixed_step",
    "reset_scratch_state",
    "MixedGraphRunner",
]
