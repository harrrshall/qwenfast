"""CUDA-graph capture per batch-size bucket + the graph-safe sampler.

One CUDA graph per bucket in
``RuntimeConfig.graph_buckets`` captures the **entire** decode step — embed
gather -> 64 layers -> final norm -> lm_head -> sampler -- against the fixed
:class:`~qwenfast.runtime.fused_model.DeviceBuffers` pointers, sharing one
``torch.cuda.graph_pool_handle()`` across all buckets so graph memory does
not scale with the number of buckets (~1 GiB total, not ~1 GiB x 15).

What is captured vs. what is not (this is the one correctness-critical
split in this module):

* **Captured** (``_run_step``): ``model.decode_forward`` and
  :func:`sample_tokens`, both pure fixed-shape tensor ops with no
  data-dependent control flow and no host sync.
* **Not captured, must run before every replay**: ``AttentionRunner.plan_decode``
  (host-side FlashInfer planning -- see ``fused_model.AttentionRunner``) and
  filling ``buf.input_ids``/``positions``/``slot_ids`` from the scheduler's
  own bookkeeping. The scheduler (``scheduler.py``) owns that "harvest
  ``out_tokens`` -> build next step's inputs on the host -> upload" loop;
  this module intentionally does **not** try to carry a
  sampled token over into the next step's ``input_ids`` inside the graph,
  because which row of the batch holds which sequence's slot can change
  between two decode steps under continuous batching (a request finishes,
  a new one is admitted into that row) -- reusing the graph's own output
  buffer as next-step input would silently feed the wrong slot's last token
  into a graph replay whose ``slot_ids``/``positions`` say otherwise.

``--no-graphs``: when ``RuntimeConfig.use_cuda_graphs`` is False, or the
device is not CUDA (e.g. a CPU-only laptop), :meth:`GraphedDecoder.capture` is a no-op
and :meth:`GraphedDecoder.step` always takes the eager path -- the exact
same ``_run_step`` body, just launched instead of replayed. This is what
makes the CPU test suite exercise this module's sampler and step-selection
logic without ever touching ``torch.cuda``.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch

from .fused_model import DeviceBuffers, FusedQwenForCausalLM, RuntimeConfig

# --------------------------------------------------------------------------- #
# 1. the graph-safe sampler
# --------------------------------------------------------------------------- #
def sample_tokens(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: torch.Tensor,
    *,
    candidates: int = 2048,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Fused greedy / temperature / top-p / top-k sampler. Torch ops only.

    ``logits``: ``[B, V]`` fp32. ``temperature``/``top_p``/``top_k``: ``[B]``
    fp32 (``top_k <= 0`` means "disabled", matching ``DeviceBuffers.top_k``'s
    convention). Returns ``[B]`` int32 token ids.

    No ``.item()``, no ``if tensor:``, no ``torch.multinomial`` (whose CPU
    path syncs and whose CUDA path is not guaranteed graph-capturable across
    torch versions) -- every row is processed identically via masking and a
    per-row Gumbel-max draw, so the whole function is a straight-line
    sequence of fixed-shape tensor ops and is safe to capture inside a CUDA
    graph.

    Bounded to the top ``candidates`` logits per row (``RuntimeConfig
    .sampler_candidates``, default 2048) before any sort: at ``vocab ==
    248320`` a full per-row sort every decode step would dominate the step
    time at large batch. Restricting to the top-2048 candidates first (a
    single ``topk``, itself well-optimized) then doing top-p/top-k *within*
    that candidate set is exact for every temperature/top_p/top_k
    combination that would ever keep fewer than 2048 tokens, which is true
    of every sane sampling config (top_k defaults are O(20-100), top_p
    rarely needs more than a few hundred candidates in practice).
    """
    b, v = logits.shape
    k = min(int(candidates), v)
    topv, topi = torch.topk(logits.float(), k, dim=-1, sorted=True)  # [B, k] descending

    greedy = topi[:, 0]

    temp = temperature.to(logits.device, torch.float32).clamp(min=1e-5).unsqueeze(-1)
    scaled = topv / temp

    # -- top-k mask (within the candidate window) --------------------------- #
    ar = torch.arange(k, device=logits.device).unsqueeze(0)  # [1, k]
    tk = top_k.to(logits.device, torch.float32).unsqueeze(-1)  # [B, 1]
    tk_eff = torch.where(tk <= 0, torch.full_like(tk, float(k)), tk.clamp(max=float(k)))
    keep_k = ar < tk_eff

    # -- top-p / nucleus mask ------------------------------------------------ #
    probs = torch.softmax(scaled.masked_fill(~keep_k, float("-inf")), dim=-1)
    cum_excl = probs.cumsum(dim=-1) - probs  # probability mass strictly before this token
    tp = top_p.to(logits.device, torch.float32).clamp(min=1e-5, max=1.0).unsqueeze(-1)
    keep_p = cum_excl < tp
    keep_p = keep_p.clone()
    keep_p[:, 0] = True  # always keep the top-1 candidate, even if top_p is tiny

    keep = keep_k & keep_p
    filtered = scaled.masked_fill(~keep, float("-inf"))

    if generator is not None and generator.device == filtered.device:
        u = torch.rand(filtered.shape, dtype=torch.float32, device=filtered.device, generator=generator)
    else:
        u = torch.rand(filtered.shape, dtype=torch.float32, device=filtered.device)
    u = u.clamp(min=1e-20, max=1.0 - 1e-7)
    gumbel = -torch.log(-torch.log(u))
    sampled_local = (filtered + gumbel).argmax(dim=-1, keepdim=True)  # [B, 1]
    sampled = topi.gather(-1, sampled_local).squeeze(-1)

    is_greedy = temperature.to(logits.device, torch.float32) <= 0
    out = torch.where(is_greedy, greedy, sampled)
    return out.to(torch.int32)


# --------------------------------------------------------------------------- #
# 2. the graphed decode step
# --------------------------------------------------------------------------- #
class GraphedDecoder:
    """Captures/replays one decode step per bucket, or runs it eagerly.

    ``step(batch, slots)`` is the one entry point the scheduler calls: it
    rounds ``batch`` up to a bucket, runs ``AttentionRunner.plan_decode``
    (host-side, always eager), then either replays that bucket's captured
    graph or runs the step body directly -- transparently to the caller.
    Returns a view of ``buf.out_tokens[:bucket]`` (sampled token ids); the
    caller is responsible for harvesting only the first ``batch`` of those
    (the rest are padding rows pointed at the scratch slot).
    """

    def __init__(self, model: FusedQwenForCausalLM, buf: DeviceBuffers, rt: RuntimeConfig):
        self.model = model
        self.buf = buf
        self.rt = rt
        self.device = model.device
        self.buckets = tuple(rt.buckets_for()) if rt.use_cuda_graphs else ()
        self._graphs: Dict[int, "torch.cuda.CUDAGraph"] = {}
        self._pool = None
        self._captured = False
        self._generator: Optional[torch.Generator] = None
        if self.device.type == "cuda":
            self._generator = torch.Generator(device=self.device)
            self._generator.manual_seed(0)

    @property
    def graphs_enabled(self) -> bool:
        return bool(self.buckets) and self.device.type == "cuda" and torch.cuda.is_available()

    def bucket_for(self, batch: int) -> int:
        buckets = self.buckets or self.rt.buckets_for()
        for b in buckets:
            if batch <= b:
                return b
        return max(buckets[-1], batch) if buckets else batch

    # -- the captured/eager step body ---------------------------------------- #
    def _run_step(self, bucket: int) -> None:
        self.model.decode_forward(self.buf, bucket, write_logits=True)
        logits = self.buf.logits[:bucket]
        tok = sample_tokens(
            logits,
            self.buf.temperature[:bucket],
            self.buf.top_p[:bucket],
            self.buf.top_k[:bucket],
            candidates=self.rt.sampler_candidates,
            generator=self._generator,
        )
        self.buf.out_tokens[:bucket].copy_(tok)

    # -- capture -------------------------------------------------------------- #
    def _fill_scratch_inputs(self, bucket: int) -> None:
        scratch = self.model.scratch_slot
        self.buf.host["input_ids"][:bucket].zero_()
        self.buf.host["positions"][:bucket].zero_()
        self.buf.host["slot_ids"][:bucket].fill_(scratch)
        self.buf.host["temperature"][:bucket].fill_(1.0)
        self.buf.host["top_p"][:bucket].fill_(1.0)
        self.buf.host["top_k"][:bucket].zero_()
        self.buf.upload(["input_ids", "positions", "slot_ids", "temperature", "top_p", "top_k"])
        self.model.attn.plan_decode([scratch] * bucket, bucket)

    def warmup(self, iters: int = 2) -> None:
        """Run each bucket eagerly a few times first.

        Two jobs: (1) let :class:`~.fused_model.ResolvedLinear` resolve and
        pin its GEMM backend outside any capture (its docstring: an
        exception mid-capture leaves capture undefined), and (2) let
        cuDNN/cuBLAS algorithm selection settle before the shapes are frozen
        into a graph.
        """
        buckets = self.buckets if self.graphs_enabled else (self.rt.buckets_for() if self.rt.use_cuda_graphs else (self.rt.max_num_seqs,))
        for b in buckets:
            self._fill_scratch_inputs(b)
            for _ in range(iters):
                self._run_step(b)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def capture(self) -> None:
        """Capture one graph per bucket into a shared mempool. No-op unless
        ``graphs_enabled`` (CUDA device and ``use_cuda_graphs``).

        Requires the attention backend to be ``"flashinfer"``: its plan()-
        outside/run()-inside split (``fused_model.AttentionRunner``) is the
        only attention path in this codebase with no host sync inside the
        captured region. The torch/eager fallback
        (``attn.flashinfer_attn.torch_fallback_decode``) is the documented
        CPU/reference path -- it does a Python loop over ``slot_ids
        .tolist()``, an unconditional device-to-host sync, which CUDA graph
        capture forbids outright ("Cannot copy between CPU and CUDA tensors
        during CUDA graph capture unless the CPU tensor is pinned" is the
        failure it produces, three calls deep). Rather than let that surface
        as a cryptic low-level CUDA error, fail here with an actionable
        message: this is a real, structural constraint (only
        FlashInfer's decode kernel is graph-safe), not a bug to route around.
        """
        if not self.graphs_enabled:
            return
        if self.model.attn.backend != "flashinfer":
            raise RuntimeError(
                f"GraphedDecoder.capture(): attn_backend={self.model.attn.backend!r} is not "
                "graph-safe -- CUDA graph capture requires the FlashInfer decode backend "
                "(RuntimeConfig(attn_backend='flashinfer') or 'auto' on a host where flashinfer "
                "is importable). The torch/eager attention fallback does a host sync "
                "(slot_ids.tolist()) that CUDA graph capture forbids; use "
                "RuntimeConfig(use_cuda_graphs=False) if you need the fallback backend."
            )
        self._pool = torch.cuda.graph_pool_handle()
        for b in self.buckets:
            self._fill_scratch_inputs(b)
            # A private warmup stream side-steps the "graph capture must not
            # run on the default stream while other work is pending" trap.
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self._run_step(b)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()

            g = torch.cuda.CUDAGraph()
            # The sampler's torch.rand uses a private CUDA generator; a
            # generator that is not registered with the graph raises
            # "RNG op during graph capture but generator is not registered".
            # Registering it lets each replay advance
            # the philox offset exactly like eager.
            if self._generator is not None and self._generator.device.type == "cuda":
                g.register_generator_state(self._generator)
            with torch.cuda.graph(g, pool=self._pool):
                self._run_step(b)
            self._graphs[b] = g
        self._captured = True

    # -- the per-step entry point --------------------------------------------- #
    def step(
        self,
        batch: int,
        slots: Sequence[int],
        seq_lens: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """One decode step for ``batch`` live rows, ``slots`` already padded
        to this bucket's size by the caller.

        Preconditions (the scheduler's job, not this method's):
        ``buf.input_ids``/``positions``/``slot_ids``/sampling-param buffers
        already hold this step's data for ``[:bucket]``. Returns
        ``buf.out_tokens[:bucket]``.
        """
        return self.replay(self.prepare_step(batch, slots, seq_lens=seq_lens))

    # -- the same step, with plan and replay split ----------------------------- #
    def prepare_step(
        self,
        batch: int,
        slots: Sequence[int],
        seq_lens: Optional[Sequence[int]] = None,
    ) -> int:
        """The **host** half of :meth:`step` -- ``plan_decode`` -- returning the
        bucket :meth:`replay` should run.

        Split out for ``--overlap``: an overlapped step plans the
        prefill chunk *and* the decode rows before it launches either, because
        once both graphs are in flight there is nowhere to put host work.
        """
        bucket = self.bucket_for(batch)
        self.model.attn.plan_decode(list(slots), bucket, seq_lens=seq_lens)
        return bucket

    def replay(self, bucket: int) -> torch.Tensor:
        """The device half of :meth:`step`. Launches on the **current** stream,
        so a side-stream caller only has to wrap this in
        ``torch.cuda.stream(...)``."""
        if bucket in self._graphs:
            self._graphs[bucket].replay()
        else:
            self._run_step(bucket)
        return self.buf.out_tokens[:bucket]


__all__ = ["sample_tokens", "GraphedDecoder"]
