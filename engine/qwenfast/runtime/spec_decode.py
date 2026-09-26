"""MTP speculative decoding.

One speculative step, for ``B`` sequences and a draft length ``k`` (``n = k+1``
window positions per sequence)::

    x0 = the last committed token of each sequence, at position L (not yet
         seen by the main model);  h_prev = the main model's post-final-norm
         hidden at position L-1

    1. draft   k sequential MTP steps.  Step j consumes (embed(window[j]),
               prev_hidden) at position L+j and emits window[j+1].
               j=0 uses h_prev; j>0 uses the MTP block's own output hidden.
    2. verify  ONE main-model forward over the whole [B, n] window
               (packed [B*n, hidden]).  GDN layers use the window kernel
               with commit=False (1 state read, 0 writes) and cache their
               per-position (conv-input, k, v, g, beta); attention layers use
               a multi-token bottom-right-causal paged call.
    3. accept  greedy: accept draft j while it equals argmax(logits[j-1]).
               m = accepted + 1 in [1, n], a **device** int32 tensor.
    4. commit  second pass over the cached GDN inputs with ``m``: the
               ``gdn_commit`` (window kernel, MASK_PAST_M) writes S_m; the
               conv ring is moved to concat(state, x)[..., m : m+W-1].
    5. carry   h_prev <- hidden[m-1] (device gather), for the next step's draft.

Why the commit is a second pass (and not the fused one-pass kernel).
``kernels_gdn.api.gdn_verify_and_commit``'s own docstring spells out the
ordering constraint: the fused kernel needs ``m`` *before* it runs, and a chain
verifier cannot know ``m`` until the last layer has produced logits.  So this is
the two-pass shape: 2 state reads + 1 write, a 1.5x SSM-traffic multiplier,
using the fused kernel in its ``MASK_PAST_M`` commit mode for phase B.
Reaching 1.0x needs the ReplaySSM-style *deferred* commit (replay the previous
window's accepted prefix at the head of the next one), which changes the window
contents and is not implemented.

Rollback.  Nothing is ever "undone":

* **SSM / conv** — never speculatively written at all.  Phase A does not write
  the state (``gdn_verify``), and the conv is called with ``m = 0``, for which
  ``causal_conv_verify_and_commit``'s ring update is the identity.  Phase B
  writes the accepted prefix exactly once.
* **KV (16 attention layers + the MTP layer)** — written speculatively for all
  ``n`` window positions, and rolled back by a *pointer move*: the committed
  length is ``L + m``, so the next step's window starts at ``L + m`` and simply
  overwrites positions ``L+m .. L+k``.  The scheduler owns the host-side length
  (``Request.num_computed_tokens``) and re-syncs ``PagedKVPool.seq_len`` from it
  at the top of every step, so a stale device length is never read.

Graph safety.  ``m`` never leaves the device inside the step: acceptance,
the commit mask, the emitted-token mask and the ``h_prev`` gather are all
tensor ops on it.  Every host-side piece (FlashInfer planning for the ``k``
draft passes and the verify pass, KV page allocation, the ``seq_len`` re-sync)
depends only on ``L``, which the host already knows from the *previous* step's
harvest — which is precisely why drafting happens at the head of the step that
verifies it, rather than at the tail of the step that produced ``h_prev``.

Sampling.  Greedy only (``SpecConfig.greedy_only``, default ``True``): with
``temperature <= 0`` on every live request, "accept while draft == argmax" makes
the emitted token stream **bit-identical** to non-speculative greedy decoding,
which is the correctness criterion for this module.  A batch containing any
``temperature > 0`` request falls back to the plain decode step for that step
(``SpecDecoder.eligible``).  A rejection sampler for non-greedy requests is not
implemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch


def spec_sample_accept(
    logits: torch.Tensor,
    drafts: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: torch.Tensor,
    *,
    candidates: int = 2048,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Speculative *sampling* acceptance for a point-mass (argmax) draft.

    ``logits`` ``[B, n, V]`` are the target's logits at the ``n = k + 1``
    window positions, ``drafts`` ``[B, k]`` the drafted tokens (window
    positions ``1..k``). The target distribution ``p_j`` at each position is
    exactly the one :func:`graphs.sample_tokens` samples from (temperature,
    then top-k and top-p inside the top ``candidates`` logits).

    The draft distribution is a point mass on the draft token ``d``, so the
    standard rejection rule ``accept with min(1, p(d) / q(d))`` becomes
    "accept with probability ``p(d)``", and the residual after a rejection,
    ``norm(max(0, p - q))``, is ``p`` with ``d`` removed. Accepting drafts left
    to right and sampling the first rejected position from its residual (or
    the bonus position from ``p_k`` when all ``k`` are accepted) emits tokens
    distributed exactly as plain sampling would, one at a time.

    Returns ``(tokens [B, n] int32, accepted [B] fp32 in 0..k)``: row ``i``
    emits ``tokens[i, :accepted[i] + 1]``. Fixed shape, no host sync, no
    ``multinomial``: safe inside a captured CUDA graph.
    """
    b, n, v = logits.shape
    k = n - 1
    c = min(int(candidates), v)
    flat = logits.reshape(b * n, v).float()
    topv, topi = torch.topk(flat, c, dim=-1, sorted=True)  # [B*n, c]

    def rows(x: torch.Tensor) -> torch.Tensor:  # per request -> per window position
        return x.to(flat.device, torch.float32).repeat_interleave(n).unsqueeze(-1)

    temp = rows(temperature).clamp(min=1e-5)
    scaled = topv / temp
    ar = torch.arange(c, device=flat.device).unsqueeze(0)
    tk = rows(top_k)
    tk_eff = torch.where(tk <= 0, torch.full_like(tk, float(c)), tk.clamp(max=float(c)))
    keep_k = ar < tk_eff
    probs = torch.softmax(scaled.masked_fill(~keep_k, float("-inf")), dim=-1)
    cum_excl = probs.cumsum(dim=-1) - probs
    tp = rows(top_p).clamp(min=1e-5, max=1.0)
    keep_p = (cum_excl < tp).clone()
    keep_p[:, 0] = True
    keep = keep_k & keep_p
    filtered = scaled.masked_fill(~keep, float("-inf"))
    p = torch.softmax(filtered, dim=-1)  # [B*n, c]; zero outside the kept set

    filtered = filtered.view(b, n, c)
    p = p.view(b, n, c)
    topi = topi.view(b, n, c)

    # p_j(d_{j+1}) for the k drafted positions (0 when d fell outside the kept set)
    is_d = topi[:, :k, :] == drafts.to(topi.dtype).unsqueeze(-1)  # [B, k, c]
    p_d = (p[:, :k, :] * is_d).sum(-1)  # [B, k]
    gen = generator if generator is not None and generator.device == flat.device else None
    u = torch.rand((b, max(k, 1)), device=flat.device, dtype=torch.float32, generator=gen)[:, :k]
    ok = (u < p_d).to(torch.float32)
    accepted = torch.cumprod(ok, dim=1).sum(dim=1) if k > 0 else torch.zeros(b, device=flat.device)

    # one gumbel-max draw per position from its residual: the draft removed at
    # positions 0..k-1 (only read when that draft was rejected), plain p at k
    resid = filtered.clone()
    if k > 0:
        resid[:, :k, :] = resid[:, :k, :].masked_fill(is_d, float("-inf"))
    g = torch.rand(resid.shape, device=flat.device, dtype=torch.float32, generator=gen).clamp(min=1e-20, max=1.0 - 1e-7)
    draw_local = (resid - torch.log(-torch.log(g))).argmax(dim=-1, keepdim=True)  # [B, n, 1]
    draw = topi.gather(-1, draw_local).squeeze(-1).to(torch.int32)  # [B, n]

    pos = torch.arange(n, device=flat.device).unsqueeze(0)  # [1, n]
    a = accepted.to(torch.long).unsqueeze(1)  # [B, 1]
    head = torch.cat([drafts.to(torch.int32), draw[:, -1:]], dim=1)  # drafts at 0..k-1
    tokens = torch.where(pos < a, head, draw)
    return tokens.to(torch.int32), accepted
import torch.nn.functional as F

from ..attn import flashinfer_attn as fi
from ..attn.kv_pool import PagedKVPool
from ..gemm import dispatch as gemm_dispatch
from .fused_model import (
    DeviceBuffers,
    FusedQwenForCausalLM,
    PrefillBatch,
    RuntimeConfig,
    StepContext,
)

__all__ = [
    "SpecConfig",
    "SpecAttentionRunner",
    "SpecDecoder",
    "build_spec_decoder",
]


# =========================================================================== #
# 1. config
# =========================================================================== #
@dataclass
class SpecConfig:
    """Knobs for one speculative-decoding configuration.

    ``k`` is the draft length (typically 1..3); the verify window is
    ``k + 1`` positions wide, and a step emits between 1 and ``k + 1`` tokens
    per sequence.

    **``k = 0`` is legal, and it is not speculation** -- it is a path control
    for diagnosing divergence.  The window is one position
    wide, there are no draft passes, ``m`` is always 1, and the step emits
    exactly one token per sequence, exactly like the plain decode step.  What
    it does *not* share with the plain decode step is the kernels: it still
    goes through ``FusedGDN.window``/``commit_window`` (the window kernel
    + the torch conv), through ``SpecAttentionRunner.window``
    (``BatchPrefillWithPagedKVCacheWrapper``), and through the packed-window
    GEMM shapes.  So a ``k = 0`` run that diverges from plain greedy decoding
    proves the divergence is a property of the *verify path's kernels*, with
    speculation removed as a variable.  Never use it for throughput; it is
    strictly slower than plain decode.

    **``k = 2`` is the default on measurement, not on a guess** (graph-timed,
    fp16 state).  The window kernel's breakeven accept length
    rises with ``k`` -- B=1: **1.41** (k=1), **1.55** (k=2), 1.87 (k=3); B=32:
    1.67, 2.06 -- so at the 2.4-3.4 mean accept length reported for
    SGLang MTP, k=2 is 1.61-1.94x against k=3's 1.34-1.61x.  k=2 dominates k=3
    at every accept length k=2 can reach.  Those ratios are *GDN-kernel* ratios
    though (window µs / plain-decode µs per layer), i.e. the SSM-traffic
    constraint, which binds at high batch; at B=1 the GDN is ~0.17 ms of a
    12.64 ms step and the binding cost is the draft head's ``lm_head``.
    ``bench_spec.py`` sweeps k=1,2,3 to confirm the ordering end to end rather
    than inheriting it.
    """

    k: int = 2
    greedy_only: bool = True
    #: run the MTP head over every prefill chunk so its KV layer has real
    #: context before the first draft.  Off => the draft head attends over an
    #: empty/stale MTP cache and acceptance collapses (quality only, never
    #: correctness -- the *target* decides acceptance).
    prefill_mtp: bool = True
    #: buckets to capture a speculative graph for.  ``None`` -> RuntimeConfig's.
    buckets: Optional[Tuple[int, ...]] = None
    #: gdn backend override for the window/commit kernels ("auto" = the model's)
    gdn_backend: Optional[str] = None

    #: keep the verify pass's logits in a persistent device buffer so a
    #: diagnostic can compare them against the plain decode step's
    #: ``DeviceBuffers.logits`` for the *same* context.  Off by
    #: default: the buffer is ``[bucket_max * n, vocab]`` fp32, i.e. 508 MiB at
    #: bucket 128 / k=3, which no production path has any use for.
    keep_window_logits: bool = False

    def __post_init__(self) -> None:
        if not (0 <= self.k <= 8):
            raise ValueError(f"spec k must be in 0..8, got {self.k}")

    @property
    def n(self) -> int:
        return self.k + 1


# =========================================================================== #
# 2. attention for a speculative step
# =========================================================================== #
def _torch_paged_multi(
    pool: PagedKVPool,
    layer: int,
    slots: Sequence[int],
    q_lens: Sequence[int],
    kv_lens: Sequence[int],
    q: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    scaling: float,
) -> torch.Tensor:
    """Bottom-right-causal SDPA over the pages -- the CPU/reference path.

    Identical in shape and masking to ``AttentionRunner._torch_paged_prefill``
    (a chunk whose ``q_len`` queries are the *last* ``q_len`` of a ``kv_len``
    prefix), reproduced here rather than reached into so that
    ``AttentionRunner`` needs no speculative-decoding changes.
    """
    groups = num_qo_heads // num_kv_heads
    outs: List[torch.Tensor] = []
    off = 0
    for slot, ql, kl in zip(slots, q_lens, kv_lens):
        if ql == 0:
            continue
        k, v = pool.gather_dense(layer, slot, kl)
        k = k.transpose(0, 1)  # [Hkv, kv_len, D]
        v = v.transpose(0, 1)
        if groups > 1:
            k = k.repeat_interleave(groups, dim=0)
            v = v.repeat_interleave(groups, dim=0)
        qi = q[off : off + ql].transpose(0, 1)  # [Hq, q_len, D]
        q_pos = torch.arange(kl - ql, kl, device=q.device)[:, None]
        k_pos = torch.arange(kl, device=q.device)[None, :]
        mask = (k_pos <= q_pos)[None]
        oi = F.scaled_dot_product_attention(
            qi.float(), k.float(), v.float(), attn_mask=mask, scale=scaling
        )
        outs.append(oi.transpose(0, 1).to(q.dtype))
        off += ql
    return torch.cat(outs, dim=0)


class SpecAttentionRunner:
    """The plan()-outside / run()-inside attention split for a spec step.

    Duck-types the two methods ``fused_model``'s layer modules call --
    ``decode(kv_layer, q, slot_ids)`` (the MTP draft passes, one query token per
    sequence) and ``window(kv_layer, q)`` (the verify pass, ``n`` query tokens
    per sequence) -- so it can be dropped into a :class:`StepContext` in place of
    :class:`~.fused_model.AttentionRunner` with no change to that class.

    One persistent FlashInfer wrapper per *phase* per bucket, because every
    phase of one step is planned before any of them runs: the draft passes and
    the verify pass have different KV lengths, and a single wrapper only holds
    one plan.  ``select()`` chooses which wrapper the next ``decode``/``window``
    call uses; it is called between the phases of the step body, so under CUDA
    graph capture the choice is baked into the graph (each phase's launch always
    reads the same wrapper's persistent buffers, which the host re-plans in
    place before every replay).
    """

    def __init__(
        self,
        pool: PagedKVPool,
        *,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        window: int,
        backend: str,
        device: torch.device,
        workspace: Optional[torch.Tensor],
        max_pages: int,
        use_cuda_graph_wrappers: bool,
        kv_scales: Optional[Dict[int, Tuple[float, float]]] = None,
    ):
        self.pool = pool
        #: the *model's* per-layer fp8-KV scale dict, by reference -- see
        #: :meth:`_scales_for`.  ``{}`` for a bf16 pool.
        self._kv_scales: Dict[int, Tuple[float, float]] = (
            kv_scales if kv_scales is not None else {}
        )
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scaling = head_dim ** -0.5
        self.window_size = window
        self.backend = backend
        self.device = device
        self.workspace = workspace
        self.max_pages = max_pages
        self._graphed = use_cuda_graph_wrappers

        self._draft_wrappers: Dict[Tuple[int, int], "fi.FlashInferDecodeAttention"] = {}
        self._window_wrappers: Dict[int, "fi.FlashInferPrefillAttention"] = {}
        self._dyn_draft: Optional["fi.FlashInferDecodeAttention"] = None
        self._dyn_window: Optional["fi.FlashInferPrefillAttention"] = None
        self._current = None
        self._phase = "window"
        # the torch path needs to know which draft pass is running; the
        # flashinfer path gets that through `select`'s wrapper choice instead.
        self._draft_index = 0

        # torch-fallback plan state (host lists, one per phase)
        self._slots: List[int] = []
        self._draft_kv_lens: List[List[int]] = []
        self._window_kv_lens: List[int] = []

    # -- wrapper construction ------------------------------------------------ #
    def _draft_wrapper(self, bucket: int, j: int):
        if not self._graphed:
            if self._dyn_draft is None:
                self._dyn_draft = fi.FlashInferDecodeAttention(
                    self.workspace,
                    num_qo_heads=self.num_qo_heads,
                    num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim,
                    page_size=self.pool.cfg.page_size,
                    max_batch_size=self.pool.cfg.max_seqs,
                    max_pages=self.max_pages,
                    kv_dtype=self.pool.storage_dtype,
                    q_dtype=None,
                    use_cuda_graph=False,
                    device=str(self.device),
                )
            return self._dyn_draft
        w = self._draft_wrappers.get((bucket, j))
        if w is None:
            w = fi.FlashInferDecodeAttention(
                self.workspace,
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                page_size=self.pool.cfg.page_size,
                max_batch_size=bucket,
                max_pages=self.max_pages,
                kv_dtype=self.pool.storage_dtype,
                q_dtype=None,
                use_cuda_graph=True,
                device=str(self.device),
            )
            self._draft_wrappers[(bucket, j)] = w
        return w

    def _window_wrapper(self, bucket: int):
        if not self._graphed:
            if self._dyn_window is None:
                self._dyn_window = fi.FlashInferPrefillAttention(
                    self.workspace,
                    num_qo_heads=self.num_qo_heads,
                    num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim,
                    page_size=self.pool.cfg.page_size,
                    kv_dtype=self.pool.storage_dtype,
                    use_cuda_graph=False,
                    device=str(self.device),
                )
            return self._dyn_window
        w = self._window_wrappers.get(bucket)
        if w is None:
            w = fi.FlashInferPrefillAttention(
                self.workspace,
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                page_size=self.pool.cfg.page_size,
                kv_dtype=self.pool.storage_dtype,
                use_cuda_graph=True,
                max_batch_size=bucket,
                max_pages=self.max_pages,
                device=str(self.device),
            )
            self._window_wrappers[bucket] = w
        return w

    # -- planning (host side, outside any capture/replay) --------------------- #
    def plan_step(self, bucket: int, slots: Sequence[int], ctx_lens: Sequence[int]) -> None:
        """Plan every phase of one speculative step.

        ``ctx_lens[i]`` is sequence ``i``'s *committed* context length ``L``
        (its next token sits at position ``L``).  Draft pass ``j`` reads
        ``L + j + 1`` KV tokens (it has just written position ``L + j``); the
        verify pass reads ``L + n``.  ``slots``/``ctx_lens`` are already padded
        to ``bucket`` -- padded rows point at the scratch slot with ``L = 0``
        (the scratch slot owns one page).
        """
        if len(slots) != bucket or len(ctx_lens) != bucket:
            raise ValueError(
                f"plan_step: slots/ctx_lens must be padded to bucket={bucket}, "
                f"got {len(slots)}/{len(ctx_lens)}"
            )
        n = self.window_size
        k = n - 1
        self._slots = list(slots)
        self._draft_kv_lens = [[int(L) + j + 1 for L in ctx_lens] for j in range(k)]
        self._window_kv_lens = [int(L) + n for L in ctx_lens]
        if self.backend != "flashinfer":
            return
        for j in range(k):
            indptr, indices, last, _ = self.pool.build_flashinfer_indices(
                self._slots, seq_lens=self._draft_kv_lens[j]
            )
            self._draft_wrapper(bucket, j).plan(indptr, indices, last)
        indptr, indices, last, _ = self.pool.build_flashinfer_indices(
            self._slots, seq_lens=self._window_kv_lens
        )
        qo = torch.arange(0, (bucket + 1) * n, n, dtype=torch.int32, device=self.device)
        self._window_wrapper(bucket).plan(qo, indptr, indices, last, causal=True)

    def plan_draft0(self, bucket: int, slots: Sequence[int], ctx_lens: Sequence[int]) -> None:
        """Plan **only** the ``j=0`` draft wrapper.

        ``SpecDecoder.on_plain_step``'s cheap MTP catch-up call needs exactly
        one draft-shaped attention plan (one query token per sequence) and
        never runs the verify pass -- calling :meth:`plan_step` here would
        also replan the window wrapper for nothing, on every plain-decode
        step, at whatever batch size the plain path is currently serving
        (which is precisely the *large*-batch regime ``--spec-max-batch``
        exists to keep spec-shaped work out of). Mirrors ``plan_step``'s
        ``j=0`` branch exactly, minus the window half.
        """
        if len(slots) != bucket or len(ctx_lens) != bucket:
            raise ValueError(
                f"plan_draft0: slots/ctx_lens must be padded to bucket={bucket}, "
                f"got {len(slots)}/{len(ctx_lens)}"
            )
        self._slots = list(slots)
        kv_lens = [int(L) + 1 for L in ctx_lens]
        self._draft_kv_lens = [kv_lens]
        if self.backend != "flashinfer":
            return
        indptr, indices, last, _ = self.pool.build_flashinfer_indices(
            self._slots, seq_lens=kv_lens
        )
        self._draft_wrapper(bucket, 0).plan(indptr, indices, last)

    def select(self, phase: str, bucket: int, j: int = 0) -> None:
        self._phase = phase
        if self.backend != "flashinfer":
            return
        self._current = (
            self._draft_wrapper(bucket, j) if phase == "draft" else self._window_wrapper(bucket)
        )

    # -- run (inside the captured region) ------------------------------------- #
    def _scales_for(self, kv_layer: int) -> Tuple[Optional[float], Optional[float]]:
        """Same contract as ``AttentionRunner._scales_for``.

        The speculative draft and verify passes must apply the same
        calibrated per-layer fp8-KV dequantization scale as the plain decode
        path; returning ``(None, None)`` here would silently skip it (about a
        17 percent attention mismatch).  That hazard is latent under
        ``--kv-cache-dtype bf16`` (both return ``(None, None)``) and fatal
        under ``fp8``.  The scales are owned by the
        model's own ``AttentionRunner`` (``set_kv_scale``, called once by the
        calibration pass before capture), so read them from there rather than
        keeping a second copy that can drift.
        """
        if self.pool.cfg.dtype != "fp8":
            return (None, None)
        return self._kv_scales.get(kv_layer, (None, None))

    def decode(self, kv_layer: int, q: torch.Tensor, slot_ids: Optional[torch.Tensor] = None):
        """One query token per sequence -- the MTP draft passes."""
        if self.backend == "flashinfer":
            k_scale, v_scale = self._scales_for(kv_layer)
            return self._current.run(q, self.pool.kv[kv_layer], k_scale=k_scale, v_scale=v_scale)
        j = self._draft_index
        kv_lens = self._draft_kv_lens[j]
        return _torch_paged_multi(
            self.pool, kv_layer, self._slots, [1] * len(self._slots), kv_lens,
            q, self.num_qo_heads, self.num_kv_heads, self.scaling,
        )

    def window(self, kv_layer: int, q: torch.Tensor):
        """``n`` query tokens per sequence -- the verify pass."""
        if self.backend == "flashinfer":
            k_scale, v_scale = self._scales_for(kv_layer)
            return self._current.run(q, self.pool.kv[kv_layer], k_scale=k_scale, v_scale=v_scale)
        n = self.window_size
        return _torch_paged_multi(
            self.pool, kv_layer, self._slots, [n] * len(self._slots), self._window_kv_lens,
            q, self.num_qo_heads, self.num_kv_heads, self.scaling,
        )


# =========================================================================== #
# 3. the speculative decoder
# =========================================================================== #
class SpecDecoder:
    """Captures/replays one *speculative* decode step per bucket.

    Mirrors :class:`~.graphs.GraphedDecoder`'s contract (``bucket_for`` /
    ``warmup`` / ``capture`` / ``step``) so the scheduler drives the two the
    same way; the difference is that ``step`` returns up to ``k+1`` tokens per
    sequence plus the accepted length ``m``.
    """

    def __init__(
        self,
        model: FusedQwenForCausalLM,
        buf: DeviceBuffers,
        rt: RuntimeConfig,
        cfg: SpecConfig,
    ):
        if model.mtp is None:
            raise RuntimeError(
                "spec decoding needs the MTP head: build the model with "
                "RuntimeConfig(enable_mtp=True) and a checkpoint that ships mtp.*"
            )
        self.model = model
        self.buf = buf
        self.rt = rt
        self.sampler_candidates = int(getattr(rt, "sampler_candidates", 2048) or 2048)
        #: dedicated rng for speculative sampling, registered with every
        #: captured graph so each replay draws fresh numbers
        self._generator: Optional[torch.Generator] = None
        if not cfg.greedy_only and model.device.type == "cuda":
            self._generator = torch.Generator(device=model.device)
            self._generator.manual_seed(1)
        self.cfg = cfg
        self.k = cfg.k
        self.n = cfg.n
        self.device = model.device
        self.dtype = model.dtype
        self.hidden_size = model.config.hidden_size

        bmax = buf.max_batch
        i32 = dict(dtype=torch.int32, device=self.device)
        # [Bmax, n] window: column 0 is the last committed token, 1..k the drafts
        self.window_tokens = torch.zeros(bmax, self.n, **i32)
        self.window_pos = torch.zeros(bmax, self.n, **i32)
        self.flat_slots = torch.zeros(bmax * self.n, **i32)
        self.m = torch.ones(bmax, **i32)
        self.out_window = torch.full((bmax, self.n), -1, **i32)
        # `m = 0` for phase A's conv call: `concat(state, x)[..., 0:W-1] == state`
        self.zero_m = torch.zeros(bmax, **i32)
        # per-*slot* carry of the main model's post-final-norm hidden at the
        # position before this step's first window token (the MTP head's `h_t`).
        # Slot-indexed, not row-indexed: continuous batching reshuffles which
        # bucket row holds which sequence between steps.
        self.h_prev = torch.zeros(
            model.n_slots + 1, self.hidden_size, dtype=self.dtype, device=self.device
        )
        # Diagnostic only: a persistent copy of the verify
        # pass's logits, so `bench_spec.probe_logit_parity` can hold the plain
        # decode step's `DeviceBuffers.logits` next to the window forward's for
        # the *same* context and the same state.  Allocated only when asked
        # for -- at bucket 128 / k=3 it is [512, 248320] fp32 = 508 MiB.
        self.window_logits: Optional[torch.Tensor] = None
        if cfg.keep_window_logits:
            self.window_logits = torch.zeros(
                bmax * self.n, model.config.vocab_size,
                dtype=torch.float32, device=self.device,
            )

        pin = self.device.type == "cuda"
        self.host: Dict[str, torch.Tensor] = {
            "window_tokens": torch.zeros(bmax, self.n, dtype=torch.int32, pin_memory=pin),
            "window_pos": torch.zeros(bmax, self.n, dtype=torch.int32, pin_memory=pin),
            "flat_slots": torch.zeros(bmax * self.n, dtype=torch.int32, pin_memory=pin),
        }

        self.attn = SpecAttentionRunner(
            model.kv_pool,
            num_qo_heads=model.config.num_attention_heads,
            num_kv_heads=model.config.num_key_value_heads,
            head_dim=model.config.head_dim,
            window=self.n,
            backend=model.attn.backend,
            device=self.device,
            workspace=getattr(model.attn, "workspace", None),
            max_pages=rt.n_kv_pages,
            use_cuda_graph_wrappers=rt.use_cuda_graphs,
            kv_scales=model.attn.kv_scales,
        )

        # Greedy-equivalence caveat, stated once where it is decided.  The
        # window kernel keeps the running SSM state in fp32 registers for all
        # `n` positions and stores only `S_m`; the plain decode kernel stores
        # (and so *rounds*) the state to the pool dtype after every token.
        # With `ssm_state_dtype="fp32"` that round trip is exact and the two
        # paths are bit-equivalent; with `"fp16"` they are not, by construction,
        # and the greedy streams may separate after a few hundred tokens. fp16
        # state is a legitimate throughput choice (5-17% faster from B=32
        # up) -- it is just not a configuration the bit-identical gate can pass.
        if rt.ssm_state_dtype != "fp32":
            import warnings

            warnings.warn(
                f"SpecDecoder: ssm_state_dtype={rt.ssm_state_dtype!r}. The verify "
                "window keeps intermediate SSM states in registers while plain "
                "decode rounds them to the pool dtype every token, so spec greedy "
                "output is NOT bit-identical to non-spec greedy at this setting. "
                "Use ssm_state_dtype='fp32' for the correctness gate.",
                RuntimeWarning,
                stacklevel=2,
            )

        self.buckets = tuple(cfg.buckets or rt.buckets_for()) if rt.use_cuda_graphs else ()
        self._graphs: Dict[int, "torch.cuda.CUDAGraph"] = {}
        self._pool = None
        # telemetry (`spec_accept_length` / `spec_acceptance_rate`)
        self.steps = 0
        self.accepted_tokens = 0  # sum of (m - 1)
        self.emitted_tokens = 0  # sum of m

    # -- bookkeeping ---------------------------------------------------------- #
    @property
    def graphs_enabled(self) -> bool:
        return bool(self.buckets) and self.device.type == "cuda" and torch.cuda.is_available()

    def bucket_for(self, batch: int) -> int:
        buckets = self.buckets or self.rt.buckets_for()
        for b in buckets:
            if batch <= b:
                return b
        return max(buckets[-1], batch) if buckets else batch

    def eligible(self, temperatures: Sequence[float]) -> bool:
        """Is this batch servable by the speculative path?"""
        if not self.cfg.greedy_only:
            return True
        return all(t <= 0.0 for t in temperatures)

    def stats(self) -> Dict[str, float]:
        steps = max(self.steps, 1)
        return {
            "spec_steps": self.steps,
            "spec_accept_length": self.emitted_tokens / steps,
            "spec_acceptance_rate": self.accepted_tokens / (steps * self.k) if self.k else 0.0,
        }

    def reset_slot(self, slot: int) -> None:
        self.h_prev[slot].zero_()

    # -- context -------------------------------------------------------------- #
    def _ctx(self, slot_ids, positions, seq_slot_ids) -> StepContext:
        cos, sin = self.model.rotary.lookup(positions)
        return StepContext(
            pool=self.model.kv_pool,
            attn=self.attn,
            state_pool=self.model.state_pool,
            conv_pool=self.model.conv_pool,
            slot_ids=slot_ids,
            positions=positions,
            cos=cos,
            sin=sin,
            seq_slot_ids=seq_slot_ids,
        )

    # -- the step body (captured or run eagerly) ------------------------------ #
    def _draft(self, bucket: int) -> None:
        model = self.model
        slots = self.buf.slot_ids[:bucket]
        prev = self.h_prev.index_select(0, slots.long())
        for j in range(self.k):
            self.attn.select("draft", bucket, j)
            self.attn._draft_index = j
            pos = self.window_pos[:bucket, j].contiguous()
            ctx = self._ctx(slots, pos, slots)
            emb = F.embedding(self.window_tokens[:bucket, j].long(), model.embed_tokens)
            prev = model.mtp(emb.to(self.dtype), prev, ctx, prefill=False)
            logits = model.lm_head(prev).float()
            self.window_tokens[:bucket, j + 1] = logits.argmax(-1).to(torch.int32)

    def _verify(self, bucket: int):
        """Phase A: one main-model forward over the whole ``[B, n]`` window.

        The whole body runs inside ``gemm_dispatch.rows_per_sequence(n)``.  That
        is a **correctness** requirement, not a tuning one: this pass packs
        ``n = k+1`` rows per sequence, so without the scope every
        ``ResolvedLinear`` sees ``M = bucket*n`` and routes on that.  At
        ``bucket=32, k=3`` that is ``M=128``, which crosses ``gemm.dispatch``'s
        measured ``M=64`` threshold and swaps 256 of the model's 305 linears
        from ``vllm_marlin_fp8_w8a16`` (true W8A16, relL2 2.7e-3 against an
        exact dequant) to ``flashinfer_fp8_blockscale`` (which quantizes the
        *activations* to fp8: relL2 2.6e-2).  A 2.6e-2
        logit perturbation flips a greedy argmax about once every 38 tokens
        over this vocabulary, so without the scope the verifier would accept
        and emit a different model's argmax than the plain decode path it must
        reproduce, and greedy streams diverge within a few dozen tokens.

        The scope is pure host-side Python and constant for the whole captured
        region, so a CUDA-graph capture bakes in the same kernel every replay
        uses.
        """
        model = self.model
        n = self.n
        self.attn.select("window", bucket)
        seq_slots = self.buf.slot_ids[:bucket]
        flat_slots = self.flat_slots[: bucket * n]
        flat_pos = self.window_pos[:bucket].reshape(-1)
        flat_ids = self.window_tokens[:bucket].reshape(-1)
        ctx = self._ctx(flat_slots, flat_pos, seq_slots)
        zero_m = self.zero_m[:bucket]
        with gemm_dispatch.rows_per_sequence(n):
            h = F.embedding(flat_ids.long(), model.embed_tokens).to(self.dtype)
            caches: List[Tuple[int, object]] = []
            for i, layer in enumerate(model.layers):
                h, cache = layer.window(h, ctx, zero_m)
                if cache is not None:
                    caches.append((i, cache))
            h = model._final_norm(h)
            logits = model.lm_head(h).float()  # [bucket*n, V]
        if self.window_logits is not None:
            # a fixed-shape copy: safe inside the captured region.
            self.window_logits[: bucket * n].copy_(logits)
        return h, logits, caches, ctx

    def _accept(self, bucket: int, logits: torch.Tensor) -> None:
        """Greedy acceptance, entirely on device.

        ``tgt[:, j] = argmax(logits at window position j)`` is what
        non-speculative greedy decoding would emit after the window's first
        ``j+1`` tokens.  Draft ``j+1`` is accepted iff it equals ``tgt[:, j]``
        *and* every earlier draft was accepted -- a cumulative product, not a
        plain sum, so a match after a mismatch does not count.  ``m = accepted
        + 1`` counts the bonus token, i.e. the number of window positions whose
        state must be committed.
        """
        n = self.n
        tgt = logits.view(bucket, n, -1).argmax(dim=-1).to(torch.int32)  # [B, n]
        drafts = self.window_tokens[:bucket, 1:]  # [B, k]
        match = (drafts == tgt[:, : self.k]).to(torch.float32)
        accepted = torch.cumprod(match, dim=1).sum(dim=1)  # [B] in 0..k (fp32)
        if not self.cfg.greedy_only:
            # sampled rows use speculative sampling; greedy rows keep the exact
            # argmax rule above (bit identical to plain greedy decoding)
            temp = self.buf.temperature[:bucket]
            s_tok, s_acc = spec_sample_accept(
                logits.view(bucket, n, -1), drafts, temp,
                self.buf.top_p[:bucket], self.buf.top_k[:bucket],
                candidates=self.sampler_candidates, generator=self._generator,
            )
            greedy_row = temp <= 0
            accepted = torch.where(greedy_row, accepted, s_acc)
            tgt = torch.where(greedy_row.unsqueeze(1), tgt, s_tok)
        m = (accepted + 1).to(torch.int32)
        self.m[:bucket].copy_(m)
        ar = torch.arange(n, device=tgt.device, dtype=torch.int32)
        emit = ar.unsqueeze(0) < m.unsqueeze(1)
        self.out_window[:bucket].copy_(torch.where(emit, tgt, torch.full_like(tgt, -1)))

    def _commit(self, bucket: int, ctx: StepContext, caches) -> None:
        m = self.m[:bucket]
        layers = self.model.layers
        for i, cache in caches:
            layers[i].commit_window(ctx, cache, m)

    def _carry_hidden(self, bucket: int, h: torch.Tensor) -> None:
        """``h_prev[slot] <- hidden[m-1]``: the hidden that conditions the next
        step's first draft.  A device gather on ``m`` -- no host sync."""
        hh = h.view(bucket, self.n, -1)
        idx = (self.m[:bucket].long() - 1).view(bucket, 1, 1).expand(bucket, 1, hh.shape[-1])
        sel = hh.gather(1, idx).squeeze(1)
        self.h_prev.index_copy_(0, self.buf.slot_ids[:bucket].long(), sel.to(self.h_prev.dtype))

    def _run_step(self, bucket: int) -> None:
        self._draft(bucket)
        h, logits, caches, ctx = self._verify(bucket)
        self._accept(bucket, logits)
        self._commit(bucket, ctx, caches)
        self._carry_hidden(bucket, h)

    # -- capture -------------------------------------------------------------- #
    def _fill_scratch_inputs(self, bucket: int, ctx_len: int = 0) -> None:
        """Point every row at the scratch slot with a committed length of
        ``ctx_len``.

        ``ctx_len`` matters for capture: FlashInfer's ``use_cuda_graph``
        wrappers bake part of their launch configuration at ``plan()`` time, so
        capturing against a 1-page plan and replaying against a 129-page one
        leans harder on that contract than it needs to.  Passing the benchmark's
        real context length here makes the captured plan the same *shape* as
        every replay's.  Defaults to 0 (the ``graphs.GraphedDecoder`` behaviour).
        """
        scratch = self.model.scratch_slot
        n = self.n
        self.model.kv_pool.ensure_capacity(scratch, ctx_len + n)
        self.host["window_tokens"][:bucket].zero_()
        self.host["window_pos"][:bucket] = (
            torch.arange(n, dtype=torch.int32).expand(bucket, n) + ctx_len
        )
        self.host["flat_slots"][: bucket * n].fill_(scratch)
        self.buf.host["slot_ids"][:bucket].fill_(scratch)
        self.upload(bucket)
        self.buf.upload(["slot_ids"])
        self._sync_seq_lens([scratch] * bucket, [ctx_len] * bucket)
        self.plan(bucket, [scratch] * bucket, [ctx_len] * bucket)

    def upload(self, bucket: int) -> None:
        n = self.n
        self.window_tokens[:bucket].copy_(self.host["window_tokens"][:bucket], non_blocking=True)
        self.window_pos[:bucket].copy_(self.host["window_pos"][:bucket], non_blocking=True)
        self.flat_slots[: bucket * n].copy_(
            self.host["flat_slots"][: bucket * n], non_blocking=True
        )

    def plan(self, bucket: int, slots: Sequence[int], ctx_lens: Sequence[int]) -> None:
        self.attn.plan_step(bucket, slots, ctx_lens)

    def warmup(self, iters: int = 2, ctx_len: int = 0) -> None:
        buckets = self.buckets if self.graphs_enabled else (self.bucket_for(1),)
        for b in buckets:
            self._fill_scratch_inputs(b, ctx_len)
            for _ in range(iters):
                self._run_step(b)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        # the warmup ran real state updates on the scratch slot
        self.model.reset_slot(self.model.scratch_slot)
        self.model._init_scratch_slot()
        self.reset_slot(self.model.scratch_slot)

    def capture(self, pool_handle=None, ctx_len: int = 0) -> None:
        """One graph per bucket, fixed ``k+1`` window.

        Shares the graph mempool with :class:`~.graphs.GraphedDecoder` when the
        caller passes its handle, so the two step shapes do not each pay a
        private ~1 GiB pool.
        """
        if not self.graphs_enabled:
            return
        if self.model.attn.backend != "flashinfer":
            raise RuntimeError(
                f"SpecDecoder.capture(): attn_backend={self.model.attn.backend!r} is not "
                "graph-safe -- the torch attention fallback does a host sync. Use "
                "RuntimeConfig(attn_backend='flashinfer') or use_cuda_graphs=False."
            )
        self._pool = pool_handle or torch.cuda.graph_pool_handle()
        for b in self.buckets:
            self._fill_scratch_inputs(b, ctx_len)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self._run_step(b)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()

            g = torch.cuda.CUDAGraph()
            if self._generator is not None:
                g.register_generator_state(self._generator)
            with torch.cuda.graph(g, pool=self._pool):
                self._run_step(b)
            self._graphs[b] = g
        self.model.reset_slot(self.model.scratch_slot)
        self.model._init_scratch_slot()
        self.reset_slot(self.model.scratch_slot)

    # -- the per-step entry point --------------------------------------------- #
    def step(
        self,
        batch: int,
        slots: Sequence[int],
        ctx_lens: Sequence[int],
        last_tokens: Sequence[int],
        sampling: Optional[Tuple[Sequence[float], Sequence[float], Sequence[float]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One speculative step.

        ``slots``/``ctx_lens``/``last_tokens`` are the *live* batch (length
        ``batch``); this pads them to the bucket itself (scratch slot, ``L=0``).
        Returns ``(out_window[:bucket] [B, n] int32, m[:bucket] [B] int32)``;
        row ``i``'s emitted tokens are ``out_window[i, :m[i]]`` and everything
        past that is ``-1``.
        """
        bucket = self.bucket_for(batch)
        scratch = self.model.scratch_slot
        pad = bucket - batch
        slots_p = list(slots) + [scratch] * pad
        lens_p = list(ctx_lens) + [0] * pad
        toks_p = list(last_tokens) + [0] * pad
        n = self.n

        wt = self.host["window_tokens"]
        wp = self.host["window_pos"]
        wt[:bucket].zero_()
        wt[:bucket, 0] = torch.tensor(toks_p, dtype=torch.int32)
        base = torch.tensor(lens_p, dtype=torch.int32).unsqueeze(1)
        wp[:bucket] = base + torch.arange(n, dtype=torch.int32).unsqueeze(0)
        self.host["flat_slots"][: bucket * n] = (
            torch.tensor(slots_p, dtype=torch.int32).unsqueeze(1).expand(bucket, n).reshape(-1)
        )
        self.buf.host["slot_ids"][:bucket] = torch.tensor(slots_p, dtype=torch.int32)
        self.upload(bucket)
        self.buf.upload(["slot_ids"])
        if not self.cfg.greedy_only:
            # (temperature, top_p, top_k) per live row; padding rows are greedy
            temps, tps, tks = sampling if sampling is not None else ([0.0] * batch, [1.0] * batch, [0.0] * batch)
            self.buf.host["temperature"][:bucket] = torch.tensor(list(temps) + [0.0] * pad, dtype=torch.float32)
            self.buf.host["top_p"][:bucket] = torch.tensor(list(tps) + [1.0] * pad, dtype=torch.float32)
            self.buf.host["top_k"][:bucket] = torch.tensor(list(tks) + [0.0] * pad, dtype=torch.float32)
            self.buf.upload(["temperature", "top_p", "top_k"])

        # Re-sync the pool's device-side committed lengths from the host's
        # (authoritative) view before planning: the previous step's speculative
        # KV writes left `seq_len` at L+k, and the rejected tail must not be
        # visible. This IS the KV rollback (a KV write-pointer move).
        self._sync_seq_lens(slots_p, lens_p)
        self.plan(bucket, slots_p, lens_p)

        if bucket in self._graphs:
            self._graphs[bucket].replay()
        else:
            self._run_step(bucket)
        return self.out_window[:bucket], self.m[:bucket]

    def _sync_seq_lens(self, slots: Sequence[int], lens: Sequence[int]) -> None:
        pool = self.model.kv_pool
        idx = torch.tensor(list(slots), dtype=torch.long, device=pool.seq_len.device)
        val = torch.tensor(list(lens), dtype=pool.seq_len.dtype, device=pool.seq_len.device)
        pool.seq_len.scatter_reduce_(0, idx, val, reduce="amax", include_self=False)

    def note_step(self, ms: Sequence[int]) -> None:
        """Fold one harvested step into the acceptance telemetry."""
        self.steps += len(ms)
        self.emitted_tokens += int(sum(ms))
        self.accepted_tokens += int(sum(m - 1 for m in ms))

    # -- plain-step hook ------------------------------------------------------ #
    def on_plain_step(
        self,
        slots: Sequence[int],
        tokens,  # Sequence[int] | torch.Tensor  (device-side under async scheduling)
        ctx_lens: Sequence[int],
    ) -> None:
        """Keep the MTP head's own KV layer and the ``h_prev`` carry advancing
        when a decode step takes the *plain* (non-speculative) path while
        spec decoding is configured overall -- ``--spec-max-batch`` routes a
        step to plain whenever the live batch exceeds the threshold, and a
        sampling request in the batch does the same (``eligible``).

        **The problem this closes.** The MTP block is its own dedicated
        attention layer with its own KV cache (``FusedQwenForCausalLM
        .mtp_kv_layer``), populated only by :meth:`_draft` and
        :meth:`on_prefill`. ``FusedQwenForCausalLM.decode_forward`` -- the
        plain step's whole body -- never touches it. Without this hook, every
        plain-decode token would leave a gap in that one layer's paged KV: not
        zeroed, whatever the page last held (slot reuse), for as long
        as the sequence lives, because nothing ever revisits a committed
        position to backfill it. That is not a *correctness* bug -- the
        verify pass's accept/reject decision reads only the target model's
        own logits over the real committed context (``_verify``/``_accept``),
        never the MTP layer, so greedy equivalence holds with or without
        this hook -- but every re-entry into spec mode
        after a plain stretch would draft off a stale ``h_prev`` and a KV
        cache with holes in it, both of which only depress accept length. At
        a threshold low enough to toggle every few steps under real traffic,
        an uncorrected gap is not an edge case, it is the common case.

        **The fix, and why it is cheap.** Run one MTP-block forward -- one
        small attention+MLP layer, not the 64-layer main stack -- using the
        *carried* ``h_prev`` as ``previous_hidden``, exactly the
        approximation :meth:`_draft` already makes for draft index ``j >= 1``
        within a real spec step (chain purely off the MTP's own recurrent
        hidden after the first draft, never re-touching the target model).
        So this is not a new source of approximation, only the existing one
        extended across the plain-step gap; a k>=2 spec run already tolerates
        exactly this for its later draft positions. The token fed in is the
        step's *input* token (``tokens[i]``, at position ``ctx_lens[i]`` --
        the same ``x0``/``L`` pair the plain step itself conditions on), so
        the KV entry this appends is precisely the one :meth:`_draft`'s
        ``j=0`` would have appended had the step gone through spec instead.

        **The alternative considered and rejected: a re-prefill catch-up on
        re-entry instead of a hook on every plain step.** Recomputing the
        gap's hidden states from scratch when spec resumes would need
        ``model.prefill_forward`` over every token generated while plain --
        i.e. re-running the *64-layer* main stack a second time for tokens
        the plain step already paid for once. That is strictly more
        expensive than this hook (paid once per plain step, one light layer)
        and only *some* multiple of it if a request spends many steps above
        the threshold before dropping back down, so it was not built.

        Bucket-padded the same way :meth:`step` is (scratch slot, ``L=0``) so
        the ``j=0`` draft-wrapper cache this reuses stays bounded to
        ``self.buckets`` regardless of how the plain path's own (unrelated,
        much larger) batch size varies.
        """
        if not slots:
            return
        bucket = self.bucket_for(len(slots))
        scratch = self.model.scratch_slot
        pad = bucket - len(slots)
        slots_p = list(slots) + [scratch] * pad
        lens_p = list(ctx_lens) + [0] * pad

        slot_t = torch.tensor(slots_p, dtype=torch.int32, device=self.device)
        pos_t = torch.tensor(lens_p, dtype=torch.int32, device=self.device)
        if isinstance(tokens, torch.Tensor):
            # Under asynchronous scheduling the step's input
            # tokens were gathered on the *device* from the previous step's
            # output and the host never saw them.  Only `F.embedding` below
            # reads them, so a device tensor is as good as the list -- and
            # bringing it to the host here would reintroduce exactly the D2H
            # the mode exists to remove.
            tok_t = tokens[: len(slots)].to(torch.int64)
            if pad:
                tok_t = torch.cat(
                    [tok_t, torch.zeros(pad, dtype=torch.int64, device=tok_t.device)]
                )
        else:
            tok_t = torch.tensor(list(tokens) + [0] * pad, dtype=torch.int64,
                                 device=self.device)

        self.attn.plan_draft0(bucket, slots_p, lens_p)
        self.attn.select("draft", bucket, 0)
        # torch-fallback `decode()` reads `self._draft_index` rather than the
        # `j` `select()` was given (see section 2), and a real spec step (k>=1)
        # can leave it non-zero; `plan_draft0` only ever fills index 0.
        self.attn._draft_index = 0

        prev = self.h_prev.index_select(0, slot_t.long())
        ctx = self._ctx(slot_t, pos_t, slot_t)
        emb = F.embedding(tok_t, self.model.embed_tokens).to(self.dtype)
        new_h = self.model.mtp(emb, prev, ctx, prefill=False)
        self.h_prev.index_copy_(0, slot_t.long(), new_h.to(self.h_prev.dtype))

    # -- prefill hook ---------------------------------------------------------- #
    def on_prefill(self, batch: PrefillBatch, hidden: torch.Tensor) -> None:
        """Populate the MTP KV layer and the ``h_prev`` carry for a prefill chunk.

        The MTP block is the 17th attention layer and keeps its own KV; if it is
        never run over the prompt, the first draft attends to an empty (or, worse,
        a recycled page's stale) cache and acceptance collapses.  So: run the head
        over exactly the same tokens and positions the main chunk just consumed,
        with ``previous_hidden[i] = h_{i-1}`` -- ``h_prev[slot]`` for each
        sequence's first token (which makes this correct across *chunked* prefill
        too, where ``h_{i-1}`` lives in the previous chunk), and this chunk's own
        hidden for the rest.  That (h_t, x_{t+1}) pairing is the input order
        that measured best (82.5% draft acceptance).

        ``lm_head`` is deliberately not applied: the draft logits are not needed
        here, and ``[8192, 248320]`` fp32 would be 8 GB.
        """
        model = self.model
        cu = batch.cu_seqlens.tolist()
        seq_slots = batch.seq_slot_ids.tolist()
        last_idx = batch.last_indices

        if self.cfg.prefill_mtp:
            prev = torch.empty_like(hidden)
            for i, slot in enumerate(seq_slots):
                lo, hi = int(cu[i]), int(cu[i + 1])
                if hi <= lo:
                    continue
                prev[lo] = self.h_prev[slot].to(hidden.dtype)
                if hi - lo > 1:
                    prev[lo + 1 : hi] = hidden[lo : hi - 1]
            cos, sin = model.rotary.lookup(batch.positions)
            ctx = StepContext(
                pool=model.kv_pool,
                attn=model.attn,
                state_pool=model.state_pool,
                conv_pool=model.conv_pool,
                slot_ids=batch.slot_ids,
                positions=batch.positions,
                cos=cos,
                sin=sin,
                seq_slot_ids=batch.seq_slot_ids,
                cu_seqlens=batch.cu_seqlens,
            )
            model.attn.plan_prefill(seq_slots, batch.q_lens, batch.kv_lens)
            emb = F.embedding(batch.token_ids.long(), model.embed_tokens).to(self.dtype)
            model.mtp(emb, prev.to(self.dtype), ctx, prefill=True)

        sel = hidden.index_select(0, last_idx).to(self.h_prev.dtype)
        self.h_prev.index_copy_(0, batch.seq_slot_ids.long(), sel)


# =========================================================================== #
# 4. builder
# =========================================================================== #
def build_spec_decoder(
    model: FusedQwenForCausalLM,
    buf: DeviceBuffers,
    rt: RuntimeConfig,
    cfg: Optional[SpecConfig] = None,
) -> SpecDecoder:
    return SpecDecoder(model, buf, rt, cfg or SpecConfig())
