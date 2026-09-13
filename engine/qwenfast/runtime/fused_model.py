"""The fused runtime model: fused weights, pool-indexed GDN kernels and a paged
KV cache, driven through a fixed device-buffer ABI.

Covers the fused weight layout, the SSM slot pool, the paged KV pool, the
device-buffer ABI, varlen chunked prefill, the kernel choices and the numerics
contract.

What this module is
-------------------
The *stateless-per-step* half of the runtime.  It owns the weights and the two
pools, and exposes exactly two forwards:

``decode_forward(buf, B)``
    One token for each of ``B`` slots.  Reads ``input_ids``/``positions``/
    ``slot_ids`` out of the fixed device buffers, updates the SSM state pool
    and the paged KV pool **in place**, writes ``[B, vocab]`` logits into
    ``buf.logits``.  No host sync, no data-dependent control flow, no
    allocation that depends on a tensor's *value*, i.e. capturable
    (``graphs.py`` does the capturing).

``prefill_forward(batch)``
    A packed varlen chunk (``cu_seqlens``, no padding) that threads GDN state
    through the slot pool and writes K/V straight into the pages.  Not
    graphed, MLP tiled at ``mlp_tile_tokens`` so an 8192-token chunk never
    materialises the full ``[8192, 34816]`` gate/up activation.

Numerics, all asserted against the reference model in ``tests/test_runtime.py``:

* ``RMSNorm`` -> fp32 accumulate, scale by **``(1 + weight)``**
* ``RMSNormGated`` (GDN out) -> fp32 accumulate, scale by **plain ``weight``**,
  then ``* silu(gate.float())``
* GDN: q/k L2-normalised *inside* the kernel, ``scale = 1/sqrt(128)``,
  ``g = -exp(A_log) * softplus(a + dt_bias)`` in fp32, ``beta = sigmoid(b)``,
  q/k in canonical 16-head form (the GDN kernels expand ``hv -> hv//3`` themselves)
* Attention: q_norm/k_norm on ``head_dim`` **before** RoPE, partial RoPE over
  the first 64 of 256 dims, ``sigmoid`` output gate applied **before**
  ``o_proj``

The module never patches the kernel, GEMM or KV-pool packages.  Two small local
workarounds around features missing there are marked ``WORKAROUND``.
"""

from __future__ import annotations

import math
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ..attn import flashinfer_attn as fi
from ..attn import fused_qk_rope
from ..attn.kv_pool import KVPoolConfig, PagedKVPool
from ..attn.rope import RotaryTable
from ..gemm import dispatch as gemm_dispatch
from ..gemm.fused_weights import (
    AttnFusedWeights,
    FP8Tensor,
    FusedModelWeights,
    GDNFusedWeights,
    MLPFusedWeights,
    MTPFusedWeights,
)
from ..kernels_gdn import api as gdn_api
from ..kernels_gdn import shapes as gdn_shapes
from ..kernels_gdn import state as gdn_state
from ..weights import QwenFastConfig
from .fused_ops import gdn_gate as fused_gdn_gate
from .fused_ops import swiglu as fused_swiglu

# --------------------------------------------------------------------------- #
# optional triton (norm kernels).  Never required; torch is canonical.
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - depends on the host
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover
    triton = None  # type: ignore
    tl = None  # type: ignore
    HAS_TRITON = False


DTYPES = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}


def resolve_dtype(name) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    try:
        return DTYPES[str(name).lower()]
    except KeyError as exc:  # pragma: no cover - argument validation
        raise ValueError(f"unknown dtype {name!r}") from exc


# =========================================================================== #
# 1. fused element-wise ops (torch canonical, optional Triton)
# =========================================================================== #
def norm_weight_1p(weight: torch.Tensor) -> torch.Tensor:
    """Precompute ``(1 + weight)`` in fp32, once, at model build time.

    ``Qwen3_5RMSNorm`` scales by ``1 + weight``.  Written the obvious
    way, ``out * (1.0 + weight.float())`` launches a cast kernel and an add
    kernel **per call**, and a decode step has 129 RMSNorm sites, so that
    is ~258 kernel launches per step spent recomputing a constant.  At the
    measured ~1.5 us/launch floor a graph-replayed step pays for tiny kernels
    that is ~0.4 ms/step, ~2.5% of the B=1 step, for nothing.  The value is bit-identical to the inline form.
    """
    return (1.0 + weight.float()).contiguous()


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """``Qwen3_5RMSNorm``: fp32 accumulate, scale by ``(1 + weight)``."""
    return rms_norm_w1p(x, norm_weight_1p(weight), eps)


def rms_norm_w1p(x: torch.Tensor, w1p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """:func:`rms_norm` with ``(1 + weight)`` already computed (fp32)."""
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    out = out * w1p
    return out.to(x.dtype)


def add_rms_norm(
    residual: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused ``residual += x; return (residual, rms_norm(residual))``.

    This is the shape every decoder layer boundary has: the block output is
    added into the residual stream and the *sum* is what the next norm sees.
    Doing it in one pass halves the residual-stream traffic (two reads + one
    write instead of three reads + two writes).  Bit-comparable to the reference model's
    ``hidden = residual + h; h = norm(hidden)``.
    """
    return add_rms_norm_w1p(residual, x, norm_weight_1p(weight), eps)


def add_rms_norm_w1p(
    residual: torch.Tensor, x: torch.Tensor, w1p: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """:func:`add_rms_norm` with ``(1 + weight)`` already computed (fp32)."""
    res = residual + x
    return res, rms_norm_w1p(res, w1p, eps)


def rms_norm_gated(
    x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """GDN output norm (``RMSNormGated``): **plain** ``weight``, then silu gate.

    Deliberately *not* ``(1 + weight)``: mixing the two norms up is a very
    easy bug to write.  Copied op-for-op from ``model.RMSNormGated.forward`` so the two
    cannot drift.
    """
    in_dtype = x.dtype
    h = x.to(torch.float32)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    h = weight * h.to(in_dtype)
    h = h * F.silu(gate.to(torch.float32))
    return h.to(in_dtype)


def swiglu(gate_up: torch.Tensor) -> torch.Tensor:
    """``silu(gate) * up`` for a fused ``[..., 2 * inter]`` activation."""
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up


# --------------------------------------------------------------------------- #
# Optional Triton fused norms.  `--norm-backend triton`.  Off by default
# (`norm_backend="torch"`): an unverified kernel in the residual stream would
# poison every downstream number, so enable it only after the GPU parity test
# has passed on the target build.
#
# Why they exist: with the GEMM backend fixed, the B=1 decode step's
# second-largest cost is not a kernel, it is the *number* of kernels.
# The graph-replayed step launched 3,922 of them, of which ~3,400 were the
# `copy` + `elementwise/other` groups (5.4 ms, 36% of the step) at a
# floor of ~1.5 us per launch regardless of how little work each does.
# Eager torch spends ~9 launches on one RMSNorm (cast, square, mean, add-eps,
# rsqrt, mul, mul-weight, cast back, ...), ~10 on residual-add + RMSNorm, and
# ~11 on the gated GDN output norm.  A decode step has 64 + 64 + 1 = 129 of
# the first two and 48 of the third, i.e. ~1,600 launches spent on three
# ops that are each a single pass over one row.
#
# Each kernel below is one launch and one pass, and each keeps the
# numerics contract exactly:
#   * `_rms_norm_kernel`      -> fp32 accumulate, scale by the precomputed
#                                fp32 `(1 + weight)`, cast back at the end.
#   * `_add_rms_norm_kernel`  -> the same, plus it writes the un-normalised
#                                residual sum (the next layer's residual).
#   * `_rms_norm_gated_kernel`-> **plain** weight, not `1 + weight` (an
#                                easy bug to write), and it
#                                reproduces eager's cast order: normalise in
#                                fp32, round to the activation dtype, scale
#                                by the weight in that dtype, then multiply
#                                by `silu(gate)` computed in fp32.
# `tests/test_runtime.py::TestTritonNorms` asserts all three against the
# torch functions on GPU; the default stays `norm_backend="torch"`.
# --------------------------------------------------------------------------- #
def _norm_block(n_cols: int) -> Tuple[int, int]:
    """(BLOCK, num_warps) for a one-row-per-program norm over ``n_cols``."""
    block = 1 << (n_cols - 1).bit_length()
    if block <= 256:
        warps = 2
    elif block <= 1024:
        warps = 4
    elif block <= 4096:
        warps = 8
    else:
        warps = 16
    return block, warps


if HAS_TRITON:  # pragma: no cover - GPU only

    @triton.jit
    def _rms_norm_kernel(
        X_ptr, W1P_ptr, OUT_ptr,
        stride_row,
        n_cols: tl.constexpr,
        eps: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < n_cols
        x = tl.load(X_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / n_cols
        inv = 1.0 / tl.sqrt(var + eps)
        w = tl.load(W1P_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        out = x * inv * w
        tl.store(OUT_ptr + row * stride_row + offs, out.to(OUT_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _add_rms_norm_kernel(
        X_ptr, RES_ptr, W1P_ptr, OUT_ptr, OUT_RES_ptr,
        stride_row,
        n_cols: tl.constexpr,
        eps: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < n_cols
        x = tl.load(X_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(RES_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
        s = x + r
        s_cast = s.to(OUT_RES_ptr.dtype.element_ty)
        tl.store(OUT_RES_ptr + row * stride_row + offs, s_cast, mask=mask)
        # Normalise the *stored* (rounded) residual, not the fp32 sum: eager
        # does `res = residual + x` into an activation-dtype tensor and then
        # norms that tensor, so anything else drifts from the torch path.
        sr = s_cast.to(tl.float32)
        var = tl.sum(sr * sr, axis=0) / n_cols
        inv = 1.0 / tl.sqrt(var + eps)
        w = tl.load(W1P_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        out = sr * inv * w
        tl.store(OUT_ptr + row * stride_row + offs, out.to(OUT_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _rms_norm_gated_kernel(
        X_ptr, GATE_ptr, W_ptr, OUT_ptr,
        stride_row,
        n_cols: tl.constexpr,
        eps: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < n_cols
        x = tl.load(X_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / n_cols
        inv = 1.0 / tl.sqrt(var + eps)
        # eager: h = weight * h.to(in_dtype)  -- the normalised value is
        # rounded to the activation dtype *before* the weight multiply.
        hn = (x * inv).to(OUT_ptr.dtype.element_ty)
        w = tl.load(W_ptr + offs, mask=mask, other=0.0)
        hw = (w * hn).to(tl.float32)
        g = tl.load(GATE_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
        silu = g / (1.0 + tl.exp(-g))
        out = hw * silu
        tl.store(OUT_ptr + row * stride_row + offs, out.to(OUT_ptr.dtype.element_ty), mask=mask)

    def triton_rms_norm(x, w1p, eps=1e-6):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        out = torch.empty_like(x2)
        n_cols = shape[-1]
        block, warps = _norm_block(n_cols)
        _rms_norm_kernel[(x2.shape[0],)](
            x2, w1p, out, x2.stride(0), n_cols=n_cols, eps=eps, BLOCK=block, num_warps=warps
        )
        return out.reshape(shape)

    def triton_add_rms_norm(residual, x, w1p, eps=1e-6):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        r2 = residual.reshape(-1, shape[-1])
        out = torch.empty_like(x2)
        out_res = torch.empty_like(x2)
        n_cols = shape[-1]
        block, warps = _norm_block(n_cols)
        _add_rms_norm_kernel[(x2.shape[0],)](
            x2, r2, w1p, out, out_res,
            x2.stride(0), n_cols=n_cols, eps=eps, BLOCK=block, num_warps=warps,
        )
        return out_res.reshape(shape), out.reshape(shape)

    def triton_rms_norm_gated(x, gate, weight, eps=1e-6):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        g2 = gate.reshape(-1, shape[-1])
        out = torch.empty_like(x2)
        n_cols = shape[-1]
        block, warps = _norm_block(n_cols)
        _rms_norm_gated_kernel[(x2.shape[0],)](
            x2, g2, weight, out, x2.stride(0), n_cols=n_cols, eps=eps, BLOCK=block, num_warps=warps
        )
        return out.reshape(shape)

else:

    def triton_rms_norm(*args, **kwargs):  # pragma: no cover
        raise RuntimeError("triton is not importable in this environment")

    def triton_add_rms_norm(*args, **kwargs):  # pragma: no cover
        raise RuntimeError("triton is not importable in this environment")

    def triton_rms_norm_gated(*args, **kwargs):  # pragma: no cover
        raise RuntimeError("triton is not importable in this environment")


# =========================================================================== #
# 2. linear dispatch with a sticky backend
# =========================================================================== #
#: One-slot mutable cell holding the backend name every
#: ``ResolvedLinear`` must use *for the duration of a prefill forward*. A list
#: rather than a module global rebound by ``global`` so the read in the hot
#: path is a single ``LOAD_GLOBAL`` + index rather than a dict lookup, and so
#: the scope is trivially re-entrant-safe on the one thread that uses it (the
#: engine loop, and there is exactly one).
_PREFILL_GEMM_BACKEND: List[Optional[str]] = [None]


@contextmanager
def prefill_gemm_scope(backend: Optional[str]):
    """Force ``backend`` for every linear inside the block (``None`` = no-op).

    **Memory warning.** This deliberately bypasses
    ``gemm.dispatch._policy_filtered``, which is the thing that stops a second
    23 GiB repacked copy of the fp8 weights being materialised under
    ``--gemm-weight-cache single`` (that copy is enough to OOM a serving
    process). Forcing a backend that owns a repack cache
    (``scaled_mm_pertensor``, ``vllm_cutlass_fp8_pertensor``, ``deepgemm``,
    ``vllm_marlin_fp8_w8a16``, ``machete_w8a16``) when a *different* one is
    already cached will allocate that second copy on the first prefill chunk.
    ``serve.py`` refuses the combination up front; use one of the cache-free
    backends (``flashinfer_fp8_blockscale``, ``vllm_block_fp8_cutlass``,
    ``vllm_block_fp8_triton``) unless the plan has 23 GiB spare.
    """
    if backend is None:
        yield
        return
    prev = _PREFILL_GEMM_BACKEND[0]
    _PREFILL_GEMM_BACKEND[0] = backend
    try:
        yield
    finally:
        _PREFILL_GEMM_BACKEND[0] = prev


class ResolvedLinear:
    """``gemm.dispatch.linear`` with the winning backend memoised **per M-bucket**.

    ``linear()`` walks a try/except fallback chain on every call.  That is
    exactly right for robustness and exactly wrong inside a CUDA graph: an
    exception thrown mid-capture leaves the capture in an undefined state.
    So the backend is resolved **once per M-bucket** during warmup (which
    ``graphs.py`` always runs before capturing) and passed explicitly from
    then on, turning the hot path into a single unconditional call.

    Two design points matter:

    1. **Resolution calls each backend directly.**  ``gemm_dispatch.linear``
       swallows a failing backend and walks on down the priority order, so it
       returns successfully no matter which ``backend=`` was asked for and
       cannot be used to confirm that a candidate works.  Resolution goes
       through ``gemm_dispatch.resolve_backend``, which invokes each
       candidate's implementation *directly* and returns the first that
       really works.  Pinning the wrong backend is expensive: the slowest fp8
       backend graph-times at 31.5 ms/step whole-model at M=1 versus
       marlin's 8.2 ms.

    2. **One backend per M-bucket, not per weight.**  Warmup runs bucket 1
       first, so a single memo would freeze the M=1 winner and reuse it at
       M=256, where, for marlin, it is 5x worse than the right choice
       (44.8 ms vs flashinfer's 15.9).  The memo is keyed on
       ``gemm_dispatch.m_bucket(M)``, matching the granularity of
       ``DEFAULT_BACKEND_PRIORITY_BY_M_BUCKET`` and of the autotune cache.

    ``backend=`` (from ``RuntimeConfig.gemm_backend``) still force-pins one
    backend for every shape, for A/B experiments.
    """

    __slots__ = ("weight", "n", "backend", "sm_version", "forced", "forced_ignored",
                 "is_fp8", "_by_bucket", "reasons")

    def __init__(self, weight, sm_version: Optional[int] = None, backend: Optional[str] = None):
        self.weight = weight
        self.is_fp8 = isinstance(weight, FP8Tensor)
        self.n = weight.weight.shape[0] if self.is_fp8 else weight.shape[0]
        self.sm_version = sm_version
        #: A forced `backend=` naming an fp8 kernel cannot be honoured for a
        #: weight the checkpoint stores in bf16: 49 of this model's 305
        #: linears (the 48 GDN `in_proj_ba` [96, 5120], whose N=96 is not
        #: 128-aligned so `fused_weights.py` never quantizes them, plus the
        #: bf16 `lm_head`). Forcing the fp8 name onto them would send every
        #: call through `dispatch.linear`'s full try/except chain (several
        #: thrown-and-caught exceptions per layer per step) and make
        #: `resolved_backends()` report a backend those layers never ran.
        #: Pin the truth at construction instead.
        if backend is not None and not self.is_fp8:
            self.forced = "bf16_native"
            self.forced_ignored = backend
        else:
            self.forced = backend
            self.forced_ignored = None
        self.backend = self.forced  # last resolved backend (introspection / profiling)
        self._by_bucket: Dict[int, str] = {}
        #: ``{m_bucket: {backend: why it was not chosen}}``, filled during
        #: warmup resolution only. Diagnostic; nothing on the hot path reads it.
        self.reasons: Dict[int, Dict[str, str]] = {}

    def resolved_backends(self) -> Dict[int, str]:
        """``{m_bucket: backend_name}`` resolved so far -- what
        ``runtime/profile_step.py`` prints per layer. Bucket ``0`` means "one
        backend for every bucket", i.e. a forced pin."""
        if self.forced is not None:
            return {0: self.forced}
        return dict(self._by_bucket)

    def rejection_reasons(self) -> Dict[int, Dict[str, str]]:
        """``{m_bucket: {backend: why}}`` for every candidate that lost during
        warmup resolution. Empty for a forced pin (nothing was resolved)."""
        return {b: dict(r) for b, r in self.reasons.items()}

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.forced is not None:
            return gemm_dispatch.linear(x, self.weight, backend=self.forced, use_autotune=False)
        if _PREFILL_GEMM_BACKEND[0] is not None:
            # Inside `prefill_gemm_scope` only, i.e. inside
            # `prefill_forward`, which is never CUDA-graph captured, so the
            # "no exceptions during capture" rule that made `_by_bucket`
            # necessary does not apply here, and neither does the memo (a
            # forced name needs no resolution).
            #
            # Same correction as `forced` above: an fp8 name cannot apply to a
            # never-quantized weight, and prefill is the *eager* path, so a
            # per-call exception walk would be paid in full there rather than
            # absorbed by a graph replay.
            return gemm_dispatch.linear(
                x,
                self.weight,
                backend=_PREFILL_GEMM_BACKEND[0] if self.is_fp8 else "bf16_native",
                use_autotune=False,
            )
        m = 1
        for d in x.shape[:-1]:
            m *= d
        # `sequence_m` is the identity everywhere except inside a
        # `gemm_dispatch.rows_per_sequence(n)` scope, which the speculative
        # verify pass opens because it packs n = k+1 token rows per *sequence*
        # while standing in for a decode step of B sequences. Without it the
        # verify pass at B=32/k=3 keys on M=128 and gets
        # `flashinfer_fp8_blockscale` where the decode step it must reproduce
        # got marlin: a measured 2.6e-2 relative difference, not a tiling
        # difference.
        bucket = gemm_dispatch.m_bucket(gemm_dispatch.sequence_m(m))
        name = self._by_bucket.get(bucket)
        if name is None:
            # Warmup path only: `graphs.py::GraphedDecoder.warmup` runs every
            # bucket eagerly before capture precisely so this never fires
            # inside a capture (where a raising backend would corrupt it).
            why: Dict[str, str] = {}
            name = gemm_dispatch.resolve_backend(
                x, self.weight, sm_version=self.sm_version, reasons=why
            )
            self._by_bucket[bucket] = name
            self.reasons[bucket] = why
            self.backend = name
        return gemm_dispatch.linear(x, self.weight, backend=name, use_autotune=False)

    def nbytes(self) -> int:
        w = self.weight
        if isinstance(w, FP8Tensor):
            return w.weight.numel() * w.weight.element_size() + w.scale_inv.numel() * w.scale_inv.element_size()
        return w.numel() * w.element_size()


# =========================================================================== #
# 3. runtime config
# =========================================================================== #
@dataclass
class RuntimeConfig:
    """Everything the runtime needs that is not in the checkpoint config."""

    device: str = "cuda:0"
    dtype: str = "bf16"  # activation dtype (bf16 everywhere)

    # --- pools ------------------------------------------------------------- #
    ssm_state_dtype: str = "fp32"  # --ssm-state-dtype {fp32,fp16}
    kv_cache_dtype: str = "bf16"  # --kv-cache-dtype {bf16,fp8}
    page_size: int = 16
    max_num_seqs: int = 512  # == n_ssm_slots (one SSM slot per running sequence)
    n_kv_pages: int = 8192
    max_pages_per_seq: int = 1024

    # --- scheduling / shapes ------------------------------------------------ #
    max_num_batched_tokens: int = 8192  # chunked-prefill budget
    prefill_decode_ratio: int = 4  # --prefill-decode-ratio (default 1:4)
    mlp_tile_tokens: int = 2048  # prefill activation cap
    max_model_len: int = 262144
    gdn_chunk_size: int = 64
    #: Token tile for the depthwise causal conv during prefill. The torch
    #: fallback (`kernels_gdn.torch_ops.conv_prefill`) computes in fp32, so an
    #: untiled `max_num_batched_tokens`-wide chunk costs
    #: `conv_dim x T x 4 B` in *and* the same again out -- 671 MiB per layer at
    #: T=8192, C=10240, enough to push a fully loaded server into OOM. Tiling makes that bound independent of
    #: `max_num_batched_tokens`: 84 MiB at the default 2048.
    conv_prefill_tile_tokens: int = 2048
    #: ``token_major`` routes the prefill conv through
    #: ``kernels_gdn.causal_conv_prefill_varlen``, which takes the ``[T, C]``
    #: activation ``FusedGDN.prefill`` already holds and (with Triton available) runs
    #: the whole packed chunk in two launches. ``channel_major`` is the
    #: original path: transpose to ``[1, C, T]``, ``F.conv1d`` in fp32 tiled at
    #: ``conv_prefill_tile_tokens``, transpose back, which also does one
    #: ``cu_seqlens.to("cpu")`` D2H sync *per GDN layer per chunk*.
    #:
    #: Kept as a knob rather than a straight replacement so the two can be
    #: benchmarked against each other on the same build, and so a numerics
    #: surprise has a one-flag rollback. ``conv_prefill_tile_tokens`` only applies to
    #: ``channel_major``.
    conv_prefill_layout: str = "token_major"
    #: Run **one** forward per step over
    #: ``[prefill chunk tokens ‖ decode rows]`` instead of alternating a
    #: prefill step and a decode step (vLLM's shape).
    #:
    #: Every GEMM, norm and MLP in the step then runs once at
    #: ``M = T_prefill + B_decode``, so the decode rows ride along in GEMMs
    #: whose weight reads the prefill chunk was already paying for. At conc
    #: 256 the separate-step design is bounded at ~2,120 out tok/s (decode
    #: 42.7 ms + 1,095 prefill tok / 13,991 tok/s); the mixed step is
    #: ~(1,095+256)/13,991 = 97 ms/step ~= 2,650 out tok/s.
    #:
    #: The mixed step runs **eager**: its shape is data-dependent (the chunk
    #: is a different varlen shape every step), so it cannot be a CUDA graph.
    #: That is affordable only because at ``M >= 2k`` the GEMMs dominate and
    #: the ~1,700 launches are hidden behind them; the graphed decode-only step
    #: is kept for steps with nothing to prefill, and the speculative step for
    #: small batches. Defaulted **off**; enable it after benchmarking the
    #: target workload.
    mixed_forward: bool = False
    #: CUDA-graph the mixed step.
    #:
    #: The paragraph above ("runs eager ... cannot be a CUDA graph") is only
    #: half true. It *is* true that the varlen packing
    #: changes every step; it is **not** true that the whole step therefore has
    #: to be launched op by op. Pad the prefill half to a fixed
    #: ``prefill_chunk_tokens`` and the decode half to a graph bucket and every
    #: kernel in the step has a fixed shape except the two that read
    #: ``cu_seqlens`` on the host (the fla chunk kernel's index prep and the
    #: gather/scatter around it). Those stay eager, in a **hole** between two
    #: captured graphs: one hole per GDN layer, ~49 graph segments per step.
    #:
    #: Why it matters: at ``--prefill-chunk-tokens 1024`` the
    #: measured mixed step is 145-153 ms for ~1,024 prefill tokens + ~120
    #: decode rows, against ~71 ms of chunk GEMM (14.4k tok/s) + ~20 ms of
    #: decode-row work. The ~60 ms remainder is host launch latency that the
    #: graphed decode step hides and the eager mixed step does not, and it is
    #: the whole reason small-chunk mixing (the every-step shape that reaches
    #: the ~2,650 out tok/s bound above) loses to 8k chunks when run eagerly.
    #:
    #: Requires ``mixed_forward``; ignored without CUDA graphs. Off by default.
    mixed_graphs: bool = False
    #: Prefill **plan rows** in a graphed mixed step: at most
    #: ``mixed_graph_segments - 1`` real segments, and always at least one
    #: padding segment (padding rows are a segment on the scratch slot). Fixed
    #: per graph because it is FlashInfer's ``qo_indptr`` length, and every row
    #: has to be non-empty; see ``mixed_graphs.pad_mixed_step``.
    mixed_graph_segments: int = 8
    #: Keep the **eager holes** (49 graph segments with one eager hole per GDN
    #: layer for the varlen conv + fla chunk kernel) instead of capturing the
    #: mixed step as ONE graph.
    #:
    #: ``False`` (the default) needs two things the holed layout does not: fla driven
    #: with caller-supplied ``chunk_indices``/``chunk_offsets``
    #: (``kernels_gdn.fla_static``, which fla 0.5.2 already supports) and the
    #: varlen conv launched over a token-axis grid pinned to the chunk cap
    #: rather than to this step's longest segment. With both, nothing in the
    #: step reads the segmentation on the host and the 48 holes close.
    #:
    #: ``True`` is the rollback and the A/B arm: same padding, same numerics,
    #: 49 replays and 48 eager holes instead of one replay. The runner also
    #: falls back to it, with a reason, on a build whose fla cannot take the
    #: index tensors.
    mixed_graph_holes: bool = False
    #: Smallest decode-row bucket a graphed mixed step is captured for.
    #: Buckets below this are dropped from the ladder and their steps pad up to
    #: it, which costs padding rows and saves whole graphs: at conc 256 the
    #: default ladder is 14 buckets and this cuts it to 4
    #: (32/64/128/256), i.e. 4 captures instead of 14, which can be the
    #: difference between a 4,096-token chunk capturing at all and running out
    #: of memory in warmup.
    #:
    #: A mixed step *always* has at least one decode row (``_run_mixed_step``
    #: returns early otherwise) and at the serving geometry has 95-166 of them,
    #: so nothing is pushed off the graphed path by this; steps with fewer rows
    #: than the floor are padded up to it rather than run eagerly.
    mixed_graph_min_bucket: int = 32
    #: Explicit decode-row bucket ladder for the mixed-step
    #: graphs, overriding ``buckets_for()``/``mixed_graph_min_bucket``
    #: entirely. Empty (the default) keeps the derived ladder.
    #:
    #: The overlap step (``overlap_streams``) sets this to ``(1,)``: its
    #: prefill graph carries exactly one *padding* decode row on the scratch
    #: slot, because the real decode rows are a second graph on a second
    #: stream. One bucket is one capture, and a 1-row bucket is the cheapest
    #: shape ``MixedGraphRunner`` can legally hold (a mixed step with zero
    #: decode rows has no bucket at all; see ``MixedPadSpec.bucket_for``).
    mixed_graph_buckets: Tuple[int, ...] = ()
    #: Run the prefill chunk and the decode step of one
    #: scheduler step **concurrently on two CUDA streams** instead of fusing
    #: them into one row-concatenated forward (``mixed_forward``).
    #:
    #: The rationale: inside a mixed step the prefill half is
    #: compute-bound (GEMMs at ~58 % of fp8 dense peak) and the decode half is
    #: HBM-bandwidth-bound (weights + KV + SSM state), and run serially they
    #: *add*. Two streams make the step ``max(prefill, decode)`` instead of
    #: ``prefill + decode`` to the extent the two halves really do contend for
    #: different resources.
    #:
    #: Turning this on changes three things that are otherwise invariants:
    #:
    #: * the mixed-step graphs get their **own** mempool (they no longer
    #:   replay one-at-a-time against the decode graphs' pool, so the two
    #:   captures' intermediates must not alias);
    #: * the FlashInfer **decode** wrappers get their own workspace buffer
    #:   (``AttentionRunner.decode_workspace``), because a prefill ``run()``
    #:   and a decode ``run()`` are now in flight at the same time and
    #:   FlashInfer's float workspace is per-call scratch;
    #: * the decode rows' GEMMs are no longer the chunk's GEMMs, so the model
    #:   weights are read twice per step (~6 ms of HBM time at B=256).
    #:
    #: Requires ``mixed_forward``, ``mixed_graphs`` and CUDA graphs.
    overlap_streams: bool = False
    #: CUDA stream priority for the decode half of an overlapped step
    #: (``torch.cuda.Stream(priority=...)``; lower is higher priority on
    #: CUDA). ``0`` is the default priority, ``-1`` asks the driver to
    #: schedule the decode blocks ahead of the prefill GEMM's.
    overlap_decode_priority: int = 0
    #: The smallest fraction of the graph's fixed chunk a
    #: step's **real** prefill tokens may fill before the step gives up the
    #: graph and runs its prefill half eagerly.
    #:
    #: A graphed prefill chunk is padded to exactly ``prefill_chunk_tokens``
    #: every step. That is free when the scheduler has that much work waiting
    #: and it is a **4x waste of the GPU** when it does not: at concurrency 32
    #: with an 8,192-token graph, most steps ingested one 2,139-token prompt
    #: padded to 8,192, so GPU utilisation went *up* (85.7 % vs 79.1 %
    #: all-eager) while throughput went *down* (719 vs 908 out tok/s). The
    #: card was busy computing padding.
    #:
    #: Below this fraction the chunk runs through ``prefill_forward``
    #: unpadded, exactly as a plain prefill step would. ``0.0`` restores the
    #: always-graphed behaviour.
    overlap_min_fill: float = 0.75
    #: Schedule step N+1 on the host while step N runs on the
    #: device, and harvest step N's tokens one step later through a pinned
    #: buffer and a CUDA event (vLLM's "async scheduling").  The sampled token
    #: of step N is an *input* of step N+1, so it is fed forward with a
    #: device-side gather and never crosses to the host on the critical path;
    #: the host learns it one step late, which costs at most one extra
    #: (discarded) token per request after EOS.  Off by default.
    async_scheduling: bool = False
    #: Hard cap on the tokens in one prefill chunk, if
    #: smaller than ``max_num_batched_tokens``. 0 == no separate cap.
    #:
    #: This is also the mixed step's prefill budget, and under
    #: ``mixed_forward`` it is the **p99 knob**: a mixed step's decode rows
    #: wait for the whole chunk, so an 8,192-token chunk parks all 256 of them
    #: for ~590 ms while a 2,048-token one parks them for ~150 ms. Aggregate
    #: throughput is (to first order) unchanged either way.
    #:
    #: This is a *latency* knob, not a throughput knob: total wall time is
    #: ``prefill_tokens/prefill_rate + decode_steps * step_time`` no matter how
    #: the two are interleaved, so neither this nor ``prefill_decode_ratio``
    #: moves aggregate tok/s. What it moves is TPOT: at conc 256 an 8192-token
    #: chunk parks every one of 256 running sequences for ~1.1 s, which is why
    #: the measured TPOT p50 is 195 ms against a 45.8 ms decode step.
    prefill_chunk_tokens: int = 0

    # --- backends ----------------------------------------------------------- #
    gdn_backend: str = "auto"  # auto|torch|fla|triton
    gemm_backend: Optional[str] = None  # None = dispatch/autotune
    #: multi|single|none -- how many permanent repacked weight copies the GEMM
    #: dispatcher may memoise per weight (`gemm.dispatch.set_weight_cache_policy`).
    #: "multi" is the offline-benchmark behaviour; **serving uses
    #: "single"**, because "multi" can silently duplicate all 23.0 GiB of FP8
    #: linear weights twice (marlin at the decode buckets + per-tensor fp8 at
    #: prefill's M-bucket 512) and OOM the server.
    gemm_weight_cache: str = "multi"
    #: v8|v7|v4 -- which measured cold-start priority table
    #: (`gemm.dispatch.set_backend_priority_profile`) the dispatcher ranks with.
    #: "v8" (**default**) is v7 unchanged at M <= 512 plus four prefill-scale
    #: buckets (1024/2048/4096/8192) that v7 clamps to the M=512 answer; see
    #: `gemm.dispatch.V8_BACKEND_PRIORITY_BY_M_BUCKET`. "v7" is the graph-timed
    #: `gemm_v7` table (deepgemm rank 1 at every M in [32, 512]); "v4" is an
    #: older table. Both kept as one-flag rollbacks, and "v7" is also the
    #: baseline for checking whether routing a real chunk on its own measured
    #: bucket beats the M=512 answer.
    gemm_priority: str = "v8"
    #: fast|strict -- the accuracy bar the GEMM dispatcher must clear
    #: (`gemm.dispatch.set_gemm_accuracy`).
    #:
    #: "fast" (default): rank backends on graph-timed speed alone. That routes
    #: every M >= 64 (i.e. every decode batch of 64+ and *every prefill
    #: chunk*) to
    #: `flashinfer_fp8_blockscale`, which quantizes the activations to fp8 and
    #: measures relL2 2.6e-2 against an exact fp32 reference, against 2.7e-3
    #: for the W8A16 backend the same engine uses below M=64.
    #:
    #: "strict": only backends measured at or under
    #: `dispatch.STRICT_REL_L2_MAX` (5e-3) may be resolved, i.e. W8A16 only.
    #: Costs real time at high M (marlin is 23.4 ms/step at M=128 against
    #: flashinfer's 10.3) and is what speculative decoding needs for its
    #: verify pass to reproduce its own decode step.
    #:
    #: **The default stays "fast"** because "strict" has a real end-to-end
    #: throughput cost; choose it when exact verify parity matters.
    gemm_accuracy: str = "fast"
    #: Backend forced for **prefill-shaped** GEMMs only
    #: (``None`` = whatever ``m_bucket`` dispatch picks). Two facts make this
    #: knob worth having:
    #:
    #: * ``gemm.dispatch.m_bucket`` clamps at 512, so M=2048 and M=8192 route
    #:   on a table measured at M=512 and never validated above it; and
    #: * ``--gemm-weight-cache single`` (the serving default)
    #:   *denies* bucket 512 its measured winner ``scaled_mm_pertensor``,
    #:   because marlin already claimed the one repack slot at warmup bucket 1.
    #:   Prefill therefore silently runs ``flashinfer_fp8_blockscale``:
    #:   30.88 ms/step at M=512 against scaled_mm's 28.08, ~10%
    #:   of a chunk's GEMM.
    #:
    #: Every fp8-activation backend measures the same 2.7e-2 relL2 class
    #: (flat in M), so at prefill M the choice is purely a speed
    #: choice, with the one exception that ``vllm_marlin_fp8_w8a16`` is the
    #: *wrong* answer here (88.83 ms at M=512, and it degrades monotonically).
    prefill_gemm_backend: Optional[str] = None
    #: Which M-bucket gets to claim the **one** weight
    #: repack cache slot that ``--gemm-weight-cache single`` allows.
    #:
    #: ``"decode"`` (default): nothing special happens, so warmup (which runs
    #: the decode buckets in ascending order) resolves bucket 1 first, marlin
    #: claims the slot, and every higher bucket is thereafter denied the
    #: cache-owning backend it would otherwise pick. On the gemm_v7 table that means the
    #: server runs ``flashinfer_fp8_blockscale`` at M=512 (31.19 ms/step)
    #: where ``deepgemm`` measures 26.65, and 16.82 at M=256 where deepgemm
    #: measures 14.13, because marlin's **3% win at M=1** (8.23 vs 8.50)
    #: holds the slot for the whole process.
    #:
    #: ``"prefill"``: :meth:`FusedQwenForCausalLM.claim_gemm_cache` resolves
    #: every linear at a prefill-shaped M *before* warmup touches the decode
    #: buckets, so the large-M winner claims the slot and bucket 1 falls to the
    #: best cache-free backend (flashinfer, 8.79 vs marlin's 8.23). Still
    #: exactly one cache, so the memory plan is unchanged to the byte.
    #:
    #: The trade in one line, at conc 256 and the v7 numbers: pay ~0.56 ms on
    #: a B=1 step to save ~73 ms on every 8192-token prefill chunk and ~2.7 ms
    #: on every B=256 decode step. Defaulted **off** because the trade depends
    #: on the priority table and the workload; benchmark before enabling.
    gemm_cache_owner: str = "decode"
    attn_backend: str = "auto"  # auto|flashinfer|torch
    norm_backend: str = "torch"  # torch|triton (see the note above)
    # torch|triton (kernel launch reduction): fuses SwiGLU
    # (FusedMLP) and the GDN gate epilogue (FusedGDN._project) from
    # 2-4 eager kernel launches each into 1. Same opt-in convention as
    # norm_backend: "torch" (unchanged eager path) stays the default
    # until TestFusedOps has passed on the target GPU; see
    # engine/qwenfast/runtime/fused_ops/ for the kernels and their parity tests.
    fused_ops_backend: str = "torch"

    # --- features ------------------------------------------------------------ #
    enable_mtp: bool = False
    mtp_hidden_first: bool = False

    # --- graphs / sampler ----------------------------------------------------- #
    use_cuda_graphs: bool = True
    graph_buckets: Tuple[int, ...] = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)
    sampler_candidates: int = 2048  # top-k/top-p candidate pool (see graphs.py)
    attn_workspace_mb: int = 512  # 192 overflowed batch_prefill_tmp_v at B=512 (needs ~207 MB)

    def torch_dtype(self) -> torch.dtype:
        return resolve_dtype(self.dtype)

    def mixed_buckets_for(self) -> Tuple[int, ...]:
        """The decode-row bucket ladder the mixed-step graphs are captured for.

        One place, because three callers have to agree on it: the
        runner that captures the graphs (``engine.build_mixed_runner``), the
        memory plan that budgets for them (``serve.plan_memory``), and the
        scheduler's padding spec. ``mixed_graph_buckets`` overrides
        everything; otherwise ``--overlap`` means one 1-row bucket (its decode
        rows are a separate graph) and anything else means the derived ladder,
        which ``MixedGraphRunner`` then trims at ``mixed_graph_min_bucket``.
        """
        if self.mixed_graph_buckets:
            return tuple(int(b) for b in self.mixed_graph_buckets)
        if self.overlap_streams:
            return (1,)
        return self.buckets_for()

    def buckets_for(self, max_seqs: Optional[int] = None) -> Tuple[int, ...]:
        cap = max_seqs if max_seqs is not None else self.max_num_seqs
        keep = tuple(b for b in self.graph_buckets if b <= cap)
        if not keep:
            return (cap,)
        if keep[-1] < cap:
            keep = keep + (cap,)
        return keep


def default_claim_gemm_cache_m(rt: "RuntimeConfig") -> int:
    """The M `claim_gemm_cache` resolves against when no explicit ``m=`` is
    given: the real prefill-chunk token cap (``prefill_chunk_tokens`` if set,
    else ``max_num_batched_tokens``), the same expression `serve.py`
    logs as ``prefill_chunk_tokens={rt.prefill_chunk_tokens or
    rt.max_num_batched_tokens}``.

    The claim must be made at the M a real prefill chunk actually presents.
    A fixed M such as 512 would hand the GEMM repack-cache slot to whoever
    wins the *decode* M=512 bucket under ``--gemm-cache-owner prefill``, so
    the cache slot would decide prefill's backend by accident. Pulled out as
    a free function (rather than inlined in ``claim_gemm_cache``) so it is
    testable on CPU without building a model.

    When ``mixed_forward`` is on there is no such thing as a step whose M is
    the chunk. A mixed step's activation is the chunk **concatenated with
    every decode row**, so its M is ``chunk + max_num_seqs``: 1,280 at chunk
    1,024 / 256 seqs, which is a different `m_bucket` from 1,024 (1536 vs
    1024). A slot claimed at the chunk's M would be claimed for a bucket the
    server never routes on, and the bucket it *does* route on would fall back
    to a cache-free backend, defeating ``--gemm-cache-owner prefill``.
    """
    m = rt.prefill_chunk_tokens or rt.max_num_batched_tokens
    if getattr(rt, "mixed_forward", False):
        m += rt.max_num_seqs
    return m


#: Shape classes whose activation is split by `FusedMLP.__call__(tiled=True)`
#: at `RuntimeConfig.mlp_tile_tokens`. These two never see a step's M above the
#: tile -- see `claim_gemm_cache_m_for`.
_MLP_TILED_SHAPE_CLASSES = ("mlp_gate_up", "mlp_down")


def claim_gemm_cache_m_for(rt: "RuntimeConfig", n: Optional[int], k: Optional[int]) -> int:
    """The M **this particular weight** will be asked for, given the step's M.

    `default_claim_gemm_cache_m` gives the *step's* M,
    and for most linears that is what they see. It is not what the two MLP
    shapes see: `FusedMLP.__call__(tiled=True)` splits the activation at
    `mlp_tile_tokens` (2,048 by default), so above that a
    [34816, 5120] or [5120, 17408] weight is called at 2,048 rows plus a
    remainder and **never** at the step's M.

    That matters because `--gemm-weight-cache single` gives a weight exactly
    one repack-cache slot, and whoever claims it decides that weight's backend
    at *every* bucket. Claiming at the step's M therefore pins the MLP weights
    for a bucket they never route on, and the two answers genuinely differ:
    measured, `mlp_down` wants `deepgemm` at bucket 3072 but
    `vllm_cutlass_fp8_pertensor` at bucket 2048, where `deepgemm` is the
    *worst* of the five (406.0 vs 339.5 µs, +19.6 %). Claiming at the step's M
    would make that weight 20 % slower for the whole run.

    So the claim is per weight, at the M that weight is actually called with.
    Pure arithmetic on the config (no model, no device), so it is CPU-testable.
    """
    from ..gemm.dispatch import shape_class

    m = default_claim_gemm_cache_m(rt)
    if shape_class(n, k) in _MLP_TILED_SHAPE_CLASSES:
        tile = getattr(rt, "mlp_tile_tokens", 0) or 0
        if tile > 0:
            return min(m, tile)
    return m


# =========================================================================== #
# 4. the device-buffer ABI
# =========================================================================== #
@dataclass
class DeviceBuffers:
    """The fixed device buffers a decode step reads.

    ```
    input_ids   [Bmax]        int32   <- the scheduler writes, the sampler
                                         also writes (next step's input)
    positions   [Bmax]        int32
    slot_ids    [Bmax]        int32   -> SSM pool row (also the KV slot row)
    kv_indptr   [Bmax+1]      int32   -> FlashInfer paged layout
    kv_indices  [max_pages]   int32
    kv_last_page[Bmax]        int32
    seq_lens    [Bmax]        int32
    temperature/top_p/top_k/presence_penalty/repetition_penalty  [Bmax] fp32
    out_tokens  [Bmax]        int32
    logits      [Bmax, vocab] fp32
    ```

    Nothing here is ever reallocated: capture takes the pointers once and
    every later step just overwrites the contents.  The scheduler fills them
    with ``copy_`` from pinned host staging tensors (one H2D per step).
    """

    max_batch: int
    vocab_size: int
    max_pages: int
    device: torch.device

    input_ids: torch.Tensor = field(init=False)
    positions: torch.Tensor = field(init=False)
    slot_ids: torch.Tensor = field(init=False)
    kv_indptr: torch.Tensor = field(init=False)
    kv_indices: torch.Tensor = field(init=False)
    kv_last_page: torch.Tensor = field(init=False)
    seq_lens: torch.Tensor = field(init=False)
    temperature: torch.Tensor = field(init=False)
    top_p: torch.Tensor = field(init=False)
    top_k: torch.Tensor = field(init=False)
    presence_penalty: torch.Tensor = field(init=False)
    repetition_penalty: torch.Tensor = field(init=False)
    out_tokens: torch.Tensor = field(init=False)
    logits: torch.Tensor = field(init=False)

    # pinned host staging (one H2D per field per step)
    host: Dict[str, torch.Tensor] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        d = self.device
        b = self.max_batch
        i32 = dict(dtype=torch.int32, device=d)
        f32 = dict(dtype=torch.float32, device=d)
        self.input_ids = torch.zeros(b, **i32)
        self.positions = torch.zeros(b, **i32)
        self.slot_ids = torch.zeros(b, **i32)
        self.kv_indptr = torch.zeros(b + 1, **i32)
        self.kv_indices = torch.zeros(self.max_pages, **i32)
        self.kv_last_page = torch.zeros(b, **i32)
        self.seq_lens = torch.zeros(b, **i32)
        self.temperature = torch.ones(b, **f32)
        self.top_p = torch.ones(b, **f32)
        self.top_k = torch.zeros(b, **f32)  # 0 / <=0 == disabled
        self.presence_penalty = torch.zeros(b, **f32)
        self.repetition_penalty = torch.ones(b, **f32)
        self.out_tokens = torch.zeros(b, **i32)
        self.logits = torch.zeros(b, self.vocab_size, **f32)

        pin = d.type == "cuda"
        for name, t in (
            ("input_ids", self.input_ids),
            ("positions", self.positions),
            ("slot_ids", self.slot_ids),
            ("kv_indptr", self.kv_indptr),
            ("kv_indices", self.kv_indices),
            ("kv_last_page", self.kv_last_page),
            ("seq_lens", self.seq_lens),
            ("temperature", self.temperature),
            ("top_p", self.top_p),
            ("top_k", self.top_k),
            ("presence_penalty", self.presence_penalty),
            ("repetition_penalty", self.repetition_penalty),
        ):
            self.host[name] = torch.zeros(t.shape, dtype=t.dtype, device="cpu", pin_memory=pin)
        self.host["out_tokens"] = torch.zeros(b, dtype=torch.int32, device="cpu", pin_memory=pin)

    def upload(self, names: Sequence[str]) -> None:
        """Copy the named host staging tensors onto the device (non-blocking)."""
        for name in names:
            getattr(self, name).copy_(self.host[name], non_blocking=True)

    def nbytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (
                self.input_ids, self.positions, self.slot_ids, self.kv_indptr,
                self.kv_indices, self.kv_last_page, self.seq_lens, self.temperature,
                self.top_p, self.top_k, self.presence_penalty, self.repetition_penalty,
                self.out_tokens, self.logits,
            )
        )


# =========================================================================== #
# 5. attention runner (FlashInfer with a torch fallback)
# =========================================================================== #
@dataclass
class KVIndices:
    """The FlashInfer paged-layout triple, plus the slot list behind it."""

    indptr: torch.Tensor  # [B+1] int32
    indices: torch.Tensor  # [total_pages] int32
    last_page_len: torch.Tensor  # [B] int32
    slots: List[int]
    seq_lens: List[int]


def build_kv_indices(
    page_lists: Sequence[Sequence[int]],
    seq_lens: Sequence[int],
    slots: Sequence[int],
    page_size: int,
    device: torch.device,
) -> KVIndices:
    """Build the FlashInfer index triple from **host-side** page tables.

    ``PagedKVPool.build_flashinfer_indices`` does the same thing but reads the
    device-resident ``page_table``/``seq_len`` one element at a time
    (``int(self.seq_len[slot])`` per slot), which is one D2H sync per sequence
    per step.  The runtime keeps a host mirror (``scheduler.KVPageManager``)
    precisely so this stays pure Python + one H2D.
    """
    indptr = [0]
    indices: List[int] = []
    last: List[int] = []
    for pages, n in zip(page_lists, seq_lens):
        n_pages = (n + page_size - 1) // page_size
        indices.extend(int(p) for p in pages[:n_pages])
        indptr.append(indptr[-1] + n_pages)
        rem = n - (n_pages - 1) * page_size if n_pages > 0 else 0
        last.append(rem if rem > 0 else (page_size if n_pages > 0 else 0))
    return KVIndices(
        indptr=torch.tensor(indptr, dtype=torch.int32, device=device),
        indices=torch.tensor(indices or [0], dtype=torch.int32, device=device),
        last_page_len=torch.tensor(last or [0], dtype=torch.int32, device=device),
        slots=list(slots),
        seq_lens=list(seq_lens),
    )


def append_kv_graph_safe(
    pool: PagedKVPool,
    layer: int,
    slot_ids: torch.Tensor,
    positions: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    """``PagedKVPool.append_kv`` minus the host sync.

    Upstream's method is documented as "a pure scatter ... safe to call inside
    a captured CUDA graph", but it contains
    ``if bool((pages < 0).any()): raise``, a D2H sync, which makes capture
    fail and every replay a stall.  This is the same scatter with the
    (host-side, scheduler-owned) precondition check dropped: the scheduler has
    already run ``ensure_capacity``, so a negative page id here is a scheduler
    bug, not a runtime condition.
    """
    page_size = pool.cfg.page_size
    slots_l = slot_ids.long()
    pos_l = positions.long()
    pages = pool.page_table[slots_l, pos_l // page_size].long()
    offset = pos_l % page_size
    if pool.cfg.dtype == "fp8":
        k_scale = pool.scale[layer, pages, 0].unsqueeze(-1)
        v_scale = pool.scale[layer, pages, 1].unsqueeze(-1)
        from ..attn.kv_pool import FP8_MAX

        k_store = (k.float() / k_scale).clamp(-FP8_MAX, FP8_MAX).to(pool.storage_dtype)
        v_store = (v.float() / v_scale).clamp(-FP8_MAX, FP8_MAX).to(pool.storage_dtype)
    else:
        k_store = k.to(pool.storage_dtype)
        v_store = v.to(pool.storage_dtype)
    pool.kv[layer, pages, 0, offset] = k_store
    pool.kv[layer, pages, 1, offset] = v_store
    # `scatter_reduce_(amax)`, not `seq_len[slots] = maximum(seq_len[slots], ...)`:
    # the latter is a fancy *write* with potentially duplicate destination
    # indices, whose CUDA result is undefined when two rows target the same
    # slot, which a decode step always does, because every CUDA-graph
    # padding row points at `scratch_slot`. In `PagedKVPool.append_kv` the
    # same pattern silently truncated multi-token prefills to seq_len=1; here
    # the padded rows all write the same value so it would be benign, but
    # "benign undefined behaviour" is not a contract.
    new_len = (pos_l + 1).to(pool.seq_len.dtype)
    pool.seq_len.scatter_reduce_(0, slots_l, new_len, reduce="amax", include_self=True)


class AttentionRunner:
    """Owns the plan()-outside / run()-inside split for the paged attention.

    One instance per model.  ``plan_decode`` is host-side bookkeeping and must
    run **outside** graph capture/replay; ``decode`` is a pure kernel launch
    against the buffers ``plan_decode`` populated, and is what gets captured.
    """

    def __init__(
        self,
        pool: PagedKVPool,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        backend: str = "auto",
        device: torch.device,
        workspace_mb: int = 512,
        buckets: Sequence[int] = (),
        max_pages: int = 0,
        use_cuda_graph_wrappers: bool = True,
        separate_decode_workspace: bool = False,
    ):
        self.pool = pool
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scaling = head_dim ** -0.5
        self.device = device

        if backend == "auto":
            backend = "flashinfer" if (fi.HAS_FLASHINFER and device.type == "cuda") else "torch"
        if backend == "flashinfer" and not fi.HAS_FLASHINFER:
            raise RuntimeError("attn_backend='flashinfer' requested but flashinfer is not importable")
        self.backend = backend

        self._decode_wrappers: Dict[int, "fi.FlashInferDecodeAttention"] = {}
        self._dynamic_decode_wrapper: Optional["fi.FlashInferDecodeAttention"] = None
        self._current_decode_wrapper: Optional["fi.FlashInferDecodeAttention"] = None
        self._prefill_wrapper = None
        #: Which wrapper :meth:`prefill` runs against. ``plan_prefill``
        #: points it at the ordinary (non-graph) prefill wrapper; ``plan_mixed_
        #: graph`` points it at the fixed-size, ``use_cuda_graph=True`` wrapper
        #: for that step shape. It is a runner-level pointer rather than an
        #: argument to :meth:`prefill` because ``prefill`` is called once per
        #: layer from inside a captured region, where nothing may branch.
        self._active_prefill = None
        #: One graph-mode prefill wrapper per mixed-step shape, keyed by the
        #: number of *plan rows* (prefill segments + decode rows). Sized
        #: exactly, never over-sized: FlashInfer derives ``batch_size`` from
        #: the length of the persistent ``qo_indptr`` buffer, so a wrapper with
        #: room for more rows than the step has would plan over a stale tail.
        self._mixed_wrappers: Dict[int, "fi.FlashInferPrefillAttention"] = {}
        self._plan_bucket: Optional[int] = None
        # Per-layer uniform fp8 KV scales (FlashInfer's kernels take
        # ONE float per call, not a per-page/per-head tensor; see
        # PagedKVPool.calibrate_uniform_scale). None for a bf16 pool, and
        # None means "pass nothing", which is what a scale of exactly 1.0
        # would mean anyway.
        self.kv_scales: Dict[int, Tuple[float, float]] = {}
        self._buckets = tuple(buckets)
        self._max_pages = max_pages or pool.cfg.n_pages
        self._use_graph_wrappers = use_cuda_graph_wrappers
        # torch-fallback plan state
        self._slots: List[int] = []
        self._q_lens: List[int] = []
        self._kv_lens: List[int] = []

        if self.backend == "flashinfer":
            self.workspace = torch.empty(
                workspace_mb * 1024 * 1024, dtype=torch.uint8, device=device
            )
            # FlashInfer's float workspace is *per-call
            # scratch*: a prefill `run()` and a decode `run()` both write it
            # while they execute. That is fine while a step is one stream
            # (nothing else is in flight), and it is a data race the moment
            # `--overlap` puts the two halves on two streams at once. So the
            # decode wrappers get their own buffer, built here (before any
            # wrapper exists) because a wrapper's workspace is fixed at
            # construction and `GraphedDecoder.capture` bakes the buffers it
            # finds into every replay.
            self.decode_workspace = (
                torch.empty(workspace_mb * 1024 * 1024, dtype=torch.uint8, device=device)
                if separate_decode_workspace
                else self.workspace
            )
            # `buckets` is only remembered for introspection: decode
            # wrappers are built lazily (`_get_decode_wrapper`), one per
            # distinct bucket actually requested, the first time it is
            # requested. Building all ~13 persistent (`use_cuda_graph=True`)
            # wrappers eagerly would waste time and memory on buckets a given
            # run never touches, and would tie `plan_decode` to whatever
            # bucket list was passed in here, which is *always empty* when
            # `use_cuda_graph_wrappers` is False (see below): non-graphed
            # decode must not need a bucket-keyed wrapper at all.
            self._prefill_wrapper = fi.FlashInferPrefillAttention(
                self.workspace,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                page_size=pool.cfg.page_size,
                kv_dtype=pool.storage_dtype,
            )
            self._active_prefill = self._prefill_wrapper

    # -- decode --------------------------------------------------------- #
    def _get_decode_wrapper(self, bucket: int) -> "fi.FlashInferDecodeAttention":
        """Build (once) and cache the FlashInfer decode wrapper for ``bucket``.

        Two regimes, matching FlashInfer's own `use_cuda_graph` contract:

        * **graphed** (``self._use_graph_wrappers``): one persistent,
          fixed-size wrapper per distinct bucket, built the first time that
          bucket is planned -- so a graph replay always finds the same
          object/buffers it captured against. Cached in
          ``self._decode_wrappers`` for the life of the model.
        * **non-graphed** (``--no-graphs`` / ``RuntimeConfig.use_cuda_graphs
          =False``): a *single* ``use_cuda_graph=False`` wrapper, reused for
          every call regardless of batch size. FlashInfer's non-graph
          wrappers take raw, variably-sized ``indptr``/``indices``/
          ``last_page_len`` tensors straight into `plan()` every call --
          there is no fixed-size persistent buffer to key by bucket, so
          "bucket" is purely a `DeviceBuffers` padding concept upstream of
          this class, not something the attention wrapper needs to know.
        """
        if not self._use_graph_wrappers:
            if self._dynamic_decode_wrapper is None:
                self._dynamic_decode_wrapper = fi.FlashInferDecodeAttention(
                    self.decode_workspace,
                    num_qo_heads=self.num_qo_heads,
                    num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim,
                    page_size=self.pool.cfg.page_size,
                    max_batch_size=self.pool.cfg.max_seqs,
                    max_pages=self._max_pages,
                    kv_dtype=self.pool.storage_dtype,
                    q_dtype=None,
                    use_cuda_graph=False,
                    device=str(self.device),
                )
            return self._dynamic_decode_wrapper
        w = self._decode_wrappers.get(bucket)
        if w is None:
            w = fi.FlashInferDecodeAttention(
                self.decode_workspace,
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                page_size=self.pool.cfg.page_size,
                max_batch_size=bucket,
                max_pages=self._max_pages,
                kv_dtype=self.pool.storage_dtype,
                q_dtype=None,
                use_cuda_graph=True,
                device=str(self.device),
            )
            self._decode_wrappers[bucket] = w
        return w

    def plan_decode(
        self, slots: Sequence[int], bucket: int, seq_lens: Optional[Sequence[int]] = None
    ) -> None:
        """Host-side.  ``slots`` is already padded to ``bucket`` (padded rows
        point at the scratch slot, which owns one page and has seq_len 1).

        ``seq_lens`` (optional): this batch's committed context lengths, in
        the same order as ``slots``.  Pass them when the caller already knows
        them (the scheduler always does) and ``build_flashinfer_indices``
        needs **zero** device reads; omit them and it costs exactly one
        vectorised D2H for ``seq_len[slots]``.  Either way it is O(1) syncs,
        not an O(B x pages_per_seq) scalar-read storm; see
        ``PagedKVPool.build_flashinfer_indices``.
        """
        self._plan_bucket = bucket
        self._slots = list(slots)
        if self.backend != "flashinfer":
            return
        if self._use_graph_wrappers and len(self._slots) != bucket:
            # See attn/flashinfer_attn.py, FlashInferDecodeAttention.plan()
            # docstring: plan()'s `indptr_buf[: kv_indptr.numel()].copy_(...)`
            # slice only prevents a shape-mismatch crash when the incoming array is
            # shorter than the persistent buffer -- it does NOT make an under-length
            # (non-bucket-padded) call correct, because the buffer's un-overwritten
            # tail then stays at its zero-initialized default, which is a
            # non-monotonic indptr. This runtime always builds `slots` bucket-padded
            # before calling here (graphs.py, scheduler.py); this assertion exists so
            # a future caller that forgets to pad fails loudly instead of silently
            # feeding FlashInfer a malformed indptr.
            raise ValueError(
                f"plan_decode: len(slots)={len(self._slots)} != bucket={bucket} "
                "-- slots must be padded to the full bucket size before calling "
                "plan_decode in graphed mode (see FlashInferDecodeAttention.plan()'s "
                "docstring on why an unpadded indptr is silently wrong, not just slow)"
            )
        # ``staged=True`` builds the plan arrays in a pinned
        # ring and returns ``indptr``/``last_page`` on the **host** (which is
        # where FlashInfer's ``plan()`` wants them), so the decode plan costs
        # zero ``cudaStreamSynchronize``.  It degrades to plain device
        # tensors on CPU builds, and ``FlashInferDecodeAttention.plan`` only
        # takes its fast path on an exact shape match, so the non-graphed
        # wrapper (whose fixed batch size is ``max_seqs``, not ``bucket``) is
        # unaffected.
        indptr, indices, last_page, _ = self.pool.build_flashinfer_indices(
            self._slots, seq_lens=seq_lens, staged=self._use_graph_wrappers
        )
        self._current_decode_wrapper = self._get_decode_wrapper(bucket)
        self._current_decode_wrapper.plan(indptr, indices, last_page)

    def decode(self, kv_layer: int, q: torch.Tensor, slot_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``q``: ``[B, num_qo_heads, head_dim]`` -> same shape.

        ``slot_ids``: the *device* buffer slice (``ctx.slot_ids`` == ``buf
        .slot_ids[:batch]``, written by the scheduler before replay)
        the torch-fallback path indexes with. Callers must pass this instead
        of relying on ``self._slots`` (the host Python list ``plan_decode``
        stashed): rebuilding a device tensor from that list on every call
        fails capture with "Cannot copy between CPU and CUDA tensors during
        CUDA graph capture unless the CPU tensor is pinned". Falls back to
        the capture-*unsafe* re-materialisation only if a caller genuinely has
        no device tensor handy (e.g. ad-hoc/test call sites), so eager
        (non-captured) use is unaffected.
        """
        if self.backend == "flashinfer":
            k_scale, v_scale = self._scales_for(kv_layer)
            return self._current_decode_wrapper.run(
                q, self.pool.kv[kv_layer], k_scale=k_scale, v_scale=v_scale
            )
        if slot_ids is None:
            slot_ids = torch.tensor(self._slots, dtype=torch.int32, device=q.device)
        return fi.torch_fallback_decode(self.pool, kv_layer, slot_ids, q, self.scaling)

    # -- prefill --------------------------------------------------------- #
    def plan_prefill(self, slots: Sequence[int], q_lens: Sequence[int], kv_lens: Sequence[int]) -> None:
        """Host-side prefill plan.  ``kv_lens[i]`` is the committed context
        length of sequence ``i`` **after** this chunk is written (i.e.
        ``start_pos + q_len``), which is exactly what ``make_prefill_batch``
        computes.

        ``kv_lens`` **must** be forwarded to ``build_flashinfer_indices``.
        Letting it default to ``seq_lens=None`` reads the *device*
        ``PagedKVPool.seq_len``, and the only writer that advances that is
        ``append_kv`` -- which, on this path, runs **after** planning:
        ``prefill_forward`` calls ``plan_prefill`` once up front and then
        each ``FusedAttention.prefill`` appends its layer's K/V inside the
        layer loop.  So the device counter still holds the length of the
        *previous* chunk at plan time (zero for a fresh sequence, since
        ``reset_slot``/``free_pages`` zero it), and FlashInfer would be
        planned over a KV range that excludes every token this chunk is
        about to write: a fresh 2000-token prompt would attend to nothing
        at all, and a continuation chunk would attend only to its prefix.
        The hazard is GPU-only: the torch fallback (``_torch_paged_prefill``)
        reads ``self._kv_lens`` directly and is correct either way, so the
        CPU suite cannot catch it.
        """
        self._plan_prefill_on(self._prefill_wrapper, slots, q_lens, kv_lens)

    def _plan_prefill_on(self, wrapper, slots, q_lens, kv_lens) -> None:
        """The body of :meth:`plan_prefill`, against a chosen wrapper.

        A graphed mixed step plans the *same* varlen shape against a
        fixed-size ``use_cuda_graph=True`` wrapper instead, so that the
        per-layer ``run()`` the graph captured keeps finding the same
        persistent index buffers. Everything else (including the ``kv_lens``
        contract this method's caller documents at length) is identical, so
        it is one body and two entry points.
        """
        self._slots = list(slots)
        self._q_lens = list(q_lens)
        self._kv_lens = list(kv_lens)
        if self.backend != "flashinfer":
            self._active_prefill = wrapper
            return
        qo = [0]
        for n in self._q_lens:
            qo.append(qo[-1] + n)
        indptr, indices, last_page, _ = self.pool.build_flashinfer_indices(
            self._slots, seq_lens=self._kv_lens
        )
        wrapper.plan(
            torch.tensor(qo, dtype=torch.int32, device=self.device),
            indptr,
            indices,
            last_page,
            causal=True,
        )
        self._active_prefill = wrapper

    # -- mixed step, CUDA-graphed ----------------------------------------- #
    def mixed_graph_wrapper(self, n_plan_rows: int):
        """The persistent, graph-mode prefill wrapper for a step with exactly
        ``n_plan_rows`` varlen rows (prefill segments + decode rows), built on
        first use and cached for the life of the model.

        ``max_batch_size=n_plan_rows`` **exactly**: FlashInfer's graph mode
        plans over the whole persistent ``qo_indptr`` buffer, so an
        over-sized wrapper would read a stale, non-monotonic tail (the failure
        ``pad_indptr_to_bucket``'s docstring records, in a new shape). The
        callers therefore pad the *step* to a fixed row count instead; see
        ``mixed_graphs.pad_mixed_step``, which also guarantees every row is
        non-empty.
        """
        if self.backend != "flashinfer":
            return None
        w = self._mixed_wrappers.get(n_plan_rows)
        if w is None:
            w = fi.FlashInferPrefillAttention(
                self.workspace,
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                page_size=self.pool.cfg.page_size,
                kv_dtype=self.pool.storage_dtype,
                q_dtype=None,
                use_cuda_graph=True,
                max_batch_size=n_plan_rows,
                max_pages=self._max_pages,
                device=str(self.device),
            )
            self._mixed_wrappers[n_plan_rows] = w
        return w

    def plan_mixed_graph(self, slots: Sequence[int], q_lens: Sequence[int],
                         kv_lens: Sequence[int]) -> None:
        """:meth:`plan_prefill` against this shape's graph-mode wrapper.

        Host-side, and must run **outside** the graph on every step, exactly
        like ``plan_decode`` does for the decode buckets.
        """
        n = len(list(q_lens))
        w = self.mixed_graph_wrapper(n)
        self._plan_prefill_on(w if w is not None else self._prefill_wrapper,
                              slots, q_lens, kv_lens)

    def _scales_for(self, kv_layer: int) -> Tuple[Optional[float], Optional[float]]:
        """``(k_scale, v_scale)`` to hand FlashInfer for this layer.

        ``(None, None)`` for a bf16 pool (FlashInfer ignores the argument)
        and for an uncalibrated fp8 pool, whose scale tensor is still all
        1.0 -- passing 1.0 and passing nothing are the same computation.
        Populated by :meth:`set_kv_scale`, which the fp8-KV calibration path
        calls once per layer.
        """
        if self.pool.cfg.dtype != "fp8":
            return (None, None)
        return self.kv_scales.get(kv_layer, (None, None))

    def set_kv_scale(self, kv_layer: int, k_scale: float, v_scale: float) -> None:
        """Record the uniform fp8 scales for ``kv_layer`` (see
        ``PagedKVPool.calibrate_uniform_scale``, whose return value this
        takes). Host-side bookkeeping; the floats are baked into the kernel
        launch, so this must be called **before** graph capture."""
        self.kv_scales[kv_layer] = (float(k_scale), float(v_scale))

    def prefill(self, kv_layer: int, q: torch.Tensor) -> torch.Tensor:
        """``q``: ``[T, num_qo_heads, head_dim]`` packed varlen -> same shape.

        Runs against whichever wrapper the last ``plan_*`` selected
        (``_active_prefill``): the ordinary one for a prefill chunk or an eager
        mixed step, this shape's graph-mode one for a captured mixed step."""
        if self.backend == "flashinfer":
            k_scale, v_scale = self._scales_for(kv_layer)
            w = self._active_prefill or self._prefill_wrapper
            return w.run(q, self.pool.kv[kv_layer], k_scale=k_scale, v_scale=v_scale)
        return self._torch_paged_prefill(kv_layer, q)

    def _torch_paged_prefill(self, kv_layer: int, q: torch.Tensor) -> torch.Tensor:
        """Chunk-aware causal SDPA over the pages, for the CPU/reference path.

        Handles a chunk that *continues* an already-prefilled prefix: the
        query rows are the chunk's ``q_len`` tokens but the keys are the whole
        ``kv_len`` prefix, so the causal mask is bottom-right aligned.
        """
        outs = []
        off = 0
        groups = self.num_qo_heads // self.num_kv_heads
        for slot, q_len, kv_len in zip(self._slots, self._q_lens, self._kv_lens):
            if q_len == 0:
                continue
            k, v = self.pool.gather_dense(kv_layer, slot, kv_len)
            k = k.transpose(0, 1)  # [Hkv, kv_len, D]
            v = v.transpose(0, 1)
            if groups > 1:
                k = k.repeat_interleave(groups, dim=0)
                v = v.repeat_interleave(groups, dim=0)
            qi = q[off : off + q_len].transpose(0, 1)  # [Hq, q_len, D]
            q_pos = torch.arange(kv_len - q_len, kv_len, device=q.device)[:, None]
            k_pos = torch.arange(kv_len, device=q.device)[None, :]
            mask = (k_pos <= q_pos)[None]
            oi = F.scaled_dot_product_attention(
                qi.float(), k.float(), v.float(), attn_mask=mask, scale=self.scaling
            )
            outs.append(oi.transpose(0, 1).to(q.dtype))
            off += q_len
        return torch.cat(outs, dim=0)


# =========================================================================== #
# 6. layer modules
# =========================================================================== #
class FusedMLP:
    """SwiGLU over the fused ``gate_up_proj``, tiled for prefill."""

    def __init__(self, w: MLPFusedWeights, rt: RuntimeConfig, sm: Optional[int]):
        self.gate_up = ResolvedLinear(w.gate_up_proj, sm, rt.gemm_backend)
        self.down = ResolvedLinear(w.down_proj, sm, rt.gemm_backend)
        self.tile = rt.mlp_tile_tokens
        self.fused_ops_backend = rt.fused_ops_backend

    def _act(self, gate_up: torch.Tensor) -> torch.Tensor:
        # `swiglu` (chunk + silu + mul) is 2 eager launches per
        # call site x 64 MLP layers ~= 1.3 ms/step at B=128 (profiled).
        if self.fused_ops_backend == "triton":
            return fused_swiglu.triton_swiglu(gate_up)
        return swiglu(gate_up)

    def __call__(self, x: torch.Tensor, tiled: bool = False) -> torch.Tensor:
        if not tiled or x.shape[0] <= self.tile:
            return self.down(self._act(self.gate_up(x)))
        outs = []
        for i in range(0, x.shape[0], self.tile):
            xi = x[i : i + self.tile]
            outs.append(self.down(self._act(self.gate_up(xi))))
        return torch.cat(outs, dim=0)

    def nbytes(self) -> int:
        return self.gate_up.nbytes() + self.down.nbytes()


class FusedGDN:
    """One Gated-DeltaNet layer over the pool-indexed GDN kernels."""

    def __init__(
        self,
        w: GDNFusedWeights,
        cfg: QwenFastConfig,
        rt: RuntimeConfig,
        gdn_ordinal: int,
        sm: Optional[int],
    ):
        self.cfg = cfg
        self.rt = rt
        self.ordinal = gdn_ordinal  # index into the 48-layer state pool
        self.in_proj_qkvz = ResolvedLinear(w.in_proj_qkvz, sm, rt.gemm_backend)
        self.in_proj_ba = ResolvedLinear(w.in_proj_ba, sm, rt.gemm_backend)
        self.out_proj = ResolvedLinear(w.out_proj, sm, rt.gemm_backend)
        _conv_w = w.conv1d_weight.squeeze(1) if w.conv1d_weight.dim() == 3 else w.conv1d_weight
        # kernels_gdn/state.py::prepare_conv_weight: a width-major-strided
        # [C, W] weight makes the triton conv kernel's per-w-index read a
        # contiguous run over channels instead of a stride-W gather
        # (3.94us vs 8.95us at B=64).
        # Value-identical (same logical tensor), so this is free to always do.
        self.conv_w = gdn_state.prepare_conv_weight(_conv_w)
        self.A_log = w.A_log
        self.dt_bias = w.dt_bias
        # `g = -exp(A_log) * softplus(a + dt_bias)` in fp32.  `-exp(A_log)`
        # and `dt_bias.float()` are *constants*; computing them inline cost 3
        # kernel launches per layer per step (48 layers => ~144 launches/step)
        # to reproduce the same 48-element vector every time.
        self.neg_exp_A = (-w.A_log.float().exp()).contiguous()
        self.dt_bias_f = w.dt_bias.float().contiguous()
        self.norm_weight = w.norm_weight
        self.norm_backend = rt.norm_backend
        self.conv_prefill_layout = rt.conv_prefill_layout
        self.key_dim = cfg.key_dim
        self.value_dim = cfg.value_dim
        self.conv_dim = cfg.conv_dim
        self.num_k_heads = cfg.linear_num_key_heads
        self.num_v_heads = cfg.linear_num_value_heads
        self.head_k = cfg.linear_key_head_dim
        self.head_v = cfg.linear_value_head_dim
        self.eps = cfg.rms_norm_eps

    # -- shared projection half ------------------------------------------ #
    def _project(self, h: torch.Tensor):
        """``h``: ``[T, hidden]`` -> ``(mixed_qkv [T, C], z, g, beta)``."""
        qkvz = self.in_proj_qkvz(h)
        mixed_qkv, z = qkvz.split([self.conv_dim, self.value_dim], dim=-1)
        ba = self.in_proj_ba(h)
        if self.rt.fused_ops_backend == "triton":
            # split(view) + sigmoid + add + softplus + mul, 4
            # eager launches per GDN layer per step -> 1 (48 layers, ~192
            # launches in the profiled elementwise/other group).
            beta, g = fused_gdn_gate.triton_gdn_gate(ba, self.neg_exp_A, self.dt_bias_f)
        else:
            b, a = ba.split([self.num_v_heads, self.num_v_heads], dim=-1)
            beta = b.sigmoid()
            g = self.neg_exp_A * F.softplus(a.float() + self.dt_bias_f)
        return mixed_qkv, z, g, beta

    def _gated_norm(self, core: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if self.norm_backend == "triton":
            return triton_rms_norm_gated(core, z, self.norm_weight, self.eps)
        return rms_norm_gated(core, z, self.norm_weight, self.eps)

    def _split_qkv(self, mixed_qkv: torch.Tensor, t: int):
        q, k, v = mixed_qkv.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(t, self.num_k_heads, self.head_k)
        k = k.reshape(t, self.num_k_heads, self.head_k)
        v = v.reshape(t, self.num_v_heads, self.head_v)
        return q, k, v

    # -- decode ------------------------------------------------------------ #
    def decode(self, h: torch.Tensor, ctx: "StepContext") -> torch.Tensor:
        b = h.shape[0]
        mixed_qkv, z, g, beta = self._project(h)
        conv_pool = gdn_state.layer_conv_state(ctx.conv_pool, self.ordinal)
        mixed_qkv = gdn_api.causal_conv_update(
            mixed_qkv, conv_pool, ctx.slot_ids, self.conv_w,
            activation="silu", backend=self.rt.gdn_backend,
        )
        q, k, v = self._split_qkv(mixed_qkv, b)
        state_pool = gdn_state.layer_state(ctx.state_pool, self.ordinal)
        core = gdn_api.gdn_decode_step(
            q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1),
            g.unsqueeze(1), beta.unsqueeze(1),
            state_pool, ctx.slot_ids, backend=self.rt.gdn_backend,
        )  # [B, 1, HV, V]
        core = core.reshape(b * self.num_v_heads, self.head_v)
        core = self._gated_norm(core, z.reshape(b * self.num_v_heads, self.head_v))
        return self.out_proj(core.reshape(b, self.value_dim))

    # -- prefill (packed varlen) -------------------------------------------- #
    def _conv_prefill(self, mixed_qkv: torch.Tensor, conv_pool: torch.Tensor,
                      ctx: "StepContext") -> torch.Tensor:
        """The packed-varlen conv over ``[T_pre, C]``; ring state committed.

        Split out of :meth:`prefill` so :meth:`mixed` runs *exactly* this on
        its prefill rows: the mixed step's whole correctness claim is
        "the segment rows see the same kernels they would have seen in a
        prefill step", and a copy-paste would be one edit away from being
        false.
        """
        if self.conv_prefill_layout == "token_major" and ctx.q_lens is not None:
            # `mixed_qkv` is already `[T, C]`; the varlen
            # entry point consumes that layout directly, so the two
            # `[C=10240, T]` transposes below (336 MB of copy each way per GDN
            # layer at T=8192) are avoided, and with them the per-layer
            # `cu_seqlens.to("cpu")` D2H sync inside the torch conv.
            mixed_qkv = gdn_api.causal_conv_prefill_varlen(
                mixed_qkv,
                self.conv_w,
                seq_lens=ctx.q_lens,
                cu_seqlens=ctx.cu_seqlens,
                conv_state_pool=conv_pool,
                slot_ids=ctx.seq_slot_ids,
                activation="silu",
                backend=self.rt.gdn_backend,
                # `None` everywhere except a captured mixed step, where
                # it pins the token-axis grid to the padded chunk's cap.
                max_seqlen=ctx.conv_max_seqlen,
            )
        else:
            x = mixed_qkv.transpose(0, 1).unsqueeze(0).contiguous()  # [1, C, T]
            x = gdn_api.causal_conv_prefill(
                x, self.conv_w,
                cu_seqlens=ctx.cu_seqlens,
                conv_state_pool=conv_pool,
                slot_ids=ctx.seq_slot_ids,
                activation="silu",
                backend=self.rt.gdn_backend,
                tile_tokens=self.rt.conv_prefill_tile_tokens,
            )
            mixed_qkv = x.squeeze(0).transpose(0, 1).contiguous()  # [T, C]
        return mixed_qkv

    def _gdn_prefill_core(self, post_conv: torch.Tensor, g: torch.Tensor,
                          beta: torch.Tensor, ctx: "StepContext") -> torch.Tensor:
        """The chunked (fla) gated delta rule over the varlen segments.

        ``post_conv``: ``[T_pre, C]`` -> ``[T_pre, HV, V]``; the per-sequence
        final states are scattered back into ``state_pool`` by the kernel.
        Split out for the same reason as :meth:`_conv_prefill`.
        """
        t = post_conv.shape[0]
        q, k, v = self._split_qkv(post_conv, t)
        state_pool = gdn_state.layer_state(ctx.state_pool, self.ordinal)
        core, _ = gdn_api.gdn_prefill_chunked(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
            g.unsqueeze(0), beta.unsqueeze(0),
            cu_seqlens=ctx.cu_seqlens,
            state_pool=state_pool,
            slot_ids=ctx.seq_slot_ids,
            output_final_state=False,
            backend=self.rt.gdn_backend,
            chunk_size=self.rt.gdn_chunk_size,
            cu_seqlens_cpu=ctx.cu_seqlens_cpu,
            # Set only on a captured mixed step; then fla
            # builds no index tensor, does no host work and no H2D, and this
            # whole call captures. `cu_seqlens_cpu` is redundant in that case
            # (nothing reads it) and is passed anyway so the two paths differ
            # in exactly one thing.
            chunk_indices=ctx.chunk_indices,
            chunk_offsets=ctx.chunk_offsets,
        )
        return core

    def prefill(self, h: torch.Tensor, ctx: "StepContext") -> torch.Tensor:
        t = h.shape[0]
        mixed_qkv, z, g, beta = self._project(h)
        conv_pool = gdn_state.layer_conv_state(ctx.conv_pool, self.ordinal)
        mixed_qkv = self._conv_prefill(mixed_qkv, conv_pool, ctx)
        core = self._gdn_prefill_core(mixed_qkv, g, beta, ctx)
        core = core.reshape(t * self.num_v_heads, self.head_v)
        core = self._gated_norm(core, z.reshape(t * self.num_v_heads, self.head_v))
        return self.out_proj(core.reshape(t, self.value_dim))

    def _varlen_half(self, mixed_qkv: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
                     conv_pool: torch.Tensor, ctx: "StepContext", tp: int) -> torch.Tensor:
        """The prefill half of a mixed step's GDN: conv then fla chunk.

        Split out of :meth:`mixed` as one callable because it is exactly the
        region the holed graphed mixed step cannot capture (``StepContext
        .graph_break``'s docstring says why), and the eager and graphed paths
        must run *the same* code there or the parity tests are testing two
        different functions.
        """
        pre = self._conv_prefill(mixed_qkv[:tp], conv_pool, ctx)
        return self._gdn_prefill_core(pre, g[:tp], beta[:tp], ctx)

    # -- mixed prefill + decode ---------------------------------------------- #
    def mixed(self, h: torch.Tensor, ctx: "StepContext") -> torch.Tensor:
        """``h``: ``[T_pre + B_dec, hidden]`` -> same rows, one GEMM each.

        The three GEMMs (``in_proj_qkvz``, ``in_proj_ba``, ``out_proj``), the
        gate epilogue and the gated RMSNorm run **once** over every row, at
        ``M = T_pre + B_dec``; only the two sequence-mixing kernels dispatch by
        row type, because only they carry per-sequence recurrent state:

        * conv -> :meth:`_conv_prefill` on the segment rows (varlen, ring
          committed per sequence) and ``causal_conv_update`` on the decode rows
          (one ring shift each) -- the same two kernels :meth:`prefill` and
          :meth:`decode` call, on disjoint slots;
        * GDN -> :meth:`_gdn_prefill_core` (fla chunk kernel) on the segment
          rows and ``gdn_decode_step`` (the Triton pool-indexed kernel) on
          the decode rows. Both read *and* write ``state_pool``; because a
          request is either prefilling or decoding and never both, the two
          calls touch disjoint slot sets and the order between them does not
          matter.

        The two halves' outputs are concatenated back into one ``[T, HV, V]``
        before the gated norm, so the residual stream stays a single tensor and
        the MLP that follows also runs once at ``M = T``. That concatenation is
        the only copy this method adds over the two separate steps.
        """
        t = h.shape[0]
        tp = int(ctx.n_prefill_tokens)
        bd = t - tp
        mixed_qkv, z, g, beta = self._project(h)
        conv_pool = gdn_state.layer_conv_state(ctx.conv_pool, self.ordinal)
        state_pool = gdn_state.layer_state(ctx.state_pool, self.ordinal)

        if ctx.graph_break is None or ctx.chunk_indices is not None:
            # With static index tensors and a pinned conv grid there is
            # nothing left in this half that reads the segmentation on the
            # host, so it captures with everything else and the eager hole is
            # not opened even if a hook is installed. That `or` is what makes
            # "holes" vs "no holes" one flag rather than two code paths.
            core_pre = self._varlen_half(mixed_qkv, g, beta, conv_pool, ctx, tp)
        else:
            # The one hole in the holed captured mixed step.
            # The closure is re-run eagerly on every replay, against the same
            # (graph-pool, therefore address-stable) `mixed_qkv`/`g`/`beta` and
            # the runner's live `ctx`; its output is copied into a static
            # buffer the next captured segment reads.
            core_pre = ctx.graph_break(
                lambda: self._varlen_half(mixed_qkv, g, beta, conv_pool, ctx, tp)
            )

        dec = gdn_api.causal_conv_update(
            mixed_qkv[tp:], conv_pool, ctx.decode_slot_ids, self.conv_w,
            activation="silu", backend=self.rt.gdn_backend,
        )
        qd, kd, vd = self._split_qkv(dec, bd)
        core_dec = gdn_api.gdn_decode_step(
            qd.unsqueeze(1), kd.unsqueeze(1), vd.unsqueeze(1),
            g[tp:].unsqueeze(1), beta[tp:].unsqueeze(1),
            state_pool, ctx.decode_slot_ids, backend=self.rt.gdn_backend,
        )  # [B_dec, 1, HV, V]

        hv, hd = self.num_v_heads, self.head_v
        core = torch.cat(
            [core_pre.reshape(tp * hv, hd), core_dec.reshape(bd * hv, hd)], dim=0
        )
        core = self._gated_norm(core, z.reshape(t * hv, hd))
        return self.out_proj(core.reshape(t, self.value_dim))

    # -- speculative verify window ------------------------------------------ #
    def window(self, h: torch.Tensor, ctx: "StepContext", zero_m: torch.Tensor):
        """Phase A of a speculative step: ``n = k+1`` window tokens/sequence.

        ``h``: ``[B*n, hidden]``, packed token-major per sequence
        (``b0t0 .. b0t_{n-1}, b1t0 ...``) exactly like :meth:`prefill`'s varlen
        packing, except every sequence contributes the *same* ``n`` tokens so
        the shape is fixed and the step stays CUDA-graph capturable.
        ``ctx.seq_slot_ids`` is the ``[B]`` per-sequence slot vector.

        Runs the conv and the GDN recurrence over the whole window **without
        committing either piece of state**:

        * GDN -> :func:`kernels_gdn.api.gdn_verify` (the window kernel with
          ``commit=False``): 1 state read, 0 writes.
        * conv -> :func:`kernels_gdn.api.causal_conv_verify_and_commit` with
          ``m = 0``, for which its ring update ``concat(state, x)[..., m:m+W-1]``
          is exactly the identity, so it degenerates to a pure "conv over the
          window" with the ring left alone.

        The accepted prefix length ``m`` is not known until the *last* layer has
        produced logits, so the commit has to be a second pass over the cached
        per-position inputs: :meth:`commit_window`, fed the ``cache`` returned
        here.  That is the two-pass shape (2 state reads + 1 write =
        1.5x plain-decode SSM traffic); the fused 1-read/1-write form
        needs ``m`` *before* the forward runs, which a chain verifier structurally
        cannot provide (see :func:`kernels_gdn.api.gdn_verify_and_commit`'s own
        note on the two orderings).

        Returns ``(out [B*n, hidden], cache)``.
        """
        bn = h.shape[0]
        b = ctx.seq_slot_ids.shape[0]
        n = bn // b
        mixed_qkv, z, g, beta = self._project(h)
        conv_pool = gdn_state.layer_conv_state(ctx.conv_pool, self.ordinal)
        x = mixed_qkv.view(b, n, self.conv_dim).transpose(1, 2).contiguous()  # [B, C, n]
        y = gdn_api.causal_conv_verify_and_commit(
            x, conv_pool, ctx.seq_slot_ids, self.conv_w, zero_m,
            activation="silu", backend=self.rt.gdn_backend,
        )
        post = y.transpose(1, 2).reshape(bn, self.conv_dim)
        q, k, v = self._split_qkv(post, bn)
        q = q.view(b, n, self.num_k_heads, self.head_k)
        k = k.view(b, n, self.num_k_heads, self.head_k)
        v = v.view(b, n, self.num_v_heads, self.head_v)
        gw = g.view(b, n, self.num_v_heads)
        bw = beta.view(b, n, self.num_v_heads)
        state_pool = gdn_state.layer_state(ctx.state_pool, self.ordinal)
        core = gdn_api.gdn_verify(
            q, k, v, gw, bw, state_pool, ctx.seq_slot_ids, backend=self.rt.gdn_backend,
        )  # [B, n, HV, V], no state write
        core = core.reshape(bn * self.num_v_heads, self.head_v)
        core = self._gated_norm(core, z.reshape(bn * self.num_v_heads, self.head_v))
        out = self.out_proj(core.reshape(bn, self.value_dim))
        return out, (x, k, v, gw, bw)

    def commit_window(self, ctx: "StepContext", cache, m: torch.Tensor) -> None:
        """Phase B: advance conv ring + SSM state through the accepted prefix.

        ``m`` is a ``[B]`` **device** int32 tensor, so this is graph-safe:
        both kernels mask on it internally (``MASK_PAST_M`` on the Triton side,
        a ``g``/``beta`` zeroing on the torch side -- with ``beta = 0`` the delta
        update vanishes and with ``g = 0`` the decay is 1, so the recurrence is
        the identity past ``m``).
        """
        x, k, v, gw, bw = cache
        conv_pool = gdn_state.layer_conv_state(ctx.conv_pool, self.ordinal)
        gdn_api.causal_conv_verify_and_commit(
            x, conv_pool, ctx.seq_slot_ids, self.conv_w, m,
            activation="silu", backend=self.rt.gdn_backend,
        )
        state_pool = gdn_state.layer_state(ctx.state_pool, self.ordinal)
        # `k` is deliberately passed where `q` is expected. `gdn_commit`'s `q`
        # only feeds the per-position *outputs*, which phase B discards -- the
        # state write is a function of k/v/g/beta alone. Reusing k saves caching
        # a second [B, n, 16, 128] tensor per layer (4 KiB/token/layer, i.e.
        # ~200 MB at B=128/k=3 across 48 layers) for a value nothing reads.
        gdn_api.gdn_commit(
            k, k, v, gw, bw, state_pool, ctx.seq_slot_ids, m,
            backend=self.rt.gdn_backend,
        )

    def nbytes(self) -> int:
        return (
            self.in_proj_qkvz.nbytes() + self.in_proj_ba.nbytes() + self.out_proj.nbytes()
            + self.conv_w.numel() * self.conv_w.element_size()
        )


class FusedAttention:
    """One gated-GQA layer over the paged KV pool.

    ``q_proj`` carries the output gate: the fused ``qkv_proj`` is
    ``[q(24*256*2) ; k(4*256) ; v(4*256)]`` and the q half splits head-major
    into ``(q, gate)``.
    """

    def __init__(
        self,
        w: AttnFusedWeights,
        cfg: QwenFastConfig,
        rt: RuntimeConfig,
        kv_layer: int,
        sm: Optional[int],
    ):
        self.cfg = cfg
        self.kv_layer = kv_layer
        self.qkv = ResolvedLinear(w.qkv_proj, sm, rt.gemm_backend)
        self.o_proj = ResolvedLinear(w.o_proj, sm, rt.gemm_backend)
        self.q_norm = w.q_norm
        self.k_norm = w.k_norm
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.gate_mult = 2 if cfg.attn_output_gate else 1
        self.q_size = self.num_heads * self.head_dim * self.gate_mult
        self.kv_size = self.num_kv_heads * self.head_dim
        self.eps = cfg.rms_norm_eps

    def _qkv(self, h: torch.Tensor, cos, sin, *, use_fused_rope: bool = False):
        t = h.shape[0]
        fused = self.qkv(h)
        qg, k, v = fused.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.gate_mult == 2:
            q, gate = fi.split_q_gate(qg, self.num_heads, self.head_dim)
        else:
            q, gate = qg.view(t, self.num_heads, self.head_dim), None
        k = k.view(t, self.num_kv_heads, self.head_dim)
        v = v.view(t, self.num_kv_heads, self.head_dim)
        # `use_fused_rope=True` (prefill/mixed only, see
        # `prefill()` below) routes through the Triton kernel in
        # `attn/fused_qk_rope.py` that fuses this norm+RoPE into one launch
        # per tensor; `use_fused_rope=False` (decode, window; the default)
        # is byte-for-byte the eager call.
        q, k = fused_qk_rope.qk_norm_rope(
            q, k, self.q_norm, self.k_norm, cos, sin, self.eps, use_fused=use_fused_rope
        )
        return q, k, v, gate

    def _finish(self, out: torch.Tensor, gate, t: int) -> torch.Tensor:
        out = out.reshape(t, self.num_heads * self.head_dim)
        if gate is not None:
            out = fi.apply_output_gate(out, gate)
        return self.o_proj(out)

    def decode(self, h: torch.Tensor, ctx: "StepContext") -> torch.Tensor:
        t = h.shape[0]
        q, k, v, gate = self._qkv(h, ctx.cos, ctx.sin)
        if ctx.kv_calibrator is not None:  # fp8-KV calibration
            ctx.kv_calibrator(self.kv_layer, k, v)
        # graph-safe: no `bool(tensor)` host sync.
        # Prefill's KV append below is the non-graphed path and keeps
        # calling the pool's own (checked) `append_kv`.
        append_kv_graph_safe(ctx.pool, self.kv_layer, ctx.slot_ids, ctx.positions, k, v)
        out = ctx.attn.decode(self.kv_layer, q.contiguous(), ctx.slot_ids)
        return self._finish(out, gate, t)

    def prefill(self, h: torch.Tensor, ctx: "StepContext") -> torch.Tensor:
        t = h.shape[0]
        # use_fused_rope=True: prefill (and mixed, which calls this method)
        # is never CUDA-graph-captured, so the Triton
        # norm+RoPE kernel is a straight win here with none of a graphed
        # decode step's replay-safety questions.
        q, k, v, gate = self._qkv(h, ctx.cos, ctx.sin, use_fused_rope=True)
        if ctx.kv_calibrator is not None:  # fp8-KV calibration
            ctx.kv_calibrator(self.kv_layer, k, v)
        if ctx.graph_safe_kv:
            # Inside a captured mixed step. Same scatter, minus the
            # `bool((pages < 0).any())` D2H the checked path does; see
            # `StepContext.graph_safe_kv`.
            append_kv_graph_safe(ctx.pool, self.kv_layer, ctx.slot_ids, ctx.positions, k, v)
        else:
            ctx.pool.append_kv(self.kv_layer, ctx.slot_ids, ctx.positions, k, v)
        out = ctx.attn.prefill(self.kv_layer, q.contiguous())
        return self._finish(out, gate, t)

    # -- mixed prefill + decode ---------------------------------------------- #
    def mixed(self, h: torch.Tensor, ctx: "StepContext") -> torch.Tensor:
        """Identical to :meth:`prefill`, and that is the whole design.

        Attention has no cross-token recurrent state, only the paged KV, so a
        decode row *is* a one-token varlen segment: ``qo_indptr`` gets a
        ``+1`` entry for it and ``kv_indptr`` gets its whole committed prefix,
        which under FlashInfer's bottom-right causal alignment is exactly the
        "attend to everything committed" a decode step computes. One
        ``BatchPrefillWithPagedKVCache`` plan (built by
        ``AttentionRunner.plan_prefill`` over ``MixedBatch.plan_*``) therefore
        covers both halves in one kernel launch per layer, which is how vLLM
        does it and is strictly better than two calls: no second plan, no
        second workspace pass, and the decode rows ride the same
        already-resident KV pages.

        The KV append is likewise uniform: ``(slot_ids, positions)`` is
        per-token over the whole step, so the decode rows' single K/V entries
        are appended by the same scatter. This is the *checked*
        ``pool.append_kv``, not the graph-safe one, because a mixed step is
        eager by construction (its shape changes every step).
        """
        return self.prefill(h, ctx)

    # -- speculative verify window ------------------------------------------ #
    def window(self, h: torch.Tensor, ctx: "StepContext", zero_m: torch.Tensor):
        """``n`` window tokens per sequence against the paged pool.

        Signature and return shape match :meth:`FusedGDN.window` (``(out,
        cache)``) so :meth:`FusedDecoderLayer.window` can treat both mixers
        alike; attention has no phase-B commit (KV rollback is a pointer move
        on ``seq_len``, owned by the scheduler), so the cache is ``None``.
        ``zero_m`` is accepted and ignored for the same symmetry reason.

        The KV write is the *graph-safe* append (this runs inside the captured
        speculative step), and the attention itself goes through
        ``ctx.attn.window``, a multi-token, bottom-right-causal paged call.
        Positions ``m .. k`` of the window are written speculatively and simply
        overwritten by the next step once the accepted length is known.
        """
        t = h.shape[0]
        q, k, v, gate = self._qkv(h, ctx.cos, ctx.sin)
        if ctx.kv_calibrator is not None:  # fp8-KV calibration
            ctx.kv_calibrator(self.kv_layer, k, v)
        append_kv_graph_safe(ctx.pool, self.kv_layer, ctx.slot_ids, ctx.positions, k, v)
        out = ctx.attn.window(self.kv_layer, q.contiguous())
        return self._finish(out, gate, t), None

    def commit_window(self, ctx: "StepContext", cache, m: torch.Tensor) -> None:
        """No-op: attention rollback is a ``seq_len`` pointer move."""
        return None

    def nbytes(self) -> int:
        return self.qkv.nbytes() + self.o_proj.nbytes()


class FusedDecoderLayer:
    def __init__(self, mixer, mlp: FusedMLP, ln_in: torch.Tensor, ln_post: torch.Tensor, eps: float,
                 norm_backend: str = "torch"):
        self.mixer = mixer
        self.mlp = mlp
        self.ln_in = ln_in
        self.ln_post = ln_post
        # `(1 + weight)` in fp32, built once (see norm_weight_1p): the inline
        # form recomputed it on every call, at 2 kernel launches x 129 norm
        # sites x every step.
        self.ln_in_1p = norm_weight_1p(ln_in)
        self.ln_post_1p = norm_weight_1p(ln_post)
        self.eps = eps
        self.norm_backend = norm_backend

    def _norm(self, x):
        if self.norm_backend == "triton":
            return triton_rms_norm(x, self.ln_in_1p, self.eps)
        return rms_norm_w1p(x, self.ln_in_1p, self.eps)

    def _add_norm(self, residual, x):
        if self.norm_backend == "triton":
            return triton_add_rms_norm(residual, x, self.ln_post_1p, self.eps)
        return add_rms_norm_w1p(residual, x, self.ln_post_1p, self.eps)

    def __call__(self, h: torch.Tensor, ctx: "StepContext", prefill: bool) -> torch.Tensor:
        residual = h
        h = self._norm(h)
        h = self.mixer.prefill(h, ctx) if prefill else self.mixer.decode(h, ctx)
        residual, h = self._add_norm(residual, h)
        h = self.mlp(h, tiled=prefill)
        return residual + h

    # -- mixed prefill + decode ---------------------------------------------- #
    def mixed(self, h: torch.Tensor, ctx: "StepContext") -> torch.Tensor:
        """One layer of a mixed step. Everything outside ``mixer.mixed``
        (both norms, the residual add and the MLP) is row-type-agnostic and
        runs once over ``[T_pre + B_dec]`` rows.

        ``tiled=True`` for the MLP, as on a prefill step: the tile is a pure
        row split (``FusedMLP`` concatenates the tiles' outputs), so it cannot
        mix rows across the split, and without it the ``[T, 2I]`` intermediate
        is what ``serve.plan_memory``'s prefill term bounds."""
        residual = h
        h = self._norm(h)
        h = self.mixer.mixed(h, ctx)
        residual, h = self._add_norm(residual, h)
        h = self.mlp(h, tiled=True)
        return residual + h

    # -- speculative verify window ------------------------------------------ #
    def window(self, h: torch.Tensor, ctx: "StepContext", zero_m: torch.Tensor):
        """One layer of the speculative verify pass.  Returns ``(h, cache)``;
        ``cache`` is ``None`` for attention layers and the per-position GDN
        inputs for linear-attention layers (fed back to
        :meth:`FusedGDN.commit_window` once ``m`` is known)."""
        residual = h
        h = self._norm(h)
        h, cache = self.mixer.window(h, ctx, zero_m)
        residual, h = self._add_norm(residual, h)
        h = self.mlp(h, tiled=False)
        return residual + h, cache

    def commit_window(self, ctx: "StepContext", cache, m: torch.Tensor) -> None:
        self.mixer.commit_window(ctx, cache, m)

    def nbytes(self) -> int:
        return self.mixer.nbytes() + self.mlp.nbytes()


class FusedMTPHead:
    """The shipped MTP head (``mtp.*``).

    Two RMSNorms -> ``fc`` over the concatenation of ``norm_e(embed(x))`` and
    ``norm_h(h)`` -> **one full-attention decoder layer** -> ``norm`` -> the
    target's shared ``lm_head``.

    ``hidden_first`` selects the concat order, which the checkpoint does not
    pin down (both orders give ``fc.weight = [5120, 10240]``): vLLM's
    ``Qwen3NextMTP`` uses ``[embedding ; hidden]``, while ``[hidden ; embedding]``
    is also plausible.  ``RuntimeConfig.mtp_hidden_first`` selects it and
    :meth:`set_order` can flip it at runtime.
    """

    def __init__(self, w: MTPFusedWeights, cfg: QwenFastConfig, rt: RuntimeConfig,
                 kv_layer: int, sm: Optional[int]):
        self.cfg = cfg
        self.hidden_first = rt.mtp_hidden_first
        self.pre_fc_norm_embedding = w.pre_fc_norm_embedding
        self.pre_fc_norm_hidden = w.pre_fc_norm_hidden
        self.pre_fc_norm_embedding_1p = norm_weight_1p(w.pre_fc_norm_embedding)
        self.pre_fc_norm_hidden_1p = norm_weight_1p(w.pre_fc_norm_hidden)
        self.norm_1p = norm_weight_1p(w.norm)
        self.fc = ResolvedLinear(w.fc_weight, sm, rt.gemm_backend)
        self.norm = w.norm
        self.eps = cfg.rms_norm_eps
        self.layer = FusedDecoderLayer(
            FusedAttention(w.attn, cfg, rt, kv_layer, sm),
            FusedMLP(w.mlp, rt, sm),
            w.input_layernorm,
            w.post_attention_layernorm,
            cfg.rms_norm_eps,
            rt.norm_backend,
        )

    def set_order(self, hidden_first: bool) -> None:
        self.hidden_first = hidden_first

    def __call__(
        self,
        inputs_embeds: torch.Tensor,
        previous_hidden: torch.Tensor,
        ctx: "StepContext",
        prefill: bool,
    ) -> torch.Tensor:
        e = rms_norm_w1p(inputs_embeds, self.pre_fc_norm_embedding_1p, self.eps)
        p = rms_norm_w1p(previous_hidden, self.pre_fc_norm_hidden_1p, self.eps)
        parts = [p, e] if self.hidden_first else [e, p]
        h = self.fc(torch.cat(parts, dim=-1))
        h = self.layer(h, ctx, prefill)
        return rms_norm_w1p(h, self.norm_1p, self.eps)

    def nbytes(self) -> int:
        return self.fc.nbytes() + self.layer.nbytes()


# =========================================================================== #
# 7. per-step context
# =========================================================================== #
@dataclass
class StepContext:
    """Everything a layer needs that varies per step."""

    pool: PagedKVPool
    attn: AttentionRunner
    state_pool: torch.Tensor
    conv_pool: torch.Tensor
    slot_ids: torch.Tensor  # [T] per-token slot (decode: one per row)
    positions: torch.Tensor  # [T]
    #: ``None`` only on the graphed mixed path, where the rotary gather
    #: is *inside* the captured region (it is a pure index_select on the static
    #: ``positions`` buffer, so it captures and replays like everything else)
    #: rather than run on the host before it. :meth:`FusedQwenForCausalLM
    #: .mixed_forward_body` fills them in on the first (capturing) pass.
    cos: Optional[torch.Tensor]
    sin: Optional[torch.Tensor]
    seq_slot_ids: Optional[torch.Tensor] = None  # [N] one per *sequence* (prefill)
    cu_seqlens: Optional[torch.Tensor] = None  # [N+1] int32
    #: ``cu_seqlens`` on the **host**, as the plain Python
    #: list the scheduler built it from. The prefill conv needs the segment
    #: boundaries on the host to launch over them; reading them back off
    #: ``cu_seqlens`` is a D2H sync, paid once per GDN layer (48x per chunk)
    #: inside ``torch_ops.conv_prefill``. The scheduler already has this
    #: list, so passing it costs nothing and removes the sync.
    q_lens: Optional[List[int]] = None
    #: ``cu_seqlens`` as a **host** int64 tensor, built once per chunk in
    #: :meth:`FusedQwenForCausalLM._context` from ``q_lens``. fla's varlen GDN
    #: path builds its chunk index tensors from a host copy of ``cu_seqlens``
    #: and reads one back off the device when it is not given one -- once per
    #: distinct tensor *object*, memoised on Python identity in a 4-deep deque.
    #: Without this that is one pipeline drain per chunk rather than 48 only because
    #: ``PrefillBatch.cu_seqlens`` is already int32 on the right device, so
    #: ``fla_ops.chunk_gdn``'s ``.to(...)`` is the identity; passing this makes
    #: it zero and makes it not depend on that. See
    #: ``kernels_gdn.fla_ops._cu_seqlens_cpu_kwarg``.
    cu_seqlens_cpu: Optional[torch.Tensor] = None
    #: The **row split** of a mixed prefill+decode step: rows
    #: ``[0, n_prefill_tokens)`` are the varlen prefill segments described by
    #: ``cu_seqlens``/``q_lens``/``seq_slot_ids``, and rows
    #: ``[n_prefill_tokens, T)`` are one-token decode rows whose per-row slots
    #: are ``decode_slot_ids``. ``None`` on every non-mixed step, which is what
    #: every mixer's ``prefill``/``decode`` method keys on to stay unchanged.
    #:
    #: The split is a *row range*, not a mask, on purpose: every sequence-mixing
    #: kernel here takes a contiguous token-major run, so both halves are plain
    #: dim-0 slices of the shared activation (no gather, no scatter, and the
    #: slices stay contiguous so the Triton kernels can take them directly).
    n_prefill_tokens: Optional[int] = None
    #: ``[B_dec]`` int32, one slot per decode row (the mixed step's analogue of
    #: ``slot_ids`` on a decode step). Attention does not need it -- its decode
    #: rows are length-1 segments in the same paged plan -- but GDN does, for
    #: both the conv ring update and the recurrent state read/write.
    decode_slot_ids: Optional[torch.Tensor] = None
    #: fla's varlen chunk index tensors, built on the
    #: **host** from ``q_lens`` and copied into persistent device buffers once
    #: per step (``kernels_gdn.fla_static.build_chunk_meta``). ``chunk_indices``
    #: is ``[NT, 2]`` (``(segment id, chunk index within that segment)`` for
    #: every 64-token chunk in the packed batch) and ``chunk_offsets`` is
    #: ``[N+1]``, the exclusive prefix sum of each segment's chunk count.
    #:
    #: When both are set, ``FusedGDN._gdn_prefill_core`` hands them to fla
    #: instead of letting it build them, which is the single fact that makes
    #: the GDN prefill half capturable: fla's own builder does host index prep,
    #: ends in a pageable H2D copy, and is memoised on tensor *identity* (so a
    #: reused ``cu_seqlens`` buffer would return the previous step's
    #: segmentation). ``None`` on every eager step, which then lets fla build
    #: the index tensors itself.
    chunk_indices: Optional[torch.Tensor] = None
    chunk_offsets: Optional[torch.Tensor] = None
    #: The **grid** the varlen prefill conv launches over in the token
    #: axis. Normally ``max(q_lens)``, which varies step to step and is
    #: therefore the one thing about that launch a CUDA graph cannot tolerate;
    #: set to the padded chunk's token cap inside a captured mixed step. The
    #: kernel masks every tile (``if t0 >= n: return``), so the extra tiles
    #: exit. Must be ``>= max(q_lens)``; ``causal_conv_prefill_varlen`` checks.
    conv_max_seqlen: Optional[int] = None
    #: The capture-time **hole** hook, set only while
    #: :class:`~.mixed_graphs.MixedGraphRunner` is recording a mixed step.
    #: ``FusedGDN.mixed`` hands it the one closure in the step that cannot be
    #: captured (the varlen conv + fla chunk kernel over the prefill
    #: segments, whose grid and index tensors are built on the host from
    #: ``cu_seqlens`` every step (``fla.ops.utils.index.prepare_chunk_indices``
    #: memoises on tensor *identity* and finishes with a pageable H2D
    #: ``.to(cu_seqlens)``: capturing it would either fail outright or, with a
    #: reused ``cu_seqlens`` buffer, silently return a stale index tensor)).
    #: The hook ends the current graph, runs the closure eagerly, copies its
    #: output into a static buffer and starts the next graph. ``None``
    #: everywhere else, which is what keeps the eager path one code path.
    graph_break: Optional[Callable[..., torch.Tensor]] = None
    #: Use ``append_kv_graph_safe`` instead of the checked
    #: ``pool.append_kv``: the checked one does ``bool((pages < 0).any())``, a
    #: D2H sync that CUDA-graph capture forbids. True only inside a captured
    #: mixed step, where the scheduler's ``ensure_capacity`` has already made
    #: the check redundant (the same argument ``decode`` has always made).
    graph_safe_kv: bool = False
    # fp8-KV calibration hook: when set, every attention
    # layer calls this with its *pre-quantization* (bf16) (kv_layer, k, v)
    # right before appending to the pool, instead of doing any real work.
    # ``FusedQwenForCausalLM.calibrate_kv_scales`` is the only caller that
    # sets it (a dedicated calibration forward, never a real request), so
    # this is always ``None`` on every context a captured graph sees:
    # a Python-level branch on a constant-per-context attribute, not a
    # tensor-value branch, so it does not affect graph-capturability.
    kv_calibrator: Optional[Callable[[int, torch.Tensor, torch.Tensor], None]] = None


@dataclass
class PrefillBatch:
    """One packed varlen prefill chunk."""

    token_ids: torch.Tensor  # [T] int
    positions: torch.Tensor  # [T] int
    slot_ids: torch.Tensor  # [T] int  (per token)
    seq_slot_ids: torch.Tensor  # [N] int (per sequence)
    cu_seqlens: torch.Tensor  # [N+1] int32
    q_lens: List[int]
    kv_lens: List[int]  # committed context length *after* this chunk
    last_indices: torch.Tensor  # [N] index into T of each sequence's last token
    #: ``seq_slot_ids`` on the host. ``prefill_forward``
    #: needs a Python list for ``plan_prefill``; getting one by calling
    #: ``seq_slot_ids.tolist()`` would be a D2H sync on the one path that is
    #: *already* eager and therefore already launch-latency-bound. The
    #: scheduler built the device tensor from this list in the first place.
    seq_slots: List[int] = field(default_factory=list)


@dataclass
class MixedBatch:
    """One mixed prefill+decode step.

    The token axis is ``[prefill chunk tokens ‖ decode rows]``: the first
    ``n_prefill_tokens`` rows are exactly ``prefill``'s packed varlen chunk,
    unchanged and in the same order, and the remaining ``B_dec`` rows are one
    token each, one per running sequence.

    Holding the prefill half as a real :class:`PrefillBatch` (rather than
    re-flattening its fields) is deliberate: it is what
    ``SpecDecoder.on_prefill`` takes, it is what ``make_prefill_batch``
    already builds, and it keeps "the prefill half of a mixed step" and "a
    prefill step" the same object, so they cannot drift.

    The three ``plan_*`` lists are the *attention* view of the whole step:
    decode rows are length-1 segments appended to the prefill segments, which
    is how one ``BatchPrefillWithPagedKVCache`` plan covers both (vLLM's
    shape). ``kv_lens`` is post-write for both halves (``start + q_len`` for
    a segment, ``num_computed_tokens + 1`` for a decode row) because
    ``plan_prefill`` runs before any layer appends its K/V (see its docstring).
    """

    prefill: PrefillBatch
    n_prefill_tokens: int
    token_ids: torch.Tensor  # [T] int32, prefill tokens ‖ each decode row's last token
    positions: torch.Tensor  # [T] int32
    slot_ids: torch.Tensor  # [T] int32, per token
    decode_slot_ids: torch.Tensor  # [B_dec] int32, per decode row
    decode_slots: List[int]  # the same, on the host
    decode_positions: List[int]  # [B_dec] each row's pre-step context length L
    decode_token_ids: List[int]  # [B_dec] each row's input token
    plan_slots: List[int]  # prefill.seq_slots + decode_slots
    plan_q_lens: List[int]  # prefill.q_lens + [1] * B_dec
    plan_kv_lens: List[int]  # post-write context length of every plan row
    #: ``[N_pre + B_dec]`` rows of the final hidden state that ``lm_head`` runs
    #: on: each segment's **last** token (a segment that finished its prompt in
    #: this chunk samples from it) followed by every decode row. Never the whole
    #: ``[T, vocab]``, which would be 8 GB of fp32 logits at T=8192.
    logits_indices: torch.Tensor

    @property
    def n_prefill_seqs(self) -> int:
        return len(self.prefill.q_lens)

    @property
    def n_decode_rows(self) -> int:
        return len(self.decode_slots)


# =========================================================================== #
# 8. the model
# =========================================================================== #
class FusedQwenForCausalLM:
    """The fused runtime model.  Build with :meth:`from_pretrained` (real checkpoint),
    :meth:`from_fused` (a pre-built :class:`FusedModelWeights`) or
    :meth:`from_m0_module` (a ``qwenfast.model.QwenFastForCausalLM``, which is
    what the CPU tests use so that the bf16/no-fp8 path goes through *exactly*
    this code)."""

    def __init__(self, fused: FusedModelWeights, rt: RuntimeConfig):
        self.fused = fused
        self.rt = rt
        cfg: QwenFastConfig = fused.config  # type: ignore[assignment]
        self.config = cfg
        self.device = torch.device(rt.device)
        self.dtype = rt.torch_dtype()
        self.sm_version = _sm_version(self.device)
        # Process-wide, because the repack caches live on the weight objects.
        # Set before any `ResolvedLinear` can run so warmup's very first
        # `resolve_backend` already sees the policy.
        gemm_dispatch.set_weight_cache_policy(rt.gemm_weight_cache)
        # Same reasoning, same place: process-wide, and set before any
        # `ResolvedLinear` can memoise a backend.
        gemm_dispatch.set_gemm_accuracy(rt.gemm_accuracy)
        # Same reasoning, same place, third time: the
        # cold-start ranking must be fixed before warmup pins anything.
        gemm_dispatch.set_backend_priority_profile(rt.gemm_priority)

        self.attn_layer_indices = list(cfg.attention_layer_indices)
        self.gdn_layer_indices = list(cfg.linear_layer_indices)
        self._attn_ordinal = {g: i for i, g in enumerate(self.attn_layer_indices)}
        self._gdn_ordinal = {g: i for i, g in enumerate(self.gdn_layer_indices)}
        self.mtp_kv_layer = len(self.attn_layer_indices)

        # -- pools.  One extra slot/row is the scratch slot that
        #    CUDA-graph padding rows point at.
        self.n_slots = rt.max_num_seqs
        self.scratch_slot = self.n_slots
        n_rows = self.n_slots + 1
        self.state_dtype = gdn_state.resolve_state_dtype(rt.ssm_state_dtype)
        self.state_pool = gdn_state.alloc_state_pool(
            n_rows,
            n_layers=len(self.gdn_layer_indices),
            n_v_heads=cfg.linear_num_value_heads,
            head_k=cfg.linear_key_head_dim,
            head_v=cfg.linear_value_head_dim,
            dtype=self.state_dtype,
            device=self.device,
        )
        self.conv_pool = gdn_state.alloc_conv_state_pool(
            n_rows,
            n_layers=len(self.gdn_layer_indices),
            conv_dim=cfg.conv_dim,
            width=cfg.linear_conv_kernel_dim,
            dtype=self.dtype,
            device=self.device,
        )
        n_kv_layers = len(self.attn_layer_indices) + (1 if fused.mtp is not None else 0)
        self.kv_pool = _make_kv_pool(cfg, rt, n_kv_layers, n_rows, self.device, self.dtype)
        _claim_all_kv_slots(self.kv_pool)

        self.attn = AttentionRunner(
            self.kv_pool,
            num_qo_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            backend=rt.attn_backend,
            device=self.device,
            workspace_mb=rt.attn_workspace_mb,
            buckets=rt.buckets_for(),  # informational only now; wrappers build lazily
            max_pages=rt.n_kv_pages,
            use_cuda_graph_wrappers=rt.use_cuda_graphs,
            separate_decode_workspace=bool(getattr(rt, "overlap_streams", False)),
        )
        self.rotary = RotaryTable(
            max_positions=min(rt.max_model_len, cfg.max_position_embeddings),
            rotary_dim=cfg.rotary_dim,
            theta=float(cfg.rope_theta),
            dtype=self.dtype,
            device=self.device,
        )

        # -- weights ---------------------------------------------------------
        self.embed_tokens = fused.embed_tokens
        self.lm_head = ResolvedLinear(
            fused.lm_head_fp8 if fused.lm_head_fp8 is not None else fused.lm_head,
            self.sm_version,
            rt.gemm_backend,
        )
        self.final_norm = fused.final_norm
        self.final_norm_1p = norm_weight_1p(fused.final_norm)
        self.layers: List[FusedDecoderLayer] = []
        for i in range(cfg.num_hidden_layers):
            ln_in, ln_post = fused.layernorms[i]
            if cfg.layer_types[i] == "linear_attention":
                mixer = FusedGDN(fused.gdn[i], cfg, rt, self._gdn_ordinal[i], self.sm_version)
            else:
                mixer = FusedAttention(fused.attn[i], cfg, rt, self._attn_ordinal[i], self.sm_version)
            self.layers.append(
                FusedDecoderLayer(
                    mixer, FusedMLP(fused.mlp[i], rt, self.sm_version),
                    ln_in, ln_post, cfg.rms_norm_eps, rt.norm_backend,
                )
            )
        self.mtp: Optional[FusedMTPHead] = None
        if fused.mtp is not None and rt.enable_mtp:
            self.mtp = FusedMTPHead(fused.mtp, cfg, rt, self.mtp_kv_layer, self.sm_version)

        # scratch slot needs one page and a non-empty context so a padded
        # FlashInfer row is well-formed.
        self._init_scratch_slot()

    # -- byte inventory (the memory plan's ground truth) --------------------- #
    def fp8_weights(self) -> List[FP8Tensor]:
        """Every ``FP8Tensor`` the dispatcher can hang a repack cache on."""
        out: List[FP8Tensor] = []
        src = self.fused
        for g in src.gdn.values():
            out += [g.in_proj_qkvz, g.out_proj]
        for a in src.attn.values():
            out += [a.qkv_proj, a.o_proj]
        for m in src.mlp.values():
            out += [m.gate_up_proj, m.down_proj]
        if src.mtp is not None:
            out += [src.mtp.attn.qkv_proj, src.mtp.attn.o_proj,
                    src.mtp.mlp.gate_up_proj, src.mtp.mlp.down_proj]
        if src.lm_head_fp8 is not None:
            out.append(src.lm_head_fp8)
        return [w for w in out if isinstance(w, FP8Tensor)]

    @property
    def max_context_len(self) -> int:
        """Longest sequence (prompt + completion) this model can *index*.

        This is the single source of truth for "how long
        may a sequence get", and it is deliberately **not** ``rt.max_model_len``:
        two independent, differently-sized structures are indexed by position,
        and the smaller one is what actually asserts.

        * ``rotary.cos/sin`` have ``min(rt.max_model_len,
          cfg.max_position_embeddings)`` rows, gathered by
          ``RotaryTable.lookup(positions)``, an ``index_select`` with **no
          bounds handling of its own**, so position ``max_positions`` is a
          device-side ``IndexKernel.cu ... index out of bounds`` assert that
          kills the CUDA context. For example, ``--max-model-len 2560``, a
          2158-token prompt and ``max_tokens=500`` reach position 2560 402
          decode steps in.
        * the KV page table is ``[rows, max_pages_per_seq]``, so a sequence can
          never hold more than ``max_pages_per_seq * page_size`` tokens
          (``Scheduler._ensure_capacity_with_preemption`` already refuses past
          that, but it can be the *looser* of the two bounds, e.g.
          165 x 16 = 2640 > 2560).

        Callers must treat this as a hard cap: reject at admission
        (:meth:`Scheduler.context_length_error`, surfaced as HTTP 400) and stop
        with ``finish_reason="length"`` if a sequence ever reaches it.
        """
        return int(
            min(
                self.rotary.max_positions,
                self.kv_pool.cfg.max_pages_per_seq * self.kv_pool.cfg.page_size,
            )
        )

    def repack_cache_nbytes(self) -> int:
        """Device bytes held by GEMM-backend repack caches.

        Zero until warmup resolves a backend that memoises one. ``serve.plan_memory``
        must include this term; without it the plan can under-predict the real
        allocation by tens of GiB."""
        return sum(gemm_dispatch.repack_cache_bytes(w) for w in self.fp8_weights())

    def pool_nbytes(self) -> Dict[str, int]:
        """``{name: bytes}`` for every pool this model owns.

        Deliberately measured off the live tensors (``numel * element_size``)
        rather than recomputed from the config, so a dtype/shape mistake shows
        up as a mismatch against ``serve.plan_memory`` instead of cancelling
        out on both sides."""
        def sz(t) -> int:
            return int(t.numel()) * int(t.element_size()) if t is not None else 0

        kv = sz(self.kv_pool.kv) + sz(getattr(self.kv_pool, "scale", None))
        kv += sz(self.kv_pool.page_table) + sz(self.kv_pool.seq_len)
        return {
            "weights": int(self.fused.nbytes()["total"]),
            "repack_caches": self.repack_cache_nbytes(),
            "kv_pool": kv,
            "ssm_state": sz(self.state_pool),
            "conv_state": sz(self.conv_pool),
            "rotary": sz(getattr(self.rotary, "cos", None)) + sz(getattr(self.rotary, "sin", None)),
            "attn_workspace": sz(getattr(self.attn, "workspace", None)),
        }

    # -- constructors ------------------------------------------------------- #
    @classmethod
    def from_fused(cls, fused: FusedModelWeights, rt: RuntimeConfig) -> "FusedQwenForCausalLM":
        return cls(fused, rt)

    @classmethod
    def from_pretrained(cls, model_dir: str, rt: RuntimeConfig, *, fused_cache: Optional[str] = None,
                        verbose: bool = False) -> "FusedQwenForCausalLM":
        from ..gemm.fused_weights import build_fused_weights, load_fused

        if fused_cache and os.path.exists(fused_cache):
            fused = load_fused(fused_cache, device=rt.device)
        else:
            fused = build_fused_weights(
                model_dir, device=rt.device, include_mtp=True, verbose=verbose
            )
        return cls(fused, rt)

    @classmethod
    def from_m0_module(cls, model, rt: RuntimeConfig) -> "FusedQwenForCausalLM":
        """Build from a reference ``QwenFastForCausalLM``: the bf16/no-fp8 path.

        Used by the CPU tests so that "fused == reference" is a statement about
        *this* code path, not about a parallel re-implementation.
        """
        return cls(fused_weights_from_module(model), rt)

    # -- pool bookkeeping ---------------------------------------------------- #
    def _init_scratch_slot(self) -> None:
        pages = self.kv_pool.allocator.alloc(1)
        # `map_page`, not two raw table writes: the pool keeps an
        # O(1) page count per slot (`PagedKVPool.pages_allocated`) and a
        # caller that writes the table behind its back would desync it.
        self.kv_pool.map_page(self.scratch_slot, 0, pages[0])
        self.kv_pool.seq_len[self.scratch_slot] = 1

    def reset_slot(self, slot: int) -> None:
        """Reset a slot for a *new* sequence (called on admission).

        Zeroes the SSM + conv state **and returns the slot's KV pages to the
        allocator** (``PagedKVPool.free_pages``).  Without the page release
        every slot recycle leaks its whole page set and a long run exhausts
        the page pool.  A caller that wants the state zeroed but the pages
        kept should zero the pools directly; no caller in this tree does.
        """
        idx = torch.tensor([slot], dtype=torch.long, device=self.device)
        self.state_pool.index_fill_(0, idx, 0)
        self.conv_pool.index_fill_(0, idx, 0)
        if slot != self.scratch_slot:
            self.kv_pool.free_pages(slot)

    # -- GEMM cache claim and fp8 KV-cache calibration ------------------------ #
    def claim_gemm_cache(self, m: Optional[int] = None, *, verbose: bool = False) -> Dict[str, int]:
        """Resolve every linear at ``M = m`` so the large-M winner takes the
        single repack-cache slot (``RuntimeConfig.gemm_cache_owner``).

        ``m`` defaults to :func:`default_claim_gemm_cache_m` -- the real
        prefill-chunk token cap, **not** a fixed 512. A fixed 512 would make
        ``--gemm-cache-owner prefill`` claim the cache slot for whoever won
        the *decode* M=512 bucket. `dispatch.M_BUCKETS` carries buckets up to
        8,192, and this call site is the other half of routing a real chunk
        on them instead of on M=512's answer. An explicit ``m=`` (as the CPU
        tests pass) overrides it.

        Runs one tiny dummy activation of ``m`` rows through each
        ``ResolvedLinear``, **not** a real forward: a real prefill chunk at
        M=8192 would allocate 8192-row activations for all 305 layers, and all
        this needs is for ``resolve_backend`` to run once per (weight, bucket)
        with the bucket set to the prefill one.

        Must be called before ``GraphedDecoder.warmup()``. After it, the decode
        buckets resolve against a weight that already holds a cache, so
        ``_cache_policy_allows`` sends them to the best *cache-free* backend
        instead of to the one whose slot is taken, which is the whole point.

        Returns ``{backend: n_layers}`` for the bucket it just pinned, so the
        caller can log what actually happened rather than what was intended.
        """
        explicit_m = m is not None
        if m is None:
            m = default_claim_gemm_cache_m(self.rt)
        if self.device.type != "cuda":
            return {}
        picked: Dict[str, int] = {}
        forced = self.rt.prefill_gemm_backend
        for lin in _iter_resolved_linears(self):
            is_fp8 = isinstance(lin.weight, FP8Tensor)
            k = lin.weight.weight.shape[1] if is_fp8 else lin.weight.shape[1]
            n = lin.weight.weight.shape[0] if is_fp8 else lin.weight.shape[0]
            # Per-weight M. An explicit `m=` still pins
            # every weight to the same value (that is what an explicit override
            # means); the default resolves each weight at the M it is called
            # with, which for the two tiled MLP shapes is `mlp_tile_tokens`.
            m_lin = m if explicit_m else claim_gemm_cache_m_for(self.rt, int(n), int(k))
            x = torch.zeros(m_lin, int(k), dtype=self.dtype, device=self.device)
            try:
                with prefill_gemm_scope(forced):
                    lin(x)
            except Exception as exc:  # noqa: BLE001 -- a failing weight is data
                if verbose:
                    print(f"[claim_gemm_cache] skipped a linear: {exc}", flush=True)
                continue
            name = lin.backend or forced or "?"
            key = name if m_lin == m else f"{name}@M{m_lin}"
            picked[key] = picked.get(key, 0) + 1
        if verbose:
            parts = ", ".join(f"{k}={v}" for k, v in sorted(picked.items()))
            print(f"[claim_gemm_cache] M={m}: {parts}", flush=True)
        return picked

    def calibrate_kv_scales(
        self,
        prompts: Sequence[Sequence[int]],
        *,
        headroom: float = 0.1,
        slots: Optional[Sequence[int]] = None,
    ) -> Dict[int, Tuple[float, float]]:
        """Static per-layer fp8 KV scale, from one representative prefill.

        The calibration strategy is a **static scale per (layer, K-or-V)**,
        set once from a calibration prefill over real prompts and reused for
        every subsequent step, not a running/per-step amax. Two reasons this is the sane default here, not just
        the simple one:

        1. **Graph safety.** ``AttentionRunner._scales_for`` hands FlashInfer
           a plain Python ``float`` (``PagedKVPool.calibrate_uniform_scale``'s
           docstring: FlashInfer's fp8 kernels take one float per call, not a
           tensor), and that float gets baked into the kernel launch the
           moment ``GraphedDecoder.capture()`` traces the step. A scale that
           changes step-to-step cannot be re-baked into an already-captured
           graph, so "recalibrate every step" is not an available design
           once graphs are on: it would have to run *outside* the graph
           and be re-captured, which defeats CUDA graphs' whole purpose here
           (a measured 4.3x). A static scale, fixed before capture, is the
           only option that is graph-safe by construction, matching exactly
           how ``ResolvedLinear`` pins its GEMM backend before capture too.
        2. **The distribution is stable enough.** K/V activations after
           RMSNorm (this model applies q/k-norm before RoPE) have a
           roughly fixed scale set by the norm's own output range, not by
           token content, unlike, say, raw hidden-state activations. One
           calibration prefill over a handful of real prompts already sees
           that range; ``headroom`` (default 10%, looser than
           ``calibrate_uniform_scale``'s own 1/448 default used for
           within-a-page calibration) reserves extra dynamic range above the
           observed amax for tokens/prompts the calibration set didn't
           happen to cover, at the cost of a little precision.

        Must be called **before** ``GraphedDecoder.warmup()``/``.capture()``,
        for the same reason as (1) above, and before any real traffic is
        admitted onto the ``slots`` it borrows (default: slots
        ``0..min(len(prompts), n_slots)-1``) -- it runs a real prefill
        through every attention layer, which writes real KV pages and
        advances SSM/conv state on those rows, then frees/resets them again
        before the next chunk (or before returning). No-op (returns ``{}``)
        when the pool is not fp8.

        **Sequential chunking:** when there are more
        calibration prompts than available slots (the extreme case being
        ``profile_step``/``bench_runtime`` built with ``max_num_seqs=1``,
        where a single slot is all there is), the prompts are calibrated
        in successive chunks of at most ``len(slots)`` (default
        ``n_slots``) prompts per forward pass, each chunk reusing the same
        slot ids and each layer's amax accumulated (via ``_record``'s
        running ``torch.maximum``) across *all* chunks before the final
        scale is computed once at the end. So a 4-prompt calibration set
        against 1 slot runs 4 sequential one-prompt prefills and still
        yields the same scale a single 4-wide batch would have, at the
        cost of 4 small forward passes instead of 1.

        Returns ``{kv_layer: (k_scale, v_scale)}`` for the layers calibrated.
        The MTP head's KV layer (``self.mtp_kv_layer``) is **not** included:
        its KV scale stays at the pool's uncalibrated default (1.0) until the
        MTP path calibrates it the same way.
        """
        if self.kv_pool.cfg.dtype != "fp8":
            return {}
        if not prompts:
            raise ValueError("calibrate_kv_scales needs at least one prompt")
        n = len(prompts)
        if slots is None:
            chunk_slots = list(range(min(n, self.n_slots)))
        else:
            chunk_slots = list(slots)
            if len(chunk_slots) > self.n_slots:
                raise ValueError(
                    f"{len(chunk_slots)} calibration slots > {self.n_slots} available slots"
                )
        if not chunk_slots:
            raise ValueError("calibrate_kv_scales needs at least one slot")
        chunk_size = len(chunk_slots)

        samples: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

        def _record(layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
            ks = k.detach().float().abs().amax()
            vs = v.detach().float().abs().amax()
            prev = samples.get(layer)
            if prev is not None:
                ks = torch.maximum(ks, prev[0])
                vs = torch.maximum(vs, prev[1])
            samples[layer] = (ks, vs)

        prompts_list = list(prompts)
        for start in range(0, n, chunk_size):
            chunk = prompts_list[start : start + chunk_size]
            slots_i = chunk_slots[: len(chunk)]
            for slot, p in zip(slots_i, chunk):
                self.reset_slot(slot)
                self.kv_pool.ensure_capacity(slot, len(p))

            batch = make_prefill_batch(chunk, [0] * len(chunk), slots_i, self.device)
            ctx = self._context(
                batch.slot_ids, batch.positions,
                seq_slot_ids=batch.seq_slot_ids, cu_seqlens=batch.cu_seqlens,
            )
            ctx.kv_calibrator = _record
            self.attn.plan_prefill(_seq_slots_of(batch), batch.q_lens, batch.kv_lens)
            h = F.embedding(batch.token_ids.long(), self.embed_tokens).to(self.dtype)
            with torch.no_grad():
                for layer in self.layers:
                    h = layer(h, ctx, prefill=True)

            # The calibration forward wrote real KV pages and advanced SSM/conv
            # state on the slots it borrowed -- reset them before the next
            # chunk (or real traffic / a benchmark's `_fake_context`) reuses
            # those rows.
            for slot in slots_i:
                self.reset_slot(slot)

        scales: Dict[int, Tuple[float, float]] = {}
        for layer, (ks, vs) in samples.items():
            k_scale = self.kv_pool.calibrate_uniform_scale(layer, 0, ks.reshape(1), headroom)
            v_scale = self.kv_pool.calibrate_uniform_scale(layer, 1, vs.reshape(1), headroom)
            self.attn.set_kv_scale(layer, k_scale, v_scale)
            scales[layer] = (k_scale, v_scale)
        return scales

    def nbytes(self) -> Dict[str, int]:
        weights = sum(layer.nbytes() for layer in self.layers)
        weights += self.embed_tokens.numel() * self.embed_tokens.element_size()
        weights += self.lm_head.nbytes()
        mtp = self.mtp.nbytes() if self.mtp is not None else 0
        ssm = self.state_pool.numel() * self.state_pool.element_size()
        conv = self.conv_pool.numel() * self.conv_pool.element_size()
        kv = self.kv_pool.nbytes()
        return {
            "weights": weights,
            "mtp": mtp,
            "ssm_state": ssm,
            "conv_state": conv,
            "kv": kv,
            "total": weights + mtp + ssm + conv + kv,
        }

    # -- forwards ------------------------------------------------------------ #
    def _final_norm(self, h: torch.Tensor) -> torch.Tensor:
        if self.rt.norm_backend == "triton":
            return triton_rms_norm(h, self.final_norm_1p, self.config.rms_norm_eps)
        return rms_norm_w1p(h, self.final_norm_1p, self.config.rms_norm_eps)

    def _context(self, slot_ids, positions, *, seq_slot_ids=None, cu_seqlens=None,
                 q_lens=None, n_prefill_tokens=None, decode_slot_ids=None,
                 rotary: bool = True, chunk_indices=None, chunk_offsets=None,
                 conv_max_seqlen=None) -> StepContext:
        # `rotary=False` (graphed mixed step only): the cos/sin gather
        # belongs *inside* the captured region there, so `mixed_forward_body`
        # does it. Every other caller wants it here, once, on the host side of
        # the step.
        cos, sin = self.rotary.lookup(positions) if rotary else (None, None)
        # One small host tensor per chunk (never per layer), from the
        # segment lengths the scheduler already has. `None` on the decode path,
        # which has no varlen segments and never reaches fla's chunk kernel.
        cu_cpu = None
        if q_lens:
            acc, cu = 0, [0]
            for n in q_lens:
                acc += int(n)
                cu.append(acc)
            cu_cpu = torch.tensor(cu, dtype=torch.int64, device="cpu")
        return StepContext(
            pool=self.kv_pool,
            attn=self.attn,
            state_pool=self.state_pool,
            conv_pool=self.conv_pool,
            slot_ids=slot_ids,
            positions=positions,
            cos=cos,
            sin=sin,
            seq_slot_ids=seq_slot_ids,
            cu_seqlens=cu_seqlens,
            q_lens=q_lens,
            cu_seqlens_cpu=cu_cpu,
            n_prefill_tokens=n_prefill_tokens,
            decode_slot_ids=decode_slot_ids,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            conv_max_seqlen=conv_max_seqlen,
        )

    def decode_forward(self, buf: DeviceBuffers, batch: int, *, write_logits: bool = True):
        """One decode step for ``batch`` (already bucket-padded) rows.

        Reads ``buf.input_ids``/``positions``/``slot_ids``; mutates the SSM
        pool and the KV pages in place; writes ``buf.logits[:batch]``.
        ``AttentionRunner.plan_decode`` must have been called for this batch.
        """
        ids = buf.input_ids[:batch]
        positions = buf.positions[:batch]
        slot_ids = buf.slot_ids[:batch]
        ctx = self._context(slot_ids, positions)
        h = F.embedding(ids.long(), self.embed_tokens).to(self.dtype)
        for layer in self.layers:
            h = layer(h, ctx, prefill=False)
        h = self._final_norm(h)
        logits = self.lm_head(h).float()
        if write_logits:
            buf.logits[:batch].copy_(logits)
            return buf.logits[:batch]
        return logits

    def prefill_forward(self, batch: PrefillBatch, *, return_hidden: bool = False,
                        all_logits: bool = False):
        """One packed varlen prefill chunk.

        Returns ``logits`` for each sequence's **last** token (``[N, vocab]``)
        unless ``all_logits``; the chunked-prefill scheduler only needs the
        last one, and materialising ``[8192, 248320]`` fp32 logits would be
        8 GB.
        """
        ctx = self._context(
            batch.slot_ids,
            batch.positions,
            seq_slot_ids=batch.seq_slot_ids,
            cu_seqlens=batch.cu_seqlens,
            q_lens=batch.q_lens,
        )
        # `batch.seq_slots` (host) rather than `batch.seq_slot_ids.tolist()`
        # (D2H sync). `_seq_slots_of` falls back to the D2H path for a
        # `PrefillBatch` that does not carry `seq_slots`.
        self.attn.plan_prefill(
            _seq_slots_of(batch), batch.q_lens, batch.kv_lens
        )
        with prefill_gemm_scope(self.rt.prefill_gemm_backend):
            h = F.embedding(batch.token_ids.long(), self.embed_tokens).to(self.dtype)
            for layer in self.layers:
                h = layer(h, ctx, prefill=True)
            h = self._final_norm(h)
            sel = h if all_logits else h.index_select(0, batch.last_indices)
            logits = self.lm_head(sel).float()
        if return_hidden:
            return logits, h
        return logits

    def mixed_forward(self, batch: MixedBatch, *, return_hidden: bool = False):
        """One **mixed** step: a prefill chunk and every decode row, together.

        Same contract as :meth:`prefill_forward` (mutates the
        SSM/conv pools and the KV pages in place, returns logits) with one
        difference in the returned shape: ``[N_pre + B_dec, vocab]``, the
        segments' last-token rows followed by every decode row, in
        ``batch.logits_indices`` order. The caller samples the rows it needs
        (a segment mid-prompt has no token to emit; every decode row does).

        Runs eager. The step's shape is a different varlen packing every call,
        so there is nothing to capture; the graphed decode-only step
        (:meth:`decode_forward`) is still what runs when nothing is pending.

        ``prefill_gemm_scope`` wraps the whole stack because at
        ``M = T_pre + B_dec >= 2k`` every GEMM in it is prefill-shaped: the
        decode rows are a <3% tail on the M the dispatcher routes on, which is
        the entire point of the mixed step.
        """
        ctx = self._context(
            batch.slot_ids,
            batch.positions,
            seq_slot_ids=batch.prefill.seq_slot_ids,
            cu_seqlens=batch.prefill.cu_seqlens,
            q_lens=batch.prefill.q_lens,
            n_prefill_tokens=batch.n_prefill_tokens,
            decode_slot_ids=batch.decode_slot_ids,
        )
        # One plan for both halves: decode rows are length-1 segments (see
        # `FusedAttention.mixed`). `plan_kv_lens` is post-write for every row.
        self.attn.plan_prefill(batch.plan_slots, batch.plan_q_lens, batch.plan_kv_lens)
        return self.mixed_forward_body(
            ctx, batch.token_ids, batch.logits_indices, return_hidden=return_hidden
        )

    def mixed_forward_body(self, ctx: "StepContext", token_ids: torch.Tensor,
                           logits_indices: torch.Tensor, *, return_hidden: bool = False):
        """The device half of a mixed step: embed -> 64 layers -> norm -> head.

        Split out of :meth:`mixed_forward` so the CUDA-graph
        runner can drive **exactly this** body with the plan already done and
        the inputs already in its static buffers: the same
        plan()-outside/run()-inside split ``graphs.GraphedDecoder`` uses
        for the decode step, one level up. Everything here is either a
        fixed-shape tensor op or, in the GDN prefill half, routed through
        ``ctx.graph_break``.
        """
        if ctx.cos is None:
            ctx.cos, ctx.sin = self.rotary.lookup(ctx.positions)
        with prefill_gemm_scope(self.rt.prefill_gemm_backend):
            h = F.embedding(token_ids.long(), self.embed_tokens).to(self.dtype)
            for layer in self.layers:
                h = layer.mixed(h, ctx)
            h = self._final_norm(h)
            sel = h.index_select(0, logits_indices)
            logits = self.lm_head(sel).float()
        if return_hidden:
            return logits, h
        return logits

    def mtp_forward(self, token_ids: torch.Tensor, hidden: torch.Tensor,
                    batch: PrefillBatch, *, hidden_first: Optional[bool] = None):
        """Run the MTP head over a packed batch and return its logits.

        ``token_ids`` are the *next* tokens (the head predicts t+2 from
        ``h_t`` and ``x_{t+1}``).
        """
        if self.mtp is None:
            raise RuntimeError("MTP head is not loaded (RuntimeConfig.enable_mtp=False)")
        if hidden_first is not None:
            self.mtp.set_order(hidden_first)
        ctx = self._context(
            batch.slot_ids,
            batch.positions,
            seq_slot_ids=batch.seq_slot_ids,
            cu_seqlens=batch.cu_seqlens,
            q_lens=batch.q_lens,
        )
        self.attn.plan_prefill(_seq_slots_of(batch), batch.q_lens, batch.kv_lens)
        emb = F.embedding(token_ids.long(), self.embed_tokens).to(self.dtype)
        h = self.mtp(emb, hidden, ctx, prefill=True)
        return self.lm_head(h).float()


# =========================================================================== #
# 9. helpers
# =========================================================================== #
def _sm_version(device: torch.device) -> Optional[int]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability(device)
    return major * 10 + minor


def _make_kv_pool(cfg: QwenFastConfig, rt: RuntimeConfig, n_layers: int, max_seqs: int,
                  device: torch.device, act_dtype: torch.dtype) -> PagedKVPool:
    pool = PagedKVPool(
        KVPoolConfig(
            n_layers=n_layers,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            page_size=rt.page_size,
            n_pages=rt.n_kv_pages,
            max_seqs=max_seqs,
            max_pages_per_seq=rt.max_pages_per_seq,
            dtype=rt.kv_cache_dtype,
            device=str(device),
        )
    )
    if rt.kv_cache_dtype == "bf16" and act_dtype not in (torch.bfloat16,):
        _retype_kv_pool(pool, act_dtype)
    return pool


def _retype_kv_pool(pool: PagedKVPool, dtype: torch.dtype) -> None:
    """WORKAROUND: fp32 KV storage for the CPU parity tests.

    ``KVPoolConfig.dtype`` only accepts ``"bf16"``/``"fp8"`` and
    ``PagedKVPool.gather_dense`` hard-casts its output to bf16.  For the fp32
    CPU parity test we need a pool that stores and returns fp32, otherwise the
    bf16 KV round-trip (~4e-3 relative) swamps the 1e-4 tolerance we want on
    the *engine* logic.  Re-typing the storage tensor and shimming
    ``gather_dense`` is a local, contained fix; upstream should grow a real
    dtype knob.
    """
    pool.kv = torch.zeros(pool.kv.shape, dtype=dtype, device=pool.kv.device)
    pool.storage_dtype = dtype
    orig = pool.gather_dense

    def gather_dense(layer, slot, length=None):
        k, v = orig(layer, slot, length)
        return k.to(dtype), v.to(dtype)

    pool.gather_dense = gather_dense  # type: ignore[assignment]


def _claim_all_kv_slots(pool: PagedKVPool) -> None:
    """WORKAROUND: the runtime, not the KV pool, owns slot ids.

    ``PagedKVPool`` owns its own sequence-slot free list, but the runtime must
    hand out **one** slot id that indexes the SSM pool *and* the KV page table
    (the ABI's ``slot_ids[B]``).  Two independent allocators cannot agree on that
    by construction, so the runtime's ``SlotManager`` owns slot ids and the KV
    pool is told every row is live; page allocation still goes through the
    pool's own ``PageAllocator``.
    """
    pool._used_slots = set(range(pool.cfg.max_seqs))  # noqa: SLF001
    pool._free_slots = []  # noqa: SLF001


# --------------------------------------------------------------------------- #
# reference module -> FusedModelWeights (the no-fp8 path used by the CPU tests)
# --------------------------------------------------------------------------- #
def fused_weights_from_module(model) -> FusedModelWeights:
    """Fuse an in-memory reference ``QwenFastForCausalLM`` into the fused layout.

    Same concatenations as ``gemm.fused_weights.build_fused_weights``, but
    reading ``nn.Linear.weight`` instead of a safetensors store and leaving
    everything unquantized (``FusedModelWeights`` accepts a plain tensor
    wherever it accepts an ``FP8Tensor``; ``dispatch.linear`` dispatches on
    ``isinstance(w, FP8Tensor)``, so a plain tensor takes the ``bf16_dequant``
    path, i.e. a straight ``F.linear``).
    """
    cfg: QwenFastConfig = model.config
    m = model.model
    layernorms: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    gdn: Dict[int, GDNFusedWeights] = {}
    attn: Dict[int, AttnFusedWeights] = {}
    mlp: Dict[int, MLPFusedWeights] = {}

    for i, layer in enumerate(m.layers):
        layernorms[i] = (layer.input_layernorm.weight.data, layer.post_attention_layernorm.weight.data)
        if cfg.layer_types[i] == "linear_attention":
            la = layer.linear_attn
            gdn[i] = GDNFusedWeights(
                layer_idx=i,
                in_proj_qkvz=torch.cat(
                    [la.in_proj_qkv.weight.data, la.in_proj_z.weight.data], dim=0
                ).contiguous(),
                in_proj_ba=torch.cat(
                    [la.in_proj_b.weight.data, la.in_proj_a.weight.data], dim=0
                ).contiguous(),
                out_proj=la.out_proj.weight.data,
                conv1d_weight=la.conv1d.weight.data,
                A_log=la.A_log.data,
                dt_bias=la.dt_bias.data,
                norm_weight=la.norm.weight.data,
            )
        else:
            attn[i] = _attn_from_module(layer.self_attn, i)
        mlp[i] = _mlp_from_module(layer.mlp, i)

    mtp = None
    if getattr(model, "mtp", None) is not None:
        head = model.mtp
        sub = head.layers[0]
        mtp = MTPFusedWeights(
            fc_weight=head.fc.weight.data,
            pre_fc_norm_embedding=head.pre_fc_norm_embedding.weight.data,
            pre_fc_norm_hidden=head.pre_fc_norm_hidden.weight.data,
            norm=head.norm.weight.data,
            input_layernorm=sub.input_layernorm.weight.data,
            post_attention_layernorm=sub.post_attention_layernorm.weight.data,
            attn=_attn_from_module(sub.self_attn, cfg.mtp_layer_idx),
            mlp=_mlp_from_module(sub.mlp, cfg.mtp_layer_idx),
        )

    return FusedModelWeights(
        config=cfg,
        embed_tokens=m.embed_tokens.weight.data,
        lm_head=model.lm_head.weight.data,
        final_norm=m.norm.weight.data,
        layernorms=layernorms,
        gdn=gdn,
        attn=attn,
        mlp=mlp,
        mtp=mtp,
    )


def _attn_from_module(sa, layer_idx: int) -> AttnFusedWeights:
    return AttnFusedWeights(
        layer_idx=layer_idx,
        qkv_proj=torch.cat(
            [sa.q_proj.weight.data, sa.k_proj.weight.data, sa.v_proj.weight.data], dim=0
        ).contiguous(),
        o_proj=sa.o_proj.weight.data,
        q_norm=sa.q_norm.weight.data,
        k_norm=sa.k_norm.weight.data,
    )


def _mlp_from_module(mp, layer_idx: int) -> MLPFusedWeights:
    return MLPFusedWeights(
        layer_idx=layer_idx,
        gate_up_proj=torch.cat([mp.gate_proj.weight.data, mp.up_proj.weight.data], dim=0).contiguous(),
        down_proj=mp.down_proj.weight.data,
    )


def _iter_resolved_linears(model):
    """Every ``ResolvedLinear`` in the stack, including the MTP head's.

    ``bench_runtime.collect_gemm_backends`` walks the same set to report
    resolved backends; this walks it to *cause* a resolution. Kept as one
    generator so the two can never disagree about what "every linear" means.
    """
    for layer in model.layers:
        mixer = layer.mixer
        for attr in ("in_proj_qkvz", "in_proj_ba", "out_proj", "qkv", "o_proj"):
            lin = getattr(mixer, attr, None)
            if isinstance(lin, ResolvedLinear):
                yield lin
        for attr in ("gate_up", "down"):
            lin = getattr(layer.mlp, attr, None)
            if isinstance(lin, ResolvedLinear):
                yield lin
    if isinstance(getattr(model, "lm_head", None), ResolvedLinear):
        yield model.lm_head


def _seq_slots_of(batch: "PrefillBatch") -> List[int]:
    """``batch.seq_slots`` if the batch carries it, else the D2H fallback."""
    slots = getattr(batch, "seq_slots", None)
    return list(slots) if slots else batch.seq_slot_ids.tolist()


def make_prefill_batch(
    token_ids: Sequence[Sequence[int]],
    start_positions: Sequence[int],
    slots: Sequence[int],
    device: torch.device,
) -> PrefillBatch:
    """Pack ``N`` per-sequence token runs into one varlen chunk."""
    flat: List[int] = []
    pos: List[int] = []
    per_token_slots: List[int] = []
    cu = [0]
    q_lens: List[int] = []
    kv_lens: List[int] = []
    last: List[int] = []
    for ids, p0, slot in zip(token_ids, start_positions, slots):
        n = len(ids)
        # `flat.extend(int(t) for t in ids)` would be ~8192 Python-level int()
        # calls per chunk; the ids are already ints (they came out of a tokenizer or
        # `Request.prompt_token_ids`) and `torch.tensor` validates them anyway.
        flat.extend(ids)
        pos.extend(range(p0, p0 + n))
        per_token_slots.extend([slot] * n)
        cu.append(cu[-1] + n)
        q_lens.append(n)
        kv_lens.append(p0 + n)
        last.append(cu[-1] - 1)
    slots = list(slots)
    return PrefillBatch(
        token_ids=torch.tensor(flat, dtype=torch.int32, device=device),
        positions=torch.tensor(pos, dtype=torch.int32, device=device),
        slot_ids=torch.tensor(per_token_slots, dtype=torch.int32, device=device),
        seq_slot_ids=torch.tensor(slots, dtype=torch.int32, device=device),
        cu_seqlens=torch.tensor(cu, dtype=torch.int32, device=device),
        q_lens=q_lens,
        kv_lens=kv_lens,
        last_indices=torch.tensor(last, dtype=torch.long, device=device),
        seq_slots=slots,
    )


def make_mixed_batch(
    prefill: PrefillBatch,
    decode_slots: Sequence[int],
    decode_token_ids: Sequence[int],
    decode_positions: Sequence[int],
    device: torch.device,
) -> MixedBatch:
    """Pack a prefill chunk and ``B_dec`` decode rows into one step.

    ``decode_positions[i]`` is the row's **pre-step** context length ``L``
    (the position the token it is about to emit will occupy), which is exactly
    ``Request.num_computed_tokens``, the same value the plain decode step puts
    in ``buf.positions``.

    Both halves must be non-empty: with no segments there is nothing to give
    the varlen conv/GDN kernels a ``cu_seqlens``, and with no decode rows this
    is just a prefill chunk. The scheduler checks both before calling.
    """
    n_dec = len(decode_slots)
    if n_dec == 0 or not prefill.q_lens:
        raise ValueError(
            "make_mixed_batch: a mixed step needs at least one prefill segment "
            f"and one decode row (got {len(prefill.q_lens)} segments, {n_dec} rows). "
            "Run prefill_forward or the graphed decode step instead."
        )
    tp = int(prefill.token_ids.shape[0])
    dec_slots = [int(s) for s in decode_slots]
    dec_toks = [int(t) for t in decode_token_ids]
    dec_pos = [int(p) for p in decode_positions]
    cat = torch.cat
    token_ids = cat([prefill.token_ids,
                     torch.tensor(dec_toks, dtype=torch.int32, device=device)])
    positions = cat([prefill.positions,
                     torch.tensor(dec_pos, dtype=torch.int32, device=device)])
    slot_ids = cat([prefill.slot_ids,
                    torch.tensor(dec_slots, dtype=torch.int32, device=device)])
    # `last_indices` already indexes each segment's last token inside [0, tp);
    # the decode rows are one row each, in order, immediately after.
    logits_idx = cat([
        prefill.last_indices,
        torch.arange(tp, tp + n_dec, dtype=torch.long, device=device),
    ])
    return MixedBatch(
        prefill=prefill,
        n_prefill_tokens=tp,
        token_ids=token_ids,
        positions=positions,
        slot_ids=slot_ids,
        decode_slot_ids=torch.tensor(dec_slots, dtype=torch.int32, device=device),
        decode_slots=dec_slots,
        decode_positions=dec_pos,
        decode_token_ids=dec_toks,
        plan_slots=_seq_slots_of(prefill) + dec_slots,
        plan_q_lens=list(prefill.q_lens) + [1] * n_dec,
        # Post-write, for both halves: `plan_prefill` runs before any layer
        # appends its K/V, so a decode row planned over `L` keys would attend to
        # everything *except* the token it is generating from (the hazard
        # `plan_prefill`'s docstring describes, in the mixed step's own shape).
        plan_kv_lens=list(prefill.kv_lens) + [p + 1 for p in dec_pos],
        logits_indices=logits_idx,
    )


__all__ = [
    "HAS_TRITON",
    "prefill_gemm_scope",
    "DTYPES",
    "resolve_dtype",
    "norm_weight_1p",
    "rms_norm",
    "rms_norm_w1p",
    "add_rms_norm",
    "add_rms_norm_w1p",
    "rms_norm_gated",
    "swiglu",
    "triton_rms_norm",
    "triton_add_rms_norm",
    "triton_rms_norm_gated",
    "ResolvedLinear",
    "RuntimeConfig",
    "default_claim_gemm_cache_m",
    "claim_gemm_cache_m_for",
    "DeviceBuffers",
    "AttentionRunner",
    "FusedMLP",
    "FusedGDN",
    "FusedAttention",
    "FusedDecoderLayer",
    "FusedMTPHead",
    "StepContext",
    "PrefillBatch",
    "MixedBatch",
    "FusedQwenForCausalLM",
    "fused_weights_from_module",
    "make_prefill_batch",
    "make_mixed_batch",
]
