"""FlashInfer decode/prefill attention, fused pre/post-attention ops, and a
torch/FA3 fallback backend. FlashInfer's paged GQA decode + fp8 KV support
is shape-dependent, so every backend here is measured at exactly our shape
(4 kv-heads, head_dim 256, GQA 6:1).

Everything downstream of q/k/v *projection* lives here: per-head QK-RMSNorm
``(1 + weight)``, partial RoPE over the first 64 of 256 dims, the paged-KV
attention kernel itself, and the post-attention sigmoid output gate. The
projections (``q_proj``/``k_proj``/``v_proj``/``o_proj`` GEMMs) belong to the
GEMM layer (fused weight layout, FP8 dispatch); this module takes
already-projected ``q``/``k``/``v`` tensors.

Numerics are pinned to ``engine/qwenfast/model.py::Attention`` /
``engine/reference/modeling_qwen3_5.py::Qwen3_5Attention`` bit-for-bit
q_norm/k_norm on ``head_dim`` *before* RoPE, RoPE only on the
first 64 dims, output gate is ``sigmoid`` applied *before* ``o_proj``, no
extra float casts beyond what the reference does.

**GQA group size 6 (flashinfer 0.6.16.post3, torch 2.13 cu13, H200):**
``BatchDecodeWithPagedKVCacheWrapper`` at our exact decode shape (24 q-heads /
4 kv-heads -> GQA group_size 6, head_dim 256) fails with
``batch_decode.cu:63: Unsupported group_size: 6`` in *both* f16 and e4m3 KV,
with the wrapper's default (``use_tensor_cores=False``) kernel selection --
the non-tensor-core decode kernel only supports a small enumerated set of
group sizes and 6 isn't one of them. vLLM avoids this by always going through
the tensor-core kernel (or FlashAttention-3) for GQA. Three independent
decode paths, all supporting arbitrary GQA group sizes at head_dim 256, are
provided below -- ``DECODE_BACKENDS`` and ``bench_attn.py`` exercise all of
them so the choice is made on measured numbers, not assumption:

1. ``FlashInferDecodeAttention`` with ``use_tensor_cores=True`` (**default**,
   changed from the library's own default specifically because of the finding
   above) -- routes through FlashInfer's tensor-core / prefill-style decode
   kernel, which supports arbitrary GQA groups.
2. ``FlashInferPrefillAttention`` reused *for decode*, with
   ``qo_indptr = decode_qo_indptr(B)`` (``arange(B+1)``, i.e. "every request
   contributes one query token") -- the standard "prefill kernel as decode"
   workaround vLLM itself uses when the dedicated decode kernel can't handle
   a shape. ``BatchPrefillWithPagedKVCacheWrapper`` has no group-size
   restriction.
3. ``fa3_decode_with_kvcache`` -- FlashAttention-3's paged-KV path via
   ``flash_attn_varlen_func(..., block_table=..., seqused_k=...)``, decode
   framed as "``max_seqlen_q=1``, one query token per sequence". Tried via
   ``vllm_flash_attn`` first (ships with vLLM's own FA3 build, confirmed
   against its ``flash_attn_interface.py`` signature), falling back to a
   standalone ``flash_attn_interface`` install if present.

Plus the always-available reference/fallback:

4. ``torch.nn.functional.scaled_dot_product_attention`` over pages gathered
   dense -- always available (CPU or GPU), used by the correctness tests and
   as the last-resort backend. Never the benchmarked "production" path.

Every GPU-only code path guards its import inside the function/class that
needs it, so this module always imports cleanly on a CPU-only host (checked by
``python -m py_compile`` and by importing it in the CPU test suite).
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .kv_pool import PagedKVPool
from .rope import apply_rotary_pos_emb

#: Kill switch for the host-side decode plan fast path in
#: ``FlashInferDecodeAttention.plan`` -- set ``QWENFAST_FI_HOST_PLAN=0`` to
#: fall back to the device round trip if a future FlashInfer changes what
#: ``plan()`` does with its ``indptr``/``last_page_len`` arguments.  The path
#: is guarded by an exact shape/device match, so it simply never engages for
#: any caller that does not opt in.
_FI_HOST_PLAN = os.environ.get("QWENFAST_FI_HOST_PLAN", "1") not in ("0", "false", "False")

# --------------------------------------------------------------------------- #
# optional GPU backends -- import-guarded, never at module scope beyond this
# --------------------------------------------------------------------------- #
try:
    import flashinfer  # type: ignore

    HAS_FLASHINFER = True
except Exception:  # pragma: no cover - depends on the host
    flashinfer = None  # type: ignore
    HAS_FLASHINFER = False

# FA3's varlen entry point (``flash_attn_varlen_func``) is the same function
# for plain packed-varlen prefill *and* paged-KV attention (decode included)
# -- passing ``block_table``/``seqused_k`` switches it into paged mode. Two
# possible sources ship it: vLLM's own fork (``vllm_flash_attn``, installed
# alongside vLLM's FA3 build -- this is what the coordinator's H200 box has)
# and the standalone Dao-AILab Hopper build (``flash_attn_interface``). Prefer
# the former (its signature is confirmed against source), fall back to the
# latter (best-effort -- its paged-KV kwarg names are not independently
# verified here, see README "Known gaps").
_fa3_varlen_fn = None
_FA3_SOURCE = None
try:
    from vllm_flash_attn import flash_attn_varlen_func as _fa3_varlen_fn  # type: ignore

    _FA3_SOURCE = "vllm_flash_attn"
except Exception:  # pragma: no cover - depends on the host
    try:
        import flash_attn_interface  # type: ignore

        _fa3_varlen_fn = getattr(flash_attn_interface, "flash_attn_varlen_func", None)
        if _fa3_varlen_fn is not None:
            _FA3_SOURCE = "flash_attn_interface"
    except Exception:
        flash_attn_interface = None  # type: ignore
HAS_FA3 = _fa3_varlen_fn is not None

try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore

    HAS_TRITON = True
except Exception:  # pragma: no cover - depends on the host
    triton = None  # type: ignore
    HAS_TRITON = False


# =========================================================================== #
# fused pre-/post-attention ops (torch; canonical, CPU-testable path)
# =========================================================================== #
def rms_norm_head(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """QK-RMSNorm over the last (head_dim) axis, scale by ``(1 + weight)``.

    Bit-identical to ``engine/qwenfast/model.py::RMSNorm`` /
    ``modeling_qwen3_5.Qwen3_5RMSNorm`` -- normalize in fp32, scale, cast back.
    ``x``: ``[..., head_dim]`` (any leading shape: ``[T, H, D]`` packed or
    ``[B, S, H, D]`` dense both work, the reduction is always over the last
    dim only: the norm is on head_dim, before RoPE).
    """
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    out = out * (1.0 + weight.float())
    return out.type_as(x)


def split_q_gate(qg: torch.Tensor, num_heads: int, head_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split fused ``q_proj`` output into ``(q, gate)``.

    ``qg``: ``[T, num_heads * 2 * head_dim]``, laid out head-major
    ``[h0_q(D) | h0_gate(D) | h1_q(D) | ...]``. Returns
    ``q: [T, num_heads, head_dim]``, ``gate: [T, num_heads * head_dim]`` --
    matches ``model.py::Attention.forward``'s
    ``.view(b, s, -1, 2*head_dim).chunk(2, -1)`` with ``b*s`` collapsed to ``T``.
    """
    t = qg.shape[0]
    qg = qg.view(t, num_heads, 2 * head_dim)
    q, gate = qg.chunk(2, dim=-1)
    return q, gate.reshape(t, num_heads * head_dim)


def apply_output_gate(attn_out: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """``attn_out * sigmoid(gate)``, applied *before* ``o_proj``.

    No extra float upcast -- matches ``model.py`` / the reference exactly
    (``torch.sigmoid(gate)`` operates in ``gate``'s own dtype, typically bf16).
    """
    return attn_out * torch.sigmoid(gate)


def fused_qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """QK-RMSNorm then partial RoPE, on packed ``[T, H, D]`` q/k.

    ``cos``/``sin``: ``[T, rotary_dim]``. Composes ``rms_norm_head`` with
    ``rope.apply_rotary_pos_emb`` (``unsqueeze_dim=1`` broadcasts over the
    head axis of a ``[T, H, D]`` tensor -- see that function's docstring).
    This is the exact op sequence ``model.py::Attention.forward`` runs
    between projection and KV-cache append, just without the ``[B, S, ...]``
    reshape (packed varlen and single-token decode are both just "T tokens").
    """
    q = rms_norm_head(q, q_norm_weight, eps)
    k = rms_norm_head(k, k_norm_weight, eps)
    q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
    return q, k


# --------------------------------------------------------------------------- #
# optional Triton fused partial-RoPE kernel (best-effort; unverified without
# a local GPU -- guarded, never the default path, torch above is canonical)
# --------------------------------------------------------------------------- #
if HAS_TRITON:

    @triton.jit
    def _partial_rope_kernel(
        X_ptr, COS_ptr, SIN_ptr, OUT_ptr,
        stride_xt, stride_xh, stride_xd,
        stride_ct, stride_cd,
        stride_ot, stride_oh, stride_od,
        n_heads: tl.constexpr,
        head_dim: tl.constexpr,
        rotary_dim: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """One program per (token, head): rotate the first ``rotary_dim`` of
        ``head_dim``, copy the rest through untouched. ``rotary_dim/2`` is
        assumed <= ``BLOCK_D``."""
        pid_t = tl.program_id(0)
        pid_h = tl.program_id(1)
        half = rotary_dim // 2
        offs = tl.arange(0, BLOCK_D)
        mask = offs < half

        x_base = X_ptr + pid_t * stride_xt + pid_h * stride_xh
        x1 = tl.load(x_base + offs * stride_xd, mask=mask, other=0.0)
        x2 = tl.load(x_base + (offs + half) * stride_xd, mask=mask, other=0.0)

        c_base = COS_ptr + pid_t * stride_ct
        s_base = SIN_ptr + pid_t * stride_ct
        cos1 = tl.load(c_base + offs * stride_cd, mask=mask, other=0.0)
        sin1 = tl.load(s_base + offs * stride_cd, mask=mask, other=0.0)
        # cos/sin tables are built as cat(freqs, freqs), so the second half
        # of the table repeats the first half's angles.
        cos2 = cos1
        sin2 = sin1

        o1 = x1 * cos1 - x2 * sin1
        o2 = x2 * cos2 + x1 * sin2

        o_base = OUT_ptr + pid_t * stride_ot + pid_h * stride_oh
        tl.store(o_base + offs * stride_od, o1, mask=mask)
        tl.store(o_base + (offs + half) * stride_od, o2, mask=mask)

        # pass-through tail (rotary_dim .. head_dim)
        tail_offs = rotary_dim + tl.arange(0, BLOCK_D)
        tail_mask = tail_offs < head_dim
        tail = tl.load(x_base + tail_offs * stride_xd, mask=tail_mask, other=0.0)
        tl.store(o_base + tail_offs * stride_od, tail, mask=tail_mask)

    def triton_partial_rope(
        x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int
    ) -> torch.Tensor:
        """``x``: ``[T, H, D]``, ``cos``/``sin``: ``[T, rotary_dim]``. GPU only;
        not exercised by the CPU test suite (no Triton CUDA backend on a Mac).
        Numerically equivalent to ``apply_rotary_pos_emb`` restricted to one
        of ``q``/``k`` -- call twice for q and k."""
        t, h, d = x.shape
        out = torch.empty_like(x)
        block_d = max(16, triton.next_power_of_2(max(rotary_dim // 2, d - rotary_dim)))
        grid = (t, h)
        _partial_rope_kernel[grid](
            x, cos, sin, out,
            x.stride(0), x.stride(1), x.stride(2),
            cos.stride(0), cos.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            n_heads=h, head_dim=d, rotary_dim=rotary_dim, BLOCK_D=block_d,
        )
        return out

else:

    def triton_partial_rope(*args, **kwargs):  # pragma: no cover
        raise RuntimeError("triton is not importable on this host; use fused_qk_norm_rope (torch) instead")


# =========================================================================== #
# KV-append convenience wrapper (thin composition over kv_pool.append_kv)
# =========================================================================== #
def append_kv_to_pool(
    pool: PagedKVPool,
    layer: int,
    slot_ids: torch.Tensor,
    positions: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    """``k``, ``v``: ``[T, num_kv_heads, head_dim]``, already normed+roped
    (k) / raw (v -- v never gets normed or roped). Thin pass-through to
    ``PagedKVPool.append_kv``, kept as a separate function so the "KV append
    into the pool" step has a named entry point distinct from the pool's own
    method."""
    pool.append_kv(layer, slot_ids, positions, k, v)


# =========================================================================== #
# torch fallback backend (gathered pages + SDPA) -- correctness reference
# =========================================================================== #
def torch_fallback_decode(
    pool: PagedKVPool,
    layer: int,
    slot_ids: torch.Tensor,
    q: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """One decode step, ``B`` sequences, via dense-gather + SDPA.

    ``q``: ``[B, num_qo_heads, head_dim]`` (already normed+roped). Gathers
    each sequence's full K/V from the pool (dequantized to bf16 if the pool
    is fp8) and runs ``F.scaled_dot_product_attention`` with no mask (every
    gathered token is causally valid by construction -- they are exactly the
    committed prefix). ``O(B * max_ctx)`` gather; this is the reference path
    for the CPU test suite, never the hot path.
    """
    num_kv_groups = q.shape[1] // pool.cfg.num_kv_heads
    outs = []
    for i, slot in enumerate(slot_ids.tolist()):
        k, v = pool.gather_dense(layer, slot)  # [ctx, Hkv, D] bf16
        k = k.transpose(0, 1)  # [Hkv, ctx, D]
        v = v.transpose(0, 1)
        if num_kv_groups > 1:
            k = k.repeat_interleave(num_kv_groups, dim=0)
            v = v.repeat_interleave(num_kv_groups, dim=0)
        qi = q[i].unsqueeze(1).to(torch.float32)  # [Hq, 1, D]
        oi = F.scaled_dot_product_attention(qi, k.float(), v.float(), is_causal=False, scale=scaling)
        outs.append(oi.squeeze(1).to(q.dtype))
    return torch.stack(outs, dim=0)  # [B, Hq, D]


def torch_fallback_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scaling: float,
    num_kv_groups: int = 1,
    causal: bool = True,
) -> torch.Tensor:
    """Varlen causal SDPA over a packed batch, no paging.

    ``q``: ``[T, Hq, D]``, ``k``/``v``: ``[T, Hkv, D]`` (already normed+roped
    q/k). ``cu_seqlens``: ``[B+1]`` int, prefix sums of per-sequence lengths
    (``cu_seqlens[0] == 0``). Used both as the prefill fallback backend and
    as the correctness reference in ``tests/test_attn.py`` (compared against
    ``model.py``'s dense left-padded ``Attention`` on the same random data).
    """
    b = cu_seqlens.numel() - 1
    outs = []
    for i in range(b):
        s, e = int(cu_seqlens[i]), int(cu_seqlens[i + 1])
        qi = q[s:e].transpose(0, 1).float()  # [Hq, L, D]
        ki = k[s:e].transpose(0, 1).float()  # [Hkv, L, D]
        vi = v[s:e].transpose(0, 1).float()
        if num_kv_groups > 1:
            ki = ki.repeat_interleave(num_kv_groups, dim=0)
            vi = vi.repeat_interleave(num_kv_groups, dim=0)
        oi = F.scaled_dot_product_attention(qi, ki, vi, is_causal=causal, scale=scaling)
        outs.append(oi.transpose(0, 1).to(q.dtype))  # [L, Hq, D]
    return torch.cat(outs, dim=0)


# =========================================================================== #
# FlashAttention-3 varlen fallback (Hopper) -- best-effort
# =========================================================================== #
def fa3_varlen_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    scaling: Optional[float] = None,
    causal: bool = True,
) -> torch.Tensor:
    """Plain packed-varlen prefill via ``flash_attn_varlen_func`` (no paged
    KV -- ``k``/``v`` are the packed ``[total_k, Hkv, D]`` tensors for this
    chunk, same convention as ``torch_fallback_prefill``). GPU (Hopper) only;
    raises if no FA3 build is importable -- callers should check ``HAS_FA3``
    first."""
    if not HAS_FA3:
        raise RuntimeError(f"no FlashAttention-3 build importable on this box (tried vllm_flash_attn, flash_attn_interface)")
    out = _fa3_varlen_fn(
        q, k, v,
        max_seqlen, cu_seqlens,
        max_seqlen, cu_seqlens,
        causal=causal,
        softmax_scale=scaling,
    )
    # FA3 returns (out, softmax_lse[, ...]) as a tuple unless return_softmax_lse=False
    # short-circuits to just `out` (varies by build); handle both.
    return out[0] if isinstance(out, tuple) else out


def fa3_decode_with_kvcache(
    pool: PagedKVPool,
    layer: int,
    slot_ids: torch.Tensor,
    q: torch.Tensor,
    scaling: float,
    causal: bool = True,
) -> torch.Tensor:
    """Decode via FlashAttention-3's paged-KV path -- the fix for
    ``BatchDecodeWithPagedKVCacheWrapper``'s ``Unsupported group_size: 6``
    (see module docstring). Frames decode as a degenerate varlen call: one
    query token per sequence (``max_seqlen_q=1``, ``cu_seqlens_q=arange(B+1)``),
    reading straight out of the pool's paged K/V storage via ``block_table``
    (``seqused_k`` gives the valid length per sequence, so ``max_seqlen_k`` can
    just be the widest allocated context).

    ``q``: ``[B, num_qo_heads, head_dim]`` (already normed+roped). Reuses
    ``PagedKVPool.block_table_for`` (page indices, -1 rows clamped to 0 --
    the kernel never reads past ``seqused_k`` so the clamp is only to avoid
    an out-of-range address, not a correctness issue).
    """
    if not HAS_FA3:
        raise RuntimeError(f"no FlashAttention-3 build importable on this box (tried vllm_flash_attn, flash_attn_interface)")
    b = slot_ids.shape[0]
    device = q.device
    block_table = pool.block_table_for(slot_ids)  # [B, max_pages_per_seq] int32, clamped >= 0
    seq_lens = pool.seq_len[slot_ids.long()].to(torch.int32)  # [B]
    cu_seqlens_q = torch.arange(0, b + 1, dtype=torch.int32, device=device)
    max_seqlen_k = block_table.shape[1] * pool.cfg.page_size
    k_cache = pool.kv[layer, :, 0]  # [n_pages, page_size, num_kv_heads, head_dim]
    v_cache = pool.kv[layer, :, 1]
    out = _fa3_varlen_fn(
        q, k_cache, v_cache,
        1, cu_seqlens_q,
        max_seqlen_k, None,
        seqused_k=seq_lens,
        block_table=block_table,
        causal=causal,
        softmax_scale=scaling,
    )
    return out[0] if isinstance(out, tuple) else out


# --------------------------------------------------------------------------- #
# JIT-compile diagnostic: a system nvcc older than the CUDA version torch was
# built for (e.g. nvcc 12.6 with a cu13 torch) breaks JIT builds with "NVCC
# compilation failed". FlashInfer JIT-compiles its kernels through an
# nvcc/ninja subprocess, so the same mismatch surfaces as uniform failures
# across every FlashInfer-backed call. This wrapper doesn't fix the mismatch
# (that's an environment/toolchain issue outside this package's scope); it
# just makes the failure legible instead of a bare nvcc/ninja stack trace
# three frames deep inside flashinfer's JIT machinery.
# --------------------------------------------------------------------------- #
@contextmanager
def _flashinfer_jit_hint():
    try:
        yield
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if any(tok in msg for tok in ("nvcc", "ninja", "NVCC", "JITCompil", "CalledProcessError", "compil")):
            raise RuntimeError(
                "FlashInfer kernel JIT compilation appears to have failed "
                f"(original: {type(exc).__name__}: {exc}). A common cause is an nvcc/CUDA "
                "mismatch: system nvcc may be older than the CUDA "
                "version torch was built for. Try: pip install nvidia-cuda-nvcc-cu13 && "
                "export CUDA_HOME=<that package's install dir>, then re-run."
            ) from exc
        raise


# =========================================================================== #
# FlashInfer wrappers: persistent workspace, plan() outside / run() inside
# the CUDA graph (FlashInfer 0.6.x use_cuda_graph=True contract:
# fixed-size paged_kv_{indptr,indices,last_page_len}_buffer storage, whose
# *contents* -- not shape -- may change between plan() calls; see
# https://docs.flashinfer.ai/api/attention.html (BatchDecodeWithPagedKVCacheWrapper),
# https://docs.flashinfer.ai/generated/flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper.html)
# =========================================================================== #
class FlashInferDecodeAttention:
    """One instance per CUDA-graph batch-size bucket: the
    persistent ``paged_kv_indptr/indices/last_page_len`` buffers are sized
    for that bucket's ``max_batch_size``/``max_pages``, and never resized.

    Usage per step:

    1. (host, outside the graph) build the *ragged* ``kv_indptr``/``kv_indices``/
       ``kv_last_page_len`` for the live batch, padded up to the bucket size
       with empty/scratch entries (padded rows point at a scratch SSM slot
       and an empty KV page).
    2. ``plan(...)`` -- copies those into the persistent buffers and calls
       FlashInfer's own ``plan()``. Must run outside graph capture/replay:
       FlashInfer's planning step does host-driven dispatch (choosing a
       split-KV strategy etc.) that is not itself graph-capturable.
    3. ``run(q, paged_kv_cache)`` -- pure kernel launch against the buffers
       ``plan()`` just populated; capture/replay this inside the graph.
    """

    def __init__(
        self,
        workspace_buffer: "torch.Tensor",
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        max_batch_size: int,
        max_pages: int,
        kv_dtype: "torch.dtype" = None,
        q_dtype: "torch.dtype" = None,
        use_cuda_graph: bool = True,
        use_tensor_cores: bool = True,
        device: str = "cuda",
    ):
        if not HAS_FLASHINFER:
            raise RuntimeError("flashinfer is not importable on this host")
        kv_dtype = kv_dtype or torch.bfloat16
        q_dtype = q_dtype or torch.bfloat16
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.kv_dtype = kv_dtype
        self.q_dtype = q_dtype
        self.max_batch_size = max_batch_size
        self.max_pages = max_pages
        self.use_cuda_graph = use_cuda_graph
        self.use_tensor_cores = use_tensor_cores

        if use_cuda_graph:
            self.indptr_buf = torch.zeros(max_batch_size + 1, dtype=torch.int32, device=device)
            self.indices_buf = torch.zeros(max_pages, dtype=torch.int32, device=device)
            self.last_page_len_buf = torch.zeros(max_batch_size, dtype=torch.int32, device=device)
        else:
            self.indptr_buf = self.indices_buf = self.last_page_len_buf = None

        # use_tensor_cores=True is required at our shape (24 q-heads / 4
        # kv-heads -> GQA group_size 6, head_dim 256): the non-tensor-core
        # decode kernel raises "Unsupported group_size: 6" on H200/flashinfer
        # 0.6.16.post3 (both bf16 and fp8 KV) -- see module docstring.
        with _flashinfer_jit_hint():
            self.wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                workspace_buffer,
                "NHD",
                use_cuda_graph=use_cuda_graph,
                use_tensor_cores=use_tensor_cores,
                paged_kv_indptr_buffer=self.indptr_buf,
                paged_kv_indices_buffer=self.indices_buf,
                paged_kv_last_page_len_buffer=self.last_page_len_buf,
            )

    def plan(self, kv_indptr: "torch.Tensor", kv_indices: "torch.Tensor", kv_last_page_len: "torch.Tensor") -> None:
        """Outside the graph. ``kv_indptr``: ``[B+1]``, already padded to
        ``max_batch_size + 1`` by the caller when ``use_cuda_graph``."""
        if self.use_cuda_graph:
            # Slice to the *incoming* tensor's length rather than a bare
            # `.copy_()` into the whole buffer: the latter raises a
            # shape-mismatch RuntimeError whenever the live batch B is
            # smaller than max_batch_size. This slice only prevents the
            # crash; it does NOT by itself make an unpadded call correct.
            # The graph replays against fixed device buffers written by the
            # scheduler, so the *caller* owns padding a live batch up to the
            # bucket size before calling plan() in graph mode. Indptr-style
            # arrays (this one, kv_indptr) must be padded by *repeating the
            # last cumulative value* into the tail (monotonic, zero-length
            # rows), not left at the buffer's zero-initialized default,
            # which would make the array non-monotonic. See
            # tests/test_attn.py's cuda-graph-buffer test for a worked
            # example of correct padding.
            if (
                _FI_HOST_PLAN
                and kv_indptr.device.type == "cpu"
                and kv_last_page_len.device.type == "cpu"
                and kv_indices.device.type == "cuda"
                and kv_indptr.numel() == self.indptr_buf.numel()
                and kv_last_page_len.numel() == self.last_page_len_buf.numel()
                and kv_indices.numel() <= self.indices_buf.numel()
            ):
                # FlashInfer's own ``plan()`` does
                #     indptr_host = indptr.to("cpu")
                #     last_page_len_host = last_page_len.to("cpu")
                # -- it needs both arrays **on the host** to build the plan.
                # Copying them to the device here (the branch below) and
                # letting FlashInfer copy them straight back is a full H2D +
                # D2H round trip per decode step, and the D2H half is a
                # ``cudaStreamSynchronize`` on a stream that, under
                # ``--async-scheduling``, has the previous step's whole
                # 44 ms of kernels on it.  So when the caller hands us pinned
                # host arrays of exactly the persistent buffers' shapes
                # (``PagedKVPool.build_flashinfer_indices(staged=True)``) we
                # pass them through untouched: FlashInfer copies them into
                # *our* ``indptr_buf``/``last_page_len_buf`` itself (they are
                # the buffers it was constructed with) with
                # ``non_blocking=True``, and reads the host originals for the
                # plan.  ``kv_indices`` stays on the device because
                # FlashInfer's indices copy is only non-blocking for a
                # device source.
                indptr, indices, last_page_len = kv_indptr, kv_indices, kv_last_page_len
            else:
                # Slice to the *incoming* tensor's length rather than a bare
                # `.copy_()` into the whole buffer -- see the docstring above.
                self.indptr_buf[: kv_indptr.numel()].copy_(kv_indptr)
                self.indices_buf[: kv_indices.numel()].copy_(kv_indices)
                self.last_page_len_buf[: kv_last_page_len.numel()].copy_(kv_last_page_len)
                indptr, indices, last_page_len = self.indptr_buf, self.indices_buf, self.last_page_len_buf
        else:
            indptr, indices, last_page_len = kv_indptr, kv_indices, kv_last_page_len
        with _flashinfer_jit_hint():
            self.wrapper.plan(
                indptr,
                indices,
                last_page_len,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                self.page_size,
                pos_encoding_mode="NONE",  # RoPE already applied by fused_qk_norm_rope
                q_data_type=self.q_dtype,
                kv_data_type=self.kv_dtype,
            )

    def run(
        self,
        q: "torch.Tensor",
        paged_kv_cache: "torch.Tensor",
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
    ) -> "torch.Tensor":
        """Inside the graph. ``q``: ``[B, num_qo_heads, head_dim]``;
        ``paged_kv_cache``: one layer's slice of the pool,
        ``[n_pages, 2, page_size, num_kv_heads, head_dim]``.

        ``k_scale``/``v_scale``: **required** (non-``None``) whenever
        ``paged_kv_cache`` is fp8 (``kv_dtype`` is ``e4m3``) and the pool's
        calibrated scale for the pages this call reads isn't exactly 1.0.
        ``BatchDecodeWithPagedKVCacheWrapper.run()``'s ``k_scale``/``v_scale``
        default to ``1.0`` when omitted (flashinfer 0.6.16 source), so an
        fp8-KV decode call without them silently skips dequantization
        scaling, while the torch fallback (``gather_dense``) multiplies by the
        pool's real per-page/per-head scale.
        ``test_decode_matches_torch_fallback_fp8_kv`` guards this.
        FlashInfer's ``k_scale``/``v_scale`` are each a single Python
        ``float`` per call (not per-page, not per-head) -- see
        ``PagedKVPool.calibrate_uniform_scale``'s docstring for how to
        calibrate a scale that's actually representable here.
        """
        return self.wrapper.run(q, paged_kv_cache, k_scale=k_scale, v_scale=v_scale)


def decode_qo_indptr(batch_size: int, device: "torch.device | str" = "cuda") -> "torch.Tensor":
    """``arange(0, B+1)`` -- the ``qo_indptr`` that turns a *decode* step
    (one query token per live sequence) into a degenerate *prefill* call:
    query ``i`` occupies ``[i, i+1)``, i.e. every request contributes exactly
    one query token. This is the "route decode through the prefill kernel"
    workaround for ``BatchDecodeWithPagedKVCacheWrapper``'s GQA group-size
    restriction (module docstring, fix #2) -- pass this as ``qo_indptr`` to
    ``FlashInferPrefillAttention.plan`` with ``causal=True`` and a ``q`` of
    shape ``[B, num_qo_heads, head_dim]``, identical to the decode wrappers'
    ``q`` convention.
    """
    return torch.arange(0, batch_size + 1, dtype=torch.int32, device=device)


def pad_indptr_to_bucket(indptr: "torch.Tensor", bucket_size: int) -> "torch.Tensor":
    """Pad an indptr-style CSR row-pointer array (monotonic non-decreasing,
    e.g. ``qo_indptr`` or ``kv_indptr``) from ``[B+1]`` up to ``[bucket_size+1]``
    by *repeating its last value*, not zero-filling.

    This is the correct way to pad a ``kv_indptr``-shaped array for a live
    batch ``B < bucket_size`` when the *corresponding* per-row query count is
    also zero for the padded rows: the padded rows get ``indptr[i] ==
    indptr[i+1]`` (zero-length, contribute nothing), and the array stays
    monotonic -- naively leaving a persistent buffer's zero-initialized tail
    in place instead breaks monotonicity as soon as the real prefix ends
    above 0. Padding a bucket is the *caller*'s (scheduler's) job, not the
    wrapper's.

    **Do NOT use this to pad ``qo_indptr`` for a graphed decode-via-prefill
    call** (``FlashInferPrefillAttention`` with ``use_cuda_graph=True``,
    module docstring fix #2) whose ``q`` buffer is bucket-sized (fixed shape
    for CUDA-graph replay, one row per bucket slot including padding rows).
    Repeating the live batch's last cumulative value makes
    ``qo_indptr[-1] == B`` (the *live* count) while ``q.shape[0] ==
    bucket_size`` -- FlashInfer's ``run()`` requires them equal and raises
    ``ValueError: q.shape[0] (bucket_size) does not match qo_indptr[-1]
    (B)`` (``test_decode_via_prefill_kernel_supports_cuda_graph_buffers``
    covers this). Every bucket row -- including padding -- carries exactly one
    live query token in that usage, so ``qo_indptr`` must be
    ``decode_qo_indptr(bucket_size)`` (arange over the *whole* bucket), and
    the padded rows' KV range must be a real, non-empty "scratch" page (not
    zero-length -- a query row with zero valid keys under ``causal=True`` is
    itself invalid) -- exactly ``runtime/fused_model.py``'s
    ``AttentionRunner.plan_decode`` / ``FusedModel.scratch_slot`` convention
    ("padded rows point at the scratch slot, which owns one page and has
    seq_len 1"): pad the *slot list* with a real scratch slot and rebuild
    indices via ``PagedKVPool.build_flashinfer_indices`` over the padded
    list, rather than padding a raw indptr array after the fact.
    """
    n = bucket_size + 1 - indptr.numel()
    if n <= 0:
        return indptr
    pad = indptr[-1].expand(n)
    return torch.cat([indptr, pad])


class FlashInferPrefillAttention:
    """``BatchPrefillWithPagedKVCacheWrapper`` wrapper for varlen causal
    prefill writing straight into the paged pool, *and* --
    via ``qo_indptr = decode_qo_indptr(B)`` -- the GQA-group-size-6 decode
    workaround described in the module docstring (fix #2). Prefill itself is
    the *ungraphed* path, so
    ``use_cuda_graph`` defaults to ``False``; pass ``use_cuda_graph=True`` (and
    ``max_batch_size``/``max_pages``) when this instance is being used as a
    graphed *decode* backend instead, mirroring
    ``FlashInferDecodeAttention``'s persistent-buffer plan()-outside/run()-inside
    split.
    """

    def __init__(
        self,
        workspace_buffer: "torch.Tensor",
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        kv_dtype: "torch.dtype" = None,
        q_dtype: "torch.dtype" = None,
        use_cuda_graph: bool = False,
        max_batch_size: Optional[int] = None,
        max_pages: Optional[int] = None,
        device: str = "cuda",
    ):
        if not HAS_FLASHINFER:
            raise RuntimeError("flashinfer is not importable on this host")
        if use_cuda_graph and (max_batch_size is None or max_pages is None):
            raise ValueError("use_cuda_graph=True requires max_batch_size and max_pages")
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.kv_dtype = kv_dtype or torch.bfloat16
        self.q_dtype = q_dtype or torch.bfloat16
        self.use_cuda_graph = use_cuda_graph

        if use_cuda_graph:
            self.qo_indptr_buf = torch.zeros(max_batch_size + 1, dtype=torch.int32, device=device)
            self.kv_indptr_buf = torch.zeros(max_batch_size + 1, dtype=torch.int32, device=device)
            self.kv_indices_buf = torch.zeros(max_pages, dtype=torch.int32, device=device)
            self.kv_last_page_len_buf = torch.zeros(max_batch_size, dtype=torch.int32, device=device)
        else:
            self.qo_indptr_buf = self.kv_indptr_buf = self.kv_indices_buf = self.kv_last_page_len_buf = None

        with _flashinfer_jit_hint():
            self.wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                workspace_buffer,
                "NHD",
                use_cuda_graph=use_cuda_graph,
                qo_indptr_buf=self.qo_indptr_buf,
                paged_kv_indptr_buf=self.kv_indptr_buf,
                paged_kv_indices_buf=self.kv_indices_buf,
                paged_kv_last_page_len_buf=self.kv_last_page_len_buf,
            )

    def plan(
        self,
        qo_indptr: "torch.Tensor",
        kv_indptr: "torch.Tensor",
        kv_indices: "torch.Tensor",
        kv_last_page_len: "torch.Tensor",
        causal: bool = True,
    ) -> None:
        """Outside the graph when ``use_cuda_graph`` -- see
        ``FlashInferDecodeAttention.plan``'s docstring for the split
        rationale, identical here."""
        if self.use_cuda_graph:
            # Same slice-not-crash fix as FlashInferDecodeAttention.plan()
            # above -- see that method's comment. Caller still owns correct
            # padding (repeat-last-value for qo_indptr/kv_indptr) when the
            # live batch is smaller than max_batch_size.
            self.qo_indptr_buf[: qo_indptr.numel()].copy_(qo_indptr)
            self.kv_indptr_buf[: kv_indptr.numel()].copy_(kv_indptr)
            self.kv_indices_buf[: kv_indices.numel()].copy_(kv_indices)
            self.kv_last_page_len_buf[: kv_last_page_len.numel()].copy_(kv_last_page_len)
            qo_indptr = self.qo_indptr_buf
            kv_indptr, kv_indices, kv_last_page_len = (
                self.kv_indptr_buf, self.kv_indices_buf, self.kv_last_page_len_buf,
            )
        with _flashinfer_jit_hint():
            self.wrapper.plan(
                qo_indptr,
                kv_indptr,
                kv_indices,
                kv_last_page_len,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                self.page_size,
                causal=causal,
                pos_encoding_mode="NONE",
                q_data_type=self.q_dtype,
                kv_data_type=self.kv_dtype,
            )

    def run(
        self,
        q: "torch.Tensor",
        paged_kv_cache: "torch.Tensor",
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
    ) -> "torch.Tensor":
        """``q``: ``[T, num_qo_heads, head_dim]`` packed varlen queries (for
        the decode-workaround usage, ``T == B``, one token per sequence).

        ``k_scale``/``v_scale``: same contract as
        ``FlashInferDecodeAttention.run`` -- required for a non-1.0 fp8 pool
        scale; a single float per call, see
        ``PagedKVPool.calibrate_uniform_scale``.
        """
        return self.wrapper.run(q, paged_kv_cache, k_scale=k_scale, v_scale=v_scale)


# Every decode path that supports our GQA group_size (6) at head_dim 256 --
# see module docstring. Order is a mild preference (tensor-core FlashInfer
# first: same library as prefill, least new surface), not a strict priority;
# bench_attn.py measures all available ones per cell.
DECODE_BACKENDS = ("flashinfer_decode_tc", "flashinfer_prefill_as_decode", "fa3_kvcache", "torch")


__all__ = [
    "HAS_FLASHINFER",
    "HAS_FA3",
    "HAS_TRITON",
    "DECODE_BACKENDS",
    "pad_indptr_to_bucket",
    "rms_norm_head",
    "split_q_gate",
    "apply_output_gate",
    "fused_qk_norm_rope",
    "triton_partial_rope",
    "append_kv_to_pool",
    "torch_fallback_decode",
    "torch_fallback_prefill",
    "fa3_varlen_prefill",
    "fa3_decode_with_kvcache",
    "decode_qo_indptr",
    "FlashInferDecodeAttention",
    "FlashInferPrefillAttention",
]
