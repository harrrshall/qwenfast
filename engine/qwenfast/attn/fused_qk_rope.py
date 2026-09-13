"""Fused per-head QK-RMSNorm + partial RoPE, one Triton kernel.

Profiled on a real 8,192-token prefill chunk, `attn.qk_norm_rope`
(`flashinfer_attn.rms_norm_head` x2 + `rope.apply_rotary_pos_emb`, all eager
torch) costs **23.0 ms across 16 layers (4.1% of the 563 ms chunk)**. That is
~9 eager kernel launches per
q/k-norm call plus ~6 more for the rotate-half RoPE, all working on the same
``[T, H, 256]`` row: the RMSNorm reduction is *entirely redundant* with the
first 64 of those 256 dims also being read, permuted and rewritten a few
launches later. This module fuses all of it -- reduction, scale, and the
rotate-half RoPE on the first ``rotary_dim`` dims -- into one Triton kernel
launch per tensor (q, k), one program per ``(token, head)``.

Numerics are pinned to ``attn.flashinfer_attn.fused_qk_norm_rope`` (itself
pinned to ``model.py::Attention``, the dense reference model):

* RMSNorm reduces over the **full** ``head_dim`` (256), in fp32, scaled by
  ``(1 + weight)`` -- *not* plain ``weight`` (that convention is
  ``RMSNormGated``'s, a different op; see
  ``fused_model.norm_weight_1p``'s docstring for why mixing the two is "a very
  easy bug").
* RoPE rotates only the first ``rotary_dim`` (64) of the *normalised* 256 dims
  (q/k-norm runs **before** RoPE) and passes the remaining
  192 through unrotated (still normalised).
* The rotate-half identity, from ``rope.apply_rotary_pos_emb`` /
  ``model.rotate_half``, with ``half = rotary_dim // 2`` and the table built as
  ``cat(freqs, freqs)`` (so ``cos[i] == cos[i + half]`` for ``i < half``,
  ``rope.RotaryTable.__init__``):

      out[i]        = x[i]*cos[i]        - x[i+half]*sin[i]         (i < half)
      out[i+half]   = x[i+half]*cos[i+half] + x[i]*sin[i+half]      (i < half)

  computed here as one masked, gathered pair-load per lane rather than the
  reference's two ``[..., :half]`` / ``[..., half:]`` slices + a ``cat`` --
  see ``_qk_norm_rope_kernel``.

Only the **prefill/mixed** path uses this (attention prefill is never
CUDA-graph-captured, so an eager Triton launch here is a straight win with
none of a graphed decode step's replay-safety questions).
Decode keeps calling ``flashinfer_attn.fused_qk_norm_rope`` exactly as before
-- ``qk_norm_rope(..., use_fused=False)`` (the default) is a pure pass-through
to that function, so nothing about the graphed decode path changes. See the
module docstring's "Decode" note and `FusedAttention._qkv` in
``runtime/fused_model.py`` for exactly which call sites set ``use_fused=True``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from .flashinfer_attn import fused_qk_norm_rope as _eager_qk_norm_rope

try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore

    HAS_TRITON = True
except Exception:  # pragma: no cover - depends on the host
    triton = None  # type: ignore
    HAS_TRITON = False


def norm_weight_1p(weight: torch.Tensor) -> torch.Tensor:
    """``(1 + weight)`` in fp32, once. Same value as the inline
    ``1.0 + weight.float()`` in ``flashinfer_attn.rms_norm_head`` -- kept as a
    tiny local helper (rather than importing ``fused_model.norm_weight_1p``)
    because ``attn`` is the lower-level package and must not import from
    ``runtime`` (``runtime.fused_model`` imports ``attn``, not the reverse)."""
    return (1.0 + weight.float()).contiguous()


# =========================================================================== #
# the fused kernel
# =========================================================================== #
if HAS_TRITON:  # pragma: no cover - exercised on GPU only (test_fused_qk_rope.py)

    @triton.jit
    def _qk_norm_rope_kernel(
        X_ptr, W1P_ptr, COS_ptr, SIN_ptr, OUT_ptr,
        stride_xt, stride_xh, stride_xd,
        stride_ct, stride_cd,
        stride_ot, stride_oh, stride_od,
        head_dim: tl.constexpr,
        rotary_dim: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """One program per ``(token, head)``.

        Reduces RMSNorm over the full ``head_dim``-length row, scales by the
        precomputed ``(1 + weight)``, then rotates the first ``rotary_dim``
        dims (rotate-half RoPE) and passes the rest straight through -- all
        from one load of the row (plus a masked gather load of each lane's
        rotate-half partner, needed only to build the roped output; the norm
        reduction itself never needs the gather).
        """
        pid_t = tl.program_id(0)
        pid_h = tl.program_id(1)
        half = rotary_dim // 2

        offs = tl.arange(0, BLOCK_D)
        mask = offs < head_dim

        x_base = X_ptr + pid_t * stride_xt + pid_h * stride_xh
        x = tl.load(x_base + offs * stride_xd, mask=mask, other=0.0)
        xf = x.to(tl.float32)

        # RMSNorm reduction over all `head_dim` dims (not just the roped
        # prefix) -- matches `rms_norm_head`, which norms before RoPE ever
        # slices anything.
        ss = tl.sum(xf * xf, axis=0) / head_dim
        rstd = 1.0 / tl.sqrt(ss + eps)

        w1p = tl.load(W1P_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        normed = xf * rstd * w1p

        # Each rotary lane's rotate-half partner: lanes [0, half) pair with
        # [half, rotary_dim), and vice versa. Lanes >= rotary_dim are never
        # read through this (rope_mask excludes them below); the gather
        # offset there is harmless (clamped to `offs` itself, in-bounds).
        rope_mask = offs < rotary_dim
        pair_offs = tl.where(offs < half, offs + half, tl.where(rope_mask, offs - half, offs))
        x_pair = tl.load(x_base + pair_offs * stride_xd, mask=mask, other=0.0).to(tl.float32)
        w1p_pair = tl.load(W1P_ptr + pair_offs, mask=mask, other=0.0).to(tl.float32)
        normed_pair = x_pair * rstd * w1p_pair

        c = tl.load(COS_ptr + pid_t * stride_ct + offs * stride_cd, mask=rope_mask, other=0.0).to(tl.float32)
        s = tl.load(SIN_ptr + pid_t * stride_ct + offs * stride_cd, mask=rope_mask, other=0.0).to(tl.float32)
        # lanes [0, half): -x_pair*sin ; lanes [half, rotary_dim): +x_pair*sin
        sign = tl.where(offs < half, -1.0, 1.0)
        roped = normed * c + sign * normed_pair * s

        out = tl.where(rope_mask, roped, normed)

        o_base = OUT_ptr + pid_t * stride_ot + pid_h * stride_oh
        tl.store(o_base + offs * stride_od, out.to(OUT_ptr.dtype.element_ty), mask=mask)

    def _launch(x: torch.Tensor, w1p: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                rotary_dim: int, eps: float) -> torch.Tensor:
        t, h, d = x.shape
        out = torch.empty_like(x)
        block_d = triton.next_power_of_2(d)
        grid = (t, h)
        _qk_norm_rope_kernel[grid](
            x, w1p, cos, sin, out,
            x.stride(0), x.stride(1), x.stride(2),
            cos.stride(0), cos.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            head_dim=d, rotary_dim=rotary_dim, eps=eps, BLOCK_D=block_d,
        )
        return out

    def fused_qk_norm_rope_triton(
        q: torch.Tensor,
        k: torch.Tensor,
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        eps: float = 1e-6,
        rotary_dim: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Triton fused QK-RMSNorm + partial RoPE. GPU only.

        ``q``/``k``: ``[T, H, head_dim]`` bf16 (or any float dtype), any
        strides (packed varlen prefill's ``q``/``k`` are non-contiguous
        views of the fused ``qkv_proj`` output -- see ``FusedAttention._qkv``
        -- strides are read from the tensors, never assumed). ``cos``/
        ``sin``: ``[T, rotary_dim]``, as built by ``rope.RotaryTable``.
        Raw (not ``1+weight``) norm weights in, matching
        ``flashinfer_attn.fused_qk_norm_rope``'s signature exactly so the two
        are interchangeable at the call site.
        """
        if rotary_dim is None:
            rotary_dim = cos.shape[-1]
        q_w1p = norm_weight_1p(q_norm_weight)
        k_w1p = norm_weight_1p(k_norm_weight)
        q_out = _launch(q, q_w1p, cos, sin, rotary_dim, eps)
        k_out = _launch(k, k_w1p, cos, sin, rotary_dim, eps)
        return q_out, k_out

else:  # pragma: no cover - depends on the host

    def fused_qk_norm_rope_triton(*args, **kwargs):
        raise RuntimeError(
            "triton is not importable on this host; use qk_norm_rope(..., use_fused=False) "
            "(the eager torch reference) instead"
        )


# =========================================================================== #
# the dispatcher -- the one line `FusedAttention._qkv` calls
# =========================================================================== #
def qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float = 1e-6,
    *,
    use_fused: bool = False,
    rotary_dim: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Drop-in for ``flashinfer_attn.fused_qk_norm_rope`` with one extra
    keyword: ``use_fused=True`` routes through the Triton kernel above when
    it is actually usable (Triton importable *and* the tensors are on CUDA --
    checking only "did Triton import" is not enough: that is also true on a
    CPU-only host, and would hand CPU tensors to a CUDA kernel).
    Every other case, ``use_fused=False`` (decode: the default, unchanged),
    or ``use_fused=True`` on a host/tensor where the Triton path
    cannot run, falls back to the exact eager reference, so this function
    is always safe to call from either path.
    """
    if use_fused and HAS_TRITON and q.is_cuda and k.is_cuda:
        return fused_qk_norm_rope_triton(q, k, q_norm_weight, k_norm_weight, cos, sin, eps, rotary_dim)
    return _eager_qk_norm_rope(q, k, q_norm_weight, k_norm_weight, cos, sin, eps)


__all__ = [
    "HAS_TRITON",
    "norm_weight_1p",
    "fused_qk_norm_rope_triton",
    "qk_norm_rope",
]
