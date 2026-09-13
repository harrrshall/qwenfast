"""Fused GDN gate epilogue: ``beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias)``.

Replaces the tail of ``fused_model.FusedGDN._project`` -- after
``in_proj_ba`` produces ``ba = [b ; a]``, eager does a split (free, a view)
then ``b.sigmoid()`` (1 launch), ``a.float() + dt_bias_f`` (1), ``F.softplus``
(1), ``neg_exp_A * ...`` (1): four launches per GDN layer per step, on a
tensor of width ``num_v_heads`` (48 in the real config): per-layer
gate/activation/split glue on the ``in_proj`` outputs. At 48 GDN layers
that is ~192 of a decode profile's ``elementwise/other`` launches. One Triton kernel replaces all four with one
launch, reading ``ba`` once and writing ``beta``/``g`` once each.

Numerics contract (unchanged): ``g`` is fp32 throughout (the GDN kernel
wants it that way); ``beta`` is produced in ``ba``'s own
(activation) dtype, matching eager's ``b.sigmoid()`` (no explicit fp32
upcast in the reference -- `torch.sigmoid` on a bf16 tensor is computed at
whatever internal precision aten uses and rounds to bf16, so this kernel is
compared against it with the same bf16-ULP + MAE bound as the other Triton
kernels in this tree, not a bit-identical assertion).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

try:  # pragma: no cover - depends on the host
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover
    triton = None  # type: ignore
    tl = None  # type: ignore
    HAS_TRITON = False


def gdn_gate_reference(
    ba: torch.Tensor, neg_exp_A: torch.Tensor, dt_bias_f: torch.Tensor, num_v_heads: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The eager op sequence this kernel replaces (mirrors
    ``FusedGDN._project``'s tail exactly, kept here too so this module's
    own tests don't need to import ``fused_model``)."""
    b, a = ba.split([num_v_heads, num_v_heads], dim=-1)
    beta = b.sigmoid()
    g = neg_exp_A * F.softplus(a.float() + dt_bias_f)
    return beta, g


def _block_and_warps(n: int) -> "tuple[int, int]":
    block = min(2048, 1 << (max(n, 1) - 1).bit_length())
    warps = 2 if block <= 256 else 4
    return block, warps


if HAS_TRITON:  # pragma: no cover - GPU only

    @triton.jit
    def _gdn_gate_kernel(
        BA_ptr, NEG_EXP_A_ptr, DT_BIAS_ptr, BETA_ptr, G_ptr,
        stride_ba_row, stride_out_row,
        h: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < h
        base = BA_ptr + row * stride_ba_row
        b = tl.load(base + offs, mask=mask, other=0.0).to(tl.float32)
        a = tl.load(base + h + offs, mask=mask, other=0.0).to(tl.float32)

        beta = 1.0 / (1.0 + tl.exp(-b))
        tl.store(BETA_ptr + row * stride_out_row + offs, beta.to(BETA_ptr.dtype.element_ty), mask=mask)

        neg_exp_A = tl.load(NEG_EXP_A_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        dt_bias = tl.load(DT_BIAS_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        z = a + dt_bias
        # torch.nn.functional.softplus default beta=1, threshold=20: linear
        # past the threshold (avoids overflow in exp(z)), log1p(exp(z)) below it.
        softplus = tl.where(z > 20.0, z, tl.log(1.0 + tl.exp(z)))
        g = neg_exp_A * softplus
        tl.store(G_ptr + row * stride_out_row + offs, g, mask=mask)

    def triton_gdn_gate(
        ba: torch.Tensor, neg_exp_A: torch.Tensor, dt_bias_f: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = neg_exp_A.shape[-1]
        shape = ba.shape
        assert shape[-1] == 2 * h, f"ba width {shape[-1]} != 2 * num_v_heads ({2 * h})"
        ba2 = ba.reshape(-1, shape[-1])
        rows = ba2.shape[0]
        beta = torch.empty(rows, h, dtype=ba.dtype, device=ba.device)
        g = torch.empty(rows, h, dtype=torch.float32, device=ba.device)
        block, warps = _block_and_warps(h)
        _gdn_gate_kernel[(rows,)](
            ba2, neg_exp_A, dt_bias_f, beta, g,
            ba2.stride(0), beta.stride(0), h=h, BLOCK=block, num_warps=warps,
        )
        out_shape = shape[:-1] + (h,)
        return beta.reshape(out_shape), g.reshape(out_shape)

else:

    def triton_gdn_gate(*args, **kwargs):  # pragma: no cover
        raise RuntimeError("triton is not importable on this host")


__all__ = ["HAS_TRITON", "gdn_gate_reference", "triton_gdn_gate"]
