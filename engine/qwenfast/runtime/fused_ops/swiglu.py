"""Fused SwiGLU: ``silu(gate) * up`` from one ``[..., 2*inter]`` tensor.

Replaces ``fused_model.swiglu`` (``gate, up = gate_up.chunk(2, -1); return
F.silu(gate) * up``) -- two kernel launches (``aten::silu``, ``aten::mul``)
per call. A decode profile (B=128, with the Triton norms on) attributes
exactly this shape to two of its biggest ``elementwise/other`` entries:
128 calls / 702.5 us and 64 calls / 591.6 us -- i.e. two launches per one of the model's 64 ``FusedMLP`` call
sites (48 GDN + 16 attention decoder layers), ~1.3 ms/step at B=128. One
Triton kernel replaces both with a single launch per call site.

Numerics: eager's ``F.silu`` computes in the input dtype (bf16 in
production); this kernel accumulates in fp32 and rounds once at the store,
which can only be *more* accurate, never a numerics regression -- verified
against the eager reference within a bf16-ULP bound, same convention as
``fused_model.py``'s Triton norm tests (``TestTritonNorms``).
"""

from __future__ import annotations

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


def swiglu_reference(gate_up: torch.Tensor) -> torch.Tensor:
    """The eager op this kernel replaces (also ``fused_model.swiglu``,
    kept here too so this module's own tests don't need to import
    ``fused_model``)."""
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up


def _block_and_warps(n: int) -> "tuple[int, int]":
    block = min(1024, 1 << (max(n, 1) - 1).bit_length())
    warps = 4 if block <= 1024 else 8
    return block, warps


if HAS_TRITON:  # pragma: no cover - GPU only

    @triton.jit
    def _swiglu_kernel(
        GATE_UP_ptr, OUT_ptr,
        stride_in_row, stride_out_row,
        inter: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        col0 = tl.program_id(1) * BLOCK
        offs = col0 + tl.arange(0, BLOCK)
        mask = offs < inter
        base = GATE_UP_ptr + row * stride_in_row
        gate = tl.load(base + offs, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(base + inter + offs, mask=mask, other=0.0).to(tl.float32)
        silu = gate / (1.0 + tl.exp(-gate))
        out = silu * up
        tl.store(OUT_ptr + row * stride_out_row + offs, out.to(OUT_ptr.dtype.element_ty), mask=mask)

    def triton_swiglu(gate_up: torch.Tensor) -> torch.Tensor:
        shape = gate_up.shape
        inter = shape[-1] // 2
        x2 = gate_up.reshape(-1, shape[-1])
        rows = x2.shape[0]
        out = torch.empty(rows, inter, dtype=gate_up.dtype, device=gate_up.device)
        block, warps = _block_and_warps(inter)
        grid = (rows, triton.cdiv(inter, block))
        _swiglu_kernel[grid](
            x2, out, x2.stride(0), out.stride(0), inter=inter, BLOCK=block, num_warps=warps
        )
        return out.reshape(*shape[:-1], inter)

else:

    def triton_swiglu(*args, **kwargs):  # pragma: no cover
        raise RuntimeError("triton is not importable on this host")


__all__ = ["HAS_TRITON", "swiglu_reference", "triton_swiglu"]
