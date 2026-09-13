"""SSM state-pool allocation, per-layer views, and gather/scatter helpers.

The pool layout is frozen and is the contract between this package, the
runtime (including its CUDA graphs) and speculative decoding.  See
:mod:`.shapes` for the numbers.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from . import shapes

# --------------------------------------------------------------------------- #
# dtype plumbing
# --------------------------------------------------------------------------- #
STATE_DTYPES = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}


def resolve_state_dtype(dtype) -> torch.dtype:
    """``'fp32' | 'fp16' | 'bf16' | torch.dtype`` -> ``torch.dtype``."""
    if isinstance(dtype, torch.dtype):
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(f"unsupported SSM state dtype {dtype}")
        return dtype
    try:
        return STATE_DTYPES[str(dtype).lower()]
    except KeyError as exc:  # pragma: no cover - argument validation
        raise ValueError(f"unknown --ssm-state-dtype {dtype!r}") from exc


# --------------------------------------------------------------------------- #
# allocation
# --------------------------------------------------------------------------- #
def alloc_state_pool(
    n_slots: int,
    n_layers: int = shapes.NUM_GDN_LAYERS,
    n_v_heads: int = shapes.NUM_V_HEADS,
    head_k: int = shapes.HEAD_K_DIM,
    head_v: int = shapes.HEAD_V_DIM,
    dtype=torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """``[n_slots, n_layers, n_v_heads, head_k, head_v]``, zeroed, slot-major.

    Slot-major so one slot is a contiguous 144 MiB (fp32) span — that is what
    makes host swap (preemption) and PD-disaggregation a single copy.
    """
    dt = resolve_state_dtype(dtype)
    return torch.zeros(
        (n_slots, n_layers, n_v_heads, head_k, head_v), dtype=dt, device=device
    )


CONV_LAYOUTS = ("width_major", "channel_major")


def alloc_conv_state_pool(
    n_slots: int,
    n_layers: int = shapes.NUM_GDN_LAYERS,
    conv_dim: int = shapes.CONV_DIM,
    width: int = shapes.CONV_KERNEL,
    dtype=torch.bfloat16,
    device: Optional[torch.device] = None,
    layout: str = "width_major",
) -> torch.Tensor:
    """The conv ring pool, zeroed.  60 KiB/layer/slot either way.

    ``layout='width_major'`` (**default**) -> ``[n_slots, n_layers, W-1, C]``
        The decode step touches **all C channels for one w index**, so channel
        must be the contiguous axis.  Measured: the channel-major version ran
        the conv update at 7-360 GB/s because every access was a stride-3
        gather of 2-byte elements, i.e. ~2 useful bytes per 32 B sector.
    ``layout='channel_major'`` -> ``[n_slots, n_layers, C, W-1]``
        M0's ``HybridCache`` layout (``[B, conv_dim, K-1]``).  Kept so a slot
        can be handed straight to the M0 reference; every entry point here
        detects the layout from the shape, so both work.
    """
    if layout not in CONV_LAYOUTS:
        raise ValueError(f"layout must be one of {CONV_LAYOUTS}, got {layout!r}")
    tail = (width - 1, conv_dim) if layout == "width_major" else (conv_dim, width - 1)
    return torch.zeros((n_slots, n_layers) + tail, dtype=dtype, device=device)


def conv_pool_is_width_major(pool: torch.Tensor, conv_dim: int) -> bool:
    """``True`` for ``[..., W-1, C]``, ``False`` for ``[..., C, W-1]``."""
    if pool.shape[-1] == conv_dim:
        return True
    if pool.shape[-2] == conv_dim:
        return False
    raise ValueError(
        f"conv pool {tuple(pool.shape)} matches neither [.., W-1, {conv_dim}] "
        f"nor [.., {conv_dim}, W-1]"
    )


def prepare_conv_weight(w: torch.Tensor) -> torch.Tensor:
    """``[C, W]`` stored **width-major** (``stride(0) == 1``).

    Same logical tensor, but the kernel's per-``j`` weight read becomes a
    contiguous run over channels instead of a stride-``W`` gather.  Do this
    once at load time; it is a weight, not an activation.
    """
    if w.dim() != 2:
        raise ValueError(f"conv weight must be [C, W], got {tuple(w.shape)}")
    return w.t().contiguous().t()


def layer_state(pool: torch.Tensor, layer_idx: int) -> torch.Tensor:
    """Per-layer view of a whole-model pool -> ``[n_slots, HV, K, V]``.

    A strided (non-contiguous-in-dim-0) view.  Every kernel in this package
    reads ``pool.stride(0)`` explicitly, so no copy is ever made.
    """
    if pool.dim() == 4:
        return pool  # already a per-layer pool
    return pool[:, layer_idx]


def layer_conv_state(pool: torch.Tensor, layer_idx: int) -> torch.Tensor:
    """Per-layer view of a conv pool -> ``[n_slots, W-1, C]`` or ``[n_slots, C, W-1]``.

    The trailing two dims keep whichever order the pool was allocated with;
    every conv entry point detects it from the shape.
    """
    if pool.dim() == 3:
        return pool
    return pool[:, layer_idx]


# --------------------------------------------------------------------------- #
# gather / scatter
# --------------------------------------------------------------------------- #
def gather_states(
    state_pool: torch.Tensor,
    slot_ids: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """``[B, HV, K, V]`` contiguous copy of the rows named by ``slot_ids``.

    This is the price the ``fla`` backend pays: fla 0.5.2 has no state-pool
    indexing (vLLM's *fork* added ``ssm_state_indices``; upstream did not), so
    the state has to be materialised contiguously, which turns the ideal
    1 read + 1 write into 2 reads + 2 writes.  The Triton backend indexes the
    pool directly and avoids this entirely.
    """
    idx = slot_ids.to(device=state_pool.device, dtype=torch.long)
    out = state_pool.index_select(0, idx)
    if dtype is not None and out.dtype != dtype:
        out = out.to(dtype)
    return out.contiguous()


def scatter_states(
    state_pool: torch.Tensor, slot_ids: torch.Tensor, states: torch.Tensor
) -> None:
    """In-place ``state_pool[slot_ids] = states`` with a dtype cast."""
    idx = slot_ids.to(device=state_pool.device, dtype=torch.long)
    state_pool.index_copy_(0, idx, states.to(state_pool.dtype))


def zero_slots(state_pool: torch.Tensor, slot_ids: torch.Tensor) -> None:
    """Reset the named slots (used when a request is admitted)."""
    idx = slot_ids.to(device=state_pool.device, dtype=torch.long)
    state_pool.index_fill_(0, idx, 0)


# --------------------------------------------------------------------------- #
# introspection
# --------------------------------------------------------------------------- #
def pool_nbytes(pool: torch.Tensor) -> int:
    return pool.numel() * pool.element_size()


def describe_pool(pool: torch.Tensor) -> Tuple[str, int, int]:
    """``(shape_str, bytes_total, bytes_per_slot)``."""
    total = pool_nbytes(pool)
    per_slot = total // max(pool.shape[0], 1)
    return (str(tuple(pool.shape)), total, per_slot)


__all__ = [
    "STATE_DTYPES",
    "CONV_LAYOUTS",
    "conv_pool_is_width_major",
    "prepare_conv_weight",
    "resolve_state_dtype",
    "alloc_state_pool",
    "alloc_conv_state_pool",
    "layer_state",
    "layer_conv_state",
    "gather_states",
    "scatter_states",
    "zero_slots",
    "pool_nbytes",
    "describe_pool",
]
