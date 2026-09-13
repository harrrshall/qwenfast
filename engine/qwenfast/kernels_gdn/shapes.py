"""Frozen shape/layout constants for the Qwen3.8-27B Gated-DeltaNet kernels.

Everything here is read off ``engine/reference/config-Qwen3.8-27B-FP8.json``
(text_config) and mirrors ``kernels/microbench/common.py``.  This module must
import with **no torch installed** (it is ``py_compile``-checked on a Mac).

Layout contract
---------------
Recurrent state pool, whole model::

    [n_slots, 48 layers, 48 v-heads, 128 k-dim, 128 v-dim]      fp32 | fp16 | bf16
    -> 3.00 MiB / layer / slot, 144 MiB / slot (fp32), 72 MiB (fp16)

A *per-layer* view is ``pool[:, layer_idx]`` -> ``[n_slots, 48, 128, 128]``.
That view is **not** contiguous in dim 0 (slot stride = 48*48*128*128) which is
exactly why every kernel here takes the slot stride explicitly and indexes with
a ``slot_ids[B]`` int32 tensor rather than requiring a compacted batch.

Conv state pool, whole model::

    [n_slots, 48 layers, 10240 channels, 3 (= K-1)]             bf16
    -> 60 KiB / layer / slot, 2.81 MiB / slot

GVA
---
16 k-heads feed 48 v-heads, 3:1.  v-head ``hv`` uses k-head ``hv // 3`` — i.e.
exactly ``repeat_interleave(3, dim=head)``.  ``fla`` >= 0.5.2 implements the
same mapping natively (``i_h = i_hv // (HV // H)``), so the canonical input
form for every entry point in this package is q/k with **16** heads; 48-head
(pre-expanded) inputs are accepted and collapsed back where it is free.
"""

from __future__ import annotations

# --- model ---------------------------------------------------------------- #
HIDDEN_SIZE = 5120
NUM_LAYERS_TOTAL = 64
NUM_GDN_LAYERS = 48

# --- Gated DeltaNet ------------------------------------------------------- #
NUM_V_HEADS = 48
NUM_K_HEADS = 16
HEAD_K_DIM = 128
HEAD_V_DIM = 128
GVA_GROUP = NUM_V_HEADS // NUM_K_HEADS  # 3

KEY_DIM = NUM_K_HEADS * HEAD_K_DIM  # 2048
VALUE_DIM = NUM_V_HEADS * HEAD_V_DIM  # 6144
CONV_DIM = 2 * KEY_DIM + VALUE_DIM  # 10240
CONV_KERNEL = 4
CONV_STATE_WIDTH = CONV_KERNEL - 1  # 3

L2NORM_EPS = 1e-6  # matches fla: x / sqrt(sum(x*x) + eps)
DEFAULT_SCALE = HEAD_K_DIM ** -0.5  # 1/sqrt(128)
DEFAULT_CHUNK_SIZE = 64

# --- sweeps / physics ----------------------------------------------------- #
DECODE_BATCH_SWEEP = [1, 8, 32, 64, 128, 256, 512]
H200_HBM_GBPS = 4800.0


def state_bytes_per_slot_per_layer(itemsize: int = 4) -> int:
    """3.00 MiB at fp32."""
    return NUM_V_HEADS * HEAD_K_DIM * HEAD_V_DIM * itemsize


def state_bytes(batch: int, itemsize: int = 4, layers: int = 1) -> int:
    return batch * layers * state_bytes_per_slot_per_layer(itemsize)


def conv_state_bytes_per_slot_per_layer(itemsize: int = 2) -> int:
    """60 KiB at bf16."""
    return CONV_DIM * CONV_STATE_WIDTH * itemsize


__all__ = [
    "HIDDEN_SIZE",
    "NUM_LAYERS_TOTAL",
    "NUM_GDN_LAYERS",
    "NUM_V_HEADS",
    "NUM_K_HEADS",
    "HEAD_K_DIM",
    "HEAD_V_DIM",
    "GVA_GROUP",
    "KEY_DIM",
    "VALUE_DIM",
    "CONV_DIM",
    "CONV_KERNEL",
    "CONV_STATE_WIDTH",
    "L2NORM_EPS",
    "DEFAULT_SCALE",
    "DEFAULT_CHUNK_SIZE",
    "DECODE_BATCH_SWEEP",
    "H200_HBM_GBPS",
    "state_bytes_per_slot_per_layer",
    "state_bytes",
    "conv_state_bytes_per_slot_per_layer",
]
