"""Partial RoPE: precomputed cos/sin table + the reference apply function.

Text-only mRoPE degenerates exactly to
standard RoPE over the first 64 of 256 head dims (theta = 1e7, 32 frequency
pairs). ``engine/qwenfast/model.py`` already carries ``rotate_half``,
``apply_rotary_pos_emb`` and ``interleave_mrope`` copied verbatim from
``engine/reference/modeling_qwen3_5.py`` (bit-identical -- diffed by hand
while building this module: same ``cos.unsqueeze``/``rotate_half``/concat
sequence, same ``apply_interleaved_mrope`` chunk-vs-interleave logic). Rather
than re-copy that code a second time and risk it drifting from the dense
reference model, this module *re-exports* those functions and adds the one
thing the reference model's
per-forward-pass ``RotaryEmbedding`` doesn't need but the paged/graphed
decode path does: a table precomputed once up front and indexed with a
gather, so a decode step never recomputes ``cos``/``sin`` from scratch.

Table size at the full 262,144-position, 32-frequency, bf16 spec:
``262144 * 64 * 2 (cos+sin) * 2 bytes = 64 MiB`` -- trivial next to the KV
pool.
"""

from __future__ import annotations

from typing import Tuple

import torch

from ..model import (  # noqa: F401 -- re-exported, see module docstring
    apply_rotary_pos_emb,
    compute_inv_freq,
    interleave_mrope,
    rotate_half,
)

DEFAULT_ROTARY_DIM = 64  # head_dim(256) * partial_rotary_factor(0.25)
DEFAULT_THETA = 1e7
DEFAULT_MAX_POSITIONS = 262_144  # config.max_position_embeddings


class RotaryTable:
    """Precomputed ``(cos, sin)`` lookup table, ``[max_positions, rotary_dim]``.

    Built once (e.g. at engine startup) and indexed with a gather per step --
    cheap, allocation-free, and CUDA-graph-friendly (``lookup`` is a pure
    ``index_select``, no data-dependent control flow).
    """

    def __init__(
        self,
        max_positions: int = DEFAULT_MAX_POSITIONS,
        rotary_dim: int = DEFAULT_ROTARY_DIM,
        theta: float = DEFAULT_THETA,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cpu",
    ):
        if rotary_dim % 2 != 0:
            raise ValueError("rotary_dim must be even (it is head_dim * partial_rotary_factor)")
        self.max_positions = max_positions
        self.rotary_dim = rotary_dim
        self.theta = theta
        self.dtype = dtype
        inv_freq = compute_inv_freq(rotary_dim, theta, device=device)  # [rotary_dim/2], fp32
        positions = torch.arange(max_positions, dtype=torch.float32, device=device)
        freqs = positions[:, None] * inv_freq[None, :]  # [max_positions, rotary_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [max_positions, rotary_dim]
        # cos/sin themselves are computed in fp32 and cast down once, so the
        # bf16 table's rounding error is exactly one rounding step, not an
        # accumulation of per-step rounding as positions grow.
        self.cos = emb.cos().to(dtype)
        self.sin = emb.sin().to(dtype)

    def to(self, device: torch.device | str) -> "RotaryTable":
        self.cos = self.cos.to(device)
        self.sin = self.sin.to(device)
        return self

    def lookup(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``positions``: any int shape ``[...]`` -> ``(cos, sin)`` of shape
        ``[..., rotary_dim]``, gathered from the precomputed table."""
        idx = positions.long()
        return self.cos[idx], self.sin[idx]


__all__ = [
    "DEFAULT_ROTARY_DIM",
    "DEFAULT_THETA",
    "DEFAULT_MAX_POSITIONS",
    "RotaryTable",
    "apply_rotary_pos_emb",
    "rotate_half",
    "interleave_mrope",
    "compute_inv_freq",
]
