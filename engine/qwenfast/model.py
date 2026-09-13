"""qwenfast reference model: minimal pure-PyTorch Qwen3.8-27B (text-only).

This is the *correctness oracle* for the whole engine: every optimisation
(fused FP8 GEMMs, CUDA-graphed decode, MTP speculation) is validated against
this file, so it deliberately favours clarity over speed.

Architecture recap::

    hidden 5120, 64 layers = 48 Gated-DeltaNet + 16 gated GQA (every 4th)
    MLP 17408, vocab 248320 (untied lm_head)
    GDN : 16 k-heads x 128, 48 v-heads x 128, conv kernel 4, fp32 state
    Attn: 24 q-heads x 256 (+ 24 gate-heads x 256 fused in q_proj), 4 kv-heads,
          QK-RMSNorm on head_dim, partial RoPE over the first 64 dims,
          sigmoid output gate
    MTP : 1 extra full-attention decoder layer + fc([embed ; hidden]) -> hidden

Text-only mRoPE note
--------------------
The reference model builds 3-D (t, h, w) position ids and interleaves the
sections.  For text-only inputs all three rows are identical, so the interleave
is a no-op and mRoPE degenerates *exactly* to standard RoPE over
``head_dim * partial_rotary_factor = 64`` dims.  We implement the standard form
and keep :func:`interleave_mrope` around so the equivalence stays testable.
"""

from __future__ import annotations


from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .weights import QwenFastConfig

# --------------------------------------------------------------------------- #
# optional fla kernels
# --------------------------------------------------------------------------- #
try:
    from fla.ops.gated_delta_rule import (  # type: ignore
        chunk_gated_delta_rule as _fla_chunk,
        fused_recurrent_gated_delta_rule as _fla_recurrent,
    )

    HAS_FLA = True
except Exception:  # pragma: no cover - depends on the host environment
    _fla_chunk = None
    _fla_recurrent = None
    HAS_FLA = False


# --------------------------------------------------------------------------- #
# primitive ops (kept 1:1 with modeling_qwen3_5.py)
# --------------------------------------------------------------------------- #
def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """Matches the `fla` l2norm used inside the GDN kernels."""
    inv = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv


class RMSNorm(nn.Module):
    """Qwen3.5/3.8 RMSNorm: normalise in fp32, scale by ``(1 + weight)``."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        out = out * (1.0 + self.weight.float())
        return out.type_as(x)

    def extra_repr(self) -> str:  # pragma: no cover
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class RMSNormGated(nn.Module):
    """GDN output norm: plain ``weight *`` (NOT ``1 + weight``) then silu gate."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        in_dtype = hidden_states.dtype
        h = hidden_states.to(torch.float32)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        h = self.weight * h.to(in_dtype)
        h = h * F.silu(gate.to(torch.float32))
        return h.to(in_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Partial RoPE: rotate the first ``cos.shape[-1]`` dims, pass the rest."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)


def interleave_mrope(freqs: torch.Tensor, mrope_section: Sequence[int]) -> torch.Tensor:
    """Reference interleaved-mRoPE section mixing (``freqs``: ``[3, B, S, D/2]``)."""
    freqs_t = freqs[0].clone()
    for dim, offset in enumerate((1, 2), start=1):
        length = mrope_section[dim] * 3
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[dim, ..., idx]
    return freqs_t


def compute_inv_freq(rotary_dim: int, theta: float, device=None) -> torch.Tensor:
    return 1.0 / (
        theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim)
    )


class RotaryEmbedding(nn.Module):
    """Standard (text-only) partial RoPE over ``rotary_dim`` head dims.

    ``inv_freq`` is a plain cached tensor rather than a registered buffer so the
    module can be built on ``meta`` and populated purely from the checkpoint.
    """

    def __init__(self, config: QwenFastConfig):
        super().__init__()
        self.rotary_dim = config.rotary_dim
        self.rope_theta = float(config.rope_theta)
        self.mrope_section = list(config.mrope_section)
        self._inv_freq: Optional[torch.Tensor] = None

    def inv_freq(self, device: torch.device) -> torch.Tensor:
        if self._inv_freq is None or self._inv_freq.device != device:
            self._inv_freq = compute_inv_freq(self.rotary_dim, self.rope_theta, device)
        return self._inv_freq

    @torch.no_grad()
    def forward(
        self, position_ids: torch.Tensor, dtype: torch.dtype = torch.bfloat16
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``position_ids``: ``[B, S]`` int -> ``(cos, sin)`` of ``[B, S, rotary_dim]``."""
        inv = self.inv_freq(position_ids.device)
        freqs = position_ids.float()[..., None] * inv[None, None, :]  # [B, S, D/2]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


# --------------------------------------------------------------------------- #
# gated delta rule (pure torch, copied from modeling_qwen3_5.py)
# --------------------------------------------------------------------------- #
def torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
):
    """Chunked (prefill) gated delta rule. Inputs are ``[B, S, H, D]``."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0
    )

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1
    )

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        # decay_mask is already lower-triangular (inclusive diagonal), so the
        # strictly-upper entries are exactly zero; `mask` is kept for parity
        # with the reference implementation.
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def torch_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = False,
):
    """Sequential (decode) gated delta rule. Inputs are ``[B, S, H, D]``."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(
        batch_size, num_heads, sequence_length, v_head_dim, dtype=value.dtype, device=value.device
    )
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def gated_delta_rule(
    query, key, value, g, beta, initial_state, recurrent: bool, use_fla: bool = True
):
    """Dispatch to `fla` when available, else the pure-torch reference."""
    if use_fla and HAS_FLA:
        fn = _fla_recurrent if recurrent else _fla_chunk
        out, state = fn(
            q=query,
            k=key,
            v=value,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        return out, state
    if recurrent:
        return torch_recurrent_gated_delta_rule(
            query, key, value, g, beta, initial_state, True, use_qk_l2norm_in_kernel=True
        )
    return torch_chunk_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )


def causal_conv1d(x: torch.Tensor, conv_state: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Depthwise causal conv with an in-place ``[B, C, K-1]`` state ring.

    ``x``: ``[B, C, S]``.  Works for both prefill (S>1, state may be zeros) and
    decode (S==1).  ``weight``: ``[C, K]``.  Activation is silu (config).
    """
    k = weight.shape[-1]
    x_in = torch.cat([conv_state, x], dim=-1).to(weight.dtype)
    conv_state.copy_(x_in[:, :, -(k - 1) :])
    out = F.conv1d(x_in, weight.unsqueeze(1), None, padding=0, groups=weight.shape[0])
    return F.silu(out).to(x.dtype)


# --------------------------------------------------------------------------- #
# cache
# --------------------------------------------------------------------------- #
class HybridCache:
    """Dense (non-paged) reference cache: per-layer GDN state + per-layer KV.

    Deliberately simple: one contiguous slab per layer, batch-major,
    left-padded prompts so that every sequence's last token sits at ``pos-1``.
    The paged/pooled version lives in ``qwenfast.runtime``.
    """

    def __init__(
        self,
        config: QwenFastConfig,
        batch_size: int,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        state_dtype: torch.dtype = torch.float32,
        with_mtp: bool = False,
    ):
        self.config = config
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        self.state_dtype = state_dtype
        self.seq_len = 0  # number of valid (padded) cache positions

        c = config
        self.conv_states: Dict[int, torch.Tensor] = {}
        self.recurrent_states: Dict[int, torch.Tensor] = {}
        for i in c.linear_layer_indices:
            self.conv_states[i] = torch.zeros(
                batch_size, c.conv_dim, c.linear_conv_kernel_dim - 1, device=device, dtype=dtype
            )
            self.recurrent_states[i] = torch.zeros(
                batch_size,
                c.linear_num_value_heads,
                c.linear_key_head_dim,
                c.linear_value_head_dim,
                device=device,
                dtype=state_dtype,
            )

        attn_layers = list(c.attention_layer_indices)
        if with_mtp:
            attn_layers.append(c.mtp_layer_idx)
        self.k_cache: Dict[int, torch.Tensor] = {}
        self.v_cache: Dict[int, torch.Tensor] = {}
        for i in attn_layers:
            shape = (batch_size, c.num_key_value_heads, max_seq_len, c.head_dim)
            self.k_cache[i] = torch.zeros(shape, device=device, dtype=dtype)
            self.v_cache[i] = torch.zeros(shape, device=device, dtype=dtype)

        # [B, max_seq_len] bool: True where the cache slot holds a real token
        self.valid = torch.zeros(batch_size, max_seq_len, device=device, dtype=torch.bool)

    # -- introspection ------------------------------------------------------ #
    def nbytes(self) -> Dict[str, int]:
        ssm = sum(t.numel() * t.element_size() for t in self.recurrent_states.values())
        conv = sum(t.numel() * t.element_size() for t in self.conv_states.values())
        kv = sum(t.numel() * t.element_size() for t in self.k_cache.values()) * 2
        return {"ssm": ssm, "conv": conv, "kv": kv, "total": ssm + conv + kv}

    # -- kv update ---------------------------------------------------------- #
    def append_kv(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, start: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Write ``[B, H, S, D]`` at ``start`` and return the live prefix."""
        s = k.shape[2]
        end = start + s
        self.k_cache[layer_idx][:, :, start:end] = k
        self.v_cache[layer_idx][:, :, start:end] = v
        return self.k_cache[layer_idx][:, :, :end], self.v_cache[layer_idx][:, :, :end]

    def reset(self) -> None:
        for t in self.conv_states.values():
            t.zero_()
        for t in self.recurrent_states.values():
            t.zero_()
        self.valid.zero_()
        self.seq_len = 0

    def clone_ssm(self) -> Dict[str, Dict[int, torch.Tensor]]:
        """Snapshot of GDN state (the reference stand-in for MTP rollback)."""
        return {
            "conv": {i: t.clone() for i, t in self.conv_states.items()},
            "rec": {i: t.clone() for i, t in self.recurrent_states.items()},
        }

    def restore_ssm(self, snap: Dict[str, Dict[int, torch.Tensor]]) -> None:
        for i, t in snap["conv"].items():
            self.conv_states[i].copy_(t)
        for i, t in snap["rec"].items():
            self.recurrent_states[i].copy_(t)


# --------------------------------------------------------------------------- #
# blocks
# --------------------------------------------------------------------------- #
class GatedDeltaNet(nn.Module):
    """48-v-head / 16-k-head Gated DeltaNet token mixer.

    Weight shapes (hidden 5120)::

        in_proj_qkv [10240, 5120]   (q 2048 | k 2048 | v 6144)
        in_proj_z   [ 6144, 5120]
        in_proj_b   [   48, 5120]   in_proj_a [48, 5120]
        conv1d      [10240, 1, 4]   depthwise
        out_proj    [ 5120, 6144]

    The fused runtime folds qkv+z+b+a into one ``[16480, 5120]`` GEMM.
    """

    def __init__(self, config: QwenFastConfig, layer_idx: int):
        super().__init__()
        c = config
        self.config = c
        self.layer_idx = layer_idx
        self.num_v_heads = c.linear_num_value_heads
        self.num_k_heads = c.linear_num_key_heads
        self.head_k_dim = c.linear_key_head_dim
        self.head_v_dim = c.linear_value_head_dim
        self.key_dim = c.key_dim
        self.value_dim = c.value_dim
        self.conv_dim = c.conv_dim
        self.conv_kernel_size = c.linear_conv_kernel_dim

        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )
        self.dt_bias = nn.Parameter(torch.zeros(self.num_v_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))
        self.norm = RMSNormGated(self.head_v_dim, eps=c.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, c.hidden_size, bias=False)
        self.in_proj_qkv = nn.Linear(c.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(c.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(c.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(c.hidden_size, self.num_v_heads, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache: Optional[HybridCache] = None,
        padding_mask: Optional[torch.Tensor] = None,
        use_fla: bool = True,
    ) -> torch.Tensor:
        if padding_mask is not None:
            hidden_states = hidden_states * padding_mask[:, :, None].to(hidden_states.dtype)

        b, s, _ = hidden_states.shape
        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)  # [B, C, S]
        z = self.in_proj_z(hidden_states).reshape(b, s, -1, self.head_v_dim)
        beta = self.in_proj_b(hidden_states).sigmoid()
        a = self.in_proj_a(hidden_states)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())

        w = self.conv1d.weight.squeeze(1)  # [C, K]
        if cache is not None:
            conv_state = cache.conv_states[self.layer_idx]
        else:
            conv_state = torch.zeros(
                b, self.conv_dim, self.conv_kernel_size - 1,
                device=hidden_states.device, dtype=hidden_states.dtype,
            )
        mixed_qkv = causal_conv1d(mixed_qkv, conv_state, w)
        mixed_qkv = mixed_qkv.transpose(1, 2)  # [B, S, C]

        query, key, value = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        query = query.reshape(b, s, -1, self.head_k_dim)
        key = key.reshape(b, s, -1, self.head_k_dim)
        value = value.reshape(b, s, -1, self.head_v_dim)
        rep = self.num_v_heads // self.num_k_heads
        if rep > 1:
            query = query.repeat_interleave(rep, dim=2)
            key = key.repeat_interleave(rep, dim=2)

        init_state = cache.recurrent_states[self.layer_idx] if cache is not None else None
        core_out, last_state = gated_delta_rule(
            query, key, value, g, beta, init_state, recurrent=(s == 1), use_fla=use_fla
        )
        if cache is not None and last_state is not None:
            cache.recurrent_states[self.layer_idx].copy_(
                last_state.to(cache.recurrent_states[self.layer_idx].dtype)
            )

        core_out = core_out.reshape(-1, self.head_v_dim)
        core_out = self.norm(core_out, z.reshape(-1, self.head_v_dim))
        core_out = core_out.reshape(b, s, -1)
        return self.out_proj(core_out)


class Attention(nn.Module):
    """Gated GQA. ``q_proj`` already contains the output gate: 24 x 256 x 2."""

    def __init__(self, config: QwenFastConfig, layer_idx: int):
        super().__init__()
        c = config
        self.config = c
        self.layer_idx = layer_idx
        self.head_dim = c.head_dim
        self.num_heads = c.num_attention_heads
        self.num_kv_heads = c.num_key_value_heads
        self.num_kv_groups = c.num_attention_heads // c.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        gate_mult = 2 if c.attn_output_gate else 1
        self.q_proj = nn.Linear(
            c.hidden_size, c.num_attention_heads * self.head_dim * gate_mult, bias=c.attention_bias
        )
        self.k_proj = nn.Linear(c.hidden_size, c.num_key_value_heads * self.head_dim, bias=c.attention_bias)
        self.v_proj = nn.Linear(c.hidden_size, c.num_key_value_heads * self.head_dim, bias=c.attention_bias)
        self.o_proj = nn.Linear(c.num_attention_heads * self.head_dim, c.hidden_size, bias=c.attention_bias)
        self.q_norm = RMSNorm(self.head_dim, eps=c.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=c.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        cache: Optional[HybridCache] = None,
        cache_start: int = 0,
        attn_bias: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        b, s, _ = hidden_states.shape
        if self.config.attn_output_gate:
            qg = self.q_proj(hidden_states).view(b, s, self.num_heads, 2 * self.head_dim)
            q, gate = qg.chunk(2, dim=-1)
            gate = gate.reshape(b, s, -1)
        else:
            q = self.q_proj(hidden_states).view(b, s, self.num_heads, self.head_dim)
            gate = None

        q = self.q_norm(q).transpose(1, 2)  # [B, Hq, S, D]
        k = self.k_norm(self.k_proj(hidden_states).view(b, s, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

        if cache is not None:
            k, v = cache.append_kv(self.layer_idx, k, v, cache_start)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias, is_causal=is_causal, scale=self.scaling
        )
        out = out.transpose(1, 2).reshape(b, s, -1)
        if gate is not None:
            out = out * torch.sigmoid(gate)
        return self.o_proj(out)


class MLP(nn.Module):
    """SwiGLU. The fused runtime folds gate_proj+up_proj into one ``[34816, 5120]`` GEMM."""

    def __init__(self, config: QwenFastConfig, intermediate_size: Optional[int] = None):
        super().__init__()
        inter = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, inter, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, inter, bias=False)
        self.down_proj = nn.Linear(inter, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, config: QwenFastConfig, layer_idx: int, block_type: Optional[str] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.block_type = block_type or config.layer_types[layer_idx]
        if self.block_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(config, layer_idx)
        else:
            self.self_attn = Attention(config, layer_idx)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        cache: Optional[HybridCache] = None,
        cache_start: int = 0,
        attn_bias: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        use_fla: bool = True,
    ) -> torch.Tensor:
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        if self.block_type == "linear_attention":
            h = self.linear_attn(h, cache=cache, padding_mask=padding_mask, use_fla=use_fla)
        else:
            h = self.self_attn(
                h,
                position_embeddings=position_embeddings,
                cache=cache,
                cache_start=cache_start,
                attn_bias=attn_bias,
                is_causal=is_causal,
            )
        hidden_states = residual + h
        residual = hidden_states
        h = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(h)
        return hidden_states


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class QwenFastModel(nn.Module):
    def __init__(self, config: QwenFastConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        cache: Optional[HybridCache] = None,
        cache_start: int = 0,
        attn_bias: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        is_causal: Optional[bool] = None,
        use_fla: bool = True,
    ) -> torch.Tensor:
        h = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if is_causal is None:
            # No explicit mask: a multi-token forward starting at cache position
            # 0 is exactly SDPA's top-left-aligned causal case. A continuation
            # (cache_start > 0) is not, so demand an explicit mask there.
            if attn_bias is None and h.shape[1] > 1 and cache_start > 0:
                raise ValueError(
                    "multi-token forward with cache_start > 0 needs an explicit attn_bias"
                )
            is_causal = attn_bias is None and h.shape[1] > 1 and cache_start == 0
        cos, sin = self.rotary_emb(position_ids, dtype=h.dtype)
        for layer in self.layers:
            h = layer(
                h,
                position_embeddings=(cos, sin),
                cache=cache,
                cache_start=cache_start,
                attn_bias=attn_bias,
                padding_mask=padding_mask,
                is_causal=is_causal,
                use_fla=use_fla,
            )
        return self.norm(h)


class MTPHead(nn.Module):
    """Shipped multi-token-prediction head (``mtp.*``).

    ``h_mtp = fc(concat)`` then one full-attention decoder layer, a final norm,
    and the *shared* ``lm_head``.  ``fc``: ``[5120, 10240]``, bf16 even in the
    FP8 checkpoint.

    .. warning::
       **The concat order is not determined by the checkpoint.**  Both orders
       give ``fc.weight`` of shape ``[5120, 10240]``, and the two sources
       disagree: vLLM's ``Qwen3NextMTP`` uses ``[embedding ; hidden]`` (our
       default), while other descriptions of the architecture write
       ``[hidden ; embedding]``.  ``transformers`` does not implement the head
       at all, so there is no tie-breaker on paper.

       Resolve it empirically on a GPU: run both orders and keep the one
       whose draft token agrees with the target model's *next-next* token far
       more often (the right order gives a high acceptance rate; the wrong one
       gives near-chance).  Flip with ``hidden_first=True``.
    """

    def __init__(self, config: QwenFastConfig, hidden_first: bool = False):
        super().__init__()
        self.config = config
        self.hidden_first = hidden_first
        self.pre_fc_norm_embedding = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(config, config.mtp_layer_idx + i, block_type="full_attention")
                for i in range(max(config.mtp_num_hidden_layers, 1))
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        cache: Optional[HybridCache] = None,
        cache_start: int = 0,
        attn_bias: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        e = self.pre_fc_norm_embedding(inputs_embeds)
        p = self.pre_fc_norm_hidden(previous_hidden_states)
        parts = [p, e] if self.hidden_first else [e, p]
        h = self.fc(torch.cat(parts, dim=-1))
        for layer in self.layers:
            h = layer(
                h,
                position_embeddings=position_embeddings,
                cache=cache,
                cache_start=cache_start,
                attn_bias=attn_bias,
                is_causal=is_causal,
            )
        return self.norm(h)


class QwenFastForCausalLM(nn.Module):
    """Text-only Qwen3.8-27B. ``mtp`` is optional and off by default."""

    def __init__(self, config: QwenFastConfig, with_mtp: bool = False,
                 mtp_hidden_first: bool = False):
        super().__init__()
        self.config = config
        self.with_mtp = with_mtp
        self.model = QwenFastModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mtp = MTPHead(config, hidden_first=mtp_hidden_first) if with_mtp else None

    # -- construction ------------------------------------------------------- #
    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        with_mtp: bool = False,
        mtp_hidden_first: bool = False,
        verbose: bool = False,
    ) -> "QwenFastForCausalLM":
        from .weights import SafetensorsStore, load_into_model, resolve_snapshot

        model_dir = resolve_snapshot(model_dir)
        config = QwenFastConfig.from_pretrained(model_dir)
        with torch.device("meta"):
            model = cls(config, with_mtp=with_mtp, mtp_hidden_first=mtp_hidden_first)
        # No `to_empty`: every parameter is assigned straight from the
        # checkpoint, so we never allocate a second copy of the 27B weights.
        store = SafetensorsStore(model_dir, device=device)
        load_into_model(model, store, dtype=dtype, strict=True, verbose=verbose)
        store.close()
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    def make_cache(
        self,
        batch_size: int,
        max_seq_len: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        state_dtype: torch.dtype = torch.float32,
    ) -> HybridCache:
        p0 = next(self.parameters())
        device = device or p0.device
        dtype = dtype or p0.dtype
        return HybridCache(
            self.config, batch_size, max_seq_len, device, dtype, state_dtype, with_mtp=self.with_mtp
        )

    # -- forward ------------------------------------------------------------ #
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        cache: Optional[HybridCache] = None,
        cache_start: int = 0,
        attn_bias: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        is_causal: Optional[bool] = None,
        num_logits: Optional[int] = None,
        return_hidden: bool = False,
        use_fla: bool = True,
    ):
        if position_ids is None:
            b, s = input_ids.shape
            position_ids = torch.arange(s, device=input_ids.device).unsqueeze(0).expand(b, s)
        h = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            cache=cache,
            cache_start=cache_start,
            attn_bias=attn_bias,
            padding_mask=padding_mask,
            is_causal=is_causal,
            use_fla=use_fla,
        )
        h_out = h if num_logits is None else h[:, -num_logits:, :]
        logits = self.lm_head(h_out)
        if return_hidden:
            return logits, h
        return logits


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #
def _left_pad(
    sequences: Sequence[Sequence[int]], pad_id: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Left-pad to a rectangle. Returns ``(input_ids, mask)``, both ``[B, L]``."""
    max_len = max(len(s) for s in sequences)
    ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(sequences), max_len), dtype=torch.bool, device=device)
    for i, s in enumerate(sequences):
        ids[i, max_len - len(s) :] = torch.tensor(list(s), dtype=torch.long, device=device)
        mask[i, max_len - len(s) :] = True
    return ids, mask


class Generator:
    """Prefill + greedy batched decode on top of :class:`QwenFastForCausalLM`.

    Left-padding is safe for the GDN layers because we zero the hidden states of
    pad positions before every linear-attention block: ``k = v = 0`` makes the
    delta update a no-op and the state starts at zero, so the recurrent state is
    bit-identical to a fresh sequence when the first real token arrives.
    """

    def __init__(self, model: QwenFastForCausalLM, pad_id: Optional[int] = None, use_fla: bool = True):
        self.model = model
        self.config = model.config
        self.pad_id = pad_id if pad_id is not None else model.config.eos_token_id
        self.use_fla = use_fla

    @staticmethod
    def _attn_mask(
        valid: torch.Tensor, q_len: int, kv_len: int
    ) -> Tuple[Optional[torch.Tensor], bool]:
        """Boolean SDPA mask ``[B, 1, q_len, kv_len]`` (True = attend).

        Returns ``(mask, is_causal)``.  When nothing is padded we return
        ``(None, q_len > 1)`` and let SDPA use its fused causal path.
        """
        device = valid.device
        live = valid[:, :kv_len]
        if bool(live.all()):
            return None, q_len > 1
        key_ok = live[:, None, None, :]  # [B, 1, 1, kv]
        q_pos = torch.arange(kv_len - q_len, kv_len, device=device)[:, None]
        k_pos = torch.arange(kv_len, device=device)[None, :]
        causal = (k_pos <= q_pos)[None, None]  # [1, 1, q, kv]
        ok = key_ok & causal
        # A left-pad query row would otherwise be fully masked -> softmax NaN,
        # which then poisons real rows through the residual stream. Let every
        # query attend to at least its own position; pad outputs are discarded.
        ok = ok | (k_pos == q_pos)[None, None]
        return ok, False

    @torch.inference_mode()
    def prefill(
        self, prompts: Sequence[Sequence[int]], cache: HybridCache
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns ``(next_token[B], last_logits[B, V], position_ids[B, 1])``."""
        device = next(self.model.parameters()).device
        ids, mask = _left_pad(prompts, self.pad_id, device)
        b, s = ids.shape
        cache.valid[:, :s] = mask
        cache.seq_len = s
        # position ids: 0.. for real tokens, 0 for pads (they are masked anyway)
        pos = (mask.long().cumsum(-1) - 1).clamp(min=0)
        bias, is_causal = self._attn_mask(cache.valid, s, s)
        logits, _hidden = self.model(
            ids,
            position_ids=pos,
            cache=cache,
            cache_start=0,
            attn_bias=bias,
            padding_mask=(None if bool(mask.all()) else mask),
            is_causal=is_causal,
            num_logits=1,
            return_hidden=True,
            use_fla=self.use_fla,
        )
        last = logits[:, -1, :]
        next_pos = pos[:, -1:] + 1
        return last.argmax(-1), last, next_pos

    @torch.inference_mode()
    def decode_step(
        self, tokens: torch.Tensor, position_ids: torch.Tensor, cache: HybridCache
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One greedy step. ``tokens``: ``[B]``, ``position_ids``: ``[B, 1]``."""
        start = cache.seq_len
        cache.valid[:, start] = True
        cache.seq_len = start + 1
        bias, _ = self._attn_mask(cache.valid, 1, cache.seq_len)
        logits = self.model(
            tokens[:, None],
            position_ids=position_ids,
            cache=cache,
            cache_start=start,
            attn_bias=bias,
            padding_mask=None,
            is_causal=False,
            num_logits=1,
            use_fla=self.use_fla,
        )[:, -1, :]
        return logits.argmax(-1), logits

    @torch.inference_mode()
    def generate(
        self,
        prompts: Sequence[Sequence[int]],
        max_new_tokens: int = 32,
        eos_token_id: Optional[int] = None,
        cache: Optional[HybridCache] = None,
    ) -> List[List[int]]:
        device = next(self.model.parameters()).device
        b = len(prompts)
        max_len = max(len(p) for p in prompts) + max_new_tokens
        if cache is None:
            cache = self.model.make_cache(b, max_len, device=device)
        eos = self.config.eos_token_id if eos_token_id is None else eos_token_id
        tok, _, pos = self.prefill(prompts, cache)
        out: List[List[int]] = [[int(t)] for t in tok.tolist()]
        done = torch.zeros(b, dtype=torch.bool, device=device)
        done |= tok == eos
        for _ in range(max_new_tokens - 1):
            if bool(done.all()):
                break
            tok, _ = self.decode_step(tok, pos, cache)
            pos = pos + 1
            for i, t in enumerate(tok.tolist()):
                if not bool(done[i]):
                    out[i].append(int(t))
            done |= tok == eos
        return out


__all__ = [
    "HAS_FLA",
    "RMSNorm",
    "RMSNormGated",
    "RotaryEmbedding",
    "apply_rotary_pos_emb",
    "rotate_half",
    "interleave_mrope",
    "l2norm",
    "causal_conv1d",
    "torch_chunk_gated_delta_rule",
    "torch_recurrent_gated_delta_rule",
    "gated_delta_rule",
    "GatedDeltaNet",
    "Attention",
    "MLP",
    "DecoderLayer",
    "MTPHead",
    "HybridCache",
    "QwenFastModel",
    "QwenFastForCausalLM",
    "Generator",
]
