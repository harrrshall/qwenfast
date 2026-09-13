"""Pure-PyTorch backend (CPU-runnable reference) for the GDN kernels.

Deliberately written *independently* of ``qwenfast.model``'s reference oracle rather
than importing it: the point of ``tests/test_kernels_gdn.py`` is to compare two
implementations that were derived separately.  The two differ in one visible
way — the UT transform here is a triangular solve
(``torch.linalg.solve_triangular``) where the oracle runs the equivalent
Neumann forward-substitution loop.

Numerics contract
-----------------
* q/k L2-normalised with ``x / sqrt(sum(x*x) + 1e-6)`` — the ``fla`` convention
  (eps *inside* the sqrt), identical to the oracle's ``rsqrt(sum + eps)``.
* ``scale = 1/sqrt(128)`` applied to q **after** the L2 norm.
* ``g`` arrives already in log space (``-exp(A_log) * softplus(a + dt_bias)``);
  ``beta`` already sigmoided.
* All recurrence accumulation in fp32 regardless of the state pool's dtype.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from . import shapes
from .state import gather_states, scatter_states

# --------------------------------------------------------------------------- #
# input normalisation
# --------------------------------------------------------------------------- #
def default_scale(q: torch.Tensor) -> float:
    """``1/sqrt(head_k_dim)`` = 1/sqrt(128) for this model; fla's default too."""
    return float(q.shape[-1]) ** -0.5


def prenormalize_qk(q, k, scale=None):
    """``(q_scaled, k_normed, qk)`` — hoist the per-token reductions out.

    The Triton kernels normally do three cross-thread reductions per program
    per token: ``l2norm(q)``, ``l2norm(k)`` and ``q.k``.  Every one of the
    ``GVA * (NV/BV)`` programs that share a k-head repeats all three — 24x at
    the fp32 B=1 tiling, 48x at the window kernel's B>=64 tiling — and on this
    kernel a cross-thread reduction is latency-bound, so a 128-element one
    costs nearly what an 8192-element one does.  (Evidence: an earlier kernel traded one
    128-element reduction for removing a serial dependency between the two
    8192-element ones, and came out exactly even.)

    Computing them once per (row, k-head) instead and passing the results in
    leaves the kernel with two reductions instead of five.  ``q``/``k`` are
    ``[B, (T,) H, K]``; the returned ``qk`` is ``[B, (T,) H]``.

    Pass the three results to any entry point as ``q=``, ``k=``, ``qk=`` with
    ``use_qk_l2norm=False``.  The engine should eventually fold this into the
    conv epilogue that produces q/k in the first place.
    """
    qn = l2norm(q.float()) * (default_scale(q) if scale is None else scale)
    kn = l2norm(k.float())
    return qn, kn, (qn * kn).sum(-1)


def apply_gate(
    a_raw: torch.Tensor,
    b_raw: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(g, beta)`` from the raw projections, all in fp32.

    ``g = -exp(A_log) * softplus(a + dt_bias)`` (log space), ``beta =
    sigmoid(b)``.  This is what ``gate_in_kernel=True`` folds into the Triton
    kernel; here it is the reference the tests compare against.
    """
    g = -A_log.float().exp() * F.softplus(a_raw.float() + dt_bias.float())
    return g, b_raw.float().sigmoid()


def maybe_gate(g, beta, A_log, dt_bias):
    """Pass through, or apply :func:`apply_gate` when ``A_log`` is given."""
    if A_log is None:
        return g, beta
    if dt_bias is None:
        raise ValueError("gate-in-kernel needs both A_log and dt_bias")
    return apply_gate(g, beta, A_log, dt_bias)


def l2norm(x: torch.Tensor, eps: float = shapes.L2NORM_EPS) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def normalize_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Accept ``[B, H, D]`` (implicit T=1) or ``[B, T, H, D]``; return 4-D/3-D."""
    if q.dim() == 3:
        q = q.unsqueeze(1)
    if k.dim() == 3:
        k = k.unsqueeze(1)
    if v.dim() == 3:
        v = v.unsqueeze(1)
    if g.dim() == 2:
        g = g.unsqueeze(1)
    if beta.dim() == 2:
        beta = beta.unsqueeze(1)
    return q, k, v, g, beta


def expand_gva(x: torch.Tensor, num_v_heads: int) -> torch.Tensor:
    """``[B, T, H, D]`` -> ``[B, T, HV, D]`` by ``repeat_interleave(HV // H)``.

    v-head ``hv`` uses k-head ``hv // (HV // H)``, matching both the HF
    reference (``repeat_interleave``) and fla's ``i_h = i_hv // (HV // H)``.
    """
    h = x.shape[-2]
    if h == num_v_heads:
        return x
    if num_v_heads % h != 0:
        raise ValueError(f"HV={num_v_heads} not divisible by H={h}")
    return x.repeat_interleave(num_v_heads // h, dim=-2)


def collapse_gva(x: torch.Tensor, num_k_heads: int) -> torch.Tensor:
    """Inverse of :func:`expand_gva` (takes the group representative)."""
    h = x.shape[-2]
    if h == num_k_heads:
        return x
    rep = h // num_k_heads
    return x[..., ::rep, :]


# --------------------------------------------------------------------------- #
# recurrent (decode) form
# --------------------------------------------------------------------------- #
def recurrent_gdn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
    use_qk_l2norm: bool = True,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Sequential gated-delta-rule.

    ``q, k``: ``[B, T, H, K]`` (H may be 16 or 48) · ``v``: ``[B, T, HV, V]``
    ``g, beta``: ``[B, T, HV]`` · ``initial_state``: ``[B, HV, K, V]``.
    Returns ``(out [B, T, HV, V], final_state [B, HV, K, V] | None)``.
    """
    q, k, v, g, beta = normalize_qkv(q, k, v, g, beta)
    out_dtype = v.dtype
    hv = v.shape[2]
    if use_qk_l2norm:
        q = l2norm(q.float())
        k = l2norm(k.float())
    q = expand_gva(q.float(), hv)
    k = expand_gva(k.float(), hv)
    v = v.float()
    g = g.float()
    beta = beta.float()

    b, t, _, kd = q.shape
    vd = v.shape[-1]
    q = q * (default_scale(q) if scale is None else scale)

    if initial_state is None:
        s = torch.zeros(b, hv, kd, vd, dtype=torch.float32, device=q.device)
    else:
        s = initial_state.float().clone()

    out = torch.empty(b, t, hv, vd, dtype=torch.float32, device=q.device)
    for i in range(t):
        k_t = k[:, i]  # [B, HV, K]
        s = s * g[:, i, :, None, None].exp()
        kv_mem = (s * k_t.unsqueeze(-1)).sum(-2)  # [B, HV, V]
        delta = (v[:, i] - kv_mem) * beta[:, i].unsqueeze(-1)
        s = s + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, i] = (s * q[:, i].unsqueeze(-1)).sum(-2)

    return out.to(out_dtype), (s if output_final_state else None)


# --------------------------------------------------------------------------- #
# chunked (prefill) form
# --------------------------------------------------------------------------- #
def _chunk_dense(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    chunk_size: int,
    scale: float,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Chunked delta rule on a *dense* ``[B, T, HV, D]`` batch (fp32 in/out)."""
    b, t, hv, kd = k.shape
    vd = v.shape[-1]
    c = chunk_size
    pad = (c - t % c) % c

    # -> [B, HV, T, D]
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))
    beta = beta.transpose(1, 2)
    g = g.transpose(1, 2)
    if pad:
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
        beta = F.pad(beta, (0, pad))
        g = F.pad(g, (0, pad))
    tt = t + pad
    nc = tt // c
    q = q * scale

    k_beta = k * beta.unsqueeze(-1)
    v_beta = v * beta.unsqueeze(-1)

    def blk(x):
        return x.reshape(b, hv, nc, c, x.shape[-1])

    q, k, v, k_beta, v_beta = (blk(x) for x in (q, k, v, k_beta, v_beta))
    g = g.reshape(b, hv, nc, c).cumsum(-1)  # cumulative log-decay inside chunk

    # decay_mask[i, j] = exp(g_i - g_j) for j <= i, else 0.  The mask has to be
    # applied *before* the exp: for j > i the exponent is +|.| and overflows to
    # inf, and inf * 0 is NaN.  (The oracle's `.tril().exp().tril()` is the same
    # trick.)
    tri = torch.tril(torch.ones(c, c, dtype=torch.bool, device=q.device))
    decay = (g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp() * tri

    # UT transform: T = (I + strict_tril(k_beta k^T . decay))^-1
    lower = (k_beta @ k.transpose(-1, -2)) * decay
    lower = lower * torch.tril(
        torch.ones(c, c, dtype=lower.dtype, device=q.device), diagonal=-1
    )
    eye = torch.eye(c, dtype=lower.dtype, device=q.device).expand_as(lower)
    tmat = torch.linalg.solve_triangular(
        eye + lower, eye, upper=False, left=True, unitriangular=True
    )

    v_hat = tmat @ v_beta
    k_cumdecay = tmat @ (k_beta * g.exp().unsqueeze(-1))

    if initial_state is None:
        s = torch.zeros(b, hv, kd, vd, dtype=torch.float32, device=q.device)
    else:
        s = initial_state.float().clone()

    out = torch.empty_like(v)
    for i in range(nc):
        q_i, k_i = q[:, :, i], k[:, :, i]
        g_i = g[:, :, i]  # [B, HV, C]
        attn = (q_i @ k_i.transpose(-1, -2)) * decay[:, :, i]
        v_new = v_hat[:, :, i] - k_cumdecay[:, :, i] @ s
        out[:, :, i] = (q_i * g_i.unsqueeze(-1).exp()) @ s + attn @ v_new
        g_last = g_i[..., -1]
        s = s * g_last[..., None, None].exp() + (
            k_i * (g_last[..., None] - g_i).exp().unsqueeze(-1)
        ).transpose(-1, -2) @ v_new

    out = out.reshape(b, hv, tt, vd)[:, :, :t].transpose(1, 2).contiguous()
    return out, (s if output_final_state else None)


def chunk_gdn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
    use_qk_l2norm: bool = True,
    scale: Optional[float] = None,
    chunk_size: int = shapes.DEFAULT_CHUNK_SIZE,
    cu_seqlens_cpu: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Chunked prefill.  ``cu_seqlens`` requires ``B == 1`` (packed varlen).

    ``cu_seqlens_cpu`` is accepted and ignored: this path already loops over
    segments on the host, so it has no device->host read to avoid. It is in the
    signature only so :func:`api.gdn_prefill_chunked` can call ``mod.chunk_gdn``
    with one keyword set regardless of which backend ``mod`` is.
    """
    q, k, v, g, beta = normalize_qkv(q, k, v, g, beta)
    out_dtype = v.dtype
    hv = v.shape[2]
    if use_qk_l2norm:
        q = l2norm(q.float())
        k = l2norm(k.float())
    q = expand_gva(q.float(), hv)
    k = expand_gva(k.float(), hv)
    v, g, beta = v.float(), g.float(), beta.float()
    sc = default_scale(q) if scale is None else scale

    if cu_seqlens is None:
        out, s = _chunk_dense(
            q, k, v, g, beta, initial_state, output_final_state, chunk_size, sc
        )
        return out.to(out_dtype), s

    if q.shape[0] != 1:
        raise ValueError(
            f"batch size must be 1 when cu_seqlens is given, got {q.shape[0]}"
        )
    cu = cu_seqlens.detach().to("cpu").tolist()
    n = len(cu) - 1
    if initial_state is not None and initial_state.shape[0] != n:
        raise ValueError(
            f"initial_state must have {n} rows (len(cu_seqlens)-1), "
            f"got {initial_state.shape[0]}"
        )
    out = torch.empty(1, q.shape[1], hv, v.shape[-1], dtype=torch.float32, device=q.device)
    finals = []
    for i in range(n):
        lo, hi = int(cu[i]), int(cu[i + 1])
        if hi == lo:
            finals.append(
                initial_state[i : i + 1].float()
                if initial_state is not None
                else torch.zeros(
                    1, hv, q.shape[-1], v.shape[-1], dtype=torch.float32, device=q.device
                )
            )
            continue
        s0 = None if initial_state is None else initial_state[i : i + 1]
        o_i, s_i = _chunk_dense(
            q[:, lo:hi],
            k[:, lo:hi],
            v[:, lo:hi],
            g[:, lo:hi],
            beta[:, lo:hi],
            s0,
            True,
            chunk_size,
            sc,
        )
        out[:, lo:hi] = o_i
        finals.append(s_i)
    final = torch.cat(finals, dim=0) if output_final_state else None
    return out.to(out_dtype), final


# --------------------------------------------------------------------------- #
# pool-aware entry points
# --------------------------------------------------------------------------- #
def decode_step(
    q, k, v, g, beta, state_pool, slot_ids, *, scale=None, use_qk_l2norm=True,
    out=None, A_log=None, dt_bias=None,
):
    """One decode step against the pool; ``state_pool`` updated in place."""
    g, beta = maybe_gate(g, beta, A_log, dt_bias)
    s0 = gather_states(state_pool, slot_ids, torch.float32)
    o, s1 = recurrent_gdn(
        q, k, v, g, beta, s0, True, use_qk_l2norm=use_qk_l2norm, scale=scale
    )
    scatter_states(state_pool, slot_ids, s1)
    if out is not None:
        out.copy_(o.reshape(out.shape).to(out.dtype))
        return out
    return o


def verify_and_commit(
    q,
    k,
    v,
    g,
    beta,
    state_pool,
    slot_ids,
    m,
    *,
    scale=None,
    use_qk_l2norm=True,
    method: str = "two_phase",
    A_log=None,
    dt_bias=None,
):
    """See :func:`..api.gdn_verify_and_commit`.

    ``method='two_phase'``  — run the window for the outputs, then run it
    again with ``g``/``beta`` zeroed past ``m`` (which makes the recurrence the
    identity there, so the final state is exactly ``S_m``).  2 state reads +
    1 write, and **no host sync** — ``m`` stays on the device.

    ``method='fused'``      — the semantics the fused Triton kernel implements:
    one pass, snapshotting the state when ``t + 1 == m``.  1 read + 1 write.
    """
    g, beta = maybe_gate(g, beta, A_log, dt_bias)
    q, k, v, g, beta = normalize_qkv(q, k, v, g, beta)
    n = v.shape[1]
    s0 = gather_states(state_pool, slot_ids, torch.float32)
    m = m.to(device=s0.device)

    if method == "two_phase":
        o, _ = recurrent_gdn(
            q, k, v, g, beta, s0, False, use_qk_l2norm=use_qk_l2norm, scale=scale
        )
        t_idx = torch.arange(n, device=s0.device)
        keep = (t_idx[None, :] < m[:, None].to(t_idx.dtype)).to(torch.float32)
        _, s1 = recurrent_gdn(
            q,
            k,
            v,
            g.float() * keep[:, :, None],
            beta.float() * keep[:, :, None],
            s0,
            True,
            use_qk_l2norm=use_qk_l2norm,
            scale=scale,
        )
        scatter_states(state_pool, slot_ids, s1)
        return o

    if method != "fused":
        raise ValueError(f"unknown verify_and_commit method {method!r}")

    # single-pass reference for the fused kernel
    hv = v.shape[2]
    qq = l2norm(q.float()) if use_qk_l2norm else q.float()
    kk = l2norm(k.float()) if use_qk_l2norm else k.float()
    qq = expand_gva(qq, hv) * (default_scale(qq) if scale is None else scale)
    kk = expand_gva(kk, hv)
    vv, gg, bb = v.float(), g.float(), beta.float()
    s = s0.clone()
    s_c = s0.clone()
    out = torch.empty(v.shape[0], n, hv, v.shape[-1], dtype=torch.float32, device=s.device)
    for t in range(n):
        s = s * gg[:, t, :, None, None].exp()
        kv_mem = (s * kk[:, t].unsqueeze(-1)).sum(-2)
        delta = (vv[:, t] - kv_mem) * bb[:, t].unsqueeze(-1)
        s = s + kk[:, t].unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, t] = (s * qq[:, t].unsqueeze(-1)).sum(-2)
        take = (m > t).view(-1, 1, 1, 1)
        s_c = torch.where(take, s, s_c)
    scatter_states(state_pool, slot_ids, s_c)
    return out.to(v.dtype)


# --------------------------------------------------------------------------- #
# causal conv
# --------------------------------------------------------------------------- #
def conv_pool_wc(pool: torch.Tensor, c: int) -> bool:
    """``True`` when the pool is width-major ``[n_slots, W-1, C]``."""
    if pool.shape[-1] == c:
        return True
    if pool.shape[-2] == c:
        return False
    raise ValueError(
        f"conv pool {tuple(pool.shape)} matches neither [n, W-1, {c}] nor [n, {c}, W-1]"
    )


def _conv_gather(pool: torch.Tensor, idx: torch.Tensor, wc: bool) -> torch.Tensor:
    """-> ``[B, C, W-1]`` regardless of the pool's storage order."""
    st = pool.index_select(0, idx)
    return st.transpose(1, 2) if wc else st


def _conv_scatter(pool: torch.Tensor, idx: torch.Tensor, new_cw: torch.Tensor, wc: bool):
    """``new_cw``: ``[B, C, W-1]``."""
    v = new_cw.transpose(1, 2) if wc else new_cw
    pool.index_copy_(0, idx, v.contiguous().to(pool.dtype))


def conv_update(
    x: torch.Tensor,
    conv_state_pool: torch.Tensor,
    slot_ids: torch.Tensor,
    w: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
) -> torch.Tensor:
    """Decode-step depthwise causal conv.  ``x``: ``[B, C]`` or ``[B, C, 1]``.

    ``conv_state_pool``: ``[n_slots, W-1, C]`` (width-major, preferred) or
    ``[n_slots, C, W-1]`` (channel-major); detected from the shape and
    ring-shifted in place.  ``w``: ``[C, W]``.  Returns ``x``'s shape.
    """
    squeeze = False
    if x.dim() == 2:
        x = x.unsqueeze(-1)
        squeeze = True
    c = w.shape[0]
    wc = conv_pool_wc(conv_state_pool, c)
    idx = slot_ids.to(device=conv_state_pool.device, dtype=torch.long)
    st = _conv_gather(conv_state_pool, idx, wc)  # [B, C, W-1]
    xs = torch.cat([st, x.to(st.dtype)], dim=-1)  # [B, C, W]
    _conv_scatter(conv_state_pool, idx, xs[..., 1:], wc)
    y = (xs.float() * w.float().unsqueeze(0)).sum(-1)  # [B, C]
    if bias is not None:
        y = y + bias.float()
    if activation == "silu":
        y = F.silu(y)
    elif activation not in (None, "identity"):
        raise ValueError(f"unsupported conv activation {activation!r}")
    y = y.to(x.dtype)
    return y if squeeze else y.unsqueeze(-1)


def conv_prefill(
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    cu_seqlens: Optional[torch.Tensor] = None,
    conv_state_pool: Optional[torch.Tensor] = None,
    slot_ids: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    tile_tokens: Optional[int] = None,
) -> torch.Tensor:
    """Prefill depthwise causal conv.  ``x``: ``[B, C, T]``.

    ``cu_seqlens`` requires ``B == 1`` (packed varlen).  When
    ``conv_state_pool``/``slot_ids`` are given, the trailing ``W-1`` inputs of
    each sequence are written back to the pool.

    ``tile_tokens`` (``None`` = untiled) caps how many timesteps are upcast to
    fp32 at once. The conv itself is cheap, but ``F.conv1d(xin.float(), ...)``
    materialises ``[n, C, t+W-1]`` fp32 *in* and ``[n, C, t]`` fp32 *out*: at
    serving shapes (C=10240, an 8192-token chunk) that is 671 MiB of
    transient per GDN layer, enough to OOM a loaded server. Because a causal depthwise conv over a tile only needs the
    previous ``W-1`` inputs as left context, tiling is *exact* -- the ring
    state carried between tiles is the same tensor the pool write-back uses --
    so this is a pure peak-memory reduction, not an approximation.
    """
    c, width = w.shape
    b = x.shape[0]
    wc = conv_pool_wc(conv_state_pool, c) if conv_state_pool is not None else False
    tile = int(tile_tokens) if tile_tokens else 0

    def _conv_tile(xin: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        y = F.conv1d(xin.float(), w.float().unsqueeze(1), None, padding=0, groups=c)
        if bias is not None:
            y = y + bias.float().view(1, -1, 1)
        if activation == "silu":
            y = F.silu(y)
        elif activation not in (None, "identity"):
            raise ValueError(f"unsupported conv activation {activation!r}")
        return y.to(out_dtype)

    def _one(seg: torch.Tensor, st: Optional[torch.Tensor]) -> torch.Tensor:
        # seg: [n, C, t], st: [n, C, W-1] or None
        if st is None:
            st = torch.zeros(
                seg.shape[0], c, width - 1, dtype=seg.dtype, device=seg.device
            )
        st = st.to(seg.dtype)
        t = seg.shape[-1]
        if tile <= 0 or t <= tile:
            return _conv_tile(torch.cat([st, seg], dim=-1), seg.dtype)
        out = torch.empty_like(seg)
        left = st
        for lo in range(0, t, tile):
            hi = min(lo + tile, t)
            piece = seg[..., lo:hi]
            out[..., lo:hi] = _conv_tile(torch.cat([left, piece], dim=-1), seg.dtype)
            # Next tile's left context: the last W-1 inputs seen so far. Taken
            # from the concatenation so a tile shorter than W-1 still works.
            left = torch.cat([left, piece], dim=-1)[..., -(width - 1):]
        return out

    if cu_seqlens is None:
        st = None
        if initial_state is not None:
            st = initial_state
        elif conv_state_pool is not None and slot_ids is not None:
            idx = slot_ids.to(device=conv_state_pool.device, dtype=torch.long)
            st = _conv_gather(conv_state_pool, idx, wc)
        y = _one(x, st)
        if conv_state_pool is not None and slot_ids is not None:
            idx = slot_ids.to(device=conv_state_pool.device, dtype=torch.long)
            base = st if st is not None else torch.zeros(
                b, c, width - 1, dtype=x.dtype, device=x.device
            )
            tail = torch.cat([base.to(x.dtype), x], dim=-1)[..., -(width - 1) :]
            _conv_scatter(conv_state_pool, idx, tail, wc)
        return y

    if b != 1:
        raise ValueError("batch size must be 1 when cu_seqlens is given")
    cu = cu_seqlens.detach().to("cpu").tolist()
    n = len(cu) - 1
    y = torch.empty_like(x)
    tails = []
    for i in range(n):
        lo, hi = int(cu[i]), int(cu[i + 1])
        st_i = None
        if initial_state is not None:
            st_i = initial_state[i : i + 1]
        elif conv_state_pool is not None and slot_ids is not None:
            sid = slot_ids[i].to(device=conv_state_pool.device, dtype=torch.long)
            st_i = _conv_gather(conv_state_pool, sid.view(1), wc)
        if hi == lo:
            tails.append(
                st_i
                if st_i is not None
                else torch.zeros(1, c, width - 1, dtype=x.dtype, device=x.device)
            )
            continue
        seg = x[:, :, lo:hi]
        y[:, :, lo:hi] = _one(seg, st_i)
        base = (
            st_i
            if st_i is not None
            else torch.zeros(1, c, width - 1, dtype=x.dtype, device=x.device)
        )
        tails.append(torch.cat([base.to(seg.dtype), seg], dim=-1)[..., -(width - 1) :])
    if conv_state_pool is not None and slot_ids is not None:
        idx = slot_ids.to(device=conv_state_pool.device, dtype=torch.long)
        _conv_scatter(conv_state_pool, idx, torch.cat(tails, dim=0), wc)
    return y


def conv_prefill_varlen(
    x: torch.Tensor,
    w: torch.Tensor,
    seq_lens: Sequence[int],
    conv_state_pool: Optional[torch.Tensor] = None,
    slot_ids: Optional[torch.Tensor] = None,
    *,
    initial_state: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
) -> torch.Tensor:
    """Token-major reference for :func:`triton_ops.conv_prefill_varlen`.

    ``x``: ``[T_total, C]``; ``seq_lens`` is the **host** list of segment
    lengths (``PrefillBatch.q_lens``). Value-identical to :func:`conv_prefill`
    on the transposed input -- ``TestConvPrefillVarlen`` pins that -- with two
    differences that are the point of it:

    * it takes ``[T, C]``, so the caller does not transpose 336 MB twice per
      GDN layer; and
    * it takes the segment lengths from the host, so there is no
      ``cu_seqlens.to("cpu")`` D2H sync per layer.
    """
    c, width = w.shape
    if x.dim() != 2 or x.shape[1] != c:
        raise ValueError(f"expected [T, {c}], got {tuple(x.shape)}")
    if activation not in ("silu", None, "identity"):
        raise ValueError(f"unsupported conv activation {activation!r}")
    wc = conv_pool_wc(conv_state_pool, c) if conv_state_pool is not None else False
    out = torch.empty_like(x)
    tails: List[torch.Tensor] = []
    wf = w.float().unsqueeze(1)
    lo = 0
    for i, n in enumerate(seq_lens):
        n = int(n)
        hi = lo + n
        st = None
        if initial_state is not None:
            st = initial_state[i : i + 1]
        elif conv_state_pool is not None and slot_ids is not None:
            sid = slot_ids[i].to(device=conv_state_pool.device, dtype=torch.long)
            st = _conv_gather(conv_state_pool, sid.view(1), wc)  # [1, C, W-1]
        if st is None:
            st = torch.zeros(1, c, width - 1, dtype=x.dtype, device=x.device)
        st = st.to(x.dtype)
        if n == 0:
            tails.append(st)
            lo = hi
            continue
        seg = x[lo:hi].t().unsqueeze(0)  # [1, C, n]
        xin = torch.cat([st, seg], dim=-1)
        y = F.conv1d(xin.float(), wf, None, padding=0, groups=c)
        if bias is not None:
            y = y + bias.float().view(1, -1, 1)
        if activation == "silu":
            y = F.silu(y)
        out[lo:hi] = y.to(x.dtype)[0].t()
        tails.append(xin[..., -(width - 1) :])
        lo = hi
    if conv_state_pool is not None and slot_ids is not None and tails:
        idx = slot_ids.to(device=conv_state_pool.device, dtype=torch.long)
        _conv_scatter(conv_state_pool, idx, torch.cat(tails, dim=0), wc)
    return out


def conv_verify_and_commit(
    x: torch.Tensor,
    conv_state_pool: torch.Tensor,
    slot_ids: torch.Tensor,
    w: torch.Tensor,
    m: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
) -> torch.Tensor:
    """Conv over an ``n``-token draft window, committing only the first ``m``.

    ``x``: ``[B, C, n]``.  Cheap by construction (conv rollback is a
    pointer move): the whole window plus ring is ``W-1+n`` values per channel.
    """
    c, width = w.shape
    n = x.shape[-1]
    wc = conv_pool_wc(conv_state_pool, c)
    idx = slot_ids.to(device=conv_state_pool.device, dtype=torch.long)
    st = _conv_gather(conv_state_pool, idx, wc)  # [B, C, W-1]
    xin = torch.cat([st.to(x.dtype), x], dim=-1)  # [B, C, W-1+n]
    y = F.conv1d(xin.float(), w.float().unsqueeze(1), None, padding=0, groups=c)
    if bias is not None:
        y = y + bias.float().view(1, -1, 1)
    if activation == "silu":
        y = F.silu(y)
    elif activation not in (None, "identity"):
        raise ValueError(f"unsupported conv activation {activation!r}")

    # commit: the ring after m accepted tokens is xin[..., m : m + W - 1]
    pos = torch.arange(width - 1, device=x.device)[None, None, :] + m.to(
        device=x.device, dtype=torch.long
    ).view(-1, 1, 1)
    new_state = torch.gather(xin, 2, pos.expand(x.shape[0], c, width - 1))
    _conv_scatter(conv_state_pool, idx, new_state, wc)
    _ = n
    return y.to(x.dtype)


__all__ = [
    "default_scale",
    "prenormalize_qk",
    "apply_gate",
    "maybe_gate",
    "conv_pool_wc",
    "l2norm",
    "normalize_qkv",
    "expand_gva",
    "collapse_gva",
    "recurrent_gdn",
    "chunk_gdn",
    "decode_step",
    "verify_and_commit",
    "conv_update",
    "conv_prefill",
    "conv_verify_and_commit",
]
