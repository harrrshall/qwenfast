"""Hand-written Triton kernels for the Qwen3.8-27B GDN decode path.

This module executes ``@triton.jit`` at import time, so it must only ever be
imported from inside a guarded ``try`` (see :mod:`.triton_ops`).  It is
``py_compile``-safe without triton installed.

Design
======

The decode step is a pure read-modify-write over the recurrent state.  The
physics ceiling is **1 state read + 1 state write** and nothing
else matters: at ``B >= 64`` this op *is* the decode step.

``_gdn_decode_kernel``
----------------------
Grid ``(cdiv(V, BV), HV=48, B)`` — **one program per (dv-block, v-head, slot)**.
Each program:

1. resolves its pool row from ``slot_ids[i_n]``  (gather stays *inside* the
   kernel: the state never moves, no ``index_select`` copy, and CUDA graphs see
   a fixed pointer);
2. loads ``q``/``k`` for k-head ``i_hv // 3`` — **GVA-aware**: the 3 v-heads
   that share a k-head each issue the same 512 B load, which the L2 coalesces;
3. loads the ``[128, BV]`` fp32 (or fp16/bf16) state tile **once**;
4. decay ``S' = S exp(g)``, then **two independent reductions** off ``S'``
   (see below), delta, rank-1 update, output — all in fp32 registers;
5. stores the tile **once**.

The output is computed as

.. code::

    S'  = S exp(g)
    kv  = S'^T k            }  independent: both read S', neither waits
    qs  = S'^T q            }
    d   = (v - kv) * beta
    o   = qs + (q.k) d
    S'' = S' + k (x) d      (pure elementwise, no reduction)

rather than the textbook ``S'' = S' + k (x) d ; o = S''^T q``.  The two are
algebraically identical — ``(k (x) d)^T q = (q.k) d`` — but the textbook order
puts **two cross-thread reductions in series** with the rank-1 update between
them, and a cross-thread reduction over the 128-wide K axis is the single most
expensive non-memory thing this kernel does.  ``q.k`` is a 128-element
reduction over registers already held and depends on nothing, so it hides under
the state load.

Why that matters, from H200 measurements.  Fitting
``t = B (bytes_per_seq / BW + c)`` to the fp32/fp16 winners at B=256::

    BW = 4.63 TB/s  = 96% of peak      <- the memory path is essentially done
    c  = 0.253 us / sequence / layer   <- 16% of an fp32 step, 27% of an fp16 one

So "70% of HBM" for fp16 state was never a bandwidth problem: halving the bytes
does not halve ``c``, so the *same* kernel scores worse on a metric whose
denominator shrank.  The only way up for fp16 is to cut ``c``, and ``c`` is
reductions + conversions.  Hence the reformulation above.

Three things in that grid/loop order were measured on an H200, not guessed:

* **Grid order is (dv, head, slot), not (slot, head, dv).**  CUDA issues CTAs
  with ``blockIdx.x`` fastest, so consecutive CTAs must walk *within* one
  slot's 3 MiB layer block.  With batch on ``x`` (the first version) adjacent
  CTAs landed 3 MiB — or 37.7 MiB, on a whole-model pool — apart, which is
  worst-case DRAM page behaviour.
* **The reduction axis is the 128-wide K dim**, the *outer* dim of the
  ``[K, V]`` tile.  That matches the frozen pool layout (and fla's
  ``state_v_first=False``) and keeps V — the dim we tile over — contiguous, so
  every state load and store is a fully coalesced run.
* **Exactly one runtime stride argument** (``stride_sn``).  Everything else is
  derived from ``constexpr`` shapes, because the first version's 31 runtime
  args put ~70 us of host-side launch cost on *every* call, which at B <= 32
  was larger than the kernel itself.  See :mod:`.triton_ops` and the README's
  "launch overhead" section.

.. _kernel-variants:

The variant axes (``PACK_DT`` / ``HOIST`` / ``SCHED``)
------------------------------------------------------
The ``base`` kernel fitted

.. code::

    t/B = bytes(dtype)/BW + c + e(dtype)
    BW = 4155 GB/s (87% of peak)   c = 0.129 us/seq/layer
    e(fp16) = 0.088                e(bf16) = 0.094

``e(fp16) ~= e(bf16)`` is the whole story: bf16->fp32 widening is a *shift*
while fp16->fp32 is a real ``cvt``, so if the half-precision penalty were the
conversion instruction the two would differ.  They do not, which points at the
**16-bit load/store path** (or at anything else that scales with the number of
*elements* rather than bytes) and not at the convert.  The three flags below
are the three candidate mechanisms, kept orthogonal so the sweep can attribute
the win rather than ship a bundle:

``PACK_DT``  (0 = off, 1 = fp16, 2 = bf16)
    Access a 16-bit state pool through an ``int32`` view: each memory
    instruction moves **one 32-bit word = two state elements**, and the
    ``[NK, BV]`` half tile becomes an ``[NK, BV/2]`` word tile.  The pair is
    split into **even / odd** register tiles (``s_lo`` = value column ``2j``,
    ``s_hi`` = ``2j+1``) and every later op is elementwise along V, so nothing
    ever has to be re-interleaved: no shuffle, no shared-memory round trip, and
    the same total register footprint (2 x ``[NK, BV/2]`` fp32).
    For ``PACK_DT=2`` the widening is done with shifts
    (``bf16 -> fp32`` is exactly ``bits << 16``), so the bf16 packed path
    contains **zero** convert instructions on the load side — which makes the
    fp16-vs-bf16 gap under ``PACK_DT`` a direct second measurement of ``e``.
``HOIST``
    Two things, both aimed at ``c``: (a) ``tl.multiple_of`` / ``tl.max_contiguous``
    on the value offsets, which is what lets the vectorizer prove a
    16-byte-aligned contiguous run and emit ``ld.global.v4`` instead of
    narrower accesses; (b) in ``_gdn_window_kernel``, the per-token pointer
    arithmetic is hoisted out of the ``T`` loop and replaced by a constant
    pointer bump — the loop body recomputed a full ``[BV]`` index tensor per
    token, five times over.
``NOMASK``
    Only legal when the host can prove ``NV % BV == 0`` (it always can for the
    model shape: 128 % {8,16,32,64} == 0).  Drops the ``[NK, BV]`` boolean
    tensor from every state load/store — a predicate register per element and,
    worse, the thing most likely to stop the access being merged.  The
    ``slot < 0`` "skip" case degrades to a *uniform* branch around the store
    (the load still happens, from slot 0, and is thrown away), which is the
    documented behaviour: for a negative slot the state is untouched and the
    output row is undefined.  Set together with ``HOIST`` by the wrapper.
``SCHED``
    Issue the state tile load **first**, before q/k/v/g/beta and before the two
    L2 norms, and pull q/k through L1 (``cache_modifier=".ca"``, they are
    re-read by the 3 GVA siblings).  In ``base`` the long-latency load sits behind
    four small loads and two cross-thread reductions; this tests the "the scalar
    loads are on the critical path" hypothesis.

``PACK_DT=0, HOIST=False, NOMASK=False, SCHED=False`` reproduces the original kernel
instruction for instruction — that is the ``base`` variant and the fp32
default, so the frozen fp32 numbers stay comparable.

``_gdn_window_kernel``
----------------------
The same body with a ``T``-step loop, plus the fused verify-and-commit logic: it
keeps a second register tile ``S_commit`` that is refreshed while
``t + 1 <= m[i_n]``.  ``m`` is read from device memory, so the whole
speculative step stays inside one CUDA graph.  ``COMMIT=False`` turns it
into a pure verify (1 read, **0** writes).  ``MASK_PAST_M`` additionally zeroes
``g``/``beta`` past ``m`` *in the kernel*, which is how the two-phase replay pass
avoids materialising a mask tensor on the host side.  It shares the state path,
so it takes the same four variant flags.

``_conv_update_kernel``
-----------------------
Depthwise causal conv + ring shift in one pass.  The ring is read and written
along the **channel** axis, so the pool wants to be **width-major**
(``[n_slots, W-1, C]``): with the channel-major layout the accesses are
stride-``W-1`` gathers of 2-byte elements, i.e. ~2 useful bytes per 32 B
sector, which is exactly why the first version sat at 7-360 GB/s.  Both layouts
are supported (the strides are ``constexpr``); width-major is the default.

Tuning knobs
------------
``BV``          tile width over the value dim.  Also sets the CTA count:
                ``B * 48 * (128/BV)``.  At B=1, BV=128 launches only 48 CTAs
                on 132 SMs — a third of the GPU idle — so BV must shrink with
                the batch.  :func:`.triton_ops.pick_decode_tiling` holds the
                measured table.
``num_warps``   **fewer, fatter warps win** — the opposite of the occupancy
                argument, and it was measured, not reasoned.  The sweep's
                winners are ``(BV=64, 2 warps)`` for B>=16, ``(32, 1)`` for
                2<=B<16, ``(16, 1)`` at B=1: all ~128 tile elements and
                200-220 registers per thread, zero spills, at 25% occupancy.
                ``(64, 8)`` — 32 regs/thread, 100% occupancy — is 25-40%
                *slower*.  The reason is the K-axis reduction: with few threads
                per CTA most of it collapses into an in-register serial
                accumulation, and only a short cross-warp tail remains.
                Occupancy is not the binding constraint; the reduction is.
``EVICT``       ``eviction_policy='evict_first'`` on the state load/store.  The
                state is pure streaming data with no reuse inside a step, so it
                should not displace q/k/v or the conv ring in L2.
``num_stages``  irrelevant here (no software-pipelined inner loop over K).
"""

from __future__ import annotations

import triton
import triton.language as tl

#: ``PACK_DT`` codes.  Kept as plain ints so the host can pass them as a
#: ``constexpr`` without importing triton types.
PACK_NONE = 0
PACK_FP16 = 1
PACK_BF16 = 2


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
@triton.jit
def _l2norm(x, eps):
    """``x / sqrt(sum(x*x) + eps)`` — fla's convention, eps inside the sqrt."""
    return x / tl.sqrt(tl.sum(x * x, axis=0) + eps)


@triton.jit
def _softplus(x):
    """``log1p(exp(x))``, linear past 20 — matches ``F.softplus``'s threshold."""
    xc = tl.minimum(x, 20.0)
    return tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(xc)))


@triton.jit
def _sigmoid(x):
    return 1.0 / (1.0 + tl.exp(-x))


@triton.jit
def _gate(b_g, b_beta, A_LOG, DT_BIAS, i_hv, GATE_IN_KERNEL: tl.constexpr):
    """``g = -exp(A_log)*softplus(a + dt_bias)``, ``beta = sigmoid(b)``.

    With ``GATE_IN_KERNEL`` the caller hands us the **raw** ``a``/``b``
    projections and we fold the gate in, saving two elementwise launches per
    layer.  Otherwise the values are already in log space / already sigmoided.
    """
    if GATE_IN_KERNEL:
        a_log = tl.load(A_LOG + i_hv).to(tl.float32)
        dtb = tl.load(DT_BIAS + i_hv).to(tl.float32)
        b_g = -tl.exp(a_log) * _softplus(b_g + dtb)
        b_beta = _sigmoid(b_beta)
    return b_g, b_beta


# --------------------------------------------------------------------------- #
# packed 16-bit state: one 32-bit word == two state elements
# --------------------------------------------------------------------------- #
@triton.jit
def _unpack2(w, PACK_DT: tl.constexpr):
    """``int32`` word -> ``(lo, hi)`` fp32, the halves in **address** order.

    ``lo`` is the low 16 bits, i.e. the element at the *lower* address (NVIDIA
    is little-endian), so ``lo`` is value column ``2j`` and ``hi`` is ``2j+1``.

    bf16 widening is a pure shift — no ``cvt`` at all — which is why the bf16
    packed path is the cleanest probe of the ``e(dtype)`` term in the roofline.

    Everything stays **signed** int32 on purpose: triton refuses to combine
    signed and unsigned operands of the same width, and every step here is
    sign-agnostic — ``>> 16`` sign-extends but the following truncation to
    ``int16`` throws those bits away again.
    """
    if PACK_DT == 2:  # bf16: widening is `bits << 16`, no cvt at all
        lo = (w << 16).to(tl.float32, bitcast=True)
        hi = ((w >> 16) << 16).to(tl.float32, bitcast=True)
    else:  # fp16
        lo = w.to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32)
        hi = (w >> 16).to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32)
    return lo, hi


@triton.jit
def _pack2(lo, hi, PACK_DT: tl.constexpr):
    """``(lo, hi)`` fp32 -> one ``int32`` word.  Inverse of :func:`_unpack2`.

    The narrowing goes through ``.to(bf16)`` / ``.to(fp16)`` rather than a
    shift so that it **round-to-nearest**s exactly like the unpacked path;
    truncating here would silently change the drift numbers.

    ``.to(tl.int32)`` from ``int16`` sign-extends, so the low half is masked
    before the OR; the high half needs no mask because ``<< 16`` drops the
    extension anyway.
    """
    if PACK_DT == 2:  # bf16
        a = lo.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32)
        b = hi.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32)
    else:  # fp16
        a = lo.to(tl.float16).to(tl.int16, bitcast=True).to(tl.int32)
        b = hi.to(tl.float16).to(tl.int16, bitcast=True).to(tl.int32)
    return (a & 0xFFFF) | (b << 16)


@triton.jit
def _ld_state(p, m, MASKED: tl.constexpr, EVICT: tl.constexpr):
    """State-tile load.  ``MASKED=False`` drops the per-element predicate."""
    if MASKED:
        if EVICT:
            return tl.load(p, mask=m, other=0, eviction_policy="evict_first")
        return tl.load(p, mask=m, other=0)
    if EVICT:
        return tl.load(p, eviction_policy="evict_first")
    return tl.load(p)


@triton.jit
def _st_state(p, x, m, MASKED: tl.constexpr, EVICT: tl.constexpr):
    """State-tile store.  Callers guard the unmasked form with ``if valid:``."""
    if MASKED:
        if EVICT:
            tl.store(p, x, mask=m, eviction_policy="evict_first")
        else:
            tl.store(p, x, mask=m)
    else:
        if EVICT:
            tl.store(p, x, eviction_policy="evict_first")
        else:
            tl.store(p, x)


@triton.jit
def _delta_step(b_s, b_q, b_k, b_v, b_g, b_beta, b_qk):
    """One gated-delta update of one ``[NK, BV]`` fp32 tile -> ``(out, S'')``.

    Factored out so the packed path can run it twice (even columns, odd
    columns) and the window kernel can run it per token.  Inlined by triton, so
    ``base`` still compiles to the v4 instruction sequence.
    """
    b_s = b_s * tl.exp(b_g)
    # Both reductions read S' (the decayed, pre-update state), so they are
    # independent and the compiler can overlap them.  See the module docstring:
    # o = S''^T q + (q.k) d, so the rank-1 update never has to land before the
    # output reduction starts.
    b_kv = tl.sum(b_s * b_k[:, None], axis=0)
    b_qs = tl.sum(b_s * b_q[:, None], axis=0)
    b_d = (b_v - b_kv) * b_beta
    b_o = b_qs + b_qk * b_d
    b_s = b_s + b_k[:, None] * b_d[None, :]
    return b_o, b_s


# --------------------------------------------------------------------------- #
# single-token decode
# --------------------------------------------------------------------------- #
@triton.jit
def _gdn_decode_kernel(
    Q,
    K,
    V,
    G,
    BETA,
    O,
    STATE,
    SLOT,
    A_LOG,
    DT_BIAS,
    QK,
    scale,
    stride_sn,
    H: tl.constexpr,
    HV: tl.constexpr,
    NK: tl.constexpr,
    NV: tl.constexpr,
    GVA: tl.constexpr,
    BV: tl.constexpr,
    L2NORM: tl.constexpr,
    EPS: tl.constexpr,
    EVICT: tl.constexpr,
    GATE_IN_KERNEL: tl.constexpr,
    PRENORM: tl.constexpr,
    PACK_DT: tl.constexpr = 0,
    HOIST: tl.constexpr = False,
    NOMASK: tl.constexpr = False,
    SCHED: tl.constexpr = False,
):
    i_v = tl.program_id(0)
    i_hv = tl.program_id(1)
    i_n = tl.program_id(2)
    i_h = i_hv // GVA

    slot = tl.load(SLOT + i_n).to(tl.int64)
    valid = slot >= 0
    slot_safe = tl.where(valid, slot, 0)

    o_k = tl.arange(0, NK)

    # ------------------------------------------------------------------ #
    # state tile addressing.  Under PACK_DT the pool is addressed through an
    # int32 view: NVW words per row, one word per (2j, 2j+1) value pair.
    # ------------------------------------------------------------------ #
    if PACK_DT > 0:
        o_w = i_v * (BV // 2) + tl.arange(0, BV // 2)
        if HOIST:
            o_w = tl.max_contiguous(tl.multiple_of(o_w, BV // 2), BV // 2)
        p_s = (
            STATE
            + slot_safe * stride_sn
            + i_hv * (NK * (NV // 2))
            + o_k[:, None] * (NV // 2)
            + o_w[None, :]
        )
        if NOMASK:
            m_s = valid
        else:
            m_s = (o_k[:, None] < NK) & (o_w[None, :] < (NV // 2)) & valid
        # value columns this program owns, de-interleaved
        o_ve = 2 * o_w
        o_vo = o_ve + 1
        m_vw = o_w < (NV // 2)
    else:
        o_v = i_v * BV + tl.arange(0, BV)
        if HOIST:
            o_v = tl.max_contiguous(tl.multiple_of(o_v, BV), BV)
        m_v = o_v < NV
        p_s = (
            STATE
            + slot_safe * stride_sn
            + i_hv * (NK * NV)
            + o_k[:, None] * NV
            + o_v[None, :]
        )
        if NOMASK:
            m_s = valid
        else:
            m_s = (o_k[:, None] < NK) & (o_v[None, :] < NV) & valid

    # ------------------------------------------------------------------ #
    # SCHED: issue the long-latency state load before anything else, so the
    # q/k loads and the two L2 norms run underneath it instead of in front.
    # ------------------------------------------------------------------ #
    if SCHED:
        b_sw = _ld_state(p_s, m_s, not NOMASK, EVICT)

    # all activation layouts are contiguous, so every stride is a constexpr
    p_qk = i_n * (H * NK) + i_h * NK
    if SCHED:
        # the 3 GVA siblings re-read this 512 B; keep it in L1
        b_q = tl.load(Q + p_qk + o_k, cache_modifier=".ca").to(tl.float32)
        b_k = tl.load(K + p_qk + o_k, cache_modifier=".ca").to(tl.float32)
    else:
        b_q = tl.load(Q + p_qk + o_k).to(tl.float32)
        b_k = tl.load(K + p_qk + o_k).to(tl.float32)
    if PRENORM:
        # q arrives L2-normalised *and* scaled, k normalised, and q.k alongside:
        # three cross-thread reductions that all GVA*(NV/BV) programs sharing
        # this k-head would otherwise each repeat.  See the module docstring.
        b_qk = tl.load(QK + i_n * H + i_h).to(tl.float32)
    else:
        if L2NORM:
            b_q = _l2norm(b_q, EPS)
            b_k = _l2norm(b_k, EPS)
        b_q = b_q * scale
        # q.k folds the rank-1 update into the output; a 128-element reduction
        # over registers we already hold, so it hides under the state load.
        b_qk = tl.sum(b_q * b_k, axis=0)

    p_v = V + i_n * (HV * NV) + i_hv * NV
    if PACK_DT > 0:
        b_v_lo = tl.load(p_v + o_ve, mask=m_vw, other=0.0).to(tl.float32)
        b_v_hi = tl.load(p_v + o_vo, mask=m_vw, other=0.0).to(tl.float32)
    else:
        b_v = tl.load(p_v + o_v, mask=m_v, other=0.0).to(tl.float32)

    p_g = i_n * HV + i_hv
    b_g = tl.load(G + p_g).to(tl.float32)
    b_beta = tl.load(BETA + p_g).to(tl.float32)
    b_g, b_beta = _gate(b_g, b_beta, A_LOG, DT_BIAS, i_hv, GATE_IN_KERNEL)

    if not SCHED:
        b_sw = _ld_state(p_s, m_s, not NOMASK, EVICT)

    p_o = O + i_n * (HV * NV) + i_hv * NV
    if PACK_DT > 0:
        s_lo, s_hi = _unpack2(b_sw, PACK_DT)
        o_lo, s_lo = _delta_step(s_lo, b_q, b_k, b_v_lo, b_g, b_beta, b_qk)
        o_hi, s_hi = _delta_step(s_hi, b_q, b_k, b_v_hi, b_g, b_beta, b_qk)
        b_out = _pack2(s_lo, s_hi, PACK_DT)
        if NOMASK:
            if valid:
                _st_state(p_s, b_out, m_s, False, EVICT)
        else:
            _st_state(p_s, b_out, m_s, True, EVICT)
        tl.store(p_o + o_ve, o_lo.to(O.dtype.element_ty), mask=m_vw)
        tl.store(p_o + o_vo, o_hi.to(O.dtype.element_ty), mask=m_vw)
    else:
        b_s = b_sw.to(tl.float32)
        b_o, b_s = _delta_step(b_s, b_q, b_k, b_v, b_g, b_beta, b_qk)
        b_out = b_s.to(p_s.dtype.element_ty)
        if NOMASK:
            if valid:
                _st_state(p_s, b_out, m_s, False, EVICT)
        else:
            _st_state(p_s, b_out, m_s, True, EVICT)
        tl.store(p_o + o_v, b_o.to(O.dtype.element_ty), mask=m_v)


# --------------------------------------------------------------------------- #
# multi-token window: decode-many / verify / verify-and-commit (two-phase + fused)
# --------------------------------------------------------------------------- #
@triton.jit
def _gdn_window_kernel(
    Q,
    K,
    V,
    G,
    BETA,
    O,
    STATE,
    SLOT,
    M,
    A_LOG,
    DT_BIAS,
    QK,
    scale,
    stride_sn,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    NK: tl.constexpr,
    NV: tl.constexpr,
    GVA: tl.constexpr,
    BV: tl.constexpr,
    L2NORM: tl.constexpr,
    EPS: tl.constexpr,
    EVICT: tl.constexpr,
    GATE_IN_KERNEL: tl.constexpr,
    PRENORM: tl.constexpr,
    COMMIT: tl.constexpr,
    USE_M: tl.constexpr,
    MASK_PAST_M: tl.constexpr,
    PACK_DT: tl.constexpr = 0,
    HOIST: tl.constexpr = False,
    NOMASK: tl.constexpr = False,
    SCHED: tl.constexpr = False,
):
    i_v = tl.program_id(0)
    i_hv = tl.program_id(1)
    i_n = tl.program_id(2)
    i_h = i_hv // GVA

    slot = tl.load(SLOT + i_n).to(tl.int64)
    valid = slot >= 0
    slot_safe = tl.where(valid, slot, 0)

    o_k = tl.arange(0, NK)

    if PACK_DT > 0:
        o_w = i_v * (BV // 2) + tl.arange(0, BV // 2)
        if HOIST:
            o_w = tl.max_contiguous(tl.multiple_of(o_w, BV // 2), BV // 2)
        p_s = (
            STATE
            + slot_safe * stride_sn
            + i_hv * (NK * (NV // 2))
            + o_k[:, None] * (NV // 2)
            + o_w[None, :]
        )
        if NOMASK:
            m_s = valid
        else:
            m_s = (o_k[:, None] < NK) & (o_w[None, :] < (NV // 2)) & valid
        o_ve = 2 * o_w
        o_vo = o_ve + 1
        m_vw = o_w < (NV // 2)
    else:
        o_v = i_v * BV + tl.arange(0, BV)
        if HOIST:
            o_v = tl.max_contiguous(tl.multiple_of(o_v, BV), BV)
        m_v = o_v < NV
        p_s = (
            STATE
            + slot_safe * stride_sn
            + i_hv * (NK * NV)
            + o_k[:, None] * NV
            + o_v[None, :]
        )
        if NOMASK:
            m_s = valid
        else:
            m_s = (o_k[:, None] < NK) & (o_v[None, :] < NV) & valid

    b_sw = _ld_state(p_s, m_s, not NOMASK, EVICT)
    if PACK_DT > 0:
        b_s_lo, b_s_hi = _unpack2(b_sw, PACK_DT)
    else:
        b_s = b_sw.to(tl.float32)

    if USE_M:
        b_m = tl.load(M + i_n).to(tl.int32)
    else:
        b_m = T

    if COMMIT:
        if PACK_DT > 0:
            b_sc_lo = b_s_lo
            b_sc_hi = b_s_hi
        else:
            b_sc = b_s

    # HOIST: the v4 loop rebuilt five full index expressions per token.  Here
    # the bases are formed once and bumped by a constexpr stride each step.
    if HOIST:
        p_qk_t = Q + i_n * (T * H * NK) + i_h * NK + o_k
        p_kk_t = K + i_n * (T * H * NK) + i_h * NK + o_k
        if PRENORM:  # QK is a dummy pointer otherwise — don't even address it
            p_qkv_t = QK + i_n * (T * H) + i_h
        p_g_t = G + i_n * (T * HV) + i_hv
        p_b_t = BETA + i_n * (T * HV) + i_hv
        p_v_t = V + i_n * (T * HV * NV) + i_hv * NV
        p_o_t = O + i_n * (T * HV * NV) + i_hv * NV

    for t in range(T):
        if HOIST:
            b_q = tl.load(p_qk_t).to(tl.float32)
            b_k = tl.load(p_kk_t).to(tl.float32)
        else:
            p_qk = i_n * (T * H * NK) + t * (H * NK) + i_h * NK
            b_q = tl.load(Q + p_qk + o_k).to(tl.float32)
            b_k = tl.load(K + p_qk + o_k).to(tl.float32)
        if PRENORM:
            if HOIST:
                b_qk = tl.load(p_qkv_t).to(tl.float32)
            else:
                b_qk = tl.load(QK + i_n * (T * H) + t * H + i_h).to(tl.float32)
        else:
            if L2NORM:
                b_q = _l2norm(b_q, EPS)
                b_k = _l2norm(b_k, EPS)
            b_q = b_q * scale
            b_qk = tl.sum(b_q * b_k, axis=0)

        if HOIST:
            if PACK_DT > 0:
                b_v_lo = tl.load(p_v_t + o_ve, mask=m_vw, other=0.0).to(tl.float32)
                b_v_hi = tl.load(p_v_t + o_vo, mask=m_vw, other=0.0).to(tl.float32)
            else:
                b_v = tl.load(p_v_t + o_v, mask=m_v, other=0.0).to(tl.float32)
            b_g = tl.load(p_g_t).to(tl.float32)
            b_beta = tl.load(p_b_t).to(tl.float32)
        else:
            p_v = V + i_n * (T * HV * NV) + t * (HV * NV) + i_hv * NV
            if PACK_DT > 0:
                b_v_lo = tl.load(p_v + o_ve, mask=m_vw, other=0.0).to(tl.float32)
                b_v_hi = tl.load(p_v + o_vo, mask=m_vw, other=0.0).to(tl.float32)
            else:
                b_v = tl.load(p_v + o_v, mask=m_v, other=0.0).to(tl.float32)
            p_g = i_n * (T * HV) + t * HV + i_hv
            b_g = tl.load(G + p_g).to(tl.float32)
            b_beta = tl.load(BETA + p_g).to(tl.float32)
        b_g, b_beta = _gate(b_g, b_beta, A_LOG, DT_BIAS, i_hv, GATE_IN_KERNEL)

        if MASK_PAST_M:
            # past the accepted prefix the recurrence must be the identity:
            # beta = 0 kills the delta update, g = 0 makes the decay 1.
            keep = t < b_m
            b_g = tl.where(keep, b_g, 0.0)
            b_beta = tl.where(keep, b_beta, 0.0)

        if PACK_DT > 0:
            b_o_lo, b_s_lo = _delta_step(
                b_s_lo, b_q, b_k, b_v_lo, b_g, b_beta, b_qk
            )
            b_o_hi, b_s_hi = _delta_step(
                b_s_hi, b_q, b_k, b_v_hi, b_g, b_beta, b_qk
            )
        else:
            b_o, b_s = _delta_step(b_s, b_q, b_k, b_v, b_g, b_beta, b_qk)

        if HOIST:
            p_o = p_o_t
        else:
            p_o = O + i_n * (T * HV * NV) + t * (HV * NV) + i_hv * NV
        if PACK_DT > 0:
            tl.store(p_o + o_ve, b_o_lo.to(O.dtype.element_ty), mask=m_vw)
            tl.store(p_o + o_vo, b_o_hi.to(O.dtype.element_ty), mask=m_vw)
        else:
            tl.store(p_o + o_v, b_o.to(O.dtype.element_ty), mask=m_v)

        if COMMIT:
            # refresh the committed snapshot while this token is accepted;
            # after the loop b_sc == S_m for m = M[i_n] (m = 0 -> untouched).
            keep_c = t + 1 <= b_m
            if PACK_DT > 0:
                b_sc_lo = tl.where(keep_c, b_s_lo, b_sc_lo)
                b_sc_hi = tl.where(keep_c, b_s_hi, b_sc_hi)
            else:
                b_sc = tl.where(keep_c, b_s, b_sc)

        if HOIST:
            p_qk_t += H * NK
            p_kk_t += H * NK
            if PRENORM:
                p_qkv_t += H
            p_g_t += HV
            p_b_t += HV
            p_v_t += HV * NV
            p_o_t += HV * NV

    if COMMIT:
        if PACK_DT > 0:
            b_out = _pack2(b_sc_lo, b_sc_hi, PACK_DT)
        else:
            b_out = b_sc.to(p_s.dtype.element_ty)
        if NOMASK:
            if valid:
                _st_state(p_s, b_out, m_s, False, EVICT)
        else:
            _st_state(p_s, b_out, m_s, True, EVICT)


# --------------------------------------------------------------------------- #
# depthwise causal conv, decode step
# --------------------------------------------------------------------------- #
@triton.jit
def _conv_update_kernel(
    X,
    STATE,
    SLOT,
    W,
    BIAS,
    O,
    stride_sn,
    C,
    STRIDE_SC: tl.constexpr,
    STRIDE_SW: tl.constexpr,
    STRIDE_WC: tl.constexpr,
    STRIDE_WW: tl.constexpr,
    WIDTH: tl.constexpr,
    BC: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SILU: tl.constexpr,
):
    i_c = tl.program_id(0)
    i_n = tl.program_id(1)

    offs = i_c * BC + tl.arange(0, BC)
    m_c = offs < C

    slot = tl.load(SLOT + i_n).to(tl.int64)
    valid = slot >= 0
    slot_safe = tl.where(valid, slot, 0)

    p_s = STATE + slot_safe * stride_sn + offs * STRIDE_SC
    m_s = m_c & valid

    b_x = tl.load(X + i_n * C + offs, mask=m_c, other=0.0).to(tl.float32)

    acc = tl.zeros([BC], dtype=tl.float32)
    for j in tl.static_range(WIDTH - 1):
        b_sj = tl.load(p_s + j * STRIDE_SW, mask=m_s, other=0.0).to(tl.float32)
        b_wj = tl.load(
            W + offs * STRIDE_WC + j * STRIDE_WW, mask=m_c, other=0.0
        ).to(tl.float32)
        acc += b_sj * b_wj
    b_wl = tl.load(
        W + offs * STRIDE_WC + (WIDTH - 1) * STRIDE_WW, mask=m_c, other=0.0
    ).to(tl.float32)
    acc += b_x * b_wl

    if HAS_BIAS:
        acc += tl.load(BIAS + offs, mask=m_c, other=0.0).to(tl.float32)
    if SILU:
        acc = acc * _sigmoid(acc)

    tl.store(O + i_n * C + offs, acc.to(O.dtype.element_ty), mask=m_c)

    # ring shift: state[j] <- state[j+1], state[WIDTH-2] <- x.
    # Each iteration reads index j+1 strictly after writing index j, so no
    # iteration ever reads a slot it has already overwritten.
    for j in tl.static_range(WIDTH - 2):
        b_next = tl.load(p_s + (j + 1) * STRIDE_SW, mask=m_s, other=0.0)
        tl.store(p_s + j * STRIDE_SW, b_next, mask=m_s)
    tl.store(
        p_s + (WIDTH - 2) * STRIDE_SW,
        b_x.to(p_s.dtype.element_ty),
        mask=m_s,
    )


# --------------------------------------------------------------------------- #
# depthwise causal conv, PREFILL (packed varlen, token-major)
# --------------------------------------------------------------------------- #
# Without a prefill conv kernel, the torch fallback
# (`kernels_gdn.torch_ops.conv_prefill`) is the
# single most expensive non-GEMM term in a prefill chunk:
#
#   * it wants `[B, C, T]`, but every activation in `FusedGDN.prefill` is
#     `[T, C]`, so the caller pays **two** `[C=10240, T=8192]` bf16 transposes
#     per GDN layer (336 MB of copy each way, x48 layers = 32 GB per chunk);
#   * `F.conv1d(xin.float(), ...)` upcasts to fp32, so the conv itself moves
#     4x the bytes it needs to, in ~10 eager kernels per tile;
#   * it is tiled (`conv_prefill_tile_tokens`) purely to bound that
#     fp32 working set, which multiplies the launch count by the tile count;
#   * and the packed-varlen path does `cu_seqlens.to("cpu").tolist()` -- one
#     **D2H sync per GDN layer per chunk**, i.e. 48 full pipeline drains, on a
#     path that is already eager and therefore already launch-bound.
#
# This kernel replaces all of that with two launches per layer that read `x`
# once in its native `[T, C]` layout and accumulate in fp32:
#
#   `_conv_prefill_kernel`        the conv itself, grid (C-tiles, T-tiles, N)
#   `_conv_prefill_state_kernel`  the per-sequence ring write-back, grid (C-tiles, N)
#
# They are two kernels rather than one because the write-back is a
# read-modify-write of the *same* rows tile 0 reads as left context: fusing
# them would race whenever a sequence spans more than one T-tile. A second
# launch on the same stream is the cheapest correct ordering.
#
# Exactness: output element `(t, c)` is
# `silu(sum_j w[c, j] * in[t - (W-1) + j, c])` where `in[p]` is `x[p]` for
# `p >= 0` and the ring state `state[slot, p + (W-1)]` for `p < 0` -- the same
# `cat([state, seg])` the torch path materialises, never materialised.


@triton.jit
def _conv_prefill_kernel(
    X,            # [T_total, C] activations, packed varlen
    STATE,        # [n_slots, W-1, C] (or [n_slots, C, W-1]) ring, read-only here
    CU,           # [N+1] int32 cumulative sequence lengths
    SLOT,         # [N] int32 slot id per sequence (-1 == no state)
    W,            # [C, W] conv weight
    BIAS,
    O,            # [T_total, C] output
    stride_sn,
    C,
    STRIDE_SC: tl.constexpr,
    STRIDE_SW: tl.constexpr,
    STRIDE_WC: tl.constexpr,
    STRIDE_WW: tl.constexpr,
    WIDTH: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SILU: tl.constexpr,
):
    i_c = tl.program_id(0)
    i_t = tl.program_id(1)
    i_n = tl.program_id(2)

    lo = tl.load(CU + i_n).to(tl.int64)
    hi = tl.load(CU + i_n + 1).to(tl.int64)
    n = hi - lo
    t0 = i_t * BT
    if t0 >= n:
        return

    offs_c = i_c * BC + tl.arange(0, BC)
    m_c = offs_c < C
    offs_t = t0 + tl.arange(0, BT)
    m_t = offs_t < n

    slot = tl.load(SLOT + i_n).to(tl.int64)
    has_state = slot >= 0
    slot_safe = tl.where(has_state, slot, 0)
    p_state = STATE + slot_safe * stride_sn

    acc = tl.zeros([BT, BC], dtype=tl.float32)
    for j in tl.static_range(WIDTH):
        # input index for tap `j` of output `offs_t`, local to this sequence
        p = offs_t - (WIDTH - 1) + j
        in_x = p >= 0
        # (a) from the chunk itself
        m_x = m_t[:, None] & in_x[:, None] & m_c[None, :]
        v_x = tl.load(
            X + (lo + tl.maximum(p, 0))[:, None] * C + offs_c[None, :],
            mask=m_x,
            other=0.0,
        ).to(tl.float32)
        # (b) from the ring state: `p in [-(W-1), -1]` -> ring index `p + W-1`
        s_idx = p + (WIDTH - 1)
        m_s = m_t[:, None] & (~in_x)[:, None] & m_c[None, :] & has_state
        v_s = tl.load(
            p_state + tl.maximum(s_idx, 0)[:, None] * STRIDE_SW
            + offs_c[None, :] * STRIDE_SC,
            mask=m_s,
            other=0.0,
        ).to(tl.float32)
        b_w = tl.load(
            W + offs_c * STRIDE_WC + j * STRIDE_WW, mask=m_c, other=0.0
        ).to(tl.float32)
        # `in_x` and `~in_x` are disjoint, so exactly one of the two loads is
        # non-zero for every (t, c): the sum *is* the selection.
        acc += (v_x + v_s) * b_w[None, :]

    if HAS_BIAS:
        acc += tl.load(BIAS + offs_c, mask=m_c, other=0.0).to(tl.float32)[None, :]
    if SILU:
        acc = acc * _sigmoid(acc)

    tl.store(
        O + (lo + offs_t)[:, None] * C + offs_c[None, :],
        acc.to(O.dtype.element_ty),
        mask=m_t[:, None] & m_c[None, :],
    )


@triton.jit
def _conv_prefill_state_kernel(
    X,            # [T_total, C]
    STATE,        # [n_slots, W-1, C] ring, read AND written
    CU,           # [N+1]
    SLOT,         # [N]
    stride_sn,
    C,
    STRIDE_SC: tl.constexpr,
    STRIDE_SW: tl.constexpr,
    WIDTH: tl.constexpr,
    BC: tl.constexpr,
):
    """``state[j] <- in[n - (W-1) + j]`` for ``j in [0, W-2)``.

    ``in`` is the same virtual `cat(old_state, x_seq)` the conv reads, so a
    sequence shorter than ``W-1`` correctly keeps the tail of its *old* state.
    Must run after :func:`_conv_prefill_kernel` (separate launch): tile 0 of
    that kernel reads the very rows this one overwrites.
    """
    i_c = tl.program_id(0)
    i_n = tl.program_id(1)

    slot = tl.load(SLOT + i_n).to(tl.int64)
    if slot < 0:
        return
    lo = tl.load(CU + i_n).to(tl.int64)
    hi = tl.load(CU + i_n + 1).to(tl.int64)
    n = hi - lo
    if n <= 0:
        return  # empty segment: the ring is already correct

    offs_c = i_c * BC + tl.arange(0, BC)
    m_c = offs_c < C
    p_state = STATE + slot * stride_sn

    # Written indices after step j are {0..j}; the next read is `n + (j+1)`,
    # and `n >= 1` makes that strictly greater than j. No WAR hazard, so the
    # stores can stay inside the loop instead of buffering W-1 register tiles.
    for j in tl.static_range(WIDTH - 1):
        p = n - (WIDTH - 1) + j
        v_x = tl.load(
            X + (lo + tl.maximum(p, 0)) * C + offs_c,
            mask=m_c & (p >= 0),
            other=0.0,
        ).to(tl.float32)
        v_s = tl.load(
            p_state + (p + (WIDTH - 1)) * STRIDE_SW + offs_c * STRIDE_SC,
            mask=m_c & (p < 0),
            other=0.0,
        ).to(tl.float32)
        tl.store(
            p_state + j * STRIDE_SW + offs_c * STRIDE_SC,
            (v_x + v_s).to(p_state.dtype.element_ty),
            mask=m_c,
        )


__all__ = [
    "PACK_NONE",
    "PACK_FP16",
    "PACK_BF16",
    "_gdn_decode_kernel",
    "_gdn_window_kernel",
    "_conv_update_kernel",
    "_conv_prefill_kernel",
    "_conv_prefill_state_kernel",
]
