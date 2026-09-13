"""Host-side wrappers for the hand-written Triton GDN kernels.

Import is guarded: on a machine without triton/CUDA (e.g. a laptop)
:func:`is_available` returns ``False`` and the API layer falls
back to ``fla`` or ``torch``.

Launch cost is a first-class concern here
-----------------------------------------
An early version measured on an H200 showed a **~70 us floor**
on every ``gdn_decode_step`` call, flat from B=1 to B=32, on top of a clean
2.68 GB/s-marginal bandwidth line whose extrapolated intercept was only 9.4 us.
That gap is host-side: argument marshalling for 31 runtime kernel args plus the
wrapper's own tensor bookkeeping.  So:

* every activation layout is **required contiguous** and its strides are
  ``constexpr``-derived from the shapes — the kernels take **one** runtime
  stride, ``state_pool.stride(0)``;
* the two-phase replay masks ``g``/``beta`` *inside* the kernel (``MASK_PAST_M``)
  instead of building a mask tensor with eager ops;
* ``VALIDATE`` can be switched off once the runtime captures the step into a CUDA
  graph, where the wrapper runs once and the floor disappears entirely.

Tuning knobs are module-level dicts overridable from the environment so they
can be swept on the GPU host without editing code::

    QWENFAST_GDN_DECODE_BV=64 QWENFAST_GDN_DECODE_WARPS=8 python -m ...

The *code path* is a knob too.  ``QWENFAST_GDN_DECODE_VARIANT`` (and
``QWENFAST_GDN_WINDOW_VARIANT``) select any combination of ``packed`` /
``hoist`` / ``sched`` — see :func:`parse_variant` and the variant-axes section of
:mod:`.triton_kernels`::

    QWENFAST_GDN_DECODE_VARIANT=packed_hoist python -m ...

``base`` (the default everywhere) is the original kernel unchanged, which is what
keeps the frozen fp32 numbers comparable.
"""

from __future__ import annotations

import os
import re
from typing import Dict, Optional, Tuple

import torch

from . import shapes
from .torch_ops import normalize_qkv, collapse_gva

_KERNELS = None
_IMPORT_ERROR: Optional[str] = None

#: Set False once the step is CUDA-graph captured to shave the wrapper's
#: per-call shape/stride checks.  Wrong layouts then fail silently, so only do
#: it behind a graph capture that was validated with VALIDATE=True.
VALIDATE = True

#: Populated after each launch with the CompiledKernel, so the bench can report
#: ``n_regs`` / ``n_spills`` without a separate ``ncu`` run.
LAST_COMPILED: dict = {}

#: Populated after each launch with the resolved variant flags of that launch
#: (``{"variant": "packed_hoist", "PACK_DT": 1, ...}``), so the bench can label
#: a row with what actually ran rather than what was requested — a variant that
#: is not applicable (e.g. ``packed`` on an fp32 pool) degrades silently.
LAST_VARIANT: dict = {}


def _load():
    global _KERNELS, _IMPORT_ERROR
    if _KERNELS is not None or _IMPORT_ERROR is not None:
        return
    try:
        import triton  # noqa: F401

        from . import triton_kernels as _k

        _KERNELS = _k
    except Exception as exc:  # pragma: no cover - depends on the host environment
        _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


def is_available() -> bool:
    """Triton importable *and* a CUDA device present."""
    if not torch.cuda.is_available():
        return False
    _load()
    return _KERNELS is not None


def unavailable_reason() -> Optional[str]:
    if not torch.cuda.is_available():
        return "torch.cuda.is_available() is False"
    _load()
    return _IMPORT_ERROR


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.lower() not in ("0", "false", "no", "")


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


# --------------------------------------------------------------------------- #
# kernel variants
# --------------------------------------------------------------------------- #
#: The three orthogonal code-path switches of the decode/window kernels.  See
#: :mod:`.triton_kernels` (``The variant axes``) for what each one
#: actually changes and why it is a candidate for the ``c`` / ``e(dtype)``
#: terms of the roofline fit.
#:
#: ``packed``  16-bit state moved as int32 words (2 elements / instruction),
#:             de-interleaved into even/odd fp32 register tiles.  Silently
#:             ignored for an fp32 pool or an odd ``BV``.
#: ``hoist``   contiguity/alignment hints + (window) pointer math lifted out of
#:             the T loop + unmasked state access when ``NV % BV == 0``.
#: ``sched``   state tile load issued first, q/k pulled through L1.
VARIANT_FLAGS = ("packed", "hoist", "sched")
_VARIANT_SPLIT = re.compile(r"[+_,\s]+")


def parse_variant(name: Optional[str]) -> Dict[str, bool]:
    """``'packed_hoist'`` -> ``{'packed': True, 'hoist': True, 'sched': False}``.

    Accepts ``+``/``_``/``,``-separated flag names in any order, plus the
    aliases ``base`` (nothing, == the original kernel) and ``all``.
    """
    n = (name or "base").strip().lower()
    if n in ("", "base", "v4", "none"):
        return {f: False for f in VARIANT_FLAGS}
    if n in ("all", "full", "everything"):
        return {f: True for f in VARIANT_FLAGS}
    parts = [p for p in _VARIANT_SPLIT.split(n) if p]
    bad = [p for p in parts if p not in VARIANT_FLAGS]
    if bad:
        raise ValueError(
            f"unknown kernel variant flag(s) {bad} in {name!r}; "
            f"pick from {VARIANT_FLAGS} or 'base'/'all'"
        )
    return {f: (f in parts) for f in VARIANT_FLAGS}


def variant_name(flags: Dict[str, bool]) -> str:
    """Inverse of :func:`parse_variant`, canonical order."""
    on = [f for f in VARIANT_FLAGS if flags.get(f)]
    return "_".join(on) if on else "base"


def all_variants() -> Tuple[str, ...]:
    """Every combination, canonical order — what a variant sweep enumerates."""
    out = ["base"]
    for mask in range(1, 1 << len(VARIANT_FLAGS)):
        out.append(
            "_".join(f for i, f in enumerate(VARIANT_FLAGS) if mask & (1 << i))
        )
    return tuple(out)


_SM_COUNT: Optional[int] = None


def sm_count() -> int:
    global _SM_COUNT
    if _SM_COUNT is None:
        try:
            _SM_COUNT = torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).multi_processor_count
        except Exception:  # pragma: no cover
            _SM_COUNT = 132  # H200
    return _SM_COUNT


# --------------------------------------------------------------------------- #
# tuning
# --------------------------------------------------------------------------- #
# BV=0 / warps=0 mean "auto" (see pick_decode_tiling); anything else pins it.
DECODE_TUNING = {
    "BV": _env_int("QWENFAST_GDN_DECODE_BV", 0),
    "num_warps": _env_int("QWENFAST_GDN_DECODE_WARPS", 0),
    "num_stages": _env_int("QWENFAST_GDN_DECODE_STAGES", 1),
    "tiles_per_sm": _env_int("QWENFAST_GDN_DECODE_TILES_PER_SM", 4),
    "max_bv": _env_int("QWENFAST_GDN_DECODE_MAX_BV", 64),
    "min_bv": _env_int("QWENFAST_GDN_DECODE_MIN_BV", 8),
    "evict": _env_flag("QWENFAST_GDN_DECODE_EVICT", True),
    # "" -> ask the table; anything else pins the variant (see VARIANTS)
    "variant": _env_str("QWENFAST_GDN_DECODE_VARIANT", ""),
}
WINDOW_TUNING = {
    "BV": _env_int("QWENFAST_GDN_WINDOW_BV", 0),
    "num_warps": _env_int("QWENFAST_GDN_WINDOW_WARPS", 0),
    "num_stages": _env_int("QWENFAST_GDN_WINDOW_STAGES", 1),
    "tiles_per_sm": _env_int("QWENFAST_GDN_WINDOW_TILES_PER_SM", 4),
    "max_bv": _env_int("QWENFAST_GDN_WINDOW_MAX_BV", 32),
    "min_bv": _env_int("QWENFAST_GDN_WINDOW_MIN_BV", 8),
    "evict": _env_flag("QWENFAST_GDN_WINDOW_EVICT", True),
    "variant": _env_str("QWENFAST_GDN_WINDOW_VARIANT", ""),
}
CONV_TUNING = {
    "BC": _env_int("QWENFAST_GDN_CONV_BC", 0),
    "num_warps": _env_int("QWENFAST_GDN_CONV_WARPS", 0),
    "num_stages": _env_int("QWENFAST_GDN_CONV_STAGES", 1),
    "tiles_per_sm": _env_int("QWENFAST_GDN_CONV_TILES_PER_SM", 2),
}
#: The prefill conv is a different shape problem from the
#: decode conv: `T` is thousands, `B` (== sequences in the chunk) is single
#: digits, so the parallelism has to come from the token axis, not the batch
#: axis. `BT x BC` is the fp32 accumulator tile each program holds in
#: registers -- 32 x 128 fp32 = 16 KiB over 4 warps is 32 regs/thread, which
#: leaves room for the `WIDTH` staged loads. Both overridable for a sweep.
CONV_PREFILL_TUNING = {
    "BT": _env_int("QWENFAST_GDN_CONVP_BT", 32),
    "BC": _env_int("QWENFAST_GDN_CONVP_BC", 128),
    "num_warps": _env_int("QWENFAST_GDN_CONVP_WARPS", 4),
    "num_stages": _env_int("QWENFAST_GDN_CONVP_STAGES", 2),
}


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


# --------------------------------------------------------------------------- #
# Measured tilings, H200, HV=48, K=V=128.  Winner by CUDA-graph time.
#
#   decode fp32   B=1 (16,1) 3.8us | B=8 (32,1) | B>=16 (64,2)  80-82% HBM
#   decode fp16   B=1 (16,1) 3.6us | B=8 (64,1) | B>=16 (64,2)  65-69% HBM
#   window        B<64 (16,1)      | B>=64 (8,1)   189us @B=64, 770us @B=256
#   conv          B<64 (2048,8) 3.5us | B<256 (1024,8) 4.1us | B>=256 (512,4) 8.7us
#
# The shape of the answer — few warps, high registers, low occupancy — is the
# opposite of the occupancy heuristic these replaced, which was 25-40% slower.
# The binding constraint is the K-axis reduction, not occupancy.  See the
# `num_warps` note in triton_kernels.
#
# Rows are (min_batch, tiling), highest threshold first.  A tiling is
# ``(BV, num_warps)`` or ``(BV, num_warps, variant)``, so a future
# sweep can bake a per-batch *code path* next to the per-batch tile without
# touching any of the readers: :func:`pick_decode_tiling` slices the first two
# and :func:`pick_decode_variant` reads the third (falling back to the separate
# variant table when a row does not carry one).
# --------------------------------------------------------------------------- #
DECODE_TABLE = {
    4: ((16, (64, 2)), (2, (32, 1)), (0, (16, 1))),  # fp32 state
    # B=8 (64,1) measured 7.4us and won despite 50 spills — at that batch there
    # is spare bandwidth to hide the spill traffic, and it was measured.
    2: ((16, (64, 2)), (2, (64, 1)), (0, (16, 1))),  # fp16 / bf16 state
}

# --------------------------------------------------------------------------- #
# Variant tables, measured.  Rows are (min_batch, variant), highest threshold first.
#
# The variant sweep covered all 8 combinations x 7 batches x 3 state dtypes,
# graph-timed.  Almost everything came back inside the noise floor, and the
# adoption bar was set at **>= 3% with the same answer from two independent
# dtypes**.  Exactly one entry cleared it:
#
#   B=512, 16-bit state:  base 517.4 us (fp16) / 519.9 (bf16)
#                 packed_sched 489.0        / 489.8      = -5.5% / -5.8%
#                                                           65% -> 69% of HBM
#
# and the ranking was identical for fp16 and bf16 across all 8 variants, so it
# is a real effect, not a lucky draw.  Everything below B=512 stayed within
# 2.5%, which is under this kernel's own run-to-run spread (~4% at B=32, ~13%
# at B=8, measured by re-running the same config in the tuning sweep) — so it
# is *not* measured to be better and is not adopted.
#
# Two entries are deliberately left at ``base`` despite a positive mean:
#   * B=128-256: `sched`/`hoist_sched` are consistently ~2.3% faster in both
#     dtypes.  Real, probably — but under the bar, and a second code path in
#     production has to earn more than that.  One-line change if it ever does.
#   * B<=8 16-bit: every variant is *worse* (B=8 base 7.9 us vs packed 8.9,
#     packed_hoist 13.2) — that batch runs the known-spilling (64,1) tiling and
#     the variants make the spill worse.  Re-tuning (BV, warps) per variant did
#     not rescue it (best packed B=8 = 7.9 us, i.e. a tie at best).
#
# The fp32 row stays ``base`` by decision: those numbers are frozen
# (80-82% of HBM) and the whole variant comparison is against them; no variant beat
# fp32 base by more than 1% at any B >= 32 anyway.
#
# `packed_sched` is **bit-identical** to `base` (TestGpuPackedDecode
# .test_baked_variants_are_bit_identical_to_base), so adopting it cannot move
# a single model output — it is purely a memory-issue-order change.
# --------------------------------------------------------------------------- #
DECODE_VARIANT_TABLE = {
    4: ((0, "base"),),  # fp32 state — frozen
    2: ((512, "packed_sched"), (0, "base")),  # fp16 / bf16
}
# The window (fused verify-and-commit) kernel: `base` won everywhere measured.
# At B=32 the packed variants are 12-13% *slower* (k=1: 54.3 us base vs 61.3
# packed) — WINDOW_TABLE drives BV down to 8-16 to fit two [128, BV] state
# tiles, and halving that again to BV/2 = 4-8 word columns is too narrow to pay
# for the unpack.
WINDOW_VARIANT_TABLE = {
    4: ((0, "base"),),
    2: ((0, "base"),),
}

# Measured (gdn_tuning_window.md).  Note how much smaller these are than the
# decode tilings: the window kernel holds two [128, BV] fp32 tiles, so the
# sweep drove BV down to 8 at large batch — 16 dv-blocks per (slot, head).
# The derived table this replaced (32,2)/(16,1)/(8,1) was wrong at B>=64.
WINDOW_TABLE = {
    4: ((64, (8, 1)), (0, (16, 1))),
    2: ((64, (8, 1)), (0, (16, 1))),
}

# Measured (gdn_tuning_conv.md).  Larger batch wants *smaller* BC: at B=1 the
# CTA count has to come from splitting C, at B=256 it already comes from B.
CONV_TABLE = ((256, (512, 4)), (64, (1024, 8)), (0, (2048, 8)))


def _table_lookup(table: dict, b: int, itemsize: int):
    """Highest ``min_batch`` row that ``b`` clears; last row is the floor."""
    rows = table.get(itemsize) or table[4]
    for min_b, entry in rows:
        if b >= min_b:
            return entry
    return rows[-1][1]


def pick_decode_tiling(
    b: int,
    hv: int,
    nv: int,
    cfg: Optional[dict] = None,
    tiles_factor: int = 8,
    itemsize: int = 4,
    table: Optional[dict] = None,
) -> Tuple[int, int]:
    """``(BV, num_warps)`` for a batch of ``b`` and a state of ``itemsize`` bytes.

    Uses the measured table above when the shape is the model's
    (``HV=48, V=128``) — which is the only shape that was swept.  For any other
    shape it falls back to the analytic rule: shrink ``BV`` until there are
    ``tiles_per_sm`` CTAs per SM (at B=1, BV=128 is 48 CTAs for 132 SMs), then
    set ``num_warps`` from ``BV``.  Explicit ``cfg`` entries always win.
    """
    cfg = cfg or DECODE_TUNING
    if table is None:
        table = WINDOW_TABLE if cfg is WINDOW_TUNING else DECODE_TABLE

    if table and hv == shapes.NUM_V_HEADS and nv == shapes.HEAD_V_DIM:
        bv, nw = _table_lookup(table, b, itemsize)[:2]
        return (cfg["BV"] or bv), (cfg["num_warps"] or nw)

    bv = cfg["BV"] or min(cfg["max_bv"], nv)
    if not cfg["BV"]:
        target = cfg["tiles_per_sm"] * sm_count()
        while bv > cfg["min_bv"] and b * hv * _cdiv(nv, bv) < target:
            bv //= 2
    bv = max(1, min(bv, nv))
    nw = cfg["num_warps"] or max(1, min(16, bv // tiles_factor))
    return bv, nw


def pick_decode_variant(
    b: int,
    hv: int,
    nv: int,
    cfg: Optional[dict] = None,
    itemsize: int = 4,
    table: Optional[dict] = None,
) -> str:
    """Which kernel variant to compile for this ``(batch, state itemsize)``.

    Resolution order: an explicit ``cfg['variant']`` (env
    ``QWENFAST_GDN_DECODE_VARIANT`` / ``_WINDOW_VARIANT``, or whatever the
    sweep pinned) > a variant carried by the tiling table row > the variant
    table > ``base``.  Non-model shapes get ``base``: nothing else was swept.
    """
    cfg = cfg or DECODE_TUNING
    if cfg.get("variant"):
        return cfg["variant"]
    if hv != shapes.NUM_V_HEADS or nv != shapes.HEAD_V_DIM:
        return "base"
    is_window = cfg is WINDOW_TUNING
    tile_table = table if table is not None else (
        WINDOW_TABLE if is_window else DECODE_TABLE
    )
    if tile_table:
        row = _table_lookup(tile_table, b, itemsize)
        if len(row) > 2 and row[2]:
            return row[2]
    vt = WINDOW_VARIANT_TABLE if is_window else DECODE_VARIANT_TABLE
    return _table_lookup(vt, b, itemsize)


# --------------------------------------------------------------------------- #
# packed-state plumbing
# --------------------------------------------------------------------------- #
#: kernel-side ``PACK_DT`` codes (mirrors triton_kernels.PACK_*)
_PACK_CODE = {torch.float16: 1, torch.bfloat16: 2}


def packed_state_view(
    state_pool: torch.Tensor, bv: int
) -> Tuple[torch.Tensor, int, int]:
    """``(tensor, stride0, PACK_DT)`` for the packed 32-bit state path.

    Returns the pool unchanged with ``PACK_DT = 0`` whenever packing does not
    apply — an fp32 pool, an odd ``BV``, an odd row length, or any layout that
    ``Tensor.view(torch.int32)`` refuses (it needs a contiguous last dim, an
    even ``size(-1)``, and even strides / storage offset, all of which the
    frozen pool layout satisfies).  Degrading silently is deliberate: the
    caller asks for a *variant*, and a variant that cannot apply to this pool
    must not become an exception in the decode path.

    The view is a host-side reinterpretation only — no copy, no kernel — and
    under CUDA-graph capture it runs once, so it costs the replay nothing.
    """
    code = _PACK_CODE.get(state_pool.dtype, 0)
    if not code:
        return state_pool, state_pool.stride(0), 0
    if bv % 2 or state_pool.shape[-1] % 2:
        return state_pool, state_pool.stride(0), 0
    try:
        view = state_pool.view(torch.int32)
    except Exception:  # noqa: BLE001 - odd stride/offset: just don't pack
        return state_pool, state_pool.stride(0), 0
    return view, view.stride(0), code


def _variant_args(
    flags: dict, state_pool: torch.Tensor, bv: int, nv: int
) -> Tuple[torch.Tensor, int, int, bool, bool, bool]:
    """-> ``(state_arg, stride_sn, PACK_DT, HOIST, NOMASK, SCHED)``.

    ``NOMASK`` is only ever enabled together with ``hoist`` *and* only when the
    grid provably covers the value axis exactly (``NV % BV == 0``) — otherwise
    the tail block would read and write out of bounds.
    """
    pack_dt = 0
    st = state_pool
    stride = state_pool.stride(0)
    if flags.get("packed"):
        st, stride, pack_dt = packed_state_view(state_pool, bv)
    hoist = bool(flags.get("hoist"))
    nomask = hoist and nv % bv == 0
    return st, stride, pack_dt, hoist, nomask, bool(flags.get("sched"))


def pick_conv_tiling(b: int, c: int, cfg: Optional[dict] = None) -> Tuple[int, int]:
    """``(BC, num_warps)`` from the measured table for C=10240, else the rule."""
    cfg = cfg or CONV_TUNING
    if c == shapes.CONV_DIM:
        for min_b, (bc, nw) in CONV_TABLE:
            if b >= min_b:
                return (cfg["BC"] or bc), (cfg["num_warps"] or nw)
    bc = cfg["BC"] or 1024
    if not cfg["BC"]:
        target = cfg["tiles_per_sm"] * sm_count()
        while bc > 128 and b * _cdiv(c, bc) < target:
            bc //= 2
    nw = cfg["num_warps"] or max(1, min(8, bc // 128))
    return bc, nw


# --------------------------------------------------------------------------- #
# input prep
# --------------------------------------------------------------------------- #
def _prep(q, k, v, g, beta, want_t: bool):
    """-> contiguous ``q,k [B,(T,)H,K]``, ``v [B,(T,)HV,V]``, ``g,beta [B,(T,)HV]``.

    q/k are collapsed back to the 16-head GVA form when the caller pre-expanded
    them, so each k-head's 512 B of q/k is fetched once per 3 v-heads.
    """
    q, k, v, g, beta = normalize_qkv(q, k, v, g, beta)
    hv = v.shape[2]
    if q.shape[2] == hv and hv != shapes.NUM_K_HEADS and hv % shapes.NUM_K_HEADS == 0:
        q = collapse_gva(q, shapes.NUM_K_HEADS)
        k = collapse_gva(k, shapes.NUM_K_HEADS)
    if hv % q.shape[2] != 0:
        raise ValueError(f"HV={hv} is not a multiple of H={q.shape[2]}")
    if not want_t:
        if v.shape[1] != 1:
            raise ValueError(f"decode_step expects T == 1, got T={v.shape[1]}")
        q, k, v, g, beta = q[:, 0], k[:, 0], v[:, 0], g[:, 0], beta[:, 0]
    return (
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        g.float().contiguous(),
        beta.float().contiguous(),
    )


def _check_pool(state_pool: torch.Tensor, slot_ids: torch.Tensor, hv: int, nk: int, nv: int):
    """The kernels derive every state stride but ``stride(0)`` from constexprs."""
    if state_pool.dim() != 4:
        raise ValueError(
            f"state_pool must be a per-layer view [n_slots, HV, K, V], "
            f"got {tuple(state_pool.shape)} — use state.layer_state(pool, layer)"
        )
    if state_pool.shape[1:] != (hv, nk, nv):
        raise ValueError(
            f"state_pool tail {tuple(state_pool.shape[1:])} != (HV, K, V) = "
            f"{(hv, nk, nv)}"
        )
    if state_pool.stride()[1:] != (nk * nv, nv, 1):
        raise ValueError(
            "state_pool must be contiguous within a slot (strides "
            f"{(nk * nv, nv, 1)}), got {state_pool.stride()[1:]}"
        )
    if slot_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"slot_ids must be int32/int64, got {slot_ids.dtype}")


def _slots(slot_ids: torch.Tensor, device) -> torch.Tensor:
    if slot_ids.dtype is torch.int32 and slot_ids.device == device and slot_ids.is_contiguous():
        return slot_ids
    return slot_ids.to(device=device, dtype=torch.int32).contiguous()


def _prenorm_args(qk, want_t: bool, b: int, t: int, h: int):
    """Validate a pre-computed ``q.k`` and return ``(ptr, PRENORM)``."""
    if qk is None:
        return None, False
    want = (b, t, h) if want_t else (b, h)
    qk = qk.float().contiguous()
    if tuple(qk.shape) != want:
        # tolerate a [B, 1, H] for the T==1 path
        if want_t or tuple(qk.shape) != (b, 1, h):
            raise ValueError(f"qk must be {want}, got {tuple(qk.shape)}")
        qk = qk.reshape(b, h)
    return qk, True


def _gate_args(a_log, dt_bias):
    """``(A_LOG, DT_BIAS, GATE_IN_KERNEL)`` — dummy pointers when unused."""
    if a_log is None:
        return None, None, False
    if dt_bias is None:
        raise ValueError("gate-in-kernel needs both A_log and dt_bias")
    return a_log.contiguous(), dt_bias.contiguous(), True


# --------------------------------------------------------------------------- #
def decode_step(
    q,
    k,
    v,
    g,
    beta,
    state_pool,
    slot_ids,
    *,
    scale: Optional[float] = None,
    use_qk_l2norm: bool = True,
    out: Optional[torch.Tensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    qk: Optional[torch.Tensor] = None,
):
    """One decode step, state gathered *inside* the kernel (no pool copy).

    Pass ``qk`` (with q/k already L2-normalised and q pre-scaled — see
    :func:`.torch_ops.prenormalize_qk`) to take the **PRENORM** path, which
    drops three of the kernel's five cross-thread reductions.

    With ``A_log``/``dt_bias`` given, ``g``/``beta`` are read as the **raw**
    ``a``/``b`` projections and the gate
    (``-exp(A_log)*softplus(a+dt_bias)`` / ``sigmoid(b)``) is folded in,
    removing two elementwise launches per layer.
    """
    _load()
    if _KERNELS is None:
        raise RuntimeError(f"triton unavailable: {_IMPORT_ERROR}")

    q, k, v, g, beta = _prep(q, k, v, g, beta, want_t=False)
    b, hv, vd = v.shape
    h, kd = q.shape[1], q.shape[2]
    if VALIDATE:
        _check_pool(state_pool, slot_ids, hv, kd, vd)
    sid = _slots(slot_ids, state_pool.device)
    a_ptr, d_ptr, gate = _gate_args(A_log, dt_bias)
    qk_ptr, prenorm = _prenorm_args(qk, False, b, 1, h)

    o = out
    if o is None:
        o = torch.empty(b, 1, hv, vd, dtype=v.dtype, device=v.device)
    # .view (not .reshape): a non-contiguous `out` must raise here rather than
    # silently get a copy that the kernel then writes into instead of `out`.
    o_flat = o.view(b, hv, vd)

    itemsize = state_pool.element_size()
    bv, nw = pick_decode_tiling(b, hv, vd, itemsize=itemsize)
    flags = parse_variant(pick_decode_variant(b, hv, vd, itemsize=itemsize))
    st_arg, stride_sn, pack_dt, hoist, nomask, sched = _variant_args(
        flags, state_pool, bv, vd
    )
    LAST_VARIANT["decode"] = {
        "requested": variant_name(flags),
        "PACK_DT": pack_dt,
        "HOIST": hoist,
        "NOMASK": nomask,
        "SCHED": sched,
    }
    grid = (_cdiv(vd, bv), hv, b)
    LAST_COMPILED["decode"] = _KERNELS._gdn_decode_kernel[grid](
        q,
        k,
        v,
        g,
        beta,
        o_flat,
        st_arg,
        sid,
        a_ptr if gate else sid,
        d_ptr if gate else sid,
        qk_ptr if prenorm else sid,
        (kd ** -0.5) if scale is None else scale,
        stride_sn,
        H=h,
        HV=hv,
        NK=kd,
        NV=vd,
        GVA=hv // h,
        BV=bv,
        L2NORM=use_qk_l2norm,
        EPS=shapes.L2NORM_EPS,
        EVICT=DECODE_TUNING["evict"],
        GATE_IN_KERNEL=gate,
        PRENORM=prenorm,
        PACK_DT=pack_dt,
        HOIST=hoist,
        NOMASK=nomask,
        SCHED=sched,
        num_warps=nw,
        num_stages=DECODE_TUNING["num_stages"],
    )
    return o


def window(
    q,
    k,
    v,
    g,
    beta,
    state_pool,
    slot_ids,
    m: Optional[torch.Tensor] = None,
    *,
    commit: bool = True,
    mask_past_m: bool = False,
    scale: Optional[float] = None,
    use_qk_l2norm: bool = True,
    out: Optional[torch.Tensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    qk: Optional[torch.Tensor] = None,
):
    """T-token window in one pass: 1 state read, 0-or-1 state write.

    ``commit=False``             -> pure verify (phase A), no state write.
    ``commit=True, m=None``      -> plain multi-token decode (commit all T).
    ``commit=True, m=[B] int32`` -> **fused commit**: outputs for all T positions and the
                                    state written back as ``S_m``, with ``m``
                                    read on the device so the step stays inside
                                    a CUDA graph.
    ``mask_past_m=True``         -> **two-phase, phase B**: zero ``g``/``beta`` past
                                    ``m`` in-kernel, making the recurrence the
                                    identity there.  No mask tensor, no eager
                                    ops, still graph-safe.
    """
    _load()
    if _KERNELS is None:
        raise RuntimeError(f"triton unavailable: {_IMPORT_ERROR}")

    q, k, v, g, beta = _prep(q, k, v, g, beta, want_t=True)
    b, t, hv, vd = v.shape
    h, kd = q.shape[2], q.shape[3]
    if VALIDATE:
        _check_pool(state_pool, slot_ids, hv, kd, vd)
    sid = _slots(slot_ids, state_pool.device)
    a_ptr, d_ptr, gate = _gate_args(A_log, dt_bias)
    qk_ptr, prenorm = _prenorm_args(qk, True, b, t, h)

    use_m = m is not None
    if mask_past_m and not use_m:
        raise ValueError("mask_past_m=True requires m")
    m_t = _slots(m, state_pool.device) if use_m else sid

    o = out
    if o is None:
        o = torch.empty(b, t, hv, vd, dtype=v.dtype, device=v.device)

    itemsize = state_pool.element_size()
    bv, nw = pick_decode_tiling(
        b, hv, vd, WINDOW_TUNING, tiles_factor=4, itemsize=itemsize,
    )
    flags = parse_variant(
        pick_decode_variant(b, hv, vd, WINDOW_TUNING, itemsize=itemsize)
    )
    st_arg, stride_sn, pack_dt, hoist, nomask, sched = _variant_args(
        flags, state_pool, bv, vd
    )
    LAST_VARIANT["window"] = {
        "requested": variant_name(flags),
        "PACK_DT": pack_dt,
        "HOIST": hoist,
        "NOMASK": nomask,
        "SCHED": sched,
    }
    grid = (_cdiv(vd, bv), hv, b)
    LAST_COMPILED["window"] = _KERNELS._gdn_window_kernel[grid](
        q,
        k,
        v,
        g,
        beta,
        o,
        st_arg,
        sid,
        m_t,
        a_ptr if gate else sid,
        d_ptr if gate else sid,
        qk_ptr if prenorm else sid,
        (kd ** -0.5) if scale is None else scale,
        stride_sn,
        T=t,
        H=h,
        HV=hv,
        NK=kd,
        NV=vd,
        GVA=hv // h,
        BV=bv,
        L2NORM=use_qk_l2norm,
        EPS=shapes.L2NORM_EPS,
        EVICT=WINDOW_TUNING["evict"],
        GATE_IN_KERNEL=gate,
        PRENORM=prenorm,
        COMMIT=commit,
        USE_M=use_m,
        MASK_PAST_M=mask_past_m,
        PACK_DT=pack_dt,
        HOIST=hoist,
        NOMASK=nomask,
        SCHED=sched,
        num_warps=nw,
        num_stages=WINDOW_TUNING["num_stages"],
    )
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
    method: str = "fused",
    **kw,
):
    """Fused kernel when ``method='fused'``; two stock passes when ``method='two_phase'``."""
    if method == "fused":
        return window(
            q, k, v, g, beta, state_pool, slot_ids, m,
            commit=True, scale=scale, use_qk_l2norm=use_qk_l2norm, **kw,
        )
    if method != "two_phase":
        raise ValueError(f"unknown verify_and_commit method {method!r}")

    # Two-phase on our own kernels: phase A verify (1 read, 0 writes), phase B replay
    # the accepted prefix.  Both phases are single launches with no eager ops —
    # the g/beta masking happens inside the kernel.
    o = window(
        q, k, v, g, beta, state_pool, slot_ids, None,
        commit=False, scale=scale, use_qk_l2norm=use_qk_l2norm, **kw,
    )
    window(
        q, k, v, g, beta, state_pool, slot_ids, m,
        commit=True, mask_past_m=True, scale=scale,
        use_qk_l2norm=use_qk_l2norm, **kw,
    )
    return o


def commit(
    q, k, v, g, beta, state_pool, slot_ids, m, *, scale=None, use_qk_l2norm=True, **kw
):
    """Two-phase phase B alone: advance the pool to ``S_m``.  1 read, 1 write."""
    window(
        q, k, v, g, beta, state_pool, slot_ids, m,
        commit=True, mask_past_m=True, scale=scale,
        use_qk_l2norm=use_qk_l2norm, **kw,
    )


# --------------------------------------------------------------------------- #
def conv_state_strides(pool: torch.Tensor, c: int) -> Tuple[int, int]:
    """``(stride_channel, stride_width)`` for either conv-pool layout.

    ``[n_slots, W-1, C]`` (width-major, the default) -> ``(1, C)``: the decode
    step touches all C channels for one w index, so this is the coalesced one.
    ``[n_slots, C, W-1]`` (channel-major, the reference model's layout) -> ``(W-1, 1)``: a
    stride-3 gather of 2-byte elements, ~2 useful bytes per 32 B sector.
    """
    if pool.dim() != 3:
        raise ValueError(
            f"conv pool must be a per-layer view, got {tuple(pool.shape)}"
        )
    if pool.shape[2] == c:
        if pool.stride()[1:] != (c, 1):
            raise ValueError("width-major conv pool must be contiguous per slot")
        return 1, c
    if pool.shape[1] == c:
        w1 = pool.shape[2]
        if pool.stride()[1:] != (w1, 1):
            raise ValueError("channel-major conv pool must be contiguous per slot")
        return w1, 1
    raise ValueError(
        f"conv pool {tuple(pool.shape)} matches neither [n, W-1, {c}] nor [n, {c}, W-1]"
    )


def conv_weight_strides(w: torch.Tensor) -> Tuple[int, int]:
    """``(stride_channel, stride_width)`` for ``w`` viewed as ``[C, W]``."""
    return w.stride(0), w.stride(1)


def conv_update(
    x: torch.Tensor,
    conv_state_pool: torch.Tensor,
    slot_ids: torch.Tensor,
    w: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
) -> torch.Tensor:
    _load()
    if _KERNELS is None:
        raise RuntimeError(f"triton unavailable: {_IMPORT_ERROR}")

    squeeze = False
    if x.dim() == 3:
        if x.shape[-1] != 1:
            raise ValueError(f"conv_update expects T == 1, got {x.shape[-1]}")
        x = x[..., 0]
        squeeze = True
    x = x.contiguous()
    c, width = w.shape
    if activation not in ("silu", None, "identity"):
        raise ValueError(f"unsupported conv activation {activation!r}")
    s_sc, s_sw = conv_state_strides(conv_state_pool, c)
    s_wc, s_ww = conv_weight_strides(w)
    sid = _slots(slot_ids, conv_state_pool.device)
    b = x.shape[0]
    o = torch.empty_like(x)

    bc, nw = pick_conv_tiling(b, c)
    grid = (_cdiv(c, bc), b)
    LAST_COMPILED["conv"] = _KERNELS._conv_update_kernel[grid](
        x,
        conv_state_pool,
        sid,
        w,
        bias if bias is not None else x,
        o,
        conv_state_pool.stride(0),
        c,
        STRIDE_SC=s_sc,
        STRIDE_SW=s_sw,
        STRIDE_WC=s_wc,
        STRIDE_WW=s_ww,
        WIDTH=width,
        BC=bc,
        HAS_BIAS=bias is not None,
        SILU=(activation == "silu"),
        num_warps=nw,
        num_stages=CONV_TUNING["num_stages"],
    )
    return o.unsqueeze(-1) if squeeze else o


def conv_prefill_varlen(
    x: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens: torch.Tensor,
    conv_state_pool: torch.Tensor,
    slot_ids: torch.Tensor,
    *,
    max_seqlen: int,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
) -> torch.Tensor:
    """Packed-varlen depthwise causal conv over a **token-major** chunk.

    ``x``: ``[T_total, C]`` (the layout ``FusedGDN.prefill`` already has, so
    this entry point costs zero transposes) · ``w``: ``[C, W]`` ·
    ``cu_seqlens``: ``[N+1]`` int32 on device · ``slot_ids``: ``[N]`` int32.
    Returns ``[T_total, C]`` and advances each sequence's ring state.

    ``max_seqlen`` is the longest segment in the chunk and is passed **from
    the host** on purpose: the scheduler already knows it
    (``PrefillBatch.q_lens``), and reading it off ``cu_seqlens`` here would
    reintroduce exactly the per-layer D2H sync this kernel exists to delete
    (``torch_ops.conv_prefill``'s ``cu_seqlens.to("cpu")``).
    """
    _load()
    if _KERNELS is None:
        raise RuntimeError(f"triton unavailable: {_IMPORT_ERROR}")
    if x.dim() != 2:
        raise ValueError(f"conv_prefill_varlen expects [T, C], got {tuple(x.shape)}")
    if activation not in ("silu", None, "identity"):
        raise ValueError(f"unsupported conv activation {activation!r}")

    c, width = w.shape
    if x.shape[1] != c:
        raise ValueError(f"x has {x.shape[1]} channels, weight has {c}")
    x = x.contiguous()  # the kernels hard-code the [T, C] row stride as C
    n_seq = int(cu_seqlens.numel()) - 1
    o = torch.empty_like(x)
    if n_seq <= 0 or x.shape[0] == 0:
        return o

    s_sc, s_sw = conv_state_strides(conv_state_pool, c)
    s_wc, s_ww = conv_weight_strides(w)
    cu = cu_seqlens.to(device=x.device, dtype=torch.int32)
    sid = _slots(slot_ids, conv_state_pool.device)

    bt = CONV_PREFILL_TUNING["BT"]
    bc = CONV_PREFILL_TUNING["BC"]
    nw = CONV_PREFILL_TUNING["num_warps"]
    grid = (_cdiv(c, bc), max(_cdiv(int(max_seqlen), bt), 1), n_seq)
    LAST_COMPILED["conv_prefill"] = _KERNELS._conv_prefill_kernel[grid](
        x,
        conv_state_pool,
        cu,
        sid,
        w,
        bias if bias is not None else x,
        o,
        conv_state_pool.stride(0),
        c,
        STRIDE_SC=s_sc,
        STRIDE_SW=s_sw,
        STRIDE_WC=s_wc,
        STRIDE_WW=s_ww,
        WIDTH=width,
        BT=bt,
        BC=bc,
        HAS_BIAS=bias is not None,
        SILU=(activation == "silu"),
        num_warps=nw,
        num_stages=CONV_PREFILL_TUNING["num_stages"],
    )
    # Second launch, not a fused epilogue: see `_conv_prefill_state_kernel`.
    LAST_COMPILED["conv_prefill_state"] = _KERNELS._conv_prefill_state_kernel[
        (_cdiv(c, bc), n_seq)
    ](
        x,
        conv_state_pool,
        cu,
        sid,
        conv_state_pool.stride(0),
        c,
        STRIDE_SC=s_sc,
        STRIDE_SW=s_sw,
        WIDTH=width,
        BC=bc,
        num_warps=nw,
        num_stages=1,
    )
    return o


def kernel_stats(name: str) -> dict:
    """``n_regs`` / ``n_spills`` / ``shared`` of the last launch, if exposed."""
    ck = LAST_COMPILED.get(name)
    if ck is None:
        return {}
    out = {}
    for attr in ("n_regs", "n_spills", "shared", "num_warps", "num_ctas"):
        v = getattr(ck, attr, None)
        if v is not None:
            out[attr] = v
    md = getattr(ck, "metadata", None)
    for attr in ("num_warps", "num_stages", "shared"):
        v = getattr(md, attr, None)
        if v is not None:
            out.setdefault(attr, v)
    out.update(LAST_VARIANT.get(name) or {})
    return out


__all__ = [
    "VALIDATE",
    "DECODE_TABLE",
    "WINDOW_TABLE",
    "CONV_TABLE",
    "DECODE_VARIANT_TABLE",
    "WINDOW_VARIANT_TABLE",
    "DECODE_TUNING",
    "WINDOW_TUNING",
    "CONV_TUNING",
    "LAST_COMPILED",
    "LAST_VARIANT",
    "VARIANT_FLAGS",
    "parse_variant",
    "variant_name",
    "all_variants",
    "packed_state_view",
    "sm_count",
    "pick_decode_tiling",
    "pick_decode_variant",
    "pick_conv_tiling",
    "conv_state_strides",
    "conv_weight_strides",
    "kernel_stats",
    "is_available",
    "unavailable_reason",
    "decode_step",
    "window",
    "verify_and_commit",
    "commit",
    "conv_update",
]
