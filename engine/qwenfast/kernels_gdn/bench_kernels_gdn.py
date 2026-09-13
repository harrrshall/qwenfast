#!/usr/bin/env python3
"""GPU microbenchmark for the GDN kernels.

Measures, per backend and per state dtype, one **decode step of one GDN layer**
and extrapolates to the 48-layer model step:

* ``mean_us`` / ``p50_us`` / ``p99_us`` per layer
* ``achieved_gbps`` against the **ideal** traffic (1 state read + 1 state
  write).  A backend that needs extra copies scores below 100% by exactly the
  factor it wastes, which is the number the custom kernels exist to move.
* ``x48_layers_ms`` and ``pct_of_step_budget`` against the corrected physics
  model of a decode step::

      step_ms(B, ctx) = 6.19 + B * (0.0629 + 6.83e-6 * ctx)

Also benched: the causal-conv decode step, the chunked prefill, and the two
speculative verify-and-commit paths (two-phase vs fused), so the
"SSM traffic multiplier <= 1.5x / <= 1.15x" targets for speculative decoding
are directly measurable.

Never raises on a missing optional backend — each variant is wrapped and
records its error string instead.  ``py_compile``-safe with no torch installed
(every torch import is inside a function body).

Example (on the GPU host)::

    source /home/venv_vllm/bin/activate
    PYTHONPATH=/home/engine python -m qwenfast.kernels_gdn.bench_kernels_gdn \\
        --out /home/qwenfast-results/gdn_kernels.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
if __package__ in (None, ""):  # allow `python bench_kernels_gdn.py`
    sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))
    from qwenfast.kernels_gdn import api, fla_ops, shapes, triton_ops
    from qwenfast.kernels_gdn import state as st, torch_ops
else:
    from . import api, fla_ops, shapes, torch_ops, triton_ops
    from . import state as st


# --------------------------------------------------------------------------- #
# physics model
# --------------------------------------------------------------------------- #
WEIGHT_MS = 6.19
SSM_MS_PER_SEQ_FP32 = 0.0629
KV_MS_PER_SEQ_PER_TOK = 6.83e-6


def physics_step_ms(batch: int, ctx: int, state_itemsize: int = 4) -> float:
    ssm = SSM_MS_PER_SEQ_FP32 * (state_itemsize / 4.0)
    return WEIGHT_MS + batch * (ssm + KV_MS_PER_SEQ_PER_TOK * ctx)


def physics_ssm_ms_48(batch: int, state_itemsize: int, bw_gbps: float) -> float:
    """Ideal ms for 1 read + 1 write of the whole 48-layer SSM state."""
    b = 2 * shapes.state_bytes(batch, state_itemsize, shapes.NUM_GDN_LAYERS)
    return b / (bw_gbps * 1e9) * 1000.0


# --------------------------------------------------------------------------- #
# timing
# --------------------------------------------------------------------------- #
def cuda_time(fn: Callable[[], Any], warmup: int, iters: int) -> Dict[str, float]:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    us = sorted(starts[i].elapsed_time(ends[i]) * 1000.0 for i in range(iters))
    return {
        "mean_us": statistics.fmean(us),
        "p50_us": us[len(us) // 2],
        "p99_us": us[min(len(us) - 1, int(0.99 * len(us)))],
        "min_us": us[0],
        "iters": iters,
    }


def graph_time(fn: Callable[[], Any], warmup: int, iters: int) -> Dict[str, Any]:
    """True GPU time per call, by capturing ``fn`` into a CUDA graph.

    This is *the* number that matters.  ``cuda_time`` brackets each eager call
    with events, so if the host cannot enqueue faster than the GPU drains, it
    measures the **host**.  The first H200 run showed exactly that: a ~70 us
    floor flat from B=1 to B=32 on a bandwidth line whose intercept was 9.4 us.
    The engine runs the whole decode step from a graph, so graph time is also what
    the engine will actually see.
    """
    import torch

    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(side)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        for _ in range(max(3, warmup)):
            g.replay()
        torch.cuda.synchronize()
        st_e = torch.cuda.Event(enable_timing=True)
        en_e = torch.cuda.Event(enable_timing=True)
        st_e.record()
        for _ in range(iters):
            g.replay()
        en_e.record()
        torch.cuda.synchronize()
        per = st_e.elapsed_time(en_e) * 1000.0 / iters
        del g
        return {"graph_us": per, "graph_status": "ok"}
    except Exception as exc:  # noqa: BLE001
        return {"graph_us": None, "graph_status": f"{type(exc).__name__}: {exc}"}


def safe(fn: Callable[[], Dict[str, Any]], label: str) -> Dict[str, Any]:
    try:
        r = fn()
        r.setdefault("status", "ok")
        return r
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "label": label,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=6),
        }


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
DTYPES = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}


def _torch_dtype(name: str):
    import torch

    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


def make_inputs(b: int, t: int, device: str, act_dtype, seed: int = 0):
    """Post-conv GDN inputs at the real shape: q/k 16 heads, v/g/beta 48."""
    import torch
    import torch.nn.functional as F

    gen = torch.Generator(device="cpu").manual_seed(seed)

    def rn(*shape):
        return torch.randn(*shape, generator=gen).to(device=device, dtype=torch.float32)

    q = rn(b, t, shapes.NUM_K_HEADS, shapes.HEAD_K_DIM).to(act_dtype)
    k = rn(b, t, shapes.NUM_K_HEADS, shapes.HEAD_K_DIM).to(act_dtype)
    v = rn(b, t, shapes.NUM_V_HEADS, shapes.HEAD_V_DIM).to(act_dtype)
    a_log = torch.log(
        torch.empty(shapes.NUM_V_HEADS).uniform_(0.01, 16, generator=gen)
    ).to(device)
    dt_bias = torch.ones(shapes.NUM_V_HEADS, device=device)
    beta = rn(b, t, shapes.NUM_V_HEADS).sigmoid()
    g = -a_log.float().exp() * F.softplus(rn(b, t, shapes.NUM_V_HEADS) + dt_bias)
    return dict(q=q, k=k, v=v, g=g, beta=beta)


def make_pool(n_slots: int, device: str, dtype_name: str, seed: int = 1):
    """A scratch state pool.  **Generated on the device.**

    At B=512 the pool is 1024 slots x 3 MiB = 3.2 GB; filling that from a CPU
    generator cost ~20 s of setup *per benched configuration*, which the kernel
    variant sweep multiplies by 8.  The values are irrelevant to the timing
    (it is a pure streaming read-modify-write), only the shape and dtype
    matter, so the generator moves to the device and stays seeded.
    """
    import torch

    try:
        gen = torch.Generator(device=device).manual_seed(seed)
        p = torch.randn(
            n_slots, shapes.NUM_V_HEADS, shapes.HEAD_K_DIM, shapes.HEAD_V_DIM,
            generator=gen, device=device,
        ) * 0.02
    except (RuntimeError, TypeError):  # no device generator (cpu-only run)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        p = torch.randn(
            n_slots, shapes.NUM_V_HEADS, shapes.HEAD_K_DIM, shapes.HEAD_V_DIM,
            generator=gen,
        ) * 0.02
        p = p.to(device=device)
    return p.to(_torch_dtype(dtype_name))


def make_slots(b: int, n_slots: int, device: str):
    import torch

    # deliberately scattered: the pool is never compacted
    idx = (torch.arange(b) * 7 + 3) % n_slots
    return idx.to(device=device, dtype=torch.int32)


# --------------------------------------------------------------------------- #
# benchmarks
# --------------------------------------------------------------------------- #
class _pin_variant:
    """Pin ``cfg['variant']`` for the duration of a timed block.

    The kernel variants (``packed`` / ``hoist`` / ``sched``, see
    :mod:`.triton_kernels`) are compile-time constexprs, so switching one is a
    fresh JIT — it has to be pinned across *both* the warmup and the graph
    capture, not just the first call.
    """

    def __init__(self, cfg: dict, variant: str):
        self.cfg, self.variant = cfg, variant
        self.saved = cfg.get("variant", "")

    def __enter__(self):
        self.cfg["variant"] = self.variant
        return self

    def __exit__(self, *exc):
        self.cfg["variant"] = self.saved
        return False


def _variant_row(name: str, key: str) -> Dict[str, Any]:
    """``variant`` (what was asked) + ``variant_effective`` (what compiled)."""
    eff = dict(triton_ops.LAST_VARIANT.get(key) or {})
    return {
        "variant": name or "table",
        "variant_effective": eff,
        # a variant that could not apply (e.g. packed on an fp32 pool) degrades
        # silently in the wrapper — flag it so the table is not misread
        "variant_applied": (
            None if not eff else (
                (eff.get("PACK_DT", 0) > 0) == ("packed" in (name or ""))
            )
        ),
    }


def bench_decode(
    backend: str, b: int, dtype_name: str, args, device: str, variant: str = ""
) -> Dict[str, Any]:
    import torch

    itemsize = _torch_dtype(dtype_name).itemsize
    act = torch.float32 if dtype_name == "fp32" else _torch_dtype(dtype_name)
    n_slots = max(b, 1) * 2
    inp = make_inputs(b, 1, device, act)
    pool = make_pool(n_slots, device, dtype_name)
    slots = make_slots(b, n_slots, device)
    out = torch.empty(
        b, 1, shapes.NUM_V_HEADS, shapes.HEAD_V_DIM, device=device, dtype=act
    )
    kw: Dict[str, Any] = {}
    if getattr(args, "prenorm", False):
        qn, kn, qk = torch_ops.prenormalize_qk(inp["q"], inp["k"])
        inp = dict(inp, q=qn.to(act), k=kn.to(act))
        kw["qk"] = qk

    def step():
        return api.gdn_decode_step(
            **inp, state_pool=pool, slot_ids=slots, backend=backend, out=out, **kw
        )

    with _pin_variant(triton_ops.DECODE_TUNING, variant):
        step()
        with torch.no_grad():
            t = cuda_time(step, args.warmup, args.iters)
            gt = graph_time(step, args.warmup, args.iters) if not args.no_graph else {}
        t.update(gt)
        vrow = _variant_row(variant, "decode")

    ideal = 2 * shapes.state_bytes(b, itemsize)
    # prefer graph time: it is the GPU-only cost, and it is what the engine runs
    best_us = t.get("graph_us") or t["mean_us"]
    best_s = best_us * 1e-6
    x48 = best_us * shapes.NUM_GDN_LAYERS / 1000.0
    ideal48 = physics_ssm_ms_48(b, itemsize, args.gpu_bw_gbps)
    bv, nw = (
        triton_ops.pick_decode_tiling(b, shapes.NUM_V_HEADS, shapes.HEAD_V_DIM)
        if backend == "triton" and triton_ops.is_available()
        else (None, None)
    )
    return {
        "op": "decode_step",
        "backend": backend,
        "state_dtype": dtype_name,
        "batch": b,
        **t,
        **(vrow if backend == "triton" else {"variant": "n/a"}),
        "timed_on": "graph" if t.get("graph_us") else "eager",
        "prenorm": bool(getattr(args, "prenorm", False)),
        "launch_overhead_us": (
            t["mean_us"] - t["graph_us"] if t.get("graph_us") else None
        ),
        "BV": bv,
        "num_warps": nw,
        "kernel_stats": triton_ops.kernel_stats("decode") if backend == "triton" else {},
        "ideal_state_bytes_rw": ideal,
        "achieved_gbps": ideal / best_s / 1e9,
        "pct_of_hbm_peak": 100.0 * (ideal / best_s / 1e9) / args.gpu_bw_gbps,
        "eager_gbps": ideal / (t["mean_us"] * 1e-6) / 1e9,
        "x48_layers_ms": x48,
        "physics_ssm_ms_48": ideal48,
        "state_bw_efficiency_pct": 100.0 * ideal48 / x48 if x48 > 0 else float("nan"),
        "physics_step_ms": physics_step_ms(b, args.ctx, itemsize),
        "pct_of_step_budget": 100.0 * x48 / physics_step_ms(b, args.ctx, itemsize),
    }


def bench_decode_raw_fla(b: int, dtype_name: str, args, device: str) -> Dict[str, Any]:
    """fla with a *contiguous* state: separates kernel quality from the pool tax."""
    import torch

    if not fla_ops.is_available():
        return {"status": "unavailable", "reason": fla_ops.unavailable_reason()}
    itemsize = _torch_dtype(dtype_name).itemsize
    act = torch.float32 if dtype_name == "fp32" else _torch_dtype(dtype_name)
    inp = make_inputs(b, 1, device, act)
    s = make_pool(b, device, "fp32")

    def step():
        o, ns = fla_ops.recurrent_gdn(**inp, initial_state=s, output_final_state=True)
        return o, ns

    step()
    with torch.no_grad():
        t = cuda_time(step, args.warmup, args.iters)
    ideal = 2 * shapes.state_bytes(b, 4)
    mean_s = t["mean_us"] * 1e-6
    return {
        "op": "decode_step_no_pool",
        "backend": "fla_raw",
        "state_dtype": "fp32",
        "requested_state_dtype": dtype_name,
        "batch": b,
        **t,
        "ideal_state_bytes_rw": ideal,
        "achieved_gbps": ideal / mean_s / 1e9,
        "pct_of_hbm_peak": 100.0 * (ideal / mean_s / 1e9) / args.gpu_bw_gbps,
        "x48_layers_ms": t["mean_us"] * shapes.NUM_GDN_LAYERS / 1000.0,
        "note": "no slot indexing; fla always returns an fp32 state, so "
                f"dtype={dtype_name} is not exercised here (itemsize {itemsize})",
    }


def bench_conv(b: int, args, device: str, layout: str = "width_major") -> Dict[str, Any]:
    """Conv decode step.  ``layout`` picks the ring-pool storage order — the
    channel-major variant is kept so the coalescing win is measurable, not
    asserted (it was 7-360 GB/s on the first run)."""
    import torch

    backend = "triton" if triton_ops.is_available() else "torch"
    c, w = shapes.CONV_DIM, shapes.CONV_KERNEL
    n_slots = max(b, 1) * 2
    x = torch.randn(b, c, device=device, dtype=torch.bfloat16)
    weight = (torch.randn(c, w, device=device) * 0.1).to(torch.bfloat16)
    if layout == "width_major":
        weight = st.prepare_conv_weight(weight)
        pool = torch.randn(n_slots, w - 1, c, device=device, dtype=torch.bfloat16)
    else:
        pool = torch.randn(n_slots, c, w - 1, device=device, dtype=torch.bfloat16)
    slots = make_slots(b, n_slots, device)

    def step():
        return api.causal_conv_update(x, pool, slots, weight, backend=backend)

    step()
    with torch.no_grad():
        t = cuda_time(step, args.warmup, args.iters)
        t.update(graph_time(step, args.warmup, args.iters) if not args.no_graph else {})
    best_us = t.get("graph_us") or t["mean_us"]
    # x + out + ring read + ring write + weights
    nbytes = (2 * b * c + 2 * b * c * (w - 1) + c * w) * 2
    bc, nw = triton_ops.pick_conv_tiling(b, c)
    return {
        "op": "causal_conv_update",
        "backend": backend,
        "layout": layout,
        "batch": b,
        **t,
        "timed_on": "graph" if t.get("graph_us") else "eager",
        "launch_overhead_us": (
            t["mean_us"] - t["graph_us"] if t.get("graph_us") else None
        ),
        "BC": bc,
        "num_warps": nw,
        "kernel_stats": triton_ops.kernel_stats("conv") if backend == "triton" else {},
        "approx_bytes_rw": nbytes,
        "achieved_gbps": nbytes / (best_us * 1e-6) / 1e9,
        "x48_layers_ms": best_us * shapes.NUM_GDN_LAYERS / 1000.0,
    }


def bench_spec(
    backend: str, method: str, b: int, dtype_name: str, k: int, args, device: str,
    variant: str = "",
) -> Dict[str, Any]:
    import torch

    itemsize = _torch_dtype(dtype_name).itemsize
    act = torch.float32 if dtype_name == "fp32" else _torch_dtype(dtype_name)
    n = k + 1
    n_slots = max(b, 1) * 2
    inp = make_inputs(b, n, device, act)
    pool = make_pool(n_slots, device, dtype_name)
    slots = make_slots(b, n_slots, device)
    m = torch.full((b,), max(k, 1), dtype=torch.int32, device=device)
    kw: Dict[str, Any] = {}
    if getattr(args, "prenorm", False):
        qn, kn, qk = torch_ops.prenormalize_qk(inp["q"], inp["k"])
        inp = dict(inp, q=qn.to(act), k=kn.to(act))
        kw["qk"] = qk

    def step():
        return api.gdn_verify_and_commit(
            **inp, state_pool=pool, slot_ids=slots, m=m,
            method=method, backend=backend, **kw,
        )

    with _pin_variant(triton_ops.WINDOW_TUNING, variant):
        step()
        with torch.no_grad():
            t = cuda_time(step, args.warmup, args.iters)
            t.update(
                graph_time(step, args.warmup, args.iters) if not args.no_graph else {}
            )
        vrow = _variant_row(variant, "window")
    best_us = t.get("graph_us") or t["mean_us"]

    plain = 2 * shapes.state_bytes(b, itemsize)  # one non-speculative step
    expected_mult = 1.5 if method == "two_phase" else 1.0
    return {
        "op": "verify_and_commit",
        "backend": backend,
        "method": method,
        "state_dtype": dtype_name,
        "batch": b,
        "k": k,
        "window": n,
        **t,
        **(vrow if backend == "triton" else {"variant": "n/a"}),
        "kernel_stats": (
            triton_ops.kernel_stats("window") if backend == "triton" else {}
        ),
        "timed_on": "graph" if t.get("graph_us") else "eager",
        "launch_overhead_us": (
            t["mean_us"] - t["graph_us"] if t.get("graph_us") else None
        ),
        "plain_step_state_bytes_rw": plain,
        "expected_traffic_multiplier": expected_mult,
        "achieved_gbps_at_expected_multiplier": expected_mult
        * plain
        / (best_us * 1e-6)
        / 1e9,
        "x48_layers_ms": best_us * shapes.NUM_GDN_LAYERS / 1000.0,
        "us_per_accepted_token": best_us / max(k, 1),
    }


def bench_prefill(backend: str, tokens: int, args, device: str) -> Dict[str, Any]:
    import torch

    inp = make_inputs(1, tokens, device, torch.bfloat16)
    s0 = make_pool(1, device, "fp32")
    cu = torch.tensor([0, tokens], dtype=torch.int32, device=device)

    def step():
        return api.gdn_prefill_chunked(
            **inp, cu_seqlens=cu, initial_state=s0, backend=backend
        )

    step()
    with torch.no_grad():
        t = cuda_time(step, max(1, args.warmup // 4), max(3, args.iters // 4))
    return {
        "op": "prefill_chunked",
        "backend": backend,
        "tokens": tokens,
        **t,
        "tokens_per_s_one_layer": tokens / (t["mean_us"] * 1e-6),
        "x48_layers_ms": t["mean_us"] * shapes.NUM_GDN_LAYERS / 1000.0,
    }


BV_SWEEP = (8, 16, 32, 64, 128)
WARP_SWEEP = (1, 2, 4, 8, 16)
BC_SWEEP = (128, 256, 512, 1024, 2048)
CONV_WARP_SWEEP = (1, 2, 4, 8)


def sweep_tuning(
    batches: List[int], dtype_name: str, args, device: str, variant: str = ""
) -> List[Dict[str, Any]]:
    """Re-derive the Triton decode kernel's ``(BV, num_warps)`` on this GPU.

    Swept per batch size, because the right answer is batch-dependent: BV sets
    the CTA count (``B * 48 * 128/BV``) and at B=1 there are only 48 CTAs at
    BV=128, while ``num_warps`` sets registers-per-thread (``4*BV/warps``) and
    therefore occupancy.  ``n_regs``/``n_spills`` come straight off the
    compiled kernel, so a spill shows up here rather than needing ``ncu``.
    """
    rows: List[Dict[str, Any]] = []
    if not triton_ops.is_available():
        return rows
    op = getattr(args, "sweep_op", "decode")
    cfg = {
        "decode": triton_ops.DECODE_TUNING,
        "window": triton_ops.WINDOW_TUNING,
        "conv": triton_ops.CONV_TUNING,
    }[op]
    saved = dict(cfg)
    tiles = 2 if op == "window" else 1  # the window kernel holds two state tiles
    # (BV, num_warps) and the kernel variant are separable: sweep the tile at a
    # *pinned* code path, so a change here cannot be a variant effect in
    # disguise.
    cfg["variant"] = variant
    try:
        for b in batches:
            if op == "conv":
                for bc in BC_SWEEP:
                    for nw in CONV_WARP_SWEEP:
                        cfg.update(BC=bc, num_warps=nw)
                        r = safe(
                            lambda b=b: bench_conv(b, args, device, "width_major"),
                            f"tune_conv_B{b}_BC{bc}_w{nw}",
                        )
                        r.update(BV=bc, num_warps=nw, batch=b,
                                 ctas=b * ((shapes.CONV_DIM + bc - 1) // bc))
                        rows.append(r)
                continue
            for bv in BV_SWEEP:
                for nw in WARP_SWEEP:
                    if 128 * bv // (32 * nw) < 1:  # fewer than 1 elem/thread
                        continue
                    if 32 * nw > 1024:
                        continue
                    if tiles * 128 * bv / (32 * nw) > 512:  # certain spill
                        continue
                    cfg.update(BV=bv, num_warps=nw)
                    if op == "decode":
                        r = safe(
                            lambda b=b: bench_decode("triton", b, dtype_name, args, device),
                            f"tune_B{b}_BV{bv}_w{nw}",
                        )
                    else:
                        r = safe(
                            lambda b=b: bench_spec(
                                "triton", "fused", b, dtype_name, args.spec_k, args, device
                            ),
                            f"tune_win_B{b}_BV{bv}_w{nw}",
                        )
                        r["kernel_stats"] = triton_ops.kernel_stats("window")
                        r["pct_of_hbm_peak"] = float("nan")
                    r["BV"] = bv
                    r["num_warps"] = nw
                    r["batch"] = b
                    r["variant"] = variant or "table"
                    r["tile_elems_per_thread"] = tiles * 128 * bv / (32 * nw)
                    r["ctas"] = b * shapes.NUM_V_HEADS * (
                        (shapes.HEAD_V_DIM + bv - 1) // bv
                    )
                    rows.append(r)
    finally:
        cfg.update(saved)
    return rows


# --------------------------------------------------------------------------- #
def fit_roofline(
    rows: List[Dict[str, Any]],
    peak_gbps: float,
    min_batch: int = 32,
    variant: Optional[str] = None,
) -> Dict[str, Any]:
    """Least-squares fit of ``t/B = bytes(dtype)/BW + c + e(dtype)``.

    Four unknowns — the memory path's real throughput ``BW``, a dtype-shared
    per-sequence cost ``c`` (reductions, elementwise math, address arithmetic),
    and a per-dtype extra ``e`` for fp16/bf16 (conversion cost), with
    ``e(fp32) = 0`` by convention — fitted over every (batch, dtype) row at
    ``B >= min_batch``.

    The earlier per-batch *exact* solve from a single fp32/fp16 pair was
    ill-conditioned: it returned BW = 1812% of peak at B=1 and 111% at B=512,
    which is not a measurement, it is two equations with barely any signal
    between them.  Small batches are latency-bound and do not obey this model
    at all, hence ``min_batch``.

    Why this is the scoreboard: "% of HBM peak" divides by a byte count that
    halves at fp16 while ``c`` does not, so the *same kernel* scores 81% at
    fp32 and 65-69% at fp16.  ``c`` is the number to drive down; ``e(fp16)``
    says how much of the fp16 gap is conversion cost specifically.
    """
    obs = []
    dtypes = []
    for r in rows:
        if r.get("status") != "ok" or r.get("op") != "decode_step":
            continue
        if r["backend"] != "triton" or r["batch"] < min_batch:
            continue
        # one fit per code path: mixing variants would average two different
        # kernels into one `c`, which is exactly the number under test
        if variant is not None and r.get("variant", "table") != variant:
            continue
        us = r.get("graph_us") or r.get("mean_us")
        if not us:
            continue
        per_seq = us / r["batch"]
        mb = r["ideal_state_bytes_rw"] / r["batch"] / 1e6  # MB per seq
        obs.append((mb, per_seq, r["state_dtype"], r["batch"]))
        if r["state_dtype"] not in dtypes:
            dtypes.append(r["state_dtype"])
    extra = [d for d in dtypes if d != "fp32"]
    if len(obs) < 2 + len(extra):
        return {"status": "insufficient data", "n_obs": len(obs)}
    try:
        import numpy as np

        A = np.zeros((len(obs), 2 + len(extra)))
        y = np.zeros(len(obs))
        for i, (mb, per_seq, dt, _) in enumerate(obs):
            A[i, 0] = mb
            A[i, 1] = 1.0
            if dt in extra:
                A[i, 2 + extra.index(dt)] = 1.0
            y[i] = per_seq
        x, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ x
        resid = float(np.sqrt(((pred - y) ** 2).mean()))
    except Exception as exc:  # noqa: BLE001
        return {"status": f"fit failed: {type(exc).__name__}: {exc}"}

    inv_bw, c = float(x[0]), float(x[1])
    bw = 1.0 / inv_bw * 1e3 if inv_bw > 0 else float("nan")  # MB/us -> GB/s
    out: Dict[str, Any] = {
        "status": "ok",
        "variant": variant,
        "min_batch": min_batch,
        "n_obs": len(obs),
        "fitted_bw_gbps": bw,
        "fitted_bw_pct_of_peak": 100.0 * bw / peak_gbps,
        "compute_us_per_seq_per_layer": c,
        "residual_rms_us_per_seq": resid,
        "per_dtype_extra_us_per_seq": {
            d: float(x[2 + i]) for i, d in enumerate(extra)
        },
    }
    # what each dtype scores now, and what it would score with c reduced
    proj = {}
    for dt in dtypes:
        mb = next(o[0] for o in obs if o[2] == dt)
        e = out["per_dtype_extra_us_per_seq"].get(dt, 0.0)
        now = mb * inv_bw + c + e
        proj[dt] = {
            "pct_of_hbm_peak": 100.0 * (mb / now * 1e3) / peak_gbps,
            "pct_if_c_halved": 100.0 * (mb / (mb * inv_bw + c / 2 + e) * 1e3) / peak_gbps,
            "pct_if_c_zero": 100.0 * (mb / (mb * inv_bw + e) * 1e3) / peak_gbps,
            "c_share_of_step_pct": 100.0 * c / now,
            "dtype_extra_share_pct": 100.0 * e / now,
        }
    out["projection"] = proj
    # Budget for the whole non-memory term if the step is to score 80% of HBM
    # *peak*, minus the dtype extra -> what `c` is allowed to be.
    #
    #   t_target = bytes / (0.80 * peak)      <- 80% of peak, by definition
    #   t_now    = bytes / BW_fitted + c + e
    #   => c <= bytes/(0.80*peak) - bytes/BW_fitted - e
    #
    # An earlier form was `mb*inv_bw*(peak/(0.80*peak) - 1)`, i.e. 25% of the
    # *memory* term — `peak` cancels, so it never looked at the target at all
    # and reported a budget ~3x too generous.  Corrected here: the value can
    # come out **negative**, which is the honest answer (80% unreachable at any
    # `c`, because `e` alone already exceeds the budget) and is exactly what
    # fp16 returns.
    out["c_needed_for_80pct"] = {
        dt: mb / (0.80 * peak_gbps) * 1e3
        - mb * inv_bw
        - out["per_dtype_extra_us_per_seq"].get(dt, 0.0)
        for dt in dtypes
        for mb in [next(o[0] for o in obs if o[2] == dt)]
    }
    return out


def spec_economics(
    decode_rows: List[Dict[str, Any]],
    spec_rows: List[Dict[str, Any]],
    accept_lengths: Sequence[float] = (2.0, 2.4, 2.5, 3.0),
) -> List[Dict[str, Any]]:
    """Turn the verify-and-commit cost into the number that decides whether speculation pays.

    The window/decode *ratio* is the wrong comparison: the window processes
    ``k+1`` tokens for one state read+write, so what matters is
    **microseconds per accepted token** against a plain decode step at the same
    batch.  At B=256 fp32 one measurement was 770 us (window, k=3) vs 413 us
    (decode), a 1.87x ratio that nonetheless *wins* from a mean accept length
    of about 2.1 upward, and is 1.34x at an accept length of 2.5, the figure
    reported for SGLang MTP.
    """
    base: Dict[Any, float] = {}
    base_any: Dict[Any, float] = {}
    for r in decode_rows:
        if r.get("status") != "ok" or r.get("op") != "decode_step":
            continue
        us = r.get("graph_us") or r.get("mean_us")
        if us:
            key = (r["backend"], r["batch"], r["state_dtype"])
            # a variant sweep runs several variants per (backend, batch, dtype); compare a
            # window row against the *same* code path where one exists, and
            # fall back to any decode row otherwise
            base[key + (r.get("variant", "table"),)] = us
            base_any.setdefault(key, us)
    out = []
    for r in spec_rows:
        if r.get("status") != "ok":
            continue
        us = r.get("graph_us") or r.get("mean_us")
        key = (r["backend"], r["batch"], r["state_dtype"])
        b0 = base.get(key + (r.get("variant", "table"),)) or base_any.get(key)
        if not us or not b0:
            continue
        row = {
            "backend": r["backend"], "method": r["method"], "batch": r["batch"],
            "state_dtype": r["state_dtype"], "k": r["k"],
            "variant": r.get("variant", "n/a"),
            "window_us": us, "plain_decode_us": b0,
            "window_over_decode": us / b0,
            "breakeven_accept_length": us / b0,
        }
        for a in accept_lengths:
            row[f"speedup_at_accept_{a}"] = b0 / (us / a)
        out.append(row)
    return out


def summarize_variants(decode_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per ``(state dtype, batch)``: every kernel variant against ``base``.

    ``speedup_vs_base`` is
    the honest scoreboard; ``pct_of_hbm_peak`` is the one the target is stated
    in (fp16 >= 80%), and the two disagree by construction because the %HBM
    denominator halves with the dtype while the fixed per-sequence cost ``c``
    does not — see :func:`fit_roofline`.
    """
    rows = [
        r for r in decode_rows
        if r.get("status") == "ok"
        and r.get("op") == "decode_step"
        and r.get("backend") == "triton"
    ]
    out: List[Dict[str, Any]] = []
    keys = sorted({(r["state_dtype"], r["batch"]) for r in rows}, key=lambda x: (x[0], x[1]))
    for dt, b in keys:
        cand = [r for r in rows if r["state_dtype"] == dt and r["batch"] == b]
        base = next(
            (r for r in cand if r.get("variant") in ("base", "table")), None
        )
        b_us = (base.get("graph_us") or base.get("mean_us")) if base else None
        for r in cand:
            us = r.get("graph_us") or r.get("mean_us")
            if not us:
                continue
            out.append({
                "state_dtype": dt,
                "batch": b,
                "variant": r.get("variant", "table"),
                "variant_effective": r.get("variant_effective", {}),
                "us": us,
                "pct_of_hbm_peak": r.get("pct_of_hbm_peak"),
                "x48_layers_ms": r.get("x48_layers_ms"),
                "speedup_vs_base": (b_us / us) if b_us else None,
                "kernel_stats": r.get("kernel_stats", {}),
            })
    return out


def variant_winners(summary: List[Dict[str, Any]]) -> Dict[str, Any]:
    """``{"fp16": {"64": {...}}}`` — the fastest variant per (dtype, batch).

    Paste-ready for ``triton_ops.DECODE_VARIANT_TABLE``.
    """
    best: Dict[str, Any] = {}
    for r in summary:
        d = best.setdefault(r["state_dtype"], {})
        cur = d.get(str(r["batch"]))
        if cur is None or r["us"] < cur["us"]:
            d[str(r["batch"])] = {
                "variant": r["variant"], "us": r["us"],
                "pct_of_hbm_peak": r["pct_of_hbm_peak"],
                "speedup_vs_base": r["speedup_vs_base"],
            }
    return best


def env_metadata(device: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch

        meta["torch"] = torch.__version__
        meta["torch_cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            meta["gpu"] = torch.cuda.get_device_name(device)
            props = torch.cuda.get_device_properties(device)
            meta["sm"] = f"{props.major}.{props.minor}"
            meta["gpu_mem_gib"] = round(props.total_memory / 2**30, 1)
    except Exception as exc:  # noqa: BLE001
        meta["torch_error"] = str(exc)
    try:
        import triton

        meta["triton"] = triton.__version__
    except Exception as exc:  # noqa: BLE001
        meta["triton_error"] = str(exc)
    meta["fla"] = fla_ops.version() or f"MISSING: {fla_ops.unavailable_reason()}"
    meta["backends"] = {
        kk: (vv if vv is True else str(vv)) for kk, vv in api.available_backends().items()
    }
    meta["tuning"] = {
        "decode": dict(triton_ops.DECODE_TUNING),
        "window": dict(triton_ops.WINDOW_TUNING),
        "conv": dict(triton_ops.CONV_TUNING),
    }
    return meta


def markdown(results: Dict[str, Any]) -> str:
    lines = ["# gdn kernels microbench", ""]
    e = results["env"]
    lines.append(
        f"GPU **{e.get('gpu', '?')}** · torch {e.get('torch', '?')} "
        f"(cu{e.get('torch_cuda', '?')}) · triton {e.get('triton', '?')} · "
        f"fla {e.get('fla', '?')}"
    )
    lines.append("")
    lines.append("## decode step (per GDN layer)")
    lines.append("")
    lines.append(
        "GB/s and x48 are computed from **graph us** (GPU-only) where the "
        "capture succeeded; `eager us` includes host launch cost."
    )
    lines.append("")
    lines.append(
        "| backend | variant | state | B | BV | warps | regs | spill | graph us | "
        "eager us | launch us | GB/s | % HBM | x48 ms | % step budget |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results["decode"]:
        if r.get("status") != "ok":
            continue
        ks = r.get("kernel_stats") or {}

        def _f(x, fmt="{:.1f}"):
            return fmt.format(x) if isinstance(x, (int, float)) else "-"

        lines.append(
            f"| {r['backend']} | {r.get('variant', '-')} | {r['state_dtype']} | "
            f"{r['batch']} | "
            f"{r.get('BV') or '-'} | {r.get('num_warps') or '-'} | "
            f"{ks.get('n_regs', '-')} | {ks.get('n_spills', '-')} | "
            f"{_f(r.get('graph_us'))} | {_f(r.get('mean_us'))} | "
            f"{_f(r.get('launch_overhead_us'))} | "
            f"{r['achieved_gbps']:.0f} | {r['pct_of_hbm_peak']:.0f}% | "
            f"{r['x48_layers_ms']:.2f} | {r.get('pct_of_step_budget', float('nan')):.0f}% |"
        )

    vs = results.get("variant_summary") or []
    if len({r["variant"] for r in vs}) > 1:
        lines += [
            "", "## kernel variants vs `base`", "",
            "`packed` = 16-bit state moved as int32 words (2 elements per "
            "instruction, de-interleaved even/odd register tiles); `hoist` = "
            "contiguity hints + unmasked state access + pointer math lifted "
            "out of the window kernel's T loop; `sched` = state load issued "
            "first, q/k through L1.  See `triton_kernels`.", "",
            "| state | B | variant | PACK_DT | us | vs base | % HBM | x48 ms | "
            "regs | spill |", "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in vs:
            eff = r.get("variant_effective") or {}
            ks = r.get("kernel_stats") or {}
            sp = r.get("speedup_vs_base")
            pct = r.get("pct_of_hbm_peak")
            sp_s = "{:.2f}x".format(sp) if sp else "-"
            pct_s = "{:.0f}%".format(pct) if pct is not None else "-"
            lines.append(
                f"| {r['state_dtype']} | {r['batch']} | {r['variant']} | "
                f"{eff.get('PACK_DT', '-')} | {r['us']:.1f} | {sp_s} | {pct_s} | "
                f"{r.get('x48_layers_ms', float('nan')):.2f} | "
                f"{ks.get('n_regs', '-')} | {ks.get('n_spills', '-')} |"
            )
        win = results.get("variant_winners") or {}
        if win:
            lines += ["", "### winner per (state dtype, batch)", "",
                      "| state | B | variant | us | % HBM | vs base |",
                      "|---|---|---|---|---|---|"]
            for dt, byb in win.items():
                for b, w in sorted(byb.items(), key=lambda kv: int(kv[0])):
                    p = w.get("pct_of_hbm_peak")
                    s = w.get("speedup_vs_base")
                    p_s = "{:.0f}%".format(p) if p is not None else "-"
                    s_s = "{:.2f}x".format(s) if s else "-"
                    lines.append(
                        f"| {dt} | {b} | `{w['variant']}` | {w['us']:.1f} | "
                        f"{p_s} | {s_s} |"
                    )
            lines += ["", "Bake the winners into "
                      "`triton_ops.DECODE_VARIANT_TABLE` (rows are "
                      "`(min_batch, variant)`, highest threshold first).", ""]
    for section, cols in (
        ("conv", ("backend", "layout", "batch", "BC", "num_warps", "graph_us",
                  "mean_us", "achieved_gbps", "x48_layers_ms")),
        (
            "spec",
            ("backend", "method", "variant", "state_dtype", "batch", "k",
             "graph_us", "mean_us", "launch_overhead_us", "x48_layers_ms",
             "us_per_accepted_token"),
        ),
        ("prefill", ("backend", "tokens", "mean_us", "tokens_per_s_one_layer")),
    ):
        rows = [r for r in results.get(section, []) if r.get("status") == "ok"]
        if not rows:
            continue
        lines += ["", f"## {section}", "", "| " + " | ".join(cols) + " |",
                  "|" + "---|" * len(cols)]
        for r in rows:
            vals = []
            for c in cols:
                x = r.get(c, "")
                vals.append(f"{x:.2f}" if isinstance(x, float) else str(x))
            lines.append("| " + " | ".join(vals) + " |")
    rl = results.get("roofline") or {}
    if rl.get("status") == "ok":
        lines += [
            "", "## fitted roofline  `t/B = bytes(dtype)/BW + c + e(dtype)`", "",
            f"Least squares over {rl['n_obs']} (batch, dtype) rows at "
            f"B >= {rl['min_batch']}; residual RMS "
            f"{rl['residual_rms_us_per_seq']:.4f} us/seq.", "",
            f"* **BW = {rl['fitted_bw_gbps']:.0f} GB/s "
            f"({rl['fitted_bw_pct_of_peak']:.0f}% of peak)** — the memory path.",
            f"* **c = {rl['compute_us_per_seq_per_layer']:.3f} us/seq/layer** — "
            "dtype-shared cost: reductions, elementwise math, addressing.",
        ]
        for d, e in (rl.get("per_dtype_extra_us_per_seq") or {}).items():
            lines.append(f"* `e({d})` = {e:+.3f} us/seq — conversion cost "
                         f"specific to that state dtype.")
        lines += ["", "| state | % HBM now | if c halved | if c = 0 | c is | "
                  "dtype extra is | c needed for 80% |",
                  "|---|---|---|---|---|---|---|"]
        need = rl.get("c_needed_for_80pct") or {}
        for d, pr in (rl.get("projection") or {}).items():
            lines.append(
                f"| {d} | {pr['pct_of_hbm_peak']:.0f}% | "
                f"{pr['pct_if_c_halved']:.0f}% | {pr['pct_if_c_zero']:.0f}% | "
                f"{pr['c_share_of_step_pct']:.0f}% of the step | "
                f"{pr['dtype_extra_share_pct']:.0f}% | {need.get(d, float('nan')):.3f} us |"
            )
        lines.append("")
    elif rl:
        lines += ["", f"## fitted roofline: {rl.get('status')}", ""]

    rbv = {
        k: v for k, v in (results.get("roofline_by_variant") or {}).items()
        if isinstance(v, dict) and v.get("status") == "ok"
    }
    if len(rbv) > 1:
        lines += [
            "", "## roofline per variant", "",
            "The variant question in one table: `c` is the dtype-shared "
            "per-sequence cost and `e(dtype)` the half-precision extra.  "
            "`packed` targets `e` (fewer, wider memory instructions), `hoist` "
            "and `sched` target `c`.  A variant that moves neither has not "
            "found the mechanism.", "",
            "| variant | BW GB/s | % peak | c us/seq | e(fp16) | e(bf16) | "
            "resid us |", "|---|---|---|---|---|---|---|",
        ]
        for name, v in rbv.items():
            ex = v.get("per_dtype_extra_us_per_seq") or {}
            e16 = ex.get("fp16")
            eb16 = ex.get("bf16")
            lines.append(
                f"| {name} | {v['fitted_bw_gbps']:.0f} | "
                f"{v['fitted_bw_pct_of_peak']:.0f}% | "
                f"{v['compute_us_per_seq_per_layer']:.3f} | "
                + ("{:.3f}".format(e16) if e16 is not None else "-") + " | "
                + ("{:.3f}".format(eb16) if eb16 is not None else "-") + " | "
                f"{v['residual_rms_us_per_seq']:.4f} |"
            )
        lines.append("")

    se = results.get("spec_economics") or []
    if se:
        lines += [
            "", "## speculative decoding economics", "",
            "The window/decode **ratio** is the wrong comparison — the window "
            "does `k+1` tokens for one state read+write.  What decides it is "
            "us per *accepted* token vs a plain decode step at the same batch, "
            "i.e. the speedup columns.  `breakeven` is the mean accept length "
            "at which speculation stops losing.", "",
            "| backend | method | B | state | k | window us | decode us | ratio | "
            "breakeven | 2.0 | 2.4 | 2.5 | 3.0 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in se:
            lines.append(
                f"| {r['backend']} | {r['method']} | {r['batch']} | "
                f"{r['state_dtype']} | {r['k']} | {r['window_us']:.0f} | "
                f"{r['plain_decode_us']:.0f} | {r['window_over_decode']:.2f}x | "
                f"{r['breakeven_accept_length']:.2f} | "
                + " | ".join(
                    f"{r[f'speedup_at_accept_{a}']:.2f}x" for a in (2.0, 2.4, 2.5, 3.0)
                )
                + " |"
            )
        lines.append("")
    tb = results.get("tuning_best") or {}
    if tb:
        lines += ["", "## tuning sweep — best (BV, num_warps) per batch", "",
                  "| B | BV | warps | regs | spills | CTAs | us | % HBM | "
                  "heuristic picks |", "|---|---|---|---|---|---|---|---|---|"]
        for b, r in sorted(tb.items(), key=lambda kv: int(kv[0])):
            ks = r.get("kernel_stats") or {}
            us = r.get("graph_us", r.get("mean_us"))
            lines.append(
                f"| {b} | {r['BV']} | {r['num_warps']} | {ks.get('n_regs', '-')} | "
                f"{ks.get('n_spills', '-')} | {r['ctas']} | "
                f"{us:.1f} | {r['pct_of_hbm_peak']:.0f}% | {r['heuristic_would_pick']} |"
            )
        lines.append(f"\n(swept op: `{results['args'].get('sweep_op', 'decode')}`)")
        lines += ["", "If `heuristic picks` differs from the winner, update "
                  "`triton_ops.pick_decode_tiling` (or pin via "
                  "`QWENFAST_GDN_DECODE_BV` / `_WARPS`).", ""]
    errs = [
        r for sec in ("decode", "conv", "spec", "prefill", "tuning")
        for r in results.get(sec, []) if r.get("status") not in ("ok", None)
    ]
    if errs:
        lines += ["", "## non-ok variants", ""]
        for r in errs:
            lines.append(f"* `{r.get('label', r.get('backend', '?'))}`: "
                         f"{r.get('error', r.get('reason', '?'))}")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--batches", default=",".join(str(x) for x in shapes.DECODE_BATCH_SWEEP))
    p.add_argument("--state-dtypes", default="fp32,fp16,bf16",
                   help="bf16 is in the default sweep because its fp32 "
                        "conversion is a shift rather than a full cvt — it is "
                        "the cheapest probe of the per-dtype term in the "
                        "roofline fit, and evals/ needs the number anyway")
    p.add_argument("--roofline-min-batch", type=int, default=32,
                   help="batches below this are latency-bound and do not obey "
                        "t = bytes/BW + c, so they are excluded from the fit")
    p.add_argument(
        "--variants", default="",
        help="comma-separated kernel variants to compare for the *triton* "
             "backend: any of base/packed/hoist/sched (combine with '+', e.g. "
             "'packed+hoist'), or 'all' for every combination.  Empty = one "
             "run with whatever the baked table picks.  Each variant is a "
             "separate JIT, and each gets its own roofline fit.",
    )
    p.add_argument(
        "--spec-variants", default="",
        help="same, for the fused verify-and-commit (window) kernel; empty "
             "reuses --variants",
    )
    p.add_argument("--prenorm", action="store_true",
                   help="bench the PRENORM path (q/k pre-normalised, q.k "
                        "precomputed) instead of normalising in-kernel")
    p.add_argument("--backends", default="triton,fla,torch")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--gpu-bw-gbps", type=float, default=shapes.H200_HBM_GBPS)
    p.add_argument("--ctx", type=int, default=2048, help="context for the step budget")
    p.add_argument("--spec-k", type=int, default=3, help="draft length (window = k+1)")
    p.add_argument("--prefill-tokens", default="512,2048,8192")
    p.add_argument("--torch-max-batch", type=int, default=32,
                   help="the torch reference is O(B) eager ops; skip it past this")
    p.add_argument("--sweep-tuning", action="store_true",
                   help="re-derive the Triton (BV, num_warps) table on this GPU")
    p.add_argument("--tuning-batches", default="1,8,64,256",
                   help="batch sizes to sweep tuning at (the answer is "
                        "batch-dependent: BV sets the CTA count)")
    p.add_argument("--sweep-op", default="decode", choices=("decode", "window", "conv"),
                   help="which kernel --sweep-tuning targets")
    p.add_argument("--tuning-variant", default="",
                   help="pin this kernel variant while sweeping (BV, num_warps) — "
                        "the tile and the code path are separable and must be "
                        "swept one at a time")
    p.add_argument("--no-graph", action="store_true",
                   help="skip CUDA-graph timing (eager numbers only). Graph "
                        "time is the GPU-only cost and is what the engine sees "
                        "when serving, so only do this to debug a capture failure.")
    p.add_argument("--conv-layouts", default="width_major,channel_major")
    p.add_argument(
        "--sections", default="decode,conv,spec,prefill",
        help="which benchmark families to run.  A kernel variant sweep wants "
             "`decode` alone: with --variants all, leaving `spec` on would "
             "multiply the (already long) window kernel by 8 for no new "
             "information.",
    )
    p.add_argument("--out", default="gdn_kernels_bench")
    args = p.parse_args(argv)

    batches = [int(x) for x in args.batches.split(",") if x]
    dtypes = [x for x in args.state_dtypes.split(",") if x]
    backends = [x for x in args.backends.split(",") if x]

    def _variants(spec: str) -> List[str]:
        if not spec:
            return [""]  # whatever the baked table picks
        if spec.strip().lower() in ("all", "*"):
            return list(triton_ops.all_variants())
        return [x.strip() for x in spec.split(",") if x.strip()]

    variants = _variants(args.variants)
    spec_variants = _variants(args.spec_variants or args.variants)
    sections = {x.strip() for x in args.sections.split(",") if x.strip()}

    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False")
        torch.cuda.set_device(args.device)
    except Exception as exc:  # noqa: BLE001
        payload = {"error": f"CUDA unavailable: {exc}", "env": env_metadata("cpu")}
        _write(args.out, payload, f"# gdn kernels microbench\n\nFAILED: {exc}\n")
        print(f"[bench_kernels_gdn] FAILED (no CUDA): {exc}", file=sys.stderr)
        return 1

    dev = args.device
    results: Dict[str, Any] = {
        "env": env_metadata(dev),
        "args": vars(args),
        "physics": {
            "formula": "step_ms(B, ctx) = 6.19 + B*(0.0629*itemsize/4 + 6.83e-6*ctx)",
            "hbm_gbps": args.gpu_bw_gbps,
        },
        "decode": [],
        "conv": [],
        "spec": [],
        "prefill": [],
        "tuning": [],
    }

    for b in batches:
        for dt in (dtypes if "decode" in sections else []):
            for be in backends:
                if be == "torch" and b > args.torch_max_batch:
                    continue
                if be == "triton" and not triton_ops.is_available():
                    continue
                if be == "fla" and not fla_ops.is_available():
                    continue
                for var in (variants if be == "triton" else [""]):
                    print(
                        f"[bench] decode {be} {dt} B={b} "
                        f"variant={var or 'table'}", file=sys.stderr,
                    )
                    results["decode"].append(
                        safe(
                            lambda be=be, b=b, dt=dt, var=var: bench_decode(
                                be, b, dt, args, dev, var
                            ),
                            f"decode_{be}_{dt}_B{b}_{var or 'table'}",
                        )
                    )
            if dt == dtypes[0]:
                results["decode"].append(
                    safe(
                        lambda b=b, dt=dt: bench_decode_raw_fla(b, dt, args, dev),
                        f"decode_fla_raw_B{b}",
                    )
                )
        for lay in ([x for x in args.conv_layouts.split(",") if x]
                    if "conv" in sections else []):
            print(f"[bench] conv {lay} B={b}", file=sys.stderr)
            results["conv"].append(
                safe(lambda b=b, lay=lay: bench_conv(b, args, dev, lay), f"conv_{lay}_B{b}")
            )

        for be in (backends if "spec" in sections else []):
            if be == "torch" and b > args.torch_max_batch:
                continue
            if be == "triton" and not triton_ops.is_available():
                continue
            if be == "fla" and not fla_ops.is_available():
                continue
            methods = ["two_phase"] + (["fused"] if be in ("triton", "torch") else [])
            for meth in methods:
                for var in (spec_variants if be == "triton" else [""]):
                    results["spec"].append(
                        safe(
                            lambda be=be, meth=meth, b=b, var=var: bench_spec(
                                be, meth, b, dtypes[0], args.spec_k, args, dev, var
                            ),
                            f"spec_{be}_{meth}_B{b}_{var or 'table'}",
                        )
                    )

    for tok in ([int(x) for x in args.prefill_tokens.split(",") if x]
                if "prefill" in sections else []):
        for be in ("fla", "torch"):
            if be == "fla" and not fla_ops.is_available():
                continue
            if be == "torch" and tok > 2048:
                continue
            print(f"[bench] prefill {be} T={tok}", file=sys.stderr)
            results["prefill"].append(
                safe(
                    lambda be=be, tok=tok: bench_prefill(be, tok, args, dev),
                    f"prefill_{be}_{tok}",
                )
            )

    if args.sweep_tuning:
        print("[bench] tuning sweep", file=sys.stderr)
        tb = [int(x) for x in args.tuning_batches.split(",") if x]
        results["tuning"] = sweep_tuning(
            tb, dtypes[0], args, dev, args.tuning_variant
        )
        ok = [r for r in results["tuning"] if r.get("status") == "ok"]
        best_by_b: Dict[str, Any] = {}
        for b in tb:
            cand = [r for r in ok if r["batch"] == b]
            if not cand:
                continue
            key = "graph_us" if cand[0].get("graph_us") else "mean_us"
            w = min(cand, key=lambda r: r[key])
            best_by_b[str(b)] = {
                "BV": w["BV"], "num_warps": w["num_warps"], key: w[key],
                "pct_of_hbm_peak": w.get("pct_of_hbm_peak", float("nan")),
                "tile_elems_per_thread": w.get("tile_elems_per_thread"),
                "kernel_stats": w.get("kernel_stats", {}),
                "ctas": w["ctas"],
                "heuristic_would_pick": triton_ops.pick_decode_tiling(
                    b, shapes.NUM_V_HEADS, shapes.HEAD_V_DIM,
                    triton_ops.WINDOW_TUNING if args.sweep_op == "window" else None,
                    itemsize=4 if dtypes[0] == "fp32" else 2,
                ) if args.sweep_op != "conv" else triton_ops.pick_conv_tiling(
                    b, shapes.CONV_DIM
                ),
            }
        results["tuning_best"] = best_by_b

    # One fit per code path — `c` and `e(dtype)` are properties of the kernel,
    # so averaging two variants into one fit would hide exactly the effect a
    # variant sweep is measuring.  `roofline` stays the first variant's fit so the
    # key keeps its original shape.
    results["roofline_by_variant"] = {
        (v or "table"): fit_roofline(
            results["decode"], args.gpu_bw_gbps, args.roofline_min_batch,
            variant=(v or "table"),
        )
        for v in variants
    }
    results["roofline"] = results["roofline_by_variant"][variants[0] or "table"]
    results["variant_summary"] = summarize_variants(results["decode"])
    results["variant_winners"] = variant_winners(results["variant_summary"])
    results["spec_economics"] = spec_economics(results["decode"], results["spec"])

    jp, mp = _write(args.out, results, markdown(results))
    print(f"[bench_kernels_gdn] wrote {jp} and {mp}")
    return 0


def _write(out: str, payload: Dict[str, Any], md: str):
    base = out[:-5] if out.endswith(".json") else out
    jp, mp = base + ".json", base + ".md"
    d = os.path.dirname(os.path.abspath(jp))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(jp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    with open(mp, "w") as f:
        f.write(md)
    return jp, mp


if __name__ == "__main__":
    raise SystemExit(main())
