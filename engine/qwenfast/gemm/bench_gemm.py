#!/usr/bin/env python3
"""GEMM bench for the *fused* weight layout.

Benchmarks every fused model GEMM shape x
every M bucket (``dispatch.M_BUCKETS``) x every registered backend
(``dispatch.available_backends()``), reporting µs and *effective TB/s of
weight read*, plus a whole-model per-decode-step extrapolation per
``(M, backend)`` -- the fused-layout analogue of
``kernels/microbench/gemm_bench.py`` (which benches the *unfused* per-tensor
shapes). Comparing the two is the fusion cross-check: fusion
should cut per-step GEMM count from 10 to 6 per GDN-layer-pair without
regressing effective bandwidth.

GPU-only (every timing call needs CUDA); safe to import on macOS
(``py_compile``) and the ``--help``/shape-table path runs without a GPU.

``vllm_block_fp8_triton`` measures about 2.1x *slower* than plain bf16
``torch.matmul`` at every M, which is why ``dispatch.py`` registers FP8
backends aimed specifically at beating bf16 (``flashinfer_fp8_blockscale``,
``vllm_marlin_fp8_w8a16``, ``deepgemm``, and ``scaled_mm_pertensor`` with a
column-major weight layout as cuBLASLt requires; see ``dispatch.py``'s module
docstring). This script benches **every** registered backend
(``dispatch.available_backends()``) generically, so a newly registered
backend is exercised with no changes here.

Example (GPU host)::

    PYTHONPATH=/home/engine python -m qwenfast.gemm.bench_gemm \\
        --out /home/qwenfast-results/gemm_fused.json --device cuda:0
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# model GEMM shapes (fused layout) -- kept in sync with fused_weights.py's
# fusion table and autotune.py's MODEL_GEMM_SHAPES.
# --------------------------------------------------------------------------- #
NUM_GDN_LAYERS = 48
NUM_ATTN_LAYERS = 16
NUM_ALL_LAYERS = 64

SHAPES: List[Dict[str, Any]] = [
    dict(name="gdn_in_proj_qkvz", n=16384, k=5120, count=NUM_GDN_LAYERS, kind="fp8", group="gdn",
         note="fused in_proj_qkv(10240)+in_proj_z(6144)"),
    dict(name="gdn_in_proj_ba", n=96, k=5120, count=NUM_GDN_LAYERS, kind="bf16", group="gdn",
         note="fused in_proj_b(48)+in_proj_a(48), kept bf16 (not 128-aligned)"),
    dict(name="gdn_out_proj", n=5120, k=6144, count=NUM_GDN_LAYERS, kind="fp8", group="gdn",
         note="unfused (already a single GEMM)"),
    dict(name="attn_qkv_proj", n=14336, k=5120, count=NUM_ATTN_LAYERS, kind="fp8", group="attn",
         note="fused q_proj(12288)+k_proj(1024)+v_proj(1024)"),
    dict(name="attn_o_proj", n=5120, k=6144, count=NUM_ATTN_LAYERS, kind="fp8", group="attn",
         note="unfused (already a single GEMM)"),
    dict(name="mlp_gate_up_proj", n=34816, k=5120, count=NUM_ALL_LAYERS, kind="fp8", group="mlp",
         note="fused gate_proj(17408)+up_proj(17408)"),
    dict(name="mlp_down_proj", n=5120, k=17408, count=NUM_ALL_LAYERS, kind="fp8", group="mlp",
         note="unfused (already a single GEMM)"),
    dict(name="lm_head", n=248320, k=5120, count=1, kind="fp8", group="lm_head",
         note="bf16 in the checkpoint; fp8 here is the optional quantize_lm_head path"),
]

MTP_SHAPES: List[Dict[str, Any]] = [
    dict(name="mtp_qkv_proj", n=14336, k=5120, count=1, kind="fp8", group="mtp"),
    dict(name="mtp_o_proj", n=5120, k=6144, count=1, kind="fp8", group="mtp"),
    dict(name="mtp_gate_up_proj", n=34816, k=5120, count=1, kind="fp8", group="mtp"),
    dict(name="mtp_down_proj", n=5120, k=17408, count=1, kind="fp8", group="mtp"),
    dict(name="mtp_fc", n=5120, k=10240, count=1, kind="bf16", group="mtp",
         note="mtp.fc, bf16 even in the FP8 checkpoint"),
]


# --------------------------------------------------------------------------- #
# prefill-scale (M > 512) sweep policy
# --------------------------------------------------------------------------- #
# `dispatch.M_BUCKETS` runs up to 8,192 (a real prefill chunk's M), and by
# default this script benches every registered backend at every bucket -- fine
# up to M=512, where marlin/machete/bf16_dequant and
# `vllm_block_fp8_triton` lose by 3-20x. At M in {1024, 2048, 4096, 8192} that ratio only
# gets worse (compute-bound GEMM: cost scales ~linearly in M for every
# backend, so a 3x-slower backend at M=512 stays ~3x slower at M=8192) while
# the wall-clock cost of measuring it 4x over (once per new bucket) does not
# shrink -- marlin alone measures about 356 ms/whole-model-step at M=2048.
# So by default only the five backends that are actually
# W8A8 (quantize their activations, the class that is competitive at this
# scale, as opposed to W8A16) get benched above 512:
LARGE_M_THRESHOLD = 512
LARGE_M_FP8_ACTIVATION_BACKENDS: List[str] = [
    "deepgemm",
    "flashinfer_fp8_blockscale",
    "vllm_block_fp8_cutlass",
    "vllm_cutlass_fp8_pertensor",
    "scaled_mm_pertensor",
]
# ... except at exactly one bucket above the threshold -- a REFERENCE cell,
# not a re-confirmation at every bucket -- where every backend (marlin,
# machete, both bf16 backends, `vllm_block_fp8_triton`) still runs, so the
# "still catastrophic, or not" claim above is a measurement each sweep makes
# fresh rather than an assumption baked in permanently. Smallest of the new
# buckets, so the reference costs the least of the four.
LARGE_M_REFERENCE_M = 1024


def applicable_backends_for(kind: str, m: int, backends: List[str]) -> List[str]:
    """The backends this sweep actually calls for one ``(shape kind, M)``
    cell, given the full requested ``backends`` list. Pulled out of ``main``'s
    loop so the M>512 skip policy is unit-testable without a GPU (the comments
    above this function are the full rationale)."""
    base = [b for b in backends if kind == "fp8" or b == "bf16_dequant"]
    if kind == "fp8" and m > LARGE_M_THRESHOLD and m != LARGE_M_REFERENCE_M:
        return [b for b in base if b in LARGE_M_FP8_ACTIVATION_BACKENDS]
    return base


# --------------------------------------------------------------------------- #
# small self-contained helpers (deliberately not importing kernels/microbench
# to keep this subpackage self-contained; same spirit/shape as common.py)
# --------------------------------------------------------------------------- #
def env_metadata(device: str = "cuda:0") -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
    }
    try:
        import torch

        meta["torch_version"] = torch.__version__
        meta["torch_cuda_version"] = torch.version.cuda
        meta["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            idx = torch.device(device).index or 0
            meta["gpu_name"] = torch.cuda.get_device_name(idx)
            props = torch.cuda.get_device_properties(idx)
            meta["gpu_total_mem_gb"] = round(props.total_memory / 1e9, 2)
            meta["gpu_capability"] = f"{props.major}.{props.minor}"
    except Exception as exc:  # noqa: BLE001
        meta["torch_import_error"] = f"{type(exc).__name__}: {exc}"
    for pkg in ("vllm", "flashinfer"):
        try:
            mod = __import__(pkg)
            meta[f"{pkg}_version"] = getattr(mod, "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001
            meta[f"{pkg}_import_error"] = f"{type(exc).__name__}: {exc}"
    return meta


def cuda_time_fn(fn, warmup: int, iters: int) -> Dict[str, float]:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    torch.cuda.synchronize()
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times_us = sorted(s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends))
    n = len(times_us)
    return {
        "mean_us": sum(times_us) / n,
        "min_us": times_us[0],
        "max_us": times_us[-1],
        "p50_us": times_us[n // 2],
        "n_iters": n,
    }


def cuda_graph_time_fn(fn, warmup: int, iters: int, n_capture: int = 20) -> Dict[str, Any]:
    """Time ``fn()`` by capturing ``n_capture`` back-to-back calls into ONE
    CUDA graph, replaying it ``iters`` times, and dividing.

    **Why this exists**: ``cuda_time_fn`` above times each
    call individually with CUDA events bracketing just that call -- but
    those events still capture any GPU-idle gap while the CPU is busy
    dispatching that call's kernel(s), so at small shapes (where GPU
    execution is a few µs but CPU-side launch/dispatch is ~50-70µs) it
    measures host launch cost, not GPU compute. Under CUDA-graph
    replay, an entire captured sequence is submitted as one launch and the
    GPU runs it back-to-back with no CPU round-trip between kernels (on the
    GDN kernels the per-launch floor drops from 70µs to 4µs). This is also the
    *correct* metric to optimize for regardless: the real engine's decode
    step is one CUDA graph per batch-size bucket, so a
    backend's *graph-replayed* cost is what actually determines a real
    decode step's latency, not its eager cost.

    Standard PyTorch graph-capture-benchmark pattern: warm up on a side
    stream first (lets the caching allocator settle before capture, and
    absorbs any one-time JIT/lazy-init inside ``fn`` so it isn't captured),
    then capture into a private graph memory pool, then time replay with
    CUDA events (same event-timing approach as ``cuda_time_fn``, just
    around ``g.replay()`` instead of ``fn()``).

    Returns a dict with ``graph_us`` (mean per-*single*-call time, i.e.
    replay time / ``n_capture``) plus percentiles and ``graph_n_capture``.
    Raises on any capture failure (unsupported op, host sync inside ``fn``,
    ...) -- callers must catch this and record "not graph-capturable"
    explicitly (some backends may not be capturable: DeepGEMM's runtime,
    Marlin, CUTLASS, ``torch._scaled_mm``, FlashInfer all need to be tested
    individually; vLLM itself captures DeepGEMM in production, so that one
    is expected to work)."""
    import torch

    # Warm up on a side stream (recommended practice ahead of capture: lets
    # cuBLAS/cuDNN handles, lazy kernel JITs, and the caching allocator's
    # pool settle outside the captured region).
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        for _ in range(max(warmup, 3)):
            fn()
    torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(n_capture):
            fn()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    torch.cuda.synchronize()
    for i in range(iters):
        starts[i].record()
        graph.replay()
        ends[i].record()
    torch.cuda.synchronize()

    replay_us = sorted(s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends))
    n = len(replay_us)
    per_call_us = [t / n_capture for t in replay_us]
    return {
        "graph_us": sum(per_call_us) / n,
        "graph_min_us": per_call_us[0],
        "graph_max_us": per_call_us[-1],
        "graph_p50_us": per_call_us[n // 2],
        "graph_n_capture": n_capture,
        "graph_n_iters": n,
    }


def gbps(nbytes: float, seconds: float) -> float:
    return (nbytes / 1e9) / seconds if seconds > 0 else float("nan")


def safe_run(fn, label: str = "") -> Dict[str, Any]:
    try:
        result = fn()
        result.setdefault("status", "ok")
        return result
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=6),
            "label": label,
        }


def _weight_bytes(n: int, k: int, kind: str) -> int:
    if kind == "fp8":
        # weight (1 B/elt) + block-128 scale_inv (bf16, 2 B/elt, N/128 x K/128)
        return n * k * 1 + ((n + 127) // 128) * ((k + 127) // 128) * 2
    return n * k * 2  # bf16


# --------------------------------------------------------------------------- #
# core sweep
# --------------------------------------------------------------------------- #
# Backends that quantize the activation themselves before the GEMM, mapped to
# the exact dispatch.py helper each one uses -- so its cost can be isolated
# under its own CUDA-graph capture (for M<=32 it may be a large fraction of
# the call). ``None``/absent means the backend takes bf16 activations
# directly (W8A16: no separate quant kernel to isolate).
_QUANT_ONLY_CALL = {
    "vllm_block_fp8_triton": lambda dm, x: dm._per_token_group_quant_fp8(x, 128),
    "vllm_block_fp8_cutlass": lambda dm, x: dm._per_token_group_quant_fp8(x, 128),
    "deepgemm": lambda dm, x: dm._per_token_group_quant_fp8(
        x, 128, column_major_scales=True, tma_aligned_scales=True, use_ue8m0=False),
    "scaled_mm_pertensor": lambda dm, x: dm._scaled_fp8_quant_pertensor(x),
    "vllm_cutlass_fp8_pertensor": lambda dm, x: dm._scaled_fp8_quant_pertensor(x),
    # flashinfer_fp8_blockscale, vllm_marlin_fp8_w8a16, bf16_dequant: no
    # separate activation-quant step (bf16 in, quantized/dequantized only on
    # the weight side, which is a one-time cached cost -- see
    # fused_weights.FP8Tensor's _*_cache fields).
}


def bench_shape_backend(n: int, k: int, m: int, kind: str, backend: str, warmup: int, iters: int,
                         device: str, graph_n_capture: int = 20,
                         time_quant_only: bool = True) -> Dict[str, Any]:
    """Time one (shape, M, backend) cell: eager, CUDA-graph-replayed, and
    (for M<=32) the backend's activation-quant step in isolation.

    **JIT-warm timing, separated**: some backends (notably
    ``flashinfer_fp8_blockscale`` and ``deepgemm``) JIT-compile a kernel the
    first time they see a new (N, K[, dtype]) shape, reportedly ~3 minutes
    per shape for flashinfer, which can consume a per-backend timeout with
    no visible cause. The first call is therefore timed with a plain wall-clock
    ``time.perf_counter()`` (JIT compilation is a CPU-side/host build step,
    not a GPU kernel -- CUDA events would only capture the kernel launch,
    not the compile) and reported separately as ``jit_warmup_s``, so it is
    visible in the output and clearly excluded from ``mean_us``/
    ``effective_tbps`` (which come from ``cuda_time_fn``'s *own* internal
    warmup+timed loop, i.e. steady-state, post-JIT calls only).

    **CUDA-graph timing**: every
    per-layer shape's eager time averages ~77µs vs. ~20µs of actual
    weight read at the lm_head shape's own measured 4.1 TB/s, i.e. ~55µs
    of fixed per-call launch overhead, the same pattern seen on
    the GDN kernels (eager 70µs floor, 4µs under graph replay). ``graph_us``
    (via :func:`cuda_graph_time_fn`) is the graph-replayed per-call cost;
    ``launch_overhead_us = eager_us - graph_us`` isolates exactly that fixed
    cost. If graph capture fails for a backend (unsupported op, host sync,
    ...), that is recorded explicitly (``graph_capturable=False,
    graph_error=...``) rather than silently falling back -- the *engine*
    needs every backend it uses to be graph-capturable (decode
    is one CUDA graph per batch bucket), so "can't capture" is itself an
    important, reportable result, not a bench-harness inconvenience.

    ``time_quant_only`` additionally captures-and-replays just the backend's
    activation-quantization step (see ``_QUANT_ONLY_CALL`` above) when one
    exists for this backend and ``m <= 32`` (where it may be
    a large fraction of the call) **or ``m > LARGE_M_THRESHOLD``** (a
    per-token-group quant of an [8192, 5120] bf16 -> fp8 is 84 MB read /
    42 MB write, ~40 us at HBM speed; if any backend's quant path is eager
    multi-kernel it will dominate). Reported as ``quant_graph_us`` /
    ``quant_frac_of_graph`` (= quant_graph_us / graph_us, a rough but useful
    "how much of the graphed call is quant vs. GEMM" split; not exactly
    additive with the full call's graph_us since the two are captured as
    separate graphs, but close at these shapes since there's no other work).
    """
    import time

    import torch

    from . import dispatch as dispatch_mod
    from .dispatch import _BACKENDS
    from .fused_weights import quantize_bf16_to_fp8_block128

    if backend not in _BACKENDS:
        raise KeyError(f"unknown backend {backend!r}; available: {list(_BACKENDS)}")
    if kind == "bf16" and backend != "bf16_dequant":
        raise RuntimeError(f"backend {backend!r} does not apply to a bf16 (unquantized) weight")

    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    if kind == "fp8":
        w_bf16 = torch.randn(n, k, device=device, dtype=torch.bfloat16)
        w = quantize_bf16_to_fp8_block128(w_bf16)
    else:
        w = torch.randn(n, k, device=device, dtype=torch.bfloat16)

    fn = _BACKENDS[backend]
    t0 = time.perf_counter()
    out = fn(x, w)  # smoke call: surfaces shape errors early, absorbs first-call JIT compile
    torch.cuda.synchronize()
    jit_warmup_s = time.perf_counter() - t0
    assert tuple(out.shape) == (m, n), f"backend {backend} returned shape {tuple(out.shape)}, expected {(m, n)}"
    timing = cuda_time_fn(lambda: fn(x, w), warmup, iters)

    weight_bytes = _weight_bytes(n, k, kind)
    mean_s = timing["mean_us"] * 1e-6
    # `effective_tbps` is the right yardstick at M=1 (a decode
    # GEMM is a weight read) and the wrong one at M>=1024 (a prefill/mixed GEMM
    # is compute-bound: about 98.7% of a chunk's FLOPs are GEMMs). At the
    # mixed step's M the question is "what fraction of the H200's fp8 tensor
    # cores is this shape reaching", so record the FLOP rate alongside, per cell
    # and per M -- 2*M*N*K, the standard GEMM count.
    gemm_flops = 2.0 * m * n * k
    cell: Dict[str, Any] = {"backend": backend, "n": n, "k": k, "m": m, "kind": kind, **timing,
                             "eager_us": timing["mean_us"],
                             "jit_warmup_s": jit_warmup_s,
                             "weight_bytes": weight_bytes,
                             "gemm_flops": gemm_flops,
                             "tflops": gemm_flops / mean_s / 1e12 if mean_s > 0 else float("nan"),
                             "effective_tbps": gbps(weight_bytes, mean_s) / 1000.0}

    if graph_n_capture > 0:
        try:
            graph_timing = cuda_graph_time_fn(lambda: fn(x, w), warmup, iters, n_capture=graph_n_capture)
            cell.update(graph_timing)
            cell["graph_capturable"] = True
            cell["launch_overhead_us"] = cell["eager_us"] - graph_timing["graph_us"]
            graph_s = graph_timing["graph_us"] * 1e-6
            cell["effective_tbps_graph"] = gbps(weight_bytes, graph_s) / 1000.0
            cell["tflops_graph"] = gemm_flops / graph_s / 1e12 if graph_s > 0 else float("nan")
        except Exception as exc:  # noqa: BLE001 -- expected for some backends; see docstring
            cell["graph_capturable"] = False
            cell["graph_error"] = f"{type(exc).__name__}: {exc}"

    quant_fn_factory = _QUANT_ONLY_CALL.get(backend)
    if time_quant_only and (m <= 32 or m > LARGE_M_THRESHOLD) and quant_fn_factory is not None:
        try:
            quant_timing = cuda_graph_time_fn(
                lambda: quant_fn_factory(dispatch_mod, x), warmup, iters, n_capture=graph_n_capture
            )
            cell["quant_graph_us"] = quant_timing["graph_us"]
            cell["quant_capturable"] = True
            if cell.get("graph_capturable"):
                cell["quant_frac_of_graph"] = quant_timing["graph_us"] / cell["graph_us"]
        except Exception as exc:  # noqa: BLE001
            cell["quant_capturable"] = False
            cell["quant_error"] = f"{type(exc).__name__}: {exc}"

    return cell


def _compute_totals(cells: List[Dict[str, Any]], shapes: List[Dict[str, Any]], m_buckets: List[int],
                     backends: List[str]) -> List[Dict[str, Any]]:
    """Whole-model per-decode-step extrapolation, per (M, backend) -- both
    eager (``total_ms_per_decode_step``, from ``mean_us``) and
    CUDA-graph-replayed (``total_ms_per_decode_step_graph``, from
    ``graph_us``; see ``cuda_graph_time_fn``'s docstring for why this is the
    metric that matters for the real, always-graphed engine). A shape whose cell wasn't
    graph-capturable falls back to its eager ``mean_us`` in the graph total
    too, so the graph total is always at least as complete as the eager one
    -- but is flagged via ``graph_fallback_shapes`` so it's visible which
    shapes didn't get a real graph measurement.

    Recomputed from scratch each time it is called (cheap -- cells is at
    most a few hundred rows), so it is safe to call after every single cell
    for the incremental snapshot writes below."""
    totals: List[Dict[str, Any]] = []
    for m in m_buckets:
        for backend in backends:
            total_us = 0.0
            total_bytes = 0.0
            graph_total_us = 0.0
            graph_total_bytes = 0.0
            ok_rows = 0
            missing: List[str] = []
            graph_fallback: List[str] = []
            for cell in cells:
                if cell["m"] != m or cell["backend"] != backend:
                    continue
                if cell.get("status") == "ok":
                    total_us += cell["mean_us"] * cell["count"]
                    total_bytes += cell["weight_bytes"] * cell["count"]
                    ok_rows += 1
                    if cell.get("graph_capturable"):
                        graph_total_us += cell["graph_us"] * cell["count"]
                    else:
                        graph_total_us += cell["mean_us"] * cell["count"]  # fallback to eager
                        graph_fallback.append(cell["shape_name"])
                    graph_total_bytes += cell["weight_bytes"] * cell["count"]
                else:
                    missing.append(cell["shape_name"])
            applicable = sum(1 for s in shapes if s["kind"] == "fp8" or backend == "bf16_dequant")
            if ok_rows == 0:
                continue
            totals.append({
                "m": m, "backend": backend,
                "total_ms_per_decode_step": total_us / 1000.0,
                "effective_tbps": gbps(total_bytes, total_us * 1e-6) / 1000.0,
                "total_ms_per_decode_step_graph": graph_total_us / 1000.0,
                "effective_tbps_graph": gbps(graph_total_bytes, graph_total_us * 1e-6) / 1000.0,
                "graph_fallback_shapes": graph_fallback,
                "shapes_measured": ok_rows, "shapes_applicable": applicable,
                "shapes_missing": missing,
                "complete": ok_rows == applicable,
                "graph_complete": ok_rows == applicable and not graph_fallback,
            })
    return totals


def derive_priority_from_results(results: Dict[str, Any]) -> Dict[int, List[str]]:
    """Rank backends per M-bucket by ascending
    ``total_ms_per_decode_step_graph`` (falling back to the eager total for
    any (M, backend) that has no complete row at all) -- the concrete,
    reproducible "re-derive the per-M-bucket priority from graph_us" step.
    Only ranks buckets/backends that actually have a `complete`
    total in this run's ``totals_per_step``; a bucket with no complete rows
    at all is omitted (nothing trustworthy to rank).

    This does NOT edit ``dispatch.py`` -- it produces the ranking so a human
    (or a follow-up patch) can compare it against
    ``dispatch.DEFAULT_BACKEND_PRIORITY_BY_M_BUCKET`` and update that table
    deliberately, the same way ``autotune.py``'s cache is consulted at
    runtime rather than baked into the static table blindly."""
    totals = results.get("totals_per_step", [])
    by_m: Dict[int, List[Dict[str, Any]]] = {}
    for t in totals:
        if t.get("complete"):
            by_m.setdefault(t["m"], []).append(t)
    priority: Dict[int, List[str]] = {}
    for m, rows in by_m.items():
        ranked = sorted(rows, key=lambda t: t["total_ms_per_decode_step_graph"])
        priority[m] = [r["backend"] for r in ranked]
    return priority


def derive_shape_priority_from_results(results: Dict[str, Any]) -> Dict[str, Dict[int, List[str]]]:
    """``{shape_class: {m_bucket: [backends, fastest first]}}`` from this run's
    **per-cell** ``graph_us``.

    The difference from :func:`derive_priority_from_results` above is the whole
    reason this exists. That one ranks by ``total_ms_per_decode_step_graph`` --
    a ``count``-weighted sum over all six fused shapes -- so it answers "which
    single backend, used for every shape, makes the whole step fastest". At
    prefill/mixed scale that sum is ~70% `mlp_gate_up_proj` + `mlp_down_proj`,
    so a backend that wins a smaller shape by 40% is invisible in it. This one
    ranks each ``(shape, M)`` cell on its own time, which is the decision
    ``dispatch.priority_for(m, n, k)`` actually makes.

    Keys are the **routing** keys, not the benched ones: the shape class comes
    from ``dispatch.shape_class(N, K)`` and the M key from
    ``dispatch.m_bucket(M)``, so the output is paste-ready into
    ``V9_BACKEND_PRIORITY_BY_SHAPE_AND_M``. A cell whose ``(N, K)`` has no shape
    class (a toy weight, another checkpoint) is dropped rather than guessed at.
    If two benched M values land in the same bucket, the **larger** M wins the
    key -- it is the one closer to the bucket's ceiling, and routing is by
    round-up.

    Only ``status == "ok"`` cells with a real ``graph_us`` are ranked; a cell
    that was not graph-capturable falls back to its eager ``mean_us`` (flagged
    in the returned metadata by :func:`shape_priority_report`, not here)."""
    from .dispatch import m_bucket, shape_class

    # Pass 1: (class, bucket) -> the largest benched M that lands in it.
    src_m: Dict[Tuple[str, int], int] = {}
    for cell in results.get("cells", []):
        if cell.get("status") != "ok":
            continue
        cls = shape_class(cell.get("n"), cell.get("k"))
        if cls is None:
            continue
        m = int(cell["m"])
        key = (cls, m_bucket(m))
        src_m[key] = max(src_m.get(key, 0), m)

    # Pass 2: rank the backends measured at exactly that M.
    times: Dict[Tuple[str, int], Dict[str, float]] = {}
    for cell in results.get("cells", []):
        if cell.get("status") != "ok":
            continue
        cls = shape_class(cell.get("n"), cell.get("k"))
        if cls is None:
            continue
        m = int(cell["m"])
        key = (cls, m_bucket(m))
        if src_m.get(key) != m:
            continue
        us = cell.get("graph_us") if cell.get("graph_capturable") else cell.get("mean_us")
        if us is None:
            continue
        times.setdefault(key, {})[cell["backend"]] = float(us)

    out: Dict[str, Dict[int, List[str]]] = {}
    for (cls, bucket), table in times.items():
        out.setdefault(cls, {})[bucket] = [b for b, _ in sorted(table.items(), key=lambda kv: kv[1])]
    return {cls: dict(sorted(buckets.items())) for cls, buckets in sorted(out.items())}


def shape_priority_report(results: Dict[str, Any]) -> List[str]:
    """Markdown lines: one table per shape class, rows = M, columns = backend
    graph µs and TFLOP/s, plus the per-cell winner and what it beats the
    M-only table's winner by. It is generated rather than transcribed so a
    number in any report can be traced to a cell in the JSON."""
    from .dispatch import m_bucket, shape_class

    by_shape: Dict[str, Dict[int, Dict[str, Dict[str, Any]]]] = {}
    for cell in results.get("cells", []):
        if cell.get("status") != "ok":
            continue
        cls = shape_class(cell.get("n"), cell.get("k"))
        if cls is None:
            continue
        by_shape.setdefault(cls, {}).setdefault(int(cell["m"]), {})[cell["backend"]] = cell
    lines = ["# Per-(shape, M) GEMM efficiency", "",
             "Graph-replayed µs per call and the FLOP rate it implies "
             "(2*M*N*K / graph_us). `bucket` is `dispatch.m_bucket(M)`, i.e. the "
             "key `priority_for(m, n, k)` routes on.", ""]
    for cls in sorted(by_shape):
        per_m = by_shape[cls]
        any_cell = next(iter(next(iter(per_m.values())).values()))
        lines += [f"## {cls}  [N={any_cell['n']}, K={any_cell['k']}] x{any_cell['count']}", "",
                  "| M | bucket | backend | graph_us | TFLOP/s | vs best |",
                  "|---|---|---|---|---|---|"]
        for m in sorted(per_m):
            cells = per_m[m]
            ranked = sorted(cells.items(),
                            key=lambda kv: kv[1].get("graph_us", kv[1].get("mean_us", 1e9)))
            best_us = ranked[0][1].get("graph_us", ranked[0][1].get("mean_us"))
            for backend, cell in ranked:
                us = cell.get("graph_us", cell.get("mean_us"))
                tf = cell.get("tflops_graph", cell.get("tflops"))
                ratio = us / best_us if best_us else float("nan")
                mark = " **<-**" if backend == ranked[0][0] else ""
                lines.append(f"| {m} | {m_bucket(m)} | {backend}{mark} | {us:.1f} | "
                             f"{tf:.0f} | {ratio:.3f}x |")
        lines.append("")
    return lines


def _write_snapshot(out_json: str, results: Dict[str, Any]) -> str:
    """Rewrite the full `<out>.json` + `<out>_summary.md` from the
    accumulated-so-far `results` dict. Called after every cell so a
    killed/timed-out run (for example during a long JIT compile) always
    leaves a valid, up-to-date JSON + markdown snapshot of everything
    completed so far."""
    os.makedirs(os.path.dirname(os.path.abspath(out_json)) or ".", exist_ok=True)
    tmp_path = out_json + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    os.replace(tmp_path, out_json)  # atomic on POSIX: readers never see a half-written file

    md_path = out_json[: -len(".json")] + "_summary.md"
    lines = ["# bench_gemm (fused layout) results", "",
              f"GPU: {results['env'].get('gpu_name', '?')} | torch {results['env'].get('torch_version', '?')}", "",
              f"_{len(results['cells'])} cells completed so far._", "",
              "## Whole-model per-decode-step extrapolation (complete rows only)", "",
              "eager = each call timed individually (includes CPU launch overhead); "
              "graph = CUDA-graph-captured/replayed (the metric that matches the real, "
              "always-graphed engine). `*` on graph_ms = one or more shapes fell back to "
              "eager because that backend wasn't graph-capturable there (see `graph_fallback_shapes` "
              "in the JSON).",
              "",
              "| M | backend | eager_ms | eager_TB/s | graph_ms | graph_TB/s |",
              "|---|---|---|---|---|---|"]
    for t in sorted(results.get("totals_per_step", []), key=lambda t: (t["m"], t["backend"])):
        if t["complete"]:
            star = "*" if t.get("graph_fallback_shapes") else ""
            lines.append(
                f"| {t['m']} | {t['backend']} | {t['total_ms_per_decode_step']:.4f} | "
                f"{t['effective_tbps']:.3f} | {t['total_ms_per_decode_step_graph']:.4f}{star} | "
                f"{t['effective_tbps_graph']:.3f}{star} |"
            )
    md_tmp = md_path + ".tmp"
    with open(md_tmp, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(md_tmp, md_path)
    return md_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=str, default="gemm_fused_bench_results",
                         help="Output path. Writes <out>.json (rewritten after every cell) and "
                              "<out>_summary.md, plus <out>.jsonl (one line appended per cell, "
                              "as it completes -- the most crash-safe of the three).")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--m-buckets", type=str, default=None,
                         help="Comma-separated override of the M sweep (default: dispatch.M_BUCKETS, "
                              "which runs 1..8192). Above 512, only "
                              "LARGE_M_FP8_ACTIVATION_BACKENDS are benched per shape except at "
                              "LARGE_M_REFERENCE_M (1024), which still benches every requested backend "
                              "as a reference cell -- see applicable_backends_for()'s docstring.")
    parser.add_argument("--backends", type=str, default=None,
                         help="Comma-separated override of backends to bench (default: all registered).")
    parser.add_argument("--shapes", type=str, default=None,
                         help="Comma-separated override of shape names to bench (default: all of SHAPES, "
                              "plus MTP_SHAPES if --include-mtp). Names are the first column of "
                              "--print-shapes' output, e.g. 'attn_qkv_proj,mlp_gate_up_proj'. Use this to "
                              "isolate a slow/timing-out shape (e.g. a fresh JIT-compile target) from the "
                              "rest of a sweep.")
    parser.add_argument("--include-mtp", action="store_true", help="Also bench the MTP-head shapes.")
    parser.add_argument("--graph-n-capture", type=int, default=20,
                         help="Number of back-to-back calls captured into one CUDA graph for "
                              "graph_us timing. Larger amortizes any fixed "
                              "per-replay overhead further; 20 is enough at these shapes.")
    parser.add_argument("--no-graph-timing", action="store_true",
                         help="Skip CUDA-graph capture/replay timing entirely (eager-only) "
                              "-- useful for a quick smoke pass, or if graph capture itself "
                              "is what's hanging on some backend.")
    parser.add_argument("--no-quant-only-timing", action="store_true",
                         help="Skip the separate activation-quant-only graph timing at M<=32.")
    parser.add_argument("--emit-priority", action="store_true",
                         help="After the sweep, also write <out>_priority.json and <out>_priority.py: "
                              "a per-M-bucket backend ranking derived from this run's graph_us (falling "
                              "back to eager_us for any (M,backend) that wasn't graph-capturable), ready "
                              "to compare against / paste into dispatch.py's "
                              "DEFAULT_BACKEND_PRIORITY_BY_M_BUCKET. With the default M sweep this now "
                              "covers the four prefill buckets (1024/2048/4096/8192) too, "
                              "ranked from whichever backends actually ran at each of them (see "
                              "LARGE_M_FP8_ACTIVATION_BACKENDS: most buckets only have five). Safe to "
                              "pass even on a partial --shapes/--backends run -- it just ranks whatever "
                              "backends were measured.")
    parser.add_argument("--emit-shape-priority", action="store_true",
                         help="After the sweep, write <out>_shape_priority.json, "
                              "<out>_shape_priority.py and <out>_shape_table.md: a per-(shape class, "
                              "M-bucket) ranking derived from each cell's OWN graph_us, ready to paste "
                              "into dispatch.V9_BACKEND_PRIORITY_BY_SHAPE_AND_M. Unlike --emit-priority "
                              "(which ranks the count-weighted whole-model total, so the two MLP shapes "
                              "decide every bucket) this keys on the shape the dispatcher actually has "
                              "in hand at the call site.")
    parser.add_argument("--print-shapes", action="store_true",
                         help="Print the shape table and exit (no GPU needed).")
    args = parser.parse_args(argv)

    shapes = list(SHAPES) + (list(MTP_SHAPES) if args.include_mtp else [])
    if args.shapes:
        wanted = set(args.shapes.split(","))
        shapes = [s for s in shapes if s["name"] in wanted]
        unknown = wanted - {s["name"] for s in shapes}
        if unknown:
            print(f"[bench_gemm] WARNING: unknown --shapes name(s), ignored: {sorted(unknown)}", file=sys.stderr)

    if args.print_shapes:
        for s in shapes:
            print(f"{s['name']:20s} [{s['n']:>7d}, {s['k']:>6d}] x{s['count']:<3d} {s['kind']:5s} {s.get('note', '')}")
        return 0

    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False -- this benchmark must run on a CUDA GPU.")
    except Exception as exc:  # noqa: BLE001
        payload = {"error": f"CUDA/torch unavailable: {exc}", "env": env_metadata(args.device)}
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out if args.out.endswith(".json") else args.out + ".json", "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[bench_gemm] FAILED (no CUDA)", file=sys.stderr)
        return 1

    from .dispatch import M_BUCKETS, available_backends

    device = args.device
    torch.cuda.set_device(device)
    m_buckets = [int(x) for x in args.m_buckets.split(",")] if args.m_buckets else list(M_BUCKETS)
    backends = args.backends.split(",") if args.backends else available_backends()

    out_json = args.out if args.out.endswith(".json") else args.out + ".json"
    out_jsonl = out_json[: -len(".json")] + ".jsonl"
    os.makedirs(os.path.dirname(os.path.abspath(out_json)) or ".", exist_ok=True)

    results: Dict[str, Any] = {"env": env_metadata(device), "args": vars(args), "shapes": shapes,
                                "cells": [], "totals_per_step": []}

    with open(out_jsonl, "w") as jsonl_f:
        for shape in shapes:
            name, n, k, kind = shape["name"], shape["n"], shape["k"], shape["kind"]
            for m in m_buckets:
                applicable_backends = applicable_backends_for(kind, m, backends)
                large_m_restricted = (
                    kind == "fp8" and m > LARGE_M_THRESHOLD and m != LARGE_M_REFERENCE_M
                )
                tag = " (large-M: fp8-activation backends only)" if large_m_restricted else ""
                print(f"[bench_gemm] shape={name} ({n}x{k}, {kind}) M={m}{tag} ...", file=sys.stderr)
                for backend in applicable_backends:
                    cell = safe_run(
                        lambda n=n, k=k, m=m, kind=kind, backend=backend: bench_shape_backend(
                            n, k, m, kind, backend, args.warmup, args.iters, device,
                            graph_n_capture=(0 if args.no_graph_timing else args.graph_n_capture),
                            time_quant_only=not args.no_quant_only_timing,
                        ),
                        label=f"{name}_{backend}_M{m}",
                    )
                    cell.update({"shape_name": name, "backend": backend, "m": m, "group": shape["group"],
                                 "count": shape["count"]})
                    if cell.get("status") == "ok":
                        print(f"[bench_gemm]   {backend}: {cell['mean_us']:.1f}us "
                              f"({cell['effective_tbps']:.3f} TB/s), jit_warmup={cell.get('jit_warmup_s', 0):.2f}s",
                              file=sys.stderr)
                    else:
                        print(f"[bench_gemm]   {backend}: ERROR {cell.get('error')}", file=sys.stderr)
                    results["cells"].append(cell)

                    # incremental write #1: append this cell to the JSONL
                    # sidecar immediately and flush -- survives a SIGKILL
                    # mid-sweep with zero risk of a half-written JSON file.
                    jsonl_f.write(json.dumps(cell, default=str) + "\n")
                    jsonl_f.flush()
                    os.fsync(jsonl_f.fileno())

                    # incremental write #2: recompute totals and rewrite the
                    # full <out>.json + <out>_summary.md snapshot, so both
                    # are always a complete, valid, up-to-date view of
                    # everything finished so far (not just a trailing log).
                    results["totals_per_step"] = _compute_totals(results["cells"], shapes, m_buckets, backends)
                    _write_snapshot(out_json, results)

    md_path = out_json[: -len(".json")] + "_summary.md"
    print(f"[bench_gemm] wrote {out_json}, {md_path}, and {out_jsonl}")

    if args.emit_shape_priority:
        shape_priority = derive_shape_priority_from_results(results)
        sp_json = out_json[: -len(".json")] + "_shape_priority.json"
        with open(sp_json, "w") as f:
            json.dump({cls: {str(m): order for m, order in buckets.items()}
                       for cls, buckets in shape_priority.items()}, f, indent=2)
        sp_py = out_json[: -len(".json")] + "_shape_priority.py"
        py_lines = [
            "# Derived from " + out_json + " by bench_gemm.py --emit-shape-priority.",
            "# Ranked on each (shape, M) cell's own graph_us. Keys are dispatch.shape_class(N, K)",
            "# and dispatch.m_bucket(M) -- paste into dispatch.V9_BACKEND_PRIORITY_BY_SHAPE_AND_M.",
            "# A (class, bucket) absent here was not measured in this run and MUST NOT be assumed;",
            "# dispatch falls through to the M-only table for it, which is the honest default.",
            "V9_BACKEND_PRIORITY_BY_SHAPE_AND_M = {",
        ]
        for cls, buckets in shape_priority.items():
            py_lines.append(f"    {cls!r}: {{")
            for m, order in buckets.items():
                py_lines.append(f"        {m}: {order!r},")
            py_lines.append("    },")
        py_lines.append("}")
        with open(sp_py, "w") as f:
            f.write("\n".join(py_lines) + "\n")
        sp_md = out_json[: -len(".json")] + "_shape_table.md"
        with open(sp_md, "w") as f:
            f.write("\n".join(shape_priority_report(results)) + "\n")
        print(f"[bench_gemm] wrote {sp_json}, {sp_py} and {sp_md} "
              f"({sum(len(b) for b in shape_priority.values())} (shape, bucket) cells ranked)")

    if args.emit_priority:
        priority = derive_priority_from_results(results)
        priority_json_path = out_json[: -len(".json")] + "_priority.json"
        with open(priority_json_path, "w") as f:
            json.dump({str(m): order for m, order in sorted(priority.items())}, f, indent=2)
        priority_py_path = out_json[: -len(".json")] + "_priority.py"
        py_lines = [
            "# Derived from " + out_json + " by bench_gemm.py --emit-priority (graph_us-ranked).",
            "# Compare against dispatch.DEFAULT_BACKEND_PRIORITY_BY_M_BUCKET before pasting in --",
            "# a bucket missing here had no `complete` row in this run (rerun with --shapes/--backends",
            "# to fill it in) and should NOT be assumed unchanged.",
            "DERIVED_BACKEND_PRIORITY_BY_M_BUCKET = {",
        ]
        for m, order in sorted(priority.items()):
            py_lines.append(f"    {m}: {order!r},")
        py_lines.append("}")
        with open(priority_py_path, "w") as f:
            f.write("\n".join(py_lines) + "\n")
        print(f"[bench_gemm] wrote {priority_json_path} and {priority_py_path} "
              f"({len(priority)}/{len(m_buckets)} M-buckets ranked)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
