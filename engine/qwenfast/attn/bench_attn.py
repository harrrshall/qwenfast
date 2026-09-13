#!/usr/bin/env python
"""GPU benchmark for the paged-KV attention stack in this subpackage
(``kv_pool.PagedKVPool`` + ``flashinfer_attn.FlashInferDecodeAttention`` /
``FlashInferPrefillAttention``).

Two sweeps, JSON out:

* **decode**: µs/step for batch ``B`` in the decode-bucket sweep x context
  ``{2048, 8192}`` x KV dtype ``{bf16, fp8}`` x every backend in
  ``--backends`` (default: ``flashinfer_decode_tc``, ``flashinfer_prefill_as_decode``,
  ``fa3_kvcache`` -- see ``flashinfer_attn.DECODE_BACKENDS``), plus the
  x16-full-attention-layer extrapolated ms/step. **Multi-backend by design**:
  with flashinfer 0.6.16.post3 on H200, ``BatchDecodeWithPagedKVCacheWrapper``
  fails outright at our GQA shape (24 q-heads / 4 kv-heads, head_dim 256 ->
  group_size 6) unless ``use_tensor_cores=True`` -- see ``flashinfer_attn``'s
  module docstring. Every cell is wrapped in a timed, exception-safe runner:
  an unavailable/broken backend records ``status: "error"``, a cell that runs
  past ``--cell-timeout-s`` records ``status: "timeout"`` -- the sweep never
  aborts and never hangs past its per-cell budget.
* **prefill**: tok/s for a single varlen causal chunk of length
  ``T in {512, 2048, 8192}``, attention kernel time only (KV-cache append is
  timed separately and reported, not folded into the tok/s number).

**Bounded by design** (an unbounded full sweep can spend most of its time in
host-loop-bound setup rather than in kernels; see "Setup cost" below):

* ``--iters 20 --warmup 5`` are the defaults. For a
  microsecond-scale decode/prefill kernel, 20 timed iterations is already a
  tight distribution; this alone cuts total kernel-launch count ~6x.
* ``--quick`` sets ``--batches 1,32,128,512 --contexts 2048
  --kv-dtypes bf16,fp8 --prefill-tokens 2048`` -- a representative sample
  across the batch range at one context, for a fast sanity pass (a handful
  of seconds/cell x ~4 batches x 2 dtypes x 3 backends, not the full
  7-batch x 2-context grid).
* ``--cell-timeout-s`` (default 60s) bounds every individual cell
  (pool build *and* backend timing) with ``signal.alarm`` -- a cell that
  blows the budget is recorded with ``status: "timeout"`` and the sweep
  moves on, instead of hanging indefinitely. Unix only (the GPU host is
  Linux); a no-op with a one-time warning on platforms without
  ``signal.SIGALRM``.
* Each cell's result prints to stderr **as it completes**
  (``... -> 123.4us (ok)`` / ``-> TIMEOUT`` / ``-> ERROR: ...``), so a
  killed/interrupted run still shows exactly how far it got.
* A dedicated **JIT warm-up phase** runs first: one tiny (B=1,
  ctx=page_size) call per (backend, kv_dtype) to force any lazy CUDA kernel
  compilation, timed separately and reported as ``jit_warmup[*].jit_s`` --
  so a cold JIT compile (which can take tens of seconds) is paid once,
  up front, visibly, rather than silently inflating -- or being hidden
  inside -- the first real cell's numbers.
* The torch fallback (``torch_fallback_decode``, an O(B) Python loop -- a
  correctness reference, never a production candidate) is **not** in the
  default ``--backends`` list; add ``torch`` explicitly if you want it.

**Setup cost:** filling the pool with a Python loop over each of the ``B``
sequences (and, for fp8, a nested loop over each sequence's pages to
calibrate its scale), or writing page-table entries one scalar CUDA write at
a time, costs up to 512 x 512 = 262,144 tiny CUDA calls **per cell** at
``B=512, ctx=8192, page_size=16``, each paying full Python-loop + CUDA-API
dispatch overhead (~10-50us): tens of seconds per large cell, across dozens
of cells. Both ``PagedKVPool.ensure_capacity`` and this script's pool setup
are therefore single vectorized tensor ops, and the pool is built once per
``(B, ctx, kv_dtype)`` and reused across backends, so a full default sweep
takes a small number of minutes.

This script does nothing when imported and only runs under
``python bench_attn.py ...`` on a CUDA host with ``flashinfer`` importable --
it is ``python -m py_compile``-safe with no torch installed (every torch/CUDA
use lives inside function bodies), matching the convention in
``kernels/microbench/*.py``.

Example (remote, see README.md) -- either invocation form works (the
sys.path bootstrap right below the imports handles the "run as a bare
script" case, which otherwise fails with "ImportError: attempted relative
import with no known parent package")::

    PYTHONPATH=/home/engine /home/venv_vllm/bin/python -m qwenfast.attn.bench_attn \\
        --quick --out /home/qwenfast-results/attn_bench_quick.json

    PYTHONPATH=/home/engine /home/venv_vllm/bin/python \\
        /home/engine/qwenfast/attn/bench_attn.py \\
        --out /home/qwenfast-results/attn_bench.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import sys
import time
import traceback
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple

# Bootstrap so this file works both as part of the package (python -m
# qwenfast.attn.bench_attn) *and* invoked directly as a script
# (python /home/engine/qwenfast/attn/bench_attn.py) -- the latter sets
# __package__ to "" / None, which breaks relative imports (`from .kv_pool
# import ...`) with "ImportError: attempted relative import with no known
# parent package". Adding `engine/` to
# sys.path and importing absolutely (`from qwenfast.attn... import ...`)
# works in both invocation styles, matching the sys.path bootstrap
# `tests/test_attn.py` already uses for the same reason. Harmless no-op when
# `engine/` is already on sys.path (e.g. via PYTHONPATH=/home/engine).
_HERE = os.path.dirname(os.path.abspath(__file__))
_ENGINE_DIR = os.path.abspath(os.path.join(_HERE, "..", ".."))  # attn/ -> qwenfast/ -> engine/
if _ENGINE_DIR not in sys.path:
    sys.path.insert(0, _ENGINE_DIR)

DEFAULT_DECODE_BATCHES = [1, 8, 32, 64, 128, 256, 512]
DEFAULT_CONTEXTS = [2048, 8192]
DEFAULT_KV_DTYPES = ["bf16", "fp8"]
DEFAULT_PREFILL_TOKENS = [512, 2048, 8192]
# Kept as plain strings (not imported from flashinfer_attn at module scope) so
# this module's torch-free import path never changes; must match
# flashinfer_attn.DECODE_BACKENDS. "torch" (the always-correct
# O(B)-python-loop reference) is opt-in via --backends -- it isn't a
# production candidate, just useful for a sanity cross-check against the GPU
# backends' numbers.
DEFAULT_DECODE_BACKENDS = ["flashinfer_decode_tc", "flashinfer_prefill_as_decode", "fa3_kvcache"]

# --quick preset (see module docstring's "Bounded by design").
QUICK_BATCHES = [1, 32, 128, 512]
QUICK_CONTEXTS = [2048]
QUICK_KV_DTYPES = ["bf16", "fp8"]
QUICK_PREFILL_TOKENS = [2048]

DEFAULT_ITERS = 20
DEFAULT_WARMUP = 5
DEFAULT_CELL_TIMEOUT_S = 60.0

NUM_ATTN_LAYERS = 16  # full-attention layers in the model (MTP is a 17th, benched separately if wanted)

NUM_QO_HEADS = 24
NUM_KV_HEADS = 4
HEAD_DIM = 256
PAGE_SIZE_DEFAULT = 16


# --------------------------------------------------------------------------- #
# small self-contained result-writer / timing / timeout helpers (kept local
# rather than importing kernels/microbench/common.py, so this package has no
# dependency outside engine/qwenfast/)
# --------------------------------------------------------------------------- #
class CellTimeout(Exception):
    """Raised when a single benchmark cell exceeds --cell-timeout-s."""


_TIMEOUT_UNSUPPORTED_WARNED = False


@contextmanager
def cell_timeout(seconds: Optional[float]):
    """Best-effort per-cell wall-clock budget via ``signal.alarm`` (Unix only).

    Not a hard real-time guarantee against a single opaque blocking C call
    with no interleaved Python bytecode, but it reliably bounds the failure
    mode that matters here (a host-side Python loop of many
    small CUDA calls, or a JIT compile that shells out to nvcc/ninja via
    ``subprocess`` -- both are interruptible). ``seconds <= 0`` or ``None``
    disables the timeout entirely.
    """
    global _TIMEOUT_UNSUPPORTED_WARNED
    import signal

    if not seconds or seconds <= 0 or not hasattr(signal, "SIGALRM"):
        if seconds and not hasattr(signal, "SIGALRM") and not _TIMEOUT_UNSUPPORTED_WARNED:
            print("[bench_attn] WARNING: signal.SIGALRM unavailable on this platform; "
                  "--cell-timeout-s has no effect.", file=sys.stderr)
            _TIMEOUT_UNSUPPORTED_WARNED = True
        yield
        return

    def _handler(signum, frame):
        raise CellTimeout(f"cell exceeded {seconds:.0f}s wall-clock budget")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    old_alarm = signal.alarm(max(1, int(seconds)))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_alarm:  # pragma: no cover -- restore a still-pending outer alarm, if any
            signal.alarm(old_alarm)


def safe_run(fn: Callable[[], Any], label: str = "", timeout_s: Optional[float] = None) -> Dict[str, Any]:
    """Run ``fn()`` under ``cell_timeout``, always returning a JSON-safe dict.

    Success: ``fn()``'s return dict (or ``{"value": ...}`` if it returned
    something else), with ``status: "ok"`` merged in. Timeout:
    ``status: "timeout"``. Any other exception: ``status: "error"``. Never
    raises -- this is what lets the sweep keep going past a broken or
    hanging backend/cell.
    """
    try:
        with cell_timeout(timeout_s):
            result = fn()
        if not isinstance(result, dict):
            result = {"value": result}
        result.setdefault("status", "ok")
        return result
    except CellTimeout as exc:
        return {"status": "timeout", "error": str(exc), "label": label}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=8), "label": label}


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
    except Exception as exc:  # pragma: no cover
        meta["torch_import_error"] = f"{type(exc).__name__}: {exc}"
    # flashinfer for fix #1/#2 (tensor-core decode / prefill-as-decode);
    # vllm_flash_attn / flash_attn_interface for fix #3 (FA3 paged decode) --
    # see flashinfer_attn.py's module docstring for what each fix is.
    for pkg in ("flashinfer", "vllm_flash_attn", "flash_attn_interface"):
        try:
            mod = __import__(pkg)
            meta[f"{pkg}_version"] = getattr(mod, "__version__", "unknown")
        except Exception as exc:
            meta[f"{pkg}_import_error"] = f"{type(exc).__name__}: {exc}"
    return meta


def cuda_time_fn(fn: Callable[[], Any], warmup: int, iters: int) -> Dict[str, float]:
    """Time ``fn()`` with CUDA events. Callers are responsible for anything
    that must happen once (pool build, ``plan()``) happening *outside*
    ``fn`` -- ``fn`` should do exactly the one thing being measured
    (typically just ``wrapper.run(...)``), never re-plan per iteration."""
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
    times_us = sorted(s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends))
    n = len(times_us)
    return {
        "mean_us": sum(times_us) / n,
        "min_us": times_us[0],
        "max_us": times_us[-1],
        "p50_us": times_us[n // 2],
        "n_iters": n,
    }


def write_results(out: str, payload: Dict[str, Any]) -> str:
    if not out.endswith(".json"):
        out = out + ".json"
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return out


def dtype_nbytes(name: str) -> int:
    return {"bf16": 2, "fp8": 1}[name]


def _print_cell_result(prefix: str, cell: Dict[str, Any], us_key: str = "mean_us") -> None:
    """Print each cell's result *as it completes* (module docstring item),
    so a killed/interrupted run's stderr shows exactly how far it got."""
    status = cell.get("status")
    if status == "ok":
        print(f"{prefix} -> {cell.get(us_key, float('nan')):.1f}us (ok)", file=sys.stderr)
    elif status == "timeout":
        print(f"{prefix} -> TIMEOUT ({cell.get('error')})", file=sys.stderr)
    else:
        print(f"{prefix} -> ERROR: {cell.get('error')}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# decode sweep -- one filled pool per (B, ctx, kv_dtype), timed against every
# requested backend (pool build is NOT repeated per backend)
# --------------------------------------------------------------------------- #
def build_decode_pool(B: int, context: int, kv_dtype: str, page_size: int, device: str) -> Tuple[Any, List[int]]:
    """Allocate a pool, fill ``B`` sequences with ``context`` random tokens
    each -- fully vectorized (no per-sequence or per-page Python loop): one
    batched ``[B, context, H, D]`` K/V tensor, one flattened ``append_kv``
    call, and (for fp8) one broadcast scale assignment covering every page
    at once. See the module docstring's "Setup cost" for why this matters:
    a per-sequence/per-page loop form dominates the cost of a full sweep.

    The fp8 scale here is deliberately a single global-ish per-head value
    (computed once from the whole batch's amax, applied to every page) --
    good enough for a *performance* benchmark's numbers to be representative
    and non-overflowing; ``kv_pool.py``'s own tests exercise the finer
    per-page ``calibrate_page_scale`` this trades away for speed.
    """
    import torch

    from qwenfast.attn.kv_pool import FP8_MAX, KVPoolConfig, PagedKVPool

    pages_per_seq = (context + page_size - 1) // page_size
    cfg = KVPoolConfig(
        n_layers=1, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM, page_size=page_size,
        n_pages=B * pages_per_seq + 1, max_seqs=B, max_pages_per_seq=pages_per_seq + 1,
        dtype=kv_dtype, device=device,
    )
    pool = PagedKVPool(cfg)
    slots = [pool.alloc_slot() for _ in range(B)]
    for s in slots:
        pool.ensure_capacity(s, context)  # vectorized (kv_pool.py fix); B cheap host-side calls

    k_all = (torch.randn(B, context, NUM_KV_HEADS, HEAD_DIM, device=device) * 0.1).to(torch.bfloat16)
    v_all = (torch.randn(B, context, NUM_KV_HEADS, HEAD_DIM, device=device) * 0.1).to(torch.bfloat16)

    if kv_dtype == "fp8":
        k_amax = k_all.float().abs().amax(dim=(0, 1, 3)).clamp(min=1e-6)  # -> [num_kv_heads]
        v_amax = v_all.float().abs().amax(dim=(0, 1, 3)).clamp(min=1e-6)
        pool.scale[0, :, 0, :] = (k_amax / FP8_MAX)[None, :]  # broadcast over every page, one op
        pool.scale[0, :, 1, :] = (v_amax / FP8_MAX)[None, :]

    slots_t = torch.tensor(slots, dtype=torch.int64, device=device)
    slot_ids_flat = torch.repeat_interleave(slots_t, context)  # [B*context], B-major
    positions_flat = torch.arange(context, dtype=torch.int64, device=device).repeat(B)  # matches B-major order
    pool.append_kv(
        0, slot_ids_flat, positions_flat,
        k_all.reshape(B * context, NUM_KV_HEADS, HEAD_DIM),
        v_all.reshape(B * context, NUM_KV_HEADS, HEAD_DIM),
    )
    return pool, slots


def run_decode_backend(pool: Any, slots: List[int], backend: str, warmup: int, iters: int, device: str) -> Dict[str, Any]:
    """Time one backend's decode step against an already-built, already-filled
    ``pool``. ``plan()`` (for the two FlashInfer-based backends) runs exactly
    once here, *before* ``cuda_time_fn``'s warmup/timed loops -- never inside
    them, never per iteration."""
    import torch

    from qwenfast.attn import flashinfer_attn as fa

    B = len(slots)
    slot_ids = torch.tensor(slots, dtype=torch.int64, device=device)
    q = torch.randn(B, NUM_QO_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    scaling = HEAD_DIM ** -0.5
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    if backend == "flashinfer_decode_tc":
        wrapper = fa.FlashInferDecodeAttention(
            workspace, NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM, pool.cfg.page_size,
            max_batch_size=B, max_pages=pool.cfg.n_pages, kv_dtype=pool.storage_dtype,
            use_cuda_graph=False, use_tensor_cores=True, device=device,
        )
        kv_indptr, kv_indices, kv_last_page_len, _ = pool.build_flashinfer_indices(slots)
        wrapper.plan(kv_indptr, kv_indices, kv_last_page_len)  # once, outside the timed loop
        step = lambda: wrapper.run(q, pool.kv[0])
    elif backend == "flashinfer_prefill_as_decode":
        wrapper = fa.FlashInferPrefillAttention(workspace, NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM, pool.cfg.page_size,
                                                 kv_dtype=pool.storage_dtype)
        qo_indptr = fa.decode_qo_indptr(B, device=device)
        kv_indptr, kv_indices, kv_last_page_len, _ = pool.build_flashinfer_indices(slots)
        wrapper.plan(qo_indptr, kv_indptr, kv_indices, kv_last_page_len, causal=True)  # once
        step = lambda: wrapper.run(q, pool.kv[0])
    elif backend == "fa3_kvcache":
        step = lambda: fa.fa3_decode_with_kvcache(pool, 0, slot_ids, q, scaling)
    elif backend == "torch":
        step = lambda: fa.torch_fallback_decode(pool, 0, slot_ids, q, scaling)
    else:
        raise ValueError(f"unknown decode backend {backend!r}; choose from {fa.DECODE_BACKENDS}")

    with torch.no_grad():
        timing = cuda_time_fn(step, warmup, iters)

    context = int(pool.seq_len[slots[0]]) if slots else 0
    kv_dtype_bytes = 1 if pool.cfg.dtype == "fp8" else 2
    kv_read_bytes = B * context * 2 * NUM_KV_HEADS * HEAD_DIM * kv_dtype_bytes
    mean_s = timing["mean_us"] * 1e-6
    return {
        **timing,
        "kv_read_bytes": kv_read_bytes,
        "achieved_gbps": (kv_read_bytes / 1e9) / mean_s if mean_s > 0 else float("nan"),
        "extrapolated_ms_16_layers": timing["mean_us"] * NUM_ATTN_LAYERS / 1000.0,
    }


# --------------------------------------------------------------------------- #
# prefill sweep
# --------------------------------------------------------------------------- #
def bench_prefill(T: int, kv_dtype: str, page_size: int, warmup: int, iters: int, device: str) -> Dict[str, Any]:
    import torch

    from qwenfast.attn.flashinfer_attn import FlashInferPrefillAttention
    from qwenfast.attn.kv_pool import FP8_MAX, KVPoolConfig, PagedKVPool

    pages_needed = (T + page_size - 1) // page_size
    cfg = KVPoolConfig(
        n_layers=1, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM, page_size=page_size,
        n_pages=pages_needed + 1, max_seqs=1, max_pages_per_seq=pages_needed + 1,
        dtype=kv_dtype, device=device,
    )
    pool = PagedKVPool(cfg)
    slot = pool.alloc_slot()
    pool.ensure_capacity(slot, T)

    q = (torch.randn(T, NUM_QO_HEADS, HEAD_DIM, device=device) * 0.1).to(torch.bfloat16)
    k = (torch.randn(T, NUM_KV_HEADS, HEAD_DIM, device=device) * 0.1).to(torch.bfloat16)
    v = (torch.randn(T, NUM_KV_HEADS, HEAD_DIM, device=device) * 0.1).to(torch.bfloat16)
    if kv_dtype == "fp8":
        # one global-ish per-head scale, one broadcast assignment -- see
        # build_decode_pool's docstring for the same tradeoff.
        k_amax = k.float().abs().amax(dim=(0, 2)).clamp(min=1e-6)
        v_amax = v.float().abs().amax(dim=(0, 2)).clamp(min=1e-6)
        pool.scale[0, :, 0, :] = (k_amax / FP8_MAX)[None, :]
        pool.scale[0, :, 1, :] = (v_amax / FP8_MAX)[None, :]

    def do_append():
        pool.append_kv(0, torch.full((T,), slot, dtype=torch.int64, device=device),
                        torch.arange(T, dtype=torch.int64, device=device), k, v)

    with torch.no_grad():
        append_timing = cuda_time_fn(do_append, warmup, iters)

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = FlashInferPrefillAttention(workspace, NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM, page_size,
                                          kv_dtype=pool.storage_dtype)
    qo_indptr = torch.tensor([0, T], dtype=torch.int32, device=device)
    kv_indptr, kv_indices, kv_last_page_len, _ = pool.build_flashinfer_indices([slot])
    wrapper.plan(qo_indptr, kv_indptr, kv_indices, kv_last_page_len, causal=True)  # once, outside the timed loop

    with torch.no_grad():
        attn_timing = cuda_time_fn(lambda: wrapper.run(q, pool.kv[0]), warmup, iters)

    mean_s = attn_timing["mean_us"] * 1e-6
    return {
        "attn_mean_us": attn_timing["mean_us"],
        "kv_append_mean_us": append_timing["mean_us"],
        "tok_per_s": T / mean_s if mean_s > 0 else float("nan"),
    }


# --------------------------------------------------------------------------- #
# JIT warm-up: one tiny call per (backend, kv_dtype), timed separately
# --------------------------------------------------------------------------- #
def jit_warmup(backends: List[str], kv_dtypes: List[str], page_size: int, device: str,
                timeout_s: Optional[float]) -> List[Dict[str, Any]]:
    """Force any lazy CUDA kernel JIT compilation for each (backend, kv_dtype)
    shape config with one B=1, ctx=page_size call, *before* the real sweep --
    so a cold compile (which can take tens of seconds) is paid once, up
    front, and reported as its own ``jit_s``, rather than silently inflating
    (or hiding inside) the first real cell that happens to hit it."""
    rows: List[Dict[str, Any]] = []
    for backend in backends:
        if backend == "torch":
            continue  # pure torch SDPA -- no kernel JIT to warm
        for kv_dtype in kv_dtypes:
            print(f"[bench_attn] JIT warm-up backend={backend} kv_dtype={kv_dtype} ...", file=sys.stderr)
            t0 = time.time()

            def _do():
                pool, slots = build_decode_pool(1, page_size, kv_dtype, page_size, device)
                return run_decode_backend(pool, slots, backend, warmup=1, iters=1, device=device)

            cell = safe_run(_do, label=f"jit_{backend}_{kv_dtype}", timeout_s=timeout_s)
            jit_s = time.time() - t0
            cell.update({"backend": backend, "kv_dtype": kv_dtype, "jit_s": jit_s})
            _print_cell_result(f"[bench_attn]   jit_s={jit_s:.2f}", cell, us_key="mean_us")
            rows.append(cell)
    return rows


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=str, default="attn_bench_results.json")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--page-size", type=int, default=PAGE_SIZE_DEFAULT)
    parser.add_argument("--cell-timeout-s", type=float, default=DEFAULT_CELL_TIMEOUT_S,
                         help="Per-cell wall-clock budget (pool build + backend timing). "
                              "<= 0 disables the timeout. Default %(default)s.")
    parser.add_argument("--quick", action="store_true",
                         help=f"Fast preset: batches={QUICK_BATCHES}, contexts={QUICK_CONTEXTS}, "
                              f"kv_dtypes={QUICK_KV_DTYPES}, prefill_tokens={QUICK_PREFILL_TOKENS} "
                              "(overridden by any of --batches/--contexts/--kv-dtypes/--prefill-tokens "
                              "you also pass explicitly).")
    parser.add_argument("--batches", type=str, default=None,
                         help=f"Comma-separated decode batch sweep (default {DEFAULT_DECODE_BATCHES}, "
                              f"or {QUICK_BATCHES} with --quick).")
    parser.add_argument("--contexts", type=str, default=None,
                         help=f"Comma-separated context sweep (default {DEFAULT_CONTEXTS}, "
                              f"or {QUICK_CONTEXTS} with --quick).")
    parser.add_argument("--kv-dtypes", type=str, default=None,
                         help=f"Comma-separated KV dtypes (default {DEFAULT_KV_DTYPES}).")
    parser.add_argument("--prefill-tokens", type=str, default=None,
                         help=f"Comma-separated prefill T sweep (default {DEFAULT_PREFILL_TOKENS}, "
                              f"or {QUICK_PREFILL_TOKENS} with --quick).")
    parser.add_argument("--backends", type=str, default=",".join(DEFAULT_DECODE_BACKENDS),
                         help="Comma-separated decode backends to benchmark, from "
                              "{flashinfer_decode_tc, flashinfer_prefill_as_decode, fa3_kvcache, torch} "
                              f"(default {DEFAULT_DECODE_BACKENDS} -- add 'torch' explicitly for the "
                              "slow-but-always-correct reference; it is never included by default).")
    parser.add_argument("--skip-decode", action="store_true")
    parser.add_argument("--skip-prefill", action="store_true")
    parser.add_argument("--skip-jit-warmup", action="store_true")
    args = parser.parse_args(argv)

    batches = ([int(x) for x in args.batches.split(",")] if args.batches
               else (QUICK_BATCHES if args.quick else DEFAULT_DECODE_BATCHES))
    contexts = ([int(x) for x in args.contexts.split(",")] if args.contexts
                else (QUICK_CONTEXTS if args.quick else DEFAULT_CONTEXTS))
    kv_dtypes = ([x.strip() for x in args.kv_dtypes.split(",") if x.strip()] if args.kv_dtypes
                 else (QUICK_KV_DTYPES if args.quick else DEFAULT_KV_DTYPES))
    prefill_tokens = ([int(x) for x in args.prefill_tokens.split(",")] if args.prefill_tokens
                      else (QUICK_PREFILL_TOKENS if args.quick else DEFAULT_PREFILL_TOKENS))
    backends = [x.strip() for x in args.backends.split(",") if x.strip()]
    timeout_s = args.cell_timeout_s if args.cell_timeout_s and args.cell_timeout_s > 0 else None

    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False -- this benchmark must run on a CUDA GPU.")
    except Exception as exc:  # noqa: BLE001
        payload = {"error": f"CUDA/torch unavailable: {exc}", "env": env_metadata(args.device)}
        json_path = write_results(args.out, payload)
        print(f"[bench_attn] FAILED (no CUDA): wrote {json_path}", file=sys.stderr)
        return 1

    device = args.device
    torch.cuda.set_device(device)

    try:
        import flashinfer  # noqa: F401
    except Exception as exc:
        print(f"[bench_attn] WARNING: flashinfer not importable ({exc}); "
              "flashinfer_* backends will error per-cell.", file=sys.stderr)

    results: Dict[str, Any] = {
        "env": env_metadata(device), "args": vars(args),
        "jit_warmup": [], "decode_cells": [], "prefill_cells": [],
    }

    if not args.skip_decode and not args.skip_jit_warmup:
        results["jit_warmup"] = jit_warmup(backends, kv_dtypes, args.page_size, device, timeout_s)

    if not args.skip_decode:
        for context in contexts:
            for B in batches:
                for kv_dtype in kv_dtypes:
                    prefix = f"[bench_attn] decode ctx={context} B={B} kv_dtype={kv_dtype}"
                    print(f"{prefix}: building pool ...", file=sys.stderr)
                    pool_cell = safe_run(
                        lambda B=B, context=context, kv_dtype=kv_dtype: build_decode_pool(
                            B, context, kv_dtype, args.page_size, device),
                        label=f"pool_ctx{context}_B{B}_{kv_dtype}", timeout_s=timeout_s)
                    if pool_cell["status"] != "ok":
                        _print_cell_result(f"{prefix}: pool build", pool_cell)
                        for backend in backends:
                            cell = {"batch": B, "context": context, "kv_dtype": kv_dtype, "backend": backend,
                                     "page_size": args.page_size, "stage": "pool_build", **pool_cell}
                            results["decode_cells"].append(cell)
                        continue
                    pool, slots = pool_cell["value"]

                    for backend in backends:
                        cell_prefix = f"{prefix} backend={backend}"
                        cell = safe_run(
                            lambda pool=pool, slots=slots, backend=backend: run_decode_backend(
                                pool, slots, backend, args.warmup, args.iters, device),
                            label=f"decode_ctx{context}_B{B}_{kv_dtype}_{backend}", timeout_s=timeout_s)
                        cell.update({"batch": B, "context": context, "kv_dtype": kv_dtype,
                                     "backend": backend, "page_size": args.page_size})
                        _print_cell_result(cell_prefix, cell)
                        results["decode_cells"].append(cell)

    if not args.skip_prefill:
        for T in prefill_tokens:
            for kv_dtype in kv_dtypes:
                prefix = f"[bench_attn] prefill T={T} kv_dtype={kv_dtype}"
                cell = safe_run(
                    lambda T=T, kv_dtype=kv_dtype: bench_prefill(
                        T, kv_dtype, args.page_size, args.warmup, args.iters, device),
                    label=f"prefill_T{T}_{kv_dtype}", timeout_s=timeout_s)
                cell.update({"T": T, "kv_dtype": kv_dtype, "page_size": args.page_size})
                _print_cell_result(prefix, cell, us_key="attn_mean_us")
                results["prefill_cells"].append(cell)

    json_path = write_results(args.out, results)
    print(f"[bench_attn] wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
