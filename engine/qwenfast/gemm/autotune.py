"""M-bucketed GEMM autotune cache.

For each fused-weight ``(N, K)`` shape and each M bucket in
``dispatch.M_BUCKETS``, times every registered backend
(``dispatch._BACKENDS``) on real random data on the current GPU, picks the
fastest one, and persists the winner to a JSON file keyed by
``sm_version`` (``torch.cuda.get_device_capability()`` as ``major*10+minor``,
e.g. 90 for H200). ``dispatch.linear()`` consults this cache (lazily,
process-cached) before falling back to the static default backend order.

Timing is CUDA-graph-captured/replayed (``bench_gemm.cuda_graph_time_fn``,
reused here rather than duplicated), not eager. Eager per-call timing at
these (small-M, small-per-layer-shape) sizes is dominated by CPU
launch/dispatch overhead (~55µs/call), not GPU compute; the GDN kernels show
the same pattern (eager 70µs floor, 4µs under graph replay). Since the engine
always runs decode inside a CUDA graph (one graph per batch-size bucket), an
eager-timed "winner" can be the wrong pick for how this backend actually
runs in production; graph-replayed time is what should decide the cache.
Both ``graph_us`` and the old eager timing (now ``eager_us``) are kept in
the persisted ``timings_us`` per backend for transparency, and ``best`` is
chosen by ``graph_us`` (falling back to ``eager_us`` for a backend that
isn't graph-capturable at this shape -- recorded via ``<name>_graph_error``
so a non-capturable backend winning "by default" is visible in the cache,
not silent).

Must run on a CUDA host (``run_autotune`` / ``autotune_shape``); importing
this module and calling the pure lookup/cache-path helpers is safe with no
GPU (macOS, ``py_compile``) -- no torch import at module scope.

Usage (GPU host)::

    PYTHONPATH=/home/engine python -m qwenfast.gemm.autotune \\
        --out-dir /home/qwenfast-results/gemm_autotune --device cuda:0
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Tuple

# The fused (N, K) shapes that actually occur in the model, matching
# kernels/microbench/common.py's GEMM_SHAPES (fused variants). K is always hidden_size=5120 except
# down_proj/mlp (K=intermediate*2/... no -- down_proj K=intermediate=17408)
# and out_proj/o_proj (K=value_dim/o_proj_in). Kept in one place so
# bench_gemm.py and this module never drift apart.
MODEL_GEMM_SHAPES: List[Tuple[str, int, int]] = [
    ("gdn_in_proj_qkvz", 16384, 5120),
    ("gdn_in_proj_ba", 96, 5120),  # bf16, not autotuned (no FP8 backend applies)
    ("gdn_out_proj", 5120, 6144),
    ("attn_qkv_proj", 14336, 5120),
    ("attn_o_proj", 5120, 6144),
    ("mlp_gate_up_proj", 34816, 5120),
    ("mlp_down_proj", 5120, 17408),
    ("lm_head", 248320, 5120),
]

_CACHE_DIR_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "autotune_cache")


def cache_path(sm_version: int, cache_dir: Optional[str] = None) -> str:
    d = cache_dir or _CACHE_DIR_DEFAULT
    return os.path.join(d, f"sm{sm_version}.json")


_LOADED: Dict[str, dict] = {}


def _load(sm_version: int, cache_dir: Optional[str] = None) -> dict:
    path = cache_path(sm_version, cache_dir)
    if path in _LOADED:
        return _LOADED[path]
    data: dict = {}
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    _LOADED[path] = data
    return data


def _key(n: int, k: int, m: int) -> str:
    return f"{n}x{k}x{m}"


def lookup(
    sm_version: Optional[int], n: int, k: int, m: int, cache_dir: Optional[str] = None
) -> Optional[str]:
    """Return the cached winning backend name for this shape, or ``None`` if
    there's no cache / no entry (caller falls back to the static default)."""
    if sm_version is None:
        return None
    from .dispatch import m_bucket

    data = _load(sm_version, cache_dir)
    entry = data.get(_key(n, k, m_bucket(m)))
    return entry.get("backend") if entry else None


def invalidate_cache(cache_dir: Optional[str] = None) -> None:
    """Drop the in-process cache of loaded JSON files (tests / after a
    manual edit of the on-disk cache)."""
    if cache_dir is None:
        _LOADED.clear()
        return
    for path in list(_LOADED):
        if os.path.dirname(path) == cache_dir:
            del _LOADED[path]


def detect_sm_version(device: str = "cuda:0") -> int:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False -- autotune needs a real GPU")
    idx = torch.device(device).index or 0
    major, minor = torch.cuda.get_device_capability(idx)
    return major * 10 + minor


# --------------------------------------------------------------------------- #
# GPU-only timing
# --------------------------------------------------------------------------- #
def _make_fp8_weight(n: int, k: int, device: str):
    import torch

    from .fused_weights import quantize_bf16_to_fp8_block128

    w_bf16 = torch.randn(n, k, device=device, dtype=torch.bfloat16)
    fw = quantize_bf16_to_fp8_block128(w_bf16)
    return fw


def _time_backend_eager(fn, x, w, warmup: int, iters: int) -> float:
    """Wall-clock-around-N-calls timing -- kept only as the ``eager_us``
    diagnostic field and the fallback for a backend that isn't
    graph-capturable at this shape. NOT used to pick ``best`` (see the
    module docstring): this measures CPU launch
    overhead as much as GPU compute at the shapes/M this engine cares
    about."""
    import torch

    for _ in range(warmup):
        fn(x, w)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(x, w)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # microseconds


def _time_backend_graph(fn, x, w, warmup: int, iters: int, n_capture: int = 20) -> float:
    """CUDA-graph-captured/replayed per-call timing -- the metric ``best`` is
    chosen by (see the module docstring). Reuses
    ``bench_gemm.cuda_graph_time_fn`` rather than duplicating the capture
    logic; raises if this backend/shape isn't graph-capturable (caller
    catches and falls back to eager)."""
    from .bench_gemm import cuda_graph_time_fn

    timing = cuda_graph_time_fn(lambda: fn(x, w), warmup, iters, n_capture=n_capture)
    return timing["graph_us"]


def autotune_shape(
    n: int,
    k: int,
    m_buckets: Optional[List[int]] = None,
    warmup: int = 10,
    iters: int = 30,
    device: str = "cuda:0",
    graph_n_capture: int = 20,
) -> Dict[str, dict]:
    """Time every registered backend at this ``(N, K)`` for every M bucket.
    Returns a dict keyed like the on-disk cache (``"{N}x{K}x{M}"``), each
    entry ``{"backend": <winner-by-graph_us>, "timings_us": {name: eager_us,
    f"{name}_graph": graph_us, f"{name}_graph_error": <str, if not
    capturable>, ...}}``."""
    import torch

    from . import dispatch

    m_buckets = m_buckets or dispatch.M_BUCKETS
    w = _make_fp8_weight(n, k, device)
    results: Dict[str, dict] = {}
    for m in m_buckets:
        x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
        timings: Dict[str, Optional[float]] = {}
        graph_timings: Dict[str, Optional[float]] = {}
        for name, fn in dispatch._BACKENDS.items():
            try:
                timings[name] = _time_backend_eager(fn, x, w, warmup, iters)
            except Exception as exc:  # noqa: BLE001
                timings[name] = None
                timings[f"{name}_error"] = f"{type(exc).__name__}: {exc}"  # type: ignore[assignment]
            try:
                graph_timings[name] = _time_backend_graph(fn, x, w, warmup, iters, n_capture=graph_n_capture)
                timings[f"{name}_graph"] = graph_timings[name]  # type: ignore[assignment]
            except Exception as exc:  # noqa: BLE001
                graph_timings[name] = None
                timings[f"{name}_graph_error"] = f"{type(exc).__name__}: {exc}"  # type: ignore[assignment]
        # rank by graph_us; a backend with no graph number (capture failed)
        # falls back to its eager number so it can still win if every
        # graph-capturable backend is worse -- but is visibly worse-off in
        # the persisted timings_us either way, since its `_graph_error` key
        # says so.
        ranked: Dict[str, float] = {}
        for name in dispatch._BACKENDS:
            g = graph_timings.get(name)
            e = timings.get(name)
            if isinstance(g, (int, float)):
                ranked[name] = g
            elif isinstance(e, (int, float)):
                ranked[name] = e
        best = min(ranked, key=ranked.get) if ranked else None  # type: ignore[arg-type]
        results[_key(n, k, m)] = {"backend": best, "timings_us": timings}
    return results


def run_autotune(
    shapes: Optional[List[Tuple[str, int, int]]] = None,
    out_dir: Optional[str] = None,
    device: str = "cuda:0",
    m_buckets: Optional[List[int]] = None,
    warmup: int = 10,
    iters: int = 30,
    graph_n_capture: int = 20,
) -> str:
    """Autotune every shape in ``shapes`` (default: ``MODEL_GEMM_SHAPES``)
    and persist/merge the results into ``cache_path(sm_version, out_dir)``.
    Returns the path written."""
    shapes = shapes if shapes is not None else MODEL_GEMM_SHAPES
    sm = detect_sm_version(device)
    path = cache_path(sm, out_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    data: dict = {}
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)

    for name, n, k in shapes:
        print(f"[autotune] {name} ({n}x{k}) ...")
        data.update(autotune_shape(n, k, m_buckets=m_buckets, warmup=warmup, iters=iters, device=device,
                                    graph_n_capture=graph_n_capture))

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    invalidate_cache(out_dir)
    return path


def _main(argv: Optional[List[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=str, default=None, help=f"default: {_CACHE_DIR_DEFAULT}")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--graph-n-capture", type=int, default=20,
                    help="Calls captured per CUDA graph for graph_us ranking (see the "
                         "module docstring). Matches bench_gemm.py's --graph-n-capture default.")
    args = p.parse_args(argv)

    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False -- autotune needs a real GPU")
    except Exception as exc:  # noqa: BLE001
        print(f"[autotune] FAILED (no CUDA): {exc}", flush=True)
        return 1

    path = run_autotune(out_dir=args.out_dir, device=args.device, warmup=args.warmup, iters=args.iters,
                        graph_n_capture=args.graph_n_capture)
    print(f"[autotune] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "MODEL_GEMM_SHAPES",
    "cache_path",
    "lookup",
    "invalidate_cache",
    "detect_sm_version",
    "autotune_shape",
    "run_autotune",
]
