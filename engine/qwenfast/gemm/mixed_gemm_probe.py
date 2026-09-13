#!/usr/bin/env python3
"""What a *mixed*-step GEMM costs, and the ways to change it.

In the every-step-mixing configuration, the graphed chunk-1,024 step is about
109 ms of device time for 1,016 prefill tokens (9,300 tok/s) against 583 ms
for 8,020 (13,750 tok/s), because a 1,024-row GEMM is simply a less efficient
GEMM. A mixed step presents ``M = prefill_chunk + decode_rows`` (1,152 / 1,280
/ 2,304 at the shapes the scheduler actually runs), and `bench_gemm.py` does
not bench any of those M values (its sweep is the power-of-two bucket list).

This module benches exactly those M, and prices the three levers that could
move them. Each is a *hypothesis about why a mixed-step GEMM is inefficient*,
and each is answered with a graph-timed number rather than an argument:

1. **DeepGEMM's tile/config heuristic** (``--probe deepgemm-config``). DeepGEMM
   picks ``(block_m, block_n, num_stages, num_sms, ...)`` per ``(M, N, K)``
   inside ``get_best_configs``. At M ~ 1,100 with only 132 SMs there may be no
   tiling that fills the machine: a [34816, 5120] weight at block_n=128 is 272
   column tiles, and ``ceil(1152/block_m)`` row tiles, so the wave quantisation
   is visible from the config alone. This probe *reads the heuristic's answer*
   for every (shape, M) rather than inferring it, and reports the implied tile
   count and wave count against ``num_sms``.

2. **Splitting the step's rows into two GEMM calls** (``--probe split``).
   The mixed step's M is a concatenation of a prefill block and a decode block.
   If the inefficiency were a *tail effect* -- one ragged wave -- then two
   calls at M=1,024 and M=128 could beat one at M=1,152. If it is a
   *fixed-M-cost* effect, two calls must lose, because they read the weight
   twice. The two hypotheses predict opposite signs, which is what makes this
   worth measuring: a 27B model's [34816, 5120] weight is 179 MB, so a second
   read is ~37 µs at 4.8 TB/s and the split has to beat that before it can
   break even.

3. **Padding M up to the next tile multiple** (``--probe pad``). If the kernel
   quantises M to a 64/128/256-row tile internally, padding to that multiple is
   free work that costs nothing extra; if it does not, padding is pure waste.
   Padding is also the *cheap* fix if it wins, because the mixed step already
   pads, so it would be a change to the pad target, not a kernel.

4. **The activation quant** (``--probe quant``). Every one of the five
   fp8-activation backends quantises its ``[M, K]`` bf16 activation first.
   At M=8192 the quant is a small share of the call; at M ~ 1,150 the GEMM
   is ~7x cheaper while the quant is only
   ~7x cheaper too, so the *fraction* is the thing to check, not the absolute.

Also benched, because it is the one backend whose small-M behaviour is a
library heuristic we do not control at all: ``torch._scaled_mm`` (cuBLASLt),
which is the ``scaled_mm_pertensor`` backend.

GPU-only; imports cleanly (and ``--print-plan`` runs) with no torch.

Example (GPU host)::

    PYTHONPATH=/home/engine python -m qwenfast.gemm.mixed_gemm_probe \\
        --out /home/qwenfast-results/mixed_probe.json \\
        --m 1152,1280,1536,2304,4352,8448
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple

# The six fused shapes the mixed step runs, plus their per-step call counts --
# kept in sync with bench_gemm.SHAPES. Only the fp8
# ones: `gdn_in_proj_ba` is bf16 in the checkpoint and `lm_head` runs at M=B,
# not at the mixed step's M.
MIXED_SHAPES: List[Dict[str, Any]] = [
    dict(name="mlp_gate_up_proj", n=34816, k=5120, count=64),
    dict(name="mlp_down_proj", n=5120, k=17408, count=64),
    dict(name="gdn_in_proj_qkvz", n=16384, k=5120, count=48),
    dict(name="gdn_out_proj", n=5120, k=6144, count=48),
    dict(name="attn_qkv_proj", n=14336, k=5120, count=16),
    dict(name="attn_o_proj", n=5120, k=6144, count=16),
]

#: The M values a mixed step actually presents: `prefill_chunk + decode_rows`
#: for the common chunk sizes against the row counts the
#: scheduler runs at conc 128/256. 8448 is the chunk-8,192 control: the
#: configuration with the best served throughput, and therefore the
#: efficiency the smaller chunks are compared against.
DEFAULT_M: List[int] = [1152, 1280, 1536, 2304, 4352, 8448]

#: The five backends that quantize their activations. Above M=512 these are the
#: only ones in contention (bench_gemm.LARGE_M_FP8_ACTIVATION_BACKENDS; marlin
#: measures about 356 ms/step at M=2048, worse than dequant-per-call).
DEFAULT_BACKENDS: List[str] = [
    "deepgemm",
    "flashinfer_fp8_blockscale",
    "vllm_block_fp8_cutlass",
    "vllm_cutlass_fp8_pertensor",
    "scaled_mm_pertensor",
]

#: How the mixed step's M decomposes, per benched M: (prefill tokens, decode
#: rows). This is what `--probe split` splits on -- the real boundary, not an
#: arbitrary one, because the two halves are contiguous in the real activation.
SPLIT_OF_M: Dict[int, Tuple[int, int]] = {
    1152: (1024, 128),
    1280: (1024, 256),
    1536: (1024, 512),
    2304: (2048, 256),
    4352: (4096, 256),
    8448: (8192, 256),
}

PROBES = ("deepgemm-config", "deepgemm-knobs", "split", "pad", "quant")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _safe(fn, label: str) -> Dict[str, Any]:
    try:
        out = fn()
        out.setdefault("status", "ok")
        return out
    except Exception as exc:  # noqa: BLE001 -- a probe that cannot run is a result
        return {"status": "error", "label": label,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=6)}


def _graph_us(fn, warmup: int, iters: int, n_capture: int) -> float:
    """Per-call µs under CUDA-graph replay -- `bench_gemm.cuda_graph_time_fn`'s
    `graph_us`, reused verbatim so every number in this module is comparable
    cell-for-cell with the sweep's."""
    from .bench_gemm import cuda_graph_time_fn

    return cuda_graph_time_fn(fn, warmup, iters, n_capture=n_capture)["graph_us"]


def _make_inputs(m: int, n: int, k: int, device: str):
    import torch

    from .fused_weights import quantize_bf16_to_fp8_block128

    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    w = quantize_bf16_to_fp8_block128(torch.randn(n, k, device=device, dtype=torch.bfloat16))
    return x, w


# --------------------------------------------------------------------------- #
# probe 1: DeepGEMM's tile/config heuristic
# --------------------------------------------------------------------------- #
def _deepgemm_module():
    """The vendored DeepGEMM package -- vLLM 0.28.0 ships its own copy at
    ``vllm.third_party.deep_gemm`` and the standalone PyPI package is not
    installed in the serving environment (see dispatch.py's
    `_deepgemm_available` docstring). Try the standalone first anyway so this reads whichever one
    the dispatcher would actually call."""
    last: Optional[BaseException] = None
    for name in ("deep_gemm", "vllm.third_party.deep_gemm"):
        try:
            mod = __import__(name, fromlist=["*"])
            return name, mod
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise RuntimeError(f"no DeepGEMM module importable: {last}")


#: DeepGEMM knobs this build exposes at the Python level, and what each one is.
#: The vLLM-vendored DeepGEMM has **no**
#: `get_best_configs` anywhere -- its entire Python surface is `__init__.py` plus
#: `utils/` and `testing/`, and every tile/stage decision lives in the compiled
#: C++ (`include/deep_gemm/scheduler/gemm.cuh`, `impls/sm90_fp8_gemm_1d*.cuh`),
#: chosen inside the kernel's own scheduler rather than by a Python heuristic a
#: caller can read or override. What it *does* expose are three global setters,
#: which is a smaller lever than picking a tile but a real one:
#:
#:   set_num_sms(n)                 how many SMs the persistent scheduler spans
#:   set_block_size_multiple_of(n)  granularity the block sizes are rounded to
#:   set_tc_util(pct)               tensor-core utilisation target
#:
#: All three are process-global and read at kernel-launch time, so a sweep of
#: them is a legitimate "what would a different config have done" probe even
#: without access to the chooser itself.
DEEPGEMM_NUM_SMS_SWEEP: Tuple[int, ...] = (132, 128, 120, 112, 96, 64)
DEEPGEMM_BLOCK_MULTIPLE_SWEEP: Tuple[int, ...] = (16, 32, 64, 128)


def probe_deepgemm_knobs(m_values: List[int], shapes: List[Dict[str, Any]], device: str,
                         warmup: int, iters: int, n_capture: int) -> Dict[str, Any]:
    """Sweep DeepGEMM's exposed globals at the mixed step's M.

    The hypothesis this tests is the same one `probe_deepgemm_config` was written
    for -- that at M ~ 1,100 the tiling leaves the machine partly idle and a
    different config would fill it. `set_num_sms` is the one knob that can move
    a *wave* boundary from the outside: if the shape is tail-quantised, capping
    the scheduler at fewer SMs than the device has can (counter-intuitively) go
    faster, because the last partial wave is what costs. If nothing moves, the
    kernel's own scheduler is already making the right call and there is no tile
    lever here at all -- which is a result, not a non-result."""
    from . import dispatch
    from .dispatch import _BACKENDS

    dispatch._ensure_deepgemm_use_env()
    dispatch._ensure_deepgemm_cuda_home()
    mod_name, mod = _deepgemm_module()
    fn = _BACKENDS["deepgemm"]
    out: Dict[str, Any] = {"module": mod_name, "rows": [],
                            "exposed": [n for n in ("get_num_sms", "set_num_sms",
                                                    "set_block_size_multiple_of",
                                                    "set_tc_util", "get_tc_util")
                                        if hasattr(mod, n)]}
    get_num_sms = getattr(mod, "get_num_sms", None)
    set_num_sms = getattr(mod, "set_num_sms", None)
    set_block_mult = getattr(mod, "set_block_size_multiple_of", None)
    out["default_num_sms"] = int(get_num_sms()) if callable(get_num_sms) else None
    default_sms = out["default_num_sms"]

    for shape in shapes:
        n, k = shape["n"], shape["k"]
        for m in m_values:
            x, w = _make_inputs(m, n, k, device)
            fn(x, w)
            base = _graph_us(lambda: fn(x, w), warmup, iters, n_capture)
            row: Dict[str, Any] = {"shape": shape["name"], "m": m, "n": n, "k": k,
                                    "base_us": base, "num_sms": [], "block_multiple": []}
            if callable(set_num_sms) and default_sms:
                for sms in DEEPGEMM_NUM_SMS_SWEEP:
                    try:
                        set_num_sms(int(sms))
                        fn(x, w)
                        us = _graph_us(lambda: fn(x, w), warmup, iters, n_capture)
                        row["num_sms"].append({"num_sms": sms, "us": us,
                                               "speedup": base / us if us else float("nan")})
                    except Exception as exc:  # noqa: BLE001
                        row["num_sms"].append({"num_sms": sms, "error": f"{type(exc).__name__}: {exc}"})
                set_num_sms(int(default_sms))
            if callable(set_block_mult):
                for mult in DEEPGEMM_BLOCK_MULTIPLE_SWEEP:
                    try:
                        set_block_mult(int(mult))
                        fn(x, w)
                        us = _graph_us(lambda: fn(x, w), warmup, iters, n_capture)
                        row["block_multiple"].append({"multiple": mult, "us": us,
                                                      "speedup": base / us if us else float("nan")})
                    except Exception as exc:  # noqa: BLE001
                        row["block_multiple"].append({"multiple": mult,
                                                      "error": f"{type(exc).__name__}: {exc}"})
                try:
                    set_block_mult(8)  # DeepGEMM's own default
                except Exception:  # noqa: BLE001
                    pass
            out["rows"].append(row)
            del x, w
    return out


def _find_get_best_configs(mod) -> Tuple[str, Any]:
    """Locate ``get_best_configs`` (DeepGEMM's per-(M, N, K) tile chooser).

    Its home has moved between DeepGEMM releases (``jit_kernels.gemm``,
    ``jit_kernels.utils``, ``utils.layout``, ...), so search rather than
    hard-code an import path -- a probe that reports "not found" because the
    module moved would be indistinguishable from one that reports "this build
    has no such heuristic", and those are very different facts."""
    import importlib
    import pkgutil

    candidates = [
        mod.__name__ + ".jit_kernels.gemm",
        mod.__name__ + ".jit_kernels.impls.sm90_bf16_gemm",
        mod.__name__ + ".jit_kernels.impls.sm90_fp8_gemm_1d1d",
        mod.__name__ + ".jit_kernels.impls.sm90_fp8_gemm_1d2d",
        mod.__name__ + ".jit_kernels.utils",
        mod.__name__ + ".utils.layout",
    ]
    seen = set()
    if hasattr(mod, "__path__"):
        for info in pkgutil.walk_packages(mod.__path__, mod.__name__ + "."):
            candidates.append(info.name)
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        try:
            sub = importlib.import_module(name)
        except Exception:  # noqa: BLE001 -- a submodule that needs CUDA/JIT is fine to skip
            continue
        fn = getattr(sub, "get_best_configs", None)
        if callable(fn):
            return name, fn
    raise RuntimeError(
        f"get_best_configs not found anywhere under {mod.__name__} "
        f"(searched {len(seen)} submodules)"
    )


def probe_deepgemm_config(m_values: List[int], shapes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Ask DeepGEMM's own heuristic what it would do at each (M, N, K), and
    turn its answer into the two numbers that decide whether the machine is
    full: how many CTAs the tiling produces, and how many **waves** of
    ``num_sms`` that is. A wave count just above an integer (1.03, 2.06) is the
    signature of tail quantisation -- the thing `--probe pad` would fix -- and a
    wave count *below* 1 means the shape cannot fill the GPU at all, which no
    amount of padding fixes."""
    import inspect

    import torch

    from . import dispatch

    dispatch._ensure_deepgemm_use_env()
    dispatch._ensure_deepgemm_cuda_home()
    mod_name, mod = _deepgemm_module()
    where, get_best_configs = _find_get_best_configs(mod)
    sig = inspect.signature(get_best_configs)
    num_sms_fn = getattr(mod, "get_num_sms", None)
    device_sms = torch.cuda.get_device_properties(0).multi_processor_count

    out: Dict[str, Any] = {
        "module": mod_name,
        "get_best_configs_at": where,
        "signature": str(sig),
        "source_file": inspect.getsourcefile(get_best_configs),
        "device_sm_count": device_sms,
        "get_num_sms": int(num_sms_fn()) if callable(num_sms_fn) else None,
        "rows": [],
    }
    params = list(sig.parameters)
    for shape in shapes:
        for m in m_values:
            row: Dict[str, Any] = {"shape": shape["name"], "m": m, "n": shape["n"], "k": shape["k"]}
            # Signatures seen in the wild: (m, n, k, num_groups, num_sms, ...)
            # and (m, n, k, num_groups, num_sms, is_grouped_contiguous=...).
            # Build positionally from whatever the first five names are.
            args: List[Any] = []
            for p in params[:5]:
                if p in ("m",):
                    args.append(m)
                elif p in ("n",):
                    args.append(shape["n"])
                elif p in ("k",):
                    args.append(shape["k"])
                elif p in ("num_groups", "num_group"):
                    args.append(1)
                elif p in ("num_sms",):
                    args.append(int(num_sms_fn()) if callable(num_sms_fn) else device_sms)
                else:
                    break
            try:
                cfg = get_best_configs(*args)
                row["args"] = {p: a for p, a in zip(params, args)}
                row["config"] = [str(c) for c in cfg] if isinstance(cfg, tuple) else str(cfg)
                block_m = block_n = None
                if isinstance(cfg, tuple) and len(cfg) >= 2:
                    if isinstance(cfg[0], int) and isinstance(cfg[1], int):
                        block_m, block_n = cfg[0], cfg[1]
                if block_m and block_n:
                    sms = int(num_sms_fn()) if callable(num_sms_fn) else device_sms
                    tiles = -(-m // block_m) * -(-shape["n"] // block_n)
                    row.update({
                        "block_m": block_m, "block_n": block_n,
                        "m_tiles": -(-m // block_m), "n_tiles": -(-shape["n"] // block_n),
                        "ctas": tiles, "num_sms": sms,
                        "waves": round(tiles / sms, 3),
                        "wave_efficiency": round(tiles / (sms * -(-tiles // sms)), 3),
                        "m_pad_waste": round((block_m * -(-m // block_m) - m) / m, 4),
                    })
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"{type(exc).__name__}: {exc}"
            out["rows"].append(row)
    return out


# --------------------------------------------------------------------------- #
# probe 2: split the mixed step's rows into two GEMM calls
# --------------------------------------------------------------------------- #
def probe_split(m: int, shape: Dict[str, Any], backend: str, device: str,
                warmup: int, iters: int, n_capture: int) -> Dict[str, Any]:
    """One call at M vs two calls at (prefill, decode). Same weight, same
    graph-capture methodology, same total rows -- so the delta is the tail
    effect and the extra weight read, and nothing else."""
    from .dispatch import _BACKENDS

    fn = _BACKENDS[backend]
    n, k = shape["n"], shape["k"]
    mp, md = SPLIT_OF_M.get(m, (m - 256, 256))
    x, w = _make_inputs(m, n, k, device)
    xa, xb = x[:mp].contiguous(), x[mp:].contiguous()
    fn(x, w)  # absorb JIT for both shapes before any timing
    fn(xa, w)
    fn(xb, w)
    one = _graph_us(lambda: fn(x, w), warmup, iters, n_capture)
    two = _graph_us(lambda: (fn(xa, w), fn(xb, w)), warmup, iters, n_capture)
    return {"m": m, "m_prefill": mp, "m_decode": md, "shape": shape["name"], "backend": backend,
            "one_call_us": one, "two_call_us": two,
            "split_speedup": one / two if two else float("nan"),
            "split_wins": two < one}


# --------------------------------------------------------------------------- #
# probe 3: pad M up to the next tile multiple
# --------------------------------------------------------------------------- #
PAD_MULTIPLES: Tuple[int, ...] = (64, 128, 256)


def probe_pad(m: int, shape: Dict[str, Any], backend: str, device: str,
              warmup: int, iters: int, n_capture: int) -> Dict[str, Any]:
    """Time M, then M padded up to the next multiple of 64 / 128 / 256.

    A padded call does strictly *more* arithmetic, so ``padded_us < base_us``
    can only mean the kernel was already paying for those rows -- i.e. M was
    being quantised internally and the tail rows were free. That is a real,
    if counter-intuitive, thing for a tile-quantised kernel and it is exactly
    what makes padding a candidate fix; anything else and the answer is "do not
    pad"."""
    from .dispatch import _BACKENDS

    fn = _BACKENDS[backend]
    n, k = shape["n"], shape["k"]
    x, w = _make_inputs(m, n, k, device)
    fn(x, w)
    base = _graph_us(lambda: fn(x, w), warmup, iters, n_capture)
    row: Dict[str, Any] = {"m": m, "shape": shape["name"], "backend": backend, "base_us": base,
                            "pads": []}
    for mult in PAD_MULTIPLES:
        padded_m = -(-m // mult) * mult
        if padded_m == m:
            row["pads"].append({"multiple": mult, "padded_m": m, "padded_us": base,
                                "speedup": 1.0, "note": "already aligned"})
            continue
        xp, _ = _make_inputs(padded_m, n, k, device)
        fn(xp, w)
        us = _graph_us(lambda: fn(xp, w), warmup, iters, n_capture)
        row["pads"].append({"multiple": mult, "padded_m": padded_m, "padded_us": us,
                            "speedup": base / us if us else float("nan"),
                            "pad_wins": us < base})
    return row


# --------------------------------------------------------------------------- #
# probe 4: the activation quantization, in isolation
# --------------------------------------------------------------------------- #
def probe_quant(m: int, shape: Dict[str, Any], backend: str, device: str,
                warmup: int, iters: int, n_capture: int) -> Dict[str, Any]:
    """The backend's activation-quant kernel alone, and its share of the whole
    call. Uses `bench_gemm._QUANT_ONLY_CALL` -- the same mapping the sweep uses
    -- so "quant" means the same kernel in both artefacts."""
    from . import dispatch as dispatch_mod
    from .bench_gemm import _QUANT_ONLY_CALL
    from .dispatch import _BACKENDS

    factory = _QUANT_ONLY_CALL.get(backend)
    if factory is None:
        return {"m": m, "shape": shape["name"], "backend": backend,
                "status": "skipped", "reason": "backend takes bf16 activations (no quant step)"}
    fn = _BACKENDS[backend]
    n, k = shape["n"], shape["k"]
    x, w = _make_inputs(m, n, k, device)
    fn(x, w)
    factory(dispatch_mod, x)
    full = _graph_us(lambda: fn(x, w), warmup, iters, n_capture)
    quant = _graph_us(lambda: factory(dispatch_mod, x), warmup, iters, n_capture)
    return {"m": m, "shape": shape["name"], "backend": backend, "k": k,
            "full_us": full, "quant_us": quant,
            "quant_frac": quant / full if full else float("nan"),
            # the quant reads [M, K] bf16 and writes [M, K] fp8 + scales
            "quant_bytes": m * k * 3,
            "quant_gbps": (m * k * 3) / (quant * 1e-6) / 1e9 if quant else float("nan")}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="mixed_gemm_probe.json")
    p.add_argument("--m", default=",".join(str(m) for m in DEFAULT_M),
                   help="Comma-separated M values (default: the mixed step's real M).")
    p.add_argument("--shapes", default=None,
                   help="Comma-separated shape names (default: all six fp8 mixed-step shapes).")
    p.add_argument("--backends", default=",".join(DEFAULT_BACKENDS))
    p.add_argument("--probe", default=",".join(PROBES),
                   help=f"Comma-separated subset of {PROBES}.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--graph-n-capture", type=int, default=10)
    p.add_argument("--split-pad-shapes", default="mlp_gate_up_proj,mlp_down_proj,gdn_in_proj_qkvz",
                   help="The split/pad probes are O(shapes x M x backends x pads) and the answer "
                        "they give is a property of the tiling, not of the layer -- so by default "
                        "they run on the three shapes that are ~85%% of a step's GEMM FLOPs rather "
                        "than on all six. Pass 'all' to run every shape.")
    p.add_argument("--print-plan", action="store_true",
                   help="Print what would be measured and exit (no GPU needed).")
    args = p.parse_args(argv)

    m_values = [int(x) for x in args.m.split(",")]
    shapes = list(MIXED_SHAPES)
    if args.shapes:
        want = set(args.shapes.split(","))
        shapes = [s for s in shapes if s["name"] in want]
    backends = args.backends.split(",")
    probes = [x for x in args.probe.split(",") if x]
    unknown = set(probes) - set(PROBES)
    if unknown:
        p.error(f"unknown --probe {sorted(unknown)}; choose from {PROBES}")
    if args.split_pad_shapes == "all":
        sp_shapes = shapes
    else:
        want = set(args.split_pad_shapes.split(","))
        sp_shapes = [s for s in shapes if s["name"] in want]

    if args.print_plan:
        print(f"M values     : {m_values}")
        print(f"shapes       : {[s['name'] for s in shapes]}")
        print(f"backends     : {backends}")
        print(f"probes       : {probes}")
        print(f"split/pad on : {[s['name'] for s in sp_shapes]}")
        cells = (len(sp_shapes) * len(m_values) * len(backends)
                 * (("split" in probes) + ("pad" in probes) * (1 + len(PAD_MULTIPLES))))
        cells += len(shapes) * len(m_values) * len(backends) * 2 * ("quant" in probes)
        print(f"graph-timed cells: ~{cells}")
        return 0

    from .bench_gemm import env_metadata

    results: Dict[str, Any] = {"env": env_metadata(args.device), "args": vars(args),
                               "deepgemm_config": None, "deepgemm_knobs": None,
                               "split": [], "pad": [], "quant": []}
    out_path = args.out if args.out.endswith(".json") else args.out + ".json"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    def flush() -> None:
        tmp = out_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(results, f, indent=2, default=str)
        os.replace(tmp, out_path)

    import torch

    if not torch.cuda.is_available():
        results["error"] = "torch.cuda.is_available() is False"
        flush()
        print("[mixed_gemm_probe] FAILED (no CUDA)", file=sys.stderr)
        return 1
    torch.cuda.set_device(args.device)

    if "deepgemm-config" in probes:
        print("[mixed_gemm_probe] deepgemm config heuristic ...", file=sys.stderr)
        results["deepgemm_config"] = _safe(
            lambda: probe_deepgemm_config(m_values, shapes), "deepgemm-config")
        flush()

    if "deepgemm-knobs" in probes:
        print("[mixed_gemm_probe] deepgemm knob sweep ...", file=sys.stderr)
        results["deepgemm_knobs"] = _safe(
            lambda: probe_deepgemm_knobs(m_values, sp_shapes, args.device,
                                         args.warmup, args.iters, args.graph_n_capture),
            "deepgemm-knobs")
        flush()

    for shape in sp_shapes:
        for m in m_values:
            for backend in backends:
                if "split" in probes:
                    print(f"[mixed_gemm_probe] split {shape['name']} M={m} {backend}", file=sys.stderr)
                    cell = _safe(lambda s=shape, m=m, b=backend: probe_split(
                        m, s, b, args.device, args.warmup, args.iters, args.graph_n_capture),
                        f"split_{shape['name']}_{m}_{backend}")
                    cell.setdefault("shape", shape["name"]); cell.setdefault("m", m)
                    cell.setdefault("backend", backend)
                    results["split"].append(cell); flush()
                if "pad" in probes:
                    print(f"[mixed_gemm_probe] pad {shape['name']} M={m} {backend}", file=sys.stderr)
                    cell = _safe(lambda s=shape, m=m, b=backend: probe_pad(
                        m, s, b, args.device, args.warmup, args.iters, args.graph_n_capture),
                        f"pad_{shape['name']}_{m}_{backend}")
                    cell.setdefault("shape", shape["name"]); cell.setdefault("m", m)
                    cell.setdefault("backend", backend)
                    results["pad"].append(cell); flush()

    if "quant" in probes:
        for shape in shapes:
            for m in m_values:
                for backend in backends:
                    print(f"[mixed_gemm_probe] quant {shape['name']} M={m} {backend}", file=sys.stderr)
                    cell = _safe(lambda s=shape, m=m, b=backend: probe_quant(
                        m, s, b, args.device, args.warmup, args.iters, args.graph_n_capture),
                        f"quant_{shape['name']}_{m}_{backend}")
                    cell.setdefault("shape", shape["name"]); cell.setdefault("m", m)
                    cell.setdefault("backend", backend)
                    results["quant"].append(cell); flush()

    flush()
    print(f"[mixed_gemm_probe] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
