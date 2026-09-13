"""Shared shapes, CUDA-timing, and result-reporting utilities for the
Gated-DeltaNet / Qwen3.8-27B kernel microbenchmark suite.

All model shapes below are read directly off
`engine/reference/config-Qwen3.8-27B-FP8.json` (text_config) and cross-checked
against the HF reference implementation `engine/reference/modeling_qwen3_5.py`
(classes `Qwen3_5GatedDeltaNet`, `Qwen3_5Attention`, `Qwen3_5MLP`). See
`docs/architecture.md` for the performance-modeling context these numbers
feed into.

This module must import cleanly with **no torch/CUDA available** (it is
`py_compile`-checked on a Mac laptop with no GPU and no torch installed) --
all torch usage lives inside function bodies, never at module scope.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import platform
import sys
import traceback
from typing import Any, Callable, Iterable, Optional

# --------------------------------------------------------------------------
# Model shapes (Qwen3.8-27B / Qwen3_5ForConditionalGeneration text backbone)
# --------------------------------------------------------------------------

HIDDEN_SIZE = 5120
NUM_LAYERS_TOTAL = 64
NUM_GDN_LAYERS = 48          # "linear_attention" layers
NUM_ATTN_LAYERS = 16         # "full_attention" layers (every 4th layer)
INTERMEDIATE_SIZE = 17408
VOCAB_SIZE = 248320
RMS_NORM_EPS = 1e-6

# --- Gated DeltaNet (linear_attention) layer, Qwen3_5GatedDeltaNet ---
GDN_NUM_V_HEADS = 48
GDN_NUM_K_HEADS = 16
GDN_HEAD_K_DIM = 128
GDN_HEAD_V_DIM = 128
GDN_KEY_DIM = GDN_HEAD_K_DIM * GDN_NUM_K_HEADS        # 2048
GDN_VALUE_DIM = GDN_HEAD_V_DIM * GDN_NUM_V_HEADS      # 6144
GDN_GVA_GROUP_SIZE = GDN_NUM_V_HEADS // GDN_NUM_K_HEADS  # 3 (q/k repeat_interleave factor)
GDN_CONV_KERNEL = 4
GDN_CONV_DIM = GDN_KEY_DIM * 2 + GDN_VALUE_DIM        # 10240 (mixed qkv conv channels)
GDN_STATE_DTYPE_DEFAULT = "float32"                    # config: mamba_ssm_dtype

# in_proj_qkv: hidden -> key_dim*2 + value_dim = 4096 + 6144 = 10240
GDN_IN_PROJ_QKV_OUT = GDN_KEY_DIM * 2 + GDN_VALUE_DIM  # 10240
# in_proj_z: hidden -> value_dim (output gate pre-activation)
GDN_IN_PROJ_Z_OUT = GDN_VALUE_DIM                      # 6144
# out_proj: value_dim -> hidden
GDN_OUT_PROJ_IN = GDN_VALUE_DIM                        # 6144

# --- Full attention (full_attention) layer, Qwen3_5Attention ---
ATTN_NUM_Q_HEADS = 24
ATTN_NUM_KV_HEADS = 4
ATTN_HEAD_DIM = 256
ATTN_PARTIAL_ROTARY_FACTOR = 0.25
# q_proj emits [query ; sigmoid-gate] fused, hence the x2
ATTN_Q_PROJ_OUT = ATTN_NUM_Q_HEADS * ATTN_HEAD_DIM * 2   # 12288
ATTN_KV_PROJ_OUT = ATTN_NUM_KV_HEADS * ATTN_HEAD_DIM     # 1024
ATTN_O_PROJ_IN = ATTN_NUM_Q_HEADS * ATTN_HEAD_DIM        # 6144

# --------------------------------------------------------------------------
# Batch / sequence sweeps requested for the suite
# --------------------------------------------------------------------------

DECODE_BATCH_SWEEP = [1, 8, 32, 64, 128, 256, 512]
PREFILL_T_SWEEP = [512, 2048, 8192]
PREFILL_BATCH = 1
ATTN_CONTEXT_SWEEP = [2048, 8192]

# --------------------------------------------------------------------------
# GEMM shape table (gemm_bench.py) -- exact per-layer linear shapes.
# `count` = number of times this GEMM fires per full-model forward step;
# `include_in_total` = whether it counts toward the whole-model extrapolation
# (the *_alt gate/up-separate rows are excluded to avoid double-counting the
# fused gate_up variant).
# --------------------------------------------------------------------------

GEMM_SHAPES = [
    # Gated DeltaNet layers (x48)
    dict(name="gdn_in_proj_qkv", in_f=HIDDEN_SIZE, out_f=GDN_IN_PROJ_QKV_OUT,
         group="gdn", count=NUM_GDN_LAYERS, include_in_total=True,
         note="hidden -> 2*key_dim(2048*2) + value_dim(6144) = 10240"),
    dict(name="gdn_in_proj_z", in_f=HIDDEN_SIZE, out_f=GDN_IN_PROJ_Z_OUT,
         group="gdn", count=NUM_GDN_LAYERS, include_in_total=True,
         note="hidden -> value_dim (output-gate pre-activation)"),
    dict(name="gdn_out_proj", in_f=GDN_OUT_PROJ_IN, out_f=HIDDEN_SIZE,
         group="gdn", count=NUM_GDN_LAYERS, include_in_total=True,
         note="value_dim -> hidden"),
    # Full attention layers (x16)
    dict(name="attn_q_proj", in_f=HIDDEN_SIZE, out_f=ATTN_Q_PROJ_OUT,
         group="attn", count=NUM_ATTN_LAYERS, include_in_total=True,
         note="hidden -> [q ; gate] fused, 24*256*2"),
    dict(name="attn_k_proj", in_f=HIDDEN_SIZE, out_f=ATTN_KV_PROJ_OUT,
         group="attn", count=NUM_ATTN_LAYERS, include_in_total=True,
         note="hidden -> 4*256"),
    dict(name="attn_v_proj", in_f=HIDDEN_SIZE, out_f=ATTN_KV_PROJ_OUT,
         group="attn", count=NUM_ATTN_LAYERS, include_in_total=True,
         note="hidden -> 4*256"),
    dict(name="attn_o_proj", in_f=ATTN_O_PROJ_IN, out_f=HIDDEN_SIZE,
         group="attn", count=NUM_ATTN_LAYERS, include_in_total=True,
         note="24*256 -> hidden"),
    # MLP, present on every one of the 64 layers (GDN and attention alike)
    dict(name="mlp_gate_up_fused", in_f=HIDDEN_SIZE, out_f=2 * INTERMEDIATE_SIZE,
         group="mlp", count=NUM_LAYERS_TOTAL, include_in_total=True,
         note="fused [gate_proj;up_proj], hidden -> 2*17408=34816"),
    dict(name="mlp_gate_proj", in_f=HIDDEN_SIZE, out_f=INTERMEDIATE_SIZE,
         group="mlp_alt", count=NUM_LAYERS_TOTAL, include_in_total=False,
         note="unfused alternative to mlp_gate_up_fused -- not double-counted in totals"),
    dict(name="mlp_up_proj", in_f=HIDDEN_SIZE, out_f=INTERMEDIATE_SIZE,
         group="mlp_alt", count=NUM_LAYERS_TOTAL, include_in_total=False,
         note="unfused alternative to mlp_gate_up_fused -- not double-counted in totals"),
    dict(name="mlp_down_proj", in_f=INTERMEDIATE_SIZE, out_f=HIDDEN_SIZE,
         group="mlp", count=NUM_LAYERS_TOTAL, include_in_total=True,
         note="17408 -> hidden"),
    # LM head
    dict(name="lm_head", in_f=HIDDEN_SIZE, out_f=VOCAB_SIZE,
         group="lm_head", count=1, include_in_total=True,
         note="untied, hidden -> vocab (248320)"),
]

GPU_BW_GBPS_DEFAULT = 4800.0   # H200 SXM peak HBM3e bandwidth
WEIGHT_BYTES_FP8 = 27e9        # ~27 GB FP8 weights, read once/step

# --------------------------------------------------------------------------
# CLI helpers
# --------------------------------------------------------------------------


def add_common_args(p: argparse.ArgumentParser, default_out: str) -> argparse.ArgumentParser:
    p.add_argument("--out", type=str, default=default_out,
                    help="Output path. Writes <out>.json and <out>.md (or, if --out "
                         "already ends in .json/.md, writes that + the sibling extension).")
    p.add_argument("--warmup", type=int, default=20, help="Warmup iterations before timing.")
    p.add_argument("--iters", type=int, default=100, help="Timed iterations.")
    p.add_argument("--device", type=str, default="cuda:0", help="CUDA device.")
    p.add_argument("--gpu-bw-gbps", type=float, default=GPU_BW_GBPS_DEFAULT,
                    help="HBM bandwidth (GB/s) used to compute the physics-model efficiency %%.")
    p.add_argument("--batches", type=str, default=None,
                    help="Comma-separated override of the batch sweep (default: "
                         f"{','.join(str(b) for b in DECODE_BATCH_SWEEP)}).")
    p.add_argument("--seed", type=int, default=0)
    return p


def parse_batches(arg: Optional[str], default: Iterable[int]) -> list[int]:
    if not arg:
        return list(default)
    return [int(x) for x in arg.split(",") if x.strip()]


# --------------------------------------------------------------------------
# Environment / device metadata
# --------------------------------------------------------------------------


def env_metadata(device: str = "cuda:0") -> dict[str, Any]:
    meta: dict[str, Any] = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
    }
    try:
        import torch  # noqa: WPS433 (intentionally local import)

        meta["torch_version"] = torch.__version__
        meta["torch_cuda_version"] = torch.version.cuda
        meta["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            idx = torch.device(device).index or 0
            meta["gpu_name"] = torch.cuda.get_device_name(idx)
            props = torch.cuda.get_device_properties(idx)
            meta["gpu_total_mem_gb"] = round(props.total_memory / 1e9, 2)
            meta["gpu_sm_count"] = props.multi_processor_count
            meta["gpu_capability"] = f"{props.major}.{props.minor}"
    except Exception as exc:  # pragma: no cover
        meta["torch_import_error"] = f"{type(exc).__name__}: {exc}"
    for pkg in ("fla", "causal_conv1d", "flashinfer", "vllm", "cula"):
        try:
            mod = __import__(pkg)
            meta[f"{pkg}_version"] = getattr(mod, "__version__", "unknown")
        except Exception as exc:
            meta[f"{pkg}_import_error"] = f"{type(exc).__name__}: {exc}"
    return meta


# --------------------------------------------------------------------------
# try/except result wrapper -- every benchmark *variant* goes through this so
# one missing/broken dependency never aborts the whole sweep.
# --------------------------------------------------------------------------


def safe_run(fn: Callable[[], dict[str, Any]], label: str = "") -> dict[str, Any]:
    """Run `fn`, catch anything, and always return a JSON-safe dict.

    On success: fn()'s return dict, with "status": "ok" merged in (fn should
    NOT set "status" itself). On failure: {"status": "error", "error": "...",
    "traceback": "..."}.
    """
    try:
        result = fn()
        if not isinstance(result, dict):
            result = {"value": result}
        result.setdefault("status", "ok")
        return result
    except Exception as exc:  # noqa: BLE001 -- intentional catch-all
        return {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
            "label": label,
        }


def try_import(module_path: str, attr: Optional[str] = None):
    """Import `module_path` (optionally pulling `attr` off it). Returns
    (obj_or_None, error_str_or_None). Never raises."""
    try:
        mod = __import__(module_path, fromlist=["_"])
        if attr is None:
            return mod, None
        return getattr(mod, attr), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------
# CUDA-event timing
# --------------------------------------------------------------------------


def cuda_time_fn(fn: Callable[[], Any], warmup: int, iters: int) -> dict[str, float]:
    """Time `fn()` (which should do exactly one unit of work and take no
    args) with CUDA events. Returns timing stats in microseconds.

    Caller is responsible for anything that must happen once (tensor setup)
    happening OUTSIDE `fn`. `fn` may mutate state in place (e.g. a recurrent
    state tensor) -- that's expected and fine for steady-state decode timing.
    """
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

    times_ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    times_us = [t * 1000.0 for t in times_ms]
    times_us.sort()
    n = len(times_us)
    mean_us = sum(times_us) / n
    p50 = times_us[n // 2]
    p10 = times_us[max(0, int(n * 0.1))]
    p90 = times_us[min(n - 1, int(n * 0.9))]
    return {
        "mean_us": mean_us,
        "min_us": times_us[0],
        "max_us": times_us[-1],
        "p10_us": p10,
        "p50_us": p50,
        "p90_us": p90,
        "n_iters": n,
    }


# --------------------------------------------------------------------------
# Output writers: JSON + Markdown
# --------------------------------------------------------------------------


def _split_out_paths(out: str) -> tuple[str, str]:
    if out.endswith(".json"):
        return out, out[: -len(".json")] + ".md"
    if out.endswith(".md"):
        return out[: -len(".md")] + ".json", out
    return out + ".json", out + ".md"


def dict_rows_to_markdown(rows: list[dict[str, Any]], columns: Optional[list[str]] = None) -> str:
    if not rows:
        return "_(no rows)_\n"
    cols = columns or list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in rows:
        cells = []
        for c in cols:
            v = row.get(c, "")
            if isinstance(v, float):
                v = f"{v:,.3f}"
            cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_results(out: str, payload: dict[str, Any], markdown_sections: list[str]) -> tuple[str, str]:
    json_path, md_path = _split_out_paths(out)
    os.makedirs(os.path.dirname(os.path.abspath(json_path)) or ".", exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    with open(md_path, "w") as f:
        f.write("\n\n".join(markdown_sections) + "\n")
    return json_path, md_path


# --------------------------------------------------------------------------
# Misc numeric helpers
# --------------------------------------------------------------------------


def dtype_nbytes(dtype_name: str) -> int:
    return {
        "float32": 4, "fp32": 4,
        "float16": 2, "fp16": 2,
        "bfloat16": 2, "bf16": 2,
        "float8_e4m3fn": 1, "fp8": 1, "fp8_e4m3": 1,
        "uint8": 1, "int8": 1,
    }[dtype_name.lower()]


def gdn_state_bytes(batch: int, dtype_name: str = "float32") -> int:
    return batch * GDN_NUM_V_HEADS * GDN_HEAD_K_DIM * GDN_HEAD_V_DIM * dtype_nbytes(dtype_name)


def gbps(nbytes: float, seconds: float) -> float:
    if seconds <= 0:
        return float("nan")
    return (nbytes / 1e9) / seconds


def tflops(flops: float, seconds: float) -> float:
    if seconds <= 0:
        return float("nan")
    return (flops / 1e12) / seconds


def gdn_chunk_algorithmic_flops(B: int, T: int, H: int, dk: int, dv: int, chunk_size: int = 64) -> float:
    """Analytical FLOP estimate for the chunked gated-delta-rule prefill
    algorithm, derived term-by-term from
    `engine/reference/modeling_qwen3_5.py::torch_chunk_gated_delta_rule`
    (same asymptotic work as `fla.ops.gated_delta_rule.chunk_gated_delta_rule`
    -- both implement the WY-representation / UT-transform chunked
    delta-rule; a fused Triton/CUDA kernel changes the constant via better
    fusion/reduced memory round-trips, not the FLOP count itself).

    Per chunk (chunk_size C), per head, the reference does:
      attn = k_beta @ key^T                     2*C^2*dk
      UT-transform (sequential lower-tri solve)  ~(2/3)*C^3   (dim-independent)
      value = attn @ v_beta                      2*C^2*dv
      k_cumdecay = attn @ (k_beta*decay)          2*C^2*dk
      attn_intra = q_i @ k_i^T                    2*C^2*dk
      v_prime = k_cumdecay @ state                2*C*dk*dv
      attn_inter = (q_i*decay) @ state             2*C*dk*dv
      out += attn_intra @ v_new                    2*C^2*dv
      state = state*decay + k_i^T @ v_new           2*C*dk*dv
    Multiplied by num_chunks = ceil(T/C), by H, by B.
    """
    C = chunk_size
    per_chunk_per_head = (
        2 * C * C * dk        # attn = k_beta @ key^T
        + (2.0 / 3.0) * C ** 3  # UT transform
        + 2 * C * C * dv       # value = attn @ v_beta
        + 2 * C * C * dk       # k_cumdecay
        + 2 * C * C * dk       # attn_intra
        + 2 * C * dk * dv      # v_prime
        + 2 * C * dk * dv      # attn_inter
        + 2 * C * C * dv       # out += attn_intra @ v_new
        + 2 * C * dk * dv      # state update
    )
    num_chunks = math.ceil(T / C)
    return B * H * num_chunks * per_chunk_per_head
