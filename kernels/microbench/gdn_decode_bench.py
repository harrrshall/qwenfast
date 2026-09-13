#!/usr/bin/env python3
"""Decode-step microbenchmark for the Gated-DeltaNet recurrent kernel used by
Qwen3.8-27B's 48 `linear_attention` layers.

For each batch size B in the sweep, times ONE decode step (T=1) of:
  (a) `fla.ops.gated_delta_rule.fused_recurrent_gated_delta_rule` (Triton), fp32 state
  (b) same, bf16 state (and fp16 state) -- tried, recorded if unsupported
  (c) cuLA (inclusionAI)'s equivalent, if importable -- best-effort, see README
  (d) a pure-torch (einsum/elementwise) reference -- small B only, for correctness
  (e) `causal_conv1d_update` (from the `causal_conv1d` package, else a torch
      fallback identical to the one in the HF reference impl) -- the depthwise
      conv step that also runs every decode step, ahead of the recurrence.

Shapes and the exact math (including the q/k GVA repeat_interleave x3, the
qk-L2-norm-in-kernel, and the log-space gate) are taken verbatim from
`engine/reference/modeling_qwen3_5.py::Qwen3_5GatedDeltaNet.forward` /
`torch_recurrent_gated_delta_rule`, restricted to the `seq_len == 1`,
cached-decode branch.

Non-interactive, CLI-driven, prints/writes JSON + Markdown to --out. Never
raises on a missing/broken optional dependency -- each variant is wrapped in
`common.safe_run` and records its error string instead.

This script does not run anything by itself when imported; only under
`python gdn_decode_bench.py ...` on a CUDA host. It is `py_compile`-safe with
no torch installed.

Example (remote, inside /home/venv_vllm):
    python gdn_decode_bench.py --out /home/qwenfast-results/gdn_decode.json
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Any, Optional

import common


def build_gdn_step_inputs(B: int, dtype, device, seed: int = 0):
    """Build one decode-step's worth of GDN inputs, matching
    Qwen3_5GatedDeltaNet.forward exactly (seq_len=1, cached-decode branch,
    post-conv, pre-repeat_interleave / post-repeat_interleave split so callers
    can choose which stage they want).

    Returns a dict of tensors on `device`:
      q_raw, k_raw: [B, 1, GDN_NUM_K_HEADS, GDN_HEAD_K_DIM]   (pre-GVA-expand)
      v:            [B, 1, GDN_NUM_V_HEADS, GDN_HEAD_V_DIM]
      q, k:         [B, 1, GDN_NUM_V_HEADS, GDN_HEAD_K_DIM]   (post-GVA-expand)
      g:            [B, 1, GDN_NUM_V_HEADS]  (log-space decay, pre-exp)
      beta:         [B, 1, GDN_NUM_V_HEADS]  (post-sigmoid, in (0,1))
      A_log, dt_bias, a_raw, b_raw: the raw per-head params, for reference
    """
    import torch

    g = torch.Generator(device="cpu").manual_seed(seed)

    def randn(*shape, gen=g):
        return torch.randn(*shape, generator=gen).to(device=device, dtype=torch.float32)

    q_raw = randn(B, 1, common.GDN_NUM_K_HEADS, common.GDN_HEAD_K_DIM)
    k_raw = randn(B, 1, common.GDN_NUM_K_HEADS, common.GDN_HEAD_K_DIM)
    v = randn(B, 1, common.GDN_NUM_V_HEADS, common.GDN_HEAD_V_DIM)

    A_log = torch.log(torch.empty(common.GDN_NUM_V_HEADS).uniform_(0.01, 16, generator=g)).to(device)
    dt_bias = torch.ones(common.GDN_NUM_V_HEADS, device=device)
    a_raw = randn(B, 1, common.GDN_NUM_V_HEADS)
    b_raw = randn(B, 1, common.GDN_NUM_V_HEADS)

    beta = b_raw.sigmoid()
    gate = -A_log.float().exp() * torch.nn.functional.softplus(a_raw.float() + dt_bias)

    rep = common.GDN_GVA_GROUP_SIZE
    q = q_raw.repeat_interleave(rep, dim=2)
    k = k_raw.repeat_interleave(rep, dim=2)

    return dict(q_raw=q_raw, k_raw=k_raw, v=v.to(dtype), q=q, k=k, g=gate, beta=beta,
                A_log=A_log, dt_bias=dt_bias, a_raw=a_raw, b_raw=b_raw)


def make_state(B: int, dtype, device, seed: int = 1):
    import torch

    g = torch.Generator(device="cpu").manual_seed(seed)
    state = torch.randn(B, common.GDN_NUM_V_HEADS, common.GDN_HEAD_K_DIM, common.GDN_HEAD_V_DIM,
                         generator=g).to(device=device, dtype=dtype)
    return state * 0.02  # keep magnitudes sane -- this is a perf/traffic bench, not a trained model


# --------------------------------------------------------------------------
# (a)/(b) fla fused_recurrent_gated_delta_rule
# --------------------------------------------------------------------------


def bench_fla_fused_recurrent(B: int, state_dtype_name: str, warmup: int, iters: int, device: str) -> dict[str, Any]:
    import torch
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    state_dtype = dtype_map[state_dtype_name]
    act_dtype = torch.float32 if state_dtype == torch.float32 else state_dtype

    inputs = build_gdn_step_inputs(B, act_dtype, device)
    state = make_state(B, state_dtype, device)

    with torch.no_grad():
        def step():
            nonlocal state
            out, new_state = fused_recurrent_gated_delta_rule(
                inputs["q"].to(act_dtype), inputs["k"].to(act_dtype), inputs["v"],
                g=inputs["g"], beta=inputs["beta"],
                initial_state=state, output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            state = new_state
            return out

        timing = common.cuda_time_fn(step, warmup, iters)

    state_bytes_rw = 2 * common.gdn_state_bytes(B, state_dtype_name)
    mean_s = timing["mean_us"] * 1e-6
    return {
        "variant": "fla_fused_recurrent",
        "state_dtype": state_dtype_name,
        "batch": B,
        **timing,
        "state_bytes_read_plus_write": state_bytes_rw,
        "achieved_gbps": common.gbps(state_bytes_rw, mean_s),
        "extrapolated_ms_48_layers": timing["mean_us"] * common.NUM_GDN_LAYERS / 1000.0,
    }


# --------------------------------------------------------------------------
# (c) cuLA (inclusionAI) -- best-effort. At the time of writing
# cuLA's public API exposes KDA (Kimi Delta Attention,
# `cula.kda.chunk_kda`) and Lightning Attention, not a confirmed drop-in
# `fused_recurrent_gated_delta_rule`. KDA is an algorithmic cousin of Gated
# DeltaNet (also delta-rule + gating) but with a different gate
# parameterization (per-channel decay vs GDN's per-head decay here), so a
# numerically-faithful drop-in is not guaranteed to exist. We try, in order,
# every import path that plausibly matches, and report exactly which one (if
# any) worked. If cuLA is not installed, or none of the entrypoints below
# exist / accept these shapes, this is recorded as status=unavailable rather
# than an error -- it's an optional, best-effort probe.
# --------------------------------------------------------------------------

_CULA_CANDIDATES = [
    ("cula.ops.gated_delta_rule", "fused_recurrent_gated_delta_rule"),
    ("cula.ops.gated_delta_rule", "fused_recurrent_gdn"),
    ("cula.gated_delta_rule", "fused_recurrent_gated_delta_rule"),
    ("cula.kda", "fused_recurrent_kda"),
    ("cula.ops.kda.decode.cute", "fused_recurrent_kda"),
]


def bench_cula(B: int, warmup: int, iters: int, device: str) -> dict[str, Any]:
    import torch

    cula_mod, imp_err = common.try_import("cula")
    if cula_mod is None:
        return {"status": "unavailable", "reason": f"import cula failed: {imp_err}"}

    inputs = build_gdn_step_inputs(B, torch.bfloat16, device)
    state = make_state(B, torch.float32, device)

    attempted = []
    for mod_path, attr in _CULA_CANDIDATES:
        fn, err = common.try_import(mod_path, attr)
        if fn is None:
            attempted.append({"path": f"{mod_path}.{attr}", "error": err})
            continue
        try:
            with torch.no_grad():
                out = fn(inputs["q"].to(torch.bfloat16), inputs["k"].to(torch.bfloat16), inputs["v"],
                          g=inputs["g"], beta=inputs["beta"], initial_state=state,
                          output_final_state=True, use_qk_l2norm_in_kernel=True)

                def step():
                    return fn(inputs["q"].to(torch.bfloat16), inputs["k"].to(torch.bfloat16), inputs["v"],
                               g=inputs["g"], beta=inputs["beta"], initial_state=state,
                               output_final_state=True, use_qk_l2norm_in_kernel=True)

                timing = common.cuda_time_fn(step, warmup, iters)
            state_bytes_rw = 2 * common.gdn_state_bytes(B, "float32")
            mean_s = timing["mean_us"] * 1e-6
            return {
                "status": "ok",
                "variant": "cula",
                "entrypoint": f"{mod_path}.{attr}",
                "caveat": "KDA/algorithmic-cousin best-effort call; verify numerics against (d) "
                          "before trusting -- gate parameterization differs from GDN's.",
                "batch": B,
                **timing,
                "state_bytes_read_plus_write": state_bytes_rw,
                "achieved_gbps": common.gbps(state_bytes_rw, mean_s),
                "extrapolated_ms_48_layers": timing["mean_us"] * common.NUM_GDN_LAYERS / 1000.0,
            }
        except Exception as exc:  # noqa: BLE001
            attempted.append({"path": f"{mod_path}.{attr}", "error": f"{type(exc).__name__}: {exc}"})
            continue

    return {"status": "unavailable", "reason": "cula imported but no known GDN/KDA entrypoint "
                                                "accepted these arguments", "attempted": attempted}


# --------------------------------------------------------------------------
# (d) pure-torch reference (small B only) -- exact port of
# torch_recurrent_gated_delta_rule's single-timestep body, for correctness.
# --------------------------------------------------------------------------


def torch_reference_step(inputs: dict, state: "Any"):
    import torch

    q = common_l2norm(inputs["q"].float())
    k = common_l2norm(inputs["k"].float())
    v = inputs["v"].float()
    scale = 1.0 / math.sqrt(common.GDN_HEAD_K_DIM)
    q = q * scale

    q_t = q[:, 0]       # [B, HV, K]
    k_t = k[:, 0]
    v_t = v[:, 0]
    g_t = inputs["g"][:, 0].exp().unsqueeze(-1).unsqueeze(-1)   # [B, HV, 1, 1]
    beta_t = inputs["beta"][:, 0].unsqueeze(-1)                 # [B, HV, 1]

    state = state.float() * g_t
    kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)            # [B, HV, V]
    delta = (v_t - kv_mem) * beta_t
    state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)     # [B, HV, K, V]
    out = (state * q_t.unsqueeze(-1)).sum(dim=-2)                # [B, HV, V]
    return out, state


def common_l2norm(x, dim: int = -1, eps: float = 1e-6):
    import torch
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def bench_torch_reference(B: int, warmup: int, iters: int, device: str) -> dict[str, Any]:
    import torch

    inputs = build_gdn_step_inputs(B, torch.float32, device)
    state = make_state(B, torch.float32, device)

    with torch.no_grad():
        def step():
            nonlocal state
            out, new_state = torch_reference_step(inputs, state)
            state = new_state
            return out

        timing = common.cuda_time_fn(step, warmup, iters)

    state_bytes_rw = 2 * common.gdn_state_bytes(B, "float32")
    mean_s = timing["mean_us"] * 1e-6
    return {
        "variant": "torch_reference",
        "batch": B,
        **timing,
        "state_bytes_read_plus_write": state_bytes_rw,
        "achieved_gbps": common.gbps(state_bytes_rw, mean_s),
        "extrapolated_ms_48_layers": timing["mean_us"] * common.NUM_GDN_LAYERS / 1000.0,
    }


def verify_correctness(B: int, device: str) -> dict[str, Any]:
    """Single (non-timed) forward of (a) fla fused_recurrent vs (d) torch
    reference, from identical inputs/state, reporting max-abs-diff."""
    import torch
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule

    inputs = build_gdn_step_inputs(B, torch.float32, device, seed=42)
    state0 = make_state(B, torch.float32, device, seed=43)

    with torch.no_grad():
        out_a, state_a = fused_recurrent_gated_delta_rule(
            inputs["q"], inputs["k"], inputs["v"],
            g=inputs["g"], beta=inputs["beta"],
            initial_state=state0.clone(), output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        out_d, state_d = torch_reference_step(inputs, state0.clone())

    out_diff = (out_a.float() - out_d.float()).abs().max().item()
    state_diff = (state_a.float() - state_d.float()).abs().max().item()
    return {
        "batch": B,
        "max_abs_diff_output": out_diff,
        "max_abs_diff_state": state_diff,
        "output_rel_ok": bool(out_diff < 1e-2),
    }


# --------------------------------------------------------------------------
# causal_conv1d_update step
# --------------------------------------------------------------------------


def _torch_causal_conv1d_update(hidden_states, conv_state, weight, bias=None, activation=None):
    """Verbatim fallback port of the HF reference's `causal_conv1d_update`
    (engine/reference/modeling_qwen3_5.py:200-216), used when the
    `causal_conv1d` CUDA package is not importable."""
    import torch
    import torch.nn.functional as F
    from transformers.activations import ACT2FN

    state_len = conv_state.shape[-1]
    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_states.shape[1])
    out = out[:, :, -hidden_states.shape[-1]:]
    if activation is not None:
        out = ACT2FN[activation](out)
    return out.to(hidden_states.dtype)


def bench_causal_conv1d_update(B: int, warmup: int, iters: int, device: str) -> dict[str, Any]:
    import torch

    dim = common.GDN_CONV_DIM
    width = common.GDN_CONV_KERNEL
    dtype = torch.bfloat16

    x = torch.randn(B, dim, device=device, dtype=dtype)
    conv_state = torch.randn(B, dim, width - 1, device=device, dtype=dtype)
    weight = torch.randn(dim, width, device=device, dtype=dtype) * 0.1
    bias = torch.zeros(dim, device=device, dtype=dtype)

    impl_name = "torch_fallback"
    fn, err = common.try_import("causal_conv1d", "causal_conv1d_update")
    if fn is not None:
        impl_name = "causal_conv1d_cuda"
        update_fn = fn
    else:
        update_fn = _torch_causal_conv1d_update

    with torch.no_grad():
        def step():
            return update_fn(x, conv_state, weight, bias, "silu")

        timing = common.cuda_time_fn(step, warmup, iters)

    # traffic: read x + conv_state + weight, write out + conv_state
    nbytes = 2  # bf16
    bytes_rw = (x.numel() + 2 * conv_state.numel() + weight.numel() + x.numel()) * nbytes
    mean_s = timing["mean_us"] * 1e-6
    return {
        "variant": "causal_conv1d_update",
        "impl": impl_name,
        "impl_available_error": err if fn is None else None,
        "batch": B,
        **timing,
        "approx_bytes_rw": bytes_rw,
        "achieved_gbps": common.gbps(bytes_rw, mean_s),
        "extrapolated_ms_48_layers": timing["mean_us"] * common.NUM_GDN_LAYERS / 1000.0,
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(parser, default_out="gdn_decode_bench_results")
    parser.add_argument("--torch-ref-max-batch", type=int, default=32,
                         help="Only run the pure-torch reference variant (d) for B <= this "
                              "(it's O(B) python-level tensor ops, gets slow/pointless at large B).")
    parser.add_argument("--verify-batch", type=int, default=8,
                         help="Batch size at which to run the (a) vs (d) correctness check.")
    parser.add_argument("--skip-cula", action="store_true")
    args = parser.parse_args(argv)

    batches = common.parse_batches(args.batches, common.DECODE_BATCH_SWEEP)

    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False -- this benchmark must run on a CUDA GPU.")
    except Exception as exc:  # noqa: BLE001
        payload = {"error": f"CUDA/torch unavailable: {exc}", "env": common.env_metadata(args.device)}
        json_path, md_path = common.write_results(args.out, payload, [f"# gdn_decode_bench\n\nFAILED: {exc}\n"])
        print(f"[gdn_decode_bench] FAILED (no CUDA): wrote {json_path}", file=sys.stderr)
        return 1

    device = args.device
    torch.cuda.set_device(device)

    results: dict[str, Any] = {"env": common.env_metadata(device), "args": vars(args), "per_batch": []}

    fla_available, fla_err = common.try_import("fla.ops.gated_delta_rule", "fused_recurrent_gated_delta_rule")
    if fla_available is None:
        print(f"[gdn_decode_bench] WARNING: fla not importable ({fla_err}); "
              "(a)/(b) will be recorded as errors for every batch.", file=sys.stderr)

    md_rows_fla_fp32, md_rows_fla_other, md_rows_cula, md_rows_conv, md_rows_ref = [], [], [], [], []

    for B in batches:
        print(f"[gdn_decode_bench] B={B} ...", file=sys.stderr)
        row: dict[str, Any] = {"batch": B}

        row["fla_fused_recurrent_fp32"] = common.safe_run(
            lambda B=B: bench_fla_fused_recurrent(B, "float32", args.warmup, args.iters, device),
            label=f"fla_fp32_B{B}")
        if row["fla_fused_recurrent_fp32"]["status"] == "ok":
            md_rows_fla_fp32.append({
                "B": B, "mean_us": row["fla_fused_recurrent_fp32"]["mean_us"],
                "p50_us": row["fla_fused_recurrent_fp32"]["p50_us"],
                "achieved_GB/s": row["fla_fused_recurrent_fp32"]["achieved_gbps"],
                "x48_layers_ms": row["fla_fused_recurrent_fp32"]["extrapolated_ms_48_layers"],
            })

        row["fla_fused_recurrent_bf16"] = common.safe_run(
            lambda B=B: bench_fla_fused_recurrent(B, "bfloat16", args.warmup, args.iters, device),
            label=f"fla_bf16_B{B}")
        row["fla_fused_recurrent_fp16"] = common.safe_run(
            lambda B=B: bench_fla_fused_recurrent(B, "float16", args.warmup, args.iters, device),
            label=f"fla_fp16_B{B}")
        for variant_key in ("fla_fused_recurrent_bf16", "fla_fused_recurrent_fp16"):
            r = row[variant_key]
            if r["status"] == "ok":
                md_rows_fla_other.append({
                    "B": B, "state_dtype": r["state_dtype"], "mean_us": r["mean_us"],
                    "achieved_GB/s": r["achieved_gbps"], "x48_layers_ms": r["extrapolated_ms_48_layers"],
                })

        if not args.skip_cula:
            row["cula"] = common.safe_run(lambda B=B: bench_cula(B, args.warmup, args.iters, device),
                                           label=f"cula_B{B}")
            if row["cula"].get("status") == "ok":
                md_rows_cula.append({
                    "B": B, "entrypoint": row["cula"]["entrypoint"], "mean_us": row["cula"]["mean_us"],
                    "achieved_GB/s": row["cula"]["achieved_gbps"],
                })

        if B <= args.torch_ref_max_batch:
            row["torch_reference"] = common.safe_run(
                lambda B=B: bench_torch_reference(B, args.warmup, args.iters, device), label=f"torchref_B{B}")
            if row["torch_reference"]["status"] == "ok":
                md_rows_ref.append({
                    "B": B, "mean_us": row["torch_reference"]["mean_us"],
                    "achieved_GB/s": row["torch_reference"]["achieved_gbps"],
                })

        row["causal_conv1d_update"] = common.safe_run(
            lambda B=B: bench_causal_conv1d_update(B, args.warmup, args.iters, device), label=f"conv_B{B}")
        if row["causal_conv1d_update"]["status"] == "ok":
            c = row["causal_conv1d_update"]
            md_rows_conv.append({
                "B": B, "impl": c["impl"], "mean_us": c["mean_us"],
                "achieved_GB/s": c["achieved_gbps"], "x48_layers_ms": c["extrapolated_ms_48_layers"],
            })

        # combined per-step-per-layer estimate (GDN math + conv), extrapolated to 48 layers
        fla_ok = row["fla_fused_recurrent_fp32"].get("status") == "ok"
        conv_ok = row["causal_conv1d_update"].get("status") == "ok"
        if fla_ok and conv_ok:
            combined_us = row["fla_fused_recurrent_fp32"]["mean_us"] + row["causal_conv1d_update"]["mean_us"]
            row["combined_gdn_layer_fp32"] = {
                "mean_us_per_layer": combined_us,
                "extrapolated_ms_48_layers": combined_us * common.NUM_GDN_LAYERS / 1000.0,
                "physics_model_state_only_ms_48_layers":
                    (2 * common.gdn_state_bytes(B, "float32") * common.NUM_GDN_LAYERS) / (args.gpu_bw_gbps * 1e9) * 1000.0,
            }
            phys = row["combined_gdn_layer_fp32"]["physics_model_state_only_ms_48_layers"]
            achieved = row["fla_fused_recurrent_fp32"]["extrapolated_ms_48_layers"]
            row["combined_gdn_layer_fp32"]["state_bw_efficiency_pct"] = (
                100.0 * phys / achieved if achieved > 0 else float("nan")
            )

        results["per_batch"].append(row)

    results["correctness_check"] = common.safe_run(
        lambda: verify_correctness(args.verify_batch, device), label="verify")

    md = [
        "# gdn_decode_bench results",
        f"GPU: {results['env'].get('gpu_name', '?')} | torch {results['env'].get('torch_version', '?')} "
        f"(cu{results['env'].get('torch_cuda_version', '?')}) | fla {results['env'].get('fla_version', 'MISSING: ' + str(results['env'].get('fla_import_error')))}",
        "## fla.fused_recurrent_gated_delta_rule, fp32 state (primary decode kernel)",
        common.dict_rows_to_markdown(md_rows_fla_fp32, ["B", "mean_us", "p50_us", "achieved_GB/s", "x48_layers_ms"]),
        "## fla.fused_recurrent_gated_delta_rule, bf16/fp16 state",
        common.dict_rows_to_markdown(md_rows_fla_other, ["B", "state_dtype", "mean_us", "achieved_GB/s", "x48_layers_ms"]),
        "## cuLA (best-effort)",
        common.dict_rows_to_markdown(md_rows_cula, ["B", "entrypoint", "mean_us", "achieved_GB/s"]),
        "## pure-torch reference (small B, for correctness only)",
        common.dict_rows_to_markdown(md_rows_ref, ["B", "mean_us", "achieved_GB/s"]),
        "## causal_conv1d_update",
        common.dict_rows_to_markdown(md_rows_conv, ["B", "impl", "mean_us", "achieved_GB/s", "x48_layers_ms"]),
        f"## Correctness: fla fused_recurrent vs pure-torch reference @ B={args.verify_batch}\n\n"
        f"```json\n{results['correctness_check']}\n```",
    ]

    json_path, md_path = common.write_results(args.out, results, md)
    print(f"[gdn_decode_bench] wrote {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
