#!/usr/bin/env python3
"""GEMM microbenchmark for every distinct weight-matrix shape in
Qwen3.8-27B's forward pass (see `common.GEMM_SHAPES`, derived from
`engine/reference/modeling_qwen3_5.py`: `Qwen3_5GatedDeltaNet`,
`Qwen3_5Attention`, `Qwen3_5MLP`, and the untied lm_head).

At decode batch sizes these GEMMs are memory-bound (read the whole weight
matrix once per forward; the physics model's "weights (FP8): ~27
GB -> 5.6ms, read once per step, independent of B" term) -- this script
measures µs and *effective TB/s of weight read* per shape, per batch size, so
that number can be checked against the ~4.8 TB/s H200 ceiling directly, and
extrapolates a whole-model per-decode-step total (48 GDN + 16 attn layers +
1 lm_head, against the physics model's "Weights (FP8): ~27 GB -> 5.6 ms"
term -- this script's own extrapolated total is the from-first-principles
cross-check of that number).

Variants:
  - bf16:        torch.matmul / F.linear                      (baseline)
  - fp8_pertensor: torch._scaled_mm, one scale for the whole tensor  (simple baseline, per spec)
  - fp8_block128:  vLLM's block-wise (128x128) scaled fp8 GEMM,
                    `vllm.model_executor.layers.quantization.utils.fp8_utils.w8a8_triton_block_scaled_mm`,
                    matching the FP8 checkpoint's actual quant scheme
                    (config-Qwen3.8-27B-FP8.json: fmt=e4m3, block quant)      (if importable)
  - fp8_cutlass:   vLLM's `vllm._custom_ops` cutlass fp8 GEMM, best-effort    (if importable)

All GEMM shapes have in_features/out_features that are multiples of 128, so
128x128 block quantization applies cleanly to every row in the table.

Every variant is wrapped in common.safe_run; a missing/incompatible vLLM
install degrades those cells to recorded errors, never a crash.

Example (remote):
    python gemm_bench.py --out /home/qwenfast-results/gemm.json
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Optional

import common

FP8_MAX = 448.0  # e4m3 max representable magnitude
BLOCK = 128


def _quantize_fp8_pertensor(x, dim_name: str):
    import torch
    amax = x.abs().amax().clamp(min=1e-8)
    scale = (amax / FP8_MAX).to(torch.float32)
    xq = (x.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return xq, scale.reshape(1)


def _quantize_fp8_block(w, block: int = BLOCK):
    """w: [out_f, in_f] -> (w_fp8 [out_f,in_f], scales [out_f//block, in_f//block])."""
    import torch
    out_f, in_f = w.shape
    ob, ib = out_f // block, in_f // block
    w_blocks = w.float().view(ob, block, ib, block)
    amax = w_blocks.abs().amax(dim=(1, 3)).clamp(min=1e-8)
    scale = amax / FP8_MAX
    w_fp8 = (w_blocks / scale[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).view(out_f, in_f)
    return w_fp8, scale


def _quantize_fp8_pertoken_group(x, block: int = BLOCK):
    """x: [M, in_f] -> (x_fp8 [M,in_f], scales [M, in_f//block])."""
    import torch
    M, in_f = x.shape
    groups = in_f // block
    x_g = x.float().view(M, groups, block)
    amax = x_g.abs().amax(dim=-1).clamp(min=1e-8)
    scale = amax / FP8_MAX
    x_fp8 = (x_g / scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).view(M, in_f)
    return x_fp8, scale


def bench_bf16(B: int, in_f: int, out_f: int, warmup: int, iters: int, device: str) -> dict[str, Any]:
    import torch
    x = torch.randn(B, in_f, device=device, dtype=torch.bfloat16)
    w = torch.randn(out_f, in_f, device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        def step():
            return torch.nn.functional.linear(x, w)

        timing = common.cuda_time_fn(step, warmup, iters)

    weight_bytes = in_f * out_f * 2
    mean_s = timing["mean_us"] * 1e-6
    return {"variant": "bf16", "B": B, "in_f": in_f, "out_f": out_f, **timing,
            "weight_bytes": weight_bytes, "effective_tbps": common.gbps(weight_bytes, mean_s) / 1000.0}


def bench_fp8_pertensor(B: int, in_f: int, out_f: int, warmup: int, iters: int, device: str) -> dict[str, Any]:
    import torch
    if not hasattr(torch, "_scaled_mm"):
        raise RuntimeError("this torch build has no torch._scaled_mm")

    x = torch.randn(B, in_f, device=device, dtype=torch.bfloat16)
    w_t = torch.randn(in_f, out_f, device=device, dtype=torch.bfloat16)  # [K, N] layout _scaled_mm wants

    x_fp8, x_scale = _quantize_fp8_pertensor(x, "act")
    w_fp8, w_scale = _quantize_fp8_pertensor(w_t, "weight")

    with torch.no_grad():
        def step():
            return torch._scaled_mm(x_fp8, w_fp8, scale_a=x_scale, scale_b=w_scale, out_dtype=torch.bfloat16)

        timing = common.cuda_time_fn(step, warmup, iters)

    weight_bytes = in_f * out_f * 1
    mean_s = timing["mean_us"] * 1e-6
    return {"variant": "fp8_pertensor_scaled_mm", "B": B, "in_f": in_f, "out_f": out_f, **timing,
            "weight_bytes": weight_bytes, "effective_tbps": common.gbps(weight_bytes, mean_s) / 1000.0}


def bench_fp8_block128(B: int, in_f: int, out_f: int, warmup: int, iters: int, device: str) -> dict[str, Any]:
    import torch

    mm_fn, mm_err = common.try_import(
        "vllm.model_executor.layers.quantization.utils.fp8_utils", "w8a8_triton_block_scaled_mm")
    if mm_fn is None:
        raise RuntimeError(f"vllm fp8_utils.w8a8_triton_block_scaled_mm not importable: {mm_err}")

    x = torch.randn(B, in_f, device=device, dtype=torch.bfloat16)
    w = torch.randn(out_f, in_f, device=device, dtype=torch.bfloat16)
    w_fp8, w_scale = _quantize_fp8_block(w, BLOCK)

    quant_fn, _ = common.try_import(
        "vllm.model_executor.layers.quantization.utils.fp8_utils", "per_token_group_quant_fp8")

    with torch.no_grad():
        def quantize_act():
            if quant_fn is not None:
                try:
                    return quant_fn(x, BLOCK)
                except Exception:
                    pass
            return _quantize_fp8_pertoken_group(x, BLOCK)

        x_fp8, x_scale = quantize_act()

        def step():
            return mm_fn(x_fp8, w_fp8, x_scale, w_scale, [BLOCK, BLOCK], output_dtype=torch.bfloat16)

        timing = common.cuda_time_fn(step, warmup, iters)

    weight_bytes = in_f * out_f * 1
    mean_s = timing["mean_us"] * 1e-6
    return {"variant": "fp8_block128_vllm_triton", "B": B, "in_f": in_f, "out_f": out_f, **timing,
            "weight_bytes": weight_bytes, "effective_tbps": common.gbps(weight_bytes, mean_s) / 1000.0,
            "act_quant_fn": "vllm.per_token_group_quant_fp8" if quant_fn is not None else "local_fallback"}


def bench_fp8_cutlass(B: int, in_f: int, out_f: int, warmup: int, iters: int, device: str) -> dict[str, Any]:
    """Best-effort: vLLM's `vllm._custom_ops` cutlass fp8 scaled_mm. The exact
    entrypoint name/signature has shifted across vLLM releases, so this tries
    a couple of plausible call shapes and reports whichever worked -- if none
    do, it's recorded as unavailable (not a hard error), matching the cuLA
    probe pattern in gdn_decode_bench.py."""
    import torch

    ops, ops_err = common.try_import("vllm._custom_ops")
    if ops is None:
        return {"status": "unavailable", "reason": f"import vllm._custom_ops failed: {ops_err}"}

    x = torch.randn(B, in_f, device=device, dtype=torch.bfloat16)
    w = torch.randn(out_f, in_f, device=device, dtype=torch.bfloat16)
    x_fp8, x_scale = _quantize_fp8_pertensor(x, "act")
    w_fp8, w_scale = _quantize_fp8_pertensor(w, "weight")
    w_fp8_t = w_fp8.t().contiguous()  # [in_f, out_f], cutlass_scaled_mm convention (B as [K,N])

    attempts = []
    if hasattr(ops, "cutlass_scaled_mm"):
        try:
            with torch.no_grad():
                def step():
                    return ops.cutlass_scaled_mm(x_fp8, w_fp8_t, x_scale, w_scale, torch.bfloat16)

                out = step()
                timing = common.cuda_time_fn(step, warmup, iters)
            weight_bytes = in_f * out_f * 1
            mean_s = timing["mean_us"] * 1e-6
            return {"status": "ok", "variant": "fp8_cutlass_vllm_custom_ops", "B": B, "in_f": in_f, "out_f": out_f,
                    **timing, "weight_bytes": weight_bytes,
                    "effective_tbps": common.gbps(weight_bytes, mean_s) / 1000.0}
        except Exception as exc:  # noqa: BLE001
            attempts.append(f"cutlass_scaled_mm(x,wT,scale_a,scale_b,out_dtype): {type(exc).__name__}: {exc}")

    return {"status": "unavailable", "reason": "vllm._custom_ops imported but cutlass_scaled_mm "
                                                "did not accept the attempted call signature(s)",
            "attempts": attempts}


VARIANT_FNS = {
    "bf16": bench_bf16,
    "fp8_pertensor": bench_fp8_pertensor,
    "fp8_block128": bench_fp8_block128,
}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(parser, default_out="gemm_bench_results")
    parser.add_argument("--include-alt-mlp", action="store_true",
                         help="Also benchmark the unfused gate_proj/up_proj rows (excluded from "
                              "whole-model totals by default to avoid double-counting vs. the fused row).")
    parser.add_argument("--skip-cutlass", action="store_true")
    args = parser.parse_args(argv)

    batches = common.parse_batches(args.batches, common.DECODE_BATCH_SWEEP)

    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False -- this benchmark must run on a CUDA GPU.")
    except Exception as exc:  # noqa: BLE001
        payload = {"error": f"CUDA/torch unavailable: {exc}", "env": common.env_metadata(args.device)}
        json_path, md_path = common.write_results(args.out, payload, [f"# gemm_bench\n\nFAILED: {exc}\n"])
        print(f"[gemm_bench] FAILED (no CUDA): wrote {json_path}", file=sys.stderr)
        return 1

    device = args.device
    torch.cuda.set_device(device)

    shapes = [s for s in common.GEMM_SHAPES if args.include_alt_mlp or s["group"] != "mlp_alt"]

    results: dict[str, Any] = {"env": common.env_metadata(device), "args": vars(args),
                                "shapes": shapes, "cells": [], "totals_per_step": []}

    for B in batches:
        for shape in shapes:
            name, in_f, out_f = shape["name"], shape["in_f"], shape["out_f"]
            print(f"[gemm_bench] B={B} shape={name} ({in_f}->{out_f}) ...", file=sys.stderr)
            for variant, fn in VARIANT_FNS.items():
                cell = common.safe_run(
                    lambda fn=fn, B=B, in_f=in_f, out_f=out_f: fn(B, in_f, out_f, args.warmup, args.iters, device),
                    label=f"{variant}_{name}_B{B}")
                cell.update({"shape_name": name, "requested_variant": variant, "batch": B,
                             "group": shape["group"], "count": shape["count"],
                             "include_in_total": shape["include_in_total"]})
                results["cells"].append(cell)

            if not args.skip_cutlass:
                cell = common.safe_run(
                    lambda B=B, in_f=in_f, out_f=out_f: bench_fp8_cutlass(B, in_f, out_f, args.warmup, args.iters, device),
                    label=f"fp8_cutlass_{name}_B{B}")
                cell.update({"shape_name": name, "requested_variant": "fp8_cutlass", "batch": B,
                             "group": shape["group"], "count": shape["count"],
                             "include_in_total": shape["include_in_total"]})
                results["cells"].append(cell)

    # Whole-model per-decode-step extrapolation, per (batch, variant).
    variants_for_total = list(VARIANT_FNS.keys()) + (["fp8_cutlass"] if not args.skip_cutlass else [])
    md_totals_rows = []
    for B in batches:
        for variant in variants_for_total:
            total_us = 0.0
            total_weight_bytes = 0.0
            ok_rows = 0
            missing = []
            for cell in results["cells"]:
                if cell["batch"] != B or cell["requested_variant"] != variant or not cell["include_in_total"]:
                    continue
                if cell.get("status") == "ok":
                    total_us += cell["mean_us"] * cell["count"]
                    total_weight_bytes += cell["weight_bytes"] * cell["count"]
                    ok_rows += 1
                else:
                    missing.append(cell["shape_name"])
            n_expected = sum(1 for s in shapes if s["include_in_total"])
            entry = {
                "batch": B, "variant": variant,
                "total_ms_per_decode_step": total_us / 1000.0,
                "total_weight_bytes": total_weight_bytes,
                "effective_tbps": common.gbps(total_weight_bytes, total_us * 1e-6) / 1000.0,
                "shapes_measured": ok_rows, "shapes_expected": n_expected,
                "shapes_missing": missing,
            }
            results["totals_per_step"].append(entry)
            if ok_rows == n_expected:
                md_totals_rows.append({
                    "B": B, "variant": variant, "total_ms": entry["total_ms_per_decode_step"],
                    "eff_TB/s": entry["effective_tbps"],
                })

    md_by_shape = []
    for shape in shapes:
        rows = [c for c in results["cells"] if c["shape_name"] == shape["name"] and c.get("status") == "ok"]
        if not rows:
            continue
        md_by_shape.append(f"### {shape['name']} ({shape['in_f']} -> {shape['out_f']}, x{shape['count']}, "
                            f"{shape['note']})")
        table_rows = [{"B": c["batch"], "variant": c["requested_variant"], "mean_us": c["mean_us"],
                        "eff_TB/s": c["effective_tbps"]} for c in rows]
        md_by_shape.append(common.dict_rows_to_markdown(table_rows, ["B", "variant", "mean_us", "eff_TB/s"]))

    md = [
        "# gemm_bench results",
        f"GPU: {results['env'].get('gpu_name', '?')} | torch {results['env'].get('torch_version', '?')} | "
        f"vllm {results['env'].get('vllm_version', 'MISSING: ' + str(results['env'].get('vllm_import_error')))}",
        "## Whole-model per-decode-step extrapolation (48 GDN + 16 attn + lm_head GEMMs, all shapes present)",
        common.dict_rows_to_markdown(md_totals_rows, ["B", "variant", "total_ms", "eff_TB/s"]),
        "## Per-shape detail",
        "\n".join(md_by_shape),
    ]

    json_path, md_path = common.write_results(args.out, results, md)
    print(f"[gemm_bench] wrote {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
