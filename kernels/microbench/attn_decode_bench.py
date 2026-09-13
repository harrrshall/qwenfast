#!/usr/bin/env python3
"""Decode-step microbenchmark for the full-attention layers (16 of 64,
`layer_types == "full_attention"`, every 4th layer) using FlashInfer's
paged-KV batch-decode kernel.

Shapes (engine/reference/config-Qwen3.8-27B-FP8.json, text_config):
  num_attention_heads (q) = 24, num_key_value_heads (kv) = 4, head_dim = 256.
  (Note: q_proj in the real model also emits a fused output-gate, doubling
  its width to 12288 -- that GEMM is benchmarked separately in gemm_bench.py.
  This script only exercises the attention kernel itself: q @ paged-KV.)

Sweeps: batch B x {context 2048, context 8192} x {KV dtype fp16, fp8 e4m3}.
Reports µs/step, achieved KV-read GB/s, and the x16-layer extrapolated ms per
model decode step (contrast against gdn_decode_bench's x48-layer GDN number
and the physics model's "KV read: 32 KiB x ctx x B" term -- that 32
KiB/token figure assumes fp8 KV; this script measures both fp8 and fp16 so
the "4x smaller KV, near-free concurrency" claim in the plan can be checked
against the *kernel*, not just the raw byte count).

Wrapped per-variant in common.safe_run -- a missing/incompatible flashinfer
install degrades to recorded errors, never a crash.

Example (remote):
    python attn_decode_bench.py --out /home/qwenfast-results/attn_decode.json
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Optional

import common

NUM_QO_HEADS = common.ATTN_NUM_Q_HEADS       # 24
NUM_KV_HEADS = common.ATTN_NUM_KV_HEADS      # 4
HEAD_DIM = common.ATTN_HEAD_DIM              # 256


def _build_paged_kv(B: int, context: int, page_size: int, kv_dtype, device):
    import torch

    pages_per_seq = (context + page_size - 1) // page_size
    total_pages = B * pages_per_seq
    last_page_len = context - (pages_per_seq - 1) * page_size
    if last_page_len == 0:
        last_page_len = page_size

    kv_indptr = torch.arange(0, B + 1, dtype=torch.int32, device=device) * pages_per_seq
    kv_indices = torch.arange(0, total_pages, dtype=torch.int32, device=device)
    kv_last_page_len = torch.full((B,), last_page_len, dtype=torch.int32, device=device)

    if kv_dtype == getattr(torch, "float8_e4m3fn", None):
        # fp8 storage: fill with small-magnitude values representable in e4m3
        base = torch.randn(total_pages, 2, page_size, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=torch.float32)
        kv_cache = (base * 0.1).to(kv_dtype)
    else:
        kv_cache = torch.randn(total_pages, 2, page_size, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=kv_dtype)

    return kv_indptr, kv_indices, kv_last_page_len, kv_cache, pages_per_seq, total_pages


def bench_flashinfer_decode(B: int, context: int, kv_dtype_name: str, page_size: int,
                             warmup: int, iters: int, device: str,
                             use_tensor_cores: bool = True) -> dict[str, Any]:
    import torch
    import flashinfer

    dtype_map = {"float16": torch.float16, "float8_e4m3fn": getattr(torch, "float8_e4m3fn", None)}
    kv_dtype = dtype_map[kv_dtype_name]
    if kv_dtype is None:
        raise RuntimeError(f"torch has no {kv_dtype_name} dtype (too old a torch build)")

    kv_indptr, kv_indices, kv_last_page_len, kv_cache, pages_per_seq, total_pages = _build_paged_kv(
        B, context, page_size, kv_dtype, device)

    q = torch.randn(B, NUM_QO_HEADS, HEAD_DIM, device=device, dtype=torch.float16)

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    # use_tensor_cores=True is required at our GQA shape (NUM_QO_HEADS/NUM_KV_HEADS
    # = 24/4 = group_size 6, HEAD_DIM=256): the library's own default
    # (use_tensor_cores=False) raises "batch_decode.cu:63: Unsupported
    # group_size: 6" on H200/flashinfer 0.6.16.post3, in both f16 and e4m3
    # KV. See
    # engine/qwenfast/attn/flashinfer_attn.py's module docstring for the
    # full finding and the other two independent workarounds benchmarked
    # there (prefill-kernel-as-decode, FlashAttention-3 paged KV).
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, "NHD", use_tensor_cores=use_tensor_cores)
    wrapper.plan(
        kv_indptr, kv_indices, kv_last_page_len,
        NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM, page_size,
        pos_encoding_mode="NONE",
        q_data_type=torch.float16,
        kv_data_type=kv_dtype,
    )

    with torch.no_grad():
        def step():
            return wrapper.run(q, kv_cache)

        timing = common.cuda_time_fn(step, warmup, iters)

    kv_dtype_bytes = common.dtype_nbytes(kv_dtype_name)
    kv_read_bytes = B * context * 2 * NUM_KV_HEADS * HEAD_DIM * kv_dtype_bytes  # K+V read, whole context
    mean_s = timing["mean_us"] * 1e-6
    return {
        "variant": "flashinfer_batch_decode",
        "batch": B, "context": context, "kv_dtype": kv_dtype_name, "page_size": page_size,
        "pages_per_seq": pages_per_seq, "total_pages": total_pages,
        **timing,
        "kv_read_bytes": kv_read_bytes,
        "achieved_gbps": common.gbps(kv_read_bytes, mean_s),
        "extrapolated_ms_16_layers": timing["mean_us"] * common.NUM_ATTN_LAYERS / 1000.0,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(parser, default_out="attn_decode_bench_results")
    parser.add_argument("--contexts", type=str, default=None,
                         help=f"Comma-separated context sweep (default: {','.join(str(c) for c in common.ATTN_CONTEXT_SWEEP)}).")
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--kv-dtypes", type=str, default="float16,float8_e4m3fn")
    parser.add_argument("--use-tensor-cores", action=argparse.BooleanOptionalAction, default=True,
                         help="Route through flashinfer's tensor-core decode kernel (default: on). "
                              "Required at our GQA group_size=6 (24 q-heads / 4 kv-heads) -- the "
                              "non-tensor-core kernel raises 'Unsupported group_size: 6' on H200/"
                              "flashinfer 0.6.16.post3. Pass --no-use-tensor-cores to reproduce/"
                              "confirm that failure.")
    args = parser.parse_args(argv)

    batches = common.parse_batches(args.batches, common.DECODE_BATCH_SWEEP)
    contexts = [int(x) for x in args.contexts.split(",")] if args.contexts else common.ATTN_CONTEXT_SWEEP
    kv_dtypes = [x.strip() for x in args.kv_dtypes.split(",") if x.strip()]

    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False -- this benchmark must run on a CUDA GPU.")
    except Exception as exc:  # noqa: BLE001
        payload = {"error": f"CUDA/torch unavailable: {exc}", "env": common.env_metadata(args.device)}
        json_path, md_path = common.write_results(args.out, payload, [f"# attn_decode_bench\n\nFAILED: {exc}\n"])
        print(f"[attn_decode_bench] FAILED (no CUDA): wrote {json_path}", file=sys.stderr)
        return 1

    device = args.device
    torch.cuda.set_device(device)

    fi_ok, fi_err = common.try_import("flashinfer")
    if fi_ok is None:
        print(f"[attn_decode_bench] WARNING: flashinfer not importable ({fi_err}); "
              "every cell will be recorded as an error.", file=sys.stderr)

    results: dict[str, Any] = {"env": common.env_metadata(device), "args": vars(args), "cells": []}
    md_rows: dict[str, list] = {c: [] for c in contexts}

    for context in contexts:
        for B in batches:
            for kv_dtype_name in kv_dtypes:
                print(f"[attn_decode_bench] context={context} B={B} kv_dtype={kv_dtype_name} ...", file=sys.stderr)
                cell = common.safe_run(
                    lambda B=B, context=context, kv_dtype_name=kv_dtype_name: bench_flashinfer_decode(
                        B, context, kv_dtype_name, args.page_size, args.warmup, args.iters, device,
                        use_tensor_cores=args.use_tensor_cores),
                    label=f"attn_decode_ctx{context}_B{B}_{kv_dtype_name}")
                cell["batch"], cell["context"], cell["kv_dtype_requested"] = B, context, kv_dtype_name
                results["cells"].append(cell)
                if cell.get("status") == "ok":
                    md_rows[context].append({
                        "B": B, "kv_dtype": kv_dtype_name, "mean_us": cell["mean_us"],
                        "achieved_GB/s": cell["achieved_gbps"], "x16_layers_ms": cell["extrapolated_ms_16_layers"],
                    })

    md = ["# attn_decode_bench results",
          f"GPU: {results['env'].get('gpu_name', '?')} | flashinfer "
          f"{results['env'].get('flashinfer_version', 'MISSING: ' + str(results['env'].get('flashinfer_import_error')))} | "
          f"page_size={args.page_size}, q_heads={NUM_QO_HEADS}, kv_heads={NUM_KV_HEADS}, head_dim={HEAD_DIM}"]
    for context in contexts:
        md.append(f"## context = {context}")
        md.append(common.dict_rows_to_markdown(md_rows[context], ["B", "kv_dtype", "mean_us", "achieved_GB/s", "x16_layers_ms"]))

    json_path, md_path = common.write_results(args.out, results, md)
    print(f"[attn_decode_bench] wrote {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
