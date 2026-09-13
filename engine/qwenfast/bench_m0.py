#!/usr/bin/env python
"""Reference-model decode-step benchmark: batch 1 and batch 32.

Reports measured ms/step and tok/s next to the analytic memory-bandwidth
ceiling::

    step_ms(B) ~= W_ms + (2 * 144 MiB * B + kv_bytes(B, ctx)) / BW

with ``W_ms`` = weight bytes / HBM bandwidth.  The ``efficiency`` column
(ceiling / measured) is the number the optimised runtime must move; the
reference model is expected to land at a few percent because it is one eager
kernel launch per op.

Example::

    /home/venv_vllm/bin/python engine/qwenfast/bench_m0.py \
        --model /home/hf/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/* \
        --batch 1 32 --prompt-len 512 --steps 32
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

from qwenfast.model import HAS_FLA, Generator, QwenFastForCausalLM  # noqa: E402
from qwenfast.weights import QwenFastConfig, resolve_snapshot  # noqa: E402

MiB = 1024 ** 2
GiB = 1024 ** 3


# --------------------------------------------------------------------------- #
def physics(cfg: QwenFastConfig, batch: int, ctx: int, weight_bytes: int,
            bw_tbs: float, state_dtype_bytes: int = 4, kv_dtype_bytes: int = 2) -> Dict[str, float]:
    """Analytic per-decode-step HBM traffic."""
    bw = bw_tbs * 1e12  # bytes/s
    n_gdn = len(cfg.linear_layer_indices)
    ssm_per_seq = n_gdn * cfg.linear_num_value_heads * cfg.linear_key_head_dim * cfg.linear_value_head_dim * state_dtype_bytes
    kv_per_token = len(cfg.attention_layer_indices) * cfg.num_key_value_heads * cfg.head_dim * 2 * kv_dtype_bytes
    weight_ms = weight_bytes / bw * 1e3
    ssm_ms = 2 * ssm_per_seq * batch / bw * 1e3  # read + write
    kv_ms = kv_per_token * ctx * batch / bw * 1e3
    step_ms = weight_ms + ssm_ms + kv_ms
    return {
        "ssm_bytes_per_seq": ssm_per_seq,
        "kv_bytes_per_token": kv_per_token,
        "weight_ms": weight_ms,
        "ssm_ms": ssm_ms,
        "kv_ms": kv_ms,
        "step_ms": step_ms,
        "tok_s": batch / (step_ms / 1e3),
    }


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 32])
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--bandwidth-tbs", type=float, default=4.8, help="HBM BW, H200 = 4.8 TB/s")
    ap.add_argument("--state-dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    ap.add_argument("--no-fla", action="store_true")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    model_dir = resolve_snapshot(args.model)
    state_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.state_dtype]

    t0 = time.time()
    model = QwenFastForCausalLM.from_pretrained(model_dir, device=args.device, dtype=torch.bfloat16, verbose=True)
    load_s = time.time() - t0
    cfg = model.config
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    use_fla = HAS_FLA and not args.no_fla
    print(f"[bench] loaded {model_dir} in {load_s:.1f}s")
    print(f"[bench] resident weights {weight_bytes / GiB:.2f} GiB (bf16 runtime), fla={'ON' if use_fla else 'OFF'}")
    if device.type == "cuda":
        print(f"[bench] gpu {torch.cuda.get_device_name(device)}, "
              f"alloc {torch.cuda.memory_allocated(device) / GiB:.1f} GiB")

    gen = Generator(model, use_fla=use_fla)
    torch.manual_seed(0)
    results: List[Dict] = []

    print()
    header = f"{'B':>4} {'prefill ms':>11} {'decode ms':>10} {'tok/s':>9} {'ceil ms':>8} {'ceil tok/s':>11} {'eff %':>7} {'cache GiB':>10}"
    print(header)
    print("-" * len(header))

    for b in args.batch:
        prompts = [
            torch.randint(10, cfg.vocab_size - 10, (args.prompt_len,)).tolist() for _ in range(b)
        ]
        max_len = args.prompt_len + args.steps + args.warmup + 4
        cache = model.make_cache(b, max_len, device=device, state_dtype=state_dtype)
        cache_gib = cache.nbytes()["total"] / GiB

        sync(device)
        t0 = time.time()
        tok, _, pos = gen.prefill(prompts, cache)
        sync(device)
        prefill_ms = (time.time() - t0) * 1e3

        for _ in range(args.warmup):
            tok, _ = gen.decode_step(tok, pos, cache)
            pos = pos + 1
        sync(device)

        t0 = time.time()
        for _ in range(args.steps):
            tok, _ = gen.decode_step(tok, pos, cache)
            pos = pos + 1
        sync(device)
        decode_ms = (time.time() - t0) * 1e3 / args.steps

        ctx = args.prompt_len + args.warmup + args.steps
        ph = physics(
            cfg, b, ctx, weight_bytes, args.bandwidth_tbs,
            state_dtype_bytes=torch.finfo(state_dtype).bits // 8,
        )
        tok_s = b / (decode_ms / 1e3)
        eff = 100.0 * ph["step_ms"] / decode_ms
        print(f"{b:>4} {prefill_ms:>11.1f} {decode_ms:>10.2f} {tok_s:>9.1f} "
              f"{ph['step_ms']:>8.2f} {ph['tok_s']:>11.1f} {eff:>7.1f} {cache_gib:>10.2f}")
        results.append(
            {
                "batch": b,
                "prompt_len": args.prompt_len,
                "prefill_ms": prefill_ms,
                "prefill_tok_s": b * args.prompt_len / (prefill_ms / 1e3),
                "decode_ms_per_step": decode_ms,
                "decode_tok_s": tok_s,
                "ceiling_ms": ph["step_ms"],
                "ceiling_tok_s": ph["tok_s"],
                "efficiency_pct": eff,
                "cache_gib": cache_gib,
                "physics": ph,
            }
        )
        del cache
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print()
    print("ceiling = weights/BW + 2*SSM*B/BW + KV*ctx*B/BW")
    print(f"          weights {weight_bytes / GiB:.1f} GiB, SSM {results[0]['physics']['ssm_bytes_per_seq'] / MiB:.0f} MiB/seq "
          f"({args.state_dtype}), KV {results[0]['physics']['kv_bytes_per_token'] / 1024:.0f} KiB/token (bf16)")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(
                {
                    "model": model_dir,
                    "device": str(device),
                    "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                    "use_fla": use_fla,
                    "state_dtype": args.state_dtype,
                    "weight_bytes": weight_bytes,
                    "load_s": load_s,
                    "results": results,
                },
                f,
                indent=2,
            )
        print(f"[bench] wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
