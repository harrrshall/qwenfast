#!/usr/bin/env python3
"""Per-token cost math (INR and USD) for the benchmark harness.

Standalone CLI:
    python cost.py --tok-per-s 4200 --gpu h200_spot
    python cost.py --tok-per-s 4200 --rate-inr 250.0

Library use (e.g. from summarize.py):
    from cost import cost_per_million_tokens, annotate_bench_json
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Optional

# Hourly GPU rental rates in INR. Illustrative defaults; update them to your provider's prices.
GPU_RATES_INR_PER_HR: dict[str, float] = {
    "h200_spot": 188.73,
    "h200_on_demand": 378.27,
    "rtx_pro_6000_spot": 93.96,
    "rtx_pro_6000_on_demand": 179.01,
}

INR_PER_USD = 96.19


def cost_per_million_tokens(tok_per_s: float, inr_per_hr: float) -> float:
    """INR to produce one million tokens at a sustained rate of `tok_per_s`."""
    if tok_per_s <= 0:
        return float("inf")
    tokens_per_hr = tok_per_s * 3600.0
    return inr_per_hr / tokens_per_hr * 1_000_000.0


def inr_to_usd(inr: float) -> float:
    return inr / INR_PER_USD


def resolve_rate(gpu: Optional[str], rate_inr: Optional[float]) -> float:
    if rate_inr is not None:
        return rate_inr
    if gpu is not None:
        try:
            return GPU_RATES_INR_PER_HR[gpu]
        except KeyError as exc:
            options = ", ".join(sorted(GPU_RATES_INR_PER_HR))
            raise SystemExit(f"unknown --gpu '{gpu}'. options: {options}") from exc
    raise SystemExit("must pass either --gpu or --rate-inr")


def annotate_bench_json(
    data: dict[str, Any], gpu: Optional[str] = None, rate_inr: Optional[float] = None
) -> dict[str, Any]:
    """Add INR/USD-per-million-token cost fields to every level in a bench result.

    Mutates and returns `data`. Costs are computed from `output_tok_per_s`
    (cost to generate output tokens, the usual pricing basis) and from
    `total_tok_per_s` (input+output) for reference.
    """
    rate = resolve_rate(gpu, rate_inr)
    data.setdefault("metadata", {})["cost_basis"] = {
        "gpu": gpu,
        "rate_inr_per_hr": rate,
        "inr_per_usd": INR_PER_USD,
    }
    for level in data.get("levels", []):
        out_rate = level.get("output_tok_per_s", 0.0)
        total_rate = level.get("total_tok_per_s", 0.0)
        cost_out_inr = cost_per_million_tokens(out_rate, rate)
        cost_total_inr = cost_per_million_tokens(total_rate, rate)
        level["cost"] = {
            "inr_per_m_output_tokens": cost_out_inr,
            "usd_per_m_output_tokens": inr_to_usd(cost_out_inr),
            "inr_per_m_total_tokens": cost_total_inr,
            "usd_per_m_total_tokens": inr_to_usd(cost_total_inr),
        }
    return data


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute INR/USD per 1M tokens from a throughput.")
    p.add_argument("--tok-per-s", type=float, help="sustained output tokens/sec")
    p.add_argument("--gpu", choices=sorted(GPU_RATES_INR_PER_HR), help="named rate from the table")
    p.add_argument("--rate-inr", type=float, help="custom INR/hr rate, overrides --gpu")
    p.add_argument(
        "--annotate",
        metavar="PATH",
        help="instead of the single computation above, annotate a bench_serve.py "
        "result JSON in place with cost fields for every concurrency level",
    )
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()

    if args.annotate:
        with open(args.annotate) as f:
            data = json.load(f)
        annotate_bench_json(data, gpu=args.gpu, rate_inr=args.rate_inr)
        with open(args.annotate, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[cost] annotated {args.annotate}")
        return

    if args.tok_per_s is None:
        raise SystemExit("pass --tok-per-s (with --gpu or --rate-inr), or --annotate PATH")

    rate = resolve_rate(args.gpu, args.rate_inr)
    inr_per_m = cost_per_million_tokens(args.tok_per_s, rate)
    usd_per_m = inr_to_usd(inr_per_m)
    label = args.gpu or f"{rate:.2f} INR/hr"
    print(f"{args.tok_per_s:.1f} tok/s on {label}:")
    print(f"  INR {inr_per_m:,.2f} / 1M tokens")
    print(f"  USD {usd_per_m:,.4f} / 1M tokens")


if __name__ == "__main__":
    main()
