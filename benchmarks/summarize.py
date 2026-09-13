#!/usr/bin/env python3
"""Merge many bench_serve.py result JSONs into one markdown comparison table.

Usage:
    python summarize.py results/*.json --out RESULTS.md
    python summarize.py results/*.json --out RESULTS.md --gpu h200_spot
"""

from __future__ import annotations

import argparse
import glob
import json
from typing import Any, Optional

from cost import GPU_RATES_INR_PER_HR, cost_per_million_tokens, resolve_rate


def load_results(paths: list[str]) -> list[dict[str, Any]]:
    files: list[str] = []
    for pattern in paths:
        matched = sorted(glob.glob(pattern))
        files.extend(matched if matched else [pattern])
    out = []
    for path in files:
        with open(path) as f:
            data = json.load(f)
        data["_source_path"] = path
        out.append(data)
    return out


def build_rows(
    datasets: list[dict[str, Any]], rate_inr: Optional[float]
) -> list[dict[str, Any]]:
    rows = []
    for data in datasets:
        meta = data.get("metadata", {})
        tag = meta.get("tag", data.get("_source_path", "run"))
        for level in data.get("levels", []):
            row = {
                "tag": tag,
                "concurrency": level["concurrency"],
                "out_tok_s": level["output_tok_per_s"],
                "total_tok_s": level["total_tok_per_s"],
                "req_s": level["request_throughput_per_s"],
                "ttft_p50_ms": level["ttft_s"]["p50"] * 1000,
                "ttft_p99_ms": level["ttft_s"]["p99"] * 1000,
                "tpot_p50_ms": level["tpot_s"]["p50"] * 1000,
                "tpot_p99_ms": level["tpot_s"]["p99"] * 1000,
                "errors": level["num_errors"],
            }
            if rate_inr is not None:
                row["inr_per_m_out"] = cost_per_million_tokens(
                    level["output_tok_per_s"], rate_inr
                )
            elif "cost" in level:
                row["inr_per_m_out"] = level["cost"]["inr_per_m_output_tokens"]
            rows.append(row)
    rows.sort(key=lambda r: (r["tag"], r["concurrency"]))
    return rows


def render_table(rows: list[dict[str, Any]], with_cost: bool) -> str:
    headers = [
        "tag",
        "conc",
        "out tok/s",
        "total tok/s",
        "req/s",
        "TTFT p50",
        "TTFT p99",
        "TPOT p50",
        "TPOT p99",
        "errors",
    ]
    if with_cost:
        headers.append("INR/M out")

    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for r in rows:
        cells = [
            str(r["tag"]),
            str(r["concurrency"]),
            f"{r['out_tok_s']:.1f}",
            f"{r['total_tok_s']:.1f}",
            f"{r['req_s']:.2f}",
            f"{r['ttft_p50_ms']:.0f}ms",
            f"{r['ttft_p99_ms']:.0f}ms",
            f"{r['tpot_p50_ms']:.1f}ms",
            f"{r['tpot_p99_ms']:.1f}ms",
            str(r["errors"]),
        ]
        if with_cost:
            cells.append(f"{r.get('inr_per_m_out', 0):.2f}" if "inr_per_m_out" in r else "-")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge bench_serve.py results into RESULTS.md")
    p.add_argument("inputs", nargs="+", help="result JSON paths or glob patterns")
    p.add_argument("--out", default="benchmarks/RESULTS.md")
    p.add_argument(
        "--gpu", choices=sorted(GPU_RATES_INR_PER_HR), default=None, help="GPU key from cost.py's rate table"
    )
    p.add_argument("--rate-inr", type=float, default=None, help="custom INR/hr rate")
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    datasets = load_results(args.inputs)
    if not datasets:
        raise SystemExit("no input files matched")

    rate_inr = None
    if args.gpu or args.rate_inr:
        rate_inr = resolve_rate(args.gpu, args.rate_inr)

    rows = build_rows(datasets, rate_inr)
    table = render_table(rows, with_cost=rate_inr is not None or any("cost" in lvl for d in datasets for lvl in d.get("levels", [])))

    lines = [
        "# Benchmark results",
        "",
        f"Merged from {len(datasets)} result file(s).",
        "",
        table,
        "",
    ]
    with open(args.out, "w") as f:
        f.write("\n".join(lines))
    print(f"[summarize] wrote {args.out} ({len(rows)} rows from {len(datasets)} files)")


if __name__ == "__main__":
    main()
