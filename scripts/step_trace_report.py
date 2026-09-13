#!/usr/bin/env python3
"""Slice an engine step trace by concurrency level and print the loop's ledger.

``python -m qwenfast.runtime.serve --step-trace-out T.json``
writes one cumulative summary plus a **per-second series** over the server's
whole life; ``benchmarks/bench_serve.py`` writes ``started_unix`` /
``ended_unix`` for every concurrency level it measured.  Joining the two on
wall-clock time is the only way to say "at concurrency 256, the loop spent
X % of the wall clock inside ``Scheduler.step()``" -- a trace averaged over a
whole sweep blends six levels, a ramp and two idle gaps into one number that
means nothing.

Usage::

    python3 scripts/step_trace_report.py TRACE.json [--bench BENCH.json] [--json]

Without ``--bench`` it prints the whole-life summary.  With it, one row per
level, plus the level's measured ``out tok/s`` and ``nvidia-smi`` utilisation
next to the loop's own ``step``-busy fraction -- the two independent views of
the same idle.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

BUCKETS = ("step", "gap", "drain", "emit", "idle")


def slice_series(trace: Dict[str, Any], t0: float, t1: float) -> Dict[str, float]:
    """Sum the per-second buckets whose start falls inside ``[t0, t1)``.

    ``kinds`` (per-second) is accumulated too when the trace carries it; an
    older trace without it is handled by the same code with an empty breakdown.
    """
    base = trace.get("wall_start_unix", 0.0)
    acc = {k: 0.0 for k in BUCKETS}
    acc.update(steps=0.0, tokens=0.0, seconds=0.0)
    kinds: Dict[str, List[float]] = {}
    bucket_s = 1.0
    for row in trace.get("series", []):
        t = base + float(row.get("t_s", 0.0))
        if t < t0 or t >= t1:
            continue
        acc["seconds"] += bucket_s
        acc["steps"] += row.get("steps", 0)
        acc["tokens"] += row.get("tokens", 0)
        for k in BUCKETS:
            acc[k] += row.get(f"{k}_ms", 0.0)
        for name, v in (row.get("kinds") or {}).items():
            slot = kinds.setdefault(name, [0.0] * 5)
            for i in range(5):
                slot[i] += v[i]
    acc["kinds"] = kinds  # type: ignore[assignment]
    return acc


def format_kinds(kinds: Dict[str, List[float]], wall_s: float) -> str:
    """``kind steps ms/step tok/step prefill-tok/step decode-rows/step``."""
    if not kinds:
        return ""
    out = []
    for name, (steps, ms, tok, pre, rows) in sorted(kinds.items()):
        if not steps:
            continue
        out.append(
            f"      {name:<8} {int(steps):>6} steps  {ms / steps:>8.1f} ms/step  "
            f"{100.0 * ms / (wall_s * 1e3):>5.1f}% of wall  "
            f"{tok / steps:>7.1f} tok/step  "
            f"{pre / steps:>8.0f} prefill-tok/step  "
            f"{rows / steps:>6.1f} decode-rows/step"
        )
    return "\n".join(out)


def as_row(acc: Dict[str, float]) -> Dict[str, Any]:
    wall_ms = acc["seconds"] * 1e3 or 1.0
    other = max(wall_ms - sum(acc[k] for k in BUCKETS), 0.0)
    row: Dict[str, Any] = {
        "wall_s": round(acc["seconds"], 1),
        "steps": int(acc["steps"]),
        "tokens": int(acc["tokens"]),
        "out_tok_s": round(acc["tokens"] / acc["seconds"], 1) if acc["seconds"] else 0.0,
    }
    for k in BUCKETS:
        row[f"{k}_pct"] = round(100.0 * acc[k] / wall_ms, 2)
    row["other_pct"] = round(100.0 * other / wall_ms, 2)
    row["host_stall_pct"] = round(
        100.0 * (acc["gap"] + acc["drain"] + acc["emit"] + other) / wall_ms, 2
    )
    row["gap_ms_per_step"] = round(acc["gap"] / acc["steps"], 3) if acc["steps"] else 0.0
    row["step_ms_mean"] = round(acc["step"] / acc["steps"], 2) if acc["steps"] else 0.0
    return row


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace")
    p.add_argument("--bench", default=None, help="bench_serve JSON to slice by")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    trace = json.load(open(args.trace))
    meta = trace.get("meta", {})

    if not args.bench:
        s = trace
        print(f"# {args.trace}  mode={meta.get('mode')}  {meta.get('label', '')}")
        print(json.dumps({k: s[k] for k in
                          ("wall_s", "steps", "output_tokens", "output_tok_s",
                           "pct_of_wall", "step_busy_pct", "host_stall_pct",
                           "per_step_ms", "gap_ms_max", "by_kind")}, indent=1))
        return 0

    bench = json.load(open(args.bench))
    rows = []
    for lvl in bench.get("levels", bench.get("results", [])):
        t0, t1 = lvl.get("started_unix", 0.0), lvl.get("ended_unix", 0.0)
        if not (t0 and t1):
            print(f"level {lvl.get('concurrency')}: no timestamps "
                  f"(result written by an older bench_serve)", file=sys.stderr)
            continue
        acc = slice_series(trace, t0, t1)
        row = as_row(acc)
        row["_kinds"] = acc.get("kinds") or {}
        row["concurrency"] = lvl["concurrency"]
        row["bench_out_tok_s"] = round(lvl.get("output_tok_per_s", 0.0), 1)
        gpu = (lvl.get("gpu") or {}).get("utilization_gpu_pct") or {}
        row["nvsmi_util_pct"] = round(gpu.get("mean", 0.0), 1)
        rows.append(row)

    if args.json:
        print(json.dumps({"meta": meta, "levels": rows}, indent=1))
        return 0

    show_kinds = any(r.get("_kinds") for r in rows)

    print(f"# {args.trace}  mode={meta.get('mode')}  vs {args.bench}")
    hdr = ("conc", "wall s", "steps", "out tok/s", "bench", "nvsmi%",
           "step%", "gap%", "drain%", "emit%", "idle%", "other%",
           "stall%", "gap ms/step", "step ms")
    print(" | ".join(f"{h:>10}" for h in hdr))
    for r in rows:
        print(" | ".join(f"{v:>10}" for v in (
            r["concurrency"], r["wall_s"], r["steps"], r["out_tok_s"],
            r["bench_out_tok_s"], r["nvsmi_util_pct"],
            r["step_pct"], r["gap_pct"], r["drain_pct"], r["emit_pct"],
            r["idle_pct"], r["other_pct"], r["host_stall_pct"],
            r["gap_ms_per_step"], r["step_ms_mean"],
        )))
        if show_kinds and r.get("_kinds"):
            print(format_kinds(r["_kinds"], r["wall_s"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
