#!/usr/bin/env python
"""Print (and diff) the step attribution `--step-profile-out` writes.

`scripts/step_trace_report.py` is the same idea one level up: it
splits the engine *loop*'s wall clock and joins it to a sweep.  This one splits
the *step*, the bucket the loop ledger cannot open, into its host phases and
its device timeline.

    python step_profile_report.py before.json [after.json]
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, Optional

PHASES = ("admit", "collect", "build", "plan", "launch", "sync", "harvest",
          "book", "other")


def load(path: str) -> Dict[str, Any]:
    with open(path) as fh:
        return json.load(fh)


def fmt(v: float) -> str:
    return f"{v:8.2f}"


def table(name: str, a: Dict[str, Any], b: Optional[Dict[str, Any]] = None) -> None:
    print(f"\n### {name}")
    ha, ga = a["host_ms"], a["gpu_ms"]
    hb = b["host_ms"] if b else None
    gb = b["gpu_ms"] if b else None
    hdr = "| phase | before ms | " + ("after ms | delta ms |" if b else "")
    print(f"steps {a['steps']}" + (f" -> {b['steps']}" if b else "")
          + f", rows/step {a['rows_per_step']}"
          + (f" -> {b['rows_per_step']}" if b else "")
          + f", chunk tok/step {a['chunk_tokens_per_step']}"
          + (f" -> {b['chunk_tokens_per_step']}" if b else ""))
    print(hdr if b else "| phase | ms |")
    print("|---|---|---|---|" if b else "|---|---|")
    for p in PHASES:
        row = f"| {p} | {fmt(ha.get(p, 0.0))} |"
        if hb is not None:
            row += f" {fmt(hb.get(p, 0.0))} | {fmt(hb.get(p, 0.0) - ha.get(p, 0.0))} |"
        print(row)
    for p in ("commit_wait", "commit_book"):
        row = f"| *{p}* | {fmt(ha.get(p, 0.0))} |"
        if hb is not None:
            row += f" {fmt(hb.get(p, 0.0))} | {fmt(hb.get(p, 0.0) - ha.get(p, 0.0))} |"
        print(row)
    row = f"| **host step total** | {fmt(ha['total'])} |"
    if hb is not None:
        row += f" {fmt(hb['total'])} | {fmt(hb['total'] - ha['total'])} |"
    print(row)
    print()
    print("| device | before ms | " + ("after ms |" if b else ""))
    print("|---|---|---|" if b else "|---|---|")
    for k, label in (("idle_pre", "idle before launch"),
                     ("kernel", "kernel"),
                     ("tail", "idle after launch"),
                     ("span", "step span")):
        row = f"| {label} | {fmt(ga.get(k, 0.0))} |"
        if gb is not None:
            row += f" {fmt(gb.get(k, 0.0))} |"
        print(row)
    row = f"| **GPU busy %** | {a['gpu_busy_pct']:8.1f} |"
    if b:
        row += f" {b['gpu_busy_pct']:8.1f} |"
    print(row)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    a = load(sys.argv[1])
    b = load(sys.argv[2]) if len(sys.argv) > 2 else None
    print(f"# step profile: {sys.argv[1]}" + (f"  vs  {sys.argv[2]}" if b else ""))
    print(f"meta: {a.get('meta')}")
    if b:
        print(f"meta: {b.get('meta')}")
    print(f"overall GPU busy: {a['overall_gpu_busy_pct']} %"
          + (f" -> {b['overall_gpu_busy_pct']} %" if b else ""))
    kinds = sorted(set(a["by_kind"]) | (set(b["by_kind"]) if b else set()))
    for k in kinds:
        ka = a["by_kind"].get(k)
        kb = b["by_kind"].get(k) if b else None
        if ka is None:
            ka, kb = kb, None
            table(f"{k} (after only)", ka)
            continue
        table(k, ka, kb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
