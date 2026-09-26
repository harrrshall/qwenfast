"""summarize a harbor terminal-bench job run with the qwenfast agent.

    python report.py <jobs_dir>/<job_name> [more job dirs...]

prints a table and writes report.json into each job dir. per trial it reads harbor's result.json
(reward, phase timings, errors), the router decision (agent/qfa-route.json) and pi's event log
(agent/pi.txt: turns, tool calls, tokens).
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def _secs(span: dict | None) -> float | None:
    if not span or not span.get("started_at") or not span.get("finished_at"):
        return None
    a = datetime.fromisoformat(span["started_at"].replace("Z", "+00:00"))
    b = datetime.fromisoformat(span["finished_at"].replace("Z", "+00:00"))
    return (b - a).total_seconds()


def _pi_stats(path: Path) -> dict:
    turns = tools = inp = out = cache = 0
    if path.exists():
        for line in path.read_text(errors="replace").splitlines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = ev.get("type")
            if t == "turn_end":
                turns += 1
            elif t == "tool_execution_start":
                tools += 1
            elif t == "message_end" and (ev.get("message") or {}).get("role") == "assistant":
                u = ev["message"].get("usage") or {}
                inp += u.get("input", 0)
                out += u.get("output", 0)
                cache += u.get("cacheRead", 0)
    return {"turns": turns, "tool_calls": tools, "input_tokens": inp, "output_tokens": out, "cache_tokens": cache}


def trial_rows(job: Path) -> list[dict]:
    rows = []
    for rf in sorted(job.glob("*/result.json")):
        d = json.loads(rf.read_text())
        route = {}
        rp = rf.parent / "agent" / "qfa-route.json"
        if rp.exists():
            route = json.loads(rp.read_text())
        reward = ((d.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        exc = (d.get("exception_info") or {}).get("exception_type")
        agent_s = _secs(d.get("agent_execution"))
        pi = _pi_stats(rf.parent / "agent" / "pi.txt")
        rows.append({
            "task": d.get("task_name"),
            "trial": d.get("trial_name"),
            "reward": reward,
            "passed": reward is not None and reward >= 1.0,
            "error": exc,
            "tier": route.get("tier"),
            "route_source": route.get("source"),
            "agent_s": agent_s,
            "setup_s": _secs(d.get("agent_setup")),
            "verify_s": _secs(d.get("verifier")),
            "total_s": _secs({"started_at": d.get("started_at"), "finished_at": d.get("finished_at")}),
            **pi,
            "decode_tok_s": (pi["output_tokens"] / agent_s) if agent_s and pi["output_tokens"] else None,
        })
    return rows


def _p(xs: list[float], q: float) -> float | None:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 1)


def summarize(rows: list[dict]) -> dict:
    graded = [r for r in rows if r["reward"] is not None]
    infra = [r for r in rows if r["reward"] is None]
    by_task: dict[str, list[bool]] = defaultdict(list)
    for r in graded:
        by_task[r["task"]].append(r["passed"])
    k = max((len(v) for v in by_task.values()), default=0)
    tiers: dict[str, list[bool]] = defaultdict(list)
    for r in graded:
        tiers[r["tier"] or "?"].append(r["passed"])
    ok = [r for r in graded if r["passed"]]
    return {
        "trials": len(rows),
        "graded": len(graded),
        "passed": len(ok),
        "pass_rate": round(len(ok) / len(graded), 4) if graded else None,
        "pass_rate_incl_infra_errors": round(len(ok) / len(rows), 4) if rows else None,
        "tasks": len(by_task),
        "attempts_per_task": k,
        "pass_at_k": round(sum(any(v) for v in by_task.values()) / len(by_task), 4) if by_task and k > 1 else None,
        "pass_hat_k": round(sum(all(v) for v in by_task.values()) / len(by_task), 4) if by_task and k > 1 else None,
        "infra_errors": {e: sum(1 for r in infra if r["error"] == e) for e in {r["error"] for r in infra}},
        "agent_errors": {e: sum(1 for r in graded if r["error"] == e) for e in {r["error"] for r in graded if r["error"]}},
        "tier_mix": {t: len(v) for t, v in tiers.items()},
        "pass_rate_by_tier": {t: round(sum(v) / len(v), 3) for t, v in tiers.items()},
        "agent_s": {"p50": _p([r["agent_s"] for r in graded], 0.5), "p90": _p([r["agent_s"] for r in graded], 0.9)},
        "agent_s_when_passed": {"p50": _p([r["agent_s"] for r in ok], 0.5)},
        "turns_p50": _p([r["turns"] for r in graded], 0.5),
        "tool_calls_p50": _p([r["tool_calls"] for r in graded], 0.5),
        "output_tokens_total": sum(r["output_tokens"] for r in rows),
        "input_tokens_total": sum(r["input_tokens"] for r in rows),
        "decode_tok_s_p50": _p([r["decode_tok_s"] for r in graded], 0.5),
        "setup_s_p50": _p([r["setup_s"] for r in rows], 0.5),
    }


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for arg in sys.argv[1:]:
        job = Path(arg)
        rows = trial_rows(job)
        summary = summarize(rows)
        (job / "report.json").write_text(json.dumps({"summary": summary, "trials": rows}, indent=2) + "\n")
        print(f"== {job.name}")
        for k, v in summary.items():
            print(f"  {k:28s} {v}")
        width = max((len(r["task"] or "") for r in rows), default=10)
        for r in sorted(rows, key=lambda r: (r["task"] or "")):
            status = "PASS" if r["passed"] else ("fail" if r["reward"] is not None else f"ERR {r['error']}")
            agent_s = f"{r['agent_s']:.0f}s" if r["agent_s"] else "-"
            print(f"    {r['task']:{width}s}  {status:28s} tier={r['tier'] or '-':6s} {agent_s:>6s} turns={r['turns']:3d} out={r['output_tokens']}")


if __name__ == "__main__":
    main()
