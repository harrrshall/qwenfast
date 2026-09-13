#!/usr/bin/env python3
"""
Eval gate for Qwen3.8-27B inference-engine / quantization / speculative-decoding
changes. Runs against any OpenAI-compatible chat-completions server.

Suites:
  - GSM8K (200 fixed problems, evals/data/gsm8k_200.jsonl): exact-match on the
    final numeric answer, extracted from a "#### <number>" marker.
  - IFEval-lite (50 prompts, evals/data/ifeval_50.jsonl): programmatic
    instruction-following checkers (see ifeval_checkers.py).

Usage (live run against a server):
    python run_eval.py --base-url http://HOST:8000/v1 --model Qwen/Qwen3.8-27B \\
        --api-key $API_KEY --concurrency 32 --out result.json --tag baseline-bf16 \\
        --enable-thinking false --max-tokens 512 --temperature 0

Usage (diff two saved runs, e.g. spec-decoding drift or quant drift, no server):
    python run_eval.py --compare-only result_a.json result_b.json

Usage (run + diff against a saved baseline in one shot):
    python run_eval.py --base-url ... --model ... --out result_new.json \\
        --reference result_baseline.json

See evals/README.md for pass/fail thresholds.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

sys.path.insert(0, str(HERE))
from ifeval_checkers import CHECKERS  # noqa: E402

try:
    import aiohttp
except ImportError:  # pragma: no cover
    aiohttp = None


THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
HASH_RE = re.compile(r"####\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)")
NUM_RE = re.compile(r"-?[0-9][0-9,]*(?:\.[0-9]+)?")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def strip_thinking(text: str) -> str:
    return THINK_RE.sub("", text or "").strip()


def normalize_number(s: str) -> str:
    s = s.strip().replace(",", "").replace("$", "")
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return repr(f)
    except (ValueError, OverflowError):
        return s


def extract_gsm8k_answer(text: str) -> str | None:
    body = strip_thinking(text)
    last = None
    for m in HASH_RE.finditer(body):
        last = m
    if last:
        return normalize_number(last.group(1))
    nums = NUM_RE.findall(body)
    if nums:
        return normalize_number(nums[-1])
    return None


def gsm8k_prompt(question: str) -> str:
    return (
        "Solve the following grade-school math problem. You may reason "
        "briefly, but the LAST line of your response must be exactly:\n"
        "#### <number>\n"
        "where <number> is the final numeric answer with no units, commas, "
        "or extra words.\n\n"
        f"Problem: {question}"
    )


# --------------------------------------------------------------------------
# HTTP call
# --------------------------------------------------------------------------

async def call_chat(session, args, messages: list[dict]):
    url = args.base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    payload = {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    if args.enable_thinking is not None:
        enable = args.enable_thinking == "true"
        payload["chat_template_kwargs"] = {"enable_thinking": enable}

    last_err = None
    for attempt in range(args.retries + 1):
        t0 = time.monotonic()
        try:
            timeout = aiohttp.ClientTimeout(total=args.timeout)
            async with session.post(url, json=payload, headers=headers, timeout=timeout) as resp:
                data = await resp.json(content_type=None)
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}: {str(data)[:300]}")
                latency = time.monotonic() - t0
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage") or {}
                completion_tokens = usage.get("completion_tokens")
                return content, completion_tokens, latency, None
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            if attempt < args.retries:
                await asyncio.sleep(0.5 * (attempt + 1))
    return "", None, None, last_err


# --------------------------------------------------------------------------
# suites
# --------------------------------------------------------------------------

def _summarize(results: list[dict], correct_key: str) -> dict:
    n = len(results)
    correct = sum(1 for r in results if r[correct_key])
    accuracy = (correct / n) if n else 0.0
    toks = [r["output_tokens"] for r in results if r["output_tokens"] is not None]
    mean_toks = (sum(toks) / len(toks)) if toks else None
    words = [len((r.get("raw_output") or "").split()) for r in results]
    mean_words = (sum(words) / len(words)) if words else None
    errors = sum(1 for r in results if r.get("error"))
    return {
        "n": n,
        "accuracy": accuracy,
        "mean_output_tokens": mean_toks,
        "mean_output_words": mean_words,
        "errors": errors,
        "items": sorted(results, key=lambda r: r["id"]),
    }


async def run_gsm8k(session, args, sem) -> dict:
    items = load_jsonl(DATA_DIR / "gsm8k_200.jsonl")
    if args.gsm8k_n:
        items = items[: args.gsm8k_n]
    results: list[dict] = []

    async def worker(item):
        async with sem:
            messages = [{"role": "user", "content": gsm8k_prompt(item["question"])}]
            content, ctoks, latency, err = await call_chat(session, args, messages)
            pred = extract_gsm8k_answer(content) if not err else None
            gold = normalize_number(item["answer"])
            correct = pred is not None and pred == gold
            results.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "gold": gold,
                    "pred": pred,
                    "correct": correct,
                    "output_tokens": ctoks,
                    "latency_s": latency,
                    "raw_output": content,
                    "error": err,
                }
            )

    await asyncio.gather(*(worker(it) for it in items))
    return _summarize(results, "correct")


async def run_ifeval(session, args, sem) -> dict:
    items = load_jsonl(DATA_DIR / "ifeval_50.jsonl")
    if args.ifeval_n:
        items = items[: args.ifeval_n]
    results: list[dict] = []

    async def worker(item):
        async with sem:
            messages = [{"role": "user", "content": item["prompt"]}]
            content, ctoks, latency, err = await call_chat(session, args, messages)
            body = strip_thinking(content) if not err else ""
            checker_fn = CHECKERS[item["checker"]]
            try:
                passed = bool(checker_fn(body, item.get("checker_args", {}))) if not err else False
            except Exception:  # noqa: BLE001
                passed = False
            results.append(
                {
                    "id": item["id"],
                    "prompt": item["prompt"],
                    "checker": item["checker"],
                    "passed": passed,
                    "output_tokens": ctoks,
                    "latency_s": latency,
                    "raw_output": content,
                    "error": err,
                }
            )

    await asyncio.gather(*(worker(it) for it in items))
    return _summarize(results, "passed")


# --------------------------------------------------------------------------
# agreement / diff mode
# --------------------------------------------------------------------------

def compute_agreement(run_a: dict, run_b: dict) -> dict:
    report: dict = {}
    total_common = 0
    total_exact = 0.0
    for task in ("gsm8k", "ifeval"):
        a_items = {it["id"]: it for it in run_a.get(task, {}).get("items", [])}
        b_items = {it["id"]: it for it in run_b.get(task, {}).get("items", [])}
        common = sorted(set(a_items) & set(b_items))
        n = len(common)
        exact_text = 0
        outcome_match = 0
        for i in common:
            ta = (a_items[i].get("raw_output") or "").strip()
            tb = (b_items[i].get("raw_output") or "").strip()
            if ta == tb:
                exact_text += 1
            if task == "gsm8k":
                if a_items[i].get("pred") == b_items[i].get("pred"):
                    outcome_match += 1
            else:
                if a_items[i].get("passed") == b_items[i].get("passed"):
                    outcome_match += 1
        outcome_key = "answer_agreement_pct" if task == "gsm8k" else "pass_agreement_pct"
        report[task] = {
            "n_common": n,
            "exact_text_agreement_pct": (100.0 * exact_text / n) if n else None,
            outcome_key: (100.0 * outcome_match / n) if n else None,
        }
        total_common += n
        total_exact += exact_text
    report["overall_exact_text_agreement_pct"] = (
        (100.0 * total_exact / total_common) if total_common else None
    )
    return report


def print_agreement(report: dict, label_a: str, label_b: str) -> None:
    print(f"\n=== Agreement: {label_a}  vs  {label_b} ===")
    for task in ("gsm8k", "ifeval"):
        t = report.get(task, {})
        n = t.get("n_common", 0)
        exact = t.get("exact_text_agreement_pct")
        outcome_key = "answer_agreement_pct" if task == "gsm8k" else "pass_agreement_pct"
        outcome = t.get(outcome_key)
        print(
            f"  {task:8s} n={n:4d}  exact_text_agreement={fmt_pct(exact)}  "
            f"{outcome_key}={fmt_pct(outcome)}"
        )
    print(f"  overall exact_text_agreement = {fmt_pct(report.get('overall_exact_text_agreement_pct'))}")


def fmt_pct(x) -> str:
    return f"{x:.1f}%" if x is not None else "n/a"


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def print_summary(result: dict) -> None:
    meta = result["meta"]
    print(f"\n=== eval gate result — tag={meta.get('tag') or '(none)'} ===")
    print(f"  model={meta['model']}  base_url={meta['base_url']}  "
          f"enable_thinking={meta['enable_thinking']}  temperature={meta['temperature']}")
    for task in ("gsm8k", "ifeval"):
        t = result[task]
        acc = t["accuracy"] * 100
        mt = t["mean_output_tokens"]
        mt_s = f"{mt:.1f}" if mt is not None else f"~{t['mean_output_words']:.1f} words (no usage field)"
        print(
            f"  {task:8s} n={t['n']:4d}  accuracy={acc:5.1f}%  "
            f"mean_output_tokens={mt_s}  errors={t['errors']}"
        )
    print(f"  wall_time_s={meta['wall_time_s']:.1f}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", help="OpenAI-compatible base URL, e.g. http://host:8000/v1")
    p.add_argument("--model", help="model name as registered on the server")
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--out", default="result.json")
    p.add_argument("--tag", default="", help="free-form label stored in the result file (e.g. 'fp8-baseline')")
    p.add_argument("--enable-thinking", choices=["true", "false"], default=None,
                    help="sets chat_template_kwargs.enable_thinking; omit to use the server default")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=float, default=120.0, help="per-request timeout, seconds")
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--gsm8k-n", type=int, default=None, help="use only the first N GSM8K items (smoke tests)")
    p.add_argument("--ifeval-n", type=int, default=None, help="use only the first N IFEval items (smoke tests)")
    p.add_argument("--reference", default=None,
                    help="path to a prior result.json; after the run, print per-item agreement vs it")
    p.add_argument("--compare-only", nargs=2, metavar=("RUN_A", "RUN_B"), default=None,
                    help="skip the server: just diff two existing result.json files and print agreement")
    return p


async def run(args) -> dict:
    if aiohttp is None:
        print("ERROR: aiohttp is required. pip install -r evals/requirements.txt", file=sys.stderr)
        sys.exit(1)

    sem = asyncio.Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    t0 = time.time()
    async with aiohttp.ClientSession(connector=connector) as session:
        gsm8k_res, ifeval_res = await asyncio.gather(
            run_gsm8k(session, args, sem),
            run_ifeval(session, args, sem),
        )
    wall = time.time() - t0

    result = {
        "meta": {
            "tag": args.tag,
            "base_url": args.base_url,
            "model": args.model,
            "enable_thinking": args.enable_thinking,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "concurrency": args.concurrency,
            "wall_time_s": wall,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "gsm8k": gsm8k_res,
        "ifeval": ifeval_res,
    }
    return result


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.compare_only:
        path_a, path_b = args.compare_only
        with open(path_a) as f:
            run_a = json.load(f)
        with open(path_b) as f:
            run_b = json.load(f)
        report = compute_agreement(run_a, run_b)
        print_agreement(report, path_a, path_b)
        return

    if not args.base_url or not args.model:
        parser.error("--base-url and --model are required unless using --compare-only")

    result = asyncio.run(run(args))
    print_summary(result)

    if args.reference:
        with open(args.reference) as f:
            reference = json.load(f)
        report = compute_agreement(reference, result)
        print_agreement(report, args.reference, args.out)
        result["agreement_vs_reference"] = {"reference_path": args.reference, **report}

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
