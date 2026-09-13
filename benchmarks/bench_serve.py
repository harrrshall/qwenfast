#!/usr/bin/env python3
"""Async load generator for OpenAI-compatible LLM inference servers.

Runs on the GPU host against a local (or remote) server exposing the
OpenAI `/v1/completions` or `/v1/chat/completions` API (vLLM, SGLang, or a
custom engine). Sweeps a list of concurrency levels, fires a fixed number
of requests per level with a bounded number of requests in flight, and
records per-request latency breakdowns (TTFT, inter-token latency, E2E).

Usage:
    python bench_serve.py --base-url http://localhost:8000/v1 \\
        --model Qwen/Qwen3.8-27B --concurrency 1,16,64,128 \\
        --input-len 2000 --output-len 500 --out results/run1.json --tag h200-vllm

See README.md for methodology and the standard sweep command.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import string
import sys
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Optional

import aiohttp

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a hard dep in practice
    np = None  # type: ignore[assignment]


# --------------------------------------------------------------------------
# Prompt datasets
# --------------------------------------------------------------------------

_SHAREGPT_LIKE_SEEDS = [
    "Explain how transformer attention works and why it scales quadratically "
    "with sequence length. Include a discussion of KV-cache memory.",
    "Write a Python function that merges two sorted linked lists in place, "
    "then explain its time and space complexity.",
    "Summarize the plot of a mystery novel where the detective turns out to "
    "be the culprit, without spoiling the twist too early.",
    "You are a helpful assistant. A user asks for a week-long vegetarian meal "
    "plan with a shopping list organized by grocery aisle.",
    "Compare REST and gRPC for a microservices architecture handling high "
    "throughput internal traffic. Give a recommendation with trade-offs.",
    "Debug this stack trace and suggest a fix: IndexError: list index out of "
    "range in a function that processes batches of variable length.",
    "Draft a polite but firm email declining a vendor's price increase and "
    "proposing a renegotiation meeting next quarter.",
    "Describe the physics of why a spinning top precesses, using Newtonian "
    "mechanics and angular momentum conservation.",
]


def _tokenizer_from_name(name: Optional[str]):
    if not name:
        return None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    except Exception as exc:  # pragma: no cover - depends on env
        print(f"[warn] could not load tokenizer '{name}': {exc}", file=sys.stderr)
        return None


def _random_words(n_words: int, rng: random.Random) -> str:
    words = []
    for _ in range(n_words):
        length = rng.randint(3, 9)
        words.append("".join(rng.choices(string.ascii_lowercase, k=length)))
    return " ".join(words)


def make_prompt(
    dataset: str,
    input_len: int,
    tokenizer,
    rng: random.Random,
    index: int,
) -> str:
    """Build a single prompt of approximately `input_len` tokens."""
    if dataset == "random":
        if tokenizer is not None:
            vocab_size = tokenizer.vocab_size
            token_ids = [rng.randrange(vocab_size) for _ in range(input_len)]
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
            # Decoding can drop/merge tokens (special tokens, byte-fallback,
            # etc.); pad or trim by word count as a cheap correction so the
            # prompt lands close to the requested token budget.
            return text if text.strip() else _random_words(input_len, rng)
        # Fallback: synthetic word salad, ~1 token/word is a reasonable
        # approximation for a subword tokenizer on random ASCII words.
        return _random_words(input_len, rng)

    if dataset == "sharegpt-like":
        seed = _SHAREGPT_LIKE_SEEDS[index % len(_SHAREGPT_LIKE_SEEDS)]
        if tokenizer is not None:
            ids = tokenizer.encode(seed)
            if len(ids) >= input_len:
                return tokenizer.decode(ids[:input_len], skip_special_tokens=True)
            # Repeat with light variation to pad up to the target length.
            out_ids = list(ids)
            filler = tokenizer.encode(
                " Additionally, consider the following context repeated for "
                "padding purposes: " + seed
            )
            while len(out_ids) < input_len:
                out_ids.extend(filler)
            return tokenizer.decode(out_ids[:input_len], skip_special_tokens=True)
        words = seed.split()
        if len(words) >= input_len:
            return " ".join(words[:input_len])
        pad = words * (input_len // max(len(words), 1) + 1)
        return " ".join(pad[:input_len])

    raise ValueError(f"unknown dataset: {dataset}")


# --------------------------------------------------------------------------
# Per-request result
# --------------------------------------------------------------------------


@dataclass
class RequestResult:
    success: bool
    ttft_s: Optional[float] = None
    e2e_s: float = 0.0
    inter_token_latencies_s: list[float] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: Optional[str] = None


@dataclass
class LevelResult:
    concurrency: int
    num_prompts: int
    duration_s: float
    results: list[RequestResult]
    metrics_before: dict[str, float]
    metrics_after: dict[str, float]
    gpu_samples: list[dict[str, float]]
    # Wall-clock bounds of the *measured* window, so an
    # engine-side per-second trace (`serve --step-trace-out`) can be sliced to
    # exactly this level. Without them a trace over a server's whole lifetime
    # blends six concurrency levels and a ramp into one average.
    started_unix: float = 0.0
    ended_unix: float = 0.0


# --------------------------------------------------------------------------
# HTTP client
# --------------------------------------------------------------------------


def build_payload(
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    stream: bool,
    extra_body: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "stream": stream,
        "ignore_eos": True,
        "temperature": 0.0,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    if endpoint == "chat":
        payload["messages"] = [{"role": "user", "content": prompt}]
    else:
        payload["prompt"] = prompt
    payload.update(extra_body)
    return payload


async def send_one_request(
    session: aiohttp.ClientSession,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    endpoint: str,
    stream: bool,
) -> RequestResult:
    start = time.perf_counter()
    ttft: Optional[float] = None
    itls: list[float] = []
    completion_tokens = 0
    prompt_tokens = 0
    usage: Optional[dict[str, Any]] = None
    last_token_time = start

    try:
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status != 200:
                body = await resp.text()
                return RequestResult(
                    success=False,
                    e2e_s=time.perf_counter() - start,
                    error=f"HTTP {resp.status}: {body[:300]}",
                )

            if not stream:
                # Some servers mislabel the content-type on non-streaming
                # JSON responses; don't fail the request over that.
                data = await resp.json(content_type=None)
                end = time.perf_counter()
                usage = data.get("usage") or {}
                choice = (data.get("choices") or [{}])[0]
                text = (
                    choice.get("message", {}).get("content")
                    if endpoint == "chat"
                    else choice.get("text")
                ) or ""
                completion_tokens = int(
                    usage.get("completion_tokens") or max(len(text.split()), 1)
                )
                prompt_tokens = int(usage.get("prompt_tokens") or 0)
                return RequestResult(
                    success=True,
                    ttft_s=end - start,
                    e2e_s=end - start,
                    inter_token_latencies_s=[],
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )

            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:") :].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                if chunk.get("usage"):
                    usage = chunk["usage"]

                choices = chunk.get("choices") or []
                token_text = ""
                if choices:
                    choice = choices[0]
                    if endpoint == "chat":
                        token_text = (choice.get("delta") or {}).get("content") or ""
                    else:
                        token_text = choice.get("text") or ""

                if token_text:
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - start
                    else:
                        itls.append(now - last_token_time)
                    last_token_time = now
                    completion_tokens += 1

            end = time.perf_counter()
            if usage:
                completion_tokens = int(usage.get("completion_tokens") or completion_tokens)
                prompt_tokens = int(usage.get("prompt_tokens") or 0)

            if ttft is None:
                return RequestResult(
                    success=False,
                    e2e_s=end - start,
                    error="no tokens received before stream end",
                )

            return RequestResult(
                success=True,
                ttft_s=ttft,
                e2e_s=end - start,
                inter_token_latencies_s=itls,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
    except Exception as exc:  # noqa: BLE001 - report all failures as results
        return RequestResult(
            success=False, e2e_s=time.perf_counter() - start, error=str(exc)
        )


# --------------------------------------------------------------------------
# Prometheus /metrics scraping (best-effort)
# --------------------------------------------------------------------------

_INTERESTING_METRIC_SUBSTRINGS = (
    "gpu_cache_usage_perc",
    "num_requests_waiting",
    "num_requests_running",
    "num_requests_swapped",
)


def _parse_prometheus_text(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # format: metric_name{labels} value  OR  metric_name value
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        name_and_labels, value_str = parts
        name = name_and_labels.split("{", 1)[0]
        if not any(s in name for s in _INTERESTING_METRIC_SUBSTRINGS):
            continue
        try:
            value = float(value_str)
        except ValueError:
            continue
        # Keep the last-seen value per metric name (collapses label sets).
        out[name] = value
    return out


async def scrape_metrics(session: aiohttp.ClientSession, metrics_url: str) -> dict[str, float]:
    try:
        async with session.get(metrics_url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
            if resp.status != 200:
                return {}
            text = await resp.text()
            return _parse_prometheus_text(text)
    except Exception:
        return {}


def derive_metrics_url(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    root = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
    return root.rstrip("/") + "/metrics"


def derive_health_url(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    root = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
    return root.rstrip("/") + "/health"


# --------------------------------------------------------------------------
# Server liveness
# --------------------------------------------------------------------------
# A server process that is *up* is not the same thing as a server that can
# *serve*. In M3 run r_07772647 the engine thread died of a CUDA OOM four
# seconds after the server reported ready; uvicorn kept answering, and this
# script spent an hour issuing requests that could never complete (~Rs 190 of
# H200 time) before anyone noticed. `/health` now reports engine liveness
# (qwenfast/server/app.py), and the sweep checks it before it starts and
# between every level, so the failure costs one request timeout instead of a
# whole window.


class EngineDead(RuntimeError):
    """The server reported an unhealthy engine. Terminal -- do not retry."""


async def check_health(
    session: aiohttp.ClientSession, health_url: Optional[str], *, timeout_s: float = 5.0
) -> Optional[str]:
    """`None` if healthy (or unknown), else the server's failure detail.

    A connection error is deliberately *not* treated as "dead": a transient
    blip should not abort a 40-minute sweep. A 5xx with a body, on the other
    hand, is the server telling us its engine is gone.
    """
    if not health_url:
        return None
    try:
        async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
            if resp.status == 200:
                return None
            body = (await resp.text())[:300]
            return f"HTTP {resp.status}: {body}"
    except Exception:
        return None


async def wait_until_ready(
    session: aiohttp.ClientSession,
    models_url: str,
    health_url: Optional[str],
    *,
    timeout_s: float,
) -> None:
    """Block until `/v1/models` answers **and** the engine reports healthy.

    Raises `EngineDead` on an unhealthy engine and `TimeoutError` if the server
    never came up -- either way the caller aborts instead of starting a sweep
    against a server that cannot serve.
    """
    deadline = time.monotonic() + timeout_s
    last = "no response"
    while time.monotonic() < deadline:
        dead = await check_health(session, health_url)
        if dead is not None:
            raise EngineDead(f"server is unhealthy before the sweep started: {dead}")
        try:
            async with session.get(
                models_url, timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    return
                last = f"HTTP {resp.status}"
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
        await asyncio.sleep(2.0)
    raise TimeoutError(f"server not ready after {timeout_s:.0f}s ({last})")


# --------------------------------------------------------------------------
# nvidia-smi sampler
# --------------------------------------------------------------------------


class GpuSampler:
    """Samples `nvidia-smi` in a background thread at a fixed interval."""

    def __init__(self, interval_s: float = 1.0) -> None:
        self.interval_s = interval_s
        self._samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _run(self) -> None:
        import subprocess

        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=True,
                )
                line = out.stdout.strip().splitlines()[0]
                util_s, mem_s, power_s = (p.strip() for p in line.split(","))
                self._samples.append(
                    {
                        "utilization_gpu_pct": float(util_s),
                        "memory_used_mib": float(mem_s),
                        "power_draw_w": float(power_s),
                    }
                )
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> list[dict[str, float]]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 5)
        return self._samples


# --------------------------------------------------------------------------
# Stats helpers
# --------------------------------------------------------------------------


def _pctile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if np is not None:
        return float(np.percentile(values, p))
    values_sorted = sorted(values)
    k = (len(values_sorted) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(values_sorted) - 1)
    if f == c:
        return values_sorted[f]
    return values_sorted[f] + (values_sorted[c] - values_sorted[f]) * (k - f)


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def summarize_level(level: LevelResult) -> dict[str, Any]:
    ok = [r for r in level.results if r.success]
    failed = [r for r in level.results if not r.success]

    ttfts = [r.ttft_s for r in ok if r.ttft_s is not None]
    # Per-request time-per-output-token (excludes the first token).
    tpots = [
        (r.e2e_s - r.ttft_s) / (r.completion_tokens - 1)
        for r in ok
        if r.ttft_s is not None and r.completion_tokens > 1
    ]
    all_itls = [itl for r in ok for itl in r.inter_token_latencies_s]

    total_output_tokens = sum(r.completion_tokens for r in ok)
    total_prompt_tokens = sum(r.prompt_tokens for r in ok)
    duration = level.duration_s or 1e-9

    gpu_summary: dict[str, Any] = {}
    if level.gpu_samples:
        for key in ("utilization_gpu_pct", "memory_used_mib", "power_draw_w"):
            vals = [s[key] for s in level.gpu_samples if key in s]
            if vals:
                gpu_summary[key] = {"mean": _mean(vals), "max": max(vals)}

    return {
        "concurrency": level.concurrency,
        "started_unix": level.started_unix,
        "ended_unix": level.ended_unix,
        "num_prompts": level.num_prompts,
        "num_success": len(ok),
        "num_errors": len(failed),
        "errors_sample": [r.error for r in failed[:5]],
        "duration_s": duration,
        "output_tok_per_s": total_output_tokens / duration,
        "total_tok_per_s": (total_output_tokens + total_prompt_tokens) / duration,
        "request_throughput_per_s": len(ok) / duration,
        "ttft_s": {
            "mean": _mean(ttfts),
            "p50": _pctile(ttfts, 50),
            "p90": _pctile(ttfts, 90),
            "p99": _pctile(ttfts, 99),
        },
        "tpot_s": {
            "mean": _mean(tpots),
            "p50": _pctile(tpots, 50),
            "p99": _pctile(tpots, 99),
        },
        "itl_s": {
            "mean": _mean(all_itls),
            "p50": _pctile(all_itls, 50),
            "p99": _pctile(all_itls, 99),
        },
        "e2e_s": {
            "mean": _mean([r.e2e_s for r in ok]),
            "p50": _pctile([r.e2e_s for r in ok], 50),
            "p99": _pctile([r.e2e_s for r in ok], 99),
        },
        "prompt_tokens_total": total_prompt_tokens,
        "completion_tokens_total": total_output_tokens,
        "metrics_before": level.metrics_before,
        "metrics_after": level.metrics_after,
        "gpu": gpu_summary,
    }


# --------------------------------------------------------------------------
# Level runner
# --------------------------------------------------------------------------


async def run_level(
    session: aiohttp.ClientSession,
    url: str,
    headers: dict[str, str],
    endpoint: str,
    stream: bool,
    model: str,
    concurrency: int,
    num_prompts: int,
    warmup: int,
    output_len: int,
    extra_body: dict[str, Any],
    prompts: list[str],
    metrics_url: Optional[str],
    use_gpu_sampler: bool,
    health_url: Optional[str] = None,
) -> LevelResult:
    sem = asyncio.Semaphore(concurrency)

    async def bound_request(prompt: str) -> RequestResult:
        async with sem:
            payload = build_payload(endpoint, model, prompt, output_len, stream, extra_body)
            return await send_one_request(session, url, headers, payload, endpoint, stream)

    # Warmup (not measured, errors *not* ignored any more: `warmup` requests
    # that all fail is the cheapest possible signal that this level cannot run,
    # and paying the per-request timeout `warmup` times per level -- six levels
    # deep -- is precisely how one dead engine cost an hour).
    if warmup > 0:
        warm_prompts = (prompts * ((warmup // len(prompts)) + 1))[:warmup]
        warm = await asyncio.gather(*(bound_request(p) for p in warm_prompts))
        if warm and not any(r.success for r in warm):
            dead = await check_health(session, health_url)
            errs = "; ".join(sorted({r.error or "?" for r in warm}))[:300]
            raise EngineDead(
                f"every warmup request at concurrency={concurrency} failed"
                + (f" and the server reports {dead}" if dead else "")
                + f" -- {errs}"
            )

    metrics_before = (
        await scrape_metrics(session, metrics_url) if metrics_url else {}
    )

    sampler = GpuSampler() if use_gpu_sampler else None
    if sampler:
        sampler.start()

    run_prompts = (prompts * ((num_prompts // len(prompts)) + 1))[:num_prompts]
    started_unix = time.time()
    start = time.perf_counter()
    results = await asyncio.gather(*(bound_request(p) for p in run_prompts))
    duration = time.perf_counter() - start
    ended_unix = time.time()

    gpu_samples = sampler.stop() if sampler else []
    metrics_after = (
        await scrape_metrics(session, metrics_url) if metrics_url else {}
    )

    return LevelResult(
        concurrency=concurrency,
        num_prompts=num_prompts,
        duration_s=duration,
        results=list(results),
        metrics_before=metrics_before,
        metrics_after=metrics_after,
        gpu_samples=gpu_samples,
        started_unix=started_unix,
        ended_unix=ended_unix,
    )


# --------------------------------------------------------------------------
# Markdown report
# --------------------------------------------------------------------------


def render_markdown_table(tag: str, level_summaries: list[dict[str, Any]]) -> str:
    headers = [
        "conc",
        "ok/err",
        "out tok/s",
        "total tok/s",
        "req/s",
        "TTFT p50",
        "TTFT p99",
        "TPOT p50",
        "TPOT p99",
        "ITL p99",
    ]
    lines = [
        f"### {tag}",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for s in level_summaries:
        row = [
            str(s["concurrency"]),
            f"{s['num_success']}/{s['num_errors']}",
            f"{s['output_tok_per_s']:.1f}",
            f"{s['total_tok_per_s']:.1f}",
            f"{s['request_throughput_per_s']:.2f}",
            f"{s['ttft_s']['p50'] * 1000:.0f}ms",
            f"{s['ttft_s']['p99'] * 1000:.0f}ms",
            f"{s['tpot_s']['p50'] * 1000:.1f}ms",
            f"{s['tpot_s']['p99'] * 1000:.1f}ms",
            f"{s['itl_s']['p99'] * 1000:.1f}ms",
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Async load generator for OpenAI-compatible LLM servers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base-url", required=True, help="e.g. http://localhost:8000/v1")
    p.add_argument("--model", required=True, help="model name sent in the request body")
    p.add_argument("--api-key", default="EMPTY", help="Bearer token, if the server checks one")
    p.add_argument(
        "--concurrency",
        default="1,16,64,128,256,512",
        help="comma-separated list of concurrency levels to sweep",
    )
    p.add_argument(
        "--num-prompts",
        type=int,
        default=None,
        help="requests per level; default max(concurrency*8, 64) per level",
    )
    p.add_argument("--input-len", type=int, default=2000, help="target prompt length in tokens")
    p.add_argument("--output-len", type=int, default=500, help="max/target output tokens")
    p.add_argument(
        "--dataset",
        choices=["random", "sharegpt-like"],
        default="random",
        help="prompt source",
    )
    p.add_argument(
        "--tokenizer",
        default=None,
        help="HF tokenizer name/path used to build/measure prompts (optional)",
    )
    p.add_argument("--stream", dest="stream", action="store_true", default=True)
    p.add_argument("--no-stream", dest="stream", action="store_false")
    p.add_argument("--warmup", type=int, default=3, help="warmup requests per level, discarded")
    p.add_argument("--out", required=True, help="path to write the result JSON")
    p.add_argument("--tag", default="run", help="label for this configuration, used downstream")
    p.add_argument(
        "--extra-body",
        default="{}",
        help='JSON merged into the request body, e.g. \'{"chat_template_kwargs":{"enable_thinking":false}}\'',
    )
    p.add_argument("--endpoint", choices=["completions", "chat"], default="completions")
    p.add_argument(
        "--nvidia-smi", action="store_true", help="sample nvidia-smi every 1s during each level"
    )
    p.add_argument(
        "--no-metrics", action="store_true", help="skip scraping <base-url-root>/metrics"
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed for prompt generation")
    p.add_argument(
        "--request-timeout", type=float, default=600.0, help="per-request timeout in seconds"
    )
    p.add_argument(
        "--ready-timeout",
        type=float,
        default=0.0,
        help="wait this many seconds for /v1/models + a healthy /health before the first "
        "level; 0 = do not wait (assume the caller already gated on readiness)",
    )
    p.add_argument(
        "--no-health-check",
        dest="health_check",
        action="store_false",
        default=True,
        help="do not poll /health between levels (the check aborts the sweep when the "
        "server reports a dead engine)",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="keep sweeping after a level in which every request failed (default: abort, "
        "because that level is measuring nothing and the next one will cost the same)",
    )
    return p.parse_args(argv)


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    concurrency_levels = [int(c.strip()) for c in args.concurrency.split(",") if c.strip()]
    extra_body = json.loads(args.extra_body)
    rng = random.Random(args.seed)
    tokenizer = _tokenizer_from_name(args.tokenizer)

    endpoint_path = "/chat/completions" if args.endpoint == "chat" else "/completions"
    url = args.base_url.rstrip("/") + endpoint_path
    headers = {"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"}
    metrics_url = None if args.no_metrics else derive_metrics_url(args.base_url)
    health_url = derive_health_url(args.base_url) if args.health_check else None
    models_url = args.base_url.rstrip("/") + "/models"

    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    connector = aiohttp.TCPConnector(limit=0)

    level_summaries: list[dict[str, Any]] = []
    aborted: Optional[str] = None
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        if args.ready_timeout > 0:
            await wait_until_ready(
                session, models_url, health_url, timeout_s=args.ready_timeout
            )
        for concurrency in concurrency_levels:
            # Between levels, not only at the start: the engine can die *during*
            # a level (an OOM on the first big prefill is exactly what happened),
            # and the next level would otherwise pay `num_prompts` timeouts to
            # learn the same thing.
            dead = await check_health(session, health_url)
            if dead is not None:
                aborted = f"aborting before concurrency={concurrency}: server unhealthy ({dead})"
                print(f"[bench] {aborted}", file=sys.stderr)
                break
            num_prompts = args.num_prompts or max(concurrency * 8, 64)
            # Build a fresh, deterministic prompt pool per level so the
            # dataset itself is reproducible across runs/configs.
            pool_size = min(num_prompts, 200)
            prompts = [
                make_prompt(args.dataset, args.input_len, tokenizer, rng, i)
                for i in range(pool_size)
            ]
            print(
                f"[bench] level concurrency={concurrency} num_prompts={num_prompts} "
                f"input_len~{args.input_len} output_len={args.output_len}",
                file=sys.stderr,
            )
            try:
                level = await run_level(
                    session=session,
                    url=url,
                    headers=headers,
                    endpoint=args.endpoint,
                    stream=args.stream,
                    model=args.model,
                    concurrency=concurrency,
                    num_prompts=num_prompts,
                    warmup=args.warmup,
                    output_len=args.output_len,
                    extra_body=extra_body,
                    prompts=prompts,
                    metrics_url=metrics_url,
                    use_gpu_sampler=args.nvidia_smi,
                    health_url=health_url,
                )
            except EngineDead as exc:
                aborted = f"aborting at concurrency={concurrency}: {exc}"
                print(f"[bench] {aborted}", file=sys.stderr)
                break
            summary = summarize_level(level)
            level_summaries.append(summary)
            if summary["num_success"] == 0 and not args.continue_on_error:
                aborted = (
                    f"aborting after concurrency={concurrency}: "
                    f"{summary['num_errors']}/{summary['num_errors']} requests failed"
                )
                print(f"[bench] {aborted}", file=sys.stderr)
                break

    result = {
        "metadata": {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "tag": args.tag,
            "args": {
                "base_url": args.base_url,
                "model": args.model,
                "concurrency": concurrency_levels,
                "num_prompts": args.num_prompts,
                "input_len": args.input_len,
                "output_len": args.output_len,
                "dataset": args.dataset,
                "tokenizer": args.tokenizer,
                "stream": args.stream,
                "warmup": args.warmup,
                "endpoint": args.endpoint,
                "extra_body": extra_body,
                "seed": args.seed,
            },
            "aborted": aborted,
        },
        "levels": level_summaries,
    }
    return result


def main() -> None:
    args = parse_args()
    try:
        result = asyncio.run(main_async(args))
    except (EngineDead, TimeoutError) as exc:
        # Terminal, and worth an explicit non-zero exit: a calling sweep
        # script can stop instead of letting the next stage run against a
        # dead server.
        print(f"[bench] FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[bench] wrote {args.out}", file=sys.stderr)

    print(f"\n## {args.tag}\n")
    print(render_markdown_table(args.tag, result["levels"]))

    aborted = result["metadata"].get("aborted")
    if aborted:
        print(f"[bench] FATAL: {aborted}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
