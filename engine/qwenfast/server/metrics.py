"""Renders an `EngineStats` snapshot as Prometheus text exposition format for `GET /metrics`.

Metric names use the `qwenfast:` namespace, plus a couple of
vLLM-alike aliases (`num_requests_running`/`waiting`) so `benchmarks/bench_serve.py`'s
best-effort scraper (`_INTERESTING_METRIC_SUBSTRINGS` in that file) picks something up even though
it was written against vLLM's metric names.
"""

from __future__ import annotations

from .engine_api import EngineStats, HistogramSnapshot

_NAMESPACE = "qwenfast"


def _gauge(name: str, help_text: str, value: float) -> list[str]:
    return [
        f"# HELP {name} {help_text}",
        f"# TYPE {name} gauge",
        f"{name} {value}",
    ]


def _histogram(name: str, help_text: str, snap: HistogramSnapshot) -> list[str]:
    lines = [f"# HELP {name} {help_text}", f"# TYPE {name} histogram"]
    for bound, count in zip(snap.bucket_bounds, snap.bucket_counts):
        lines.append(f'{name}_bucket{{le="{bound}"}} {count}')
    lines.append(f'{name}_bucket{{le="+Inf"}} {snap.count}')
    lines.append(f"{name}_sum {snap.sum}")
    lines.append(f"{name}_count {snap.count}")
    return lines


def render_prometheus_text(stats: EngineStats, extra_lines: list[str] | None = None) -> str:
    """`extra_lines` is appended verbatim — the public-API counters from `usage.py` come in that
    way, so this module keeps knowing only about the engine."""
    lines: list[str] = []

    lines += _gauge(
        f"{_NAMESPACE}:num_requests_running",
        "Number of requests currently being decoded/prefilled.",
        stats.num_requests_running,
    )
    lines += _gauge(
        f"{_NAMESPACE}:num_requests_waiting",
        "Number of requests admitted but not yet running.",
        stats.num_requests_waiting,
    )
    # vLLM-alike aliases for benchmarks/bench_serve.py's generic scraper.
    lines += _gauge("num_requests_running", "Alias of qwenfast:num_requests_running.", stats.num_requests_running)
    lines += _gauge("num_requests_waiting", "Alias of qwenfast:num_requests_waiting.", stats.num_requests_waiting)

    lines += _gauge(
        f"{_NAMESPACE}:decode_batch_size",
        "Current decode batch size (== num_requests_running in this engine).",
        stats.num_requests_running,
    )
    lines += _gauge(
        f"{_NAMESPACE}:ssm_slots_used", "SSM recurrent-state slots currently occupied.", stats.ssm_slots_used
    )
    lines += _gauge(f"{_NAMESPACE}:ssm_slots_total", "Total SSM slot pool size.", stats.ssm_slots_total)
    lines += _gauge(f"{_NAMESPACE}:kv_pages_used", "Paged-KV pages currently occupied.", stats.kv_slots_used)
    lines += _gauge(f"{_NAMESPACE}:kv_pages_total", "Total paged-KV pool size.", stats.kv_slots_total)
    # gpu_cache_usage_perc alias, matched by bench_serve.py's scraper substring list.
    kv_frac = (stats.kv_slots_used / stats.kv_slots_total) if stats.kv_slots_total else 0.0
    lines += _gauge("gpu_cache_usage_perc", "Alias of kv_pages_used / kv_pages_total.", kv_frac)

    lines += _gauge(
        f"{_NAMESPACE}:prompt_tokens_total", "Cumulative prompt tokens processed.", stats.prompt_tokens_total
    )
    lines += _gauge(
        f"{_NAMESPACE}:generation_tokens_total",
        "Cumulative generated tokens.",
        stats.generation_tokens_total,
    )
    lines += _gauge(
        f"{_NAMESPACE}:tokens_per_second", "Generated tokens / engine uptime.", stats.tokens_per_second
    )

    lines += _histogram(
        f"{_NAMESPACE}:time_to_first_token_seconds", "Time from admission to first output token.", stats.ttft
    )
    lines += _histogram(
        f"{_NAMESPACE}:time_per_output_token_seconds", "Per-token inter-token latency after the first.", stats.tpot
    )

    if stats.spec_accept_length is not None:
        lines += _gauge(
            f"{_NAMESPACE}:spec_accept_length", "Mean accepted MTP draft length.", stats.spec_accept_length
        )
    if stats.spec_acceptance_rate is not None:
        lines += _gauge(
            f"{_NAMESPACE}:spec_acceptance_rate", "Fraction of drafted tokens accepted.", stats.spec_acceptance_rate
        )

    lines += _gauge(f"{_NAMESPACE}:uptime_seconds", "Engine process uptime.", stats.uptime_s)

    if extra_lines:
        lines += extra_lines

    return "\n".join(lines) + "\n"
