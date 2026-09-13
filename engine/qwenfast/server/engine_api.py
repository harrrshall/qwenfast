"""The `AsyncEngine` interface: the contract between the HTTP server and the engine.

This module owns nothing about tokenization, HTTP, or the OpenAI wire format — it is the
narrow boundary between the server (this subpackage) and whatever actually runs the model
(`mock_engine.py` for tests, or the real `qwenfast` runtime, which plugs in behind the same
interface).

Contract (frozen, do not widen without updating every implementation):

    engine.add_request(request_id, prompt_token_ids, sampling_params) -> AsyncIterator[StepOutput]

    Each `StepOutput` carries the *newly generated* token ids since the previous yield (usually
    exactly one, but an implementation may batch more than one per yield — the server must not
    assume `len(new_token_ids) == 1`), whether the request is finished, why, and a stats snapshot.
    The generator is exhausted (raises `StopAsyncIteration`) after the yield where `finished=True`.

    engine.abort(request_id) -> None
        Idempotent. Signals the engine to stop producing tokens for `request_id` as soon as
        possible. The corresponding `add_request` iterator must terminate (with `finished=True`,
        `finish_reason="abort"`, possibly zero new tokens) shortly after — it must not hang.

    engine.get_stats() -> EngineStats
        A point-in-time snapshot for `/metrics`. Cheap and non-blocking; safe to call from the
        HTTP event loop on every scrape.
"""

from __future__ import annotations

import abc
import time
from bisect import bisect_left
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

# --------------------------------------------------------------------------
# Sampling parameters
# --------------------------------------------------------------------------


@dataclass
class SamplingParams:
    """Engine-facing sampling knobs. The server is responsible for resolving OpenAI-request
    fields (including the model-card thinking/non-thinking defaults) down to this shape before
    calling `add_request`.
    """

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1  # -1 == disabled
    min_p: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    max_tokens: int = 16
    ignore_eos: bool = False
    stop_token_ids: tuple[int, ...] = ()
    seed: Optional[int] = None
    n: int = 1  # only 1 is supported end-to-end; the server rejects n != 1
    logprobs: Optional[int] = None  # number of top logprobs to return per token, if any

    # Stop *strings* are matched against detokenized text, which the engine never sees (it only
    # deals in token ids) — that matching happens in the server's streaming loop. It is still
    # useful for an engine to know about them (e.g. to end a batched multi-token yield early), so
    # they are threaded through even though `SamplingParams` otherwise stays token-space-only.
    stop: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.n != 1:
            raise ValueError("SamplingParams.n != 1 is not supported (server enforces this too)")
        if self.max_tokens < 0:
            raise ValueError("max_tokens must be >= 0")


# --------------------------------------------------------------------------
# Step / stats types
# --------------------------------------------------------------------------

FinishReason = Optional[str]  # "stop" | "length" | "abort" | None (not finished yet)


@dataclass
class StepOutput:
    """One yield from `AsyncEngine.add_request`."""

    new_token_ids: list[int]
    finished: bool
    finish_reason: FinishReason
    stats: "RequestStats"
    # Optional per-new-token top-k logprobs, parallel to `new_token_ids`, each entry a
    # `{token_id: logprob}` mapping (including the sampled token itself). None if not requested.
    logprobs: Optional[list[dict[int, float]]] = None


@dataclass
class RequestStats:
    """Per-request stats snapshot attached to every `StepOutput`."""

    prompt_tokens: int
    completion_tokens: int
    ttft_s: Optional[float] = None  # wall time from admission to first token, set once
    spec_accept_length: Optional[float] = None  # mean accepted draft length so far (MTP, M4+)


@dataclass
class HistogramSnapshot:
    """A Prometheus-style cumulative histogram: `buckets` maps each upper bound (`le`) to the
    cumulative count of observations <= that bound; `+Inf` is implied to equal `count`.
    """

    bucket_bounds: tuple[float, ...]
    bucket_counts: tuple[int, ...]  # cumulative, same length/order as bucket_bounds
    sum: float
    count: int

    @staticmethod
    def empty(bucket_bounds: tuple[float, ...]) -> "HistogramSnapshot":
        return HistogramSnapshot(
            bucket_bounds=bucket_bounds,
            bucket_counts=tuple(0 for _ in bucket_bounds),
            sum=0.0,
            count=0,
        )


class Histogram:
    """A minimal cumulative Prometheus-style histogram. Not thread-safe; the engine loop and the
    metrics scrape both run on the same asyncio event loop / single engine thread in this design,
    so no lock is needed.
    """

    def __init__(self, bucket_bounds: tuple[float, ...]) -> None:
        self._bounds = tuple(sorted(bucket_bounds))
        self._counts = [0] * len(self._bounds)
        self._sum = 0.0
        self._count = 0

    def observe(self, value: float) -> None:
        """One sample. Cumulative counts, so every bucket at or above the
        value's own bucket is incremented.

        This is called **once per emitted token per
        stream** from the engine thread's `_dispatch` -- 256 times per decode
        step at conc 256 -- and it used to be a Python `for` over all 11-14
        bounds, i.e. ~3,000 interpreted loop iterations per step on the thread
        that is also issuing the model's kernel launches. `bisect` finds the
        first bucket in C and the slice assignment runs in C too; the counts
        are identical, which `test_metrics.py` pins."""
        self._sum += value
        self._count += 1
        i = bisect_left(self._bounds, value)
        n = len(self._bounds)
        if i < n:
            counts = self._counts
            for j in range(i, n):
                counts[j] += 1

    def snapshot(self) -> HistogramSnapshot:
        return HistogramSnapshot(
            bucket_bounds=self._bounds,
            bucket_counts=tuple(self._counts),
            sum=self._sum,
            count=self._count,
        )


# Default bucket layouts, chosen for the expected ranges of step times and TTFT.
DEFAULT_TTFT_BUCKETS_S: tuple[float, ...] = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
)
DEFAULT_TPOT_BUCKETS_S: tuple[float, ...] = (
    0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0,
)


@dataclass
class EngineStats:
    """A `/metrics`-ready snapshot of the server's metric list."""

    num_requests_running: int
    num_requests_waiting: int
    prompt_tokens_total: int = 0
    generation_tokens_total: int = 0
    tokens_per_second: float = 0.0
    ttft: HistogramSnapshot = field(
        default_factory=lambda: HistogramSnapshot.empty(DEFAULT_TTFT_BUCKETS_S)
    )
    tpot: HistogramSnapshot = field(
        default_factory=lambda: HistogramSnapshot.empty(DEFAULT_TPOT_BUCKETS_S)
    )
    kv_slots_used: int = 0
    kv_slots_total: int = 0
    ssm_slots_used: int = 0
    ssm_slots_total: int = 0
    spec_accept_length: Optional[float] = None
    spec_acceptance_rate: Optional[float] = None
    uptime_s: float = 0.0


# --------------------------------------------------------------------------
# The interface
# --------------------------------------------------------------------------


class AsyncEngine(abc.ABC):
    """Everything the server needs from a model-serving backend."""

    @abc.abstractmethod
    def add_request(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
    ) -> AsyncIterator[StepOutput]:
        """Admit a request and return an async iterator of `StepOutput`s.

        Implementations are typically async generators, e.g.::

            async def add_request(self, request_id, prompt_token_ids, sampling_params):
                ...
                while not done:
                    ...
                    yield StepOutput(...)
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def abort(self, request_id: str) -> None:
        """Best-effort, idempotent cancellation. Safe to call after the request has already
        finished or was never admitted."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_stats(self) -> EngineStats:
        """Synchronous, non-blocking. Called on every `/metrics` scrape."""
        raise NotImplementedError

    def health(self) -> Optional[str]:
        """`None` when the engine can serve; otherwise a one-line failure reason.

        Optional, and default-healthy, so every existing implementation keeps
        working. An engine that owns a background device thread **should**
        override it: `/health` reports it, and `/v1/*` turns a non-`None`
        answer into a 503. Without it, an engine thread that died of a CUDA OOM
        leaves the HTTP server answering `{"status": "ok"}` while clients wait on
        streams that will never produce a token."""
        return None

    async def start(self) -> None:
        """Optional lifecycle hook — called once before the HTTP server starts accepting
        traffic. Default no-op; a real engine overrides this to spin up its device thread."""
        return None

    async def shutdown(self) -> None:
        """Optional lifecycle hook — called once on server shutdown. Default no-op."""
        return None


def now() -> float:
    """Monotonic clock, shared so every implementation's TTFT/TPOT math agrees."""
    return time.monotonic()
