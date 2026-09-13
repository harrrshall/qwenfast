"""A fake `AsyncEngine` that emits tokens at a configurable rate with a configurable prefill
delay, so `app.py` is fully testable (streaming format, usage accounting, think/tool-call
parsing, stop strings, abort-on-disconnect, concurrency, `/metrics`) without a GPU or the real
`qwenfast` runtime.

Two ways to get output tokens for a request:

* **Scripted** (what the test-suite uses almost everywhere): call `engine.script(request_id,
  text=...)` before issuing the HTTP request. The exact string is tokenized (with the real
  tokenizer if one was supplied, else a tiny fallback) and replayed token-by-token. This is what
  makes it possible to deterministically test things like "`</think>` straddles a chunk boundary"
  against the real streaming/detokenization/parsing pipeline instead of just the parser classes
  in isolation.
* **Unscripted** (what `bench_serve.py`/`run_eval.py` drive against): a small fixed phrase is
  repeated to fill `max_tokens`, so throughput/latency harnesses get plausible-looking output
  without any setup.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from typing import AsyncIterator, Optional, Sequence

from .engine_api import (
    AsyncEngine,
    EngineStats,
    Histogram,
    RequestStats,
    SamplingParams,
    StepOutput,
    DEFAULT_TPOT_BUCKETS_S,
    DEFAULT_TTFT_BUCKETS_S,
)

# A short, harmless filler phrase used when a request has no scripted output. Repeated (and
# retokenized) to fill however many tokens `max_tokens` asks for.
_DEFAULT_FILLER = (
    "The quick brown fox jumps over the lazy dog near the river while the "
    "sun sets slowly behind the distant mountains and a cool breeze begins. "
)

_ABORT_POLL_INTERVAL_S = 0.01  # granularity at which a sleeping request notices `abort()`


class MockEngine(AsyncEngine):
    def __init__(
        self,
        tokenizer=None,
        *,
        decode_tokens_per_second: float = 200.0,
        prefill_tokens_per_second: float = 100_000.0,
        prefill_base_delay_s: float = 0.005,
        max_concurrent_requests: int = 256,
        ssm_slots_total: int = 512,
        kv_slots_total: int = 200_000,
        eos_token_id: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.decode_tokens_per_second = decode_tokens_per_second
        self.prefill_tokens_per_second = prefill_tokens_per_second
        self.prefill_base_delay_s = prefill_base_delay_s
        self.eos_token_id = eos_token_id if eos_token_id is not None else _tokenizer_eos(tokenizer)

        self._admission = asyncio.Semaphore(max_concurrent_requests)
        self._waiting = 0
        self._running: set[str] = set()
        self._abort_flags: set[str] = set()
        self._scripts: dict[str, list[int]] = {}

        self._ssm_slots_total = ssm_slots_total
        self._kv_slots_total = kv_slots_total
        self._kv_pages_per_token = 1  # trivial accounting for the mock

        self._ttft_hist = Histogram(DEFAULT_TTFT_BUCKETS_S)
        self._tpot_hist = Histogram(DEFAULT_TPOT_BUCKETS_S)
        self._prompt_tokens_total = 0
        self._generation_tokens_total = 0
        self._start_time = time.monotonic()

        self._filler_ids = _tokenize_repeatable(self.tokenizer, _DEFAULT_FILLER)

    # ----------------------------------------------------------------
    # Test hook
    # ----------------------------------------------------------------

    def script(self, request_id: str, *, text: Optional[str] = None, token_ids: Optional[Sequence[int]] = None) -> None:
        """Register the exact output token sequence `request_id`'s next `add_request` call will
        replay. Either `text` (tokenized with `self.tokenizer`, or the fallback) or `token_ids`
        directly.
        """
        if (text is None) == (token_ids is None):
            raise ValueError("script() needs exactly one of text= or token_ids=")
        if text is not None:
            token_ids = _tokenize(self.tokenizer, text)
        self._scripts[request_id] = list(token_ids)

    # ----------------------------------------------------------------
    # AsyncEngine
    # ----------------------------------------------------------------

    async def add_request(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
    ) -> AsyncIterator[StepOutput]:
        self._waiting += 1
        try:
            async with self._admission:
                self._waiting -= 1
                self._running.add(request_id)
                self._prompt_tokens_total += len(prompt_token_ids)
                try:
                    async for step in self._run(request_id, prompt_token_ids, sampling_params):
                        yield step
                finally:
                    self._running.discard(request_id)
                    self._abort_flags.discard(request_id)
        finally:
            self._scripts.pop(request_id, None)

    async def _run(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
    ) -> AsyncIterator[StepOutput]:
        admitted_at = time.monotonic()

        output_ids = self._scripts.get(request_id)
        if output_ids is None:
            output_ids = list(
                itertools.islice(itertools.cycle(self._filler_ids), max(sampling_params.max_tokens, 1))
            )

        prefill_s = self.prefill_base_delay_s + len(prompt_token_ids) / self.prefill_tokens_per_second
        if await self._sleep_interruptible(request_id, prefill_s):
            yield StepOutput(
                new_token_ids=[],
                finished=True,
                finish_reason="abort",
                stats=RequestStats(prompt_tokens=len(prompt_token_ids), completion_tokens=0),
            )
            return

        completion_tokens = 0
        ttft_s: Optional[float] = None
        last_emit_t = admitted_at
        per_token_interval = 1.0 / self.decode_tokens_per_second if self.decode_tokens_per_second > 0 else 0.0
        stop_ids = set(sampling_params.stop_token_ids)

        for token_id in output_ids:
            if completion_tokens >= sampling_params.max_tokens:
                break
            if request_id in self._abort_flags:
                yield StepOutput(
                    new_token_ids=[],
                    finished=True,
                    finish_reason="abort",
                    stats=RequestStats(
                        prompt_tokens=len(prompt_token_ids),
                        completion_tokens=completion_tokens,
                        ttft_s=ttft_s,
                    ),
                )
                return

            if await self._sleep_interruptible(request_id, per_token_interval):
                yield StepOutput(
                    new_token_ids=[],
                    finished=True,
                    finish_reason="abort",
                    stats=RequestStats(
                        prompt_tokens=len(prompt_token_ids),
                        completion_tokens=completion_tokens,
                        ttft_s=ttft_s,
                    ),
                )
                return

            is_stop_token = (
                not sampling_params.ignore_eos
                and (token_id == self.eos_token_id or token_id in stop_ids)
            )
            if is_stop_token:
                yield StepOutput(
                    new_token_ids=[],
                    finished=True,
                    finish_reason="stop",
                    stats=RequestStats(
                        prompt_tokens=len(prompt_token_ids),
                        completion_tokens=completion_tokens,
                        ttft_s=ttft_s,
                    ),
                )
                return

            now_t = time.monotonic()
            completion_tokens += 1
            self._generation_tokens_total += 1
            if ttft_s is None:
                ttft_s = now_t - admitted_at
                self._ttft_hist.observe(ttft_s)
            else:
                self._tpot_hist.observe(now_t - last_emit_t)
            last_emit_t = now_t

            finished = completion_tokens >= sampling_params.max_tokens
            finish_reason = "length" if finished else None
            yield StepOutput(
                new_token_ids=[token_id],
                finished=finished,
                finish_reason=finish_reason,
                stats=RequestStats(
                    prompt_tokens=len(prompt_token_ids),
                    completion_tokens=completion_tokens,
                    ttft_s=ttft_s,
                ),
            )
            if finished:
                return

        # Scripted/filler tokens ran out before max_tokens: end as "stop" (nothing left to say),
        # mirroring a real model that emitted EOS.
        yield StepOutput(
            new_token_ids=[],
            finished=True,
            finish_reason="stop",
            stats=RequestStats(
                prompt_tokens=len(prompt_token_ids), completion_tokens=completion_tokens, ttft_s=ttft_s
            ),
        )

    async def abort(self, request_id: str) -> None:
        self._abort_flags.add(request_id)

    def get_stats(self) -> EngineStats:
        return EngineStats(
            num_requests_running=len(self._running),
            num_requests_waiting=self._waiting,
            prompt_tokens_total=self._prompt_tokens_total,
            generation_tokens_total=self._generation_tokens_total,
            tokens_per_second=self._generation_tokens_total
            / max(time.monotonic() - self._start_time, 1e-9),
            ttft=self._ttft_hist.snapshot(),
            tpot=self._tpot_hist.snapshot(),
            kv_slots_used=min(len(self._running) * 8, self._kv_slots_total),
            kv_slots_total=self._kv_slots_total,
            ssm_slots_used=len(self._running),
            ssm_slots_total=self._ssm_slots_total,
            spec_accept_length=None,  # MTP speculation is not modeled here
            spec_acceptance_rate=None,
            uptime_s=time.monotonic() - self._start_time,
        )

    # ----------------------------------------------------------------
    # internals
    # ----------------------------------------------------------------

    async def _sleep_interruptible(self, request_id: str, total_s: float) -> bool:
        """Sleep `total_s`, checking for `abort()` every `_ABORT_POLL_INTERVAL_S`. Returns True
        if aborted before the sleep completed."""
        remaining = total_s
        while remaining > 0:
            if request_id in self._abort_flags:
                return True
            chunk = min(_ABORT_POLL_INTERVAL_S, remaining)
            await asyncio.sleep(chunk)
            remaining -= chunk
        return request_id in self._abort_flags


# --------------------------------------------------------------------------
# tokenization helpers (mock-engine-local; the server has its own, more careful,
# incremental detokenizer in tokenization.py)
# --------------------------------------------------------------------------


def _tokenizer_eos(tokenizer) -> Optional[int]:
    if tokenizer is None:
        return None
    return getattr(tokenizer, "eos_token_id", None)


def _tokenize(tokenizer, text: str) -> list[int]:
    if tokenizer is not None:
        return list(tokenizer.encode(text, add_special_tokens=False))
    from .fake_tokenizer import FakeTokenizer

    return list(FakeTokenizer.shared().encode(text))


def _tokenize_repeatable(tokenizer, text: str) -> list[int]:
    ids = _tokenize(tokenizer, text)
    return ids if ids else [0]
