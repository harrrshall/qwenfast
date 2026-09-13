"""Streaming must emit one delta chunk per *token*, not per engine step.

Speculative decoding commits several tokens in a single step. If the server folds those into
one SSE chunk, any client that counts stream chunks to display a live tokens/sec figure
under-reads by the accept length (2-4x on this engine) until the final `usage` frame lands.
These tests pin the per-token behaviour and the byte-identical text it must still produce.

Run with:
    PYTHONPATH=engine <venv>/bin/pytest engine/qwenfast/server/tests/test_spec_chunking.py -v
"""

from __future__ import annotations

import json
from typing import AsyncIterator, Optional

import httpx
import pytest

from qwenfast.server.app import create_app
from qwenfast.server.engine_api import (
    AsyncEngine,
    EngineStats,
    RequestStats,
    SamplingParams,
    StepOutput,
)
from qwenfast.server.tokenization import IncrementalDetokenizer


class SpecStepEngine(AsyncEngine):
    """Yields `tokens_per_step` committed tokens per step, the way MTP/spec decoding does."""

    def __init__(self, token_ids: list[int], *, tokens_per_step: int) -> None:
        self.token_ids = list(token_ids)
        self.tokens_per_step = tokens_per_step
        self.aborted: list[str] = []

    async def add_request(
        self, request_id: str, prompt_token_ids: list[int], sampling_params: SamplingParams
    ) -> AsyncIterator[StepOutput]:
        emitted = 0
        n = len(self.token_ids)
        for i in range(0, n, self.tokens_per_step):
            batch = self.token_ids[i : i + self.tokens_per_step]
            emitted += len(batch)
            finished = emitted >= n
            yield StepOutput(
                new_token_ids=batch,
                finished=finished,
                finish_reason="length" if finished else None,
                stats=RequestStats(
                    prompt_tokens=len(prompt_token_ids),
                    completion_tokens=emitted,
                    ttft_s=0.01,
                ),
            )

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)

    def get_stats(self) -> EngineStats:
        return EngineStats(num_requests_running=0, num_requests_waiting=0)


async def _stream_chunks(app, body: dict) -> list[dict]:
    transport = httpx.ASGITransport(app=app)
    out: list[dict] = []
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30.0) as c:
        async with c.stream("POST", "/v1/chat/completions", json=body) as resp:
            assert resp.status_code == 200
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                out.append(json.loads(data))
    return out


def _content_chunks(chunks: list[dict]) -> list[str]:
    return [
        ch["choices"][0]["delta"]["content"]
        for ch in chunks
        if ch.get("choices") and ch["choices"][0].get("delta", {}).get("content")
    ]


@pytest.mark.parametrize("tokens_per_step", [1, 2, 3, 4])
async def test_one_chunk_per_token(tokenizer, tokens_per_step):
    """The number of content chunks equals the number of generated tokens, independent of how
    many tokens each engine step commits."""
    text = "The quick brown fox jumps over the lazy dog and keeps running for a while."
    token_ids = list(tokenizer.encode(text, add_special_tokens=False))

    engine = SpecStepEngine(token_ids, tokens_per_step=tokens_per_step)
    app = create_app(engine, tokenizer, model_name="qwenfast-spec")

    chunks = await _stream_chunks(
        app,
        {
            "model": "qwenfast-spec",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )

    contents = _content_chunks(chunks)
    usage = next((c["usage"] for c in chunks if c.get("usage")), None)

    assert usage is not None, "final usage frame missing"
    assert usage["completion_tokens"] == len(token_ids)
    # The point of the change: chunk count tracks token count, so a chunk-counting meter is
    # exact even before `usage` arrives.
    assert len(contents) == len(token_ids), (
        f"tokens_per_step={tokens_per_step}: got {len(contents)} chunks "
        f"for {len(token_ids)} tokens"
    )


@pytest.mark.parametrize("tokens_per_step", [1, 3])
async def test_text_is_byte_identical_across_step_sizes(tokenizer, tokens_per_step):
    """Splitting a step's tokens must not change the assembled text."""
    text = "Hello — naïve café 日本語 emoji 🚀 done."
    token_ids = list(tokenizer.encode(text, add_special_tokens=False))

    engine = SpecStepEngine(token_ids, tokens_per_step=tokens_per_step)
    app = create_app(engine, tokenizer, model_name="qwenfast-spec")

    chunks = await _stream_chunks(
        app,
        {
            "model": "qwenfast-spec",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    assembled = "".join(_content_chunks(chunks))
    expected = tokenizer.decode(token_ids, skip_special_tokens=True)
    assert assembled == expected


def test_add_tokens_split_matches_add_tokens(tokenizer):
    """`"".join(add_tokens_split(ids)) == add_tokens(ids)` — including multi-byte characters
    that straddle a token boundary, where individual entries are empty."""
    text = "naïve café 日本語 🚀🚀 tail"
    ids = list(tokenizer.encode(text, add_special_tokens=False))
    prompt = [1, 2, 3]

    batched = IncrementalDetokenizer(tokenizer, prompt).add_tokens(tuple(ids))
    split = IncrementalDetokenizer(tokenizer, prompt).add_tokens_split(tuple(ids))

    assert "".join(split) == batched
    assert len(split) == len(ids)
