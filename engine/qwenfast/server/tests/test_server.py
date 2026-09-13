"""pytest + httpx tests for `qwenfast.server`, run against `MockEngine` (no GPU needed).

Covers: streaming chunk format, usage accounting, `<think>`-tag
parsing split across chunks, Hermes tool-call parsing, stop strings, abort on client disconnect,
concurrency of 64 streams, and `/metrics` content.

Run with:
    PYTHONPATH=engine <venv>/bin/pytest engine/qwenfast/server/tests/test_server.py -v
"""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest

from qwenfast.server.app import create_app
from qwenfast.server.mock_engine import MockEngine
from qwenfast.server.tokenization import (
    IncrementalDetokenizer,
    StopStringMatcher,
    ThinkTagParser,
    ToolCallStreamParser,
)

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def make_engine_and_app(tokenizer, *, api_key=None, **engine_kwargs):
    engine_kwargs.setdefault("decode_tokens_per_second", 2000.0)
    engine_kwargs.setdefault("prefill_base_delay_s", 0.001)
    engine = MockEngine(tokenizer=tokenizer, **engine_kwargs)
    app = create_app(engine, tokenizer, model_name="qwenfast-mock", api_key=api_key)
    return engine, app


async def client_for(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30.0)


async def collect_sse(resp) -> list[dict]:
    events = []
    async for line in resp.aiter_lines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        if payload == "[DONE]":
            break
        events.append(json.loads(payload))
    return events


# --------------------------------------------------------------------------
# 1. streaming chunk format
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_streaming_chunk_format(tokenizer):
    engine, app = make_engine_and_app(tokenizer)
    async with await client_for(app) as client:
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "qwenfast-mock",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 5,
                "stream": True,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        ) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            raw = await resp.aread()

    lines = [l for l in raw.decode().split("\n\n") if l.strip()]
    assert lines[-1] == "data: [DONE]"
    events = [json.loads(l[len("data: ") :]) for l in lines[:-1]]
    assert events, "expected at least one SSE data event"

    for ev in events:
        assert ev["object"] == "chat.completion.chunk"
        assert "id" in ev and ev["id"].startswith("chatcmpl-")
        assert ev["model"] == "qwenfast-mock"
        assert "choices" in ev

    # First content-bearing chunk announces the assistant role (OpenAI convention).
    first_choice_events = [ev for ev in events if ev["choices"]]
    assert first_choice_events[0]["choices"][0]["delta"].get("role") == "assistant"
    # Exactly one terminal chunk carries a non-null finish_reason.
    finishes = [ev["choices"][0]["finish_reason"] for ev in first_choice_events if ev["choices"]]
    assert finishes.count("length") == 1
    assert finishes[-1] == "length"


@pytest.mark.asyncio
async def test_completions_streaming_chunk_format(tokenizer):
    engine, app = make_engine_and_app(tokenizer)
    async with await client_for(app) as client:
        async with client.stream(
            "POST",
            "/v1/completions",
            json={"model": "qwenfast-mock", "prompt": "hello world", "max_tokens": 4, "stream": True},
        ) as resp:
            assert resp.status_code == 200
            events = await collect_sse(resp)

    assert events
    for ev in events:
        assert ev["object"] == "text_completion"
        assert "choices" in ev
    joined = "".join(ev["choices"][0]["text"] for ev in events if ev["choices"] and "text" in ev["choices"][0])
    assert joined.strip() != ""
    assert events[-1]["choices"][0]["finish_reason"] in ("length", "stop")


# --------------------------------------------------------------------------
# 2. usage accounting
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_usage_accounting_non_stream(tokenizer):
    engine, app = make_engine_and_app(tokenizer)
    async with await client_for(app) as client:
        r = await client.post(
            "/v1/chat/completions",
            json={
                "model": "qwenfast-mock",
                "messages": [{"role": "user", "content": "count to ten please"}],
                "max_tokens": 7,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
    assert r.status_code == 200
    body = r.json()
    usage = body["usage"]
    assert usage["completion_tokens"] == 7
    assert usage["prompt_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert body["choices"][0]["finish_reason"] == "length"


@pytest.mark.asyncio
async def test_usage_accounting_stream_include_usage(tokenizer):
    engine, app = make_engine_and_app(tokenizer)
    async with await client_for(app) as client:
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "qwenfast-mock",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 6,
                "stream": True,
                "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": False},
            },
        ) as resp:
            events = await collect_sse(resp)

    usage_events = [ev for ev in events if "usage" in ev]
    assert len(usage_events) == 1
    usage = usage_events[0]["usage"]
    assert usage_events[0]["choices"] == []  # OpenAI convention: usage-only chunk has no choices
    assert usage["completion_tokens"] == 6
    assert usage["total_tokens"] == usage["prompt_tokens"] + 6


@pytest.mark.asyncio
async def test_no_usage_without_include_usage(tokenizer):
    engine, app = make_engine_and_app(tokenizer)
    async with await client_for(app) as client:
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "qwenfast-mock",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 3,
                "stream": True,
            },
        ) as resp:
            events = await collect_sse(resp)
    assert not any("usage" in ev for ev in events)


# --------------------------------------------------------------------------
# 3. <think> tag parsing, split across chunks (direct parser unit tests --
#    deterministic regardless of tokenizer/tokenization boundaries)
# --------------------------------------------------------------------------


def test_think_parser_close_tag_split_across_feeds():
    p = ThinkTagParser(start_in_think=True)
    out_r, out_c = [], []

    for frag in ["thinking about it", "... almost done</th", "ink>\n\nthe answer", " is 42"]:
        r, c = p.feed(frag)
        out_r.append(r)
        out_c.append(c)
    r, c = p.flush()
    out_r.append(r)
    out_c.append(c)

    assert "".join(out_r) == "thinking about it... almost done"
    assert "".join(out_c) == "the answer is 42"


def test_think_parser_close_tag_split_one_char_at_a_time():
    p = ThinkTagParser(start_in_think=True)
    text = "abc</think>def"
    reasoning, content = [], []
    for ch in text:
        r, c = p.feed(ch)
        reasoning.append(r)
        content.append(c)
    r, c = p.flush()
    reasoning.append(r)
    content.append(c)
    assert "".join(reasoning) == "abc"
    assert "".join(content) == "def"


def test_think_parser_no_close_tag_stays_reasoning():
    p = ThinkTagParser(start_in_think=True)
    r1, c1 = p.feed("never closes")
    r2, c2 = p.flush()
    assert r1 + r2 == "never closes"
    assert c1 == c2 == ""


def test_think_parser_disabled_thinking_is_pure_content():
    p = ThinkTagParser(start_in_think=False)
    r, c = p.feed("just an answer, no tags")
    assert r == ""
    assert c == "just an answer, no tags"


def test_think_parser_defensive_explicit_open_tag():
    # If the model (against instructions) echoes an explicit <think> tag mid-content, the parser
    # still tracks it rather than leaking the literal tag text.
    p = ThinkTagParser(start_in_think=False)
    r, c = p.feed("before <think>musing</think> after")
    # A single feed() call resolves every complete tag it contains in one pass, so both the
    # pre-tag and post-tag content land in this call's delta; nothing is left for flush().
    assert c == "before  after"
    assert r == "musing"
    r2, c2 = p.flush()
    assert (r2, c2) == ("", "")


# --------------------------------------------------------------------------
# 4. Hermes tool-call streaming parser (direct unit tests + one end-to-end HTTP test)
# --------------------------------------------------------------------------


def test_tool_call_parser_streaming_reconstructs_valid_json():
    p = ToolCallStreamParser()
    script = (
        'before text <tool_call>\n{"name": "get_weather", "arguments": '
        '{"city": "Zurich", "unit": "celsius"}}\n</tool_call> after text'
    )
    content_parts = []
    arg_fragments: dict[int, list[str]] = {}
    name_by_index: dict[int, str] = {}

    # feed one character at a time -- the hardest case for a streaming parser
    for ch in script:
        c, deltas = p.feed(ch)
        content_parts.append(c)
        for d in deltas:
            if d.name is not None:
                name_by_index[d.index] = d.name
            if d.arguments_delta:
                arg_fragments.setdefault(d.index, []).append(d.arguments_delta)
    c, deltas = p.flush()
    content_parts.append(c)
    for d in deltas:
        if d.name is not None:
            name_by_index[d.index] = d.name
        if d.arguments_delta:
            arg_fragments.setdefault(d.index, []).append(d.arguments_delta)

    assert "".join(content_parts) == "before text  after text"
    assert p.has_tool_calls
    assert name_by_index == {0: "get_weather"}
    full_args = "".join(arg_fragments[0])
    assert json.loads(full_args) == {"city": "Zurich", "unit": "celsius"}


def test_tool_call_parser_multiple_calls():
    p = ToolCallStreamParser()
    script = (
        '<tool_call>\n{"name": "a", "arguments": {"x": 1}}\n</tool_call>'
        '<tool_call>\n{"name": "b", "arguments": {"y": [1, 2, "}"]}}\n</tool_call>'
    )
    names = {}
    args = {}
    for chunk_size in (1,):
        for i in range(0, len(script), chunk_size):
            _, deltas = p.feed(script[i : i + chunk_size])
            for d in deltas:
                if d.name is not None:
                    names[d.index] = d.name
                if d.arguments_delta:
                    args.setdefault(d.index, []).append(d.arguments_delta)
    p.flush()
    assert names == {0: "a", 1: "b"}
    assert json.loads("".join(args[0])) == {"x": 1}
    assert json.loads("".join(args[1])) == {"y": [1, 2, "}"]}  # nested '}' inside a string


@pytest.mark.asyncio
async def test_tool_calls_end_to_end_http(tokenizer):
    engine, app = make_engine_and_app(tokenizer)
    async with await client_for(app) as client:
        engine.script(
            "toolcall-req",
            text='Reasoning about it.</think>\n\nSure! <tool_call>\n'
            '{"name": "get_weather", "arguments": {"city": "Zurich"}}\n</tool_call>',
        )
        r = await client.post(
            "/v1/chat/completions",
            json={
                "model": "qwenfast-mock",
                "messages": [{"role": "user", "content": "weather in zurich?"}],
                "max_tokens": 128,
                "chat_template_kwargs": {"enable_thinking": True},
            },
            headers={"X-Request-Id": "toolcall-req"},
        )
    assert r.status_code == 200
    body = r.json()
    msg = body["choices"][0]["message"]
    assert body["choices"][0]["finish_reason"] == "tool_calls"
    assert msg["reasoning_content"].strip() == "Reasoning about it."
    assert msg["content"].strip() == "Sure!"
    assert len(msg["tool_calls"]) == 1
    call = msg["tool_calls"][0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Zurich"}


# --------------------------------------------------------------------------
# 5. incremental detokenization
# --------------------------------------------------------------------------


def test_incremental_detokenizer_matches_batch_decode(tokenizer):
    text = "The quick brown fox jumps over the lazy dog. 你好，世界！ Zürich café naïve"
    ids = tokenizer.encode(text, add_special_tokens=False)
    prompt_ids = ids[:2]
    gen_ids = ids[2:]

    detok = IncrementalDetokenizer(tokenizer, prompt_ids)
    streamed = "".join(detok.add_token(t) for t in gen_ids)

    expected = tokenizer.decode(ids, skip_special_tokens=True)[
        len(tokenizer.decode(prompt_ids, skip_special_tokens=True)) :
    ]
    assert streamed == expected


def test_stop_string_matcher_holds_back_partial_suffix():
    m = StopStringMatcher(["STOP"])
    out1, stopped1 = m.feed("hello wor")
    out2, stopped2 = m.feed("ld ST")
    out3, stopped3 = m.feed("OP more text")
    assert out1 == "hello wor"
    assert not stopped1
    assert out2 == "ld "
    assert not stopped2
    assert out3 == ""
    assert stopped3


# --------------------------------------------------------------------------
# 6. stop strings, end to end over HTTP
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_string_truncates_and_finishes_early(tokenizer):
    engine, app = make_engine_and_app(tokenizer)
    async with await client_for(app) as client:
        engine.script("stop-req", text="one two three STOPWORD four five six seven")
        r = await client.post(
            "/v1/chat/completions",
            json={
                "model": "qwenfast-mock",
                "messages": [{"role": "user", "content": "count"}],
                "max_tokens": 64,
                "stop": ["STOPWORD"],
                "chat_template_kwargs": {"enable_thinking": False},
            },
            headers={"X-Request-Id": "stop-req"},
        )
    assert r.status_code == 200
    body = r.json()
    content = body["choices"][0]["message"]["content"]
    assert "STOPWORD" not in content
    assert content.strip() == "one two three"
    assert body["choices"][0]["finish_reason"] == "stop"
    # fewer tokens than max_tokens were actually generated -- generation really stopped early
    assert body["usage"]["completion_tokens"] < 64


# --------------------------------------------------------------------------
# 7. abort on client disconnect
# --------------------------------------------------------------------------


class _FakeDisconnectingRequest:
    """Stands in for a starlette `Request`: reports disconnected after `disconnect_after` calls
    to `is_disconnected()`. Used to test the abort wiring in `app._generate_chat` directly and
    deterministically -- real ASGI-transport-level disconnect timing in a test harness is not
    something this codebase controls, so we test the code path we own instead.
    """

    def __init__(self, disconnect_after: int = 2):
        self._calls = 0
        self._disconnect_after = disconnect_after

    async def is_disconnected(self) -> bool:
        self._calls += 1
        return self._calls > self._disconnect_after


@pytest.mark.asyncio
async def test_abort_on_disconnect_stops_generation(tokenizer):
    from qwenfast.server.app import _generate_chat
    from qwenfast.server.engine_api import SamplingParams

    engine = MockEngine(tokenizer=tokenizer, decode_tokens_per_second=50.0, prefill_base_delay_s=0.0)
    request_id = "disconnect-req"
    engine.script(request_id, text=" ".join(f"word{i}" for i in range(200)))
    prompt_ids = tokenizer.encode("hi", add_special_tokens=False)
    sp = SamplingParams(max_tokens=200, ignore_eos=True)

    fake_request = _FakeDisconnectingRequest(disconnect_after=2)
    executor = _thread_pool()
    events = []
    try:
        async for ev in _generate_chat(
            engine, tokenizer, executor, request_id, prompt_ids, sp, False, [], fake_request
        ):
            events.append(ev)
    finally:
        executor.shutdown(wait=False)

    assert events[-1].finished
    # We stopped well before all 200 tokens were produced.
    assert engine.get_stats().generation_tokens_total < 200
    # The engine no longer considers the request running/waiting.
    stats = engine.get_stats()
    assert stats.num_requests_running == 0
    assert stats.num_requests_waiting == 0


def _thread_pool():
    from concurrent.futures import ThreadPoolExecutor

    return ThreadPoolExecutor(max_workers=1)


@pytest.mark.asyncio
async def test_mock_engine_abort_is_prompt(tokenizer):
    engine = MockEngine(tokenizer=tokenizer, decode_tokens_per_second=20.0, prefill_base_delay_s=0.0)
    request_id = "abort-req"
    from qwenfast.server.engine_api import SamplingParams

    prompt_ids = [1, 2, 3]
    sp = SamplingParams(max_tokens=1000, ignore_eos=True)

    gen = engine.add_request(request_id, prompt_ids, sp)
    step = await gen.__anext__()
    assert not step.finished

    t0 = time.monotonic()
    await engine.abort(request_id)
    async for step in gen:
        pass
    elapsed = time.monotonic() - t0
    assert step.finished
    assert step.finish_reason == "abort"
    assert elapsed < 1.0  # well under one token interval (1/20 s) * poll granularity


# --------------------------------------------------------------------------
# 8. concurrency of 64 streams
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_64_concurrent_streams(tokenizer):
    engine, app = make_engine_and_app(
        tokenizer, decode_tokens_per_second=1000.0, max_concurrent_requests=128
    )
    n = 64

    async def one_stream(client: httpx.AsyncClient, i: int) -> dict:
        req_id = f"conc-{i}"
        engine.script(req_id, text=f"response number {i} with a few words of filler text here")
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "qwenfast-mock",
                "messages": [{"role": "user", "content": f"request {i}"}],
                "max_tokens": 10,
                "stream": True,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            headers={"X-Request-Id": req_id},
        ) as resp:
            events = await collect_sse(resp)
            return {"status": resp.status_code, "n_events": len(events)}

    async with await client_for(app) as client:
        results = await asyncio.gather(*(one_stream(client, i) for i in range(n)))

    assert len(results) == n
    assert all(r["status"] == 200 for r in results)
    assert all(r["n_events"] > 0 for r in results)

    # everyone finished cleanly -- engine has no leaked running/waiting slots
    stats = engine.get_stats()
    assert stats.num_requests_running == 0
    assert stats.num_requests_waiting == 0


# --------------------------------------------------------------------------
# 9. /metrics content
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metrics_endpoint_content(tokenizer):
    engine, app = make_engine_and_app(tokenizer, decode_tokens_per_second=50.0)

    async with await client_for(app) as client:
        # kick off a couple of in-flight streams so running > 0 is observable
        async def slow_stream(i: int):
            req_id = f"metrics-req-{i}"
            engine.script(req_id, text=" ".join(f"w{i}" for i in range(50)))
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "qwenfast-mock",
                    "messages": [{"role": "user", "content": "go"}],
                    "max_tokens": 40,
                    "stream": True,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                headers={"X-Request-Id": req_id},
            ) as resp:
                await collect_sse(resp)

        tasks = [asyncio.create_task(slow_stream(i)) for i in range(3)]
        await asyncio.sleep(0.05)  # let them get admitted

        r = await client.get("/metrics")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/plain")
        text = r.text

        for name in (
            "qwenfast:num_requests_running",
            "qwenfast:num_requests_waiting",
            "qwenfast:ssm_slots_used",
            "qwenfast:ssm_slots_total",
            "qwenfast:kv_pages_used",
            "qwenfast:decode_batch_size",
            "qwenfast:time_to_first_token_seconds",
            "qwenfast:time_per_output_token_seconds",
        ):
            assert name in text, f"missing metric {name}"

        assert "qwenfast:time_to_first_token_seconds_bucket" in text
        assert "qwenfast:time_to_first_token_seconds_sum" in text
        assert "qwenfast:time_to_first_token_seconds_count" in text

        await asyncio.gather(*tasks)

    # after everything finishes, running should have drained back to 0
    stats_after = engine.get_stats()
    assert stats_after.num_requests_running == 0


# --------------------------------------------------------------------------
# 10. auth, /health, /v1/models
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_and_models_no_auth_required(tokenizer):
    engine, app = make_engine_and_app(tokenizer, api_key="secret123")
    async with await client_for(app) as client:
        r = await client.get("/health")
        assert r.status_code == 200
        r = await client.get("/metrics")
        assert r.status_code == 200


@pytest.mark.asyncio
async def test_api_key_required_for_v1(tokenizer):
    engine, app = make_engine_and_app(tokenizer, api_key="secret123")
    async with await client_for(app) as client:
        r = await client.get("/v1/models")
        assert r.status_code == 401

        r = await client.get("/v1/models", headers={"Authorization": "Bearer wrong"})
        assert r.status_code == 401

        r = await client.get("/v1/models", headers={"Authorization": "Bearer secret123"})
        assert r.status_code == 200
        body = r.json()
        assert body["data"][0]["id"] == "qwenfast-mock"

        r = await client.post(
            "/v1/chat/completions",
            json={"model": "qwenfast-mock", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 2},
        )
        assert r.status_code == 401


@pytest.mark.asyncio
async def test_no_api_key_means_open_access(tokenizer):
    engine, app = make_engine_and_app(tokenizer, api_key=None)
    async with await client_for(app) as client:
        r = await client.get("/v1/models")
        assert r.status_code == 200
