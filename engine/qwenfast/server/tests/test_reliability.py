"""The reliability layer — context clamping, error shapes, fairness,
timeouts, draining, and the error-class breakdown.

Everything here runs against `MockEngine` on CPU — no GPU, no weights, no network:

    PYTHONPATH=engine <venv>/bin/pytest engine/qwenfast/server/tests/test_reliability.py -v

Each test is written against a failure observed in real public traffic rather than against the
code, so a regression that re-introduces one of those refusals fails a named test.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time

import httpx
import pytest

from qwenfast.server.app import create_app
from qwenfast.server.auth import ApiKey, ConcurrencyLimiter
from qwenfast.server.content_log import QueryLogger
from qwenfast.server.mock_engine import MockEngine
from qwenfast.server.usage import UsageRecord, UsageRecorder, percentiles

PUBLIC = "qf-public-000000000000000000000000"
OTHER = "qf-other-0000000000000000000000000"
ADMIN = "qf-admin-0000000000000000000000000"


def write_keys(path, entries) -> None:
    path.write_text(json.dumps({"keys": entries}))


def entries():
    return [{"key": PUBLIC, "name": "public"}, {"key": OTHER, "name": "other"}]


def build(tokenizer, keys_path=None, *, tps=4000.0, **kwargs):
    engine = MockEngine(
        tokenizer=tokenizer, decode_tokens_per_second=tps, prefill_base_delay_s=0.001
    )
    app = create_app(
        engine,
        tokenizer,
        model_name="qwenfast-mock",
        api_keys_file=str(keys_path) if keys_path else None,
        admin_key=kwargs.pop("admin_key", ADMIN),
        **kwargs,
    )
    return engine, app


def client_for(app):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test", timeout=30.0
    )


def bearer(key):
    return {"Authorization": f"Bearer {key}"}


def chat_body(**over):
    body = {
        "model": "qwenfast-mock",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    body.update(over)
    return body


# ==========================================================================
# 1. The context clamp — task 2(a), and 9 of the 57 measured refusals
# ==========================================================================


@pytest.mark.asyncio
async def test_max_tokens_is_clamped_into_the_remaining_context_not_refused(tokenizer, tmp_path):
    """The exact shape of 9 of the 57: a tiny prompt with `max_tokens` == the whole context.

    Measured rows had prompts of 13-557 tokens and were refused because
    `prompt + 4096 > 4096`. There is nothing wrong with such a request: the
    only sane reading of `max_tokens` is a ceiling.
    """
    keys = tmp_path / "keys.json"
    write_keys(keys, entries())
    _, app = build(tokenizer, keys, max_context_len=64, min_completion_tokens=8)
    async with client_for(app) as c:
        r = await c.post(
            "/v1/chat/completions", json=chat_body(max_tokens=64), headers=bearer(PUBLIC)
        )
        assert r.status_code == 200, r.text
        used = r.json()["usage"]
        # Clamped to whatever the prompt left, and the header says what that was.
        assert used["completion_tokens"] <= 64 - used["prompt_tokens"]
        assert int(r.headers["X-Qwenfast-Max-Tokens"]) == 64 - used["prompt_tokens"]
        assert r.json()["choices"][0]["finish_reason"] == "length"


@pytest.mark.asyncio
async def test_a_request_that_already_fits_is_untouched(tokenizer, tmp_path):
    keys = tmp_path / "keys.json"
    write_keys(keys, entries())
    _, app = build(tokenizer, keys, max_context_len=4096, min_completion_tokens=16)
    async with client_for(app) as c:
        r = await c.post(
            "/v1/chat/completions", json=chat_body(max_tokens=5), headers=bearer(PUBLIC)
        )
        assert r.status_code == 200
        assert r.json()["usage"]["completion_tokens"] == 5
        assert r.headers["X-Qwenfast-Max-Tokens"] == "5"


@pytest.mark.asyncio
async def test_only_a_prompt_with_no_room_left_is_a_400(tokenizer, tmp_path):
    keys = tmp_path / "keys.json"
    write_keys(keys, entries())
    _, app = build(tokenizer, keys, max_context_len=48, min_completion_tokens=16)
    async with client_for(app) as c:
        r = await c.post(
            "/v1/chat/completions",
            json=chat_body(messages=[{"role": "user", "content": "word " * 400}]),
            headers=bearer(PUBLIC),
        )
        assert r.status_code == 400
        err = r.json()["error"]
        assert err["code"] == "context_length_exceeded"
        assert err["param"] == "messages"
        assert err["type"] == "invalid_request_error"
        # The message must state the numbers, so a caller knows how much to cut.
        assert "48 tokens" in err["message"]


@pytest.mark.asyncio
async def test_streaming_clamps_and_reports_length(tokenizer, tmp_path):
    keys = tmp_path / "keys.json"
    write_keys(keys, entries())
    _, app = build(tokenizer, keys, max_context_len=64, min_completion_tokens=8)
    async with client_for(app) as c:
        async with c.stream(
            "POST",
            "/v1/chat/completions",
            json=chat_body(max_tokens=999, stream=True),
            headers=bearer(PUBLIC),
        ) as r:
            assert r.status_code == 200
            text = "".join([chunk async for chunk in r.aiter_text()])
    assert '"finish_reason": "length"' in text.replace('"finish_reason":"length"', '"finish_reason": "length"')
    assert text.rstrip().endswith("data: [DONE]")


@pytest.mark.asyncio
async def test_the_prompt_cap_still_works_and_is_labelled_separately(tokenizer, tmp_path):
    """`--max-prompt-tokens` is retained, but reports as its own error class.

    47 of the 57 came from this cap. Keeping the flag means an operator who wants a hard
    ceiling still has one; the recommended production config drops it in favour of
    `--min-completion-tokens`, and the two must be distinguishable on the dashboard.
    """
    keys = tmp_path / "keys.json"
    write_keys(keys, entries())
    db = tmp_path / "usage.sqlite"
    _, app = build(
        tokenizer, keys, usage_db=str(db), max_prompt_tokens=8, max_context_len=4096
    )
    async with client_for(app) as c:
        r = await c.post(
            "/v1/chat/completions",
            json=chat_body(messages=[{"role": "user", "content": "word " * 200}]),
            headers=bearer(PUBLIC),
        )
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "context_length_exceeded"
        assert "accepts at most 8" in r.json()["error"]["message"]
        app.state.usage.flush()
    rows = sqlite3.connect(db).execute("SELECT status, error FROM usage").fetchall()
    assert rows == [(400, "prompt_too_long")]


# ==========================================================================
# 2. Error shape — task 2(a), the OpenAI envelope
# ==========================================================================


@pytest.mark.asyncio
async def test_every_refusal_uses_the_openai_envelope(tokenizer, tmp_path):
    keys = tmp_path / "keys.json"
    write_keys(keys, entries())
    _, app = build(tokenizer, keys, max_messages=2, max_request_bytes=400)
    async with client_for(app) as c:
        cases = [
            (await c.get("/v1/models"), 401),
            (await c.get("/admin/usage", headers=bearer(PUBLIC)), 403),
            (
                await c.post(
                    "/v1/chat/completions",
                    json=chat_body(messages=[{"role": "user", "content": "x"}] * 9),
                    headers=bearer(PUBLIC),
                ),
                400,
            ),
            (
                await c.post(
                    "/v1/chat/completions",
                    json=chat_body(messages=[{"role": "user", "content": "x" * 3000}]),
                    headers=bearer(PUBLIC),
                ),
                413,
            ),
        ]
    for res, expected in cases:
        assert res.status_code == expected, res.text
        body = res.json()
        assert set(body) == {"error"}, body
        for field in ("message", "type", "param", "code"):
            assert field in body["error"], (expected, body)
        assert body["error"]["message"]


@pytest.mark.asyncio
async def test_a_malformed_body_is_a_400_not_a_422(tokenizer, tmp_path):
    """FastAPI's default is 422 with `{"detail": [...]}`, which no OpenAI SDK understands."""
    keys = tmp_path / "keys.json"
    write_keys(keys, entries())
    db = tmp_path / "usage.sqlite"
    _, app = build(tokenizer, keys, usage_db=str(db))
    async with client_for(app) as c:
        r = await c.post(
            "/v1/chat/completions",
            content=b"{not json at all",
            headers={**bearer(PUBLIC), "Content-Type": "application/json"},
        )
        assert r.status_code == 400
        assert r.json()["error"]["type"] == "invalid_request_error"

        # A schema violation names the offending field.
        r2 = await c.post(
            "/v1/chat/completions", json={"model": "m"}, headers=bearer(PUBLIC)
        )
        assert r2.status_code == 400
        assert r2.json()["error"]["param"] == "messages"
        app.state.usage.flush()

    classes = [
        row[0]
        for row in sqlite3.connect(db).execute(
            "SELECT error FROM usage WHERE status = 400"
        )
    ]
    # Both were previously invisible: rejected before any handler ran, so nothing metered them.
    assert classes == ["invalid_request", "invalid_request"]


@pytest.mark.asyncio
async def test_too_many_messages_is_its_own_class(tokenizer, tmp_path):
    keys = tmp_path / "keys.json"
    write_keys(keys, entries())
    db = tmp_path / "usage.sqlite"
    _, app = build(tokenizer, keys, usage_db=str(db), max_messages=3)
    async with client_for(app) as c:
        r = await c.post(
            "/v1/chat/completions",
            json=chat_body(messages=[{"role": "user", "content": "hi"}] * 9),
            headers=bearer(PUBLIC),
        )
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "too_many_messages"
        app.state.usage.flush()
    rows = sqlite3.connect(db).execute("SELECT error FROM usage").fetchall()
    assert rows == [("too_many_messages",)]


# ==========================================================================
# 3. Fairness — per-key and per-IP concurrency
# ==========================================================================


def test_concurrency_limiter_is_all_or_nothing():
    """A refusal must not leave a half-taken slot behind, or the cap ratchets to zero."""
    lim = ConcurrencyLimiter(per_key=2, per_ip=1)
    a = ApiKey(secret="a", name="a")
    assert lim.acquire(a, "1.1.1.1").allowed
    refused = lim.acquire(a, "1.1.1.1")
    assert not refused.allowed and refused.scope == "ip"
    # The refused attempt must not have consumed the key's second slot.
    assert lim.acquire(a, "2.2.2.2").allowed
    assert not lim.acquire(a, "3.3.3.3").allowed  # now the *key* cap is full
    lim.release(a, "1.1.1.1")
    lim.release(a, "2.2.2.2")
    assert lim.snapshot()["in_flight_by_key"] == {}


def test_admin_keys_are_never_concurrency_capped():
    lim = ConcurrencyLimiter(per_key=1, per_ip=1)
    admin = ApiKey(secret="x", name="admin", admin=True)
    for _ in range(5):
        assert lim.acquire(admin, "1.1.1.1").allowed


def test_a_per_key_max_streams_overrides_the_server_default():
    lim = ConcurrencyLimiter(per_key=2)
    generous = ApiKey(secret="g", name="g", max_streams=4)
    assert lim.limit_for(generous) == 4
    assert lim.limit_for(ApiKey(secret="p", name="p")) == 2


@pytest.mark.asyncio
async def test_per_key_stream_cap_returns_429_with_retry_after(tokenizer, tmp_path):
    """One caller must not be able to sit in every sequence slot."""
    keys = tmp_path / "keys.json"
    write_keys(keys, entries())
    _, app = build(tokenizer, keys, tps=20.0, max_streams_per_key=1)

    async with client_for(app) as c:
        slow = asyncio.create_task(
            c.post(
                "/v1/chat/completions",
                json=chat_body(max_tokens=30),
                headers=bearer(PUBLIC),
            )
        )
        await asyncio.sleep(0.15)  # let it get past admission and into decode

        second = await c.post(
            "/v1/chat/completions", json=chat_body(), headers=bearer(PUBLIC)
        )
        assert second.status_code == 429
        assert second.headers["Retry-After"]
        assert second.json()["error"]["code"] == "concurrency_limit_exceeded"
        assert second.json()["error"]["type"] == "rate_limit_error"

        # A *different* key is unaffected: this is fairness, not a global cap.
        other = await c.post("/v1/chat/completions", json=chat_body(), headers=bearer(OTHER))
        assert other.status_code == 200

        assert (await asyncio.wait_for(slow, timeout=20)).status_code == 200

    # And the slot is given back, so the cap is not a one-shot.
    async with client_for(app) as c:
        assert (
            await c.post("/v1/chat/completions", json=chat_body(), headers=bearer(PUBLIC))
        ).status_code == 200


@pytest.mark.asyncio
async def test_per_ip_cap_uses_the_forwarded_header(tokenizer, tmp_path):
    """Behind Cloudflare every socket peer is the same address, so the header is the only signal."""
    keys = tmp_path / "keys.json"
    write_keys(keys, entries())
    _, app = build(tokenizer, keys, tps=20.0, max_streams_per_ip=1)

    async with client_for(app) as c:
        slow = asyncio.create_task(
            c.post(
                "/v1/chat/completions",
                json=chat_body(max_tokens=30),
                headers={**bearer(PUBLIC), "X-Forwarded-For": "9.9.9.9, 10.0.0.1"},
            )
        )
        await asyncio.sleep(0.15)

        same_ip = await c.post(
            "/v1/chat/completions",
            json=chat_body(),
            headers={**bearer(OTHER), "X-Forwarded-For": "9.9.9.9"},
        )
        assert same_ip.status_code == 429, "same IP, different key, must still be capped"

        other_ip = await c.post(
            "/v1/chat/completions",
            json=chat_body(),
            headers={**bearer(PUBLIC), "X-Forwarded-For": "8.8.8.8"},
        )
        assert other_ip.status_code == 200
        assert (await asyncio.wait_for(slow, timeout=20)).status_code == 200


# ==========================================================================
# 4. Timeouts and disconnects
# ==========================================================================


@pytest.mark.asyncio
async def test_a_slow_stream_is_cut_cleanly_by_the_request_timeout(tokenizer, tmp_path):
    keys = tmp_path / "keys.json"
    write_keys(keys, entries())
    db = tmp_path / "usage.sqlite"
    _, app = build(tokenizer, keys, usage_db=str(db), tps=20.0, request_timeout_s=0.25)
    async with client_for(app) as c:
        async with c.stream(
            "POST",
            "/v1/chat/completions",
            json=chat_body(max_tokens=500, stream=True, stream_options={"include_usage": True}),
            headers=bearer(PUBLIC),
        ) as r:
            assert r.status_code == 200
            text = "".join([chunk async for chunk in r.aiter_text()])

    # Cleanly terminated: a final chunk, a usage frame, and [DONE] — not a dropped socket.
    assert "[DONE]" in text
    assert "length" in text
    app.state.usage.flush()
    rows = sqlite3.connect(db).execute("SELECT status, error FROM usage").fetchall()
    assert rows == [(200, "request_timeout")]
    assert app.state.usage.errors_by_class()["request_timeout"] == 1


@pytest.mark.asyncio
async def test_a_non_streaming_timeout_is_a_504_in_the_openai_shape(tokenizer, tmp_path):
    keys = tmp_path / "keys.json"
    write_keys(keys, entries())
    _, app = build(tokenizer, keys, tps=20.0, request_timeout_s=0.25)
    async with client_for(app) as c:
        r = await c.post(
            "/v1/chat/completions", json=chat_body(max_tokens=500), headers=bearer(PUBLIC)
        )
    assert r.status_code == 504
    assert r.json()["error"]["code"] == "request_timeout"
    # Nothing was left holding a slot.
    assert app.state.inflight.current == 0


@pytest.mark.asyncio
async def test_a_client_disconnect_frees_the_admission_slot_immediately(tokenizer, tmp_path):
    """Task 3(b): confirm the pre-existing disconnect detection really releases the slot.

    The check runs every `DISCONNECT_CHECK_EVERY` steps, but the `finally` in the streaming
    generator is what actually frees the slot, and it runs as soon as Starlette closes the
    response — which is what this asserts.
    """
    keys = tmp_path / "keys.json"
    write_keys(keys, entries())
    engine, app = build(
        tokenizer, keys, usage_db=None, tps=40.0, max_inflight_requests=1
    )
    async with client_for(app) as c:
        async with c.stream(
            "POST",
            "/v1/chat/completions",
            json=chat_body(max_tokens=400, stream=True),
            headers=bearer(PUBLIC),
        ) as r:
            assert r.status_code == 200
            # Read one frame, then walk away mid-stream.
            async for _ in r.aiter_bytes():
                break

        # Give the generator's `finally` a beat to run.
        for _ in range(200):
            if app.state.inflight.current == 0:
                break
            await asyncio.sleep(0.01)

        assert app.state.inflight.current == 0, "admission slot leaked on disconnect"
        assert app.state.concurrency.snapshot()["in_flight_by_key"] == {}

        # The single slot is genuinely reusable.
        again = await c.post("/v1/chat/completions", json=chat_body(), headers=bearer(PUBLIC))
        assert again.status_code == 200


# ==========================================================================
# 5. Draining
# ==========================================================================


@pytest.mark.asyncio
async def test_draining_refuses_new_work_and_reports_unhealthy(tokenizer, tmp_path):
    keys = tmp_path / "keys.json"
    write_keys(keys, entries())
    db = tmp_path / "usage.sqlite"
    _, app = build(tokenizer, keys, usage_db=str(db))
    async with client_for(app) as c:
        assert (await c.get("/health")).status_code == 200

        app.state.drain.begin("test")

        health = await c.get("/health")
        assert health.status_code == 503
        assert health.json()["status"] == "draining"

        # /status agrees, and says so in a field a script can read.
        st = await c.get("/status")
        assert st.status_code == 503
        assert st.json()["draining"] is True

        r = await c.post("/v1/chat/completions", json=chat_body(), headers=bearer(PUBLIC))
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "server_draining"
        assert r.headers["Retry-After"]
        app.state.usage.flush()

    rows = sqlite3.connect(db).execute("SELECT error FROM usage").fetchall()
    assert rows == [("draining",)]


@pytest.mark.asyncio
async def test_an_in_flight_stream_survives_the_start_of_a_drain(tokenizer, tmp_path):
    """Draining stops *accepting*; it must not kill what is already running."""
    keys = tmp_path / "keys.json"
    write_keys(keys, entries())
    _, app = build(tokenizer, keys, tps=200.0)
    async with client_for(app) as c:
        async with c.stream(
            "POST",
            "/v1/chat/completions",
            json=chat_body(max_tokens=20, stream=True),
            headers=bearer(PUBLIC),
        ) as r:
            app.state.drain.begin("mid-stream")
            text = "".join([chunk async for chunk in r.aiter_text()])
    assert text.rstrip().endswith("data: [DONE]")
    assert '"stop"' in text or "length" in text


# ==========================================================================
# 6. Observability — /status, /v1/limits, /admin/usage
# ==========================================================================


@pytest.mark.asyncio
async def test_status_is_public_and_leaks_nothing(tokenizer, tmp_path):
    keys = tmp_path / "keys.json"
    write_keys(keys, entries())
    _, app = build(
        tokenizer, keys, max_context_len=4096, max_inflight_requests=64,
        max_streams_per_key=8, content_log_db=str(tmp_path / "q.sqlite"),
    )
    async with client_for(app) as c:
        r = await c.get("/status")  # no key at all
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["model"] == "qwenfast-mock"
        assert body["load"]["capacity"] == 64
        assert body["limits"]["max_context_length"] == 4096
        assert body["limits"]["max_concurrent_streams_per_key"] == 8

        # Nothing about keys, cost, usage totals, or the content log. (The
        # `..._per_key` limit *names* are fine — they are deployment constants;
        # what must not appear is per-key usage, cost, or the content log.)
        text = r.text.lower()
        for forbidden in ("cost", "inr", "usd", "key_store", "content_log",
                          "queries", "\"per_key\"", "errors_by_class"):
            assert forbidden not in text, forbidden
        assert PUBLIC not in r.text and ADMIN not in r.text


@pytest.mark.asyncio
async def test_v1_limits_tells_a_client_the_envelope(tokenizer, tmp_path):
    keys = tmp_path / "keys.json"
    write_keys(keys, [{"key": PUBLIC, "name": "public", "rpm": 60, "max_tokens": 2048,
                       "max_streams": 4}])
    _, app = build(
        tokenizer, keys, max_context_len=4096, min_completion_tokens=256,
        max_output_tokens=4096, max_messages=128, max_streams_per_key=8,
    )
    async with client_for(app) as c:
        assert (await c.get("/v1/limits")).status_code == 401
        r = await c.get("/v1/limits", headers=bearer(PUBLIC))
        assert r.status_code == 200
        body = r.json()
        assert body["max_context_length"] == 4096
        assert body["min_completion_tokens"] == 256
        assert body["max_tokens_is_clamped"] is True
        assert body["key"]["name"] == "public"
        assert body["key"]["max_tokens"] == 2048
        assert body["key"]["max_concurrent_streams"] == 4  # per-key override, not the server 8
        assert PUBLIC not in r.text


@pytest.mark.asyncio
async def test_models_carries_the_context_window(tokenizer, tmp_path):
    keys = tmp_path / "keys.json"
    write_keys(keys, entries())
    _, app = build(tokenizer, keys, max_context_len=8192)
    async with client_for(app) as c:
        body = (await c.get("/v1/models", headers=bearer(PUBLIC))).json()
    assert body["data"][0]["context_window"] == 8192


@pytest.mark.asyncio
async def test_admin_usage_breaks_errors_down_by_class(tokenizer, tmp_path):
    """The headline deliverable: "57 errors" becomes a breakdown you can act on."""
    keys = tmp_path / "keys.json"
    write_keys(keys, entries())
    db = tmp_path / "usage.sqlite"
    _, app = build(
        tokenizer, keys, usage_db=str(db), max_context_len=64,
        min_completion_tokens=8, max_messages=3,
    )
    async with client_for(app) as c:
        # one of each of the three shapes seen in production
        await c.post(
            "/v1/chat/completions",
            json=chat_body(messages=[{"role": "user", "content": "word " * 400}]),
            headers=bearer(PUBLIC),
        )
        await c.post(
            "/v1/chat/completions",
            json=chat_body(messages=[{"role": "user", "content": "hi"}] * 9),
            headers=bearer(PUBLIC),
        )
        await c.post("/v1/chat/completions", json=chat_body(), headers=bearer(PUBLIC))
        app.state.usage.flush()

        body = (await c.get("/admin/usage", headers=bearer(ADMIN))).json()

    classes = body["errors_by_class"]
    assert classes["context_length_exceeded"] == 1
    assert classes["too_many_messages"] == 1
    assert body["totals"]["requests"] == 3
    assert body["totals"]["errors"] == 2

    # p50/p99 latency, in-flight, and the queue-reject breakdown are all present.
    assert body["latency"]["ttft_ms"]["p50"] is not None
    assert "p99" in body["latency"]["tpot_ms"]
    assert body["capacity"]["inflight"] == 0
    assert set(body["capacity"]["queue_rejects"]) == {
        "overload_503", "concurrency_429", "rate_limited_429"
    }
    assert body["limits"]["max_context_length"] == 64


def test_percentiles_are_nearest_rank():
    assert percentiles([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == {"p50": 5, "p90": 9, "p99": 10}
    assert percentiles([]) == {"p50": None, "p90": None, "p99": None}
    assert percentiles([42.0])["p99"] == 42.0


def test_usage_db_migrates_in_place_without_losing_history(tmp_path):
    """The production database is 11 hours of traffic the cost model is derived from.

    `service_started_at` is `MIN(ts)`, so a schema change that started a fresh file would
    silently reset the cost-per-token figure on the dashboard.
    """
    db = tmp_path / "usage.sqlite"
    # Build the v1 schema by hand — no `error` column.
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, key_name TEXT NOT NULL,
            endpoint TEXT NOT NULL, prompt_tokens INTEGER NOT NULL DEFAULT 0,
            completion_tokens INTEGER NOT NULL DEFAULT 0, ttft_ms REAL, duration_ms REAL,
            status INTEGER NOT NULL, stream INTEGER NOT NULL DEFAULT 0, finish_reason TEXT
        );
        CREATE TABLE meta (k TEXT PRIMARY KEY, v TEXT);
        """
    )
    old_ts = time.time() - 40_000
    conn.execute(
        "INSERT INTO usage (ts, key_name, endpoint, prompt_tokens, completion_tokens,"
        " ttft_ms, duration_ms, status, stream) VALUES (?,?,?,?,?,?,?,?,?)",
        (old_ts, "public", "chat.completions", 100, 200, 50.0, 500.0, 200, 1),
    )
    conn.execute(
        "INSERT INTO usage (ts, key_name, endpoint, prompt_tokens, completion_tokens,"
        " ttft_ms, duration_ms, status, stream) VALUES (?,?,?,?,?,?,?,?,?)",
        (old_ts + 1, "public", "chat.completions", 9000, 0, None, 10.0, 400, 1),
    )
    conn.commit()
    conn.close()

    rec = UsageRecorder(str(db), start_writer=False)
    assert rec.requests == 2
    assert rec.completion_tokens == 200
    assert abs(rec.service_started_at - old_ts) < 1.0, "cost baseline must survive"
    # The pre-existing 400 has no reason recorded, and says so rather than guessing.
    assert rec.errors_by_class() == {"unclassified": 1}

    # New rows land in the new column alongside the old ones.
    rec.record(
        UsageRecord(
            ts=time.time(), key_name="public", endpoint="chat.completions",
            prompt_tokens=10, completion_tokens=0, ttft_ms=None, duration_ms=1.0,
            status=400, stream=False, error="context_length_exceeded",
        )
    )
    assert rec.errors_by_class()["context_length_exceeded"] == 1


# ==========================================================================
# 7. Content logging (opt-in; private to the GPU host)
# ==========================================================================


@pytest.mark.asyncio
async def test_content_logging_is_off_unless_asked_for(tokenizer, tmp_path):
    keys = tmp_path / "keys.json"
    write_keys(keys, entries())
    _, app = build(tokenizer, keys)
    async with client_for(app) as c:
        await c.post("/v1/chat/completions", json=chat_body(), headers=bearer(PUBLIC))
        assert app.state.query_log.enabled is False
        r = await c.get("/admin/queries", headers=bearer(ADMIN))
        assert r.json() == {
            "enabled": False,
            "detail": "content logging is off (--log-content)",
        }


@pytest.mark.asyncio
async def test_content_log_records_prompt_and_reply(tokenizer, tmp_path):
    keys = tmp_path / "keys.json"
    write_keys(keys, entries())
    qdb = tmp_path / "queries.sqlite"
    _, app = build(tokenizer, keys, content_log_db=str(qdb))
    async with client_for(app) as c:
        await c.post(
            "/v1/chat/completions",
            json=chat_body(messages=[{"role": "user", "content": "how tall is everest"}]),
            headers={**bearer(PUBLIC), "X-Forwarded-For": "203.0.113.9"},
        )
        # ...and a streamed one, which accumulates its text a delta at a time.
        async with c.stream(
            "POST",
            "/v1/chat/completions",
            json=chat_body(messages=[{"role": "user", "content": "streamed question"}],
                           stream=True),
            headers=bearer(PUBLIC),
        ) as r:
            async for _ in r.aiter_bytes():
                pass
        app.state.query_log.flush()

        rows = app.state.query_log.export(limit=10)
    assert len(rows) == 2
    prompts = " ".join(r["messages_json"] for r in rows)
    assert "how tall is everest" in prompts
    assert "streamed question" in prompts
    assert all(r["response_text"] for r in rows), "the reply must be captured too"
    assert any(r["client_ip"] == "203.0.113.9" for r in rows)
    # Key *material* must never be written; the name is what identifies the caller.
    assert all(PUBLIC not in json.dumps(r) for r in rows)
    assert all(r["key_name"] == "public" for r in rows)


@pytest.mark.asyncio
async def test_admin_queries_exports_and_counts(tokenizer, tmp_path):
    keys = tmp_path / "keys.json"
    write_keys(keys, entries())
    qdb = tmp_path / "queries.sqlite"
    _, app = build(tokenizer, keys, content_log_db=str(qdb))
    async with client_for(app) as c:
        for who, text in ((PUBLIC, "alpha question"), (PUBLIC, "beta"), (OTHER, "gamma")):
            await c.post(
                "/v1/chat/completions",
                json=chat_body(messages=[{"role": "user", "content": text}]),
                headers=bearer(who),
            )
        app.state.query_log.flush()

        # Admin only.
        assert (await c.get("/admin/queries")).status_code == 401
        assert (await c.get("/admin/queries", headers=bearer(PUBLIC))).status_code == 403

        body = (await c.get("/admin/queries?limit=10", headers=bearer(ADMIN))).json()
        assert body["count"] == 3

        csv_text = (
            await c.get("/admin/queries?format=csv", headers=bearer(ADMIN))
        ).text
        assert csv_text.startswith("id,ts,key_name,client_ip")
        assert "alpha question" in csv_text

        jsonl = (await c.get("/admin/queries?format=jsonl", headers=bearer(ADMIN))).text
        assert len(jsonl.strip().splitlines()) == 3

        # The abuse-spotting view.
        counts = (
            await c.get("/admin/queries?counts=key", headers=bearer(ADMIN))
        ).json()["counts"]
        by_name = {c_["key_name"]: c_["queries"] for c_ in counts}
        assert by_name == {"public": 2, "other": 1}

        # And the regex filter.
        hits = (
            await c.get("/admin/queries?grep=alpha", headers=bearer(ADMIN))
        ).json()
        assert hits["count"] == 1
        bad = await c.get("/admin/queries?grep=%5B", headers=bearer(ADMIN))
        assert bad.status_code == 400


def test_content_log_retention_prunes_old_rows(tmp_path):
    qdb = tmp_path / "queries.sqlite"
    log = QueryLogger(str(qdb), retention_days=1.0, start_writer=False)
    conn = sqlite3.connect(qdb)
    now = time.time()
    for ts in (now - 3 * 86400, now - 2 * 86400, now - 60):
        conn.execute(
            "INSERT INTO queries (ts, key_name, endpoint, messages_json, response_text,"
            " prompt_tokens, completion_tokens, status) VALUES (?,?,?,?,?,?,?,?)",
            (ts, "public", "chat.completions", "[]", "hi", 1, 1, 200),
        )
    conn.commit()
    conn.close()

    assert log.prune() == 2
    assert len(log.export(limit=10)) == 1


def test_content_log_never_raises_on_a_bad_path(tmp_path):
    """Metering rules apply here too: a broken database must not fail a response."""
    log = QueryLogger(str(tmp_path / "nope" / "x" / "\0bad"), start_writer=False)
    assert log.enabled is False
    log.record(
        QueryRecordFactory()
    )  # no-op, must not raise
    assert log.export() == []


def QueryRecordFactory():
    from qwenfast.server.content_log import QueryRecord

    return QueryRecord(
        ts=time.time(), key_name="k", client_ip="-", endpoint="chat.completions",
        model="m", messages_json="[]", response_text="", reasoning_text=None,
        prompt_tokens=0, completion_tokens=0, status=200, finish_reason=None,
        ttft_ms=None, duration_ms=None, request_id="r",
    )
