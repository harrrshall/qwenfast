"""Tests for the public-endpoint layer: auth, rate limits, caps, metering, `/admin/usage`.

Everything here runs against `MockEngine` on CPU — no GPU, no weights, no network. Run with:

    PYTHONPATH=engine <venv>/bin/pytest engine/qwenfast/server/tests/test_public_api.py -v
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time

import httpx
import pytest

from qwenfast.server.app import create_app
from qwenfast.server.auth import ApiKey, KeyStore, RateLimiter, parse_keys_document
from qwenfast.server.mock_engine import MockEngine
from qwenfast.server.usage import UsageRecord, UsageRecorder

PUBLIC = "qf-public-000000000000000000000000"
LIMITED = "qf-limited-00000000000000000000000"
ADMIN = "qf-admin-0000000000000000000000000"
DEMO = "qf-demo-00000000000000000000000000"


def write_keys(path, entries) -> None:
    path.write_text(json.dumps({"keys": entries}))


def default_entries():
    return [
        {"key": PUBLIC, "name": "public"},
        {"key": LIMITED, "name": "limited", "rpm": 2, "tpm": 50, "max_tokens": 4},
    ]


def build(tokenizer, keys_path=None, **kwargs):
    engine = MockEngine(
        tokenizer=tokenizer,
        decode_tokens_per_second=kwargs.pop("tps", 4000.0),
        prefill_base_delay_s=0.001,
    )
    app = create_app(
        engine,
        tokenizer,
        model_name="qwenfast-mock",
        api_keys_file=str(keys_path) if keys_path else None,
        admin_key=kwargs.pop("admin_key", ADMIN),
        demo_key=kwargs.pop("demo_key", DEMO),
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


# --------------------------------------------------------------------------
# 1. auth
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_and_bad_keys_are_401(tokenizer, tmp_path):
    keys = tmp_path / "keys.json"
    write_keys(keys, default_entries())
    _, app = build(tokenizer, keys)
    async with client_for(app) as c:
        assert (await c.get("/v1/models")).status_code == 401
        assert (await c.get("/v1/models", headers=bearer("qf-nope"))).status_code == 401
        r = await c.post("/v1/chat/completions", json=chat_body())
        assert r.status_code == 401
        # The 401 body must not echo anything key-shaped back.
        assert PUBLIC not in r.text and ADMIN not in r.text
        ok = await c.get("/v1/models", headers=bearer(PUBLIC))
        assert ok.status_code == 200
        assert ok.json()["data"][0]["id"] == "qwenfast-mock"


@pytest.mark.asyncio
async def test_health_and_metrics_stay_unauthenticated(tokenizer, tmp_path):
    keys = tmp_path / "keys.json"
    write_keys(keys, default_entries())
    _, app = build(tokenizer, keys)
    async with client_for(app) as c:
        assert (await c.get("/health")).status_code == 200
        m = await c.get("/metrics")
        assert m.status_code == 200
        assert "qwenfast:api_requests_total" in m.text
        assert "qwenfast:api_keys_loaded" in m.text
        # 4 keys: two from the file plus the static admin + demo keys.
        assert "qwenfast:api_keys_loaded 4" in m.text


@pytest.mark.asyncio
async def test_disabled_key_is_rejected(tokenizer, tmp_path):
    keys = tmp_path / "keys.json"
    write_keys(keys, [{"key": PUBLIC, "name": "public", "disabled": True}])
    _, app = build(tokenizer, keys)
    async with client_for(app) as c:
        assert (await c.get("/v1/models", headers=bearer(PUBLIC))).status_code == 401


@pytest.mark.asyncio
async def test_open_endpoint_when_no_auth_configured(tokenizer):
    _, app = build(tokenizer, None, admin_key=None, demo_key=None)
    async with client_for(app) as c:
        assert (await c.get("/v1/models")).status_code == 200
        assert (await c.post("/v1/chat/completions", json=chat_body())).status_code == 200


# --------------------------------------------------------------------------
# 2. admin
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_admin_usage_requires_admin_key(tokenizer, tmp_path):
    keys = tmp_path / "keys.json"
    write_keys(keys, default_entries())
    _, app = build(tokenizer, keys)
    async with client_for(app) as c:
        assert (await c.get("/admin/usage")).status_code == 401
        assert (await c.get("/admin/usage", headers=bearer(PUBLIC))).status_code == 403
        r = await c.get("/admin/usage", headers=bearer(ADMIN))
        assert r.status_code == 200
        body = r.json()
        assert body["totals"]["requests"] == 0
        assert {k["name"] for k in body["keys"]} == {"public", "limited", "admin", "demo"}
        # Never leak key material through the admin view either.
        assert PUBLIC not in r.text and ADMIN not in r.text


@pytest.mark.asyncio
async def test_admin_usage_math_and_cost(tokenizer, tmp_path):
    keys = tmp_path / "keys.json"
    write_keys(keys, default_entries())
    _, app = build(tokenizer, keys, gpu_rate_inr_per_hour=188.73, inr_per_usd=87.5)
    async with client_for(app) as c:
        for _ in range(3):
            r = await c.post("/v1/chat/completions", json=chat_body(max_tokens=6), headers=bearer(PUBLIC))
            assert r.status_code == 200
            assert r.json()["usage"]["completion_tokens"] == 6
        body = (await c.get("/admin/usage", headers=bearer(ADMIN))).json()

    assert body["totals"]["requests"] == 3
    assert body["totals"]["completion_tokens"] == 18
    assert body["totals"]["total_tokens"] == (
        body["totals"]["prompt_tokens"] + body["totals"]["completion_tokens"]
    )
    per_key = {k["name"]: k for k in body["per_key"]}
    assert per_key["public"]["requests"] == 3
    assert per_key["public"]["completion_tokens"] == 18

    cost = body["cost"]
    assert cost["gpu_rate_inr_per_hour"] == 188.73
    # cost = billed_hours x rate, and $ = INR / FX.
    assert cost["inr"] == pytest.approx(cost["billed_hours"] * 188.73, rel=1e-9)
    assert cost["usd"] == pytest.approx(cost["inr"] / 87.5, rel=1e-9)
    # per-1M-output = total INR scaled by (1e6 / completion tokens).
    assert cost["per_1m_output_tokens_inr"] == pytest.approx(cost["inr"] / 18 * 1e6, rel=1e-9)
    assert len(body["hourly"]) == 24
    assert sum(h["completion_tokens"] for h in body["hourly"]) == 18
    assert body["engine_health"] == "ok"


# --------------------------------------------------------------------------
# 3. rate limiting
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rpm_limit_returns_429_with_retry_after(tokenizer, tmp_path):
    keys = tmp_path / "keys.json"
    write_keys(keys, default_entries())
    _, app = build(tokenizer, keys)
    async with client_for(app) as c:
        assert (await c.post("/v1/chat/completions", json=chat_body(max_tokens=1), headers=bearer(LIMITED))).status_code == 200
        assert (await c.post("/v1/chat/completions", json=chat_body(max_tokens=1), headers=bearer(LIMITED))).status_code == 200
        r = await c.post("/v1/chat/completions", json=chat_body(max_tokens=1), headers=bearer(LIMITED))
        assert r.status_code == 429
        assert 1 <= int(r.headers["Retry-After"]) <= 60
        assert "request rate limit" in r.text
        # A different key is unaffected — the window is per key.
        assert (await c.post("/v1/chat/completions", json=chat_body(max_tokens=1), headers=bearer(PUBLIC))).status_code == 200
        body = (await c.get("/admin/usage", headers=bearer(ADMIN))).json()
        assert body["totals"]["rate_limited"] == 1
        assert {k["name"]: k["errors"] for k in body["per_key"]}["limited"] == 1


def test_tpm_limit_locks_out_after_the_tokens_land():
    """Tokens are only known post-hoc, so the limiter must admit, then refuse the *next* one."""
    now = [1000.0]
    limiter = RateLimiter(clock=lambda: now[0])
    key = ApiKey(secret="s", name="k", tpm=100)

    assert limiter.check_and_admit(key).allowed
    limiter.record_tokens("k", 150)  # one big response blows the budget
    d = limiter.check_and_admit(key)
    assert not d.allowed and "token rate limit" in d.reason
    assert d.retry_after_s == pytest.approx(60.0, abs=0.01)

    now[0] += 61.0  # the window slides past it
    assert limiter.check_and_admit(key).allowed


def test_rpm_window_slides():
    now = [0.0]
    limiter = RateLimiter(clock=lambda: now[0])
    key = ApiKey(secret="s", name="k", rpm=2)
    assert limiter.check_and_admit(key).allowed
    now[0] += 30
    assert limiter.check_and_admit(key).allowed
    assert not limiter.check_and_admit(key).allowed
    now[0] += 31  # first request falls out of the window
    assert limiter.check_and_admit(key).allowed


# --------------------------------------------------------------------------
# 4. request caps
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_tokens_is_clamped_per_key_and_server_wide(tokenizer, tmp_path):
    keys = tmp_path / "keys.json"
    write_keys(keys, default_entries())
    _, app = build(tokenizer, keys, max_output_tokens=16)
    async with client_for(app) as c:
        # per-key cap of 4 wins over the request's 100
        r = await c.post("/v1/chat/completions", json=chat_body(max_tokens=100), headers=bearer(LIMITED))
        assert r.json()["usage"]["completion_tokens"] == 4
        # the unlimited key still gets the server-wide 16
        r = await c.post("/v1/chat/completions", json=chat_body(max_tokens=100), headers=bearer(PUBLIC))
        assert r.json()["usage"]["completion_tokens"] == 16
        # and a request below both caps is untouched
        r = await c.post("/v1/chat/completions", json=chat_body(max_tokens=3), headers=bearer(PUBLIC))
        assert r.json()["usage"]["completion_tokens"] == 3


@pytest.mark.asyncio
async def test_prompt_and_message_caps(tokenizer, tmp_path):
    keys = tmp_path / "keys.json"
    write_keys(keys, default_entries())
    _, app = build(tokenizer, keys, max_prompt_tokens=8, max_messages=3)
    async with client_for(app) as c:
        long_prompt = chat_body(messages=[{"role": "user", "content": "word " * 200}])
        r = await c.post("/v1/chat/completions", json=long_prompt, headers=bearer(PUBLIC))
        assert r.status_code == 400
        assert "accepts at most 8" in r.text

        many = chat_body(messages=[{"role": "user", "content": "hi"} for _ in range(9)])
        r = await c.post("/v1/chat/completions", json=many, headers=bearer(PUBLIC))
        assert r.status_code == 400
        assert "too many messages" in r.text


@pytest.mark.asyncio
async def test_oversized_body_is_413(tokenizer, tmp_path):
    keys = tmp_path / "keys.json"
    write_keys(keys, default_entries())
    _, app = build(tokenizer, keys, max_request_bytes=512)
    async with client_for(app) as c:
        big = chat_body(messages=[{"role": "user", "content": "x" * 5000}])
        r = await c.post("/v1/chat/completions", json=big, headers=bearer(PUBLIC))
        assert r.status_code == 413
        assert "max 512" in r.text


# --------------------------------------------------------------------------
# 5. overload guard
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_queue_cap_returns_503(tokenizer, tmp_path):
    keys = tmp_path / "keys.json"
    write_keys(keys, default_entries())
    # One slow request fills the single admission slot; a second arriving while it decodes must
    # be refused immediately rather than queued behind it.
    _, app = build(tokenizer, keys, max_inflight_requests=1, tps=20.0)
    async with client_for(app) as c:
        slow = asyncio.create_task(
            c.post("/v1/chat/completions", json=chat_body(max_tokens=30), headers=bearer(PUBLIC))
        )
        await asyncio.sleep(0.15)  # let it get past admission and into decode

        r = await c.post("/v1/chat/completions", json=chat_body(max_tokens=1), headers=bearer(PUBLIC))
        assert r.status_code == 503
        assert "at capacity" in r.text
        assert r.headers["Retry-After"] == "5"

        assert (await asyncio.wait_for(slow, timeout=20)).status_code == 200

        # Slot released: the next request is admitted again.
        r = await c.post("/v1/chat/completions", json=chat_body(max_tokens=1), headers=bearer(PUBLIC))
        assert r.status_code == 200
        body = (await c.get("/admin/usage", headers=bearer(ADMIN))).json()
        assert body["totals"]["rejected_overload"] == 1
        assert body["capacity"]["capacity"] == 1


# --------------------------------------------------------------------------
# 6. metering
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_usage_rows_are_written_to_sqlite(tokenizer, tmp_path):
    keys = tmp_path / "keys.json"
    write_keys(keys, default_entries())
    db = tmp_path / "usage.sqlite"
    _, app = build(tokenizer, keys, usage_db=str(db))
    async with client_for(app) as c:
        await c.post("/v1/chat/completions", json=chat_body(max_tokens=5), headers=bearer(PUBLIC))
        async with c.stream(
            "POST",
            "/v1/chat/completions",
            json=chat_body(max_tokens=7, stream=True),
            headers=bearer(PUBLIC),
        ) as resp:
            async for _ in resp.aiter_lines():
                pass
        await c.post("/v1/completions", json={"model": "m", "prompt": "hi", "max_tokens": 3}, headers=bearer(PUBLIC))
        app.state.usage.flush()

    rows = sqlite3.connect(db).execute(
        "SELECT key_name, endpoint, completion_tokens, status, stream, ttft_ms, duration_ms"
        " FROM usage ORDER BY id"
    ).fetchall()
    assert len(rows) == 3
    assert [r[0] for r in rows] == ["public", "public", "public"]
    assert [r[1] for r in rows] == ["chat.completions", "chat.completions", "completions"]
    assert [r[2] for r in rows] == [5, 7, 3]
    assert all(r[3] == 200 for r in rows)
    assert [r[4] for r in rows] == [0, 1, 0]
    assert all(r[5] is not None and r[5] >= 0 for r in rows)  # ttft recorded
    assert all(r[6] > 0 for r in rows)


@pytest.mark.asyncio
async def test_errors_are_metered_too(tokenizer, tmp_path):
    keys = tmp_path / "keys.json"
    write_keys(keys, default_entries())
    db = tmp_path / "usage.sqlite"
    _, app = build(tokenizer, keys, usage_db=str(db), max_prompt_tokens=4)
    async with client_for(app) as c:
        r = await c.post(
            "/v1/chat/completions",
            json=chat_body(messages=[{"role": "user", "content": "word " * 200}]),
            headers=bearer(PUBLIC),
        )
        assert r.status_code == 400
        app.state.usage.flush()
    rows = sqlite3.connect(db).execute("SELECT key_name, status FROM usage").fetchall()
    assert rows == [("public", 400)]


def test_usage_totals_survive_a_restart(tmp_path):
    """The watchdog restarts the server on death; the dashboard's headline must not reset."""
    db = tmp_path / "usage.sqlite"
    first = UsageRecorder(str(db))
    t0 = time.time() - 3600
    for i in range(5):
        first.record(
            UsageRecord(
                ts=t0 + i, key_name="alice", endpoint="chat.completions",
                prompt_tokens=10, completion_tokens=100, ttft_ms=12.0, duration_ms=900.0, status=200,
            )
        )
    first.flush()
    first.shutdown()

    second = UsageRecorder(str(db))
    s = second.summary()
    assert s["totals"]["requests"] == 5
    assert s["totals"]["completion_tokens"] == 500
    assert s["totals"]["prompt_tokens"] == 50
    assert s["per_key"][0]["name"] == "alice"
    # Service uptime is measured from the first row ever, not from this process's start.
    assert s["service_uptime_s"] > 3500
    assert s["cost"]["inr"] > 0
    second.shutdown()


def test_recorder_never_raises_on_a_bad_db_path(tmp_path):
    bad = tmp_path / "nope" / "\0" / "usage.sqlite"
    rec = UsageRecorder(str(bad))
    rec.record(
        UsageRecord(ts=time.time(), key_name="k", endpoint="e", prompt_tokens=1,
                    completion_tokens=2, ttft_ms=None, duration_ms=1.0, status=200)
    )
    assert rec.summary()["totals"]["completion_tokens"] == 2
    assert rec.summary()["storage"]["error"] is not None


# --------------------------------------------------------------------------
# 7. key hot-reload
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_keys_file_hot_reload(tokenizer, tmp_path):
    keys = tmp_path / "keys.json"
    write_keys(keys, [{"key": PUBLIC, "name": "public"}])
    _, app = build(tokenizer, keys)
    store: KeyStore = app.state.key_store
    store._poll_interval_s = 0.0  # poll on every request in the test

    async with client_for(app) as c:
        assert (await c.get("/v1/models", headers=bearer(PUBLIC))).status_code == 200
        assert (await c.get("/v1/models", headers=bearer(LIMITED))).status_code == 401

        # Adding a key takes effect without a restart...
        write_keys(keys, [{"key": PUBLIC, "name": "public"}, {"key": LIMITED, "name": "limited"}])
        assert (await c.get("/v1/models", headers=bearer(LIMITED))).status_code == 200

        # ...and so does revoking one.
        write_keys(keys, [{"key": LIMITED, "name": "limited"}])
        assert (await c.get("/v1/models", headers=bearer(PUBLIC))).status_code == 401
        assert (await c.get("/v1/models", headers=bearer(LIMITED))).status_code == 200

        # A broken edit must NOT lock everyone out: the last good set keeps serving.
        keys.write_text("{ this is not json")
        assert (await c.get("/v1/models", headers=bearer(LIMITED))).status_code == 200
        body = (await c.get("/admin/usage", headers=bearer(ADMIN))).json()
        assert body["key_store"]["error"] is not None

        # Fixing the file clears the error.
        write_keys(keys, [{"key": LIMITED, "name": "limited"}, {"key": PUBLIC, "name": "public"}])
        assert (await c.get("/v1/models", headers=bearer(PUBLIC))).status_code == 200
        body = (await c.get("/admin/usage", headers=bearer(ADMIN))).json()
        assert body["key_store"]["error"] is None


@pytest.mark.asyncio
async def test_sighup_triggers_reload(tokenizer, tmp_path):
    keys = tmp_path / "keys.json"
    write_keys(keys, [{"key": PUBLIC, "name": "public"}])
    _, app = build(tokenizer, keys)
    store: KeyStore = app.state.key_store
    store._poll_interval_s = 3600.0  # mtime polling is effectively off

    async with client_for(app) as c:
        write_keys(keys, [{"key": PUBLIC, "name": "public"}, {"key": LIMITED, "name": "limited"}])
        assert (await c.get("/v1/models", headers=bearer(LIMITED))).status_code == 401
        store.request_reload()  # what the SIGHUP handler does
        assert (await c.get("/v1/models", headers=bearer(LIMITED))).status_code == 200


def test_keys_document_validation():
    with pytest.raises(ValueError):
        parse_keys_document({"keys": [{"name": "no-key"}]})
    with pytest.raises(ValueError):
        parse_keys_document({"keys": [{"key": "a", "name": "x"}, {"key": "a", "name": "y"}]})
    with pytest.raises(ValueError):
        parse_keys_document({"keys": [{"key": "a", "name": "x"}, {"key": "b", "name": "x"}]})
    with pytest.raises(ValueError):
        parse_keys_document({"keys": "not-a-list"})
    ok = parse_keys_document(
        {"keys": [{"key": "a", "name": "x", "rpm": 60, "tpm": 0, "max_tokens": None}]}
    )
    assert ok[0].rpm == 60
    assert ok[0].tpm is None  # 0 means "no limit"
    assert ok[0].max_tokens is None


# --------------------------------------------------------------------------
# 8. the demo key
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_demo_key_is_metered_separately(tokenizer, tmp_path):
    keys = tmp_path / "keys.json"
    write_keys(keys, default_entries())
    _, app = build(tokenizer, keys)
    async with client_for(app) as c:
        assert (await c.post("/v1/chat/completions", json=chat_body(max_tokens=2), headers=bearer(DEMO))).status_code == 200
        body = (await c.get("/admin/usage", headers=bearer(ADMIN))).json()
        per_key = {k["name"]: k for k in body["per_key"]}
        assert per_key["demo"]["completion_tokens"] == 2
        assert "public" not in per_key  # the demo traffic is not attributed to a public key
    # The demo key is not an admin key.
    async with client_for(app) as c:
        assert (await c.get("/admin/usage", headers=bearer(DEMO))).status_code == 403
