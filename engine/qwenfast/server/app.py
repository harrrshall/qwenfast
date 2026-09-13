"""OpenAI-compatible HTTP server for `qwenfast`.

Framework choice: **FastAPI on top of Starlette/uvicorn** (`--http h11`). Justification against the
obvious alternative, aiohttp:

* Per-token streaming overhead is dominated by (a) the SSE `data: {json}\\n\\n` framing/encode and
  (b) the write syscall into the ASGI transport -- both are essentially identical in cost between
  Starlette's `StreamingResponse` and an aiohttp `web.StreamResponse`; neither library does
  anything expensive in the hot per-chunk path. What differs is everything *around* that path:
  request validation, auth, and OpenAI's fairly involved (and blissfully repetitive across
  `/v1/completions` and `/v1/chat/completions`) request schema. FastAPI/Pydantic pay that cost
  once at request-parse time, not per token, so they don't touch the streaming hot loop at all.
* uvicorn's `h11` HTTP implementation (a pure-Python, RFC-conformant HTTP/1.1 parser) is what the
  server pins (`--http h11`) over the C-accelerated `httptools`: h11 has cleaner backpressure
  and chunked-transfer semantics under long-lived SSE streams, and our bottleneck is the model
  (milliseconds/token) not HTTP parsing (microseconds/request) -- so the parser's raw throughput is
  irrelevant and its correctness/robustness under many concurrent long streams (this server's
  actual workload, with `max_num_seqs` up to 512) wins.
* aiohttp is a fine choice too (`benchmarks/mock_server.py`, a *test double*, uses it for exactly
  that reason: zero-dependency, minimal). We didn't pick it for the real server because it would
  mean hand-rolling the request-body validation FastAPI/Pydantic give for free, for no measurable
  gain on the metric that matters here (tokens/s and TTFT/TPOT, not requests/s of tiny bodies).

Everything below is framework glue around three pure, testable pieces: `tokenization.py`
(detokenization + think/tool-call parsing), `engine_api.py`/`mock_engine.py` (token generation),
and `metrics.py` (the `/metrics` renderer).
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from .auth import ApiKey, ConcurrencyLimiter, KeyStore, RateLimiter
from .engine_api import AsyncEngine, RequestStats, SamplingParams
from .content_log import QueryLogger, QueryRecord, messages_to_json
from .errors import ApiError, error_class_of, install_error_handlers
from .metrics import render_prometheus_text
from .protocol import ChatCompletionRequest, CompletionRequest
from .usage import (
    DEFAULT_GPU_RATE_INR_PER_HOUR,
    DEFAULT_INR_PER_USD,
    UsageRecord,
    UsageRecorder,
)
from .tokenization import (
    IncrementalDetokenizer,
    StopStringMatcher,
    ThinkTagParser,
    ToolCallDelta,
    ToolCallStreamParser,
    render_chat_prompt,
)

# Sampling-parameter recommendations from the model card, applied
# whenever the request doesn't specify a field itself.
THINKING_DEFAULTS: dict[str, float] = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
}
NON_THINKING_DEFAULTS: dict[str, float] = {
    "temperature": 0.7,
    "top_p": 0.80,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
}
# /v1/completions has no chat template / thinking concept; the non-thinking profile is the closer
# match (plain instruction-style completion), and is used as its default sampling profile.
COMPLETION_DEFAULTS = NON_THINKING_DEFAULTS


def _sse(obj: dict) -> bytes:
    return f"data: {json.dumps(obj)}\n\n".encode("utf-8")


_DONE = b"data: [DONE]\n\n"


# --------------------------------------------------------------------------
# Shared sampling-params resolution
# --------------------------------------------------------------------------


def _resolve_sampling_params(
    req, defaults: dict[str, float], default_max_tokens: int
) -> SamplingParams:
    def pick(name: str):
        v = getattr(req, name, None)
        return v if v is not None else defaults.get(name)

    return SamplingParams(
        temperature=pick("temperature"),
        top_p=pick("top_p"),
        top_k=pick("top_k"),
        min_p=pick("min_p"),
        presence_penalty=pick("presence_penalty"),
        repetition_penalty=pick("repetition_penalty"),
        max_tokens=req.max_tokens if req.max_tokens is not None else default_max_tokens,
        ignore_eos=req.ignore_eos,
        seed=req.seed,
        n=req.n,
        logprobs=req.logprobs_count(),
    )


# --------------------------------------------------------------------------
# Chat generation pipeline (think-tag + tool-call streaming parse)
# --------------------------------------------------------------------------


@dataclass
class ChatEvent:
    reasoning_delta: str = ""
    content_delta: str = ""
    tool_call_deltas: list[ToolCallDelta] = field(default_factory=list)
    finished: bool = False
    finish_reason: Optional[str] = None
    stats: Optional[RequestStats] = None


#: How many engine steps apart the two streaming paths ask
#: Starlette whether the client is still there. Starlette implements
#: ``is_disconnected()`` as an ``anyio.CancelScope`` around an ``await
#: self.receive()`` that it then cancels -- a scope object, a cancelled await
#: and a task step -- and it was being paid **once per token per stream**, i.e.
#: 256 times per decode step at concurrency 256, on the same event loop that
#: has to hand the GIL back to the engine thread between kernel launches.
#:
#: At ~40 ms/token a 16-step window means a vanished client holds its slot for
#: at most ~0.6 s longer than before, against 240 fewer event-loop round trips
#: per step. The **first** step is always checked (a client that was already
#: gone when the request was admitted is caught at once -- one check per
#: request, not one per token), and so is the last.
DISCONNECT_CHECK_EVERY = 16


async def _detok(loop, executor, fn, token_ids):
    """Run one incremental-detokenizer call, in the pool or inline.

    ``executor is None`` (``--detok-workers 0``) means inline: no
    ``run_in_executor``, no self-pipe wakeup, no extra event-loop callback --
    see :func:`create_app`'s note. Kept as one helper so both stream paths
    make the same choice and a future third one cannot forget."""
    if executor is None:
        return fn(token_ids)
    return await loop.run_in_executor(executor, fn, token_ids)


async def _generate_chat(
    engine: AsyncEngine,
    tokenizer,
    executor: ThreadPoolExecutor,
    request_id: str,
    prompt_ids: list[int],
    sp: SamplingParams,
    enable_thinking: bool,
    stop_strings: list[str],
    http_request: Optional[Request],
    on_disconnect=None,
) -> AsyncIterator[ChatEvent]:
    loop = asyncio.get_running_loop()
    detok = IncrementalDetokenizer(tokenizer, prompt_ids)
    think = ThinkTagParser(start_in_think=enable_thinking)
    toolp = ToolCallStreamParser()
    stopmatch = StopStringMatcher(stop_strings)

    abort_requested = False
    terminal_sent = False
    n_steps = 0
    last_stats = RequestStats(prompt_tokens=len(prompt_ids), completion_tokens=0)

    # NOTE: we always fully drain `engine.add_request`'s generator via this `async for`, even
    # after deciding to abort -- that is what guarantees the engine's own cleanup (releasing its
    # admission slot, discarding abort flags, etc.) runs deterministically, without relying on
    # `aclose()`/`GeneratorExit` semantics across an early `return` out of an `async for`.
    async for step in engine.add_request(request_id, prompt_ids, sp):
        last_stats = step.stats

        n_steps += 1
        if (not abort_requested and http_request is not None
                and (n_steps == 1 or step.finished
                     or n_steps % DISCONNECT_CHECK_EVERY == 0)
                and await http_request.is_disconnected()):
            await engine.abort(request_id)
            abort_requested = True
            if on_disconnect is not None:
                on_disconnect()

        if not abort_requested and step.new_token_ids:
            # One delta per *token*, not one per step. Speculative decoding commits several
            # tokens in a single step, and emitting them as one chunk makes any client that
            # counts stream chunks -- e.g. a live tokens/sec meter -- under-read by the accept
            # length (2-4x) until the final `usage` frame settles it. `add_tokens_split` pays
            # the same single thread-pool hop and concatenates to byte-identical text.
            deltas = await _detok(
                loop, executor, detok.add_tokens_split, tuple(step.new_token_ids)
            )
            for raw in deltas:
                if not raw:
                    continue
                forward, stopped = stopmatch.feed(raw)
                if forward:
                    r, c = think.feed(forward)
                    p, tds = toolp.feed(c)
                    if r or p or tds:
                        yield ChatEvent(reasoning_delta=r, content_delta=p, tool_call_deltas=tds, stats=last_stats)
                if stopped:
                    await engine.abort(request_id)
                    abort_requested = True
                    finish_reason = "tool_calls" if toolp.has_tool_calls else "stop"
                    yield ChatEvent(finished=True, finish_reason=finish_reason, stats=last_stats)
                    terminal_sent = True
                    # Anything after the stop string within this step is not part of the
                    # response; drop the remaining tokens of the batch.
                    break

        if step.finished:
            if not terminal_sent:
                r2, c2 = think.flush()
                p2, tds2 = toolp.feed(c2) if c2 else ("", [])
                p3, tds3 = toolp.flush()
                tds_all = tds2 + tds3
                p_all = p2 + p3
                if r2 or p_all or tds_all:
                    yield ChatEvent(
                        reasoning_delta=r2, content_delta=p_all, tool_call_deltas=tds_all, stats=last_stats
                    )
                finish_reason = "tool_calls" if toolp.has_tool_calls else step.finish_reason
                yield ChatEvent(finished=True, finish_reason=finish_reason, stats=last_stats)
                terminal_sent = True
            # Deliberately no `return`/`break` here: `step.finished` is the *last* value the
            # engine yields before its own generator returns on the *next* `__anext__()` call, so
            # exiting early would skip that final resume and leak the engine's cleanup (admission
            # slot release, running-set bookkeeping) until GC happens to schedule it. Just let the
            # `async for` make that one extra call and end naturally via StopAsyncIteration.


@dataclass
class CompletionEvent:
    text_delta: str = ""
    finished: bool = False
    finish_reason: Optional[str] = None
    stats: Optional[RequestStats] = None


async def _generate_completion(
    engine: AsyncEngine,
    tokenizer,
    executor: ThreadPoolExecutor,
    request_id: str,
    prompt_ids: list[int],
    sp: SamplingParams,
    stop_strings: list[str],
    http_request: Optional[Request],
    on_disconnect=None,
) -> AsyncIterator[CompletionEvent]:
    """Raw-text variant for `/v1/completions`: no think-tag / tool-call parsing (out of scope for
    the base completions API), just incremental detokenization + stop-string matching."""
    loop = asyncio.get_running_loop()
    detok = IncrementalDetokenizer(tokenizer, prompt_ids)
    stopmatch = StopStringMatcher(stop_strings)

    abort_requested = False
    terminal_sent = False
    n_steps = 0
    last_stats = RequestStats(prompt_tokens=len(prompt_ids), completion_tokens=0)

    async for step in engine.add_request(request_id, prompt_ids, sp):
        last_stats = step.stats

        n_steps += 1
        if (not abort_requested and http_request is not None
                and (n_steps == 1 or step.finished
                     or n_steps % DISCONNECT_CHECK_EVERY == 0)
                and await http_request.is_disconnected()):
            await engine.abort(request_id)
            abort_requested = True
            if on_disconnect is not None:
                on_disconnect()

        if not abort_requested and step.new_token_ids:
            raw = await _detok(loop, executor, detok.add_tokens, tuple(step.new_token_ids))
            if raw:
                forward, stopped = stopmatch.feed(raw)
                if forward:
                    yield CompletionEvent(text_delta=forward, stats=last_stats)
                if stopped:
                    await engine.abort(request_id)
                    abort_requested = True
                    yield CompletionEvent(finished=True, finish_reason="stop", stats=last_stats)
                    terminal_sent = True

        if step.finished:
            if not terminal_sent:
                yield CompletionEvent(finished=True, finish_reason=step.finish_reason, stats=last_stats)
                terminal_sent = True
            # See the comment in `_generate_chat`: no early return -- let the `async for` drain
            # the engine generator's own next (immediate) `StopAsyncIteration` naturally.


# --------------------------------------------------------------------------
# App factory
# --------------------------------------------------------------------------


class _InFlight:
    """The global admission counter behind the 503 overload guard.

    Mutated only from the asyncio event loop (every `/v1/*` handler runs there), so a plain int
    is correct and a lock would only add cost. `capacity == 0` disables the guard entirely.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = max(0, int(capacity))
        self.current = 0
        self.peak = 0

    def try_acquire(self) -> bool:
        if self.capacity and self.current >= self.capacity:
            return False
        self.current += 1
        self.peak = max(self.peak, self.current)
        return True

    def release(self) -> None:
        if self.current > 0:
            self.current -= 1


@dataclass
class _Meter:
    """One request's usage row, filled in as the response is produced."""

    key_name: str
    endpoint: str
    started: float
    stream: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_ms: Optional[float] = None
    status: int = 200
    finish_reason: Optional[str] = None
    #: `usage.ERROR_CLASSES` label. Set for every refusal, and also for the two
    #: things that are *not* failures but that an operator must be able to see:
    #: a stream the client walked away from, and one cut short by the deadline.
    error: Optional[str] = None

    def first_token(self) -> None:
        if self.ttft_ms is None:
            self.ttft_ms = (time.perf_counter() - self.started) * 1000.0


class _Drain:
    """Whether this process has been asked to stop, and when.

    A restart used to be: SIGTERM, uvicorn stops the loop, every in-flight SSE
    stream dies mid-sentence, and every *new* request in the same second races
    the shutdown and gets a connection reset. Draining splits that in two --
    **stop accepting** (new `/v1/*` requests get a 503 with `Retry-After`, and
    `/health` goes 503 so the watchdog and any load balancer stop routing here)
    and **let the running ones finish**, which uvicorn's own graceful shutdown
    then waits for, bounded by `--drain-timeout`.
    """

    def __init__(self) -> None:
        self.started_at: Optional[float] = None
        self.reason: str = ""

    @property
    def draining(self) -> bool:
        return self.started_at is not None

    def begin(self, reason: str = "shutdown") -> None:
        if self.started_at is None:
            self.started_at = time.time()
            self.reason = reason

    def as_dict(self) -> dict:
        return {
            "draining": self.draining,
            "since": self.started_at,
            "reason": self.reason or None,
        }


class _RequestDeadline:
    """A wall-clock cap on one request, enforced by cancelling its task.

    The cheap version of this -- check a deadline each time the engine yields --
    cannot fire in the one case that matters: an engine that has stopped
    yielding at all. That is precisely the dead-engine failure (a dead device
    thread, clients waiting on streams that will never produce a token), and it
    is what leaves a browser tab spinning for the full 300 s keep-alive.

    So the timer cancels the task, and `absorb()` tells the handler that the
    `CancelledError` it just caught was *ours* (a timeout, which gets a clean
    terminal SSE frame) rather than the client's (a disconnect, which must
    propagate so Starlette can tear the response down). One `call_later` per
    request; nothing per token.
    """

    def __init__(self, seconds: Optional[float]) -> None:
        self.fired = False
        self._handle = None
        self._task = None
        if seconds and seconds > 0:
            self._task = asyncio.current_task()
            self._handle = asyncio.get_running_loop().call_later(float(seconds), self._fire)

    def _fire(self) -> None:
        self.fired = True
        task = self._task
        if task is not None and not task.done():
            task.cancel()

    def cancel(self) -> None:
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None

    def absorb(self) -> bool:
        """True if the in-flight cancellation was this deadline's, and is now cleared."""
        if not self.fired:
            return False
        self.cancel()
        task = asyncio.current_task()
        uncancel = getattr(task, "uncancel", None)
        if uncancel is not None:  # 3.11+: clear the cancelling counter so we may await again
            try:
                uncancel()
            except Exception:  # noqa: BLE001
                pass
        return True


ANON_KEY = ApiKey(secret="", name="anonymous")


def create_app(
    engine: AsyncEngine,
    tokenizer,
    *,
    model_name: str,
    api_key: Optional[str] = None,
    default_max_tokens: int = 512,
    detok_workers: int = 4,
    # -- public-endpoint options (docs/api.md). All default to "off", so every existing caller
    # (tests, the demo server, benchmarks) gets exactly the pre-existing behaviour.
    api_keys_file: Optional[str] = None,
    admin_key: Optional[str] = None,
    demo_key: Optional[str] = None,
    key_store: Optional[KeyStore] = None,
    usage_db: Optional[str] = None,
    usage_recorder: Optional[UsageRecorder] = None,
    gpu_rate_inr_per_hour: float = DEFAULT_GPU_RATE_INR_PER_HOUR,
    inr_per_usd: float = DEFAULT_INR_PER_USD,
    max_inflight_requests: int = 0,
    max_output_tokens: Optional[int] = None,
    max_prompt_tokens: Optional[int] = None,
    max_request_bytes: int = 1_000_000,
    max_messages: int = 256,
    # -- Reliability options; all default to "off"/"legacy".
    max_context_len: Optional[int] = None,
    min_completion_tokens: int = 0,
    request_timeout_s: float = 0.0,
    max_streams_per_key: int = 0,
    max_streams_per_ip: int = 0,
    client_ip_header: str = "x-forwarded-for",
    drain_timeout_s: float = 0.0,
    # -- opt-in content logging (private to the GPU host; see `content_log.py`).
    content_log_db: Optional[str] = None,
    content_retention_days: float = 0.0,
    query_logger: Optional[QueryLogger] = None,
) -> FastAPI:
    # `--detok-workers 0` runs the incremental detokenizer
    # **on the event loop** instead of hopping to a thread pool.
    #
    # Why that is the faster answer at 256 concurrent streams, and not the
    # obviously-wrong one it looks like: each `run_in_executor` is a
    # `SimpleQueue.put` + a worker wakeup + a `Future.set_result` that itself
    # calls `loop.call_soon_threadsafe` -- an event-loop lock, a `write(2)` on
    # the self-pipe and a selector wakeup. That is *per token per stream*, so
    # 256 of them per decode step, the same 256-per-step pattern that was
    # removed on the `_dispatch` side, reintroduced in the opposite direction.
    # The work being offloaded is two
    # `tokenizer.decode` calls over a window of a few tokens: tens of
    # microseconds, and a fast tokenizer releases the GIL inside them anyway.
    #
    # The pool is still built (and still the default) because a *slow* Python
    # tokenizer, or a chat stream that decodes per token, is a different
    # trade; `None` here means inline.
    executor = (
        ThreadPoolExecutor(max_workers=detok_workers, thread_name_prefix="detok")
        if detok_workers > 0 else None
    )

    # A single static `--api-key` is just a one-entry key store with no limits, so there is
    # exactly one auth code path to reason about (and to test).
    static: list[ApiKey] = []
    if api_key:
        static.append(ApiKey(secret=api_key, name="static"))
    if demo_key:
        static.append(ApiKey(secret=demo_key, name="demo"))
    if admin_key:
        static.append(ApiKey(secret=admin_key, name="admin", admin=True))
    keys = key_store or KeyStore(api_keys_file, static_keys=static)
    limiter = RateLimiter()
    usage = usage_recorder or UsageRecorder(
        usage_db, gpu_rate_inr_per_hour=gpu_rate_inr_per_hour, inr_per_usd=inr_per_usd
    )
    inflight = _InFlight(max_inflight_requests)
    concurrency = ConcurrencyLimiter(per_key=max_streams_per_key, per_ip=max_streams_per_ip)
    drain = _Drain()
    qlog = query_logger or QueryLogger(content_log_db, retention_days=content_retention_days)

    def context_length() -> Optional[int]:
        """The engine's context window, or None if it will not say.

        Explicit `max_context_len` wins (the CLI passes `--max-model-len`, which
        is what the engine was *configured* with); otherwise ask the engine,
        which is the only thing that knows its own page-table geometry. A
        backend that answers neither -- `MockEngine` in most tests -- simply
        gets no clamping, and the old reject-only behaviour via
        `require_fits_context`."""
        if max_context_len:
            return int(max_context_len)
        n = getattr(engine, "max_context_len", None)
        try:
            return int(n) if n else None
        except Exception:  # noqa: BLE001
            return None

    def _install_drain_signals() -> list:
        """SIGTERM/SIGINT flip the server into draining, then run whatever
        handler was already installed.

        Chaining matters: uvicorn installs *its* handlers before the lifespan
        runs, and they are what actually starts the graceful shutdown. Replacing
        them would leave a server that refuses new work and then never exits --
        which is worse than no drain at all."""
        import signal

        installed = []
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                previous = signal.getsignal(sig)

                def _handler(signum, frame, _previous=previous):
                    drain.begin("signal")
                    if callable(_previous) and _previous not in (signal.SIG_DFL, signal.SIG_IGN):
                        _previous(signum, frame)

                signal.signal(sig, _handler)
                installed.append((sig, previous))
            except Exception:  # noqa: BLE001 - not the main thread, or not POSIX
                continue
        return installed

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        keys.install_sighup_handler()
        restore = _install_drain_signals()
        await engine.start()
        try:
            yield
        finally:
            drain.begin("lifespan-shutdown")
            import signal as _signal

            for sig, previous in restore:
                try:
                    _signal.signal(sig, previous)
                except Exception:  # noqa: BLE001
                    pass
            await engine.shutdown()
            if executor is not None:
                executor.shutdown(wait=False)
            usage.shutdown()
            qlog.shutdown()

    app = FastAPI(title="qwenfast", version="0.1.0", lifespan=lifespan)
    app.state.engine = engine
    app.state.tokenizer = tokenizer
    app.state.model_name = model_name
    app.state.key_store = keys
    app.state.rate_limiter = limiter
    app.state.usage = usage
    app.state.inflight = inflight
    app.state.concurrency = concurrency
    app.state.drain = drain
    app.state.drain_timeout_s = drain_timeout_s
    app.state.query_log = qlog

    def _note_handler_error(request, status_code: int, error_class: str) -> None:
        """Meter a refusal that never reached a handler (schema 400, 404, ...).

        Without this the usage table only ever sees failures the handlers
        themselves raised, so a caller sending malformed JSON all day is
        invisible in exactly the view that exists to find such callers."""
        if getattr(request.state, "metered", False):
            return
        usage.record(
            UsageRecord(
                ts=time.time(),
                key_name="anonymous",
                endpoint=str(getattr(request, "url", "")).rsplit("/", 1)[-1][:40] or "unknown",
                prompt_tokens=0,
                completion_tokens=0,
                ttft_ms=None,
                duration_ms=0.0,
                status=status_code,
                stream=False,
                error=error_class,
            )
        )

    install_error_handlers(app, on_error=_note_handler_error)

    @app.middleware("http")
    async def limit_body_size(request: Request, call_next):
        """413 on an oversized body *before* Pydantic materialises it.

        A public endpoint gets 50 MB JSON bodies eventually, whether by accident or on purpose,
        and parsing one costs far more than reading its `Content-Length`."""
        if max_request_bytes:
            declared = request.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > max_request_bytes:
                request.state.metered = True
                usage.record(
                    UsageRecord(
                        ts=time.time(), key_name="anonymous", endpoint="body",
                        prompt_tokens=0, completion_tokens=0, ttft_ms=None,
                        duration_ms=0.0, status=413, stream=False,
                        error="payload_too_large",
                    )
                )
                return JSONResponse(
                    {
                        "error": {
                            "message": (
                                f"request body is {declared} bytes; max {max_request_bytes}"
                            ),
                            "type": "invalid_request_error",
                            "param": None,
                            "code": "payload_too_large",
                        }
                    },
                    status_code=413,
                )
        return await call_next(request)

    async def require_api_key(authorization: Optional[str] = Header(None)) -> ApiKey:
        """Bearer auth against the (hot-reloadable) key store.

        Returns the matched key so handlers can apply its per-key `max_tokens` clamp. When no
        auth is configured at all the endpoint is open and every caller is `anonymous`.
        """
        keys.maybe_reload()
        if not keys.enabled:
            return ANON_KEY
        token = None
        # Scheme is matched case-insensitively ("Bearer", "bearer", "BEARER"):
        # people paste headers in every casing and RFC 6750 says the scheme
        # is case-insensitive.
        if authorization and authorization[:7].lower() == "bearer ":
            token = authorization[7:].strip()
        elif authorization:
            token = authorization.strip()
        found = keys.lookup(token)
        if found is None:
            usage.note_auth_failure()
            raise ApiError(
                401,
                "Invalid API key. Pass `Authorization: Bearer <key>`; "
                "see docs/api.md for how to get one.",
                code="invalid_api_key",
                param="Authorization",
            )
        return found

    async def require_admin_key(authorization: Optional[str] = Header(None)) -> ApiKey:
        key = await require_api_key(authorization)
        if not key.admin:
            raise ApiError(403, "Admin key required", code="permission_denied")
        return key

    def enforce_rate_limit(key: ApiKey) -> None:
        decision = limiter.check_and_admit(key)
        if decision.allowed:
            return
        usage.note_rate_limited()
        retry = max(1, int(decision.retry_after_s + 0.999))
        raise ApiError(
            429,
            decision.reason,
            code="rate_limit_exceeded",
            error_class="rate_limited",
            headers={"Retry-After": str(retry)},
        )

    def reject_if_draining() -> None:
        if drain.draining:
            raise ApiError(
                503,
                "This server is shutting down and is not accepting new requests. "
                "Retry in a few seconds — a replacement is starting.",
                code="server_draining",
                error_class="draining",
                headers={"Retry-After": "10"},
            )

    def client_ip(request: Request) -> str:
        """Best-effort caller identity for the per-IP cap.

        Behind a hosting provider's edge proxy every socket peer can be the same
        private address, so the socket alone
        would make the per-IP cap a second global cap. `--client-ip-header`
        (default `x-forwarded-for`, first hop) is therefore the primary source,
        with the peer as the fallback for a direct connection."""
        if client_ip_header:
            raw = request.headers.get(client_ip_header)
            if raw:
                first = raw.split(",")[0].strip()
                if first:
                    return first[:64]
        peer = getattr(request, "client", None)
        return (peer.host if peer else "-")[:64]

    def admit_request(key: ApiKey, ip: str):
        """Take a fairness slot and a global admission slot, or take neither.

        Returns an idempotent release callable. Order matters: the fairness cap
        is checked first because its answer (`429`, your budget) is more useful
        than the global one (`503`, everybody's budget) when both would refuse,
        and because a caller already holding 8 streams should not consume the
        last global slot only to be told to go away."""
        decision = concurrency.acquire(key, ip)
        if not decision.allowed:
            usage.note_concurrency_rejected()
            raise ApiError(
                429,
                decision.reason,
                code="concurrency_limit_exceeded",
                error_class="concurrency_limited",
                headers={"Retry-After": str(decision.retry_after_s)},
            )
        try:
            enforce_capacity()
        except BaseException:
            concurrency.release(key, ip)
            raise

        state = {"released": False}

        def release() -> None:
            if state["released"]:
                return
            state["released"] = True
            inflight.release()
            concurrency.release(key, ip)

        return release

    def enforce_capacity() -> None:
        """503 rather than an unbounded queue.

        Past the cap the engine's own admission queue is already deep enough that a new request
        would wait longer than any sane client's timeout; answering immediately with a 503 and a
        `Retry-After` is both more honest and what makes a load generator back off instead of
        piling on."""
        if not inflight.try_acquire():
            usage.note_overload()
            raise ApiError(
                503,
                f"Server at capacity ({inflight.capacity} concurrent requests). "
                "Retry shortly — this is a single-GPU public endpoint.",
                code="server_overloaded",
                error_class="overloaded",
                headers={"Retry-After": "5"},
            )

    def clamp_output_tokens(sp: SamplingParams, key: ApiKey) -> None:
        """Per-key and server-wide `max_tokens` ceilings, applied silently (OpenAI-style)."""
        limits = [n for n in (key.max_tokens, max_output_tokens) if n]
        if limits:
            sp.max_tokens = min(sp.max_tokens, min(limits))

    def enforce_prompt_size(n_prompt: int) -> None:
        if max_prompt_tokens and n_prompt > max_prompt_tokens:
            raise ApiError(
                400,
                f"prompt is {n_prompt} tokens; this endpoint accepts at most "
                f"{max_prompt_tokens}. Send fewer/shorter messages.",
                code="context_length_exceeded",
                param="messages",
                error_class="prompt_too_long",
            )

    def fit_to_context(n_prompt: int, sp: SamplingParams) -> Optional[int]:
        """Clamp `max_tokens` to the room the prompt leaves; 400 only if there is none.

        This is the difference between a long conversation that keeps working
        and one that stops dead. Without it, `prompt + max_tokens > context` is a
        `400`, so a 13-token question with `max_tokens: 4096` against a
        4,096-token context is refused outright, and a chat page that never
        shortens its history hits the wall and stays there.

        OpenAI's own behaviour is to clamp: `max_tokens` is a *ceiling*, not a
        reservation, and a response that stops early says `finish_reason:
        "length"`. So: clamp to `context - prompt`, answer normally, and reject
        only when the prompt by itself leaves less than `min_completion_tokens`
        of room -- the one case where there is no answer to give.

        Returns the room in tokens (None when the engine will not state a
        context length, in which case nothing is clamped and the old
        engine-side check still guards the device).
        """
        ctx = context_length()
        if not ctx:
            return None
        room = ctx - int(n_prompt)
        floor = max(1, int(min_completion_tokens or 1))
        if room < floor:
            raise ApiError(
                400,
                f"This model's maximum context length is {ctx} tokens. Your messages "
                f"came to {n_prompt} tokens, which leaves {max(room, 0)} for a reply "
                f"(at least {floor} required). Please shorten the conversation.",
                code="context_length_exceeded",
                param="messages",
                error_class="context_length_exceeded",
            )
        if sp.max_tokens > room:
            sp.max_tokens = room
        return room

    def finish_meter(meter: _Meter, stats: Optional[RequestStats] = None) -> None:
        """Hand one finished request to the recorder. Never raises, never blocks."""
        if stats is not None:
            meter.prompt_tokens = stats.prompt_tokens
            meter.completion_tokens = stats.completion_tokens
        try:
            limiter.record_tokens(meter.key_name, meter.completion_tokens)
            usage.record(
                UsageRecord(
                    ts=time.time(),
                    key_name=meter.key_name,
                    endpoint=meter.endpoint,
                    prompt_tokens=meter.prompt_tokens,
                    completion_tokens=meter.completion_tokens,
                    ttft_ms=meter.ttft_ms,
                    duration_ms=(time.perf_counter() - meter.started) * 1000.0,
                    status=meter.status,
                    stream=meter.stream,
                    finish_reason=meter.finish_reason,
                    error=meter.error,
                )
            )
        except Exception:  # noqa: BLE001 - metering never fails a response
            pass

    def log_query(
        meter: _Meter,
        messages,
        response_text: str,
        reasoning_text: Optional[str],
        *,
        request_id: Optional[str],
        ip: str,
        finish_reason: Optional[str],
    ) -> None:
        """Append one content row, if content logging is on. Never raises.

        Gated on `qlog.enabled` before anything is built, so the default (off)
        costs one attribute read per request and allocates nothing."""
        if not qlog.enabled:
            return
        try:
            qlog.record(
                QueryRecord(
                    ts=time.time(),
                    key_name=meter.key_name,
                    client_ip=ip,
                    endpoint=meter.endpoint,
                    model=model_name,
                    messages_json=messages_to_json(messages),
                    response_text=response_text or "",
                    reasoning_text=reasoning_text or None,
                    prompt_tokens=meter.prompt_tokens,
                    completion_tokens=meter.completion_tokens,
                    status=meter.status,
                    finish_reason=finish_reason or meter.finish_reason,
                    ttft_ms=meter.ttft_ms,
                    duration_ms=(time.perf_counter() - meter.started) * 1000.0,
                    request_id=request_id or "",
                )
            )
        except Exception:  # noqa: BLE001 - content logging never fails a response
            pass

    def engine_health() -> Optional[str]:
        """`None` if the engine can serve, else why not (`AsyncEngine.health`).

        Never let introspection itself take the server down: an engine that
        raises out of `health()` is, for our purposes, unhealthy."""
        probe = getattr(engine, "health", None)
        if probe is None:
            return None
        try:
            return probe()
        except Exception as exc:  # noqa: BLE001
            return f"health probe raised: {exc}"

    def require_fits_context(n_prompt: int, sp: SamplingParams) -> None:
        """400 a request that would run off the end of the engine's context.

        Duck-typed on purpose: the check lives in the engine (which knows its
        rotary-table and page-table geometry), and this module only knows how
        to turn its verdict into an OpenAI-shaped error. An engine without the
        hook -- `MockEngine`, or any future backend -- is simply not checked
        here.

        Without it a 2158-token prompt with `max_tokens=500` against
        `--max-model-len 2560` would be accepted, decode 402 tokens, and then
        assert inside `RotaryTable.lookup`: a device-side assert, which kills the
        CUDA context and with it the engine thread, so every later request gets
        a 503.
        """
        checker = getattr(engine, "context_length_error", None)
        if checker is None:
            return
        try:
            detail = checker(n_prompt, sp.max_tokens)
        except Exception:  # noqa: BLE001 - never let the guard itself 500
            return
        if detail:
            raise HTTPException(status_code=400, detail=detail)

    def require_live_engine() -> None:
        reason = engine_health()
        if reason is not None:
            # 503, not 500: this is "the backend is gone", and it is what tells
            # a load generator to stop rather than keep queueing work.
            raise HTTPException(status_code=503, detail=reason)

    # -- unauthenticated endpoints ---------------------------------------------------

    def limits_payload() -> dict:
        """The numbers a client needs in order to *not* be refused.

        Published so callers do not have to discover the prompt cap and the
        context length by hitting them, once per message for the rest of a
        conversation."""
        ctx = context_length()
        return {
            "max_context_length": ctx,
            "max_prompt_tokens": max_prompt_tokens or (ctx - min_completion_tokens if ctx else None),
            "default_max_tokens": default_max_tokens,
            "max_output_tokens": max_output_tokens,
            "min_completion_tokens": min_completion_tokens or None,
            "max_messages": max_messages,
            "max_request_bytes": max_request_bytes,
            "max_concurrent_streams_per_key": concurrency.per_key or None,
            "max_concurrent_streams_per_ip": concurrency.per_ip or None,
            "request_timeout_s": request_timeout_s or None,
            # Stated explicitly so a client knows an over-long `max_tokens` is
            # clamped rather than refused.
            "max_tokens_is_clamped": True,
        }

    @app.get("/status")
    async def status() -> JSONResponse:
        """Public, keyless "is it worth me calling right now?".

        Deliberately carries no usage, no cost, and no key names -- everything
        on here is either a constant of the deployment or a number a caller
        could infer from being throttled anyway."""
        reason = engine_health()
        try:
            stats = engine.get_stats()
            load = {
                "in_flight": inflight.current,
                "capacity": inflight.capacity or None,
                "running": stats.num_requests_running,
                "waiting": stats.num_requests_waiting,
                "kv_pages_used": stats.kv_slots_used,
                "kv_pages_total": stats.kv_slots_total,
            }
            uptime_s = stats.uptime_s
        except Exception:  # noqa: BLE001 - /status must answer even if the engine is sick
            load = {"in_flight": inflight.current, "capacity": inflight.capacity or None}
            uptime_s = None
        healthy = reason is None and not drain.draining
        return JSONResponse(
            {
                "status": "ok" if healthy else ("draining" if drain.draining else "unhealthy"),
                "healthy": healthy,
                "detail": reason,
                "model": model_name,
                "load": load,
                "limits": limits_payload(),
                "uptime_s": uptime_s,
                "process_uptime_s": max(0.0, time.time() - usage.process_started_at),
                "draining": drain.draining,
            },
            status_code=200 if healthy else 503,
        )

    @app.get("/health")
    async def health() -> JSONResponse:
        # An engine whose device thread has died must NOT report healthy,
        # otherwise clients and supervisors keep sending work to a dead engine.
        # Draining reports unhealthy on purpose: it is what makes the watchdog
        # and any load balancer stop sending work here *before* the process
        # actually goes away, which is the difference between a clean restart
        # and a handful of connection resets.
        if drain.draining:
            return JSONResponse(
                {"status": "draining", "detail": "shutting down"}, status_code=503
            )
        reason = engine_health()
        if reason is not None:
            return JSONResponse({"status": "unhealthy", "detail": reason}, status_code=503)
        return JSONResponse({"status": "ok"})

    @app.get("/metrics")
    async def metrics() -> PlainTextResponse:
        extra = usage.prometheus_lines()
        extra += [
            "# HELP qwenfast:api_inflight_requests Requests currently held by the admission guard.",
            "# TYPE qwenfast:api_inflight_requests gauge",
            f"qwenfast:api_inflight_requests {inflight.current}",
            "# HELP qwenfast:api_inflight_capacity Configured concurrent-request cap (0 == unlimited).",
            "# TYPE qwenfast:api_inflight_capacity gauge",
            f"qwenfast:api_inflight_capacity {inflight.capacity}",
            "# HELP qwenfast:api_keys_loaded Number of API keys currently accepted.",
            "# TYPE qwenfast:api_keys_loaded gauge",
            f"qwenfast:api_keys_loaded {len(keys.names())}",
            "# HELP qwenfast:api_draining 1 while the process is refusing new requests.",
            "# TYPE qwenfast:api_draining gauge",
            f"qwenfast:api_draining {1 if drain.draining else 0}",
        ]
        text = render_prometheus_text(engine.get_stats(), extra_lines=extra)
        return PlainTextResponse(text, media_type="text/plain; version=0.0.4")

    # -- admin ------------------------------------------------------------------------

    @app.get("/admin/usage")
    async def admin_usage(
        hours: int = 24, _admin: ApiKey = Depends(require_admin_key)
    ) -> JSONResponse:
        """Everything the private dashboard needs, in one round trip.

        Deliberately *not* on the `/v1` prefix: the Vercel proxy forwards `/v1/*` verbatim to the
        public internet, and usage data is the operator's, not the callers'.
        """
        hours = max(1, min(int(hours), 168))
        try:
            stats = engine.get_stats()
            engine_view = {
                "num_requests_running": stats.num_requests_running,
                "num_requests_waiting": stats.num_requests_waiting,
                "generation_tokens_total": stats.generation_tokens_total,
                "prompt_tokens_total": stats.prompt_tokens_total,
                "lifetime_tokens_per_second": stats.tokens_per_second,
                "per_stream_tokens_per_second": (
                    stats.tpot.count / stats.tpot.sum if stats.tpot.sum > 0 else None
                ),
                "mean_ttft_ms": (
                    stats.ttft.sum / stats.ttft.count * 1000.0 if stats.ttft.count else None
                ),
                "spec_accept_length": stats.spec_accept_length,
                "kv_pages_used": stats.kv_slots_used,
                "kv_pages_total": stats.kv_slots_total,
                "uptime_s": stats.uptime_s,
            }
        except Exception as exc:  # noqa: BLE001 - the dashboard must render even if the engine is sick
            engine_view = {"error": f"{exc.__class__.__name__}: {exc}"}

        body = usage.summary(hours=hours)
        body["model"] = model_name
        body["engine"] = engine_view
        body["engine_health"] = engine_health() or "ok"
        body["capacity"] = {
            "inflight": inflight.current,
            "peak_inflight": inflight.peak,
            "capacity": inflight.capacity,
            "concurrency": concurrency.snapshot(),
            "queue_rejects": {
                "overload_503": body["totals"].get("rejected_overload", 0),
                "concurrency_429": body["totals"].get("rejected_concurrency", 0),
                "rate_limited_429": body["totals"].get("rate_limited", 0),
            },
        }
        body["limits"] = limits_payload()
        body["drain"] = drain.as_dict()
        body["content_log"] = qlog.stats()
        body["keys"] = [
            {**k, **limiter.snapshot(k["name"])} for k in keys.describe()
        ]
        body["key_store"] = {
            "path": keys.path,
            "loaded_at": keys.loaded_at,
            "reload_count": keys.reload_count,
            "error": keys.load_error,
        }
        return JSONResponse(body)

    # -- authenticated /v1 endpoints --------------------------------------------------

    @app.get("/admin/queries")
    async def admin_queries(
        since: Optional[float] = None,
        hours: Optional[float] = None,
        limit: int = 100,
        key: Optional[str] = None,
        grep: Optional[str] = None,
        counts: Optional[str] = None,
        format: str = "json",
        _admin: ApiKey = Depends(require_admin_key),
    ):
        """Export the content log. Admin key only; never reachable from `/v1`.

        `counts=key` / `counts=ip` returns the grouped view instead of rows --
        the first thing to look at when deciding whether a key is being abused,
        and small enough to eyeball. `grep` filters rows by regex over the
        prompt and the reply.
        """
        if not qlog.enabled:
            return JSONResponse(
                {"enabled": False, "detail": "content logging is off (--log-content)"},
                status_code=200,
            )
        if since is None and hours:
            since = time.time() - float(hours) * 3600.0
        limit = max(1, min(int(limit), 10_000))

        if counts:
            return JSONResponse(
                {
                    "enabled": True,
                    "group_by": "client_ip" if counts in ("ip", "client_ip") else "key_name",
                    "since": since,
                    "counts": qlog.counts(
                        since=since, by=("ip" if counts in ("ip", "client_ip") else "key_name")
                    ),
                }
            )
        try:
            rows = (
                qlog.grep(grep, since=since, limit=limit)
                if grep
                else qlog.export(since=since, limit=limit, key_name=key)
            )
        except ValueError as exc:
            raise ApiError(400, str(exc), code="invalid_grep", param="grep") from exc

        fmt = (format or "json").lower()
        if fmt == "csv":
            from .content_log import rows_to_csv

            return PlainTextResponse(
                rows_to_csv(rows),
                media_type="text/csv; charset=utf-8",
                headers={"Content-Disposition": 'attachment; filename="queries.csv"'},
            )
        if fmt == "jsonl":
            from .content_log import rows_to_jsonl

            return PlainTextResponse(rows_to_jsonl(rows), media_type="application/x-ndjson")
        return JSONResponse({"enabled": True, "count": len(rows), "rows": rows})

    @app.get("/v1/limits")
    async def limits(key: ApiKey = Depends(require_api_key)) -> JSONResponse:
        """Everything a client needs to stay inside the endpoint's envelope.

        Keyed, because the per-key part of the answer depends on which key is
        asking; the deployment-wide part is also on `/status`, without a key."""
        body = dict(limits_payload())
        body["object"] = "limits"
        body["model"] = model_name
        body["key"] = {
            "name": key.name,
            "max_tokens": key.max_tokens,
            "rpm": key.rpm,
            "tpm": key.tpm,
            "max_concurrent_streams": concurrency.limit_for(key) or None,
        }
        return JSONResponse(body)

    @app.get("/v1/models", dependencies=[Depends(require_api_key)])
    async def list_models() -> JSONResponse:
        ctx = context_length()
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {
                        "id": model_name,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "qwenfast",
                        # Non-standard, additive fields: an OpenAI SDK ignores
                        # them, and a client that wants the envelope without a
                        # second round trip to /v1/limits can read them here.
                        "context_window": ctx,
                        "max_output_tokens": max_output_tokens,
                        "max_prompt_tokens": max_prompt_tokens,
                    }
                ],
            }
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(
        req: ChatCompletionRequest,
        http_request: Request,
        key: ApiKey = Depends(require_api_key),
    ):
        meter = _Meter(
            key_name=key.name,
            endpoint="chat.completions",
            started=time.perf_counter(),
            stream=bool(req.stream),
        )
        ip = client_ip(http_request)
        try:
            reject_if_draining()
            require_live_engine()
            enforce_rate_limit(key)
            if len(req.messages) > max_messages:
                raise ApiError(
                    400,
                    f"too many messages ({len(req.messages)}); max {max_messages}",
                    code="too_many_messages",
                    param="messages",
                    error_class="too_many_messages",
                )
            enable_thinking = req.resolve_enable_thinking()
            reasoning_effort = req.resolve_reasoning_effort()
            preserve_thinking = req.resolve_preserve_thinking()

            try:
                prompt_ids = render_chat_prompt(
                    tokenizer,
                    [m.model_dump(exclude_none=True) for m in req.messages],
                    tools=req.tools,
                    enable_thinking=enable_thinking,
                    reasoning_effort=reasoning_effort,
                    preserve_thinking=preserve_thinking,
                )
            except Exception as exc:  # noqa: BLE001 - surface template errors as a 400
                raise ApiError(
                    400, f"chat template error: {exc}", code="invalid_chat_template",
                    param="messages", error_class="invalid_request",
                ) from exc

            defaults = THINKING_DEFAULTS if enable_thinking else NON_THINKING_DEFAULTS
            sp = _resolve_sampling_params(req, defaults, default_max_tokens)
            meter.prompt_tokens = len(prompt_ids)
            enforce_prompt_size(len(prompt_ids))
            clamp_output_tokens(sp, key)
            # Clamp `max_tokens` into the room the prompt leaves, rather than
            # refusing the request — see `fit_to_context`. The engine's own
            # check still runs after it, and must now never fire.
            fit_to_context(len(prompt_ids), sp)
            require_fits_context(len(prompt_ids), sp)
            stop_strings = req.stop_list()
            # Admission is acquired LAST, and nothing between here and the `finally` that
            # releases it may raise — otherwise a slot leaks and the cap ratchets down to zero.
            release_admission = admit_request(key, ip)
        except HTTPException as exc:
            http_request.state.metered = True
            meter.status = exc.status_code
            meter.error = error_class_of(exc, exc.status_code)
            finish_meter(meter)
            log_query(
                meter, req.messages, "", None, request_id=None, ip=ip,
                finish_reason=None,
            )
            raise
        # `X-Request-Id`, if the client sets one, becomes the engine-facing request id (in
        # addition to the OpenAI-shaped `id` field in the response, which always gets a fresh
        # `chatcmpl-...`). This is what lets tests drive `MockEngine.script(request_id=...)`
        # deterministically; real clients never need to set it.
        request_id = http_request.headers.get("x-request-id") or f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        if req.stream:

            async def event_stream():
                first = True
                last_stats: Optional[RequestStats] = None
                logged = qlog.enabled
                content_parts: list[str] = []
                reasoning_parts: list[str] = []
                deadline = _RequestDeadline(request_timeout_s)

                def _note_disconnect() -> None:
                    meter.error = "client_disconnect"

                try:
                    try:
                        async for ev in _generate_chat(
                            engine, tokenizer, executor, request_id, prompt_ids, sp,
                            enable_thinking, stop_strings, http_request,
                            on_disconnect=_note_disconnect,
                        ):
                            if ev.stats is not None:
                                last_stats = ev.stats
                            if not ev.finished:
                                delta: dict[str, Any] = {}
                                if first:
                                    delta["role"] = "assistant"
                                    first = False
                                if ev.reasoning_delta:
                                    delta["reasoning_content"] = ev.reasoning_delta
                                if ev.content_delta:
                                    delta["content"] = ev.content_delta
                                if ev.tool_call_deltas:
                                    delta["tool_calls"] = [_tool_call_delta_json(td) for td in ev.tool_call_deltas]
                                if logged:
                                    if ev.content_delta:
                                        content_parts.append(ev.content_delta)
                                    if ev.reasoning_delta:
                                        reasoning_parts.append(ev.reasoning_delta)
                                if delta:
                                    meter.first_token()
                                    yield _sse(_chat_chunk(request_id, created, model_name, delta, None))
                            else:
                                meter.finish_reason = ev.finish_reason
                                delta = {} if first else {}
                                yield _sse(
                                    _chat_chunk(request_id, created, model_name, delta, ev.finish_reason)
                                )
                                if req.stream_options and req.stream_options.include_usage and ev.stats:
                                    usage_json = _usage_dict(ev.stats)
                                    yield _sse(_chat_chunk(request_id, created, model_name, None, None, usage=usage_json))
                    except asyncio.CancelledError:
                        # Ours (the deadline) or the client's? Only the first is
                        # recoverable; the second must propagate so Starlette can
                        # tear the response down.
                        if not deadline.absorb():
                            meter.error = meter.error or "client_disconnect"
                            raise
                        meter.error = "request_timeout"
                        # `length`, not a bespoke reason: OpenAI SDKs type
                        # `finish_reason` as a closed set, and a truncated answer
                        # *is* a length stop from the caller's point of view. The
                        # real reason is on the usage row.
                        meter.finish_reason = "length"
                        await engine.abort(request_id)
                        yield _sse(
                            _chat_chunk(request_id, created, model_name, {} if first else {}, "length")
                        )
                        if req.stream_options and req.stream_options.include_usage and last_stats:
                            yield _sse(
                                _chat_chunk(
                                    request_id, created, model_name, None, None,
                                    usage=_usage_dict(last_stats),
                                )
                            )
                finally:
                    # Metering and the admission release both live here, not after the loop, so
                    # a client disconnect (which closes the generator) still books the tokens the
                    # GPU actually produced and still frees the admission slot.
                    deadline.cancel()
                    await engine.abort(request_id)
                    release_admission()
                    finish_meter(meter, last_stats)
                    if logged:
                        log_query(
                            meter, req.messages, "".join(content_parts),
                            "".join(reasoning_parts) or None,
                            request_id=request_id, ip=ip, finish_reason=meter.finish_reason,
                        )
                yield _DONE

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    # What the server actually resolved `max_tokens` to after the
                    # per-key cap and the context clamp. A client that wants to
                    # know it was trimmed can read it; nothing needs to.
                    "X-Qwenfast-Max-Tokens": str(sp.max_tokens),
                },
            )

        # Non-streaming: drain fully, then build one response.
        reasoning_accum: list[str] = []
        content_accum: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        finish_reason = "stop"
        final_stats = RequestStats(prompt_tokens=len(prompt_ids), completion_tokens=0)
        deadline = _RequestDeadline(request_timeout_s)
        try:
            try:
                async for ev in _generate_chat(
                    engine, tokenizer, executor, request_id, prompt_ids, sp, enable_thinking,
                    stop_strings, http_request,
                ):
                    if ev.reasoning_delta:
                        reasoning_accum.append(ev.reasoning_delta)
                    if ev.content_delta:
                        content_accum.append(ev.content_delta)
                    if ev.reasoning_delta or ev.content_delta or ev.tool_call_deltas:
                        meter.first_token()
                    for td in ev.tool_call_deltas:
                        _accumulate_tool_call(tool_calls, td)
                    if ev.stats:
                        final_stats = ev.stats
                    if ev.finished:
                        finish_reason = ev.finish_reason or "stop"
            except asyncio.CancelledError:
                if not deadline.absorb():
                    meter.error = meter.error or "client_disconnect"
                    raise
                # Nothing has been sent yet, so unlike the streaming path this
                # can still be an honest error rather than a truncated answer.
                meter.error = "request_timeout"
                await engine.abort(request_id)
                raise ApiError(
                    504,
                    f"request exceeded the {request_timeout_s:.0f}s server time limit and was "
                    "cancelled. Ask for fewer tokens, or use stream=true so partial output "
                    "is delivered as it is produced.",
                    code="request_timeout",
                    error_class="request_timeout",
                )
        except HTTPException as exc:
            meter.status = exc.status_code
            meter.error = meter.error or error_class_of(exc, exc.status_code)
            raise
        finally:
            deadline.cancel()
            await engine.abort(request_id)
            release_admission()
            meter.finish_reason = finish_reason
            finish_meter(meter, final_stats)
            log_query(
                meter, req.messages, "".join(content_accum),
                "".join(reasoning_accum) or None,
                request_id=request_id, ip=ip, finish_reason=finish_reason,
            )

        message: dict[str, Any] = {"role": "assistant"}
        reasoning_text = "".join(reasoning_accum)
        content_text = "".join(content_accum)
        if reasoning_text:
            message["reasoning_content"] = reasoning_text
        if tool_calls:
            message["tool_calls"] = [
                _finalize_tool_call(tool_calls[i]) for i in sorted(tool_calls)
            ]
            message["content"] = content_text or None
        else:
            message["content"] = content_text

        return JSONResponse(
            {
                "id": request_id,
                "object": "chat.completion",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
                "usage": _usage_dict(final_stats),
            },
            headers={"X-Qwenfast-Max-Tokens": str(sp.max_tokens)},
        )

    @app.post("/v1/completions")
    async def completions(
        req: CompletionRequest,
        http_request: Request,
        key: ApiKey = Depends(require_api_key),
    ):
        meter = _Meter(
            key_name=key.name,
            endpoint="completions",
            started=time.perf_counter(),
            stream=bool(req.stream),
        )
        ip = client_ip(http_request)
        try:
            reject_if_draining()
            require_live_engine()
            enforce_rate_limit(key)
            prompt_ids = _encode_completion_prompt(tokenizer, req.prompt)
            sp = _resolve_sampling_params(req, COMPLETION_DEFAULTS, default_max_tokens)
            meter.prompt_tokens = len(prompt_ids)
            enforce_prompt_size(len(prompt_ids))
            clamp_output_tokens(sp, key)
            fit_to_context(len(prompt_ids), sp)
            require_fits_context(len(prompt_ids), sp)
            stop_strings = req.stop_list()
            release_admission = admit_request(key, ip)  # last: see the note in chat_completions
        except HTTPException as exc:
            http_request.state.metered = True
            meter.status = exc.status_code
            meter.error = error_class_of(exc, exc.status_code)
            finish_meter(meter)
            log_query(
                meter, [{"role": "user", "content": _prompt_preview(req.prompt)}], "", None,
                request_id=None, ip=ip, finish_reason=None,
            )
            raise
        request_id = http_request.headers.get("x-request-id") or f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        if req.stream:

            async def event_stream():
                last_stats: Optional[RequestStats] = None
                logged = qlog.enabled
                text_parts: list[str] = []
                deadline = _RequestDeadline(request_timeout_s)

                def _note_disconnect() -> None:
                    meter.error = "client_disconnect"

                try:
                    try:
                        async for ev in _generate_completion(
                            engine, tokenizer, executor, request_id, prompt_ids, sp,
                            stop_strings, http_request, on_disconnect=_note_disconnect,
                        ):
                            if ev.stats is not None:
                                last_stats = ev.stats
                            if not ev.finished:
                                if ev.text_delta:
                                    if logged:
                                        text_parts.append(ev.text_delta)
                                    meter.first_token()
                                    yield _sse(_completion_chunk(request_id, created, model_name, ev.text_delta, None))
                            else:
                                meter.finish_reason = ev.finish_reason
                                yield _sse(_completion_chunk(request_id, created, model_name, "", ev.finish_reason))
                                if req.stream_options and req.stream_options.include_usage and ev.stats:
                                    usage_json = _usage_dict(ev.stats)
                                    yield _sse(
                                        _completion_chunk(request_id, created, model_name, None, None, usage=usage_json)
                                    )
                    except asyncio.CancelledError:
                        if not deadline.absorb():
                            meter.error = meter.error or "client_disconnect"
                            raise
                        meter.error = "request_timeout"
                        meter.finish_reason = "length"
                        await engine.abort(request_id)
                        yield _sse(_completion_chunk(request_id, created, model_name, "", "length"))
                        if req.stream_options and req.stream_options.include_usage and last_stats:
                            yield _sse(
                                _completion_chunk(
                                    request_id, created, model_name, None, None,
                                    usage=_usage_dict(last_stats),
                                )
                            )
                finally:
                    deadline.cancel()
                    await engine.abort(request_id)
                    release_admission()
                    finish_meter(meter, last_stats)
                    if logged:
                        log_query(
                            meter,
                            [{"role": "user", "content": _prompt_preview(req.prompt)}],
                            "".join(text_parts), None,
                            request_id=request_id, ip=ip, finish_reason=meter.finish_reason,
                        )
                yield _DONE

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "X-Qwenfast-Max-Tokens": str(sp.max_tokens),
                },
            )

        text_accum: list[str] = []
        finish_reason = "stop"
        final_stats = RequestStats(prompt_tokens=len(prompt_ids), completion_tokens=0)
        deadline = _RequestDeadline(request_timeout_s)
        try:
            try:
                async for ev in _generate_completion(
                    engine, tokenizer, executor, request_id, prompt_ids, sp, stop_strings,
                    http_request,
                ):
                    if ev.text_delta:
                        meter.first_token()
                        text_accum.append(ev.text_delta)
                    if ev.stats:
                        final_stats = ev.stats
                    if ev.finished:
                        finish_reason = ev.finish_reason or "stop"
            except asyncio.CancelledError:
                if not deadline.absorb():
                    meter.error = meter.error or "client_disconnect"
                    raise
                meter.error = "request_timeout"
                await engine.abort(request_id)
                raise ApiError(
                    504,
                    f"request exceeded the {request_timeout_s:.0f}s server time limit and was "
                    "cancelled. Ask for fewer tokens, or use stream=true.",
                    code="request_timeout",
                    error_class="request_timeout",
                )
        except HTTPException as exc:
            meter.status = exc.status_code
            meter.error = meter.error or error_class_of(exc, exc.status_code)
            raise
        finally:
            deadline.cancel()
            await engine.abort(request_id)
            release_admission()
            meter.finish_reason = finish_reason
            finish_meter(meter, final_stats)
            log_query(
                meter, [{"role": "user", "content": _prompt_preview(req.prompt)}],
                "".join(text_accum), None,
                request_id=request_id, ip=ip, finish_reason=finish_reason,
            )

        return JSONResponse(
            {
                "id": request_id,
                "object": "text_completion",
                "created": created,
                "model": model_name,
                "choices": [
                    {"index": 0, "text": "".join(text_accum), "finish_reason": finish_reason, "logprobs": None}
                ],
                "usage": _usage_dict(final_stats),
            },
            headers={"X-Qwenfast-Max-Tokens": str(sp.max_tokens)},
        )

    return app


# --------------------------------------------------------------------------
# response-building helpers
# --------------------------------------------------------------------------


def _usage_dict(stats: RequestStats) -> dict[str, int]:
    return {
        "prompt_tokens": stats.prompt_tokens,
        "completion_tokens": stats.completion_tokens,
        "total_tokens": stats.prompt_tokens + stats.completion_tokens,
    }


def _chat_chunk(
    id_: str,
    created: int,
    model: str,
    delta: Optional[dict],
    finish_reason: Optional[str],
    *,
    usage: Optional[dict] = None,
) -> dict:
    chunk: dict[str, Any] = {
        "id": id_,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": (
            [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
        ),
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def _completion_chunk(
    id_: str,
    created: int,
    model: str,
    text: Optional[str],
    finish_reason: Optional[str],
    *,
    usage: Optional[dict] = None,
) -> dict:
    chunk: dict[str, Any] = {
        "id": id_,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": (
            []
            if text is None
            else [{"index": 0, "text": text, "finish_reason": finish_reason, "logprobs": None}]
        ),
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def _tool_call_delta_json(td: ToolCallDelta) -> dict:
    function: dict[str, Any] = {}
    if td.name is not None:
        function["name"] = td.name
    if td.arguments_delta:
        function["arguments"] = td.arguments_delta
    out: dict[str, Any] = {"index": td.index, "type": "function", "function": function}
    if td.id is not None:
        out["id"] = td.id
    return out


def _accumulate_tool_call(store: dict[int, dict[str, Any]], td: ToolCallDelta) -> None:
    entry = store.setdefault(td.index, {"id": None, "name": "", "arguments": []})
    if td.id is not None:
        entry["id"] = td.id
    if td.name is not None:
        entry["name"] = td.name
    if td.arguments_delta:
        entry["arguments"].append(td.arguments_delta)


def _finalize_tool_call(entry: dict[str, Any]) -> dict:
    return {
        "id": entry["id"] or "call_0",
        "type": "function",
        "function": {"name": entry["name"], "arguments": "".join(entry["arguments"])},
    }


def _prompt_preview(prompt) -> str:
    """`/v1/completions` has no messages array; give the content log one anyway."""
    try:
        if isinstance(prompt, list):
            if prompt and isinstance(prompt[0], int):
                return f"<{len(prompt)} token ids>"
            return "\n".join(str(p) for p in prompt)
        return str(prompt)
    except Exception:  # noqa: BLE001
        return "<unrenderable prompt>"


def _encode_completion_prompt(tokenizer, prompt) -> list[int]:
    if isinstance(prompt, list):
        if prompt and isinstance(prompt[0], int):
            return list(prompt)
        if len(prompt) != 1:
            raise ApiError(
                400, "batched string prompts (n>1) are not supported",
                code="unsupported_parameter", param="prompt",
            )
        prompt = prompt[0]
    return list(tokenizer.encode(prompt, add_special_tokens=False))
