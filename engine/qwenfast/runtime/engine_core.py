"""The engine loop in its own process.

Why
---
vLLM holds the H200 at **99.96 %** utilisation through the serving sweep,
while the in-process engine's best served configuration sat at **71.5 %** at
concurrency 256, and the offline step cost multiplied by that utilisation
reproduces the served throughput to within a few points at every measured
configuration.  Removing four per-token host costs moved the number from 1,419
to 1,531 out tok/s, but each of those was a constant-factor reduction of a cost
whose *structure* is unchanged:

    `QwenFastEngine` runs the engine loop as a **thread in the same CPython
    interpreter as uvicorn**, so every event-loop operation on the serving side
    competes for the GIL with a loop that is pure Python between kernel
    launches.

vLLM's structural answer -- an ``EngineCore`` process that owns CUDA, the
model and the scheduler, talking to the API server over ZeroMQ -- is what this
module implements.  Two interpreters, two GILs: the HTTP process can spend as
long as it likes in ``json.dumps``, ``h11`` and the incremental detokenizer
without ever making the engine loop wait for a bytecode-count switch.

Shape
-----
::

    HTTP process                                 engine-core process
    ------------                                 -------------------
    EngineCoreClient (AsyncEngine)               run_core()
      tokenise, auth, meter                        FusedQwenForCausalLM
      asyncio, SSE, detokenise                     Scheduler + graphs + spec
      per-request asyncio.Queue                    one CUDA stream (or two)
            |     ^                                     |     ^
     PUSH   |     | PULL  (one thread, batched)   PULL  |     | PUSH
            v     |                                     v     |
          ipc://...-in / ipc://...-out  (ZeroMQ, unix domain sockets)

**One message per step**, never one per token: the core sends a single
``(MSG_STEP, [(request_id, new_token_ids, finished, finish_reason,
prompt_tokens, completion_tokens, ttft_s), ...])`` carrying the whole batch.
At concurrency 256 that is 1 message where the in-process design did 256
``asyncio.Queue`` writes behind one ``call_soon_threadsafe``, and it is why
the core loop never waits on the HTTP side: the send is non-blocking, and a
send that would block is parked in a local backlog and retried after the next
step rather than stalling the GPU.

The reverse direction is drained non-blockingly once per iteration, so an
``add_request`` costs the loop one ``zmq.Again`` when there is nothing to read.

What is *not* here
------------------
Nothing about the model.  ``run_core`` builds a perfectly ordinary
:class:`~.engine.QwenFastEngine` through the caller's own builder and drives
``engine.scheduler`` directly, so speculative decoding, the graphed mixed
step, ``--overlap`` and the fill guard all run unchanged inside the core --
they live inside ``Scheduler.step()``, which this module does not touch.

In-process mode stays the default (``serve.py`` without ``--engine-process``)
and is what every existing test drives.
"""

from __future__ import annotations

import asyncio
import atexit
import importlib
import multiprocessing
import os
import shutil
import signal
import sys
import tempfile
import threading
import time
import traceback
import uuid
from collections import deque
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple

from ..server.engine_api import (
    AsyncEngine,
    DEFAULT_TPOT_BUCKETS_S,
    DEFAULT_TTFT_BUCKETS_S,
    EngineStats,
    Histogram,
    RequestStats,
    SamplingParams,
    StepOutput,
)

__all__ = [
    "EngineCoreClient",
    "run_core",
    "resolve_builder",
    "CODEC",
]

# =========================================================================== #
# 0. wire format
# =========================================================================== #
# HTTP -> core
MSG_ADD = 1
MSG_ABORT = 2
MSG_SHUTDOWN = 3
MSG_PING = 4
# core -> HTTP
MSG_READY = 10
MSG_STEP = 11
MSG_STATS = 12
MSG_FATAL = 13
MSG_PONG = 14


def _make_codec() -> Tuple[str, Callable[[Any], bytes], Callable[[bytes], Any]]:
    """``msgspec.msgpack`` when it is installed, else ``pickle``.

    Both processes run the same interpreter out of the same venv, so they
    cannot disagree.  msgpack is ~3x faster on the hot message (a list of 256
    small tuples) and, more importantly, cannot execute anything on decode;
    pickle is the fallback so a bare venv still works and so the CPU tests do
    not acquire a dependency.
    """
    try:
        import msgspec.msgpack as _mp  # type: ignore

        enc = _mp.Encoder()
        dec = _mp.Decoder()
        return "msgpack", enc.encode, dec.decode
    except Exception:  # noqa: BLE001
        import pickle

        return (
            "pickle",
            lambda o: pickle.dumps(o, protocol=pickle.HIGHEST_PROTOCOL),
            pickle.loads,
        )


CODEC, _encode, _decode = _make_codec()

#: Input-queue depth, in messages, before ``add_request`` refuses.  Real
#: admission backpressure is ``--max-inflight-requests`` (HTTP 503) and the
#: scheduler's own waiting queue, exactly as in-process; this is the last
#: resort that turns "the core stopped reading" into an error instead of a
#: silent, unbounded memory climb on the serving side.
INPUT_HWM = 8192

#: How long the core waits on its input socket when the scheduler has nothing
#: to do.  Replaces the in-process ``time.sleep(0.0005)`` spin: a poll wakes
#: the instant a request lands, so an idle server's TTFT is not quantised to
#: the sleep interval.
IDLE_POLL_MS = 1


def resolve_builder(spec: str) -> Callable[..., Any]:
    """``"package.module:function"`` -> the function.

    A string, not a pickled callable, so the child resolves it by import in
    its own fresh interpreter and a builder that is only importable (not
    picklable) still works.
    """
    if ":" not in spec:
        raise ValueError(f"builder must be 'module:function', got {spec!r}")
    mod_name, _, fn_name = spec.partition(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name, None)
    if fn is None:
        raise AttributeError(f"{mod_name} has no attribute {fn_name!r}")
    return fn


def _set_pdeathsig() -> None:
    """Ask the kernel to SIGKILL this process if its parent dies.

    A ``daemon=True`` child is reaped on a *clean* parent exit; it is not
    reaped when uvicorn is ``kill -9``'d, and an orphaned core holds ~119 GiB
    of a shared H200 until someone notices.  Linux-only, best effort.
    """
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, signal.SIGKILL, 0, 0, 0)  # PR_SET_PDEATHSIG
    except Exception:  # noqa: BLE001 - macOS, musl, anything: not fatal
        pass


# =========================================================================== #
# 1. the core process
# =========================================================================== #
def run_core(payload: Dict[str, Any]) -> None:
    """Child entry point.  Owns CUDA, the model and the scheduler.

    ``payload`` is everything the child needs, and is deliberately plain data
    (``multiprocessing`` spawn pickles it):

    ``builder`` / ``builder_kwargs``
        ``"module:function"`` returning an **unstarted**
        :class:`~.engine.QwenFastEngine`.
    ``in_addr`` / ``out_addr``
        ZeroMQ endpoints the parent has already bound.
    """
    _set_pdeathsig()
    import zmq

    in_addr = payload["in_addr"]
    out_addr = payload["out_addr"]
    stats_interval_s = float(payload.get("stats_interval_s", 0.25))
    step_trace_out = payload.get("step_trace_out")
    idle_poll_ms = int(payload.get("idle_poll_ms", IDLE_POLL_MS))
    label = payload.get("label", "engine-core")

    ctx = zmq.Context(io_threads=1)
    out = ctx.socket(zmq.PUSH)
    # Unbounded on the output side, matching the in-process design's unbounded
    # `asyncio.Queue`: the reader on the other end is a dedicated thread that
    # only decodes and appends, and a bounded queue here would mean the *GPU*
    # waiting for the HTTP process.
    out.setsockopt(zmq.SNDHWM, 0)
    out.setsockopt(zmq.LINGER, 2000)
    out.connect(out_addr)
    inp = ctx.socket(zmq.PULL)
    inp.setsockopt(zmq.RCVHWM, 0)
    inp.setsockopt(zmq.LINGER, 0)
    inp.connect(in_addr)

    def send(msg) -> None:
        out.send(_encode(msg))

    # -- build ------------------------------------------------------------- #
    try:
        builder = resolve_builder(payload["builder"])
        engine = builder(**payload.get("builder_kwargs", {}))
        engine.prepare()  # warmup + graph capture + the post-capture memory gate
    except BaseException as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        print(tb, file=sys.stderr, flush=True)
        try:
            send((MSG_FATAL, f"engine core failed to start: {type(exc).__name__}: {exc}", tb))
            out.close(linger=2000)
            inp.close(linger=0)
            # `close(linger=...)` only schedules the flush; `term()` blocks until
            # the FATAL frame has left, so the client sees the real cause.
            ctx.term()
        except Exception:  # noqa: BLE001
            pass
        os._exit(1)

    sched = engine.scheduler
    from .scheduler import GenParams, Request
    from .step_trace import StepTrace

    trace = StepTrace(
        step_trace_out,
        meta={
            "mode": "engine-process",
            "pid": os.getpid(),
            "codec": CODEC,
            "label": label,
            **dict(payload.get("trace_meta", {})),
        },
    )
    engine.step_trace = trace

    send((
        MSG_READY,
        {
            "pid": os.getpid(),
            "codec": CODEC,
            "max_context_len": int(sched.max_context_len),
            "eos_token_id": engine.eos_token_id,
            "max_num_seqs": int(engine.rt.max_num_seqs),
            "kv_pages_total": int(sched.stats().kv_pages_total),
            "ssm_slots_total": int(sched.stats().ssm_slots_total),
            "spec": engine.spec is not None,
            "overlap": bool(getattr(engine.rt, "overlap_streams", False)),
        },
    ))
    print(f"[core] ready pid={os.getpid()} codec={CODEC} "
          f"max_context_len={sched.max_context_len}", flush=True)

    poller = zmq.Poller()
    poller.register(inp, zmq.POLLIN)

    stopping = False
    backlog: "deque[bytes]" = deque()
    next_stats = 0.0
    prompt_tokens_total = 0

    def flush_backlog() -> None:
        while backlog:
            try:
                out.send(backlog[0], zmq.NOBLOCK)
            except zmq.Again:
                return
            backlog.popleft()

    def send_step(msg) -> None:
        """Never blocks the loop.  A send that cannot complete right now is
        parked and retried on the next iteration -- the GPU does not wait on
        the HTTP process, which is the entire point of this module."""
        flush_backlog()
        raw = _encode(msg)
        if backlog:
            backlog.append(raw)
            return
        try:
            out.send(raw, zmq.NOBLOCK)
        except zmq.Again:
            backlog.append(raw)

    def drain_input() -> bool:
        """Apply everything waiting on the input socket.  Returns True on
        shutdown."""
        nonlocal stopping, prompt_tokens_total
        while True:
            try:
                raw = inp.recv(zmq.NOBLOCK)
            except zmq.Again:
                return stopping
            msg = _decode(raw)
            op = msg[0]
            if op == MSG_ADD:
                _rid, ids, p = msg[1], msg[2], msg[3]
                params = GenParams(
                    temperature=p[0], top_p=p[1], top_k=p[2], max_tokens=p[3],
                    ignore_eos=bool(p[4]), stop_token_ids=tuple(p[5]),
                    eos_token_id=p[6],
                )
                prompt_tokens_total += len(ids)
                sched.add_request(
                    Request(request_id=_rid, prompt_token_ids=list(ids), params=params)
                )
            elif op == MSG_ABORT:
                sched.abort(msg[1])
            elif op == MSG_PING:
                send((MSG_PONG, msg[1]))
            elif op == MSG_SHUTDOWN:
                stopping = True
                return True

    def maybe_stats(now: float) -> None:
        nonlocal next_stats
        if now < next_stats:
            return
        next_stats = now + stats_interval_s
        s = sched.stats()
        send_step((
            MSG_STATS,
            {
                "num_running": s.num_running,
                "num_waiting": s.num_waiting,
                "kv_pages_used": s.kv_pages_used,
                "kv_pages_total": s.kv_pages_total,
                "ssm_slots_used": s.ssm_slots_used,
                "ssm_slots_total": s.ssm_slots_total,
                "spec_accept_length": s.spec_accept_length if engine.spec is not None else None,
                "spec_acceptance_rate": s.spec_acceptance_rate if engine.spec is not None else None,
                "prompt_tokens_total": prompt_tokens_total,
                "step_busy_pct": trace.step_busy_pct,
            },
        ))

    # -- the loop ----------------------------------------------------------- #
    perf = time.perf_counter
    try:
        while True:
            t_start = perf()
            if drain_input():
                # Under `--async-scheduling` the last launched
                # step's tokens are still on the device when SHUTDOWN arrives.
                # Commit and send them, or the final token of every in-flight
                # request is dropped on the way out.
                try:
                    tail = sched.drain()
                except Exception:  # noqa: BLE001 - shutdown must not raise
                    tail = []
                if tail:
                    send_step((MSG_STEP, [
                        (e.request.request_id, e.new_token_ids, e.finished,
                         e.finish_reason, len(e.request.prompt_token_ids),
                         len(e.request.output_token_ids), e.request.ttft_s)
                        for e in tail
                    ]))
                    flush_backlog()
                break
            t_drained = perf()

            if not sched.has_work():
                flush_backlog()
                poller.poll(timeout=idle_poll_ms)
                t_end = perf()
                trace.record_idle(t_start=t_start, t_drained=t_drained, t_end=t_end)
                maybe_stats(t_end)
                continue

            events = sched.step()
            t_stepped = perf()

            if events:
                rows = [
                    (
                        e.request.request_id,
                        e.new_token_ids,
                        e.finished,
                        e.finish_reason,
                        len(e.request.prompt_token_ids),
                        len(e.request.output_token_ids),
                        e.request.ttft_s,
                    )
                    for e in events
                ]
                send_step((MSG_STEP, rows))
            else:
                flush_backlog()
            t_end = perf()
            trace.record(
                t_start=t_start, t_drained=t_drained, t_stepped=t_stepped, t_end=t_end,
                scheduler=sched, events=events,
            )
            maybe_stats(t_end)
            if not sched.last_step_progressed:
                # Nothing could be scheduled -- every waiting request is blocked
                # on capacity held by a running one.  Wait on the socket rather
                # than spin (the in-process loop's `time.sleep(0.0005)`).
                poller.poll(timeout=idle_poll_ms)
    except BaseException as exc:  # noqa: BLE001 - last line of defence
        tb = traceback.format_exc()
        print(tb, file=sys.stderr, flush=True)
        try:
            send((MSG_FATAL, f"engine core died: {type(exc).__name__}: {exc}", tb))
        except Exception:  # noqa: BLE001
            pass
        trace.write()
        print(trace.one_line(), flush=True)
        try:
            out.close(linger=2000)
        except Exception:  # noqa: BLE001
            pass
        os._exit(1)

    # -- clean shutdown ------------------------------------------------------ #
    path = trace.write()
    print(trace.one_line(), flush=True)
    if path:
        print(f"[core] step trace -> {path}", flush=True)
    try:
        flush_backlog()
        out.close(linger=1000)
        inp.close(linger=0)
        ctx.term()
    except Exception:  # noqa: BLE001
        pass
    # `os._exit`, not `return`: the child holds a CUDA context and 119 GiB of
    # pool allocations, and torch's interpreter-shutdown teardown of those has
    # been known to hang for minutes.  The parent is waiting on `join()`.
    os._exit(0)


# =========================================================================== #
# 2. the client (HTTP-process side)
# =========================================================================== #
def _deliver(deliveries: "List[Tuple[asyncio.Queue, Optional[StepOutput]]]") -> None:
    """Run on the event loop: push one batch of step outputs into their queues.

    Identical in shape to ``engine._deliver``: no awaits, O(len), and one
    event-loop wakeup for the whole batch however many steps it spans.
    """
    for q, out in deliveries:
        q.put_nowait(out)


class EngineCoreClient(AsyncEngine):
    """The ``AsyncEngine`` the HTTP process talks to when the engine loop lives
    in its own process.

    Deliberately the same surface -- and the same failure semantics -- as
    :class:`~.engine.QwenFastEngine`, so ``server/app.py`` cannot tell them
    apart: ``health()`` returns a one-line reason once the core is gone,
    every in-flight generator is released with the ``None`` sentinel, and
    ``/health`` 503s exactly as it did when the engine was a thread.
    """

    def __init__(
        self,
        builder: str,
        builder_kwargs: Optional[Dict[str, Any]] = None,
        *,
        start_timeout_s: float = 2400.0,
        stats_interval_s: float = 0.25,
        step_trace_out: Optional[str] = None,
        idle_poll_ms: int = IDLE_POLL_MS,
        ipc_dir: Optional[str] = None,
        trace_meta: Optional[Dict[str, Any]] = None,
        label: str = "engine-core",
        eos_token_id: Optional[int] = None,
        verbose: bool = True,
    ) -> None:
        self.builder = builder
        self.builder_kwargs = dict(builder_kwargs or {})
        self.start_timeout_s = float(start_timeout_s)
        self.stats_interval_s = float(stats_interval_s)
        self.step_trace_out = step_trace_out
        self.idle_poll_ms = int(idle_poll_ms)
        self.trace_meta = dict(trace_meta or {})
        self.label = label
        self.verbose = verbose

        self._ipc_dir = ipc_dir
        self._own_ipc_dir = ipc_dir is None
        self._ctx = None
        self._to_core = None
        self._from_core = None
        self._proc: Optional[multiprocessing.process.BaseProcess] = None
        self._reader: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._send_lock = threading.Lock()

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._lock = threading.Lock()
        self._out_queues: Dict[str, "asyncio.Queue[Optional[StepOutput]]"] = {}
        self._last_emit_t: Dict[str, float] = {}

        self._fatal_error: Optional[str] = None
        self._fatal_traceback: Optional[str] = None

        self._start_time = time.monotonic()
        self._prompt_tokens_total = 0
        self._generation_tokens_total = 0
        self._ttft_hist = Histogram(DEFAULT_TTFT_BUCKETS_S)
        self._tpot_hist = Histogram(DEFAULT_TPOT_BUCKETS_S)
        self._core_stats: Dict[str, Any] = {}
        self._ready: Dict[str, Any] = {}

        # Filled from the core's READY message; `server/app.py` reads both.
        self.max_context_len = 0
        self.eos_token_id = eos_token_id

    # -- lifecycle ----------------------------------------------------------- #
    def _bind(self) -> None:
        import zmq

        if self._ipc_dir is None:
            self._ipc_dir = tempfile.mkdtemp(prefix="qwenfast-ipc-")
        tag = uuid.uuid4().hex[:8]
        self._in_addr = f"ipc://{os.path.join(self._ipc_dir, f'in-{tag}.sock')}"
        self._out_addr = f"ipc://{os.path.join(self._ipc_dir, f'out-{tag}.sock')}"
        self._ctx = zmq.Context(io_threads=1)
        self._to_core = self._ctx.socket(zmq.PUSH)
        self._to_core.setsockopt(zmq.SNDHWM, INPUT_HWM)
        self._to_core.setsockopt(zmq.LINGER, 1000)
        self._to_core.bind(self._in_addr)
        self._from_core = self._ctx.socket(zmq.PULL)
        self._from_core.setsockopt(zmq.RCVHWM, 0)
        self._from_core.setsockopt(zmq.LINGER, 0)
        self._from_core.bind(self._out_addr)

    async def start(self) -> None:
        self._loop = asyncio.get_event_loop()
        self._bind()
        payload = {
            "builder": self.builder,
            "builder_kwargs": self.builder_kwargs,
            "in_addr": self._in_addr,
            "out_addr": self._out_addr,
            "stats_interval_s": self.stats_interval_s,
            "step_trace_out": self.step_trace_out,
            "idle_poll_ms": self.idle_poll_ms,
            "trace_meta": self.trace_meta,
            "label": self.label,
        }
        # `spawn`, not `fork`: the child initialises CUDA, and a forked child
        # of a parent that has ever touched CUDA cannot.  It also gives the
        # core a clean interpreter -- which is the entire point.
        mp = multiprocessing.get_context("spawn")
        # `daemon=False`, deliberately.  A daemonic process may not create
        # children of its own, and the core's *startup* shells out: Triton
        # autotune and FlashInfer's JIT both invoke a compiler through
        # `subprocess`, and some builds compile modules in parallel.  Losing a
        # 70-minute GPU window to `AssertionError: daemonic processes are not
        # allowed to have children` is not a trade worth making for automatic
        # reaping, so the reaping is done explicitly instead, three ways:
        # `shutdown()` on the normal path, the `atexit` hook below if uvicorn
        # dies without running its lifespan, and `PR_SET_PDEATHSIG` in the
        # child for the `kill -9` case that neither can catch.  (`atexit` is
        # LIFO and `multiprocessing`'s own join-children hook was registered
        # when this module was imported, so this one runs first and the join
        # finds a dead child rather than hanging on a live one.)
        self._proc = mp.Process(
            target=run_core, args=(payload,), name="qwenfast-engine-core", daemon=False
        )
        self._proc.start()
        atexit.register(self._kill_child)
        if self.verbose:
            print(
                f"[serve] engine core pid={self._proc.pid} codec={CODEC} "
                f"ipc={self._ipc_dir}",
                flush=True,
            )
        info = await self._loop.run_in_executor(None, self._await_ready)
        self._ready = info
        self.max_context_len = int(info["max_context_len"])
        if self.eos_token_id is None:
            self.eos_token_id = info.get("eos_token_id")
        self._reader = threading.Thread(
            target=self._read_loop, name="qwenfast-core-reader", daemon=True
        )
        self._reader.start()

    def _await_ready(self) -> Dict[str, Any]:
        """Block (on a worker thread) until the core says it is up.

        Loads the checkpoint, fuses the weights and captures every graph, so
        the timeout is minutes, not seconds.  A core that dies on the way (a
        CUDA OOM during load or capture, or the post-capture memory gate)
        raises here, which fails uvicorn's lifespan and exits the server with
        a non-zero status instead of leaving a live HTTP port in front of a
        dead engine.
        """
        import zmq

        deadline = time.monotonic() + self.start_timeout_s
        poller = zmq.Poller()
        poller.register(self._from_core, zmq.POLLIN)
        while True:
            if dict(poller.poll(timeout=500)):
                msg = _decode(self._from_core.recv())
                if msg[0] == MSG_READY:
                    return msg[1]
                if msg[0] == MSG_FATAL:
                    raise RuntimeError(msg[1] + "\n" + (msg[2] or ""))
                continue  # a STATS that raced READY; ignore
            if self._proc is not None and not self._proc.is_alive():
                # The core may have sent FATAL just before exiting: drain it first.
                if dict(poller.poll(timeout=200)):
                    msg = _decode(self._from_core.recv())
                    if msg[0] == MSG_FATAL:
                        raise RuntimeError(msg[1] + "\n" + (msg[2] or ""))
                    if msg[0] == MSG_READY:
                        return msg[1]
                raise RuntimeError(
                    f"engine core exited with code {self._proc.exitcode} before becoming ready"
                )
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"engine core did not become ready within {self.start_timeout_s:.0f}s"
                )

    async def shutdown(self) -> None:
        self._stop.set()
        try:
            self._send((MSG_SHUTDOWN,), raise_on_full=False)
        except Exception:  # noqa: BLE001
            pass
        if self._reader is not None:
            self._reader.join(timeout=2.0)
            self._reader = None
        p = self._proc
        if p is not None:
            p.join(timeout=20.0)
            if p.is_alive():
                p.terminate()
                p.join(timeout=10.0)
            if p.is_alive():
                p.kill()
                p.join(timeout=5.0)
            self._proc = None
            try:
                atexit.unregister(self._kill_child)
            except Exception:  # noqa: BLE001
                pass
        for s in (self._to_core, self._from_core):
            try:
                if s is not None:
                    s.close(linger=0)
            except Exception:  # noqa: BLE001
                pass
        self._to_core = self._from_core = None
        try:
            if self._ctx is not None:
                self._ctx.term()
        except Exception:  # noqa: BLE001
            pass
        self._ctx = None
        if self._own_ipc_dir and self._ipc_dir:
            shutil.rmtree(self._ipc_dir, ignore_errors=True)
            self._ipc_dir = None

    def _kill_child(self) -> None:
        """Last-resort reaper (``atexit``).  Never raises, never blocks long."""
        p = self._proc
        if p is None:
            return
        try:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5.0)
            if p.is_alive():
                p.kill()
                p.join(timeout=2.0)
        except Exception:  # noqa: BLE001
            pass

    # -- liveness ------------------------------------------------------------ #
    def health(self) -> Optional[str]:
        if self._fatal_error is not None:
            return self._fatal_error
        p = self._proc
        if p is not None and not p.is_alive() and not self._stop.is_set():
            return f"engine core process is not running (exit code {p.exitcode})"
        return None

    def _die(self, reason: str, tb: Optional[str] = None) -> None:
        if self._fatal_error is None:
            self._fatal_error = reason
            self._fatal_traceback = tb
            print(f"[serve] ENGINE CORE FAILED: {reason}", flush=True)
            if tb:
                print(tb, file=sys.stderr, flush=True)
        with self._lock:
            queues = list(self._out_queues.values())
        loop = self._loop
        if loop is None:
            return
        # `None` is `_add_request`'s "engine gone mid-stream" sentinel; every
        # waiter returns at once instead of hanging until the client times out.
        try:
            loop.call_soon_threadsafe(_deliver, [(q, None) for q in queues])
        except RuntimeError:  # loop already closed
            pass

    # -- the reader thread ---------------------------------------------------- #
    def _read_loop(self) -> None:
        import zmq

        poller = zmq.Poller()
        poller.register(self._from_core, zmq.POLLIN)
        sock = self._from_core
        loop = self._loop
        while not self._stop.is_set():
            try:
                ready = dict(poller.poll(timeout=100))
            except Exception:  # noqa: BLE001 - socket closed under us
                return
            if not ready:
                p = self._proc
                if p is not None and not p.is_alive() and not self._stop.is_set():
                    self._die(
                        f"engine core process exited with code {p.exitcode}"
                    )
                    return
                continue

            deliveries: List[Tuple[asyncio.Queue, Optional[StepOutput]]] = []
            fatal: Optional[Tuple[str, str]] = None
            # Drain everything that has arrived, so N steps' worth of output
            # costs the event loop **one** wakeup rather than N.
            while True:
                try:
                    raw = sock.recv(zmq.NOBLOCK)
                except zmq.Again:
                    break
                except Exception:  # noqa: BLE001
                    return
                msg = _decode(raw)
                op = msg[0]
                if op == MSG_STEP:
                    self._on_step(msg[1], deliveries)
                elif op == MSG_STATS:
                    self._core_stats = msg[1]
                elif op == MSG_FATAL:
                    fatal = (msg[1], msg[2] if len(msg) > 2 else "")
                    break
                elif op == MSG_READY:
                    self._ready = msg[1]
            if deliveries and loop is not None:
                try:
                    loop.call_soon_threadsafe(_deliver, deliveries)
                except RuntimeError:
                    return
            if fatal is not None:
                self._die(fatal[0], fatal[1])
                return

    def _on_step(self, rows, deliveries) -> None:
        """Turn one step's batched message into per-request ``StepOutput``s.

        Runs on the reader thread, i.e. in the HTTP process -- which is the
        point: the TTFT/TPOT histograms and the ``StepOutput`` objects used to
        be built on the engine thread, between two kernel launches.
        """
        now = time.monotonic()
        with self._lock:
            queues = dict(self._out_queues)
        for rid, new_ids, finished, finish_reason, n_prompt, n_completion, ttft_s in rows:
            if new_ids:
                self._generation_tokens_total += len(new_ids)
                last = self._last_emit_t.get(rid)
                if last is None:
                    if ttft_s is not None:
                        self._ttft_hist.observe(ttft_s)
                else:
                    self._tpot_hist.observe(now - last)
                self._last_emit_t[rid] = now
            if finished:
                self._last_emit_t.pop(rid, None)
            q = queues.get(rid)
            if q is None:
                continue
            deliveries.append((
                q,
                StepOutput(
                    new_token_ids=list(new_ids),
                    finished=bool(finished),
                    finish_reason=finish_reason,
                    stats=RequestStats(
                        prompt_tokens=n_prompt,
                        completion_tokens=n_completion,
                        ttft_s=ttft_s,
                        spec_accept_length=None,
                    ),
                ),
            ))

    # -- sending -------------------------------------------------------------- #
    def _send(self, msg, *, raise_on_full: bool = True) -> None:
        import zmq

        sock = self._to_core
        if sock is None:
            if raise_on_full:
                raise RuntimeError("engine core transport is closed")
            return
        raw = _encode(msg)
        with self._send_lock:
            try:
                sock.send(raw, zmq.NOBLOCK)
            except zmq.Again:
                if raise_on_full:
                    raise RuntimeError(
                        "engine core input queue is full "
                        f"({INPUT_HWM} messages); the core is not draining"
                    )
            except Exception as exc:  # noqa: BLE001
                if raise_on_full:
                    raise RuntimeError(f"engine core transport error: {exc}") from exc

    # -- AsyncEngine ----------------------------------------------------------- #
    def context_length_error(self, n_prompt: int, max_tokens: int) -> Optional[str]:
        from .scheduler import context_length_error

        if not self.max_context_len:
            return None
        return context_length_error(n_prompt, max_tokens, self.max_context_len)

    def add_request(
        self,
        request_id: str,
        prompt_token_ids: List[int],
        sampling_params: SamplingParams,
    ) -> AsyncIterator[StepOutput]:
        return self._add_request(request_id, prompt_token_ids, sampling_params)

    async def _add_request(
        self,
        request_id: str,
        prompt_token_ids: List[int],
        sampling_params: SamplingParams,
    ) -> AsyncIterator[StepOutput]:
        dead = self.health()
        if dead is not None:
            raise RuntimeError(dead)
        too_long = self.context_length_error(len(prompt_token_ids), sampling_params.max_tokens)
        if too_long is not None:
            raise ValueError(too_long)

        out_q: "asyncio.Queue[Optional[StepOutput]]" = asyncio.Queue()
        with self._lock:
            self._out_queues[request_id] = out_q

        params = (
            sampling_params.temperature,
            sampling_params.top_p,
            sampling_params.top_k,
            sampling_params.max_tokens,
            sampling_params.ignore_eos,
            list(sampling_params.stop_token_ids),
            self.eos_token_id,
        )
        self._prompt_tokens_total += len(prompt_token_ids)
        try:
            self._send((MSG_ADD, request_id, list(prompt_token_ids), params))
        except Exception:
            with self._lock:
                self._out_queues.pop(request_id, None)
            raise

        try:
            while True:
                item = await out_q.get()
                if item is None:
                    dead = self.health()
                    if dead is not None:
                        raise RuntimeError(dead)
                    return
                yield item
                if item.finished:
                    return
        finally:
            with self._lock:
                self._out_queues.pop(request_id, None)

    async def abort(self, request_id: str) -> None:
        try:
            self._send((MSG_ABORT, request_id), raise_on_full=False)
        except Exception:  # noqa: BLE001 - abort is best effort by contract
            pass

    def get_stats(self) -> EngineStats:
        """Non-blocking by construction: the core *pushes* a snapshot every
        ``--engine-stats-interval`` seconds and this returns the last one.  A
        request/response round trip here would put a ``/metrics`` scrape --
        which Prometheus does every 15 s, forever -- into the engine loop's
        critical path."""
        cs = self._core_stats
        uptime = max(time.monotonic() - self._start_time, 1e-9)
        return EngineStats(
            num_requests_running=int(cs.get("num_running", 0)),
            num_requests_waiting=int(cs.get("num_waiting", 0)),
            prompt_tokens_total=self._prompt_tokens_total,
            generation_tokens_total=self._generation_tokens_total,
            tokens_per_second=self._generation_tokens_total / uptime,
            ttft=self._ttft_hist.snapshot(),
            tpot=self._tpot_hist.snapshot(),
            kv_slots_used=int(cs.get("kv_pages_used", 0)),
            kv_slots_total=int(cs.get("kv_pages_total", self._ready.get("kv_pages_total", 0))),
            ssm_slots_used=int(cs.get("ssm_slots_used", 0)),
            ssm_slots_total=int(cs.get("ssm_slots_total", self._ready.get("ssm_slots_total", 0))),
            spec_accept_length=cs.get("spec_accept_length"),
            spec_acceptance_rate=cs.get("spec_acceptance_rate"),
            uptime_s=uptime,
        )
