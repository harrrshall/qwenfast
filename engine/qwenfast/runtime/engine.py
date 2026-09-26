"""The ``AsyncEngine`` contract implemented over the qwenfast runtime.

``QwenFastEngine`` owns one background thread running the single-threaded
engine loop (one CUDA stream, one
:class:`~.scheduler.Scheduler`); everything asyncio-facing (``add_request``,
``abort``, ``get_stats``) talks to that thread through thread-safe queues,
per ``server/engine_api.py``'s contract:

    engine.add_request(request_id, prompt_token_ids, sampling_params) -> AsyncIterator[StepOutput]
    engine.abort(request_id) -> None
    engine.get_stats() -> EngineStats

This module is the only place in ``runtime/`` that imports from
``qwenfast.server`` -- a read-only dependency on the frozen interface in
``server/engine_api.py``. ``server/cli.py``'s ``--engine qwenfast`` branch
is the entry point that constructs it.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from dataclasses import dataclass
from typing import AsyncIterator, Dict, Optional

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
from .fused_model import DeviceBuffers, FusedQwenForCausalLM, RuntimeConfig
from .graphs import GraphedDecoder
from .mixed_graphs import MixedGraphRunner
from .scheduler import GenParams, Request, Scheduler, StepEvent
from .spec_decode import SpecConfig, SpecDecoder, build_spec_decoder

_POLL_SLEEP_S = 0.0005  # spin interval when the scheduler has no work


# =========================================================================== #
# 1. component builders (the `runtime/__init__.py` lazy re-exports)
# =========================================================================== #
@dataclass
class EngineComponents:
    """The pieces benchmarks such as ``bench_runtime.py`` want direct access
    to, without going through the async request queue."""

    model: FusedQwenForCausalLM
    buf: DeviceBuffers
    decoder: GraphedDecoder
    rt: RuntimeConfig
    #: Speculative decoder: present only when ``build_engine(spec=SpecConfig(...))`` was used.
    spec: Optional[SpecDecoder] = None
    #: Graphed mixed-step runner: present only when ``rt.mixed_forward and rt.mixed_graphs``.
    mixed: Optional[MixedGraphRunner] = None


def build_engine(
    model_dir: str,
    rt: Optional[RuntimeConfig] = None,
    *,
    fused_cache: Optional[str] = None,
    verbose: bool = False,
    spec: Optional[SpecConfig] = None,
    **rt_kwargs,
) -> EngineComponents:
    """Load weights + build the model/buffers/graphed-decoder triple.

    Does **not** call ``decoder.warmup()``/``.capture()`` -- callers that
    want a graphed engine must do that themselves (``QwenFastEngine.start``
    does; offline scripts call it explicitly so they control when the
    capture-time cost is paid).
    """
    if rt is None:
        rt = RuntimeConfig(**rt_kwargs)
    elif rt_kwargs:
        raise ValueError("pass either `rt=` or `**rt_kwargs`, not both")

    model = FusedQwenForCausalLM.from_pretrained(model_dir, rt, fused_cache=fused_cache, verbose=verbose)
    max_batch = rt.buckets_for()[-1] if rt.use_cuda_graphs else rt.max_num_seqs
    buf = DeviceBuffers(
        max_batch=max_batch,
        vocab_size=model.config.vocab_size,
        max_pages=rt.n_kv_pages,
        device=model.device,
    )
    decoder = GraphedDecoder(model, buf, rt)
    spec_decoder = build_spec_decoder(model, buf, rt, spec) if spec is not None else None
    return EngineComponents(
        model=model, buf=buf, decoder=decoder, rt=rt, spec=spec_decoder,
        mixed=build_mixed_runner(model, rt),
    )


def build_mixed_runner(
    model: FusedQwenForCausalLM, rt: RuntimeConfig
) -> Optional[MixedGraphRunner]:
    """The graphed mixed-step runner, or ``None``.

    One runner for the one ``prefill_chunk_tokens`` the server is configured
    with (see :class:`~.mixed_graphs.MixedGraphRunner`); ``None`` unless both
    ``--mixed-forward`` and ``--mixed-graphs`` are on, so a build that has not
    opted in is byte-identical to one without the feature.
    """
    if not (rt.mixed_forward and rt.mixed_graphs):
        return None
    chunk = rt.prefill_chunk_tokens or rt.max_num_batched_tokens
    chunk = min(int(chunk), int(rt.max_num_batched_tokens))
    # `--overlap` runs the decode rows as their own graph on a second
    # stream, so its "mixed" graph is a prefill chunk plus exactly one padding
    # decode row -- one bucket, one capture.
    # An explicit ladder (`--mixed-graph-buckets`, and `--overlap`'s implied
    # `(1,)`) means exactly those buckets: `mixed_graph_min_bucket` is the
    # *derived* ladder's trim rule, and applying it to a hand-written one
    # would silently drop the bucket that was asked for.
    explicit = bool(rt.mixed_graph_buckets) or rt.overlap_streams
    return MixedGraphRunner(model, rt, chunk_tokens=chunk,
                            buckets=rt.mixed_buckets_for(),
                            min_bucket=1 if explicit else None)


def build_async_engine(
    model_dir: str,
    rt: Optional[RuntimeConfig] = None,
    spec: Optional[SpecConfig] = None,
    spec_max_batch: Optional[int] = None,
    **rt_kwargs,
) -> "QwenFastEngine":
    """The factory ``server/cli.py``'s ``--engine qwenfast`` branch uses:
    build the model and wrap it in the
    ``AsyncEngine`` contract. Graph capture happens lazily in
    :meth:`QwenFastEngine.start`, which the server calls once before it
    starts accepting traffic (``AsyncEngine.start``).

    ``spec_max_batch``: forwarded to the
    :class:`~.scheduler.Scheduler` as the per-step batch-size cap above which
    an eligible spec step still runs plain (spec decoding wins at B<=8 and
    loses at B>=128). Meaningless without ``spec`` and ignored if it is ``None``.
    """
    comps = build_engine(model_dir, rt=rt, spec=spec, **rt_kwargs)
    return QwenFastEngine(
        comps.model, comps.decoder, comps.rt, spec=comps.spec,
        spec_max_batch=spec_max_batch, mixed=comps.mixed,
    )


# =========================================================================== #
# 2. the engine
# =========================================================================== #
class QwenFastEngine(AsyncEngine):
    def __init__(
        self,
        model: FusedQwenForCausalLM,
        decoder: GraphedDecoder,
        rt: RuntimeConfig,
        *,
        eos_token_id: Optional[int] = None,
        capture_graphs: bool = True,
        spec: Optional[SpecDecoder] = None,
        spec_max_batch: Optional[int] = None,
        mixed: Optional[MixedGraphRunner] = None,
    ):
        self.model = model
        self.decoder = decoder
        self.rt = rt
        self.spec = spec
        # The graphed mixed step, captured in `start()`
        # next to the decode graphs and sharing their mempool.
        self.mixed = mixed
        self.eos_token_id = eos_token_id if eos_token_id is not None else model.config.eos_token_id
        self._capture_graphs = capture_graphs

        self.scheduler = Scheduler(
            model, decoder, rt, spec=spec, spec_max_batch=spec_max_batch,
            mixed_runner=mixed,
        )

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._pending: "queue.Queue[Request]" = queue.Queue()
        self._abort_queue: "queue.Queue[str]" = queue.Queue()
        self._out_queues: Dict[str, "asyncio.Queue[Optional[StepOutput]]"] = {}
        self._last_emit_t: Dict[str, float] = {}

        self._start_time = time.monotonic()
        self._prompt_tokens_total = 0
        self._generation_tokens_total = 0
        self._ttft_hist = Histogram(DEFAULT_TTFT_BUCKETS_S)
        self._tpot_hist = Histogram(DEFAULT_TPOT_BUCKETS_S)

        # -- liveness ----------------------------------------------------------
        # The engine loop runs on a daemon thread. An exception in
        # `scheduler.step()` (a CUDA OOM, say) kills that thread, and without
        # this nothing else would notice: uvicorn keeps answering, /health keeps
        # saying "ok", and every in-flight generator waits on `out_q.get()`
        # until the client's own timeout. The error is recorded here, every
        # waiter is released, and `health()` reports it so the HTTP layer can
        # return 503 and a benchmark can abort.
        self._fatal_error: Optional[BaseException] = None
        self._engine_died_at: Optional[float] = None
        #: Optional callback run inside `start()` *after* graph capture and
        #: *before* the loop thread starts. `runtime.serve` installs the
        #: post-capture memory gate here; raising aborts startup.
        self.post_capture_check = None
        self.memory_plan: Optional[Dict[str, float]] = None
        self.memory_budget_gib: Optional[float] = None
        #: ``None`` = not tracing (the default, and free).
        #: A :class:`~.step_trace.StepTrace` here splits the loop's wall clock
        #: into step / drain / emit / gap / idle -- the instrument that can see
        #: the between-step idle `profile_serving.attribute` is arithmetically
        #: blind to. ``runtime.serve --step-trace-out`` installs one; the
        #: engine-core process installs its own.
        self.step_trace = None
        #: ``None`` = not profiling (the default, free).
        #: A :class:`~.step_trace.StepProfiler` here opens the *step* up the
        #: way ``step_trace`` opens the loop up: host phases plus CUDA events,
        #: for a bounded window of steps. ``runtime.serve --step-profile-out``
        #: installs one and points ``scheduler.profiler`` at it.
        self.step_profiler = None

    # -- lifecycle ------------------------------------------------------------ #
    async def start(self) -> None:
        self._loop = asyncio.get_event_loop()
        self.prepare()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="qwenfast-engine", daemon=True)
        self._thread.start()

    def prepare(self) -> None:
        """Everything ``start()`` does *except* launch the loop thread.

        Split out for the engine-core process, which runs exactly
        this -- warmup, graph capture, the post-capture memory gate -- and then
        drives ``self.scheduler`` from its own loop instead of a thread, so
        there is one place where "how a qwenfast engine is brought up" is
        written and the two modes cannot drift.  Synchronous, because none of
        it is async and the core process has no event loop at all.
        """
        self.decoder.warmup()
        if self._capture_graphs:
            self.decoder.capture()
        if self.spec is not None:
            self.spec.warmup()
            if self._capture_graphs:
                # share the decoder's mempool: two step shapes, one ~1 GiB pool
                self.spec.capture(pool_handle=self.decoder._pool)
        if self.mixed is not None:
            # Same order and the same shared mempool: warm every shape
            # eagerly (GEMM backends, Triton/fla/FlashInfer compilation) and
            # only then capture, because none of that may happen inside a
            # capture. `--no-graphs` leaves the runner warm but uncaptured, and
            # the scheduler then runs the same padded step eagerly.
            self.mixed.warmup()
            if self._capture_graphs:
                # `--overlap` replays this graph *concurrently* with a
                # decode graph, so it needs its own mempool -- sharing one is
                # only safe while exactly one graph runs at a time.
                self.mixed.capture(
                    pool_handle=None if self.rt.overlap_streams else self.decoder._pool
                )
        if self.post_capture_check is not None:
            # Deliberately *before* the loop thread starts and before the HTTP
            # layer is serving: if the measured footprint does not fit, the
            # process must fail to start, not fail on the first request.
            self.post_capture_check()

    # -- liveness --------------------------------------------------------------- #
    def health(self) -> Optional[str]:
        """``None`` when the engine can serve, else a one-line reason.

        The two ways to be unhealthy are "the loop thread raised" and "the loop
        thread is gone without having been asked to stop"; both are terminal.
        """
        if self._fatal_error is not None:
            return f"engine thread died: {type(self._fatal_error).__name__}: {self._fatal_error}"
        t = self._thread
        if t is not None and not t.is_alive() and not self._stop_event.is_set():
            return "engine thread is not running"
        return None

    def _die(self, exc: BaseException) -> None:
        """Record a fatal engine error and release everyone waiting on it."""
        self._fatal_error = exc
        self._engine_died_at = time.monotonic()
        with self._lock:
            queues = list(self._out_queues.values())
        loop = self._loop
        for q in queues:
            # `None` is `_add_request`'s "engine shut down mid-stream" sentinel;
            # the waiter returns immediately instead of hanging forever.
            if loop is not None:
                loop.call_soon_threadsafe(q.put_nowait, None)

    async def shutdown(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self.step_trace is not None and self.step_trace.n_steps:
            path = self.step_trace.write()
            print(self.step_trace.one_line(), flush=True)
            if path:
                print(f"[serve] step trace -> {path}", flush=True)
        if self.step_profiler is not None and self.step_profiler.seen:
            path = self.step_profiler.write()
            print(self.step_profiler.one_line(), flush=True)
            if path:
                print(f"[serve] step profile -> {path}", flush=True)

    # -- admission limits ------------------------------------------------------- #
    @property
    def max_context_len(self) -> int:
        """Longest ``prompt + completion`` this engine can serve, in tokens.

        ``server/app.py`` reads this (and :meth:`context_length_error`) off the
        engine object rather than importing anything from ``runtime/``, so a
        mock engine without them simply skips the check."""
        return self.scheduler.max_context_len

    def context_length_error(self, n_prompt: int, max_tokens: int) -> Optional[str]:
        """``None`` if the request fits, else the message to 400 with."""
        return self.scheduler.context_length_error(n_prompt, max_tokens)

    # -- AsyncEngine ------------------------------------------------------------ #
    def add_request(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
    ) -> AsyncIterator[StepOutput]:
        return self._add_request(request_id, prompt_token_ids, sampling_params)

    async def _add_request(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
    ) -> AsyncIterator[StepOutput]:
        # Fail fast rather than admit work to a dead loop: the caller sees a
        # 503 on request 1 instead of a stream that never produces a token.
        dead = self.health()
        if dead is not None:
            raise RuntimeError(dead)

        # A request whose prompt + completion runs past
        # the rotary table is a *client* error, not an engine failure. The HTTP
        # layer normally turns this into a 400 before we get here; raising is
        # the backstop for direct callers.
        too_long = self.context_length_error(len(prompt_token_ids), sampling_params.max_tokens)
        if too_long is not None:
            raise ValueError(too_long)

        out_q: "asyncio.Queue[Optional[StepOutput]]" = asyncio.Queue()
        with self._lock:
            self._out_queues[request_id] = out_q

        params = GenParams(
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k,
            max_tokens=sampling_params.max_tokens,
            ignore_eos=sampling_params.ignore_eos,
            stop_token_ids=tuple(sampling_params.stop_token_ids),
            eos_token_id=self.eos_token_id,
        )
        req = Request(request_id=request_id, prompt_token_ids=list(prompt_token_ids), params=params)
        self._prompt_tokens_total += len(prompt_token_ids)
        self._pending.put(req)

        try:
            while True:
                item = await out_q.get()
                if item is None:  # engine shut down mid-stream
                    dead = self.health()
                    if dead is not None:
                        # A crash, not a clean shutdown: surface it so the HTTP
                        # layer can 5xx the in-flight stream rather than end it
                        # silently and look like a very short completion.
                        raise RuntimeError(dead)
                    return
                yield item
                if item.finished:
                    return
        finally:
            with self._lock:
                self._out_queues.pop(request_id, None)

    async def abort(self, request_id: str) -> None:
        self._abort_queue.put(request_id)

    def get_stats(self) -> EngineStats:
        s = self.scheduler.stats()
        uptime = max(time.monotonic() - self._start_time, 1e-9)
        return EngineStats(
            num_requests_running=s.num_running,
            num_requests_waiting=s.num_waiting,
            prompt_tokens_total=self._prompt_tokens_total,
            generation_tokens_total=self._generation_tokens_total,
            tokens_per_second=self._generation_tokens_total / uptime,
            ttft=self._ttft_hist.snapshot(),
            tpot=self._tpot_hist.snapshot(),
            kv_slots_used=s.kv_pages_used,
            kv_slots_total=s.kv_pages_total,
            ssm_slots_used=s.ssm_slots_used,
            ssm_slots_total=s.ssm_slots_total,
            # Real once `spec` is configured (`Scheduler.stats()` reads
            # `SpecDecoder.stats()`); `None` -- not 0.0 -- when spec decoding
            # is off at all, so `/metrics` omits the gauge entirely rather
            # than reporting a misleading zero (`metrics.render_prometheus_text`
            # only emits it when not `None`).
            spec_accept_length=s.spec_accept_length if self.spec is not None else None,
            spec_acceptance_rate=s.spec_acceptance_rate if self.spec is not None else None,
            prefix_hits=s.prefix_hits,
            prefix_lookups=s.prefix_lookups,
            prefix_hit_tokens=s.prefix_hit_tokens,
            prefix_prompt_tokens=s.prefix_prompt_tokens,
            prefix_entries=s.prefix_entries,
            uptime_s=uptime,
        )

    # -- background thread ------------------------------------------------------- #
    def _run_loop(self) -> None:
        try:
            self._run_loop_body()
        except BaseException as exc:  # noqa: BLE001 -- last line of defence
            # Anything that reaches here (a CUDA OOM in `scheduler.step()` is
            # the real case) has already taken the engine down; the job of this
            # handler is to make that *observable* instead of a silent hang.
            import traceback

            traceback.print_exc()
            self._die(exc)
            raise

    def _run_loop_body(self) -> None:
        trace = self.step_trace
        perf = time.perf_counter
        sched = self.scheduler
        while not self._stop_event.is_set():
            t_start = perf()
            self._drain_pending()
            t_drained = perf()
            if not sched.has_work():
                time.sleep(_POLL_SLEEP_S)
                if trace is not None:
                    trace.record_idle(t_start=t_start, t_drained=t_drained, t_end=perf())
                continue
            events = sched.step()
            t_stepped = perf()
            if events:
                self._dispatch(events)
            if trace is not None:
                trace.record(
                    t_start=t_start, t_drained=t_drained, t_stepped=t_stepped,
                    t_end=perf(), scheduler=sched, events=events,
                )
            if not sched.last_step_progressed:
                # Nothing could be scheduled this call -- e.g. every waiting
                # request is blocked on page/slot capacity held by requests
                # that are still running. `has_work()` alone can't see this
                # (waiting stays non-empty), so without this the loop would
                # spin at 100% CPU calling `step()` in a tight no-op loop.
                time.sleep(_POLL_SLEEP_S)
        # Under `--async-scheduling` the last launched step's
        # tokens are still on the device when the stop event is set. Commit
        # them: without this the final token of every in-flight request is
        # silently dropped on shutdown.
        try:
            tail = sched.drain()
        except Exception:  # noqa: BLE001 - shutdown must not raise
            tail = []
        if tail:
            self._dispatch(tail)

    def _drain_pending(self) -> None:
        while True:
            try:
                req = self._pending.get_nowait()
            except queue.Empty:
                break
            self.scheduler.add_request(req)
        while True:
            try:
                request_id = self._abort_queue.get_nowait()
            except queue.Empty:
                break
            self.scheduler.abort(request_id)

    def _dispatch(self, events: list[StepEvent]) -> None:
        """Hand this step's per-request deltas to the asyncio side.

        The deliveries are batched into **one**
        ``call_soon_threadsafe`` for the whole step, not one per event. Each
        call takes the event loop's internal lock and writes a byte to its
        self-pipe to wake the selector; at concurrency 256 a decode step
        produced 256 of those, i.e. 256 lock acquisitions and 256 write(2)
        syscalls on the engine thread, plus 256 spurious wakeups on the
        uvicorn thread -- all of it contending for the GIL with an engine loop
        that is itself pure Python between kernel launches. This is overhead
        the standalone graphed-decode benchmark does not have.
        One call per step delivers the same queue writes with one wakeup.
        """
        now = time.monotonic()
        deliveries: list = []
        # One lock acquisition for the step, not one per
        # event: `_out_queues` is only ever *read* here, and taking the lock
        # 256 times per decode step is 256 more chances for the GIL to be
        # handed to the uvicorn thread in the middle of the engine loop's
        # bookkeeping. Same guarantee -- the mapping cannot change under this
        # loop -- at 1/256th the contention.
        with self._lock:
            queues = dict(self._out_queues)
        for ev in events:
            req = ev.request
            if ev.new_token_ids:
                self._generation_tokens_total += len(ev.new_token_ids)
                last = self._last_emit_t.get(req.request_id)
                if req.first_token_at is not None and last is None:
                    self._ttft_hist.observe(req.ttft_s or 0.0)
                elif last is not None:
                    self._tpot_hist.observe(now - last)
                self._last_emit_t[req.request_id] = now
            if ev.finished:
                self._last_emit_t.pop(req.request_id, None)

            stats = RequestStats(
                prompt_tokens=len(req.prompt_token_ids),
                completion_tokens=len(req.output_token_ids),
                ttft_s=req.ttft_s,
                spec_accept_length=None,
            )
            out = StepOutput(
                new_token_ids=list(ev.new_token_ids),
                finished=ev.finished,
                finish_reason=ev.finish_reason,
                stats=stats,
            )
            q = queues.get(req.request_id)
            if q is not None:
                deliveries.append((q, out))
        if deliveries and self._loop is not None:
            self._loop.call_soon_threadsafe(_deliver, deliveries)


def _deliver(deliveries: "list[tuple[asyncio.Queue, StepOutput]]") -> None:
    """Run on the event loop thread: push one step's outputs into their queues.

    ``asyncio.Queue.put_nowait`` on an unbounded queue only appends and wakes a
    waiting getter, so this is O(len(deliveries)) with no awaits and cannot
    block the loop for meaningfully longer than the 256 separate callbacks it
    replaces would have.
    """
    for q, out in deliveries:
        q.put_nowait(out)


__all__ = ["EngineComponents", "build_engine", "build_async_engine", "QwenFastEngine"]
