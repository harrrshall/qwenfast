"""Where the engine loop's wall clock goes, measured *on the loop itself*.

``bench_serve --nvidia-smi`` shows that the served
sweep can leave the H200 idle 27-44 % of the wall clock while vLLM holds it at
99.96 %, and that multiplying the offline ledger by that utilisation
reproduces the served throughput.  What that cannot say is *what the
idle is*: ``profile_serving.attribute`` divides output tokens by the **sum of
step wall times** (``total_wall = sum(r.wall_ms ...)``), so it is
arithmetically incapable of seeing a gap *between* steps, and ``nvidia-smi``
samples once a second from outside the process and cannot attribute anything.

:class:`StepTrace` closes that hole with five ``perf_counter()`` calls per
step (not per token, so it is free at any concurrency) and splits the
loop's wall clock into five disjoint buckets:

``step``
    inside ``Scheduler.step()``: plan, launch, replay, sample, D2H harvest,
    bookkeeping.  The only bucket during which the GPU can be busy.
``drain``
    admitting new requests and applying aborts (in-process: two
    ``queue.Queue`` drains; in the engine-core process: one non-blocking ZeroMQ
    drain).
``emit``
    handing the step's outputs to the serving side (in-process: building
    ``StepOutput``s + one ``call_soon_threadsafe``; in the core: building one
    batched message + one ``send``).
``idle``
    the loop had no work at all -- nothing was admitted, so the GPU is
    *legitimately* idle and this is not a stall.
``gap``
    everything else: the wall clock between the end of one iteration's ``emit``
    and the start of the next iteration's ``drain``.  Pure interpreter --
    the ``while`` test, the attribute loads, and, in the in-process
    configuration, **every microsecond the engine thread spends waiting for
    the GIL** that uvicorn's event loop is holding.

``gap`` is therefore the measurement this module exists for.  It is the one
bucket whose structure (rather than magnitude) differs between the in-process
engine thread and an engine core in its own interpreter, and comparing it
across the two is the main use of this trace.

A per-second series is kept alongside the totals so a summary can be recomputed
over any sub-window offline (a served sweep ramps, and the first seconds of a
level are not the level).
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

__all__ = ["StepTrace", "StepProfiler", "classify_step", "STEP_PHASES", "GPU_MARKS"]

#: Upper bounds, in milliseconds, of the ``gap`` histogram.  Chosen to
#: bracket the interesting range: a GIL handoff is ~5 ms of ``sys.setswitch
#: interval`` at worst, a ``call_soon_threadsafe`` round trip is tens of
#: microseconds, and anything past 50 ms is a step's worth of stall.
GAP_BUCKETS_MS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0)


def classify_step(scheduler, events) -> str:
    """``"overlap" | "mixed" | "prefill" | "spec" | "decode"`` for the step
    that just ran, by exactly the rule ``profile_serving._classify`` uses.

    Duplicated rather than imported because ``profile_serving`` is a 2,800-line
    offline harness that pulls in the whole benchmark surface, and this runs
    inside the server.
    """
    if getattr(scheduler, "last_step_mixed", False):
        return "overlap" if getattr(scheduler, "last_step_overlapped", False) else "mixed"
    if getattr(scheduler, "last_chunk_tokens", 0):
        return "prefill"
    rows = len(getattr(scheduler, "running", ()) or ())
    n = sum(len(e.new_token_ids) for e in events)
    return "spec" if rows and n > rows else "decode"


class StepTrace:
    """Cumulative loop accounting.  Not thread-safe: only the engine loop
    touches it, which is the point -- an instrument that needed a lock would
    be measuring its own lock."""

    def __init__(
        self,
        path: Optional[str] = None,
        *,
        meta: Optional[Dict[str, Any]] = None,
        series_bucket_s: float = 1.0,
    ) -> None:
        self.path = path
        self.meta: Dict[str, Any] = dict(meta or {})
        self.series_bucket_s = float(series_bucket_s)

        self.t0 = time.perf_counter()
        self.wall_start = time.time()
        self.n_steps = 0
        self.n_idle = 0
        self.n_stalled = 0  # iterations where the scheduler could not progress
        self.tokens = 0
        self.sums_ms = {"step": 0.0, "drain": 0.0, "emit": 0.0, "gap": 0.0, "idle": 0.0}
        self.by_kind: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"steps": 0, "ms": 0.0, "tokens": 0, "prefill_tokens": 0, "decode_rows": 0}
        )
        self._gap_counts = [0] * (len(GAP_BUCKETS_MS) + 1)
        self._gap_max_ms = 0.0
        self._last_end: Optional[float] = None

        self._series: List[Dict[str, float]] = []
        self._bucket_index = -1
        self._bucket: Optional[Dict[str, float]] = None

    # -- recording ---------------------------------------------------------- #
    def _bucket_for(self, t: float) -> Dict[str, float]:
        idx = int((t - self.t0) / self.series_bucket_s)
        if idx != self._bucket_index or self._bucket is None:
            self._bucket = {
                "t_s": round(idx * self.series_bucket_s, 3),
                "steps": 0, "tokens": 0,
                "step_ms": 0.0, "drain_ms": 0.0, "emit_ms": 0.0,
                "gap_ms": 0.0, "idle_ms": 0.0,
                # Per-kind, per-second.  The cumulative `by_kind` blends
                # every concurrency level of a sweep into one average, which
                # is exactly the mistake `series` exists to avoid, and the
                # kind breakdown is where the answer is (a
                # decode step and an eager prefill chunk have nothing in
                # common but the loop they run on).
                "kinds": {},
            }
            self._series.append(self._bucket)
            self._bucket_index = idx
        return self._bucket

    def _gap_since(self, t_start: float) -> float:
        if self._last_end is None:
            return 0.0
        return max(t_start - self._last_end, 0.0) * 1e3

    def record(
        self,
        *,
        t_start: float,
        t_drained: float,
        t_stepped: float,
        t_end: float,
        scheduler=None,
        events=(),
    ) -> None:
        """One iteration that ran a step.  All four timestamps are
        ``perf_counter()`` values taken by the loop."""
        gap_ms = self._gap_since(t_start)
        drain_ms = (t_drained - t_start) * 1e3
        step_ms = (t_stepped - t_drained) * 1e3
        emit_ms = (t_end - t_stepped) * 1e3
        self._last_end = t_end

        self.n_steps += 1
        s = self.sums_ms
        s["gap"] += gap_ms
        s["drain"] += drain_ms
        s["step"] += step_ms
        s["emit"] += emit_ms

        n_tok = 0
        for e in events:
            n_tok += len(e.new_token_ids)
        self.tokens += n_tok

        if gap_ms > self._gap_max_ms:
            self._gap_max_ms = gap_ms
        i = 0
        for b in GAP_BUCKETS_MS:
            if gap_ms <= b:
                break
            i += 1
        self._gap_counts[i] += 1

        if scheduler is not None:
            kind = classify_step(scheduler, events)
            k = self.by_kind[kind]
            k["steps"] += 1
            k["ms"] += step_ms
            k["tokens"] += n_tok
            k["prefill_tokens"] += int(getattr(scheduler, "last_chunk_tokens", 0) or 0)
            k["decode_rows"] += int(getattr(scheduler, "last_mixed_decode_rows", 0) or 0)
            if not getattr(scheduler, "last_step_progressed", True):
                self.n_stalled += 1

        b = self._bucket_for(t_start)
        b["steps"] += 1
        b["tokens"] += n_tok
        b["step_ms"] += step_ms
        b["drain_ms"] += drain_ms
        b["emit_ms"] += emit_ms
        b["gap_ms"] += gap_ms
        if scheduler is not None:
            bk = b["kinds"].setdefault(kind, [0, 0.0, 0, 0, 0])
            bk[0] += 1                 # steps
            bk[1] += step_ms           # ms
            bk[2] += n_tok             # output tokens
            bk[3] += int(getattr(scheduler, "last_chunk_tokens", 0) or 0)
            bk[4] += int(getattr(scheduler, "last_mixed_decode_rows", 0) or 0)

    def record_idle(self, *, t_start: float, t_drained: float, t_end: float) -> None:
        """One iteration that found no work and waited."""
        gap_ms = self._gap_since(t_start)
        drain_ms = (t_drained - t_start) * 1e3
        idle_ms = (t_end - t_drained) * 1e3
        self._last_end = t_end
        self.n_idle += 1
        self.sums_ms["gap"] += gap_ms
        self.sums_ms["drain"] += drain_ms
        self.sums_ms["idle"] += idle_ms
        b = self._bucket_for(t_start)
        b["drain_ms"] += drain_ms
        b["idle_ms"] += idle_ms
        b["gap_ms"] += gap_ms

    # -- reporting ---------------------------------------------------------- #
    @property
    def step_busy_pct(self) -> float:
        """The cheap half of :meth:`summary` -- safe to call on the loop.

        The fraction of the wall clock spent inside ``Scheduler.step()``, i.e.
        the loop's own view of the utilisation ``nvidia-smi`` samples from
        outside the process.  ``summary()`` builds every per-kind dict and the
        whole series, so calling *it* four times a second (as the engine core's
        stats push wanted to) would be the instrument distorting the thing it
        measures.
        """
        wall_ms = (time.perf_counter() - self.t0) * 1e3
        return round(100.0 * self.sums_ms["step"] / wall_ms, 2) if wall_ms else 0.0

    def summary(self) -> Dict[str, Any]:
        wall_ms = (time.perf_counter() - self.t0) * 1e3
        wall = wall_ms or 1.0
        s = self.sums_ms
        accounted = sum(s.values())
        # Everything the five buckets did not see: the first iteration's
        # pre-loop setup and any clock skew.  Reported rather than folded into
        # `gap`, so the buckets stay disjoint and honest.
        unaccounted = max(wall_ms - accounted, 0.0)

        busy_ms = s["step"]
        # The fraction of the wall clock in which the GPU *could* have been
        # busy.  Not the same as `nvidia-smi`'s utilisation (a step is host
        # work too), but its ceiling: nothing outside `step` launches a kernel.
        pct = {k: round(100.0 * v / wall, 2) for k, v in s.items()}
        pct["unaccounted"] = round(100.0 * unaccounted / wall, 2)

        out: Dict[str, Any] = {
            "meta": self.meta,
            "wall_s": round(wall_ms / 1e3, 3),
            "wall_start_unix": self.wall_start,
            "steps": self.n_steps,
            "idle_iterations": self.n_idle,
            "stalled_iterations": self.n_stalled,
            "output_tokens": self.tokens,
            "output_tok_s": round(self.tokens / (wall_ms / 1e3), 1) if wall_ms else 0.0,
            "ms": {k: round(v, 1) for k, v in s.items()},
            "ms_unaccounted": round(unaccounted, 1),
            "pct_of_wall": pct,
            "step_busy_pct": round(100.0 * busy_ms / wall, 2),
            "host_stall_pct": round(
                100.0 * (s["gap"] + s["drain"] + s["emit"] + unaccounted) / wall, 2
            ),
            # `idle` is deliberately absent: it is paid per *idle iteration*,
            # not per step, and dividing it by the step count produces a
            # number with no meaning (a mostly-idle server reads as "263 ms of
            # idle per step").
            "per_step_ms": {
                k: round(s[k] / self.n_steps, 4)
                for k in ("step", "drain", "emit", "gap")
                if self.n_steps
            },
            "idle_ms_per_idle_iteration": (
                round(s["idle"] / self.n_idle, 4) if self.n_idle else 0.0
            ),
            "gap_ms_max": round(self._gap_max_ms, 3),
            "gap_hist": {
                "bounds_ms": list(GAP_BUCKETS_MS),
                "counts": list(self._gap_counts),  # len == bounds + 1 (the +Inf row)
            },
            "by_kind": {
                k: {
                    "steps": int(v["steps"]),
                    "ms": round(v["ms"], 1),
                    "ms_per_step": round(v["ms"] / v["steps"], 2) if v["steps"] else 0.0,
                    "tokens": int(v["tokens"]),
                    "wall_pct": round(100.0 * v["ms"] / wall, 2),
                    "prefill_tokens_per_step": (
                        round(v["prefill_tokens"] / v["steps"], 1) if v["steps"] else 0.0
                    ),
                    "decode_rows_per_step": (
                        round(v["decode_rows"] / v["steps"], 1) if v["steps"] else 0.0
                    ),
                }
                for k, v in sorted(self.by_kind.items())
            },
            "series": self._series,
        }
        return out

    def one_line(self) -> str:
        s = self.summary()
        p = s["pct_of_wall"]
        return (
            f"[step-trace] wall {s['wall_s']:.1f}s steps {s['steps']} "
            f"out_tok/s {s['output_tok_s']:.0f} | "
            f"step {p['step']:.1f}% gap {p['gap']:.1f}% drain {p['drain']:.1f}% "
            f"emit {p['emit']:.1f}% idle {p['idle']:.1f}% other {p['unaccounted']:.1f}% | "
            f"host-stall {s['host_stall_pct']:.1f}% | "
            f"gap/step {s['per_step_ms'].get('gap', 0.0):.3f} ms max {s['gap_ms_max']:.3f} ms"
        )

    def write(self) -> Optional[str]:
        """Dump the summary to ``self.path``.  Never raises: a trace that
        cannot be written must not take a server down on its way out."""
        if not self.path:
            return None
        try:
            d = os.path.dirname(os.path.abspath(self.path))
            if d:
                os.makedirs(d, exist_ok=True)
            with open(self.path, "w") as fh:
                json.dump(self.summary(), fh, indent=1, default=_json_default)
            return self.path
        except Exception as exc:  # noqa: BLE001
            print(f"[step-trace] could not write {self.path}: {exc}", flush=True)
            return None


def _json_default(o):
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return str(o)


# =========================================================================== #
# StepProfiler: where the *inside* of a step goes
# =========================================================================== #
#: The host phases of one ``Scheduler.step()``, in the order they occur.  Each
#: is the wall clock between its mark and the previous one, so the list is a
#: partition of the step -- ``_PHASES[0]`` is measured from the step's start.
STEP_PHASES = (
    "admit",      # aborts + `_admit_decode_rows` (preemption, page capacity)
    "collect",    # `_collect_prefill_chunk`: admission, slots, KV page allocation
    "build",      # host tensors for the step + their H2D uploads
    "plan",       # FlashInfer `plan_decode` / `plan_mixed_graph` (host-side)
    "launch",     # graph replay / eager forward -- kernel *launch*, not execution
    "sync",       # waiting for the device (the harvest's `synchronize`)
    "harvest",    # the pinned D2H copy + `tolist`
    "book",       # per-request bookkeeping: stop conditions, StepEvent objects
)

#: The device marks.  CUDA events recorded on the stream at the same
#: boundaries; the gap between two of them on the *device* timeline is the
#: quantity `nvidia-smi` averages and `StepTrace` cannot see.
GPU_MARKS = ("g_start", "g_launch", "g_done", "g_end")


class StepProfiler:
    """Per-step host **and** device attribution, for a bounded window of steps.

    ``StepTrace`` can localise idle time to "inside ``Scheduler.step()``" but
    no further: it measures the loop, not the step, and ``nvidia-smi``
    samples from outside the process once a second.  This is the instrument
    that opens the step up.

    Two clocks, recorded together:

    * **host** -- ``perf_counter()`` at each of :data:`STEP_PHASES`' boundaries,
      so a step is split into admit / collect / build / plan / launch / sync /
      harvest / book.
    * **device** -- four ``cuda.Event``s on the stream (:data:`GPU_MARKS`).
      ``g_start`` is recorded at the top of the step, ``g_launch`` immediately
      before the first replay, ``g_done`` immediately after the last launch,
      ``g_end`` at the very end.  Because an event on an *idle* stream is
      timestamped the moment the driver processes it, ``elapsed(g_start,
      g_launch)`` is exactly **the device idle the host's pre-launch work
      buys**, and ``elapsed(g_launch, g_done)`` is the step's kernel time.
      That pair is the whole measurement: a synchronous step has a large
      ``g_start -> g_launch``, an asynchronously scheduled one should have
      ~none, because the device is still running step N-1 while the host
      builds step N.

    Costs four events and ~12 ``perf_counter()`` calls per step, and is capped
    at ``max_steps`` records after ``warmup_steps``, so it is switched on for a
    window and never left on.  Event elapsed times are resolved lazily in
    :meth:`summary` (after one ``synchronize()``), which is what makes it safe
    on the asynchronous path: querying an event of the step you just launched
    would reintroduce exactly the sync the window exists to remove.
    """

    def __init__(
        self,
        *,
        max_steps: int = 400,
        warmup_steps: int = 100,
        min_rows: int = 0,
        path: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        device: Optional[Any] = None,
    ) -> None:
        self.max_steps = int(max_steps)
        self.warmup_steps = int(warmup_steps)
        #: Only steps with at least this many decode rows count -- towards the
        #: warmup, towards the cap, and into the summary.  A served run does a
        #: gate request, a warmup and a ramp before it reaches the level under
        #: test, all of them at batch 1-8, and a window counted in *steps*
        #: would land in that instead of in the regime the window is about.
        self.min_rows = int(min_rows)
        self.path = path
        self.meta: Dict[str, Any] = dict(meta or {})
        self.device = device
        self.seen = 0
        self._recording = False
        self._t: Dict[str, float] = {}
        self._t0 = 0.0
        self._events: Dict[str, Any] = {}
        self._records: List[Dict[str, Any]] = []
        self._cuda = False
        try:  # pragma: no cover - trivially environment-dependent
            import torch

            self._torch = torch
            self._cuda = bool(
                torch.cuda.is_available()
                and (device is None or getattr(device, "type", "cpu") == "cuda")
            )
        except Exception:  # noqa: BLE001  pragma: no cover
            self._torch = None

    # -- recording ---------------------------------------------------------- #
    @property
    def done(self) -> bool:
        return self.seen >= self.warmup_steps + self.max_steps

    def begin(self) -> None:
        """Top of ``Scheduler.step()``.  Cheap enough to call unconditionally."""
        self.seen += 1
        self._recording = self.warmup_steps < self.seen <= self.warmup_steps + self.max_steps
        if not self._recording:
            return
        self._t = {}
        self._events = {}
        self._t0 = time.perf_counter()
        self.gpu_mark("g_start")

    def mark(self, name: str) -> None:
        if self._recording:
            self._t[name] = time.perf_counter()

    def gpu_mark(self, name: str) -> None:
        if not self._recording or not self._cuda:
            return
        ev = self._torch.cuda.Event(enable_timing=True)
        ev.record()
        self._events[name] = ev

    def end(self, *, kind: str = "?", rows: int = 0, chunk_tokens: int = 0,
            tokens: int = 0) -> None:
        if not self._recording:
            return
        if rows < self.min_rows:
            # Not the regime under test: unwind this step entirely so it
            # consumes neither the warmup nor the cap.
            self._recording = False
            self.seen -= 1
            return
        self.gpu_mark("g_end")
        self._t["end"] = time.perf_counter()
        self._records.append(
            {
                "kind": kind,
                "rows": int(rows),
                "chunk_tokens": int(chunk_tokens),
                "tokens": int(tokens),
                "t0": self._t0,
                "host": dict(self._t),
                "ev": dict(self._events),
            }
        )
        self._recording = False

    # -- reporting ---------------------------------------------------------- #
    @staticmethod
    def _phase_ms(t0: float, host: Dict[str, float]) -> Dict[str, float]:
        """Turn the marks into disjoint phase durations.

        A mark that was never taken (a decode step has no ``collect``) simply
        contributes 0 and the next phase measures from the previous mark, so
        the phases always add up to the step.
        """
        out: Dict[str, float] = {}
        prev = t0
        for name in STEP_PHASES:
            t = host.get(name)
            if t is None:
                out[name] = 0.0
                continue
            out[name] = (t - prev) * 1e3
            prev = t
        end = host.get("end", prev)
        out["other"] = (end - prev) * 1e3
        out["total"] = (end - t0) * 1e3
        # The deferred commit is not a phase of the step it belongs to:
        # it is the *previous* step's harvest, running while the device
        # works on this one -- so it is reported beside the phases rather than
        # inside them.  It is part of `other`.
        c0, c1, c2 = host.get("c0"), host.get("c1"), host.get("c2")
        out["commit_wait"] = (c1 - c0) * 1e3 if (c0 and c1) else 0.0
        out["commit_book"] = (c2 - c1) * 1e3 if (c1 and c2) else 0.0
        return out

    def summary(self) -> Dict[str, Any]:
        by_kind: Dict[str, Dict[str, Any]] = {}
        if self._cuda and self._records:
            self._torch.cuda.synchronize()
        for rec in self._records:
            k = rec["kind"]
            agg = by_kind.setdefault(
                k,
                {
                    "steps": 0, "rows": 0, "chunk_tokens": 0, "tokens": 0,
                    "host_ms": {n: 0.0 for n in
                                STEP_PHASES + ("other", "total",
                                               "commit_wait", "commit_book")},
                    "gpu_ms": {"idle_pre": 0.0, "kernel": 0.0, "tail": 0.0, "span": 0.0},
                    "gpu_steps": 0,
                },
            )
            agg["steps"] += 1
            agg["rows"] += rec["rows"]
            agg["chunk_tokens"] += rec["chunk_tokens"]
            agg["tokens"] += rec["tokens"]
            ph = self._phase_ms(rec["t0"], rec["host"])
            for n, v in ph.items():
                agg["host_ms"][n] = agg["host_ms"].get(n, 0.0) + v
            ev = rec["ev"]
            if len(ev) == 4:
                try:
                    g = agg["gpu_ms"]
                    g["idle_pre"] += ev["g_start"].elapsed_time(ev["g_launch"])
                    g["kernel"] += ev["g_launch"].elapsed_time(ev["g_done"])
                    g["tail"] += ev["g_done"].elapsed_time(ev["g_end"])
                    g["span"] += ev["g_start"].elapsed_time(ev["g_end"])
                    agg["gpu_steps"] += 1
                except Exception:  # noqa: BLE001  pragma: no cover
                    pass

        out_kinds: Dict[str, Any] = {}
        for k, agg in sorted(by_kind.items()):
            n = max(agg["steps"], 1)
            gn = max(agg["gpu_steps"], 1)
            host = {name: round(v / n, 3) for name, v in agg["host_ms"].items()}
            gpu = {name: round(v / gn, 3) for name, v in agg["gpu_ms"].items()}
            span = gpu["span"] or 1.0
            out_kinds[k] = {
                "steps": agg["steps"],
                "rows_per_step": round(agg["rows"] / n, 1),
                "chunk_tokens_per_step": round(agg["chunk_tokens"] / n, 1),
                "tokens_per_step": round(agg["tokens"] / n, 2),
                "host_ms": host,
                "gpu_ms": gpu,
                # The headline: of the device timeline this step spans, how
                # much was a kernel running?  1 - this is the idle that
                # the loop-level trace can only localise to "inside the step".
                "gpu_busy_pct": round(100.0 * gpu["kernel"] / span, 2),
                "gpu_idle_pre_pct": round(100.0 * gpu["idle_pre"] / span, 2),
                "gpu_idle_tail_pct": round(100.0 * gpu["tail"] / span, 2),
            }

        tot_span = sum(a["gpu_ms"]["span"] for a in by_kind.values())
        tot_kern = sum(a["gpu_ms"]["kernel"] for a in by_kind.values())
        return {
            "meta": self.meta,
            "steps_seen": self.seen,
            "steps_recorded": len(self._records),
            "warmup_steps": self.warmup_steps,
            "min_rows": self.min_rows,
            "overall_gpu_busy_pct": round(100.0 * tot_kern / tot_span, 2) if tot_span else 0.0,
            "by_kind": out_kinds,
        }

    def one_line(self) -> str:
        s = self.summary()
        bits = []
        for k, v in s["by_kind"].items():
            bits.append(
                f"{k} n={v['steps']} step={v['host_ms']['total']:.1f}ms "
                f"gpu={v['gpu_ms']['kernel']:.1f}ms busy={v['gpu_busy_pct']:.0f}%"
            )
        return "[step-profile] " + " | ".join(bits)

    def write(self) -> Optional[str]:
        if not self.path:
            return None
        try:
            d = os.path.dirname(os.path.abspath(self.path))
            if d:
                os.makedirs(d, exist_ok=True)
            with open(self.path, "w") as fh:
                json.dump(self.summary(), fh, indent=1, default=_json_default)
            return self.path
        except Exception as exc:  # noqa: BLE001
            print(f"[step-profile] could not write {self.path}: {exc}", flush=True)
            return None
