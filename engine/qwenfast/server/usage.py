"""Per-request metering for the public endpoint: in-memory totals + a SQLite audit log.

Design constraint: **nothing in here may run on the streaming hot path.** A token
arriving from the engine must not wait on a disk write, an `fsync`, or a lock held by one. So:

* the request handler builds a `UsageRecord` (a plain dataclass) once, when the response is
  already finished, and hands it to `UsageRecorder.record()`;
* `record()` updates the in-memory aggregates under a short lock and `put_nowait`s the row onto a
  **bounded** queue. If the queue is full the row is dropped and counted — losing an audit row is
  strictly better than blocking a generation loop, and the aggregates (which is what the dashboard
  reads) are updated regardless;
* a single daemon thread drains the queue, batching up to `BATCH_MAX` rows per transaction.

The aggregates are *hydrated from SQLite at startup*, so "total tokens generated" survives a
server restart — which the watchdog in `scripts/remote_public_server.sh` makes a routine event,
and which would otherwise make the dashboard's headline number lie after every crash.

Cost model. GPU cost is wall-clock, not per-token: a rented GPU bills a fixed hourly rate
(`--gpu-rate-inr-per-hour`) whether or not anyone is calling. So the dashboard reports cost against *service uptime* — wall
time since the first request ever recorded in this database — and derives cost per 1M output
tokens from that. Process uptime is reported separately; the difference between the two is exactly
the watchdog-restart downtime.
"""

from __future__ import annotations

import json
import os
import queue
import sqlite3
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional

SCHEMA_VERSION = 2  # 2 adds usage.error (the rejection reason); see `_migrate`.
QUEUE_MAX = 20_000
BATCH_MAX = 256
FLUSH_INTERVAL_S = 1.0
RECENT_WINDOW_S = 60.0
HOURLY_KEEP_H = 26  # 24 shown + slack so the oldest bucket is always complete
LATENCY_WINDOW_S = 3600.0   # p50/p99 TTFT/TPOT are reported over this trailing window
LATENCY_MAX_SAMPLES = 20_000

DEFAULT_GPU_RATE_INR_PER_HOUR = 188.73
DEFAULT_INR_PER_USD = 87.5


@dataclass
class UsageRecord:
    ts: float
    key_name: str
    endpoint: str
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: Optional[float]
    duration_ms: float
    status: int
    stream: bool = False
    finish_reason: Optional[str] = None
    #: Short machine-readable reason this request did not produce a normal
    #: answer -- one of `ERROR_CLASSES`. `None` for a plain 200. This is the
    #: column exists so that a refused request records *which* 400 it was, not
    #: just `status=400`.
    error: Optional[str] = None


#: The closed set of values `UsageRecord.error` may take. Closed on purpose:
#: `/admin/usage` renders a breakdown keyed by these, and an open-ended string
#: (an exception message, say) would both explode the cardinality and leak
#: user content into the operator's dashboard.
ERROR_CLASSES: tuple[str, ...] = (
    "context_length_exceeded",   # prompt alone does not leave room to answer
    "prompt_too_long",           # over --max-prompt-tokens
    "too_many_messages",         # over --max-messages
    "invalid_request",           # malformed JSON / schema / chat-template error
    "payload_too_large",         # over --max-request-bytes (413)
    "unauthorized",              # 401
    "forbidden",                 # 403
    "rate_limited",              # 429 from the per-key rpm/tpm window
    "concurrency_limited",       # 429 from the per-key / per-IP stream cap
    "overloaded",                # 503 from the global admission cap
    "engine_unavailable",        # 503 because the engine is not healthy
    "draining",                  # 503 because the process is shutting down
    "request_timeout",           # cut short by --request-timeout
    "client_disconnect",         # the caller hung up mid-stream
    "unclassified",              # a row written before this column existed
)


_DDL = """
CREATE TABLE IF NOT EXISTS usage (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              REAL    NOT NULL,
    key_name        TEXT    NOT NULL,
    endpoint        TEXT    NOT NULL,
    prompt_tokens   INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    ttft_ms         REAL,
    duration_ms     REAL,
    status          INTEGER NOT NULL,
    stream          INTEGER NOT NULL DEFAULT 0,
    finish_reason   TEXT,
    error           TEXT
);
CREATE INDEX IF NOT EXISTS usage_ts ON usage (ts);
CREATE INDEX IF NOT EXISTS usage_key_ts ON usage (key_name, ts);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
"""


def _hour_bucket(ts: float) -> int:
    return int(ts // 3600) * 3600


def _tpot_ms(ttft_ms, duration_ms, completion_tokens) -> Optional[float]:
    """Mean time-per-output-token for one finished request, in ms.

    Deliberately *not* `duration / tokens`: the first token costs prefill, and
    mixing that into TPOT makes a long prompt look like a slow decode. This is
    the decode window only -- `(duration - ttft) / (tokens - 1)` -- and is
    `None` for anything that produced fewer than two tokens, where the quantity
    is not defined rather than zero.
    """
    try:
        if ttft_ms is None or duration_ms is None or int(completion_tokens) < 2:
            return None
        window = float(duration_ms) - float(ttft_ms)
        if window <= 0:
            return None
        return window / (int(completion_tokens) - 1)
    except Exception:  # noqa: BLE001 - a metrics helper may never raise
        return None


def percentiles(values: list[float], qs=(0.5, 0.9, 0.99)) -> Dict[str, Optional[float]]:
    """Nearest-rank percentiles over an unsorted list. `{}`-safe.

    Nearest-rank (rather than interpolating) because these are latency samples
    read by a human on a dashboard: `p99` should be *a request that actually
    happened*, not an average of two of them.
    """
    out: Dict[str, Optional[float]] = {}
    ordered = sorted(v for v in values if v is not None)
    n = len(ordered)
    for q in qs:
        key = f"p{int(round(q * 100))}"
        if not n:
            out[key] = None
            continue
        rank = max(1, min(n, int(-(-q * n // 1))))  # ceil(q*n), clamped to [1, n]
        out[key] = ordered[rank - 1]
    return out


@dataclass
class _KeyTotals:
    requests: int = 0
    errors: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    last_seen: float = 0.0

    def as_dict(self, name: str) -> dict:
        return {
            "name": name,
            "requests": self.requests,
            "errors": self.errors,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "last_seen": self.last_seen or None,
        }


class UsageRecorder:
    """In-memory usage aggregates with a background SQLite writer.

    `db_path=None` keeps everything in memory (what the unit tests that do not care about
    persistence use, and a safe degradation if the results directory is not writable).
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        *,
        gpu_rate_inr_per_hour: float = DEFAULT_GPU_RATE_INR_PER_HOUR,
        inr_per_usd: float = DEFAULT_INR_PER_USD,
        clock=time.time,
        start_writer: bool = True,
    ) -> None:
        self.db_path = db_path
        self.gpu_rate_inr_per_hour = gpu_rate_inr_per_hour
        self.inr_per_usd = inr_per_usd
        self._clock = clock

        self._lock = threading.Lock()
        self.requests = 0
        self.errors = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.dropped_rows = 0
        self.rate_limited = 0
        self.rejected_overload = 0
        self.rejected_concurrency = 0
        self.auth_failures = 0
        self.timeouts = 0
        self._per_key: Dict[str, _KeyTotals] = defaultdict(_KeyTotals)
        self._errors_by_class: Dict[str, int] = defaultdict(int)
        self._hourly: Dict[int, dict] = {}
        self._recent: Deque[tuple[float, int]] = deque()
        self._ttft_recent: Deque[float] = deque(maxlen=200)
        # (ts, ttft_ms|None, tpot_ms|None) over the trailing hour, for the p50/p99
        # block on /admin/usage. Bounded twice -- by age when read, by count when
        # written -- so a busy hour cannot grow it without limit.
        self._latency: Deque[tuple[float, Optional[float], Optional[float]]] = deque(
            maxlen=LATENCY_MAX_SAMPLES
        )

        self.process_started_at = self._clock()
        self.service_started_at = self.process_started_at
        self.db_error: Optional[str] = None

        self._queue: "queue.Queue[Optional[UsageRecord]]" = queue.Queue(maxsize=QUEUE_MAX)
        self._writer: Optional[threading.Thread] = None
        self._stopping = threading.Event()

        if db_path:
            try:
                self._init_db()
                self._hydrate()
            except Exception as exc:  # noqa: BLE001 - metering must never block serving
                self.db_error = f"{exc.__class__.__name__}: {exc}"
                self.db_path = None
        if self.db_path and start_writer:
            self._writer = threading.Thread(
                target=self._writer_loop, name="usage-writer", daemon=True
            )
            self._writer.start()

    # -- database ---------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        # WAL + NORMAL: the writer thread never blocks a reader (`/admin/usage` opens its own
        # short-lived connection), and we accept losing at most the last transaction on a hard
        # power loss — this is telemetry, not billing.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        parent = os.path.dirname(os.path.abspath(self.db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        conn = self._connect()
        try:
            conn.executescript(_DDL)
            self._migrate(conn)
            conn.execute(
                "INSERT OR REPLACE INTO meta (k, v) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Additive, in-place upgrades of an existing `usage` table.

        The public endpoint's database is eleven hours of real traffic that the
        cost model is computed from (`service_started_at` is `MIN(ts)`), so a
        schema change must **not** be a fresh file. Every migration here is an
        `ADD COLUMN` with a NULL default: old rows keep their meaning, and
        `error IS NULL` on a >= 400 row reads as `"unclassified"` -- which is
        exactly what those rows are.
        """
        have = {row[1] for row in conn.execute("PRAGMA table_info(usage)")}
        if "error" not in have:
            conn.execute("ALTER TABLE usage ADD COLUMN error TEXT")

    def _hydrate(self) -> None:
        """Rebuild the aggregates from the database so restarts do not reset the dashboard."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(prompt_tokens),0), COALESCE(SUM(completion_tokens),0),"
                " COALESCE(SUM(status >= 400), 0), MIN(ts)"
                " FROM usage"
            ).fetchone()
            if row and row[0]:
                self.requests, self.prompt_tokens, self.completion_tokens, self.errors = (
                    int(row[0]), int(row[1]), int(row[2]), int(row[3] or 0)
                )
                if row[4]:
                    self.service_started_at = float(row[4])
            for name, reqs, pt, ct, errs, last in conn.execute(
                "SELECT key_name, COUNT(*), COALESCE(SUM(prompt_tokens),0),"
                " COALESCE(SUM(completion_tokens),0), COALESCE(SUM(status >= 400),0), MAX(ts)"
                " FROM usage GROUP BY key_name"
            ):
                t = self._per_key[name]
                t.requests, t.prompt_tokens, t.completion_tokens = int(reqs), int(pt), int(ct)
                t.errors = int(errs or 0)
                t.last_seen = float(last or 0.0)
            for err, n in conn.execute(
                "SELECT COALESCE(error, 'unclassified'), COUNT(*) FROM usage"
                " WHERE status >= 400 GROUP BY 1"
            ):
                self._errors_by_class[str(err)] += int(n)
            # Latency samples for the p50/p99 block, restricted to the window we
            # actually report over so a restart does not lose the last hour.
            lat_cutoff = self._clock() - LATENCY_WINDOW_S
            for ts, ttft, dur, ctok in conn.execute(
                "SELECT ts, ttft_ms, duration_ms, completion_tokens FROM usage"
                " WHERE ts >= ? AND status = 200 ORDER BY ts",
                (lat_cutoff,),
            ):
                self._latency.append(
                    (float(ts), float(ttft) if ttft is not None else None,
                     _tpot_ms(ttft, dur, ctok))
                )
            cutoff = self._clock() - HOURLY_KEEP_H * 3600
            for bucket, reqs, ct, pt in conn.execute(
                "SELECT CAST(ts / 3600 AS INTEGER) * 3600, COUNT(*),"
                " COALESCE(SUM(completion_tokens),0), COALESCE(SUM(prompt_tokens),0)"
                " FROM usage WHERE ts >= ? GROUP BY 1",
                (cutoff,),
            ):
                self._hourly[int(bucket)] = {
                    "requests": int(reqs),
                    "completion_tokens": int(ct),
                    "prompt_tokens": int(pt),
                }
        finally:
            conn.close()

    def _writer_loop(self) -> None:  # pragma: no cover - exercised via flush() in tests
        conn = None
        pending: list[UsageRecord] = []
        while True:
            try:
                item = self._queue.get(timeout=FLUSH_INTERVAL_S)
                if item is None:
                    self._flush(conn, pending)
                    if conn is not None:
                        conn.close()
                    return
                pending.append(item)
                while len(pending) < BATCH_MAX:
                    try:
                        nxt = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if nxt is None:
                        self._flush(conn if conn else self._safe_connect(), pending)
                        if conn is not None:
                            conn.close()
                        return
                    pending.append(nxt)
            except queue.Empty:
                pass
            if pending:
                if conn is None:
                    conn = self._safe_connect()
                    if conn is None:
                        pending.clear()
                        continue
                self._flush(conn, pending)

    def _safe_connect(self) -> Optional[sqlite3.Connection]:
        try:
            return self._connect()
        except Exception as exc:  # noqa: BLE001
            self.db_error = f"{exc.__class__.__name__}: {exc}"
            return None

    def _flush(self, conn: Optional[sqlite3.Connection], pending: list[UsageRecord]) -> None:
        if not pending:
            return
        if conn is None:
            pending.clear()
            return
        try:
            conn.executemany(
                "INSERT INTO usage (ts, key_name, endpoint, prompt_tokens, completion_tokens,"
                " ttft_ms, duration_ms, status, stream, finish_reason, error)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        r.ts, r.key_name, r.endpoint, r.prompt_tokens, r.completion_tokens,
                        r.ttft_ms, r.duration_ms, r.status, 1 if r.stream else 0, r.finish_reason,
                        r.error,
                    )
                    for r in pending
                ],
            )
            conn.commit()
        except Exception as exc:  # noqa: BLE001 - a broken DB must not stop metering in memory
            self.db_error = f"{exc.__class__.__name__}: {exc}"
        finally:
            pending.clear()

    # -- recording --------------------------------------------------------------

    def record(self, rec: UsageRecord) -> None:
        """Never blocks, never raises."""
        try:
            with self._lock:
                self.requests += 1
                if rec.status >= 400:
                    self.errors += 1
                self.prompt_tokens += rec.prompt_tokens
                self.completion_tokens += rec.completion_tokens
                t = self._per_key[rec.key_name]
                t.requests += 1
                if rec.status >= 400:
                    t.errors += 1
                t.prompt_tokens += rec.prompt_tokens
                t.completion_tokens += rec.completion_tokens
                t.last_seen = max(t.last_seen, rec.ts)

                bucket = _hour_bucket(rec.ts)
                h = self._hourly.setdefault(
                    bucket, {"requests": 0, "completion_tokens": 0, "prompt_tokens": 0}
                )
                h["requests"] += 1
                h["completion_tokens"] += rec.completion_tokens
                h["prompt_tokens"] += rec.prompt_tokens
                cutoff_h = _hour_bucket(rec.ts) - HOURLY_KEEP_H * 3600
                for stale in [b for b in self._hourly if b < cutoff_h]:
                    del self._hourly[stale]

                if rec.completion_tokens:
                    self._recent.append((rec.ts, rec.completion_tokens))
                    cutoff = rec.ts - RECENT_WINDOW_S
                    while self._recent and self._recent[0][0] <= cutoff:
                        self._recent.popleft()
                if rec.ttft_ms is not None:
                    self._ttft_recent.append(rec.ttft_ms)
                if rec.status >= 400 or rec.error:
                    self._errors_by_class[rec.error or "unclassified"] += 1
                if rec.error == "request_timeout":
                    self.timeouts += 1
                if rec.status == 200:
                    self._latency.append(
                        (rec.ts, rec.ttft_ms,
                         _tpot_ms(rec.ttft_ms, rec.duration_ms, rec.completion_tokens))
                    )

            if self.db_path:
                try:
                    self._queue.put_nowait(rec)
                except queue.Full:
                    with self._lock:
                        self.dropped_rows += 1
        except Exception:  # noqa: BLE001 - metering is never worth a failed response
            pass

    def note_rate_limited(self) -> None:
        with self._lock:
            self.rate_limited += 1

    def note_overload(self) -> None:
        with self._lock:
            self.rejected_overload += 1

    def note_concurrency_rejected(self) -> None:
        with self._lock:
            self.rejected_concurrency += 1

    def note_auth_failure(self) -> None:
        with self._lock:
            self.auth_failures += 1

    def flush(self, timeout: float = 5.0) -> None:
        """Block until the queue is drained (tests and shutdown only)."""
        if not self.db_path:
            return
        deadline = time.monotonic() + timeout
        while not self._queue.empty() and time.monotonic() < deadline:
            time.sleep(0.01)
        # One more beat for the in-flight batch's commit.
        time.sleep(0.05)

    def shutdown(self, timeout: float = 5.0) -> None:
        if self._writer is None:
            return
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._writer.join(timeout=timeout)

    # -- reporting --------------------------------------------------------------

    def recent_tokens_per_second(self, now: Optional[float] = None) -> float:
        """Output tok/s over the trailing 60 s, from *completed* requests.

        Requests are attributed at completion, so a single very long stream in flight reads as 0
        until it ends. The dashboard pairs this with the engine's own `/metrics` rate, which is
        instantaneous, and says which is which.
        """
        now = self._clock() if now is None else now
        cutoff = now - RECENT_WINDOW_S
        with self._lock:
            while self._recent and self._recent[0][0] <= cutoff:
                self._recent.popleft()
            total = sum(n for _, n in self._recent)
        return total / RECENT_WINDOW_S

    def hourly_series(self, hours: int = 24, now: Optional[float] = None) -> list[dict]:
        """Dense (no gaps) list of the last `hours` hourly buckets, oldest first."""
        now = self._clock() if now is None else now
        current = _hour_bucket(now)
        with self._lock:
            snapshot = dict(self._hourly)
        out = []
        for i in range(hours - 1, -1, -1):
            bucket = current - i * 3600
            h = snapshot.get(bucket, {})
            out.append(
                {
                    "hour": bucket,
                    "requests": h.get("requests", 0),
                    "completion_tokens": h.get("completion_tokens", 0),
                    "prompt_tokens": h.get("prompt_tokens", 0),
                }
            )
        return out

    def latency_window(self, now: Optional[float] = None, window_s: float = LATENCY_WINDOW_S) -> dict:
        """p50/p90/p99 TTFT and TPOT over the trailing `window_s` seconds.

        These are the two numbers a caller of a shared single-GPU endpoint can
        actually feel -- how long until the first token, and how fast the rest
        arrive -- and the mean alone hid both: one 30 s prefill and 200 fast
        replies average out to "fine".
        """
        now = self._clock() if now is None else now
        cutoff = now - window_s
        with self._lock:
            while self._latency and self._latency[0][0] < cutoff:
                self._latency.popleft()
            ttfts = [t for _, t, _ in self._latency if t is not None]
            tpots = [p for _, _, p in self._latency if p is not None]
        return {
            "window_s": window_s,
            "samples": len(ttfts),
            "ttft_ms": percentiles(ttfts),
            "tpot_ms": percentiles(tpots),
        }

    def errors_by_class(self) -> Dict[str, int]:
        with self._lock:
            return dict(sorted(self._errors_by_class.items(), key=lambda kv: -kv[1]))

    def errors_by_class_since(self, since: float) -> Dict[str, int]:
        """The same breakdown restricted to a time window, straight from SQLite.

        In-memory counters are all-time (they are hydrated from the database at
        startup); "what has been failing *today*" needs the query."""
        if not self.db_path:
            return {}
        try:
            conn = self._connect()
            try:
                return {
                    str(k): int(n)
                    for k, n in conn.execute(
                        "SELECT COALESCE(error, 'unclassified'), COUNT(*) FROM usage"
                        " WHERE status >= 400 AND ts >= ? GROUP BY 1 ORDER BY 2 DESC",
                        (since,),
                    )
                }
            finally:
                conn.close()
        except Exception as exc:  # noqa: BLE001 - the dashboard must still render
            self.db_error = f"{exc.__class__.__name__}: {exc}"
            return {}

    def cost(self, now: Optional[float] = None) -> dict:
        now = self._clock() if now is None else now
        service_uptime_s = max(0.0, now - self.service_started_at)
        hours = service_uptime_s / 3600.0
        inr = hours * self.gpu_rate_inr_per_hour
        with self._lock:
            completion = self.completion_tokens
            total = self.prompt_tokens + self.completion_tokens
        per_m_out_inr = (inr / completion * 1e6) if completion else None
        per_m_total_inr = (inr / total * 1e6) if total else None
        return {
            "gpu_rate_inr_per_hour": self.gpu_rate_inr_per_hour,
            "inr_per_usd": self.inr_per_usd,
            "billed_hours": hours,
            "inr": inr,
            "usd": inr / self.inr_per_usd if self.inr_per_usd else None,
            "per_1m_output_tokens_inr": per_m_out_inr,
            "per_1m_output_tokens_usd": (
                per_m_out_inr / self.inr_per_usd if per_m_out_inr and self.inr_per_usd else None
            ),
            "per_1m_total_tokens_inr": per_m_total_inr,
        }

    def summary(self, now: Optional[float] = None, *, hours: int = 24) -> dict:
        now = self._clock() if now is None else now
        with self._lock:
            totals = {
                "requests": self.requests,
                "errors": self.errors,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
                "rate_limited": self.rate_limited,
                "rejected_overload": self.rejected_overload,
                "rejected_concurrency": self.rejected_concurrency,
                "auth_failures": self.auth_failures,
                "timeouts": self.timeouts,
                "dropped_audit_rows": self.dropped_rows,
            }
            per_key = [t.as_dict(name) for name, t in self._per_key.items()]
            ttft = list(self._ttft_recent)
        per_key.sort(key=lambda d: d["completion_tokens"], reverse=True)
        return {
            "now": now,
            "process_uptime_s": max(0.0, now - self.process_started_at),
            "service_uptime_s": max(0.0, now - self.service_started_at),
            "service_started_at": self.service_started_at,
            "totals": totals,
            "recent_output_tokens_per_second": self.recent_tokens_per_second(now),
            "recent_ttft_ms_mean": (sum(ttft) / len(ttft)) if ttft else None,
            "latency": self.latency_window(now),
            "errors_by_class": self.errors_by_class(),
            "errors_by_class_window": self.errors_by_class_since(now - hours * 3600),
            "per_key": per_key,
            "hourly": self.hourly_series(hours, now),
            "cost": self.cost(now),
            "storage": {"db_path": self.db_path, "error": self.db_error},
        }

    def prometheus_lines(self) -> list[str]:
        """Extra counters appended to `/metrics` (see `metrics.render_prometheus_text`)."""
        with self._lock:
            values = [
                ("qwenfast:api_requests_total", "API requests metered (all keys).", self.requests),
                ("qwenfast:api_errors_total", "Metered API requests that returned >= 400.", self.errors),
                ("qwenfast:api_prompt_tokens_total", "Prompt tokens billed to API keys.", self.prompt_tokens),
                (
                    "qwenfast:api_completion_tokens_total",
                    "Completion tokens billed to API keys.",
                    self.completion_tokens,
                ),
                ("qwenfast:api_rate_limited_total", "Requests refused with 429.", self.rate_limited),
                ("qwenfast:api_overload_total", "Requests refused with 503 by the queue cap.", self.rejected_overload),
                (
                    "qwenfast:api_concurrency_limited_total",
                    "Requests refused with 429 by the per-key / per-IP stream cap.",
                    self.rejected_concurrency,
                ),
                ("qwenfast:api_timeouts_total", "Requests cut short by --request-timeout.", self.timeouts),
                ("qwenfast:api_auth_failures_total", "Requests refused with 401.", self.auth_failures),
                ("qwenfast:api_dropped_audit_rows_total", "Usage rows dropped by a full writer queue.", self.dropped_rows),
            ]
            by_class = dict(self._errors_by_class)
        lines: list[str] = []
        for name, help_text, value in values:
            lines += [f"# HELP {name} {help_text}", f"# TYPE {name} counter", f"{name} {value}"]
        if by_class:
            lines += [
                "# HELP qwenfast:api_errors_by_class_total Metered failures by reason class.",
                "# TYPE qwenfast:api_errors_by_class_total counter",
            ]
            lines += [
                f'qwenfast:api_errors_by_class_total{{class="{cls}"}} {n}'
                for cls, n in sorted(by_class.items())
            ]
        return lines


def dump_json(obj: dict) -> str:  # tiny helper used by the dashboard script
    return json.dumps(obj, separators=(",", ":"))


__all__ = [
    "ERROR_CLASSES",
    "UsageRecord",
    "UsageRecorder",
    "DEFAULT_GPU_RATE_INR_PER_HOUR",
    "DEFAULT_INR_PER_USD",
    "percentiles",
]
