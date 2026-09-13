"""Opt-in content logging: the full prompt and reply of every request, to SQLite.

This is a **separate database and a separate switch** from `usage.py` on
purpose. `usage.sqlite` holds counters -- token counts, latencies, status codes
-- and is safe to keep forever. `queries.sqlite` holds *what people typed*, and
is governed by different rules:

* it is written only when `--log-content` is passed (default off);
* it lives in its own file, so "stop retaining content" is `rm queries.sqlite`
  and does not touch the cost/usage history the dashboard is computed from;
* `--content-retention-days` prunes it on a schedule, so retention is a
  decision someone made rather than one nobody made;
* **no key material is ever written.** Rows carry the key *name* (`public`,
  `demo`), never the secret, and no request header is stored.

This file holds user-submitted text and client IPs. It is private to the GPU host:
nothing about it appears on the public API, on `/status`, or on the demo page,
and the only two readers are the admin-key `/admin/queries` endpoint and the
operator's local `scripts/export_queries.py` / `scripts/usage_dashboard.py`.
Retention is whatever `--content-retention-days` says (default: forever).

Everything else follows `usage.py`'s design constraint exactly: nothing here may
run on the streaming hot path. The handler builds one `QueryRecord` after the
response is finished and hands it to `record()`, which appends to a bounded
queue drained by a single daemon thread. A full queue drops the row and counts
the drop; losing a logged query is strictly better than stalling a generation
loop.
"""

from __future__ import annotations

import csv
import io
import json
import os
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable, Optional

QUEUE_MAX = 4_000
BATCH_MAX = 64
FLUSH_INTERVAL_S = 1.0
PRUNE_INTERVAL_S = 3600.0
SCHEMA_VERSION = 1

#: Hard ceiling on any single stored text field. A 1 MB body cap (see
#: `--max-request-bytes`) means one row could otherwise be a megabyte of JSON;
#: at a few hundred requests an hour that is a disk-filling bug, not a feature.
MAX_TEXT_CHARS = 200_000


@dataclass
class QueryRecord:
    ts: float
    key_name: str
    client_ip: str
    endpoint: str
    model: str
    messages_json: str
    response_text: str
    reasoning_text: Optional[str]
    prompt_tokens: int
    completion_tokens: int
    status: int
    finish_reason: Optional[str]
    ttft_ms: Optional[float]
    duration_ms: Optional[float]
    request_id: str


_DDL = """
CREATE TABLE IF NOT EXISTS queries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              REAL    NOT NULL,
    key_name        TEXT    NOT NULL,
    client_ip       TEXT,
    endpoint        TEXT    NOT NULL,
    model           TEXT,
    messages_json   TEXT,
    response_text   TEXT,
    reasoning_text  TEXT,
    prompt_tokens   INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    status          INTEGER NOT NULL DEFAULT 200,
    finish_reason   TEXT,
    ttft_ms         REAL,
    duration_ms     REAL,
    request_id      TEXT
);
CREATE INDEX IF NOT EXISTS queries_ts ON queries (ts);
CREATE INDEX IF NOT EXISTS queries_key_ts ON queries (key_name, ts);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
"""

#: The column order every export uses, so JSON, JSONL and CSV agree.
COLUMNS: tuple[str, ...] = (
    "id", "ts", "key_name", "client_ip", "endpoint", "model", "messages_json",
    "response_text", "reasoning_text", "prompt_tokens", "completion_tokens",
    "status", "finish_reason", "ttft_ms", "duration_ms", "request_id",
)


def _clip(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    text = str(text)
    if len(text) <= MAX_TEXT_CHARS:
        return text
    return text[:MAX_TEXT_CHARS] + f"\n[...truncated at {MAX_TEXT_CHARS} chars]"


def messages_to_json(messages: Iterable[Any]) -> str:
    """Serialise the request's messages array. Never raises.

    Takes whatever the handler has -- pydantic models or plain dicts -- and
    produces compact JSON, with a placeholder rather than an exception if some
    field refuses to serialise.
    """
    try:
        out = []
        for m in messages:
            if hasattr(m, "model_dump"):
                out.append(m.model_dump(exclude_none=True))
            elif isinstance(m, dict):
                out.append(m)
            else:
                out.append({"role": getattr(m, "role", "?"), "content": str(m)})
        return _clip(json.dumps(out, ensure_ascii=False, separators=(",", ":"))) or "[]"
    except Exception:  # noqa: BLE001 - logging must never fail a response
        return '[{"role":"?","content":"<unserialisable>"}]'


class QueryLogger:
    """Bounded-queue writer for `queries.sqlite`. Disabled unless a path is given.

    `QueryLogger(None)` is the always-off object every existing caller gets;
    `enabled` is False, `record()` is a no-op, and nothing is opened. That keeps
    the flag's default (off) free of any code path in the handlers beyond one
    boolean test.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        *,
        retention_days: float = 0.0,
        clock=time.time,
        start_writer: bool = True,
    ) -> None:
        self.db_path = db_path
        self.retention_days = max(0.0, float(retention_days or 0.0))
        self._clock = clock
        self._lock = threading.Lock()
        self.rows_written = 0
        self.dropped_rows = 0
        self.rows_pruned = 0
        self.db_error: Optional[str] = None
        self._last_prune = 0.0

        self._queue: "queue.Queue[Optional[QueryRecord]]" = queue.Queue(maxsize=QUEUE_MAX)
        self._writer: Optional[threading.Thread] = None

        if db_path:
            try:
                self._init_db()
            except Exception as exc:  # noqa: BLE001 - never block serving
                self.db_error = f"{exc.__class__.__name__}: {exc}"
                self.db_path = None
        if self.db_path and start_writer:
            self._writer = threading.Thread(
                target=self._writer_loop, name="query-writer", daemon=True
            )
            self._writer.start()

    @property
    def enabled(self) -> bool:
        return bool(self.db_path)

    # -- database ---------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
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
            conn.execute(
                "INSERT OR REPLACE INTO meta (k, v) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            conn.commit()
        finally:
            conn.close()
        # Content is not world-readable even if the results directory is.
        try:
            os.chmod(self.db_path, 0o600)
        except OSError:
            pass

    def _writer_loop(self) -> None:  # pragma: no cover - exercised via flush() in tests
        conn = None
        pending: list[QueryRecord] = []
        while True:
            try:
                item = self._queue.get(timeout=FLUSH_INTERVAL_S)
                if item is None:
                    self._flush(conn or self._safe_connect(), pending)
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
                        self._flush(conn or self._safe_connect(), pending)
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
            if conn is not None:
                self._maybe_prune(conn)

    def _safe_connect(self) -> Optional[sqlite3.Connection]:
        try:
            return self._connect()
        except Exception as exc:  # noqa: BLE001
            self.db_error = f"{exc.__class__.__name__}: {exc}"
            return None

    def _flush(self, conn: Optional[sqlite3.Connection], pending: list[QueryRecord]) -> None:
        if not pending:
            return
        if conn is None:
            pending.clear()
            return
        try:
            conn.executemany(
                "INSERT INTO queries (ts, key_name, client_ip, endpoint, model, messages_json,"
                " response_text, reasoning_text, prompt_tokens, completion_tokens, status,"
                " finish_reason, ttft_ms, duration_ms, request_id)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        r.ts, r.key_name, r.client_ip, r.endpoint, r.model, r.messages_json,
                        r.response_text, r.reasoning_text, r.prompt_tokens, r.completion_tokens,
                        r.status, r.finish_reason, r.ttft_ms, r.duration_ms, r.request_id,
                    )
                    for r in pending
                ],
            )
            conn.commit()
            with self._lock:
                self.rows_written += len(pending)
        except Exception as exc:  # noqa: BLE001
            self.db_error = f"{exc.__class__.__name__}: {exc}"
        finally:
            pending.clear()

    def _maybe_prune(self, conn: sqlite3.Connection) -> None:
        if not self.retention_days:
            return
        now = time.monotonic()
        if now - self._last_prune < PRUNE_INTERVAL_S:
            return
        self._last_prune = now
        self.prune(conn)

    def prune(self, conn: Optional[sqlite3.Connection] = None) -> int:
        """Delete rows older than `retention_days`. Returns the number removed."""
        if not self.retention_days or not self.db_path:
            return 0
        owned = conn is None
        conn = conn or self._safe_connect()
        if conn is None:
            return 0
        try:
            cutoff = self._clock() - self.retention_days * 86400.0
            cur = conn.execute("DELETE FROM queries WHERE ts < ?", (cutoff,))
            conn.commit()
            n = int(cur.rowcount or 0)
            with self._lock:
                self.rows_pruned += n
            return n
        except Exception as exc:  # noqa: BLE001
            self.db_error = f"{exc.__class__.__name__}: {exc}"
            return 0
        finally:
            if owned:
                conn.close()

    # -- recording --------------------------------------------------------------

    def record(self, rec: QueryRecord) -> None:
        """Never blocks, never raises. No-op when content logging is off."""
        if not self.db_path:
            return
        try:
            rec.messages_json = _clip(rec.messages_json) or ""
            rec.response_text = _clip(rec.response_text) or ""
            rec.reasoning_text = _clip(rec.reasoning_text)
            try:
                self._queue.put_nowait(rec)
            except queue.Full:
                with self._lock:
                    self.dropped_rows += 1
        except Exception:  # noqa: BLE001
            pass

    def flush(self, timeout: float = 5.0) -> None:
        if not self.db_path:
            return
        deadline = time.monotonic() + timeout
        while not self._queue.empty() and time.monotonic() < deadline:
            time.sleep(0.01)
        time.sleep(0.05)

    def shutdown(self, timeout: float = 5.0) -> None:
        if self._writer is None:
            return
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._writer.join(timeout=timeout)

    # -- reading ----------------------------------------------------------------

    def export(
        self,
        *,
        since: Optional[float] = None,
        until: Optional[float] = None,
        limit: int = 100,
        key_name: Optional[str] = None,
    ) -> list[dict]:
        """Rows newest-first, for `/admin/queries` and `scripts/export_queries.py`."""
        if not self.db_path:
            return []
        where, params = ["1=1"], []
        if since is not None:
            where.append("ts >= ?")
            params.append(float(since))
        if until is not None:
            where.append("ts <= ?")
            params.append(float(until))
        if key_name:
            where.append("key_name = ?")
            params.append(key_name)
        sql = (
            f"SELECT {', '.join(COLUMNS)} FROM queries WHERE {' AND '.join(where)}"
            " ORDER BY ts DESC LIMIT ?"
        )
        params.append(max(1, int(limit)))
        try:
            conn = self._connect()
            try:
                return [dict(zip(COLUMNS, row)) for row in conn.execute(sql, params)]
            finally:
                conn.close()
        except Exception as exc:  # noqa: BLE001
            self.db_error = f"{exc.__class__.__name__}: {exc}"
            return []

    def counts(
        self, *, since: Optional[float] = None, by: str = "key_name", limit: int = 50
    ) -> list[dict]:
        """Query counts grouped by key or by client IP, busiest first.

        The cheapest abuse signal there is: one key or one address responsible
        for a disproportionate share of traffic is the thing to look at first,
        and it is a `GROUP BY` rather than a scroll through the text.
        """
        if not self.db_path:
            return []
        column = "client_ip" if by in ("ip", "client_ip") else "key_name"
        where, params = ["1=1"], []
        if since is not None:
            where.append("ts >= ?")
            params.append(float(since))
        params.append(max(1, int(limit)))
        try:
            conn = self._connect()
            try:
                rows = conn.execute(
                    f"SELECT COALESCE({column}, '-'), COUNT(*),"
                    f" COALESCE(SUM(prompt_tokens), 0), COALESCE(SUM(completion_tokens), 0),"
                    f" MIN(ts), MAX(ts) FROM queries WHERE {' AND '.join(where)}"
                    f" GROUP BY 1 ORDER BY 2 DESC LIMIT ?",
                    params,
                ).fetchall()
            finally:
                conn.close()
        except Exception as exc:  # noqa: BLE001
            self.db_error = f"{exc.__class__.__name__}: {exc}"
            return []
        return [
            {
                by if by in ("ip", "client_ip") else "key_name": r[0],
                "queries": int(r[1]),
                "prompt_tokens": int(r[2]),
                "completion_tokens": int(r[3]),
                "first_ts": r[4],
                "last_ts": r[5],
            }
            for r in rows
        ]

    def grep(self, pattern: str, *, since: Optional[float] = None, limit: int = 200) -> list[dict]:
        """Rows whose prompt or reply matches `pattern` (Python regex, case-insensitive).

        Done in Python rather than SQL so the pattern is a real regex and not
        SQLite's `LIKE`; the row budget is bounded by `limit * 20` scanned so a
        pathological pattern cannot walk a million rows.
        """
        import re

        try:
            rx = re.compile(pattern, re.IGNORECASE | re.DOTALL)
        except re.error as exc:
            raise ValueError(f"bad --grep regex: {exc}") from exc
        out: list[dict] = []
        for row in self.export(since=since, limit=max(limit * 20, limit)):
            haystack = f"{row.get('messages_json') or ''}\n{row.get('response_text') or ''}"
            if rx.search(haystack):
                out.append(row)
                if len(out) >= limit:
                    break
        return out

    def stats(self) -> dict:
        with self._lock:
            base = {
                "enabled": self.enabled,
                "db_path": self.db_path,
                "rows_written": self.rows_written,
                "dropped_rows": self.dropped_rows,
                "rows_pruned": self.rows_pruned,
                "retention_days": self.retention_days or None,
                "error": self.db_error,
            }
        if self.db_path:
            try:
                conn = self._connect()
                try:
                    n, first, last = conn.execute(
                        "SELECT COUNT(*), MIN(ts), MAX(ts) FROM queries"
                    ).fetchone()
                    base.update({"rows": int(n or 0), "oldest_ts": first, "newest_ts": last})
                finally:
                    conn.close()
            except Exception as exc:  # noqa: BLE001
                base["error"] = f"{exc.__class__.__name__}: {exc}"
        return base


def rows_to_csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(COLUMNS), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buf.getvalue()


def rows_to_jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


__all__ = [
    "COLUMNS",
    "QueryLogger",
    "QueryRecord",
    "messages_to_json",
    "rows_to_csv",
    "rows_to_jsonl",
]
