"""Multi-key auth + per-key rate limiting for the public `qwenfast` endpoint.

Two objects, both deliberately dependency-free and both written so that *no* failure in here can
take the engine down (the rule: a bug in the HTTP layer must degrade to "this request got a
4xx/5xx", never to "the device thread died"):

* `KeyStore` — loads `{"keys": [{"key": …, "name": …, "rpm": …, "tpm": …, "max_tokens": …}]}` from
  a JSON file and hot-reloads it. Reload happens on two triggers: `SIGHUP` (the operator's
  explicit "I just added a key") and an mtime poll (the fallback for `scp`-and-forget). A reload
  that fails to parse is *discarded* — the previously-good key set keeps serving, and the error is
  remembered for `/admin/usage` to report. That asymmetry is the whole point: a fat-fingered edit
  of the keys file must not lock every user out of a running server.

* `RateLimiter` — a per-key sliding 60-second window over both requests and *tokens*. Tokens are
  the meaningful unit here (they are what a heavy user actually consumes), but they are only known
  *after* generation, so the limiter admits on the request
  count and the *already-recorded* token history, then records the actual usage when the stream
  ends. A key that just emitted 200k tokens is therefore locked out on its next request, not
  mid-stream — which is the right shape for streaming: never kill a response in flight.

Key material never appears in logs or in any response body. Everything user-visible refers to a
key by its `name`, and `/admin/usage` is keyed by name too.
"""

from __future__ import annotations

import hmac
import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Deque, Dict, Iterable, Optional

DEFAULT_RELOAD_POLL_S = 2.0
WINDOW_S = 60.0


@dataclass(frozen=True)
class ApiKey:
    """One entry from the keys file. `secret` is never rendered anywhere."""

    secret: str
    name: str
    rpm: Optional[int] = None          # requests / 60 s; None = unlimited
    tpm: Optional[int] = None          # completion tokens / 60 s; None = unlimited
    max_tokens: Optional[int] = None   # per-request completion cap; None = server default
    #: Concurrent in-flight requests this key may hold. None = the server-wide
    #: `--max-streams-per-key`. This is the *fairness* knob: `rpm`/`tpm` bound
    #: how much a key may ask for over a minute, and bound nothing at all about
    #: how many of the GPU's 128 sequence slots it may sit in at once.
    max_streams: Optional[int] = None
    admin: bool = False
    disabled: bool = False

    def redacted(self) -> dict:
        return {
            "name": self.name,
            "rpm": self.rpm,
            "tpm": self.tpm,
            "max_tokens": self.max_tokens,
            "max_streams": self.max_streams,
            "admin": self.admin,
            "disabled": self.disabled,
        }


def _as_int(value, field_name: str) -> Optional[int]:
    if value is None:
        return None
    try:
        n = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer, got {value!r}") from exc
    if n <= 0:
        return None  # 0 / negative == "no limit", so a config can turn one off explicitly
    return n


def parse_keys_document(doc: object) -> list[ApiKey]:
    """Validate a parsed keys document. Raises `ValueError` on anything malformed.

    Strict on purpose: the caller (`KeyStore.reload`) turns an exception into "keep the old key
    set", so being strict here is what makes a broken edit a no-op instead of a partial apply.
    """
    if not isinstance(doc, dict):
        raise ValueError("keys file must be a JSON object")
    entries = doc.get("keys")
    if not isinstance(entries, list):
        raise ValueError('keys file must have a "keys" array')

    out: list[ApiKey] = []
    seen_secrets: set[str] = set()
    seen_names: set[str] = set()
    for i, raw in enumerate(entries):
        if not isinstance(raw, dict):
            raise ValueError(f"keys[{i}] must be an object")
        secret = raw.get("key")
        if not isinstance(secret, str) or not secret.strip():
            raise ValueError(f'keys[{i}] is missing a non-empty "key"')
        secret = secret.strip()
        name = raw.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f'keys[{i}] is missing a non-empty "name"')
        name = name.strip()
        if secret in seen_secrets:
            raise ValueError(f"keys[{i}] ({name}) duplicates an earlier key value")
        if name in seen_names:
            raise ValueError(f"keys[{i}] duplicates the name {name!r}")
        seen_secrets.add(secret)
        seen_names.add(name)
        out.append(
            ApiKey(
                secret=secret,
                name=name,
                rpm=_as_int(raw.get("rpm"), f"keys[{i}].rpm"),
                tpm=_as_int(raw.get("tpm"), f"keys[{i}].tpm"),
                max_tokens=_as_int(raw.get("max_tokens"), f"keys[{i}].max_tokens"),
                max_streams=_as_int(raw.get("max_streams"), f"keys[{i}].max_streams"),
                admin=bool(raw.get("admin", False)),
                disabled=bool(raw.get("disabled", False)),
            )
        )
    return out


class KeyStore:
    """The set of accepted API keys, hot-reloadable from a JSON file.

    Thread-safe (a plain lock around a dict swap): the reload can be driven from a signal handler
    or from any request thread, while lookups happen on the event loop.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        *,
        static_keys: Iterable[ApiKey] = (),
        poll_interval_s: float = DEFAULT_RELOAD_POLL_S,
        clock=time.monotonic,
    ) -> None:
        self.path = path
        self._static = tuple(static_keys)
        self._poll_interval_s = poll_interval_s
        self._clock = clock
        self._lock = threading.Lock()
        self._by_secret: Dict[str, ApiKey] = {}
        self._mtime: Optional[float] = None
        self._last_poll = -1e9
        self._forced = False
        self.load_error: Optional[str] = None
        self.loaded_at: Optional[float] = None
        self.reload_count = 0
        self._apply(list(self._static))
        if path:
            self.reload(force=True)
            # We just read the file; start the poll clock now rather than leaving `_last_poll`
            # at -inf, which would make the very first request pay a redundant `stat`.
            self._last_poll = self._clock()

    # -- loading ----------------------------------------------------------------

    def _apply(self, keys: list[ApiKey]) -> None:
        table = {k.secret: k for k in self._static}
        for k in keys:
            table[k.secret] = k
        with self._lock:
            self._by_secret = table

    def reload(self, *, force: bool = False) -> bool:
        """Re-read the keys file. Returns True if the in-memory set was replaced.

        Never raises: a missing or malformed file leaves the current set in place and records
        `load_error`.
        """
        if not self.path:
            return False
        try:
            stat = os.stat(self.path)
        except OSError as exc:
            self.load_error = f"cannot stat keys file: {exc.__class__.__name__}"
            return False
        if not force and self._mtime is not None and stat.st_mtime == self._mtime:
            return False
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                doc = json.load(fh)
            keys = parse_keys_document(doc)
        except Exception as exc:  # noqa: BLE001 - a bad edit must not lock anyone out
            self.load_error = f"{exc.__class__.__name__}: {exc}"
            # Remember the mtime anyway so we do not re-parse the same broken file every poll.
            self._mtime = stat.st_mtime
            return False
        self._apply(keys)
        self._mtime = stat.st_mtime
        self.load_error = None
        self.loaded_at = time.time()
        self.reload_count += 1
        return True

    def request_reload(self) -> None:
        """Signal-handler-safe: just flips a flag; the next `maybe_reload` does the I/O."""
        self._forced = True

    def maybe_reload(self) -> None:
        """Cheap enough to call on every request: an mtime `stat` at most every `poll_interval_s`."""
        if not self.path:
            return
        if self._forced:
            self._forced = False
            self.reload(force=True)
            return
        now = self._clock()
        if now - self._last_poll < self._poll_interval_s:
            return
        self._last_poll = now
        self.reload()

    def install_sighup_handler(self) -> bool:
        """Best-effort `SIGHUP` → reload. Returns False off the main thread / on Windows."""
        try:
            import signal

            previous = signal.getsignal(signal.SIGHUP)

            def _handler(signum, frame):  # pragma: no cover - exercised by hand
                self.request_reload()
                if callable(previous) and previous not in (signal.SIG_DFL, signal.SIG_IGN):
                    previous(signum, frame)

            signal.signal(signal.SIGHUP, _handler)
            return True
        except Exception:  # noqa: BLE001 - not being able to catch SIGHUP is not fatal
            return False

    # -- lookup -----------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """False means "no auth configured" — every request is allowed through."""
        return bool(self.path) or bool(self._static)

    def lookup(self, secret: Optional[str]) -> Optional[ApiKey]:
        """Constant-time-ish match. Returns None for unknown, disabled, or absent keys."""
        if not secret:
            return None
        with self._lock:
            table = self._by_secret
        found = table.get(secret)
        if found is None:
            # Fall back to a comparison that does not leak *which* key was close, for the
            # (rare, small-table) case where a timing oracle on dict lookup would matter.
            for candidate in table.values():
                if hmac.compare_digest(candidate.secret, secret):
                    found = candidate
                    break
        if found is None or found.disabled:
            return None
        return found

    def names(self) -> list[str]:
        with self._lock:
            return sorted(k.name for k in self._by_secret.values())

    def describe(self) -> list[dict]:
        with self._lock:
            keys = list(self._by_secret.values())
        return [k.redacted() for k in sorted(keys, key=lambda k: k.name)]


# --------------------------------------------------------------------------
# Sliding-window rate limiting
# --------------------------------------------------------------------------


@dataclass
class _Window:
    requests: Deque[float] = field(default_factory=deque)
    tokens: Deque[tuple[float, int]] = field(default_factory=deque)
    token_sum: int = 0


@dataclass(frozen=True)
class RateDecision:
    allowed: bool
    retry_after_s: float = 0.0
    reason: str = ""


class RateLimiter:
    """Per-key sliding 60 s windows over request count and completion tokens."""

    def __init__(self, *, window_s: float = WINDOW_S, clock=time.monotonic) -> None:
        self._window_s = window_s
        self._clock = clock
        self._lock = threading.Lock()
        self._windows: Dict[str, _Window] = {}
        self.rate_limited_total = 0

    def _prune(self, w: _Window, now: float) -> None:
        cutoff = now - self._window_s
        while w.requests and w.requests[0] <= cutoff:
            w.requests.popleft()
        while w.tokens and w.tokens[0][0] <= cutoff:
            w.token_sum -= w.tokens.popleft()[1]
        if w.token_sum < 0:  # defensive; cannot happen with the arithmetic above
            w.token_sum = 0

    def check_and_admit(self, key: ApiKey) -> RateDecision:
        """Admit one request for `key`, or refuse with a `Retry-After` hint.

        On admission the request is immediately counted, so N concurrent requests cannot all
        observe the same pre-admission count and slip through together.
        """
        if key.rpm is None and key.tpm is None:
            return RateDecision(True)
        now = self._clock()
        with self._lock:
            w = self._windows.setdefault(key.name, _Window())
            self._prune(w, now)
            if key.rpm is not None and len(w.requests) >= key.rpm:
                retry = max(0.0, self._window_s - (now - w.requests[0]))
                self.rate_limited_total += 1
                return RateDecision(
                    False, retry, f"request rate limit ({key.rpm}/min) exceeded"
                )
            if key.tpm is not None and w.token_sum >= key.tpm:
                retry = max(0.0, self._window_s - (now - w.tokens[0][0])) if w.tokens else self._window_s
                self.rate_limited_total += 1
                return RateDecision(
                    False, retry, f"token rate limit ({key.tpm} tokens/min) exceeded"
                )
            w.requests.append(now)
        return RateDecision(True)

    def record_tokens(self, key_name: str, tokens: int) -> None:
        if tokens <= 0:
            return
        now = self._clock()
        with self._lock:
            w = self._windows.setdefault(key_name, _Window())
            self._prune(w, now)
            w.tokens.append((now, tokens))
            w.token_sum += tokens

    def snapshot(self, key_name: str) -> dict:
        now = self._clock()
        with self._lock:
            w = self._windows.get(key_name)
            if w is None:
                return {"requests_in_window": 0, "tokens_in_window": 0}
            self._prune(w, now)
            return {"requests_in_window": len(w.requests), "tokens_in_window": w.token_sum}


# --------------------------------------------------------------------------
# Concurrency fairness: per-key and per-IP in-flight caps
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ConcurrencyDecision:
    allowed: bool
    reason: str = ""
    scope: str = ""          # "key" | "ip" -- which cap refused, for metrics
    retry_after_s: int = 2


class ConcurrencyLimiter:
    """Concurrent in-flight requests per API key **and** per client IP.

    Why both, and why this is not the same thing as `--max-inflight-requests`:
    the global admission cap is a *server* protection (it stops unbounded
    queueing), and it is entirely happy to let one caller hold all 128 sequence
    slots for ten minutes while everyone else gets a `503`. `rpm` does not stop
    that either -- 8 requests a minute is well inside `rpm 1000`, and 8 streams
    of 4,096 tokens each is most of the machine.

    So: a per-key cap (the unit a key owner controls) *and* a per-IP cap (the
    unit an anonymous sharer of one public key controls), both enforced, both
    reported. A refusal is a `429` with `Retry-After`, not a `503`: it is the
    caller's own budget that is full, not the server's.

    Thread-safety: a plain lock. Acquire/release happen once per request on the
    event loop, never per token, so the lock is never contended in the hot path.

    The IP is whatever `--client-ip-header` (default `x-forwarded-for`, first
    entry) says, falling back to the socket peer. Behind a CDN and a hosting
    provider's edge proxy every socket peer can be the *same* private address,
    so without the header the per-IP cap would be a second,
    stricter global cap -- and with it, the value is client-assertable. That is
    an accepted trade: this cap exists to keep an honest heavy user from
    monopolising the GPU, not to stop an attacker, who has to be stopped by the
    key.
    """

    def __init__(self, *, per_key: int = 0, per_ip: int = 0) -> None:
        self.per_key = max(0, int(per_key))
        self.per_ip = max(0, int(per_ip))
        self._lock = threading.Lock()
        self._by_key: Dict[str, int] = {}
        self._by_ip: Dict[str, int] = {}
        self.rejected_total = 0
        self.peak_per_key = 0
        self.peak_per_ip = 0

    @property
    def enabled(self) -> bool:
        return bool(self.per_key or self.per_ip)

    def limit_for(self, key: ApiKey) -> int:
        """The key's own cap if it sets one, else the server-wide default.

        An admin key is never capped -- the operator must be able to look at a
        server that is refusing everybody else."""
        if key.admin:
            return 0
        if key.max_streams:
            return int(key.max_streams)
        return self.per_key

    def acquire(self, key: ApiKey, ip: Optional[str]) -> ConcurrencyDecision:
        """Take one slot against both counters, or take neither and say why."""
        key_cap = self.limit_for(key)
        ip_cap = 0 if key.admin else self.per_ip
        name = key.name
        ip = ip or "-"
        with self._lock:
            if key_cap and self._by_key.get(name, 0) >= key_cap:
                self.rejected_total += 1
                return ConcurrencyDecision(
                    False,
                    f"key '{name}' already has {key_cap} requests in flight "
                    f"(per-key concurrency limit)",
                    scope="key",
                )
            if ip_cap and self._by_ip.get(ip, 0) >= ip_cap:
                self.rejected_total += 1
                return ConcurrencyDecision(
                    False,
                    f"your client already has {ip_cap} requests in flight "
                    f"(per-IP concurrency limit)",
                    scope="ip",
                )
            nk = self._by_key.get(name, 0) + 1
            ni = self._by_ip.get(ip, 0) + 1
            self._by_key[name] = nk
            self._by_ip[ip] = ni
            self.peak_per_key = max(self.peak_per_key, nk)
            self.peak_per_ip = max(self.peak_per_ip, ni)
        return ConcurrencyDecision(True)

    def release(self, key: ApiKey, ip: Optional[str]) -> None:
        """Idempotent-ish: never goes below zero, and forgets empty buckets so a
        long-lived server does not accumulate one dict entry per IP ever seen."""
        name = key.name
        ip = ip or "-"
        with self._lock:
            for table, k in ((self._by_key, name), (self._by_ip, ip)):
                n = table.get(k, 0) - 1
                if n > 0:
                    table[k] = n
                else:
                    table.pop(k, None)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "per_key_limit": self.per_key or None,
                "per_ip_limit": self.per_ip or None,
                "in_flight_by_key": dict(sorted(self._by_key.items())),
                "distinct_ips_in_flight": len(self._by_ip),
                "peak_per_key": self.peak_per_key,
                "peak_per_ip": self.peak_per_ip,
                "rejected_total": self.rejected_total,
            }


def clamp_key(key: ApiKey, **overrides) -> ApiKey:
    """`dataclasses.replace` under a friendlier name (used by the CLI to apply flag defaults)."""
    return replace(key, **overrides)


__all__ = [
    "ApiKey",
    "ConcurrencyDecision",
    "ConcurrencyLimiter",
    "KeyStore",
    "RateDecision",
    "RateLimiter",
    "clamp_key",
    "parse_keys_document",
]
