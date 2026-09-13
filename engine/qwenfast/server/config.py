"""The public-endpoint CLI flags, in one place so the two entry points cannot drift.

`python -m qwenfast.server` (`server/cli.py`) and `python -m qwenfast.runtime.serve`
(`runtime/serve.py`) both expose these; both turn them into the same `create_app(**kwargs)`.
Dependency-free on purpose — importing this must not pull in torch, fastapi, or uvicorn, because
`runtime/serve.py::build_arg_parser` is also used by the offline benches.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Optional

from .usage import DEFAULT_GPU_RATE_INR_PER_HOUR, DEFAULT_INR_PER_USD


def add_public_api_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = p.add_argument_group("public API options (docs/api.md)")
    g.add_argument(
        "--api-keys-file",
        default=None,
        help='JSON file of accepted keys: {"keys":[{"key":"qf-…","name":"alice",'
        '"rpm":60,"tpm":200000,"max_tokens":2048}]}. Hot-reloaded on SIGHUP or mtime change.',
    )
    g.add_argument(
        "--admin-key",
        default=None,
        help="key that may read /admin/usage (also a valid /v1 key). Prefer --admin-key-file.",
    )
    g.add_argument(
        "--admin-key-file",
        default=None,
        help="read the admin key from this file (keeps it out of the process command line)",
    )
    g.add_argument(
        "--demo-key",
        default=None,
        help="key used only by the demo chat proxy; metered separately from public keys",
    )
    g.add_argument("--demo-key-file", default=None, help="read --demo-key from this file")
    g.add_argument(
        "--usage-db",
        default=None,
        help="SQLite file for the per-request usage log (e.g. /home/qwenfast-results/usage.sqlite)",
    )
    g.add_argument(
        "--gpu-rate-inr-per-hour",
        type=float,
        default=DEFAULT_GPU_RATE_INR_PER_HOUR,
        help="instance price used for the cost figures on /admin/usage",
    )
    g.add_argument(
        "--inr-per-usd", type=float, default=DEFAULT_INR_PER_USD, help="FX rate for the $ column"
    )
    g.add_argument(
        "--max-inflight-requests",
        type=int,
        default=0,
        help="concurrent /v1 requests admitted before returning 503 (0 = unlimited). "
        "A sane value is ~2x --max-num-seqs.",
    )
    g.add_argument(
        "--max-output-tokens",
        type=int,
        default=0,
        help="server-wide max_tokens ceiling; a larger request value is silently clamped (0 = off)",
    )
    g.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=0,
        help="reject prompts longer than this with a 400 (0 = only the context-length check)",
    )
    g.add_argument(
        "--max-request-bytes", type=int, default=1_000_000, help="413 above this Content-Length"
    )
    g.add_argument("--max-messages", type=int, default=256, help="400 above this many chat messages")

    r = p.add_argument_group("reliability options")
    r.add_argument(
        "--min-completion-tokens",
        type=int,
        default=0,
        help="reject a prompt only when it leaves less than this much room in the context; "
        "otherwise max_tokens is CLAMPED to the room left instead of being a 400. "
        "This is the flag that replaces --max-prompt-tokens: 256 in a 4096 context means "
        "'a prompt may be up to 3840 tokens, and the reply is whatever fits'.",
    )
    r.add_argument(
        "--request-timeout",
        type=float,
        default=0.0,
        help="seconds before an in-flight request is cancelled and its stream cleanly "
        "terminated (0 = no limit). Guards against an engine that stops yielding.",
    )
    r.add_argument(
        "--max-streams-per-key",
        type=int,
        default=0,
        help="concurrent in-flight requests one API key may hold before 429 (0 = unlimited). "
        "Per-key `max_streams` in the keys file overrides it.",
    )
    r.add_argument(
        "--max-streams-per-ip",
        type=int,
        default=0,
        help="concurrent in-flight requests one client IP may hold before 429 (0 = unlimited)",
    )
    r.add_argument(
        "--client-ip-header",
        default="x-forwarded-for",
        help="header the per-IP cap reads the caller's address from; empty = socket peer only",
    )
    r.add_argument(
        "--drain-timeout",
        type=float,
        default=30.0,
        help="on SIGTERM, stop accepting new requests and give in-flight ones this many "
        "seconds to finish before the process exits",
    )

    c = p.add_argument_group("content logging (private to the GPU host)")
    c.add_argument(
        "--log-content",
        action="store_true",
        help="record the full messages array and the reply of every /v1 request to "
        "--content-log-db, for abuse monitoring. Off by default. Never logs keys or headers.",
    )
    c.add_argument(
        "--content-log-db",
        default=None,
        help="SQLite file for the content log (default: queries.sqlite beside --usage-db)",
    )
    c.add_argument(
        "--content-retention-days",
        type=float,
        default=0.0,
        help="delete content rows older than this many days (0 = keep forever)",
    )
    return p


def _default_content_db(usage_db: Optional[str]) -> Optional[str]:
    """`queries.sqlite` next to `usage.sqlite`, so one `--usage-db` places both."""
    if not usage_db:
        return None
    return os.path.join(os.path.dirname(os.path.abspath(usage_db)), "queries.sqlite")


def _read_secret_file(path: Optional[str]) -> Optional[str]:
    """First non-empty line of `path`, or None. Never logs the contents."""
    if not path:
        return None
    try:
        with open(os.path.expanduser(path), "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    return line
    except OSError:
        return None
    return None


def public_api_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Translate the parsed flags into `create_app(**kwargs)`. Tolerates a namespace that has
    none of them (so an older launch script keeps working)."""

    def get(name, default=None):
        return getattr(args, name, default)

    admin = get("admin_key") or _read_secret_file(get("admin_key_file"))
    demo = get("demo_key") or _read_secret_file(get("demo_key_file"))
    usage_db = get("usage_db")
    content_db = None
    if get("log_content", False):
        content_db = get("content_log_db") or _default_content_db(usage_db)
    return {
        "api_keys_file": get("api_keys_file"),
        "admin_key": admin,
        "demo_key": demo,
        "usage_db": get("usage_db"),
        "gpu_rate_inr_per_hour": get("gpu_rate_inr_per_hour", DEFAULT_GPU_RATE_INR_PER_HOUR),
        "inr_per_usd": get("inr_per_usd", DEFAULT_INR_PER_USD),
        "max_inflight_requests": get("max_inflight_requests", 0) or 0,
        "max_output_tokens": get("max_output_tokens", 0) or None,
        "max_prompt_tokens": get("max_prompt_tokens", 0) or None,
        "max_request_bytes": get("max_request_bytes", 1_000_000),
        "max_messages": get("max_messages", 256),
        # -- reliability knobs
        "max_context_len": get("max_model_len", 0) or None,
        "min_completion_tokens": get("min_completion_tokens", 0) or 0,
        "request_timeout_s": get("request_timeout", 0.0) or 0.0,
        "max_streams_per_key": get("max_streams_per_key", 0) or 0,
        "max_streams_per_ip": get("max_streams_per_ip", 0) or 0,
        "client_ip_header": get("client_ip_header", "x-forwarded-for") or "",
        "drain_timeout_s": get("drain_timeout", 0.0) or 0.0,
        # -- content logging
        "content_log_db": content_db,
        "content_retention_days": get("content_retention_days", 0.0) or 0.0,
    }


def describe_public_config(kwargs: dict[str, Any], key_names: list[str]) -> str:
    """A one-block startup banner. **Prints names only — never key material.**"""
    lines = [
        "public API:",
        f"  keys file          : {kwargs.get('api_keys_file') or '(none — endpoint is open)'}",
        f"  keys loaded        : {len(key_names)} ({', '.join(key_names) if key_names else '-'})",
        f"  usage db           : {kwargs.get('usage_db') or '(memory only)'}",
        f"  admin key          : {'set' if kwargs.get('admin_key') else 'not set'}",
        f"  demo key           : {'set' if kwargs.get('demo_key') else 'not set'}",
        f"  max inflight       : {kwargs.get('max_inflight_requests') or 'unlimited'}",
        f"  max output tokens  : {kwargs.get('max_output_tokens') or 'unlimited'}",
        f"  max prompt tokens  : {kwargs.get('max_prompt_tokens') or '(context-length check only)'}",
        f"  context window     : {kwargs.get('max_context_len') or '(engine-reported)'}",
        f"  min completion     : {kwargs.get('min_completion_tokens') or '(clamp only, no floor)'}",
        f"  request timeout    : {kwargs.get('request_timeout_s') or 'none'}s",
        f"  streams per key/IP : {kwargs.get('max_streams_per_key') or 'unlimited'}"
        f" / {kwargs.get('max_streams_per_ip') or 'unlimited'}",
        f"  drain timeout      : {kwargs.get('drain_timeout_s') or 0}s",
        f"  content log        : {kwargs.get('content_log_db') or 'OFF'}"
        + (
            f" (retain {kwargs['content_retention_days']}d)"
            if kwargs.get("content_retention_days")
            else ""
        ),
        f"  GPU rate           : INR {kwargs.get('gpu_rate_inr_per_hour')}/h",
    ]
    return "\n".join(lines)


__all__ = ["add_public_api_args", "public_api_kwargs", "describe_public_config"]
