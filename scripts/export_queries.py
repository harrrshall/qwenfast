#!/usr/bin/env python3
"""Pull the content log off the server, for abuse review.

    python3 scripts/export_queries.py --since 24h --format csv
    python3 scripts/export_queries.py --counts key            # who is sending the most
    python3 scripts/export_queries.py --counts ip --since 6h
    python3 scripts/export_queries.py --grep '(?i)\\bssn\\b|credit.card' --since 7d

Reads `/admin/queries` on the public server with the admin key from `.secrets/admin_key` (or
`QWENFAST_ADMIN_KEY`) and writes into `exports/` (gitignored). Nothing here is deployed and
nothing here is public: the content log lives only on the GPU host, and this is the operator's local
window onto it.

Requires the server to be running with `--log-content`; without it the endpoint answers
`{"enabled": false}` and this says so rather than writing an empty file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE = "http://127.0.0.1:8000"
EXPORT_DIR = ROOT / "exports"


def read_admin_key() -> str:
    key = os.environ.get("QWENFAST_ADMIN_KEY")
    if key:
        return key.strip()
    path = ROOT / ".secrets" / "admin_key"
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return line
    except OSError:
        pass
    raise SystemExit(
        "No admin key. Set QWENFAST_ADMIN_KEY or create .secrets/admin_key "
        "(python3 scripts/make_api_keys.py --admin)."
    )


_DURATION = re.compile(r"^(\d+(?:\.\d+)?)\s*([smhdw])$", re.I)
_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def parse_since(value: str | None) -> float | None:
    """`24h` / `7d` / `90m` -> a unix timestamp; an ISO date also works."""
    if not value:
        return None
    import time

    m = _DURATION.match(value.strip())
    if m:
        return time.time() - float(m.group(1)) * _UNITS[m.group(2).lower()]
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError as exc:
        raise SystemExit(f"--since: not a duration (24h, 7d) or ISO date: {value!r}") from exc


def fetch(base: str, key: str, params: dict, timeout: float = 60.0) -> tuple[str, str]:
    """Returns (body, content-type). Errors become a readable SystemExit."""
    url = f"{base.rstrip('/')}/admin/queries?" + urllib.parse.urlencode(
        {k: v for k, v in params.items() if v is not None}
    )
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            return res.read().decode("utf-8"), res.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:400]
        if exc.code in (401, 403):
            raise SystemExit(f"HTTP {exc.code}: the admin key was rejected. {body}") from exc
        raise SystemExit(f"HTTP {exc.code} from {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"cannot reach {base}: {exc.reason}") from exc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="export_queries.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--base", default=os.environ.get("QWENFAST_BASE_URL", DEFAULT_BASE))
    p.add_argument("--since", default=None, help="duration (24h, 7d, 90m) or ISO date")
    p.add_argument("--limit", type=int, default=1000, help="maximum rows")
    p.add_argument("--key", default=None, help="only this API key name")
    p.add_argument(
        "--grep",
        default=None,
        help="Python regex; keeps only rows whose prompt or reply matches. "
        "Case-insensitive. The filter runs on the server, so the text is not "
        "transferred unless it matches.",
    )
    p.add_argument(
        "--counts",
        choices=["key", "ip"],
        default=None,
        help="print per-key or per-IP query counts instead of rows — the quickest way to "
        "spot one caller responsible for a disproportionate share of traffic",
    )
    p.add_argument("--format", choices=["json", "jsonl", "csv"], default="jsonl")
    p.add_argument("--out", default=None, help="output file (default: exports/<timestamped>)")
    p.add_argument("--stdout", action="store_true", help="write to stdout instead of a file")
    args = p.parse_args(argv)

    key = read_admin_key()
    since = parse_since(args.since)

    if args.counts:
        body, _ = fetch(args.base, key, {"counts": args.counts, "since": since})
        doc = json.loads(body)
        if not doc.get("enabled", True):
            print("content logging is OFF on the server (start it with --log-content)")
            return 1
        rows = doc.get("counts", [])
        label = "client_ip" if args.counts == "ip" else "key_name"
        width = max([len(str(r.get(label, "-"))) for r in rows] + [len(label)])
        print(f"{label:<{width}}  {'queries':>8}  {'prompt_tok':>11}  {'out_tok':>9}  last")
        for r in rows:
            last = r.get("last_ts")
            when = (
                datetime.fromtimestamp(last, timezone.utc).strftime("%Y-%m-%d %H:%M")
                if last
                else "-"
            )
            print(
                f"{str(r.get(label, '-')):<{width}}  {r['queries']:>8}  "
                f"{r['prompt_tokens']:>11}  {r['completion_tokens']:>9}  {when}"
            )
        if not rows:
            print("(no rows in that window)")
        return 0

    body, ctype = fetch(
        args.base,
        key,
        {
            "since": since,
            "limit": args.limit,
            "key": args.key,
            "grep": args.grep,
            "format": args.format,
        },
    )

    # The JSON form wraps the rows; jsonl/csv come back ready to write.
    if args.format == "json":
        doc = json.loads(body)
        if not doc.get("enabled", True):
            print("content logging is OFF on the server (start it with --log-content)")
            return 1
        count = doc.get("count", 0)
        body = json.dumps(doc.get("rows", []), indent=1, ensure_ascii=False)
    else:
        if body.lstrip().startswith('{"enabled": false'):
            print("content logging is OFF on the server (start it with --log-content)")
            return 1
        count = len([ln for ln in body.splitlines() if ln.strip()])
        if args.format == "csv":
            count = max(0, count - 1)  # header

    if args.stdout:
        sys.stdout.write(body)
        return 0

    EXPORT_DIR.mkdir(exist_ok=True)
    if args.out:
        out = Path(args.out)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suffix = {"json": "json", "jsonl": "jsonl", "csv": "csv"}[args.format]
        out = EXPORT_DIR / f"queries-{stamp}.{suffix}"
    out.write_text(body, encoding="utf-8")
    try:
        os.chmod(out, 0o600)
    except OSError:
        pass
    print(f"{count} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
