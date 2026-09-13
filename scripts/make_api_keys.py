#!/usr/bin/env python3
"""Generate API keys for the public qwenfast endpoint into `.secrets/api_keys.json`.

    python3 scripts/make_api_keys.py alice bob carol
    python3 scripts/make_api_keys.py --rpm 120 --tpm 500000 --max-tokens 4096 public-1 public-2
    python3 scripts/make_api_keys.py --admin                 # (re)generate the admin key
    python3 scripts/make_api_keys.py --demo                  # (re)generate the Vercel demo key
    python3 scripts/make_api_keys.py --list                  # names + limits only
    python3 scripts/make_api_keys.py --show alice            # print ONE key, for handing out
    python3 scripts/make_api_keys.py --revoke bob

Keys are `qf-` + 32 URL-safe random characters from `secrets.token_urlsafe` (≈192 bits of
entropy, which is 3x what an attacker could ever brute-force against a rate-limited endpoint).

**This script never prints key material unless you ask for exactly one key with `--show`.** New
keys are written to the file and only their *names* are echoed. That is deliberate: the normal
run of this script happens in a terminal scrollback or a shared session log, and a public
endpoint's keys should not end up in either.

Output layout (`.secrets/` is gitignored):

    .secrets/api_keys.json   {"keys": [...]}   → upload to /home/.secrets/api_keys.json
    .secrets/admin_key       one line          → upload to /home/.secrets/admin_key
    .secrets/demo_key        one line          → set as QWENFAST_DEMO_KEY on Vercel
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import stat
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SECRETS_DIR = REPO_ROOT / ".secrets"
KEYS_FILE = SECRETS_DIR / "api_keys.json"
ADMIN_FILE = SECRETS_DIR / "admin_key"
DEMO_FILE = SECRETS_DIR / "demo_key"

KEY_CHARS = 32
DEFAULT_RPM = 60
DEFAULT_TPM = 200_000
DEFAULT_MAX_TOKENS = 2048


def new_key() -> str:
    # token_urlsafe returns ~1.34 chars per byte; ask for more and trim to a fixed width so every
    # key is the same length (easier to spot a truncated paste).
    body = secrets.token_urlsafe(48).replace("-", "").replace("_", "")[:KEY_CHARS]
    while len(body) < KEY_CHARS:  # pragma: no cover - astronomically unlikely
        body += secrets.token_urlsafe(16).replace("-", "").replace("_", "")
    return "qf-" + body[:KEY_CHARS]


def load() -> dict:
    if not KEYS_FILE.exists():
        return {"keys": []}
    try:
        doc = json.loads(KEYS_FILE.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{KEYS_FILE} is not valid JSON ({exc}); fix or delete it first")
    if not isinstance(doc, dict) or not isinstance(doc.get("keys"), list):
        raise SystemExit(f'{KEYS_FILE} must be an object with a "keys" array')
    return doc


def save(doc: dict) -> None:
    SECRETS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = KEYS_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n")
    os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)  # 0600
    tmp.replace(KEYS_FILE)


def write_secret_file(path: Path, value: str) -> None:
    SECRETS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(value + "\n")
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="*", help="key names to create (one key per name)")
    p.add_argument("--rpm", type=int, default=DEFAULT_RPM, help="requests per minute (0 = unlimited)")
    p.add_argument("--tpm", type=int, default=DEFAULT_TPM, help="output tokens per minute (0 = unlimited)")
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="per-request cap (0 = server default)")
    p.add_argument("--admin", action="store_true", help="(re)generate the admin key into .secrets/admin_key")
    p.add_argument("--demo", action="store_true", help="(re)generate the Vercel demo key into .secrets/demo_key")
    p.add_argument("--list", action="store_true", help="list key names and limits (no key material)")
    p.add_argument("--show", metavar="NAME", help="print the key value for ONE name (for handing it out)")
    p.add_argument("--revoke", metavar="NAME", action="append", default=[], help="remove a key by name")
    p.add_argument("--rotate", metavar="NAME", action="append", default=[], help="replace a key's value, keeping its name and limits")
    args = p.parse_args(argv)

    doc = load()
    by_name = {k["name"]: k for k in doc["keys"] if isinstance(k, dict) and "name" in k}

    if args.list:
        if not by_name:
            print("no keys yet")
            return 0
        width = max(len(n) for n in by_name)
        print(f"{'name'.ljust(width)}  rpm      tpm        max_tokens")
        for name in sorted(by_name):
            k = by_name[name]
            print(
                f"{name.ljust(width)}  {str(k.get('rpm') or '-').ljust(7)}  "
                f"{str(k.get('tpm') or '-').ljust(9)}  {k.get('max_tokens') or '-'}"
            )
        return 0

    if args.show:
        k = by_name.get(args.show)
        if not k:
            print(f"no key named {args.show!r}", file=sys.stderr)
            return 1
        print(k["key"])  # the one place key material is ever printed
        return 0

    changed = False

    for name in args.revoke:
        if name in by_name:
            doc["keys"] = [k for k in doc["keys"] if k.get("name") != name]
            by_name.pop(name)
            changed = True
            print(f"revoked: {name}")
        else:
            print(f"not found (nothing revoked): {name}", file=sys.stderr)

    for name in args.rotate:
        k = by_name.get(name)
        if not k:
            print(f"not found (nothing rotated): {name}", file=sys.stderr)
            continue
        k["key"] = new_key()
        changed = True
        print(f"rotated: {name}")

    for name in args.names:
        if name in by_name:
            print(f"exists (unchanged): {name}")
            continue
        entry = {"key": new_key(), "name": name}
        if args.rpm:
            entry["rpm"] = args.rpm
        if args.tpm:
            entry["tpm"] = args.tpm
        if args.max_tokens:
            entry["max_tokens"] = args.max_tokens
        doc["keys"].append(entry)
        by_name[name] = entry
        changed = True
        print(f"created: {name}")

    if changed:
        save(doc)
        print(f"wrote {len(doc['keys'])} keys to {KEYS_FILE.relative_to(REPO_ROOT)} (mode 0600)")

    if args.admin:
        write_secret_file(ADMIN_FILE, new_key())
        print(f"wrote admin key to {ADMIN_FILE.relative_to(REPO_ROOT)} (mode 0600)")
    if args.demo:
        write_secret_file(DEMO_FILE, new_key())
        print(f"wrote demo key to {DEMO_FILE.relative_to(REPO_ROOT)} (mode 0600)")

    if not any([changed, args.admin, args.demo]):
        p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
