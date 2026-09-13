"""Programmatic checkers for the IFEval-lite prompt set (evals/data/ifeval_50.jsonl).

Each checker takes (response_text, checker_args) and returns True/False.
`response_text` should already have any <think>...</think> block stripped
by the caller (run_eval.py does this before dispatching).

Kept deliberately dependency-free (stdlib only) so this module can be
imported standalone by tests without pulling in aiohttp.
"""
from __future__ import annotations

import json
import re


def bullet_count(text: str, args: dict) -> bool:
    """Exactly `n` non-empty lines, each starting with `marker` (default '- ')."""
    n = args["n"]
    marker = args.get("marker", "- ")
    lines = [l for l in text.strip().splitlines() if l.strip() != ""]
    if len(lines) != n:
        return False
    prefix = marker.strip()
    return all(l.lstrip().startswith(prefix) for l in lines)


def all_caps(text: str, args: dict) -> bool:
    """Response contains no lowercase alphabetic characters (and >=1 letter)."""
    stripped = text.strip()
    if not any(c.isalpha() for c in stripped):
        return False
    return not any(c.islower() for c in stripped)


def all_lower(text: str, args: dict) -> bool:
    """Response contains no uppercase alphabetic characters (and >=1 letter)."""
    stripped = text.strip()
    if not any(c.isalpha() for c in stripped):
        return False
    return not any(c.isupper() for c in stripped)


def word_count_at_least(text: str, args: dict) -> bool:
    """The given word (whole-word, case-insensitive) appears >= count times."""
    word = args["word"]
    count = args["count"]
    hits = re.findall(r"\b" + re.escape(word) + r"\b", text, flags=re.IGNORECASE)
    return len(hits) >= count


def json_keys(text: str, args: dict) -> bool:
    """Response is valid JSON, a top-level object, with exactly the given keys."""
    keys = set(args["keys"])
    body = text.strip()
    # tolerate a ```json ... ``` fenced block
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", body, flags=re.DOTALL)
    if fence:
        body = fence.group(1).strip()
    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return False
    if not isinstance(parsed, dict):
        return False
    return set(parsed.keys()) == keys


CHECKERS = {
    "bullet_count": bullet_count,
    "all_caps": all_caps,
    "all_lower": all_lower,
    "word_count_at_least": word_count_at_least,
    "json_keys": json_keys,
}
