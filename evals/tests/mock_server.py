#!/usr/bin/env python3
"""
Tiny aiohttp mock of an OpenAI-compatible /v1/chat/completions endpoint, used
to self-test run_eval.py end-to-end without a real GPU server.

Modes (env var MOCK_MODE, default "good"):
  good    - GSM8K: always answers "#### 42" (per the task spec). IFEval-lite:
            parses the prompt well enough to satisfy most checkers, so the
            self-test exercises both the pass and fail paths of each checker.
  nohash  - GSM8K answers omit the "####" marker, to exercise run_eval.py's
            fallback "last number in the text" extraction path.
  badjson - IFEval json_keys prompts get deliberately malformed JSON, to
            confirm the checker (and the eval gate) correctly marks it failed.

Deterministic (no randomness) so two runs against this server, diffed with
--compare-only, should show 100% exact-text agreement -- simulating a
greedy/temperature-0 spec-decoding drift check.

Usage:
    MOCK_MODE=good python mock_server.py --port 8123
"""
from __future__ import annotations

import argparse
import json
import os
import re

from aiohttp import web

MODE = os.environ.get("MOCK_MODE", "good")


def gsm8k_reply() -> str:
    if MODE == "nohash":
        return "Working through the steps, I get a final value of 42."
    return "Step 1: combine the given quantities.\n#### 42"


def ifeval_reply(prompt: str) -> str:
    m = re.search(r"exactly (\d+) bullet", prompt)
    if m:
        n = int(m.group(1))
        return "\n".join(f"- item {i + 1}" for i in range(n))

    if "entirely in uppercase" in prompt:
        return "THIS IS A FULLY UPPERCASE RESPONSE ABOUT THE TOPIC ASKED."

    if "entirely in lowercase" in prompt:
        return "this is a fully lowercase response about the topic asked."

    wm = re.search(r'word "([^"]+)" must appear at least (\w+)', prompt)
    if wm:
        word = wm.group(1)
        count_word = wm.group(2)
        count = {"twice": 2, "three": 3}.get(count_word, 2)
        # emit one extra than required-ish: exactly `count` occurrences
        sentence = " ".join([word] * count) + f". More text about {word} follows."
        return sentence

    km = re.findall(r'the keys? "([^"]+)"(?:\s+and\s+"([^"]+)")?', prompt)
    if km:
        keys = [k for pair in km for k in pair if k]
        if MODE == "badjson":
            return "{" + ", ".join(f'"{k}": "value"' for k in keys)  # missing closing brace
        obj = {k: "example value" for k in keys}
        return json.dumps(obj)

    return "A generic response."


async def chat_completions(request: web.Request) -> web.Response:
    body = await request.json()
    messages = body.get("messages", [])
    user_text = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            user_text = m.get("content", "")
            break

    if "####" in user_text or "grade-school math" in user_text.lower():
        content = gsm8k_reply()
    else:
        content = ifeval_reply(user_text)

    completion_tokens = max(1, len(content.split()))
    prompt_tokens = max(1, len(user_text.split()))

    resp = {
        "id": "mock-cmpl",
        "object": "chat.completion",
        "model": body.get("model", "mock-model"),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return web.json_response(resp)


def build_app() -> web.Application:
    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat_completions)
    return app


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8123)
    args = p.parse_args()
    web.run_app(build_app(), port=args.port, print=lambda *a: None)


if __name__ == "__main__":
    main()
