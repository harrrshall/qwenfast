#!/usr/bin/env python3
"""Minimal OpenAI-compatible mock server for exercising bench_serve.py locally.

Streams fake tokens at a fixed rate so the harness's timing/parsing code can
be verified in seconds, with no GPU and no real model. Not a fidelity target
for real engine behavior -- just enough surface to drive a self-test.

Usage:
    python mock_server.py --port 9999 --ttft-ms 50 --tok-interval-ms 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import Any

from aiohttp import web


def _make_chunk(
    endpoint: str, model: str, text: str, finish_reason: str | None, usage: dict[str, int] | None
) -> dict[str, Any]:
    choice: dict[str, Any] = {"index": 0, "finish_reason": finish_reason}
    if endpoint == "chat":
        choice["delta"] = {"content": text} if text else {}
    else:
        choice["text"] = text
    chunk: dict[str, Any] = {
        "id": "mock-0",
        "object": "chat.completion.chunk" if endpoint == "chat" else "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [choice],
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


async def _non_stream_response(
    request: web.Request, endpoint: str, body: dict[str, Any]
) -> web.Response:
    model = body.get("model", "mock")
    max_tokens = int(body.get("max_tokens", 16))
    prompt_text = _prompt_text(endpoint, body)
    prompt_tokens = max(len(prompt_text.split()), 1)

    await asyncio.sleep(request.app["ttft_ms"] / 1000.0)
    text = "".join(f"tok{i} " for i in range(max_tokens))
    choice: dict[str, Any] = {"index": 0, "finish_reason": "stop"}
    if endpoint == "chat":
        choice["message"] = {"role": "assistant", "content": text}
    else:
        choice["text"] = text
    payload = {
        "id": "mock-0",
        "object": "chat.completion" if endpoint == "chat" else "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [choice],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": max_tokens,
            "total_tokens": prompt_tokens + max_tokens,
        },
    }
    return web.json_response(payload)


def _prompt_text(endpoint: str, body: dict[str, Any]) -> str:
    if endpoint == "completions":
        return body.get("prompt", "")
    return " ".join(m.get("content", "") for m in body.get("messages", []))


async def _stream_response(request: web.Request, endpoint: str) -> web.StreamResponse | web.Response:
    body = await request.json()
    if not body.get("stream", True):
        return await _non_stream_response(request, endpoint, body)

    model = body.get("model", "mock")
    max_tokens = int(body.get("max_tokens", 16))
    want_usage = bool(body.get("stream_options", {}).get("include_usage"))
    prompt_tokens = max(len(_prompt_text(endpoint, body).split()), 1)

    ttft_s = request.app["ttft_ms"] / 1000.0
    tok_interval_s = request.app["tok_interval_ms"] / 1000.0

    resp = web.StreamResponse(
        status=200,
        headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
    )
    await resp.prepare(request)

    await asyncio.sleep(ttft_s)
    for i in range(max_tokens):
        finish = "stop" if i == max_tokens - 1 else None
        usage = (
            {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": i + 1,
                "total_tokens": prompt_tokens + i + 1,
            }
            if (finish and want_usage)
            else None
        )
        chunk = _make_chunk(endpoint, model, f"tok{i} ", finish, usage)
        await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
        if i < max_tokens - 1:
            await asyncio.sleep(tok_interval_s)

    await resp.write(b"data: [DONE]\n\n")
    await resp.write_eof()
    return resp


async def handle_completions(request: web.Request) -> web.StreamResponse | web.Response:
    return await _stream_response(request, "completions")


async def handle_chat_completions(request: web.Request) -> web.StreamResponse | web.Response:
    return await _stream_response(request, "chat")


async def handle_metrics(request: web.Request) -> web.Response:
    text = (
        "# HELP vllm:gpu_cache_usage_perc mock gauge\n"
        "# TYPE vllm:gpu_cache_usage_perc gauge\n"
        'vllm:gpu_cache_usage_perc{model="mock"} 0.42\n'
        "# HELP vllm:num_requests_waiting mock gauge\n"
        "# TYPE vllm:num_requests_waiting gauge\n"
        'vllm:num_requests_waiting{model="mock"} 0\n'
    )
    return web.Response(text=text, content_type="text/plain")


async def handle_models(request: web.Request) -> web.Response:
    return web.json_response(
        {"object": "list", "data": [{"id": "mock", "object": "model", "owned_by": "mock"}]}
    )


async def handle_health(request: web.Request) -> web.Response:
    return web.Response(text="ok")


def build_app(ttft_ms: float, tok_interval_ms: float) -> web.Application:
    app = web.Application()
    app["ttft_ms"] = ttft_ms
    app["tok_interval_ms"] = tok_interval_ms
    app.router.add_post("/v1/completions", handle_completions)
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    app.router.add_get("/metrics", handle_metrics)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_get("/health", handle_health)
    return app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mock OpenAI-compatible streaming server.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9999)
    p.add_argument("--ttft-ms", type=float, default=30.0, help="fake time-to-first-token")
    p.add_argument("--tok-interval-ms", type=float, default=8.0, help="fake inter-token gap")
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    app = build_app(args.ttft_ms, args.tok_interval_ms)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
