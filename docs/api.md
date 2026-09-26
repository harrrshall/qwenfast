# http api reference

the `qwenfast` server speaks the openai chat and completions wire format. any openai sdk works against it by changing `base_url` and `api_key`. this page documents every endpoint, header, request field and error code the server implements. the implementation lives in [engine/qwenfast/server/app.py](../engine/qwenfast/server/app.py) and [engine/qwenfast/server/protocol.py](../engine/qwenfast/server/protocol.py).

the examples below assume a server on `http://localhost:8000` started with `python -m qwenfast.runtime.serve` (see the [runtime readme](../engine/qwenfast/runtime/README.md)) and a model name of `qwen3.8-27b` (`--served-model-name`).

## quick start

### curl

```bash
export QWENFAST_KEY=sk-...
export QWENFAST_URL=http://localhost:8000/v1

curl -s $QWENFAST_URL/chat/completions \
  -H "Authorization: Bearer $QWENFAST_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "explain speculative decoding in two sentences."}],
    "max_tokens": 256,
    "temperature": 0
  }' | jq -r '.choices[0].message.content'
```

streaming, with `stream_options.include_usage` so the last frame carries exact token counts:

```bash
curl -N $QWENFAST_URL/chat/completions \
  -H "Authorization: Bearer $QWENFAST_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.8-27b", "stream": true, "stream_options": {"include_usage": true},
       "messages": [{"role": "user", "content": "count to twenty."}]}'
```

### openai python sdk

```python
from openai import OpenAI

client = OpenAI(api_key="sk-...", base_url="http://localhost:8000/v1")

# non-streaming
r = client.chat.completions.create(
    model="qwen3.8-27b",
    messages=[{"role": "user", "content": "write a haiku about gpus."}],
    max_tokens=128,
    temperature=0,
)
print(r.choices[0].message.content)
print(r.usage)

# streaming
stream = client.chat.completions.create(
    model="qwen3.8-27b",
    messages=[{"role": "user", "content": "explain paged kv cache."}],
    max_tokens=512,
    stream=True,
    stream_options={"include_usage": True},
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## authentication

every `/v1/*` route requires `Authorization: Bearer <key>`. the scheme is matched case-insensitively; a bare token with no scheme is also accepted. a missing, unknown or disabled key is a `401` with `error.code = "invalid_api_key"`.

keys come from one of two server flags:

| flag | effect |
|---|---|
| `--api-key KEY` | a single shared key |
| `--api-keys-file PATH` | a json file of named keys with per-key limits, hot-reloaded on mtime change or `SIGHUP` |

with neither flag the endpoint is open and every caller is metered as `anonymous`. the keys file format is described in the [server readme](../engine/qwenfast/server/README.md). key material never appears in logs, metrics or response bodies; every user-visible reference uses the key's `name`.

`/health`, `/status` and `/metrics` need no key. `/admin/*` needs a key marked as admin (`--admin-key`, `--admin-key-file`, or `"admin": true` in the keys file); a valid non-admin key gets `403` with `error.code = "permission_denied"`.

## endpoints

| method | path | auth | purpose |
|---|---|---|---|
| `GET` | `/v1/models` | key | the one served model plus its context envelope |
| `POST` | `/v1/chat/completions` | key | chat, streaming and non-streaming, tools, thinking |
| `POST` | `/v1/completions` | key | raw text completion, streaming and non-streaming |
| `GET` | `/v1/limits` | key | the deployment limits and the caller's per-key limits |
| `GET` | `/status` | none | health, load, capacity, limits, uptime |
| `GET` | `/health` | none | `200` when serving, `503` when unhealthy or draining |
| `GET` | `/metrics` | none | prometheus text exposition |
| `GET` | `/admin/usage` | admin | totals, per-key breakdown, hourly series, engine view |
| `GET` | `/admin/queries` | admin | export of the optional content log |

### `GET /v1/models`

```json
{
  "object": "list",
  "data": [{
    "id": "qwen3.8-27b",
    "object": "model",
    "created": 1700000000,
    "owned_by": "qwenfast",
    "context_window": 4096,
    "max_output_tokens": 4096,
    "max_prompt_tokens": null
  }]
}
```

`context_window`, `max_output_tokens` and `max_prompt_tokens` are additive fields an openai sdk ignores. `context_window` is the engine's `--max-model-len`.

### `POST /v1/chat/completions`

request fields (all optional except `model` and `messages`):

| field | type | notes |
|---|---|---|
| `model` | string | any value is accepted; the response always reports the served name |
| `messages` | array | `role`, `content` (string or a list of `{"type": "text", "text": ...}` parts), `name`, `tool_calls`, `tool_call_id` |
| `max_tokens` | int | default `--default-max-tokens`; clamped, see below |
| `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `repetition_penalty` | number | defaults depend on thinking mode, see below |
| `stop` | string or array | stop strings, matched on the detokenized text |
| `seed` | int | deterministic sampling |
| `stream` | bool | server-sent events |
| `stream_options` | object | `{"include_usage": true}` adds a final usage frame |
| `tools`, `tool_choice` | array, string or object | function calling, see [tool calls](#tool-calls) |
| `chat_template_kwargs` | object | `enable_thinking`, `reasoning_effort`, `preserve_thinking` |
| `reasoning_effort` | `"xhigh"`, `"medium"`, `"low"` | top-level alias of `chat_template_kwargs.reasoning_effort` |
| `logprobs`, `top_logprobs` | bool or int | accepted and passed to the engine; the response body does not carry logprobs |
| `ignore_eos` | bool | keep generating past end of sequence (benchmarking) |
| `n` | int | must be `1` |

non-streaming response:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "qwen3.8-27b",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "..."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 20, "completion_tokens": 64, "total_tokens": 84}
}
```

`finish_reason` is `stop`, `length` or `tool_calls`.

sampling defaults come from the model card's two profiles and depend on whether thinking is on:

| parameter | thinking | non-thinking |
|---|---|---|
| `temperature` | 1.0 | 0.7 |
| `top_p` | 0.95 | 0.80 |
| `top_k` | 20 | 20 |
| `min_p` | 0.0 | 0.0 |
| `presence_penalty` | 0.0 | 1.5 |
| `repetition_penalty` | 1.0 | 1.0 |

any field the request sets overrides the profile value.

### `POST /v1/completions`

`prompt` is a string, a list of token ids, or a single-element list of strings (a longer list of strings is a `400` with `error.code = "unsupported_parameter"`). the sampling fields, `stop`, `seed`, `stream`, `stream_options`, `logprobs` and `n` behave as for chat; the non-thinking profile supplies the defaults. `echo` and `suffix` are accepted and ignored.

```json
{
  "id": "cmpl-...",
  "object": "text_completion",
  "created": 1700000000,
  "model": "qwen3.8-27b",
  "choices": [{"index": 0, "text": "...", "finish_reason": "stop", "logprobs": null}],
  "usage": {"prompt_tokens": 8, "completion_tokens": 32, "total_tokens": 40}
}
```

### `GET /v1/limits`

everything a client needs to stay inside the envelope, including the calling key's own limits. values that are not configured are `null`.

```json
{
  "object": "limits",
  "model": "qwen3.8-27b",
  "max_context_length": 4096,
  "max_prompt_tokens": 3840,
  "default_max_tokens": 1024,
  "max_output_tokens": 4096,
  "min_completion_tokens": 256,
  "max_messages": 128,
  "max_request_bytes": 1000000,
  "max_concurrent_streams_per_key": 8,
  "max_concurrent_streams_per_ip": 8,
  "request_timeout_s": 300,
  "max_tokens_is_clamped": true,
  "key": {"name": "alice", "max_tokens": 4096, "rpm": 1000, "tpm": 1000000, "max_concurrent_streams": 8}
}
```

### `GET /status`

keyless. carries no usage, cost or key names.

```json
{
  "status": "ok",
  "healthy": true,
  "detail": null,
  "model": "qwen3.8-27b",
  "load": {"in_flight": 3, "capacity": 256, "running": 3, "waiting": 0,
           "kv_pages_used": 1200, "kv_pages_total": 32961},
  "limits": {"...": "same block as /v1/limits without the key section"},
  "uptime_s": 3600.0,
  "process_uptime_s": 3600.0,
  "draining": false
}
```

`status` is `ok`, `draining` or `unhealthy`; the http status is `200` when healthy and `503` otherwise.

### `GET /health`

`{"status": "ok"}` with `200`, or `{"status": "unhealthy", "detail": "..."}` / `{"status": "draining", "detail": "shutting down"}` with `503`. the probe asks the engine directly, so a dead device thread reports unhealthy.

### `GET /metrics`

prometheus text format. the engine's serving metrics plus the api counters:

`qwenfast:api_requests_total`, `qwenfast:api_errors_total`, `qwenfast:api_prompt_tokens_total`, `qwenfast:api_completion_tokens_total`, `qwenfast:api_rate_limited_total`, `qwenfast:api_overload_total`, `qwenfast:api_concurrency_limited_total`, `qwenfast:api_timeouts_total`, `qwenfast:api_auth_failures_total`, `qwenfast:api_dropped_audit_rows_total`, `qwenfast:api_errors_by_class_total{class="..."}`, and the gauges `qwenfast:api_inflight_requests`, `qwenfast:api_inflight_capacity`, `qwenfast:api_keys_loaded`, `qwenfast:api_draining`.

### `GET /admin/usage`

admin key only. `?hours=N` (1 to 168, default 24) selects the hourly window. returns the usage totals and per-key breakdown, the hourly series, `engine` (running, waiting, lifetime token counts, per-stream tokens per second, mean ttft, `spec_accept_length`, kv pages), `engine_health`, `capacity` (in-flight, peak, per-key and per-ip concurrency snapshot, `queue_rejects`), `limits`, `drain`, `content_log`, `keys` (redacted: names and limits only) and `key_store` (path, load time, reload count, last parse error).

### `GET /admin/queries`

admin key only. exports the content log written when the server runs with `--log-content`; without it the answer is `{"enabled": false, ...}`. query parameters: `since` (unix seconds) or `hours`, `limit` (1 to 10000, default 100), `key` (key name), `grep` (regex over prompt and reply; an invalid pattern is a `400` with `error.code = "invalid_grep"`), `counts=key|ip` for a grouped view, `format=json|jsonl|csv`.

## streaming

set `"stream": true`. the response is `text/event-stream`; every frame is `data: <json>\n\n` and the stream ends with `data: [DONE]`.

chat frames are `chat.completion.chunk` objects. the first delta carries `"role": "assistant"`; later deltas carry `content`, `reasoning_content` or `tool_calls`. speculative decoding can commit several tokens per engine step; the server still emits one delta per token, so chunk counting matches the token count. the terminal frame has an empty delta and a non-null `finish_reason`. with `stream_options.include_usage` one more frame follows with `"choices": []` and a `usage` object.

completion frames are `text_completion` objects with a `text` field, same terminal and usage frames.

the response headers are `Cache-Control: no-cache`, `X-Accel-Buffering: no` and `X-Qwenfast-Max-Tokens`.

a streaming request that hits `--request-timeout` is closed in band: a final chunk with `finish_reason: "length"`, the usage frame if requested, then `[DONE]`. a non-streaming request in the same situation gets a `504`.

## thinking mode and `reasoning_content`

thinking is off by default. turn it on per request:

```json
{
  "model": "qwen3.8-27b",
  "messages": [{"role": "user", "content": "what is 17 * 23?"}],
  "chat_template_kwargs": {"enable_thinking": true}
}
```

the server parses the model's think tags and returns the reasoning in `message.reasoning_content` (non-streaming) or `delta.reasoning_content` (streaming). `content` holds only the final answer. `reasoning_effort` (`xhigh`, `medium`, `low`) and `preserve_thinking` are passed to the chat template; `reasoning_effort` may also be sent at the top level of the request.

turning thinking on switches the sampling defaults to the thinking profile.

## tool calls

pass `tools` in the openai function schema. the qwen3.8 chat template asks the model for qwen3 coder xml calls (`<tool_call><function=name><parameter=key>value</parameter></function></tool_call>`), and the server parses them back into openai tool calls, converting each parameter to the json type its schema declares. hermes json calls (`<tool_call>{"name": ..., "arguments": {...}}</tool_call>`) are accepted too. tool call arguments sent back in `messages` may be the usual json string; the server turns them into the object the template needs, maps a `developer` role to `system`, and folds later system messages into the first one.

```json
{
  "model": "qwen3.8-27b",
  "messages": [{"role": "user", "content": "what is the weather in paris?"}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "current weather for a city",
      "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    }
  }]
}
```

a response that calls a tool has `finish_reason: "tool_calls"` and

```json
"message": {
  "role": "assistant",
  "content": null,
  "tool_calls": [{
    "id": "call_0",
    "type": "function",
    "function": {"name": "get_weather", "arguments": "{\"city\": \"paris\"}"}
  }]
}
```

when streaming, `delta.tool_calls` entries carry `index`, `type`, `function.name` on the first delta and `function.arguments` fragments afterwards, the usual openai accumulation pattern. send the result back as a `{"role": "tool", "tool_call_id": ..., "content": ...}` message.

## `max_tokens` clamping

`max_tokens` is a ceiling. the server resolves it in this order:

1. the request value, or `--default-max-tokens` when absent.
2. the key's `max_tokens` from the keys file and the server-wide `--max-output-tokens`, whichever is smaller.
3. the room the prompt leaves in the context (`max_context_length - prompt_tokens`).

the resolved value is returned in the `X-Qwenfast-Max-Tokens` response header on every chat and completion response, streaming or not. a reply that runs to the ceiling ends with `finish_reason: "length"`.

a request is refused with `400` `context_length_exceeded` only when the prompt by itself leaves less than `--min-completion-tokens` of room, or when `--max-prompt-tokens` is set and the prompt exceeds it. `GET /v1/limits` states the numbers up front; `max_tokens_is_clamped: true` advertises this behaviour.

## rate limits and fairness

per key, from the keys file:

| field | meaning |
|---|---|
| `rpm` | requests admitted per rolling 60 s window |
| `tpm` | completion tokens per rolling 60 s window |
| `max_tokens` | per-request completion ceiling, applied silently |
| `max_streams` | concurrent in-flight requests; overrides `--max-streams-per-key` |

the token window is enforced after the fact: tokens are known when a response ends, so a key that has just exhausted its `tpm` is refused on its next request. a stream is never cut in the middle.

server-wide:

| flag | default | effect |
|---|---|---|
| `--default-max-tokens` | 512 | `max_tokens` when the request omits it |
| `--max-output-tokens` | 0 (off) | ceiling on `max_tokens` for every key |
| `--min-completion-tokens` | 0 | a prompt is refused only if it leaves less than this much room |
| `--max-prompt-tokens` | 0 (off) | hard reject above this prompt length |
| `--max-messages` | 256 | `400` above this many chat messages |
| `--max-request-bytes` | 1000000 | `413` above this `Content-Length` |
| `--max-inflight-requests` | 0 (unlimited) | global admission cap; beyond it `503` with `Retry-After: 5` |
| `--max-streams-per-key` | 0 (unlimited) | concurrent requests one key may hold; beyond it `429` |
| `--max-streams-per-ip` | 0 (unlimited) | same per client address, read from `--client-ip-header` (default `x-forwarded-for`) |
| `--request-timeout` | 0 (none) | seconds before an in-flight request is cancelled |

admin keys are never subject to the concurrency caps. every `429` and `503` carries a `Retry-After` header in seconds.

## errors

every refusal, including schema and json failures, uses the openai envelope. all four fields are always present:

```json
{
  "error": {
    "message": "this model's maximum context length is 4096 tokens. ...",
    "type": "invalid_request_error",
    "param": "messages",
    "code": "context_length_exceeded"
  }
}
```

`type` is `invalid_request_error`, `rate_limit_error` or `server_error`. `param` names the offending field when there is one (for schema failures it is the dotted json path, for example `messages.3.role`). branch on `code`:

| http | `error.code` | meaning |
|---|---|---|
| 400 | `context_length_exceeded` | the prompt leaves too little room to answer, or exceeds `--max-prompt-tokens` |
| 400 | `too_many_messages` | more than `--max-messages` chat messages |
| 400 | `invalid_request_body` | malformed json or a schema violation |
| 400 | `invalid_chat_template` | the chat template rejected the messages or tools |
| 400 | `unsupported_parameter` | a batched string `prompt` on `/v1/completions` |
| 400 | `invalid_grep` | bad regex on `/admin/queries` |
| 401 | `invalid_api_key` | missing, unknown or disabled key |
| 403 | `permission_denied` | valid key, admin route |
| 404 | `not_found` | unknown route |
| 405 | `method_not_allowed` | wrong http method |
| 413 | `payload_too_large` | body over `--max-request-bytes` |
| 429 | `rate_limit_exceeded` | the key's `rpm` or `tpm` window is full |
| 429 | `concurrency_limit_exceeded` | the key or client address already holds its maximum number of streams |
| 500 | `internal_error` | unhandled server failure |
| 503 | `server_overloaded` | the global admission cap is full |
| 503 | `server_draining` | the process is shutting down; `Retry-After: 10` |
| 503 | `service_unavailable` | the engine is not healthy |
| 504 | `request_timeout` | a non-streaming request exceeded `--request-timeout` |

## speed note

speculative decoding (the model's mtp head) engages only when every request in a decode batch is greedy. send `"temperature": 0` to get it. a sampled request (any `temperature > 0`) is served by the plain decode path for the steps it shares with the batch, which is slower per token.

## not supported

`n > 1`, embeddings, images and other non-text content parts, the batch api, and logprobs in the response body.
