# `qwenfast.server`

the openai-compatible http server for `qwenfast`. it turns `/v1/chat/completions` and `/v1/completions` requests into token ids, hands them to an `AsyncEngine`, and streams the result back as sse chunks with incremental detokenization, `<think>` splitting, stop-string matching and hermes tool-call parsing. the whole server runs without a gpu against `mock_engine.py`, which is how its test suite works.

the http surface is documented in [docs/api.md](../../../docs/api.md). this file covers the code.

## file map

| file | role |
| --- | --- |
| `app.py` | fastapi app: `create_app(engine, tokenizer, ...)`, every route, the chat and completion streaming loops, admission, clamping, metering |
| `engine_api.py` | the `AsyncEngine` interface plus `SamplingParams`, `StepOutput`, `RequestStats`, `EngineStats`, `Histogram` |
| `protocol.py` | pydantic v2 request schemas: `ChatCompletionRequest`, `CompletionRequest`, `ChatTemplateKwargs` |
| `tokenization.py` | `IncrementalDetokenizer`, `StopStringMatcher`, `ThinkTagParser`, `ToolCallStreamParser`, `render_chat_prompt` |
| `auth.py` | `ApiKey`, `KeyStore` (hot-reloaded keys file), `RateLimiter` (per-key rpm and tpm), `ConcurrencyLimiter` (per-key and per-ip streams) |
| `usage.py` | `UsageRecorder`: in-memory totals, sqlite audit log, `/admin/usage` summary, extra `/metrics` counters |
| `content_log.py` | `QueryLogger`: opt-in sqlite log of prompts and replies behind `/admin/queries` |
| `metrics.py` | `render_prometheus_text(EngineStats)` for `/metrics` |
| `errors.py` | `ApiError` and the handlers that render every failure as the openai `{"error": {...}}` envelope |
| `config.py` | `add_public_api_args` and `public_api_kwargs`: the public-endpoint flags shared by both entry points |
| `cli.py`, `__main__.py` | `python -m qwenfast.server` |
| `mock_engine.py` | `MockEngine`: a paced fake engine, no gpu |
| `fake_tokenizer.py` | `FakeTokenizer`: offline lossless fallback tokenizer for the tests |
| `tests/` | cpu test suite, see below |

## the `AsyncEngine` contract

`engine_api.py` is the only boundary between the server and the model runtime. it knows nothing about http, tokenizers or the openai wire format.

```python
class AsyncEngine(abc.ABC):
    def add_request(self, request_id: str, prompt_token_ids: list[int],
                    sampling_params: SamplingParams) -> AsyncIterator[StepOutput]: ...
    async def abort(self, request_id: str) -> None: ...
    def get_stats(self) -> EngineStats: ...
    def health(self) -> Optional[str]: ...        # optional, default None (healthy)
    async def start(self) -> None: ...            # optional lifecycle hook
    async def shutdown(self) -> None: ...         # optional lifecycle hook
```

- `add_request` is normally an async generator. each `StepOutput` carries `new_token_ids` (one or more; speculative decoding commits several per step, and the server never assumes exactly one), `finished`, `finish_reason` (`"stop"`, `"length"`, `"abort"` or `None`), a `RequestStats` snapshot (`prompt_tokens`, `completion_tokens`, `ttft_s`, `spec_accept_length`) and optional per-token `logprobs`. the generator ends after the yield with `finished=True`.
- `abort` is idempotent and best-effort. the matching `add_request` iterator must terminate shortly after with `finish_reason="abort"`.
- `get_stats` is synchronous and cheap; it runs on every `/metrics` scrape and `/status` call.
- `health` returns `None` when the engine can serve, otherwise a one-line reason. `/health` reports it and every `/v1/*` route turns a reason into a 503 with `code: service_unavailable`.
- `SamplingParams` rejects `n != 1` and negative `max_tokens` in `__post_init__`. stop strings are threaded through as `stop` even though matching happens in the server on detokenized text.
- `EngineStats` is the `/metrics` snapshot: running and waiting counts, cumulative token counts, ttft and tpot `HistogramSnapshot`s, kv and ssm slot usage, optional speculative-decoding acceptance figures, uptime.

one invariant matters when writing a new engine: `_generate_chat` and `_generate_completion` in `app.py` always drain the generator with a plain `async for` to its natural end, even after they decide to abort. cleanup in a `finally:` inside the generator (releasing an admission slot, discarding an abort flag) therefore runs deterministically without `aclose()` handling. `MockEngine` relies on this and `test_abort_on_disconnect_stops_generation` pins it.

## the chat pipeline

`app.py::_generate_chat` builds four state machines per request and feeds every engine step through them in this order:

```
token ids --[IncrementalDetokenizer]--> raw text deltas
          --[StopStringMatcher]--> text before any requested stop string
          --[ThinkTagParser]--> (reasoning_content deltas, content deltas)
          --[ToolCallStreamParser]--> (content deltas, ToolCallDelta events)
```

every stage holds back only what it must so that text already sent to the client is never retracted: a partial tag or stop-string suffix, or an incomplete utf-8 sequence.

- `IncrementalDetokenizer` uses the prefix-offset technique: decode `window + new` and `window`, the delta is the new suffix, held back entirely while it ends in the utf-8 replacement character. `add_tokens_split` returns one delta per token for a multi-token step so a stream emits one chunk per token even when speculative decoding commits several tokens at once (`tests/test_spec_chunking.py`). decoding runs in a small `ThreadPoolExecutor` sized by `--detok-workers`; `--detok-workers 0` runs it inline on the event loop.
- `StopStringMatcher.feed(text)` returns `(forward, stopped)`. on a match the server aborts the engine request, drops the rest of the step and finishes with `finish_reason: "stop"`. matching runs on the raw text before think parsing, so a stop string can fire inside a think block.
- `ThinkTagParser` splits around `<think>` and `</think>`. the chat template appends `<think>\n` to the prompt when thinking is enabled, so the generated text usually contains only the closing tag; the parser therefore starts inside a think block (`start_in_think=enable_thinking`) and still recognises an explicit opening tag in-stream. the two newlines the template emits after `</think>` are stripped from the first content delta.
- `ToolCallStreamParser` parses hermes-style `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` blocks out of the content stream. `function.arguments` is streamed as raw json text fragments, so concatenating every fragment for one call index yields the exact json the model produced. a stream that ends inside a tool call (`max_tokens` reached) is finalised with whatever arrived. when any tool call was seen the response finishes with `finish_reason: "tool_calls"`.
- `render_chat_prompt` applies the tokenizer's own chat template with `enable_thinking`, `reasoning_effort`, `preserve_thinking` and `tools` forwarded from the request.

`/v1/completions` runs only the detokenizer and the stop matcher; it returns raw `text`.

sampling defaults come from the model card. `THINKING_DEFAULTS` (temperature 1.0, top_p 0.95, top_k 20) apply when `chat_template_kwargs.enable_thinking` is true, `NON_THINKING_DEFAULTS` (temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5) otherwise and for `/v1/completions`. any field the request sets wins.

request flow for `/v1/chat/completions`, in order: drain check, engine health, per-key rate limit, `--max-messages`, chat template, sampling params, `--max-prompt-tokens`, per-key and `--max-output-tokens` clamp, context clamp (`fit_to_context`: `max_tokens` is trimmed to `context - prompt`, a 400 only when less than `--min-completion-tokens` remain), then admission (`--max-inflight-requests`, per-key and per-ip stream caps). the resolved `max_tokens` is returned in the `X-Qwenfast-Max-Tokens` header. an `X-Request-Id` request header becomes the engine-facing request id, which is how tests drive `MockEngine.script`.

## running against the mock engine

`--engine mock` needs a tokenizer and nothing else. download just the tokenizer files:

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.8-27B', allow_patterns=['tokenizer*','*.json','*.jinja'], local_dir='/path/to/tok')
"
PYTHONPATH=engine python -m qwenfast.server --engine mock --model /path/to/tok --port 8000 --api-key mykey
```

```bash
curl -s http://localhost:8000/v1/chat/completions -H "Authorization: Bearer mykey" \
  -H 'Content-Type: application/json' -d '{
    "model": "/path/to/tok", "messages": [{"role":"user","content":"hi"}],
    "max_tokens": 16, "chat_template_kwargs": {"enable_thinking": false}
  }'
```

`MockEngine` paces output with `--mock-decode-tokens-per-second` (default 200), models prefill latency with `--mock-prefill-base-delay-s` (0.005) plus `--mock-prefill-tokens-per-second` (100000), and bounds admission with `--mock-max-concurrent-requests` (256), so running and waiting counts show up in `/metrics`. without a script it repeats a filler phrase up to `max_tokens`; `engine.script(request_id, text=...)` replays an exact string for one request id, which is what the tests use.

`--engine qwenfast` builds the real gpu runtime through `qwenfast.runtime.serve.build_engine_from_args` and installs the runtime flags into the same parser. `python -m qwenfast.runtime.serve` is the equivalent entry point on the runtime side; both call the same `create_app`. see the [runtime readme](../runtime/README.md).

general flags: `--model` (tokenizer directory or repo id; also the weights directory for the real engine), `--served-model-name`, `--host` (default `0.0.0.0`), `--port` (8000), `--api-key`, `--default-max-tokens` (512), `--detok-workers` (4), `--log-level`. the public-endpoint flags live in `config.py` and are listed in the next section.

## auth and per-key limits

`KeyStore` accepts keys from four places, all merged into one table: a single `--api-key` (named `static`), `--demo-key` (named `demo`, metered separately), `--admin-key` or `--admin-key-file` (named `admin`, may read `/admin/*`, never concurrency-capped), and a `--api-keys-file`:

```json
{
  "keys": [
    {"key": "qf-...", "name": "alice", "rpm": 60, "tpm": 200000, "max_tokens": 2048, "max_streams": 4},
    {"key": "qf-...", "name": "ops", "admin": true},
    {"key": "qf-...", "name": "old", "disabled": true}
  ]
}
```

| field | meaning |
| --- | --- |
| `key` | the secret sent as `Authorization: Bearer <key>`; required, unique |
| `name` | label used in logs, `/admin/usage` and error messages; required, unique |
| `rpm` | requests per sliding 60 s window; omit or 0 for unlimited |
| `tpm` | completion tokens per sliding 60 s window; omit or 0 for unlimited |
| `max_tokens` | per-request completion cap, applied silently |
| `max_streams` | concurrent in-flight requests for this key; overrides `--max-streams-per-key` |
| `admin` | may read `/admin/usage` and `/admin/queries`; exempt from stream caps |
| `disabled` | key is refused with 401 |

the file is re-read on `SIGHUP` and when its mtime changes (polled at most every 2 s, on request). a file that fails to parse is ignored: the previous key set keeps serving and the error is reported on `/admin/usage`. key material never appears in logs or responses. when no key source is configured every `/v1` route is open and the caller is `anonymous`. `/health`, `/status` and `/metrics` never require a key.

`RateLimiter` admits on request count and on tokens already recorded in the window, then records the actual completion tokens when the stream ends, so a stream is never cut mid-flight; the next request is the one refused. a refusal is a 429 with `Retry-After` and `code: rate_limit_exceeded`. `ConcurrencyLimiter` enforces `--max-streams-per-key` and `--max-streams-per-ip` (ip from `--client-ip-header`, default `x-forwarded-for`, falling back to the socket peer), also as a 429. `--max-inflight-requests` is the global admission cap and answers 503.

server-wide flags from `config.py`:

| flag | default | effect |
| --- | --- | --- |
| `--api-keys-file` | none | json keys file, hot-reloaded |
| `--admin-key`, `--admin-key-file` | none | admin key inline or from a file |
| `--demo-key`, `--demo-key-file` | none | key for the demo proxy, metered as `demo` |
| `--usage-db` | none | sqlite file for the usage log; totals survive restarts |
| `--max-inflight-requests` | 0 | global concurrent `/v1` requests before 503 (0 = unlimited) |
| `--max-output-tokens` | 0 | server-wide `max_tokens` ceiling, clamped silently |
| `--max-prompt-tokens` | 0 | 400 above this many prompt tokens |
| `--max-request-bytes` | 1000000 | 413 above this content length |
| `--max-messages` | 256 | 400 above this many chat messages |
| `--min-completion-tokens` | 0 | 400 only when the prompt leaves less room than this |
| `--request-timeout` | 0 | seconds before an in-flight request is cancelled (504 when not streaming) |
| `--max-streams-per-key`, `--max-streams-per-ip` | 0 | concurrent streams before 429 |
| `--client-ip-header` | `x-forwarded-for` | header the per-ip cap reads |
| `--drain-timeout` | 30 | on `SIGTERM`, stop accepting and give in-flight streams this long |
| `--log-content` | off | record prompts and replies to `--content-log-db` |
| `--content-log-db` | `queries.sqlite` beside `--usage-db` | sqlite file for the content log |
| `--content-retention-days` | 0 | prune content rows older than this |

`GET /v1/limits` returns the resulting envelope for the calling key, and `/status` returns the deployment-wide part without a key.

## errors

`errors.py` renders every failure, including fastapi validation errors and unmatched routes, as `{"error": {"message", "type", "param", "code"}}`. validation failures are a 400 with `code: invalid_request_body`. `ApiError` also carries a private `error_class` from `usage.ERROR_CLASSES` so `/admin/usage` can break failures down by reason. the full list of `error.code` values is in [docs/api.md](../../../docs/api.md).

## metrics

`GET /metrics` renders `EngineStats` in prometheus text format under the `qwenfast:` namespace: `num_requests_running`, `num_requests_waiting`, `decode_batch_size`, `ssm_slots_used|total`, `kv_pages_used|total`, `prompt_tokens_total`, `generation_tokens_total`, `tokens_per_second`, the `time_to_first_token_seconds` and `time_per_output_token_seconds` histograms, `spec_accept_length` and `spec_acceptance_rate` (only once an engine reports them; `MockEngine` does not), and `uptime_seconds`. the vllm-style aliases `num_requests_running`, `num_requests_waiting` and `gpu_cache_usage_perc` are emitted too so a generic scraper such as `benchmarks/bench_serve.py` picks them up.

`UsageRecorder.prometheus_lines` appends the api counters: `qwenfast:api_requests_total`, `api_errors_total`, `api_prompt_tokens_total`, `api_completion_tokens_total`, `api_rate_limited_total`, `api_overload_total`, `api_concurrency_limited_total`, `api_timeouts_total`, `api_auth_failures_total`, `api_dropped_audit_rows_total`, `api_errors_by_class_total{class=...}`, plus the gauges `api_inflight_requests`, `api_inflight_capacity`, `api_keys_loaded` and `api_draining`.

metering never runs on the streaming hot path: the handler builds one `UsageRecord` after the response finishes and `UsageRecorder.record` updates in-memory totals under a short lock and queues the row for a single writer thread. a full queue drops the row and counts the drop.

## tests

```bash
PYTHONPATH=engine python -m pytest engine/qwenfast/server/tests
```

everything runs on cpu against `MockEngine`. `tests/conftest.py` resolves the tokenizer in this order: `QWENFAST_TOKENIZER_DIR`, a local snapshot, `Qwen/Qwen3.8-27B` from the hub, then `FakeTokenizer` (fully offline, lossless, no `transformers` needed). `pytest.ini` sets `asyncio_mode = auto`.

| file | covers |
| --- | --- |
| `test_server.py` | sse chunk format for chat and completions, usage accounting with and without `stream_options.include_usage`, think parsing split down to one character per feed, tool-call streaming (single, multiple, nested braces and quotes), incremental detokenization against batch decode, stop strings with partial-suffix hold-back, abort on disconnect, 64 concurrent streams draining back to zero running, `/metrics` content, `--api-key` gating |
| `test_public_api.py` | 401 on missing or bad keys, open endpoints when no auth is configured, admin gating, rpm and tpm windows with `Retry-After`, per-key and server-wide `max_tokens` clamps, prompt and message caps, 413 bodies, 503 on the queue cap, sqlite usage rows and restart hydration, keys-file hot reload and `SIGHUP`, keys document validation, separate demo-key metering |
| `test_reliability.py` | context clamp and refusal cases, the error envelope on every refusal, 400 for malformed bodies, per-key and per-ip stream caps, request timeouts for streaming and non-streaming, disconnect freeing the admission slot, draining, `/status`, `/v1/limits`, `/v1/models` context window, `/admin/usage` error breakdown, usage db migration, content logging and `/admin/queries` |
| `test_spec_chunking.py` | one sse chunk per token when an engine yields several tokens per step, byte-identical text across step sizes |
| `test_supervisor_lock.py` | `scripts/supervisor_lock.sh` through `sh`: single-holder lock, stale-lock reclaim, racing supervisors |

## known simplifications

- `n` must be 1; `logprobs` is accepted and threaded through but `MockEngine` never populates it.
- `echo` on `/v1/completions` and `tool_choice: "required"` are accepted by the schema and not enforced.
- non-text content parts in chat messages are dropped; the server is text-only.
