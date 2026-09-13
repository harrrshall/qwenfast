# qwenfast demo

a streaming chat page for the `qwenfast` engine serving `qwen3.8-27b`. it is a next.js 15 app router app with two jobs:

- a chat ui at `/` that streams tokens from the engine and shows per-response and engine-side speed readouts.
- a transparent openai-compatible proxy under `/v1/*`, so an openai sdk pointed at the deployed app talks to the engine through it.

the browser never talks to the gpu box directly. every request goes through a node runtime route handler that forwards to the engine at `QWENFAST_BASE_URL` and pipes the response body back unbuffered, so sse frames arrive one token at a time.

## file map

| path | what it does |
| --- | --- |
| `app/page.tsx` | the chat page: composer, message list, client-side history trimming, per-response speed meter |
| `app/enginebar.tsx` | header strip that polls `/api/metrics` and renders the engine's own numbers |
| `app/markdown.tsx` | markdown renderer for assistant turns |
| `app/api/chat/route.ts` | keyed hop for the page: `POST` streams a chat turn, `GET` warms the upstream connection |
| `app/api/limits/route.ts` | keyed hop that returns the engine's `/v1/limits` envelope to the page, cached for a minute |
| `app/api/metrics/route.ts` | parses the engine's prometheus exposition into json |
| `app/api/diag/route.ts` | serving-path diagnostics: region, resolved endpoint, round-trip samples |
| `app/v1/*/route.ts` | the transparent proxy routes |
| `lib/proxy.ts` | proxy implementation: header forwarding, error passthrough, idle-stream watchdog |
| `lib/upstream.ts` | pooled http dispatcher, endpoint resolution, health probe |
| `lib/trim.ts` | keeps a conversation inside the prompt budget by dropping its oldest turns |
| `stub/server.mjs` | openai-shaped sse stub engine for local development and route tests |
| `tests/` | unit tests for `lib/trim.ts` and end-to-end route tests against the stub |
| `vercel.json` | region pin and function durations |

## environment variables

| variable | read by | meaning |
| --- | --- | --- |
| `QWENFAST_BASE_URL` | `lib/upstream.ts` | engine origin, scheme and host only, no `/v1` suffix. set it explicitly; the built-in default is only a placeholder |
| `QWENFAST_BASE_URLS` | `lib/upstream.ts` | comma-separated list of candidate origins. when set it replaces the single url; the first healthy one wins |
| `QWENFAST_DEMO_KEY` | `app/api/chat`, `app/api/limits` | api key the page's keyed hops send as `Authorization: Bearer`. the `/v1/*` proxy never uses it |
| `QWENFAST_IDLE_STREAM_MS` | `lib/proxy.ts` | how long a proxied stream may stay silent before the proxy closes it in-band. default `60000` |

the key is metered by the engine under its own name, so usage from the demo page and usage from direct api callers stay distinguishable in the engine's usage report.

## run locally against the stub

the stub speaks the engine's wire format: sse chat completions with one chunk per token, `/v1/models`, `/v1/limits`, `/health`, a `/metrics` exposition and the openai error envelope. auth is on by default with two keys, `qf-stub-public` and `qf-stub-demo`.

```bash
npm install
npm run stub                     # listens on :8000
```

in a second terminal:

```bash
QWENFAST_BASE_URL=http://127.0.0.1:8000 QWENFAST_DEMO_KEY=qf-stub-demo npm run dev
```

open `http://localhost:3000`. stub knobs, all optional: `PORT`, `STUB_TPS` (tokens per second, default 150), `STUB_KEYS` (`name:key,name:key`), `STUB_ADMIN_KEY`, `STUB_NO_AUTH=1` to disable auth, `STUB_CTX`, `STUB_CTX_CHARS`, `STUB_MIN_COMPLETION`, `STUB_MAX_MESSAGES`, and `STUB_STALL=1` or a message containing `STALL_NOW` to make a stream hang so the proxy watchdog can be exercised.

## run locally against a real server

start the engine's http server (see [the server readme](../engine/qwenfast/server/README.md)), then point the app at it with a key the server accepts:

```bash
QWENFAST_BASE_URL=http://127.0.0.1:8000 QWENFAST_DEMO_KEY=<key> npm run dev
```

the page defaults every turn to `temperature: 0`, `max_tokens: 1024` and `chat_template_kwargs.enable_thinking: false`. greedy matters: the engine runs speculative decoding only for greedy requests, so a non-zero temperature routes the turn to the plain decode path.

## routes

### page routes (keyed with `QWENFAST_DEMO_KEY`)

| route | method | behaviour |
| --- | --- | --- |
| `/api/chat` | `POST` | body `{ messages, temperature? }`. forwards to `/v1/chat/completions` with `stream: true` and `stream_options.include_usage`, pipes the sse body back, and sets `X-Upstream-Ms` and `Server-Timing` to the time until upstream headers arrived. engine refusals come back as `{ error, offline, code }` with the engine's own message and `error.code` |
| `/api/chat` | `GET` | resolves the endpoint and opens a pooled connection. the page calls it on load and on composer focus |
| `/api/limits` | `GET` | the engine's `/v1/limits` envelope, cached for 60 s, with a conservative fallback when the engine is down |
| `/api/metrics` | `GET` | json summary of the engine's prometheus metrics: running and waiting requests, per-stream tokens per second implied by mean tpot, ttft, speculative accept length and rate, kv utilisation, uptime. returns `{ ok: false }` when the engine is down |
| `/api/diag` | `GET` | region, preferred and resolved origins, and four round-trip samples to `/health` |

### proxy routes (transparent)

| route | method |
| --- | --- |
| `/v1/chat/completions` | `POST` |
| `/v1/completions` | `POST` |
| `/v1/models` | `GET` |
| `/v1/limits` | `GET` |

the proxy contract, implemented in `lib/proxy.ts`:

- the client's `Authorization` header is forwarded verbatim. the proxy holds no key of its own; auth, rate limits and metering are decided by the engine.
- the upstream status code and body are returned unchanged, including the error envelope, so a `429` keeps its `Retry-After` and an sdk can read `error.code`.
- `X-Forwarded-For` is forwarded so the engine's per-ip concurrency cap sees the real caller.
- `Content-Type`, `Retry-After`, `X-Request-Id`, `Cache-Control`, `X-Accel-Buffering` and `X-Qwenfast-Max-Tokens` are passed back.
- the body is piped, so sse streams arrive token by token. a stream that stays silent for `QWENFAST_IDLE_STREAM_MS` is closed in-band with a final `data: {"error": ...}` frame followed by `data: [DONE]`.
- when no candidate origin answers `/health`, the proxy returns `503` with `error.code` `engine_offline` and `Retry-After: 30`. a `502` or `504` from the edge is reported the same way and triggers a re-probe on the next request.
- client disconnects abort the upstream request, so an abandoned stream stops costing gpu time.

## deploy to vercel

```bash
vercel --yes           # preview
vercel --prod --yes    # production
```

set `QWENFAST_BASE_URL` and `QWENFAST_DEMO_KEY` in the project's environment variables. `vercel.json` gives the streaming routes a 300 s `maxDuration` and pins the function region.

## design notes on latency

two choices in this app come straight off time to first token:

- region pinning. `vercel.json` sets `regions` so the function runs in the region closest to the engine. every request pays one round trip to the engine before the first token arrives, so function placement is a first-order term. `GET /api/diag` reports the region it ran in and round-trip samples, so the choice can be re-checked after a move.
- connection pooling. serverless functions pool nothing by default, so each request would pay a fresh tcp and tls handshake. `lib/upstream.ts` installs a module-scoped `undici` `Agent` through `setGlobalDispatcher` with a 10 minute keep-alive, so only the first request on a warm instance pays the handshake. the page calls `GET /api/chat` on load and on composer focus to open that socket before the user sends anything.

## speed readouts

the page keeps two independent readouts:

- client-measured, per response: tokens per second, ttft with the network share broken out from `X-Upstream-Ms`, inter-token latency, token count and the engine's speculative accept length. the settled number comes from the final chunk's `usage`. the running number counts sse content chunks, scaled by a tokens-per-chunk ratio calibrated from previous responses; the ratio starts at 1.0, so the live meter can only under-read before it calibrates.
- engine-measured, live: `app/api/metrics/route.ts` reads the engine's prometheus endpoint and the header strip polls it at 1 s while streaming and 5 s when idle. it renders nothing while the engine is down.

## tests

```bash
npm test               # unit tests for lib/trim.ts
npm run test:routes    # end-to-end route tests against the stub
npm run test:all       # both
```

`tests/trim.test.ts` covers the token estimate, the prompt budget and the trimming rules. `tests/routes.test.ts` builds the app if `.next` is stale, starts the stub and `next start` on their own ports, and checks the deployed code path: keyed proxying, error passthrough with `error.code` intact, sse streaming with `X-Qwenfast-Max-Tokens`, `X-Forwarded-For` forwarding, the idle-stream watchdog, and the `/api/limits` and `/api/chat` page routes.
