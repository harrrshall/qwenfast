// Tiny OpenAI-compatible SSE stub for local development of the demo UI.
// Emits one chunk per "token" at ~150 tok/s, matching qwenfast's wire format.
//   node stub/server.mjs        # listens on :8000
import http from "node:http";

const PORT = Number(process.env.PORT || 8000);
const TPS = Number(process.env.STUB_TPS || 150);
const MODEL = "qwen3.8-27b";

// Auth, mirroring server/auth.py closely enough to exercise the proxy end to end: /v1/* needs a
// known Bearer key, /admin/usage needs the admin key. STUB_KEYS is "name:key,name:key"; leaving
// it unset keeps the stub open (the pre-existing behaviour for plain UI work).
const KEYS = new Map(
  (process.env.STUB_KEYS ?? "public:qf-stub-public,demo:qf-stub-demo")
    .split(",")
    .map((pair) => pair.split(":"))
    .filter((kv) => kv.length === 2)
    .map(([name, key]) => [key.trim(), name.trim()]),
);
const ADMIN_KEY = process.env.STUB_ADMIN_KEY ?? "qf-stub-admin";
const AUTH_ON = process.env.STUB_NO_AUTH !== "1";

// Mirrors the engine's request envelope closely enough to exercise the client-side trimming
// and the error passthrough. STUB_CTX_CHARS is a *character* budget standing in for the
// server's token budget; anything longer gets the same 400 the real server sends.
const CTX = Number(process.env.STUB_CTX ?? 4096);
const MIN_COMPLETION = Number(process.env.STUB_MIN_COMPLETION ?? 256);
const CTX_CHARS = Number(process.env.STUB_CTX_CHARS ?? 6000);
const MAX_MESSAGES = Number(process.env.STUB_MAX_MESSAGES ?? 128);
// A request whose text contains STALL_MARKER gets one SSE frame and then silence forever, so
// the proxy's idle-stream watchdog has something to actually time out against. Per-request
// rather than per-process so one stub can serve both the happy path and the stall test.
const STALL_MARKER = process.env.STUB_STALL_MARKER ?? "STALL_NOW";
const STALL_ALL = process.env.STUB_STALL === "1";

// What the last request carried, so a test can assert the proxy forwarded it. Never stores
// the Authorization value itself — only whether one was present.
const LAST = { xff: null, hadAuth: false, path: null };

const REPLY = `Here is a short answer, streamed one token at a time.

Some **bold** text, a bit of \`inline_code\`, and a list:

- first item
- second item
- third item

\`\`\`python
def speedup(baseline, ours):
    return baseline / ours
\`\`\`

That is all.`;

// Crude tokenization: whitespace-preserving word pieces.
const TOKENS = REPLY.match(/\s+|\S+/g) ?? [];

const STATE = { running: 0, generated: 0, reqs: 0, start: Date.now(), perKey: new Map() };

const bearer = (req) => {
  const h = req.headers.authorization ?? "";
  return h.startsWith("Bearer ") ? h.slice(7).trim() : null;
};

/** Returns the key name, or writes the 401/403 and returns null. */
const authorize = (req, res, { admin = false } = {}) => {
  if (!AUTH_ON) return "anonymous";
  const token = bearer(req);
  if (admin) {
    if (token !== ADMIN_KEY) {
      json(res, token && KEYS.has(token) ? 403 : 401, {
        error: { message: admin ? "Admin key required" : "Invalid API key", type: "invalid_request_error" },
      });
      return null;
    }
    return "admin";
  }
  const name = token ? KEYS.get(token) ?? (token === ADMIN_KEY ? "admin" : null) : null;
  if (!name) {
    json(res, 401, { error: { message: "Invalid API key", type: "invalid_request_error" } });
    return null;
  }
  return name;
};

const bump = (name, completion) => {
  const t = STATE.perKey.get(name) ?? { requests: 0, prompt_tokens: 0, completion_tokens: 0, last_seen: 0 };
  t.requests += 1;
  t.prompt_tokens += 24;
  t.completion_tokens += completion;
  t.last_seen = Date.now() / 1000;
  STATE.perKey.set(name, t);
};

const limitsBody = () => ({
  max_context_length: CTX,
  max_prompt_tokens: null,
  default_max_tokens: 1024,
  max_output_tokens: 4096,
  min_completion_tokens: MIN_COMPLETION,
  max_messages: MAX_MESSAGES,
  max_request_bytes: 1000000,
  max_concurrent_streams_per_key: 8,
  max_concurrent_streams_per_ip: 8,
  request_timeout_s: 300,
  max_tokens_is_clamped: true,
});

/** OpenAI's error envelope, byte-compatible with server/errors.py. */
const apiError = (res, status, message, code, param = null) =>
  json(res, status, {
    error: { message, type: status === 429 ? "rate_limit_error" : "invalid_request_error", param, code },
  });

const readBody = (req) =>
  new Promise((resolve) => {
    let raw = "";
    req.on("data", (c) => (raw += c));
    req.on("end", () => {
      try {
        resolve(JSON.parse(raw || "{}"));
      } catch {
        resolve({});
      }
    });
  });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const json = (res, code, obj) => {
  res.writeHead(code, { "Content-Type": "application/json" });
  res.end(JSON.stringify(obj));
};

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, "http://localhost");

  // Introspection endpoints must not overwrite what they are there to report.
  if (url.pathname === "/debug/last") return json(res, 200, LAST);
  if (url.pathname === "/health") return json(res, 200, { status: "ok" });

  LAST.xff = req.headers["x-forwarded-for"] ?? null;
  LAST.hadAuth = Boolean(req.headers.authorization);
  LAST.path = url.pathname;

  if (url.pathname === "/status") {
    return json(res, 200, {
      status: "ok",
      healthy: true,
      model: MODEL,
      load: { in_flight: STATE.running, capacity: 256 },
      limits: limitsBody(),
      uptime_s: (Date.now() - STATE.start) / 1000,
      draining: false,
    });
  }

  if (url.pathname === "/v1/limits") {
    if (!authorize(req, res)) return;
    return json(res, 200, { object: "limits", model: MODEL, ...limitsBody() });
  }

  if (url.pathname === "/metrics") {
    // Mirrors server/metrics.py's exposition so /api/metrics parsing is exercised.
    const gen = STATE.generated;
    const up = (Date.now() - STATE.start) / 1000;
    const tpot = 1 / TPS;
    res.writeHead(200, { "Content-Type": "text/plain; version=0.0.4" });
    return res.end(
      [
        `qwenfast:num_requests_running ${STATE.running}`,
        `qwenfast:num_requests_waiting 0`,
        `qwenfast:ssm_slots_used 3`,
        `qwenfast:ssm_slots_total 512`,
        `qwenfast:kv_pages_used 2431`,
        `qwenfast:kv_pages_total 20000`,
        `qwenfast:prompt_tokens_total 240`,
        `qwenfast:generation_tokens_total ${gen}`,
        `qwenfast:tokens_per_second ${(gen / Math.max(up, 1e-6)).toFixed(3)}`,
        `qwenfast:time_to_first_token_seconds_sum ${(0.12 * Math.max(STATE.reqs, 1)).toFixed(4)}`,
        `qwenfast:time_to_first_token_seconds_count ${Math.max(STATE.reqs, 1)}`,
        `qwenfast:time_per_output_token_seconds_sum ${(tpot * Math.max(gen, 1)).toFixed(4)}`,
        `qwenfast:time_per_output_token_seconds_count ${Math.max(gen, 1)}`,
        `qwenfast:spec_accept_length 2.41`,
        `qwenfast:spec_acceptance_rate 0.80`,
        `qwenfast:uptime_seconds ${up.toFixed(1)}`,
        "",
      ].join("\n"),
    );
  }

  if (url.pathname === "/admin/usage") {
    if (!authorize(req, res, { admin: true })) return;
    const uptime = (Date.now() - STATE.start) / 1000;
    const hourNow = Math.floor(Date.now() / 3_600_000) * 3600;
    const inr = (uptime / 3600) * 188.73;
    const per_key = [...STATE.perKey.entries()].map(([name, t]) => ({
      name,
      requests: t.requests,
      errors: 0,
      prompt_tokens: t.prompt_tokens,
      completion_tokens: t.completion_tokens,
      total_tokens: t.prompt_tokens + t.completion_tokens,
      last_seen: t.last_seen,
    }));
    return json(res, 200, {
      now: Date.now() / 1000,
      model: MODEL,
      process_uptime_s: uptime,
      service_uptime_s: uptime,
      engine_health: "ok",
      totals: {
        requests: STATE.reqs,
        errors: 0,
        prompt_tokens: 24 * STATE.reqs,
        completion_tokens: STATE.generated,
        total_tokens: 24 * STATE.reqs + STATE.generated,
        rate_limited: 0,
        rejected_overload: 0,
        auth_failures: 0,
        dropped_audit_rows: 0,
      },
      recent_output_tokens_per_second: STATE.generated / Math.max(uptime, 1),
      recent_ttft_ms_mean: 120,
      per_key,
      hourly: Array.from({ length: 24 }, (_, i) => ({
        hour: hourNow - (23 - i) * 3600,
        requests: i === 23 ? STATE.reqs : 0,
        completion_tokens: i === 23 ? STATE.generated : 0,
        prompt_tokens: i === 23 ? 24 * STATE.reqs : 0,
      })),
      cost: {
        gpu_rate_inr_per_hour: 188.73,
        inr_per_usd: 87.5,
        billed_hours: uptime / 3600,
        inr,
        usd: inr / 87.5,
        per_1m_output_tokens_inr: STATE.generated ? (inr / STATE.generated) * 1e6 : null,
        per_1m_output_tokens_usd: STATE.generated ? (inr / 87.5 / STATE.generated) * 1e6 : null,
        per_1m_total_tokens_inr: null,
      },
      engine: {
        num_requests_running: STATE.running,
        num_requests_waiting: 0,
        generation_tokens_total: STATE.generated,
        per_stream_tokens_per_second: TPS,
        mean_ttft_ms: 120,
        spec_accept_length: 2.41,
        uptime_s: uptime,
      },
      capacity: { inflight: STATE.running, peak_inflight: STATE.running, capacity: 256 },
      keys: [...KEYS.values()].map((name) => ({ name, rpm: 120, tpm: 500000, max_tokens: 4096, admin: false })),
      key_store: { path: "(stub)", loaded_at: STATE.start / 1000, reload_count: 1, error: null },
      storage: { db_path: "(stub)", error: null },
    });
  }

  if (url.pathname === "/v1/models") {
    if (!authorize(req, res)) return;
    return json(res, 200, {
      object: "list",
      data: [{ id: MODEL, object: "model", created: 0, owned_by: "qwenfast" }],
    });
  }

  if (url.pathname === "/v1/chat/completions" && req.method === "POST") {
    const keyName = authorize(req, res);
    if (!keyName) return;

    const body = await readBody(req);
    const messages = Array.isArray(body.messages) ? body.messages : [];
    if (messages.length > MAX_MESSAGES) {
      return apiError(
        res, 400,
        `too many messages (${messages.length}); max ${MAX_MESSAGES}`,
        "too_many_messages", "messages",
      );
    }
    const chars = messages.reduce((n, m) => n + String(m?.content ?? "").length, 0);
    if (chars > CTX_CHARS) {
      const approx = Math.ceil(chars / 3);
      return apiError(
        res, 400,
        `This model's maximum context length is ${CTX} tokens. Your messages came to ` +
          `${approx} tokens, which leaves 0 for a reply (at least ${MIN_COMPLETION} required). ` +
          `Please shorten the conversation.`,
        "context_length_exceeded", "messages",
      );
    }

    const id = `chatcmpl-stub-${Date.now()}`;
    const created = Math.floor(Date.now() / 1000);
    res.writeHead(200, {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
      "X-Qwenfast-Max-Tokens": String(body.max_tokens ?? 1024),
    });

    const chunk = (delta, finish = null, usage = undefined) => {
      const body = {
        id,
        object: "chat.completion.chunk",
        created,
        model: MODEL,
        choices: usage ? [] : [{ index: 0, delta, finish_reason: finish }],
      };
      if (usage) body.usage = usage;
      res.write(`data: ${JSON.stringify(body)}\n\n`);
    };

    let closed = false;
    req.on("close", () => (closed = true));
    STATE.running += 1;
    STATE.reqs += 1;
    res.on("close", () => (STATE.running = Math.max(0, STATE.running - 1)));

    const stall =
      STALL_ALL || messages.some((m) => String(m?.content ?? "").includes(STALL_MARKER));

    await sleep(120); // simulated prefill / TTFT
    chunk({ role: "assistant", content: TOKENS[0] });
    STATE.generated += 1;

    if (stall) return; // never finishes: the proxy's idle watchdog must end this

    for (let i = 1; i < TOKENS.length; i++) {
      if (closed) return;
      await sleep(1000 / TPS);
      chunk({ content: TOKENS[i] });
      STATE.generated += 1;
    }

    bump(keyName, TOKENS.length);
    chunk({}, "stop");
    chunk(null, null, {
      prompt_tokens: 24,
      completion_tokens: TOKENS.length,
      total_tokens: 24 + TOKENS.length,
    });
    res.write("data: [DONE]\n\n");
    return res.end();
  }

  apiError(res, 404, `no route for ${url.pathname}`, "not_found");
});

server.listen(PORT, () =>
  console.log(
    `qwenfast stub on http://127.0.0.1:${PORT} (~${TPS} tok/s, auth ${AUTH_ON ? "on" : "off"}, ` +
      `${KEYS.size} keys)`,
  ),
);
