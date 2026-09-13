/**
 * Route tests for the demo app, driven against the bundled stub engine.
 *
 *   npm run build && npm run test:routes
 *
 * These run the *real* Next server (`next start`) against `stub/server.mjs`, so what is under
 * test is the deployed code path — `app/v1/*` → `lib/proxy.ts` → upstream, and `app/api/*` →
 * upstream — rather than a re-implementation of it. The stub speaks the same wire format as
 * `server/app.py`, including the OpenAI error envelope and `/v1/limits`.
 *
 * The build is the slow part, so it is done once in `before` and only when `.next` is stale.
 */

import test, { after, before } from "node:test";
import assert from "node:assert/strict";
import { spawn, type ChildProcess } from "node:child_process";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const STUB_PORT = 8811;
const APP_PORT = 3111;
const STUB = `http://127.0.0.1:${STUB_PORT}`;
const APP = `http://127.0.0.1:${APP_PORT}`;
const KEY = "qf-stub-public";
const DEMO_KEY = "qf-stub-demo";
/** Short enough that the stall test finishes in seconds, not a minute. */
const IDLE_MS = 1_500;

let stub: ChildProcess | undefined;
let app: ChildProcess | undefined;

function run(cmd: string, args: string[], env: NodeJS.ProcessEnv = {}): ChildProcess {
  return spawn(cmd, args, {
    cwd: ROOT,
    env: { ...process.env, ...env },
    stdio: ["ignore", "pipe", "pipe"],
  });
}

async function waitFor(url: string, timeoutMs = 90_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    try {
      const r = await fetch(url, { signal: AbortSignal.timeout(2_000) });
      if (r.ok || r.status === 401) return;
    } catch {
      /* not up yet */
    }
    if (Date.now() > deadline) throw new Error(`timed out waiting for ${url}`);
    await new Promise((r) => setTimeout(r, 300));
  }
}

/** Read an SSE body to completion, with a hard ceiling so a hang fails loudly. */
async function readStream(res: Response, capMs = 20_000): Promise<string> {
  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  const deadline = Date.now() + capMs;
  let out = "";
  for (;;) {
    if (Date.now() > deadline) throw new Error("stream did not terminate");
    const { done, value } = await reader.read();
    if (done) break;
    out += decoder.decode(value, { stream: true });
  }
  return out;
}

before(async () => {
  if (!existsSync(path.join(ROOT, ".next", "BUILD_ID"))) {
    await new Promise<void>((resolve, reject) => {
      const build = run("npx", ["next", "build"]);
      build.on("exit", (code) =>
        code === 0 ? resolve() : reject(new Error(`next build failed (${code})`)),
      );
    });
  }

  stub = run("node", ["stub/server.mjs"], {
    PORT: String(STUB_PORT),
    STUB_TPS: "2000",
    // Small enough that a few hundred characters trips the context error.
    STUB_CTX_CHARS: "600",
    STUB_MAX_MESSAGES: "12",
  });
  await waitFor(`${STUB}/health`);

  app = run("npx", ["next", "start", "-p", String(APP_PORT)], {
    QWENFAST_BASE_URLS: STUB,
    QWENFAST_DEMO_KEY: DEMO_KEY,
    QWENFAST_IDLE_STREAM_MS: String(IDLE_MS),
  });
  await waitFor(`${APP}/api/limits`);
});

after(() => {
  stub?.kill("SIGKILL");
  app?.kill("SIGKILL");
});

// ---------------------------------------------------------------- /v1 proxy

test("GET /v1/models proxies a keyed request", async () => {
  const r = await fetch(`${APP}/v1/models`, { headers: { Authorization: `Bearer ${KEY}` } });
  assert.equal(r.status, 200);
  const body = await r.json();
  assert.equal(body.data[0].id, "qwen3.8-27b");
});

test("an upstream 401 comes back unchanged, in the OpenAI envelope", async () => {
  const r = await fetch(`${APP}/v1/models`, { headers: { Authorization: "Bearer nope" } });
  assert.equal(r.status, 401);
  const body = await r.json();
  assert.equal(body.error.type, "invalid_request_error");
  assert.ok(body.error.message);
});

test("an upstream 400 passes through with its code and message intact", async () => {
  // This is the contract the whole error-shape change exists for: an SDK must be able to read
  // `error.code` after the proxy hop, not just a status number.
  const r = await fetch(`${APP}/v1/chat/completions`, {
    method: "POST",
    headers: { Authorization: `Bearer ${KEY}`, "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "qwen3.8-27b",
      messages: [{ role: "user", content: "x".repeat(4000) }],
    }),
  });
  assert.equal(r.status, 400);
  const body = await r.json();
  assert.equal(body.error.code, "context_length_exceeded");
  assert.equal(body.error.param, "messages");
  assert.match(body.error.message, /maximum context length/i);
});

test("SSE streams through, and X-Qwenfast-Max-Tokens is passed back", async () => {
  const r = await fetch(`${APP}/v1/chat/completions`, {
    method: "POST",
    headers: { Authorization: `Bearer ${KEY}`, "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "qwen3.8-27b",
      stream: true,
      max_tokens: 321,
      messages: [{ role: "user", content: "hi" }],
    }),
  });
  assert.equal(r.status, 200);
  assert.equal(r.headers.get("content-type")?.includes("text/event-stream"), true);
  assert.equal(r.headers.get("x-qwenfast-max-tokens"), "321");
  const text = await readStream(r);
  assert.ok(text.includes("data: "));
  assert.ok(text.trimEnd().endsWith("data: [DONE]"));
});

test("X-Forwarded-For reaches the engine, so the per-IP cap sees the real caller", async () => {
  await fetch(`${APP}/v1/models`, {
    headers: { Authorization: `Bearer ${KEY}`, "X-Forwarded-For": "203.0.113.7" },
  });
  const last = await (await fetch(`${STUB}/debug/last`)).json();
  assert.ok(String(last.xff).includes("203.0.113.7"), `xff was ${last.xff}`);
});

test("a stalled upstream stream is closed in-band instead of hanging", async () => {
  const started = Date.now();
  const r = await fetch(`${APP}/v1/chat/completions`, {
    method: "POST",
    headers: { Authorization: `Bearer ${KEY}`, "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "qwen3.8-27b",
      stream: true,
      messages: [{ role: "user", content: "STALL_NOW" }],
    }),
  });
  const text = await readStream(r, 20_000);
  const elapsed = Date.now() - started;

  // It ended, it ended promptly, and it said why — a bare close would be indistinguishable
  // from a complete answer.
  assert.ok(elapsed < 15_000, `took ${elapsed}ms`);
  assert.ok(text.includes("upstream_stalled"), text.slice(-400));
  assert.ok(text.trimEnd().endsWith("data: [DONE]"));
});

// ---------------------------------------------------------------- demo routes

test("GET /api/limits returns the engine's envelope for the chat page", async () => {
  const r = await fetch(`${APP}/api/limits`);
  assert.equal(r.status, 200);
  const body = await r.json();
  assert.equal(typeof body.max_context_length, "number");
  assert.equal(body.max_messages, 12); // what the stub was started with
});

test("POST /api/chat surfaces the engine's own 400 message and code", async () => {
  const r = await fetch(`${APP}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages: [{ role: "user", content: "x".repeat(4000) }] }),
  });
  assert.equal(r.status, 400);
  const body = await r.json();
  assert.equal(body.code, "context_length_exceeded");
  assert.match(body.error, /maximum context length/i);
  // Not the old generic line.
  assert.doesNotMatch(body.error, /Engine returned/);
});

test("POST /api/chat streams a normal turn", async () => {
  const r = await fetch(`${APP}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages: [{ role: "user", content: "hello" }] }),
  });
  assert.equal(r.status, 200);
  const text = await readStream(r);
  assert.ok(text.includes('"content"'));
  assert.ok(text.includes("[DONE]"));
});
