// end to end: the real daemon process, real pi sessions, real bash tool execution, against a
// scripted openai compatible server. no gpu, no jarvislabs.

import assert from "node:assert/strict";
import { spawn, type ChildProcess } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, before, test } from "node:test";
import { startMock, type MockServer } from "./mock-openai.ts";

const ROOT = new URL("..", import.meta.url).pathname;
let mock: MockServer;
let home: string;
let port = 0;
let daemon: ChildProcess | undefined;

function startDaemon(): ChildProcess {
  const child = spawn(process.execPath, [join(ROOT, "src/daemon.ts")], {
    env: {
      ...process.env,
      QFA_HOME: home,
      QFA_PORT: String(port),
      QFA_JARVIS: "0",
      QFA_BIG_URL: mock.url,
      QFA_SMALL_URL: mock.url,
      QFA_API_KEY: "test-key",
    },
    stdio: "ignore",
  });
  return child;
}

async function api(method: string, path: string, body?: unknown): Promise<any> {
  for (let i = 0; ; i++) {
    try {
      const r = await fetch(`http://127.0.0.1:${port}${path}`, {
        method,
        headers: body ? { "content-type": "application/json" } : undefined,
        body: body ? JSON.stringify(body) : undefined,
      });
      return await r.json();
    } catch (err) {
      if (i > 100) throw err;
      await new Promise((r) => setTimeout(r, 200));
    }
  }
}

async function waitFor(id: string, statuses: string[], ms = 60000): Promise<any> {
  const end = Date.now() + ms;
  for (;;) {
    const t = await api("GET", `/tasks/${id}`);
    if (statuses.includes(t.status)) return t;
    if (Date.now() > end) throw new Error(`timeout waiting for ${statuses}: ${JSON.stringify(t).slice(0, 800)}`);
    await new Promise((r) => setTimeout(r, 300));
  }
}

async function waitHealthy(): Promise<void> {
  for (let i = 0; i < 100; i++) {
    const h = await api("GET", "/health");
    if (h.backend.endpoints.big.healthy && h.backend.endpoints.small.healthy) return;
    await new Promise((r) => setTimeout(r, 300));
  }
  throw new Error("backend never healthy");
}

before(async () => {
  mock = await startMock({
    judgeReply: "SMALL",
    command: () => "echo hello > out.txt && cat out.txt",
    finalText: (model) => (model.startsWith("qwen3.6") ? "wrote it\nSTATUS: DONE" : "done on big\nSTATUS: DONE"),
  });
  home = mkdtempSync(join(tmpdir(), "qfa-e2e-"));
  port = 20000 + Math.floor(Math.random() * 20000);
  daemon = startDaemon();
  await waitHealthy();
});

after(async () => {
  daemon?.kill("SIGKILL");
  await mock.close();
});

test("a small task runs tools and finishes on the small tier", async () => {
  const cwd = mkdtempSync(join(tmpdir(), "qfa-ws-"));
  const t = await api("POST", "/tasks", { prompt: "create out.txt containing hello", cwd });
  const done = await waitFor(t.id, ["done", "failed"]);
  assert.equal(done.status, "done", JSON.stringify(done));
  assert.equal(done.attempts[0].tier, "small");
  assert.equal(readFileSync(join(cwd, "out.txt"), "utf8"), "hello\n");
  const agentReq = mock.requests.find((r) => r.body.stream && r.model === "qwen3.6-35b-a3b");
  assert.ok(agentReq, "agent request went to the small model");
  assert.equal(agentReq.body.chat_template_kwargs?.enable_thinking, false);
  assert.ok(agentReq.body.tools?.some((x: any) => x.function?.name === "bash"));
});

test("blocked on small escalates to the big model with thinking", async () => {
  mock.script.finalText = (model) => (model.startsWith("qwen3.6") ? "cannot\nSTATUS: BLOCKED: too hard" : "fixed\nSTATUS: DONE");
  const cwd = mkdtempSync(join(tmpdir(), "qfa-ws-"));
  const t = await api("POST", "/tasks", { prompt: "list the files", cwd });
  const done = await waitFor(t.id, ["done", "failed"]);
  assert.equal(done.status, "done", JSON.stringify(done));
  assert.deepEqual(done.attempts.map((a: any) => [a.tier, a.outcome]), [["small", "blocked"], ["medium", "done"]]);
  const bigReq = mock.requests.filter((r) => r.body.stream && r.model === "qwen3.8-27b").at(-1)!;
  assert.equal(bigReq.body.chat_template_kwargs.enable_thinking, true);
  assert.equal(bigReq.body.chat_template_kwargs.reasoning_effort, "medium");
  mock.script.finalText = () => "ok\nSTATUS: DONE";
});

test("verify failure escalates and feeds the output back", async () => {
  const cwd = mkdtempSync(join(tmpdir(), "qfa-ws-"));
  // passes only once the file says "fixed", which the big model's command writes
  mock.script.command = (model) => (model.startsWith("qwen3.6") ? "echo broken > state" : "echo fixed > state");
  const t = await api("POST", "/tasks", { prompt: "make it work", cwd, verify: "grep -q fixed state" });
  const done = await waitFor(t.id, ["done", "failed"]);
  assert.equal(done.status, "done", JSON.stringify(done));
  assert.equal(done.attempts[0].outcome, "verify-failed");
  const esc = mock.requests.filter((r) => r.model === "qwen3.8-27b" && r.body.stream).at(-1)!;
  assert.match(JSON.stringify(esc.body.messages), /did not finish: verify-failed/);
  mock.script.command = () => "echo hello > out.txt && cat out.txt";
});

test("a killed daemon resumes the interrupted task from its session", async () => {
  mock.script.delayAfterToolMs = 6000;
  const cwd = mkdtempSync(join(tmpdir(), "qfa-ws-"));
  const t = await api("POST", "/tasks", { prompt: "create out.txt containing hello", cwd, tier: "medium" });
  // the tool call has run (out.txt exists) and the model is "thinking" about the result: kill now
  for (let i = 0; i < 100 && !existsSync(join(cwd, "out.txt")); i++) await new Promise((r) => setTimeout(r, 100));
  await new Promise((r) => setTimeout(r, 1000));
  daemon!.kill("SIGKILL");
  await new Promise((r) => setTimeout(r, 500));
  mock.script.delayAfterToolMs = 0;
  daemon = startDaemon();
  const done = await waitFor(t.id, ["done", "failed"]);
  assert.equal(done.status, "done", JSON.stringify(done));
  assert.equal(done.attempts[0].outcome, "interrupted");
  assert.ok(done.attempts[0].sessionFile && existsSync(done.attempts[0].sessionFile));
  assert.equal(done.attempts.at(-1).sessionFile, done.attempts[0].sessionFile, "resumed the same session file");
  const resumeReq = mock.requests.filter((r) => r.body.stream).at(-2)!;
  const text = JSON.stringify(resumeReq.body.messages);
  assert.match(text, /You were interrupted/);
  assert.match(text, /create out.txt containing hello/, "resumed context still holds the original task");
});

test("inbox files become tasks", async () => {
  const cwd = mkdtempSync(join(tmpdir(), "qfa-ws-"));
  writeFileSync(join(home, "inbox", "job.md"), `cwd: ${cwd}\ntier: small\n\ncreate out.txt`);
  let id: string | undefined;
  for (let i = 0; i < 100 && !id; i++) {
    const list = await api("GET", "/tasks");
    id = list.find((x: any) => x.prompt === "create out.txt")?.id;
    await new Promise((r) => setTimeout(r, 300));
  }
  assert.ok(id);
  const done = await waitFor(id!, ["done", "failed"]);
  assert.equal(done.status, "done");
  assert.equal(done.source, "inbox:job.md");
});

async function chat(body: Record<string, unknown>): Promise<{ status: number; tier: string | null; text: string }> {
  const r = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
    method: "POST",
    headers: { "content-type": "application/json", authorization: "Bearer anything" },
    body: JSON.stringify(body),
  });
  return { status: r.status, tier: r.headers.get("x-qwenfast-tier"), text: await r.text() };
}

test("gateway lists the tier models", async () => {
  const r = await api("GET", "/v1/models");
  assert.deepEqual(r.data.map((m: any) => m.id), ["auto", "qwen3.8-27b-xhigh", "qwen3.8-27b", "qwen3.6-35b-a3b"]);
});

test("gateway routes auto conversations and keeps the highest tier", async () => {
  const system = { role: "system", content: "you are qwenfast code" };
  const first = { role: "user", content: "list the files in src" };
  const a = await chat({ model: "qwenfast/auto", stream: true, messages: [system, first] });
  assert.equal(a.status, 200);
  assert.equal(a.tier, "small");
  assert.match(a.text, /data: /);
  const upA = mock.requests.at(-1)!;
  assert.equal(upA.model, "qwen3.6-35b-a3b");
  assert.equal(upA.body.chat_template_kwargs.enable_thinking, false);

  const hard = {
    role: "user",
    content: "Debug the intermittent deadlock in the scheduler, find the root cause across the codebase and refactor the locking design.",
  };
  const b = await chat({ model: "auto", stream: true, messages: [system, first, { role: "assistant", content: "ok" }, hard] });
  assert.equal(b.tier, "large");
  const upB = mock.requests.at(-1)!;
  assert.equal(upB.model, "qwen3.8-27b");
  assert.equal(upB.body.chat_template_kwargs.reasoning_effort, "xhigh");

  // an easy follow up in the same conversation stays on the tier the conversation needed
  const c = await chat({
    model: "auto", stream: true,
    messages: [system, first, { role: "assistant", content: "ok" }, hard, { role: "assistant", content: "done" }, { role: "user", content: "thanks, show git status" }],
  });
  assert.equal(c.tier, "large");
});

test("gateway maps explicit models and normalizes reasoning effort", async () => {
  const r = await chat({ model: "qwen3.8-27b", reasoning_effort: "high", messages: [{ role: "user", content: "hi there" }] });
  assert.equal(r.status, 200);
  assert.equal(r.tier, "medium");
  const up = mock.requests.at(-1)!;
  assert.equal(up.body.reasoning_effort, undefined);
  assert.equal(up.body.chat_template_kwargs.reasoning_effort, "medium");
  assert.equal(up.body.chat_template_kwargs.enable_thinking, true);
  const bad = await fetch(`http://127.0.0.1:${port}/v1/embeddings`, { method: "POST", body: "{}" });
  assert.equal(bad.status, 404);
});

test("gateway routing survives a daemon restart", async () => {
  const system = { role: "system", content: "restart probe" };
  const hard = { role: "user", content: "Investigate the root cause of the race condition in the allocator and redesign the locking across the codebase." };
  const a = await chat({ model: "auto", stream: true, messages: [system, hard] });
  assert.equal(a.tier, "large");
  await new Promise((r) => setTimeout(r, 2500)); // state is saved debounced
  daemon!.kill("SIGKILL");
  await new Promise((r) => setTimeout(r, 500));
  daemon = startDaemon();
  await waitHealthy();
  const b = await chat({ model: "auto", stream: true, messages: [system, hard, { role: "assistant", content: "ok" }, { role: "user", content: "show git status" }] });
  assert.equal(b.tier, "large", "the conversation kept its tier across the restart");
});
