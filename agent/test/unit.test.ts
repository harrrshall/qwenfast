import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { endpointsFromMachine, parseEnvFile, sshTarget } from "../src/backend.ts";
import { deepMerge, defaultConfig } from "../src/config.ts";
import { tierModel } from "../src/models.ts";
import { escalate, heuristicScore, parseJudgeReply, route, tierForScore } from "../src/router.ts";
import { parseStatus, runShell } from "../src/runner.ts";
import { parseInboxFile, TaskStore } from "../src/store.ts";

test("heuristic sends chores to small and hard work to large", () => {
  const easy = ["list the files in src", "what is the version in package.json?", "fix the typo in README.md", "summarize CHANGELOG.md in one sentence"];
  for (const p of easy) assert.equal(tierForScore(heuristicScore(p).score), "small", p);
  const hard = [
    "Debug the intermittent deadlock in the scheduler: tests are failing on CI about 1 in 20 runs. Find the root cause and fix it across the codebase, then optimize lock contention.",
    "Design and implement a new distributed rate limiter service with consensus between nodes, prove the invariant holds, and refactor the api layer to use it.",
  ];
  for (const p of hard) assert.equal(tierForScore(heuristicScore(p).score), "large", p);
  const mid = "Add a --json flag to the export command in cli.py that prints the rows as json instead of a table, and add a test for it in tests/test_cli.py.";
  assert.equal(tierForScore(heuristicScore(mid).score), "medium");
});

test("judge is consulted only when unsure and can move one tier", async () => {
  let calls = 0;
  const judge = async () => {
    calls++;
    return "large" as const;
  };
  const sure = await route("list the files", { judge });
  assert.equal(sure.source, "heuristic");
  assert.equal(calls, 0);
  // a borderline prompt (score between tiers) goes to the judge; small guess can only rise to medium
  const p = "update the readme with the new flag and rename the old option in config.py";
  const s = heuristicScore(p).score;
  const r = await route(p, { judge });
  if (r.source === "judge") {
    assert.equal(calls, 1);
    assert.ok(!(r.tier === "large" && tierForScore(s) === "small"));
  }
  const forced = await route("anything", { forced: "large", judge });
  assert.equal(forced.tier, "large");
  // no keyword signal at all: the judge is asked even though 0.5 sits in the confident band
  let asked = 0;
  const neutral = await route("please take care of the thing we talked about in the design doc for the team", {
    judge: async () => { asked++; return "medium"; },
  });
  assert.equal(asked, 1);
  assert.equal(neutral.source, "judge");
  const failing = await route(p, { judge: async () => { throw new Error("down"); } });
  assert.ok(["heuristic", "judge-failed"].includes(failing.source));
});

test("escalation ladder", () => {
  assert.equal(escalate("small"), "medium");
  assert.equal(escalate("medium"), "large");
  assert.equal(escalate("large"), "large");
  assert.equal(parseJudgeReply("  Medium."), "medium");
  assert.throws(() => parseJudgeReply("dunno"));
});

test("status line parsing", () => {
  assert.deepEqual(parseStatus("did it\nSTATUS: DONE"), { status: "done" });
  assert.deepEqual(parseStatus("no\nSTATUS: BLOCKED: needs a gpu\n"), { status: "blocked", reason: "needs a gpu" });
  assert.deepEqual(parseStatus("just text"), { status: "none" });
});

test("model definitions carry qwen thinking kwargs", () => {
  const cfg = defaultConfig("/tmp/x");
  const big = tierModel("large", cfg.tiers.large, "https://h/");
  assert.equal(big.baseUrl, "https://h/v1");
  assert.equal(big.thinkingLevelMap?.high, "xhigh");
  assert.equal(big.compat?.thinkingFormat, "chat-template");
  assert.ok(big.compat?.chatTemplateKwargs?.reasoning_effort);
  const small = tierModel("small", cfg.tiers.small, "http://s");
  assert.equal(small.provider, "qwenfast-small");
  assert.equal(small.compat?.chatTemplateKwargs?.reasoning_effort, undefined);
});

test("jarvislabs endpoint mapping and env parsing", () => {
  const m = { http_ports: "8000,8001,8080", endpoints: ["https://a1", "https://a2", "https://a3", "https://a4"] };
  assert.deepEqual(endpointsFromMachine(m), { big: "https://a2", small: "https://a3" });
  assert.deepEqual(endpointsFromMachine({ http_ports: "8001", endpoints: ["x", "y"] }), { small: "y" });
  assert.equal(sshTarget("ssh -o StrictHostKeyChecking=no root@203.0.113.10"), "root@203.0.113.10");
  assert.equal(sshTarget(undefined), undefined);
  assert.deepEqual(parseEnvFile("export A=1\nB='two'\n# c\n"), { A: "1", B: "two" });
  assert.deepEqual(deepMerge({ a: { b: 1, c: 2 }, d: 3 }, { a: { c: 9 } }), { a: { b: 1, c: 9 }, d: 3 });
});

test("store survives a restart and requeues interrupted tasks", () => {
  const dir = mkdtempSync(join(tmpdir(), "qfa-store-"));
  const s1 = new TaskStore(dir);
  const t = s1.create({ prompt: "p", cwd: "/tmp" });
  t.status = "running";
  t.attempts.push({ n: 1, tier: "small", model: "m", startedAt: new Date().toISOString(), sessionFile: "/x.jsonl" });
  s1.save(t);
  const s2 = new TaskStore(dir);
  const { requeued } = s2.load();
  assert.deepEqual(requeued, [t.id]);
  const back = s2.get(t.id)!;
  assert.equal(back.status, "queued");
  assert.equal(back.resume, true);
  assert.equal(back.attempts[0].outcome, "interrupted");
  assert.equal(JSON.parse(readFileSync(join(dir, t.id, "task.json"), "utf8")).status, "queued");
});

test("inbox header parsing", () => {
  const p = parseInboxFile("cwd: /tmp\ntier: large\nverify: make test\n\nDo the thing\nwell");
  assert.deepEqual(p, { cwd: "/tmp", tier: "large", verify: "make test", prompt: "Do the thing\nwell" });
  assert.deepEqual(parseInboxFile("just a prompt"), { prompt: "just a prompt" });
});

test("runShell captures output and kills on timeout", async () => {
  const dir = mkdtempSync(join(tmpdir(), "qfa-sh-"));
  writeFileSync(join(dir, "f"), "x");
  const ok = await runShell("cat f; exit 3", dir, 5000);
  assert.equal(ok.code, 3);
  assert.equal(ok.output, "x");
  const slow = await runShell("sleep 5", dir, 200);
  assert.notEqual(slow.code, 0);
});

test("jl calls fall back to the backup key on an auth failure", async () => {
  const { Backend } = await import("../src/backend.ts");
  const dir = mkdtempSync(join(tmpdir(), "qfa-jl-"));
  writeFileSync(
    join(dir, "jl"),
    `#!/bin/sh\nif [ "$JL_API_KEY" = good ]; then echo '{"machine_id": 7, "status": "Paused", "http_ports": "8000,8001", "endpoints": []}'; else echo '{"error": "401 unauthorized: invalid api key"}'; exit 1; fi\n`,
    { mode: 0o755 },
  );
  writeFileSync(join(dir, "primary.env"), "JARVISLABS_API_KEY=revoked\n");
  writeFileSync(join(dir, "backup.env"), "JL_API_KEY=good\n");
  writeFileSync(join(dir, "box"), "7\n");
  const cfg = defaultConfig(dir);
  cfg.jarvis.envFile = join(dir, "primary.env");
  cfg.jarvis.backupEnvFile = join(dir, "backup.env");
  cfg.jarvis.machineIdFile = join(dir, "box");
  cfg.jarvis.autoResume = false;
  const logs: string[] = [];
  const oldPath = process.env.PATH;
  process.env.PATH = `${dir}:${oldPath}`;
  try {
    const b = new Backend(cfg, (m) => logs.push(m), () => 0);
    await b.tick();
    assert.equal(b.box.status, "Paused");
    assert.ok(logs.some((l) => l.includes("retrying with the backup key")), logs.join("\n"));
  } finally {
    process.env.PATH = oldPath;
  }
});
