// qwenfast agent daemon: a long running, crash tolerant task runner on top of the pi sdk.
//
//   node agent/src/daemon.ts            (launchd keeps it alive: agent/launchd/install.sh)
//
// intake:   POST http://127.0.0.1:7788/tasks  {"prompt": ..., "cwd"?, "tier"?, "verify"?, "priority"?}
//           or drop a .md / .txt file into $QFA_HOME/inbox/ (first line "cwd: /path" optional)
// status:   GET /health, GET /tasks, GET /tasks/<id>, POST /tasks/<id>/cancel, POST /tasks/<id>/retry
//
// survival rules: every state change is an atomic file write; an unhandled error is logged and
// the loop continues; SIGTERM stops intake and lets running attempts be resumed by the next
// process; the process exits (and launchd restarts it) if its heap grows past the limit.

import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { appendFileSync, existsSync, readdirSync, readFileSync, renameSync, statSync, unlinkSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { Backend } from "./backend.ts";
import { loadConfig, readApiKey, type TierName } from "./config.ts";
import { Gateway } from "./gateway.ts";
import { httpJudge } from "./router.ts";
import { Runner } from "./runner.ts";
import { parseInboxFile, TaskStore, type Task } from "./store.ts";

const cfg = loadConfig();
const LOG = join(cfg.home, "logs", "daemon.log");
const HEAP_LIMIT = Number(process.env.QFA_HEAP_LIMIT_MB ?? 3072) * 1024 * 1024;

function rotate(): void {
  try {
    if (!existsSync(LOG) || statSync(LOG).size < cfg.log.maxBytes) return;
    for (let i = cfg.log.keep - 1; i >= 1; i--) if (existsSync(`${LOG}.${i}`)) renameSync(`${LOG}.${i}`, `${LOG}.${i + 1}`);
    renameSync(LOG, `${LOG}.1`);
  } catch {
    /* rotation is best effort */
  }
}

function log(msg: string): void {
  const line = `${new Date().toISOString()} ${msg}\n`;
  try {
    appendFileSync(LOG, line);
  } catch {
    /* disk full: keep running */
  }
  if (process.stdout.isTTY) process.stdout.write(line);
}

process.on("uncaughtException", (err) => log(`uncaught exception: ${err?.stack ?? err}`));
process.on("unhandledRejection", (err) => log(`unhandled rejection: ${(err as Error)?.stack ?? err}`));

const store = new TaskStore(join(cfg.home, "tasks"));
const { loaded, requeued } = store.load();
log(`daemon ${process.pid} starting; ${loaded} tasks on disk, ${requeued.length} requeued for resume`);

const running = new Map<string, Promise<void>>();
let gateway: Gateway | undefined;
const backend = new Backend(cfg, log, () => store.counts().queued + running.size + (gateway?.inflight ?? 0));
const apiKey = readApiKey(cfg);
const judge = cfg.router.llmJudge ? httpJudge(() => backend.url("small"), cfg.tiers.small.model, apiKey) : undefined;
const runner = new Runner({ cfg, store, backend, apiKey, judge, log });
gateway = new Gateway(cfg, backend, apiKey, log, judge);
let draining = false;
const startedAt = Date.now();

function schedule(): void {
  if (draining) return;
  while (running.size < cfg.concurrency) {
    const t = store.nextQueued(new Set(running.keys()));
    if (!t) return;
    backend.noteDemand();
    const p = runner
      .run(t)
      .catch((err) => log(`task ${t.id} crashed: ${err}`))
      .finally(() => {
        running.delete(t.id);
        setImmediate(schedule);
      });
    running.set(t.id, p);
  }
}

// -- intake -----------------------------------------------------------------------------------

const TIERS = new Set(["small", "medium", "large"]);

function submit(body: Record<string, unknown>, source: string): Task {
  const prompt = String(body.prompt ?? "").trim();
  if (!prompt) throw new Error("prompt is required");
  const cwd = resolve(String(body.cwd ?? cfg.workspace));
  if (!existsSync(cwd) || !statSync(cwd).isDirectory()) throw new Error(`cwd does not exist: ${cwd}`);
  const tier = body.tier && body.tier !== "auto" ? String(body.tier) : undefined;
  if (tier && !TIERS.has(tier)) throw new Error(`tier must be small, medium, large or auto`);
  const t = store.create({
    prompt,
    cwd,
    tier: tier as TierName | undefined,
    verify: body.verify ? String(body.verify) : undefined,
    timeoutMinutes: body.timeoutMinutes ? Number(body.timeoutMinutes) : undefined,
    priority: body.priority ? Number(body.priority) : 0,
    source,
  });
  log(`task ${t.id} queued from ${source}`);
  setImmediate(schedule);
  return t;
}

function scanInbox(): void {
  const dir = join(cfg.home, "inbox");
  for (const name of readdirSync(dir)) {
    if (!/\.(md|txt)$/.test(name)) continue;
    const file = join(dir, name);
    try {
      if (Date.now() - statSync(file).mtimeMs < 2000) continue; // still being written
      const t = submit(parseInboxFile(readFileSync(file, "utf8")), `inbox:${name}`);
      renameSync(file, join(store.taskDir(t.id), "inbox-" + name));
    } catch (err) {
      log(`inbox ${name} rejected: ${err}`);
      try {
        renameSync(file, `${file}.rejected`);
      } catch {
        /* ignore */
      }
    }
  }
}

// -- http api ---------------------------------------------------------------------------------

function send(res: ServerResponse, code: number, body: unknown): void {
  res.writeHead(code, { "content-type": "application/json" });
  res.end(JSON.stringify(body, null, 2));
}

async function readBody(req: IncomingMessage): Promise<Record<string, unknown>> {
  const chunks: Buffer[] = [];
  let n = 0;
  for await (const c of req) {
    n += (c as Buffer).length;
    if (n > 4 << 20) throw new Error("body too large");
    chunks.push(c as Buffer);
  }
  const text = Buffer.concat(chunks).toString() || "{}";
  return JSON.parse(text) as Record<string, unknown>;
}

function summary(t: Task) {
  const last = t.attempts.at(-1);
  return {
    id: t.id, status: t.status, tier: last?.tier ?? t.route?.tier, attempts: t.attempts.length,
    createdAt: t.createdAt, updatedAt: t.updatedAt, prompt: t.prompt.slice(0, 120), error: t.error,
  };
}

const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url ?? "/", "http://x");
    const parts = url.pathname.split("/").filter(Boolean);
    if (parts[0] === "v1") return await gateway!.handle(req, res, url.pathname);
    if (req.method === "POST" && url.pathname === "/box/wake") {
      backend.wake(Number(url.searchParams.get("minutes") ?? 15));
      return send(res, 202, { ok: true, box: backend.snapshot().box });
    }
    if (req.method === "POST" && url.pathname === "/box/pause") {
      return send(res, 200, { ok: true, result: await backend.pauseNow() });
    }
    if (req.method === "GET" && url.pathname === "/health") {
      return send(res, 200, {
        ok: true, pid: process.pid, uptimeS: Math.round((Date.now() - startedAt) / 1000), draining,
        tasks: store.counts(), running: [...running.keys()], backend: backend.snapshot(), gateway: gateway?.snapshot(),
        heapMB: Math.round(process.memoryUsage().heapUsed / 1048576),
      });
    }
    if (parts[0] === "tasks") {
      if (req.method === "POST" && parts.length === 1) {
        if (draining) return send(res, 503, { error: "draining" });
        return send(res, 201, store.get(submit(await readBody(req), "http").id));
      }
      if (req.method === "GET" && parts.length === 1) {
        const status = url.searchParams.get("status");
        return send(res, 200, store.list().filter((t) => !status || t.status === status).map(summary));
      }
      const t = parts[1] ? store.get(parts[1]) : undefined;
      if (!t) return send(res, 404, { error: "no such task" });
      if (req.method === "GET" && parts.length === 2) return send(res, 200, t);
      if (req.method === "GET" && parts[2] === "events") {
        const f = join(store.taskDir(t.id), "events.log");
        res.writeHead(200, { "content-type": "text/plain" });
        return res.end(existsSync(f) ? readFileSync(f, "utf8").split("\n").slice(-Number(url.searchParams.get("tail") ?? 200)).join("\n") : "");
      }
      if (req.method === "POST" && parts[2] === "cancel") {
        if (t.status === "queued") {
          t.status = "cancelled";
          store.save(t);
        } else runner.cancel(t.id);
        return send(res, 200, summary(t));
      }
      if (req.method === "POST" && parts[2] === "retry") {
        if (t.status === "running") return send(res, 409, { error: "running" });
        t.status = "queued";
        t.error = undefined;
        t.route = undefined;
        t.attempts = [];
        store.save(t);
        setImmediate(schedule);
        return send(res, 200, summary(t));
      }
    }
    send(res, 404, { error: "not found" });
  } catch (err) {
    send(res, 400, { error: String((err as Error)?.message ?? err) });
  }
});
server.on("error", (err) => {
  log(`http server error: ${err}`);
  process.exit(1); // port in use: another daemon is running; launchd retries later
});
server.listen(cfg.port, cfg.host, () => log(`listening on http://${cfg.host}:${cfg.port}`));

// -- lifecycle --------------------------------------------------------------------------------

backend.start();
const heartbeatFile = join(cfg.home, "heartbeat.json");
const tick = setInterval(() => {
  try {
    scanInbox();
    schedule();
    rotate();
    writeFileSync(heartbeatFile, JSON.stringify({ pid: process.pid, at: new Date().toISOString(), tasks: store.counts(), running: running.size }));
    if (process.memoryUsage().heapUsed > HEAP_LIMIT) {
      log(`heap ${Math.round(process.memoryUsage().heapUsed / 1048576)} MB over limit; restarting`);
      void shutdown("heap limit");
    }
  } catch (err) {
    log(`tick error: ${err}`);
  }
}, 5000);

async function shutdown(why: string): Promise<void> {
  if (draining) return;
  draining = true;
  log(`shutting down (${why}); ${running.size} running tasks will resume on next start`);
  clearInterval(tick);
  backend.stop();
  server.close();
  // running attempts stay "running" on disk and are requeued with resume=true by the next process
  setTimeout(() => process.exit(0), 1500).unref();
}
process.on("SIGTERM", () => void shutdown("SIGTERM"));
process.on("SIGINT", () => void shutdown("SIGINT"));
try {
  if (existsSync(join(cfg.home, "stop"))) unlinkSync(join(cfg.home, "stop"));
} catch {
  /* ignore */
}
schedule();
