// durable task store: one directory per task under $QFA_HOME/tasks/<id>/
//
//   task.json     the record (atomically replaced on every transition: write tmp, fsync, rename)
//   sessions/     pi session jsonl files, one per attempt (resume continues the latest)
//   events.log    human readable trace of tool calls and results
//
// the daemon can be killed at any instant; on start every task left "running" is put back to
// "queued" with resume=true so the next attempt continues its pi session instead of starting over.

import { randomBytes } from "node:crypto";
import { appendFileSync, closeSync, existsSync, fsyncSync, mkdirSync, openSync, readdirSync, readFileSync, renameSync, writeSync } from "node:fs";
import { join } from "node:path";
import type { TierName } from "./config.ts";

export type TaskStatus = "queued" | "running" | "done" | "failed" | "cancelled";

export interface Attempt {
  n: number;
  tier: TierName;
  model: string;
  startedAt: string;
  endedAt?: string;
  outcome?: "done" | "blocked" | "error" | "timeout" | "verify-failed" | "interrupted" | "cancelled" | "backend-unavailable";
  detail?: string;
  sessionFile?: string;
  tokens?: { input: number; output: number };
  toolCalls?: number;
  turns?: number;
  seconds?: number;
}

export interface Task {
  id: string;
  prompt: string;
  cwd: string;
  status: TaskStatus;
  createdAt: string;
  updatedAt: string;
  /** forced starting tier; unset lets the router decide */
  tier?: TierName;
  /** shell command that must exit 0 for the task to count as done */
  verify?: string;
  timeoutMinutes?: number;
  priority: number;
  route?: { tier: TierName; score: number; source: string; reasons: string[] };
  attempts: Attempt[];
  resume?: boolean;
  result?: string;
  error?: string;
  source?: string;
}

export function newTaskId(): string {
  const t = new Date().toISOString().replace(/[-:T]/g, "").slice(0, 14);
  return `t${t}-${randomBytes(3).toString("hex")}`;
}

export function atomicWriteJson(path: string, value: unknown): void {
  const tmp = `${path}.${process.pid}.tmp`;
  const fd = openSync(tmp, "w", 0o600);
  try {
    writeSync(fd, JSON.stringify(value, null, 2) + "\n");
    fsyncSync(fd);
  } finally {
    closeSync(fd);
  }
  renameSync(tmp, path);
}

export class TaskStore {
  private readonly tasks = new Map<string, Task>();
  readonly dir: string;

  constructor(dir: string) {
    this.dir = dir;
    mkdirSync(dir, { recursive: true });
  }

  taskDir(id: string): string {
    return join(this.dir, id);
  }

  /** loads every task; tasks interrupted mid-run are requeued for resumption */
  load(): { loaded: number; requeued: string[] } {
    const requeued: string[] = [];
    for (const name of readdirSync(this.dir)) {
      const file = join(this.dir, name, "task.json");
      if (!existsSync(file)) continue;
      let t: Task;
      try {
        t = JSON.parse(readFileSync(file, "utf8")) as Task;
      } catch {
        continue; // a torn file cannot happen with atomic writes; skip anything foreign
      }
      if (t.status === "running") {
        const last = t.attempts.at(-1);
        if (last && !last.endedAt) {
          last.endedAt = new Date().toISOString();
          last.outcome = "interrupted";
          last.detail = "daemon restarted during the attempt";
        }
        t.status = "queued";
        t.resume = true;
        requeued.push(t.id);
        this.tasks.set(t.id, t);
        this.save(t);
        continue;
      }
      this.tasks.set(t.id, t);
    }
    return { loaded: this.tasks.size, requeued };
  }

  create(input: Pick<Task, "prompt" | "cwd"> & Partial<Pick<Task, "tier" | "verify" | "timeoutMinutes" | "priority" | "source">>): Task {
    const now = new Date().toISOString();
    const t: Task = {
      id: newTaskId(),
      prompt: input.prompt,
      cwd: input.cwd,
      status: "queued",
      createdAt: now,
      updatedAt: now,
      tier: input.tier,
      verify: input.verify,
      timeoutMinutes: input.timeoutMinutes,
      priority: input.priority ?? 0,
      attempts: [],
      source: input.source,
    };
    mkdirSync(join(this.taskDir(t.id), "sessions"), { recursive: true });
    this.tasks.set(t.id, t);
    this.save(t);
    return t;
  }

  save(t: Task): void {
    t.updatedAt = new Date().toISOString();
    mkdirSync(this.taskDir(t.id), { recursive: true });
    atomicWriteJson(join(this.taskDir(t.id), "task.json"), t);
  }

  get(id: string): Task | undefined {
    return this.tasks.get(id);
  }

  list(): Task[] {
    return [...this.tasks.values()].sort((a, b) => a.createdAt.localeCompare(b.createdAt));
  }

  /** next queued task: highest priority first, then oldest */
  nextQueued(exclude: Set<string>): Task | undefined {
    return this.list()
      .filter((t) => t.status === "queued" && !exclude.has(t.id))
      .sort((a, b) => b.priority - a.priority || a.createdAt.localeCompare(b.createdAt))[0];
  }

  counts(): Record<TaskStatus, number> {
    const c: Record<TaskStatus, number> = { queued: 0, running: 0, done: 0, failed: 0, cancelled: 0 };
    for (const t of this.tasks.values()) c[t.status]++;
    return c;
  }

  event(id: string, line: string): void {
    try {
      appendFileSync(join(this.taskDir(id), "events.log"), `${new Date().toISOString()} ${line}\n`);
    } catch {
      /* logging must never take a task down */
    }
  }
}

/** inbox files: optional header lines `cwd: ...`, `tier: ...`, `verify: ...`, `priority: ...`, then the prompt */
export function parseInboxFile(text: string): Record<string, string> {
  const out: Record<string, string> = {};
  const lines = text.split("\n");
  let i = 0;
  for (; i < lines.length; i++) {
    const m = lines[i].match(/^(cwd|tier|verify|priority|timeoutMinutes):\s*(.*)$/);
    if (!m) break;
    out[m[1]] = m[2].trim();
  }
  out.prompt = lines.slice(i).join("\n").trim();
  return out;
}
