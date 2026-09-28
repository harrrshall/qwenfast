// runs one task to completion: route -> attempt (a pi agent session) -> verify -> escalate.
//
// an attempt ends in one of:
//   done               final message says STATUS: DONE (or no status) and `verify` passes
//   blocked            the model says it cannot finish          -> next tier
//   verify-failed      the verify command exited non zero        -> next tier, output fed back
//   error / timeout    provider error after pi's own retries, or the wall clock budget
//   backend-unavailable the endpoint was down; does not use up an attempt, waits and retries
//   interrupted        the daemon died; the next attempt continues the same pi session
//
// an escalated attempt starts a fresh session on the stronger model, told what the previous one
// did and that the working tree may hold its partial changes. a resumed attempt reopens the
// interrupted session file, so no work is repeated after a crash or restart.

import { spawn } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { AgentSession } from "@earendil-works/pi-coding-agent";
import { createAgentSession, DefaultResourceLoader, ModelRuntime, SessionManager, SettingsManager } from "@earendil-works/pi-coding-agent";
import type { Backend } from "./backend.ts";
import type { Config, EndpointName, TierName } from "./config.ts";
import { tierModel } from "./models.ts";
import { escalate, route, type Judge } from "./router.ts";
import type { Attempt, Task, TaskStore } from "./store.ts";

export const AUTONOMY_PROMPT = `You are running as an autonomous agent. No human is watching and nobody will answer questions.
- Never ask for confirmation or clarification: pick the most reasonable interpretation, state your assumption briefly, and continue.
- Work until the task is actually complete. Check your work with the tools (run the code, the tests, or a command that proves the result) before you finish.
- Keep tool output small (use head, tail, grep) so the context stays focused.
- End your final message with exactly one status line:
  STATUS: DONE
  or, only if the task truly cannot be completed in this environment:
  STATUS: BLOCKED: <one line reason>`;

const DEFAULT_TOOLS = ["read", "bash", "edit", "write", "grep", "find", "ls"];

export interface RunnerDeps {
  cfg: Config;
  store: TaskStore;
  backend: Backend;
  apiKey: string;
  judge?: Judge;
  log: (msg: string) => void;
}

export function parseStatus(text: string | undefined): { status: "done" | "blocked" | "none"; reason?: string } {
  if (!text) return { status: "none" };
  const lines = text.trim().split("\n").reverse().slice(0, 6);
  for (const l of lines) {
    const m = l.match(/STATUS:\s*(DONE|BLOCKED)(?::\s*(.*))?/i);
    if (m) return m[1].toUpperCase() === "DONE" ? { status: "done" } : { status: "blocked", reason: (m[2] ?? "").trim() };
  }
  return { status: "none" };
}

export function runShell(cmd: string, cwd: string, timeoutMs: number): Promise<{ code: number; output: string }> {
  return new Promise((resolve) => {
    const child = spawn("sh", ["-c", cmd], { cwd, env: process.env, detached: true });
    let out = "";
    const add = (b: Buffer) => {
      out += b.toString();
      if (out.length > 200_000) out = out.slice(-100_000);
    };
    child.stdout.on("data", add);
    child.stderr.on("data", add);
    const timer = setTimeout(() => {
      try {
        process.kill(-child.pid!, "SIGKILL");
      } catch {
        /* already gone */
      }
    }, timeoutMs);
    child.on("close", (code, signal) => {
      clearTimeout(timer);
      resolve({ code: code ?? (signal ? 124 : 1), output: out.slice(-4000) });
    });
    child.on("error", (err) => {
      clearTimeout(timer);
      resolve({ code: 127, output: String(err) });
    });
  });
}

export class Runner {
  private readonly aborts = new Map<string, AbortController>();

  private readonly d: RunnerDeps;

  constructor(d: RunnerDeps) {
    this.d = d;
  }

  /**
   * pi resolves credentials through its provider registry, so both servers are declared as
   * providers in our own agent dir's models.json (never the user's ~/.pi). the key is referenced
   * by env interpolation, not written to disk. rewritten when an endpoint url changes (a resumed
   * box gets new urls), and a fresh runtime is built per attempt so it always reads the current one.
   */
  private async modelRuntime(): Promise<ModelRuntime> {
    const { cfg, backend } = this.d;
    process.env.QFA_PI_KEY = this.d.apiKey;
    const providers: Record<string, unknown> = {};
    for (const [name, tier] of Object.entries(cfg.tiers) as Array<[TierName, (typeof cfg.tiers)[TierName]]>) {
      const url = backend.url(tier.endpoint);
      if (!url) continue;
      const m = tierModel(name, tier, url);
      const entry = (providers[m.provider] ??= { baseUrl: m.baseUrl, api: m.api, apiKey: "$QFA_PI_KEY", models: [] }) as { models: unknown[] };
      if (!entry.models.some((x) => (x as { id: string }).id === m.id)) {
        const { provider: _p, baseUrl: _b, api: _a, ...rest } = m;
        entry.models.push({ ...rest, name: m.id });
      }
    }
    const dir = join(cfg.home, "pi");
    mkdirSync(dir, { recursive: true });
    const text = JSON.stringify({ providers }, null, 2) + "\n";
    const file = join(dir, "models.json");
    if (!existsSync(file) || readFileSync(file, "utf8") !== text) writeFileSync(file, text);
    return ModelRuntime.create({ authPath: join(dir, "auth.json"), modelsPath: file });
  }

  cancel(id: string): boolean {
    const c = this.aborts.get(id);
    if (!c) return false;
    c.abort(new Error("cancelled"));
    return true;
  }

  /** the endpoint a tier should use right now, falling back to the other server if one is down */
  pickEndpoint(tier: TierName): { endpoint: EndpointName; tier: TierName } {
    const want = this.d.cfg.tiers[tier].endpoint;
    if (this.d.backend.healthy(want)) return { endpoint: want, tier };
    if (want === "small" && this.d.backend.healthy("big")) return { endpoint: "big", tier: "medium" };
    return { endpoint: want, tier };
  }

  async run(task: Task): Promise<void> {
    const { store, cfg, log } = this.d;
    const ctrl = new AbortController();
    this.aborts.set(task.id, ctrl);
    try {
      task.status = "running";
      store.save(task);

      if (!task.route) {
        const r = await route(task.prompt, { forced: task.tier, judge: this.d.backend.healthy("small") ? this.d.judge : undefined });
        task.route = r;
        store.save(task);
        store.event(task.id, `route ${r.tier} (${r.source}, score ${Number.isNaN(r.score) ? "-" : r.score.toFixed(2)}): ${r.reasons.join(" ")}`);
        log(`${task.id} routed to ${r.tier} via ${r.source}`);
      }

      let failures = task.attempts.filter((a) => a.outcome && !["backend-unavailable", "interrupted"].includes(a.outcome)).length;
      let backendMisses = 0;
      let feedback: string | undefined;

      while (!ctrl.signal.aborted) {
        const last = task.attempts.at(-1);
        const resuming = !!task.resume && !!last?.sessionFile && existsSync(last.sessionFile);
        let tier: TierName;
        if (resuming || last?.outcome === "backend-unavailable" || last?.outcome === "interrupted") tier = last!.tier;
        else if (last) tier = escalate(last.tier);
        else tier = task.route!.tier;

        if (failures >= cfg.task.maxAttempts) {
          task.status = "failed";
          task.error = `gave up after ${failures} failed attempts; last: ${last?.outcome} ${last?.detail ?? ""}`.trim();
          break;
        }

        // wait for a server that can serve this tier
        let pick = this.pickEndpoint(tier);
        if (!this.d.backend.healthy(pick.endpoint)) {
          store.event(task.id, `waiting for ${pick.endpoint} endpoint`);
          const ok = await this.d.backend.waitFor(pick.endpoint, cfg.task.backendWaitMinutes * 60000, ctrl.signal);
          if (ctrl.signal.aborted) break;
          if (!ok) {
            backendMisses++;
            task.attempts.push({
              n: task.attempts.length + 1, tier, model: cfg.tiers[tier].model, startedAt: new Date().toISOString(),
              endedAt: new Date().toISOString(), outcome: "backend-unavailable", detail: `${pick.endpoint} down for ${cfg.task.backendWaitMinutes} min`,
            });
            store.save(task);
            if (backendMisses >= 48) {
              task.status = "failed";
              task.error = "backend unavailable for too long";
              break;
            }
            continue;
          }
          pick = this.pickEndpoint(tier);
        }
        tier = pick.tier;

        const attempt = await this.attempt(task, tier, pick.endpoint, resuming ? last!.sessionFile : undefined, feedback, ctrl.signal);
        task.resume = false;
        store.save(task);

        if (attempt.outcome === "done") {
          task.status = "done";
          task.error = undefined;
          break;
        }
        if (attempt.outcome === "cancelled") break;
        if (attempt.outcome === "backend-unavailable") {
          backendMisses++;
          continue;
        }
        failures++;
        feedback = `A previous attempt (${attempt.tier} tier, ${attempt.model}) did not finish: ${attempt.outcome}${attempt.detail ? ` - ${attempt.detail}` : ""}.`;
        if (task.result) feedback += `\nIts last message was:\n${task.result.slice(-2500)}`;
        feedback += "\nThe working directory may contain its partial changes; inspect them (git status, git diff, ls) before continuing, and finish the task properly.";
        log(`${task.id} attempt ${attempt.n} ${attempt.outcome} on ${attempt.tier}; escalating`);
      }
      if (ctrl.signal.aborted && task.status === "running") task.status = "cancelled";
    } catch (err) {
      task.status = "failed";
      task.error = `runner crashed: ${String((err as Error)?.stack ?? err).slice(0, 800)}`;
      log(`${task.id} runner error: ${task.error}`);
    } finally {
      this.aborts.delete(task.id);
      store.save(task);
      store.event(task.id, `final status ${task.status}${task.error ? `: ${task.error}` : ""}`);
      log(`${task.id} -> ${task.status}`);
    }
  }

  private async attempt(
    task: Task,
    tier: TierName,
    endpoint: EndpointName,
    resumeFile: string | undefined,
    feedback: string | undefined,
    signal: AbortSignal,
  ): Promise<Attempt> {
    const { cfg, store, backend } = this.d;
    const tcfg = { ...cfg.tiers[tier], endpoint };
    const model = tierModel(tier, tcfg, backend.url(endpoint)!);
    const a: Attempt = { n: task.attempts.length + 1, tier, model: tcfg.model, startedAt: new Date().toISOString() };
    task.attempts.push(a);
    store.save(task);
    store.event(task.id, `attempt ${a.n}: ${tier} tier on ${endpoint} (${tcfg.model}, thinking ${tcfg.thinking})${resumeFile ? " resuming session" : ""}`);

    const t0 = Date.now();
    const sessionsDir = join(store.taskDir(task.id), "sessions");
    const agentDir = join(cfg.home, "pi");
    let session: AgentSession | undefined;
    let turns = 0;
    let toolCalls = 0;
    let timedOut = false;
    let turnLimit = false;
    const onAbort = () => void session?.abort();
    signal.addEventListener("abort", onAbort);
    const timeoutMs = (task.timeoutMinutes ?? cfg.task.timeoutMinutes) * 60000;
    const timer = setTimeout(() => {
      timedOut = true;
      void session?.abort();
    }, timeoutMs);

    try {
      const settingsManager = SettingsManager.inMemory({
        compaction: { enabled: true, reserveTokens: 24000, keepRecentTokens: 24000 },
        retry: { enabled: true, maxRetries: 8, baseDelayMs: 3000, maxAgentDelayMs: 300_000, provider: { timeoutMs: 1_800_000, maxRetries: 3 } },
      });
      const resourceLoader = new DefaultResourceLoader({
        cwd: task.cwd,
        agentDir,
        settingsManager,
        appendSystemPrompt: [AUTONOMY_PROMPT],
      });
      await resourceLoader.reload();
      const sessionManager = resumeFile ? SessionManager.open(resumeFile, sessionsDir, task.cwd) : SessionManager.create(task.cwd, sessionsDir);
      const modelRuntime = await this.modelRuntime();
      const created = await createAgentSession({
        cwd: task.cwd,
        agentDir,
        model: modelRuntime.getModel(model.provider, model.id) ?? model,
        thinkingLevel: tcfg.thinking,
        modelRuntime,
        resourceLoader,
        sessionManager,
        settingsManager,
        tools: DEFAULT_TOOLS,
      });
      session = created.session;
      a.sessionFile = session.sessionFile;
      store.save(task);

      session.subscribe((ev) => {
        switch (ev.type) {
          case "tool_execution_start":
            toolCalls++;
            store.event(task.id, `tool ${ev.toolName} ${JSON.stringify(ev.args).slice(0, 300)}`);
            break;
          case "tool_execution_end":
            if (ev.isError) store.event(task.id, `tool ${ev.toolName} error ${JSON.stringify(ev.result).slice(0, 300)}`);
            break;
          case "turn_end":
            turns++;
            backend.noteDemand();
            if (turns >= cfg.task.maxTurns && !turnLimit) {
              turnLimit = true;
              void session?.abort();
            }
            break;
          case "auto_retry_start":
            store.event(task.id, `retry ${ev.attempt}/${ev.maxAttempts} in ${ev.delayMs} ms: ${ev.errorMessage.slice(0, 200)}`);
            break;
          case "compaction_end":
            store.event(task.id, `compaction (${ev.reason})${ev.errorMessage ? ` error ${ev.errorMessage}` : ""}`);
            break;
          case "message_end": {
            const m = ev.message as { role?: string; content?: Array<{ type: string; text?: string }> };
            if (m.role === "assistant") {
              const text = (m.content ?? []).filter((c) => c.type === "text").map((c) => c.text).join("").trim();
              if (text) store.event(task.id, `assistant ${text.slice(0, 400).replace(/\n/g, " ")}`);
            }
            break;
          }
        }
      });

      const prompt = resumeFile
        ? "You were interrupted (the host restarted). Continue the task from where you left off; check the current state of the files first."
        : feedback
          ? `${task.prompt}\n\n---\n${feedback}`
          : task.prompt;
      await session.prompt(prompt);

      const lastMsg = [...session.messages].reverse().find((m) => (m as { role?: string }).role === "assistant") as
        | { stopReason?: string; errorMessage?: string }
        | undefined;
      const text = session.getLastAssistantText();
      task.result = text;
      const stats = session.getSessionStats();
      a.tokens = { input: stats.tokens.input, output: stats.tokens.output };

      if (signal.aborted) {
        a.outcome = "cancelled";
      } else if (timedOut) {
        a.outcome = "timeout";
        a.detail = `wall clock ${Math.round(timeoutMs / 60000)} min`;
      } else if (turnLimit) {
        a.outcome = "error";
        a.detail = `turn limit ${cfg.task.maxTurns}`;
      } else if (lastMsg?.stopReason === "error") {
        a.outcome = backend.healthy(endpoint) ? "error" : "backend-unavailable";
        a.detail = (lastMsg.errorMessage ?? "provider error").slice(0, 400);
      } else {
        const st = parseStatus(text);
        if (st.status === "blocked") {
          a.outcome = "blocked";
          a.detail = st.reason;
        } else if (task.verify) {
          const v = await runShell(task.verify, task.cwd, 20 * 60000);
          store.event(task.id, `verify exit ${v.code}`);
          if (v.code === 0) a.outcome = "done";
          else {
            a.outcome = "verify-failed";
            a.detail = `\`${task.verify}\` exited ${v.code}:\n${v.output.slice(-2000)}`;
          }
        } else {
          a.outcome = "done";
          if (st.status === "none") a.detail = "no status line";
        }
      }
    } catch (err) {
      a.outcome = signal.aborted ? "cancelled" : backend.healthy(endpoint) ? "error" : "backend-unavailable";
      a.detail = String((err as Error)?.message ?? err).slice(0, 400);
    } finally {
      clearTimeout(timer);
      signal.removeEventListener("abort", onAbort);
      try {
        session?.dispose();
      } catch {
        /* disposal errors are irrelevant after the attempt */
      }
      a.endedAt = new Date().toISOString();
      a.seconds = Math.round((Date.now() - t0) / 1000);
      a.turns = turns;
      a.toolCalls = toolCalls;
      store.save(task);
      store.event(task.id, `attempt ${a.n} ${a.outcome}${a.detail ? `: ${a.detail.slice(0, 300)}` : ""} (${a.seconds}s, ${turns} turns, ${toolCalls} tools)`);
    }
    return a;
  }
}

