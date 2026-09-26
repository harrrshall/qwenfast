// openai compatible gateway on the daemon (127.0.0.1:7788/v1): the one endpoint every local client
// (qwen fast code, pi, scripts) talks to. it
//
//   * exposes the tiers as models, plus `auto`, which routes each conversation by complexity;
//   * maps a tier to the right server, model name and thinking settings;
//   * sends traffic through the ssh tunnel when it is up (the https proxy cuts streams at ~2 min);
//   * wakes a paused gpu box and holds requests (sse keep-alive comments) until it answers;
//   * counts its in-flight requests as demand, so the idle pause never fires under a live session.
//
// routing for `auto` is sticky per conversation (keyed by its system prompt and first user
// message). every new user message is routed; the conversation keeps the highest tier it has
// needed so far, so a follow-up never lands on a weaker model mid-task. a turn that keeps looping on
// the small tier (many tool round trips for one user message) is promoted to medium.

import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { atomicWriteJson } from "./store.ts";
import type { IncomingMessage, ServerResponse } from "node:http";
import type { Backend } from "./backend.ts";
import type { Config, TierName } from "./config.ts";
import { LADDER, route, type Judge, type RouteDecision } from "./router.ts";

export const GATEWAY_MODELS: Array<{ id: string; tier: TierName | "auto"; description: string }> = [
  { id: "auto", tier: "auto", description: "routes each conversation to the right tier by complexity" },
  { id: "qwen3.8-27b-xhigh", tier: "large", description: "qwen3.8-27b, thinking at xhigh effort: hard debugging, design, refactors" },
  { id: "qwen3.8-27b", tier: "medium", description: "qwen3.8-27b, thinking at medium effort: everyday coding" },
  { id: "qwen3.6-35b-a3b", tier: "small", description: "qwen3.6-35b-a3b (3b active), no thinking: fast chores, titles, summaries" },
];

const EFFORT: Record<TierName, { enable_thinking: boolean; reasoning_effort?: string }> = {
  small: { enable_thinking: false },
  medium: { enable_thinking: true, reasoning_effort: "medium" },
  large: { enable_thinking: true, reasoning_effort: "xhigh" },
};

/** tool round trips within one user turn on the small tier before it is promoted */
const SMALL_TURN_LIMIT = 24;

interface Conversation {
  tier: TierName;
  lastUserHash: string;
  turnRequests: number;
  updated: number;
  decisions: RouteDecision[];
}

type Msg = { role?: string; content?: unknown };

export function textOf(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((p) => (p && typeof p === "object" && (p as { type?: string }).type === "text" ? String((p as { text?: string }).text ?? "") : ""))
      .join("");
  }
  return "";
}

/** the text a person typed: system reminders and attached context blocks are not the task */
export function userIntent(text: string): string {
  return text
    .replace(/<system-reminder>[\s\S]*?<\/system-reminder>/g, " ")
    .replace(/<(env|environment|context|file|files)[^>]*>[\s\S]*?<\/\1>/g, " ")
    .trim();
}

export function conversationKey(messages: Msg[]): string {
  const system = messages.find((m) => m.role === "system");
  const firstUser = messages.find((m) => m.role === "user");
  return createHash("sha1")
    .update(textOf(system?.content).slice(0, 4000))
    .update("\0")
    .update(textOf(firstUser?.content))
    .digest("hex")
    .slice(0, 16);
}

function higher(a: TierName, b: TierName): TierName {
  return LADDER.indexOf(a) >= LADDER.indexOf(b) ? a : b;
}

export class Gateway {
  inflight = 0;
  private readonly conversations = new Map<string, Conversation>();
  private readonly cfg: Config;
  private readonly backend: Backend;
  private readonly judge?: Judge;
  private readonly apiKey: string;
  private readonly log: (m: string) => void;
  readonly stats = { requests: 0, byTier: { small: 0, medium: 0, large: 0 } as Record<TierName, number>, errors: 0 };

  private saveTimer?: NodeJS.Timeout;
  private readonly stateFile: string;

  constructor(cfg: Config, backend: Backend, apiKey: string, log: (m: string) => void, judge?: Judge) {
    this.cfg = cfg;
    this.backend = backend;
    this.apiKey = apiKey;
    this.log = log;
    this.judge = judge;
    // routing state survives daemon restarts, so a long session never drops a tier mid task
    this.stateFile = join(cfg.home, "gateway.json");
    try {
      if (existsSync(this.stateFile)) {
        const saved = JSON.parse(readFileSync(this.stateFile, "utf8")) as { conversations?: Array<[string, Conversation]>; stats?: Gateway["stats"] };
        for (const [k, v] of saved.conversations ?? []) this.conversations.set(k, v);
        if (saved.stats) Object.assign(this.stats, saved.stats, { byTier: { ...this.stats.byTier, ...saved.stats.byTier } });
      }
    } catch (err) {
      log(`gateway state unreadable, starting fresh: ${String(err)}`);
    }
  }

  private save(): void {
    if (this.saveTimer) return;
    this.saveTimer = setTimeout(() => {
      this.saveTimer = undefined;
      try {
        atomicWriteJson(this.stateFile, { conversations: [...this.conversations.entries()], stats: this.stats });
      } catch (err) {
        this.log(`gateway state not saved: ${String(err)}`);
      }
    }, 2000);
    this.saveTimer.unref();
  }

  /** decide the tier for one request; exported for tests through `pick` */
  async pick(model: string, messages: Msg[]): Promise<{ tier: TierName; key?: string; decision?: RouteDecision }> {
    const fixed = GATEWAY_MODELS.find((m) => m.id === model || `qwenfast/${m.id}` === model);
    if (fixed && fixed.tier !== "auto") return { tier: fixed.tier };

    const key = conversationKey(messages);
    let conv = this.conversations.get(key);
    const lastUser = [...messages].reverse().find((m) => m.role === "user");
    const last = messages.at(-1);
    const intent = userIntent(textOf(lastUser?.content));
    const userHash = createHash("sha1").update(intent).digest("hex").slice(0, 12);
    const newTurn = last?.role === "user" && (!conv || conv.lastUserHash !== userHash);

    if (!conv || newTurn) {
      const decision = await route(intent || textOf(lastUser?.content), {
        judge: this.backend.healthy("small") ? this.judge : undefined,
      });
      const tier = conv ? higher(conv.tier, decision.tier) : decision.tier;
      conv = { tier, lastUserHash: userHash, turnRequests: 0, updated: Date.now(), decisions: [...(conv?.decisions ?? []), decision].slice(-20) };
      this.conversations.set(key, conv);
      this.log(`gateway ${key}: ${newTurn && conv.decisions.length > 1 ? "turn" : "new"} -> ${tier} (${decision.source}: ${decision.reasons.join(" ")})`);
    }
    conv.turnRequests++;
    conv.updated = Date.now();
    if (conv.tier === "small" && conv.turnRequests > SMALL_TURN_LIMIT) {
      conv.tier = "medium";
      this.log(`gateway ${key}: ${conv.turnRequests} round trips on small in one turn -> medium`);
    }
    this.prune();
    this.save();
    return { tier: conv.tier, key, decision: conv.decisions.at(-1) };
  }

  private prune(): void {
    if (this.conversations.size < 2000) return;
    const cutoff = Date.now() - 7 * 24 * 3600_000;
    for (const [k, v] of this.conversations) if (v.updated < cutoff) this.conversations.delete(k);
  }

  snapshot() {
    return {
      ...this.stats,
      inflight: this.inflight,
      conversations: [...this.conversations.entries()]
        .sort((a, b) => b[1].updated - a[1].updated)
        .slice(0, 20)
        .map(([key, c]) => ({ key, tier: c.tier, turnRequests: c.turnRequests, updated: new Date(c.updated).toISOString() })),
    };
  }

  models() {
    return {
      object: "list",
      data: GATEWAY_MODELS.map((m) => ({ id: m.id, object: "model", created: 0, owned_by: "qwenfast", description: m.description })),
    };
  }

  /** upstream body for a tier: the server's model name, qwen thinking kwargs, nothing the server rejects */
  upstreamBody(body: Record<string, unknown>, tier: TierName): Record<string, unknown> {
    const t = this.cfg.tiers[tier];
    const out: Record<string, unknown> = { ...body, model: t.model };
    delete out.reasoning_effort; // qwenfast accepts only its own three levels; the tier decides
    delete out.reasoning;
    const kwargs = { ...(body.chat_template_kwargs as Record<string, unknown> | undefined) };
    const effort = EFFORT[tier];
    kwargs.enable_thinking ??= effort.enable_thinking;
    if (t.endpoint === "big" && kwargs.enable_thinking && !kwargs.reasoning_effort && effort.reasoning_effort) {
      kwargs.reasoning_effort = effort.reasoning_effort;
    }
    if (t.endpoint === "small") delete kwargs.reasoning_effort;
    out.chat_template_kwargs = kwargs;
    if (typeof out.max_tokens === "number") out.max_tokens = Math.min(out.max_tokens, t.maxTokens);
    return out;
  }

  async handle(req: IncomingMessage, res: ServerResponse, path: string): Promise<void> {
    if (req.method === "GET" && path === "/v1/models") {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify(this.models()));
      return;
    }
    if (req.method !== "POST" || path !== "/v1/chat/completions") {
      res.writeHead(404, { "content-type": "application/json" });
      res.end(JSON.stringify({ error: { message: `unsupported: ${req.method} ${path}`, type: "not_found" } }));
      return;
    }
    const chunks: Buffer[] = [];
    for await (const c of req) chunks.push(c as Buffer);
    let body: Record<string, unknown>;
    try {
      body = JSON.parse(Buffer.concat(chunks).toString() || "{}");
    } catch {
      res.writeHead(400, { "content-type": "application/json" });
      res.end(JSON.stringify({ error: { message: "invalid json", type: "invalid_request_error" } }));
      return;
    }

    this.inflight++;
    this.backend.noteDemand();
    const abort = new AbortController();
    res.on("close", () => abort.abort());
    const stream = body.stream === true;
    let keepAlive: NodeJS.Timeout | undefined;
    try {
      const messages = (body.messages as Msg[]) ?? [];
      let { tier } = await this.pick(String(body.model ?? "auto"), messages);
      let endpoint = this.cfg.tiers[tier].endpoint;
      if (!this.backend.healthy(endpoint) && endpoint === "small" && this.backend.healthy("big")) {
        tier = "medium";
        endpoint = "big";
      }
      if (!this.backend.healthy(endpoint)) {
        // the box is paused, resuming or restarting: keep the client connected while it comes up
        if (stream) {
          res.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-cache", connection: "keep-alive" });
          res.write(": qwenfast: waiting for the gpu box\n\n");
          keepAlive = setInterval(() => res.write(": waiting\n\n"), 10_000);
        }
        const ok = await this.backend.waitFor(endpoint, this.cfg.task.backendWaitMinutes * 60_000, abort.signal);
        if (!ok) throw new Error(`the ${endpoint} model server did not come up in ${this.cfg.task.backendWaitMinutes} minutes`);
      }
      if (keepAlive) clearInterval(keepAlive);
      this.stats.requests++;
      this.stats.byTier[tier]++;
      this.save();

      const base = this.backend.url(endpoint)!.replace(/\/+$/, "");
      const upstream = await fetch(`${base}/v1/chat/completions`, {
        method: "POST",
        signal: abort.signal,
        headers: { "content-type": "application/json", authorization: `Bearer ${this.apiKey}` },
        body: JSON.stringify(this.upstreamBody(body, tier)),
      });
      if (!res.headersSent) {
        res.writeHead(upstream.status, {
          "content-type": upstream.headers.get("content-type") ?? "application/json",
          "cache-control": "no-cache",
          "x-qwenfast-tier": tier,
        });
      } else if (!upstream.ok) {
        const text = await upstream.text();
        res.write(`data: ${JSON.stringify({ error: { message: text.slice(0, 500), code: upstream.status } })}\n\n`);
        res.end();
        return;
      }
      if (upstream.body) {
        for await (const chunk of upstream.body as unknown as AsyncIterable<Uint8Array>) {
          if (!res.write(chunk)) await new Promise((r) => res.once("drain", r));
        }
      }
      res.end();
    } catch (err) {
      if (abort.signal.aborted) return;
      this.stats.errors++;
      const message = String((err as Error)?.message ?? err);
      this.log(`gateway error: ${message}`);
      if (!res.headersSent) {
        res.writeHead(503, { "content-type": "application/json" });
        res.end(JSON.stringify({ error: { message, type: "service_unavailable" } }));
      } else {
        res.write(`data: ${JSON.stringify({ error: { message, type: "service_unavailable" } })}\n\n`);
        res.end();
      }
    } finally {
      if (keepAlive) clearInterval(keepAlive);
      this.inflight--;
      this.backend.noteDemand();
    }
  }
}
