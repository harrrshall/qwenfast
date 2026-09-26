// qfa: small client for the agent daemon.
//
//   qfa submit "fix the flaky test in tests/test_io.py" [--cwd DIR] [--tier auto|small|medium|large]
//              [--verify "pytest -q"] [--priority N] [--wait]
//   qfa ls [status]        qfa show ID        qfa events ID [N]
//   qfa cancel ID          qfa retry ID       qfa health
//   qfa route "prompt"     (dry run of the router's heuristic, no model call)

import { defaultConfig, type TierName } from "./config.ts";
import { tierModel } from "./models.ts";
import { heuristicScore, httpJudge, route, tierForScore } from "./router.ts";

const BASE = process.env.QFA_URL ?? `http://127.0.0.1:${process.env.QFA_PORT ?? 7788}`;

async function call(method: string, path: string, body?: unknown): Promise<unknown> {
  const r = await fetch(`${BASE}${path}`, {
    method,
    headers: body ? { "content-type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const text = await r.text();
  if (!r.ok) throw new Error(`${r.status}: ${text}`);
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

const BOOLEAN_FLAGS = new Set(["wait", "judge"]);

function flags(args: string[]): { pos: string[]; opt: Record<string, string | true> } {
  const pos: string[] = [];
  const opt: Record<string, string | true> = {};
  for (let i = 0; i < args.length; i++) {
    if (args[i].startsWith("--")) {
      const k = args[i].slice(2);
      const v = args[i + 1];
      if (BOOLEAN_FLAGS.has(k) || v === undefined || v.startsWith("--")) opt[k] = true;
      else {
        opt[k] = v;
        i++;
      }
    } else pos.push(args[i]);
  }
  return { pos, opt };
}

const out = (v: unknown) => console.log(typeof v === "string" ? v : JSON.stringify(v, null, 2));

async function main(): Promise<void> {
  const [cmd, ...rest] = process.argv.slice(2);
  const { pos, opt } = flags(rest);
  switch (cmd) {
    case "submit": {
      const t = (await call("POST", "/tasks", {
        prompt: pos.join(" "),
        cwd: opt.cwd ?? process.cwd(),
        tier: opt.tier,
        verify: opt.verify,
        priority: opt.priority,
        timeoutMinutes: opt.timeout,
      })) as { id: string };
      console.log(t.id);
      if (opt.wait) {
        for (;;) {
          await new Promise((r) => setTimeout(r, 3000));
          const s = (await call("GET", `/tasks/${t.id}`)) as { status: string; result?: string; error?: string };
          if (!["queued", "running"].includes(s.status)) {
            out(s.result ?? s.error ?? s.status);
            process.exitCode = s.status === "done" ? 0 : 1;
            break;
          }
        }
      }
      return;
    }
    case "ls":
      return out(await call("GET", `/tasks${pos[0] ? `?status=${pos[0]}` : ""}`));
    case "show":
      return out(await call("GET", `/tasks/${pos[0]}`));
    case "events":
      return out(await call("GET", `/tasks/${pos[0]}/events?tail=${pos[1] ?? 100}`));
    case "cancel":
      return out(await call("POST", `/tasks/${pos[0]}/cancel`));
    case "retry":
      return out(await call("POST", `/tasks/${pos[0]}/retry`));
    case "health":
      return out(await call("GET", "/health"));
    case "route": {
      // --judge asks the small model for borderline prompts, exactly as the daemon does
      // (needs QFA_SMALL_URL and QFA_API_KEY); without it only the lexical heuristic runs
      const prompt = opt.file ? (await import("node:fs")).readFileSync(String(opt.file), "utf8") : pos.join(" ");
      if (opt.judge) {
        const cfg = defaultConfig("/tmp");
        const judge = httpJudge(() => process.env.QFA_SMALL_URL, cfg.tiers.small.model, process.env.QFA_API_KEY ?? "");
        return out(await route(prompt, { judge }));
      }
      const { score, reasons } = heuristicScore(prompt);
      return out({ tier: tierForScore(score), score: Number(score.toFixed(2)), reasons });
    }
    case "models-json": {
      // the pi models.json for the qwenfast tiers: --big URL --small URL, key read from $QFA_PI_KEY
      const cfg = defaultConfig("/tmp");
      const urls: Record<string, string | undefined> = { big: opt.big as string, small: opt.small as string };
      const providers: Record<string, { baseUrl: string; api: string; apiKey: string; models: unknown[] }> = {};
      for (const [name, tier] of Object.entries(cfg.tiers) as Array<[TierName, (typeof cfg.tiers)[TierName]]>) {
        const url = urls[tier.endpoint];
        if (!url) continue;
        const m = tierModel(name, tier, url);
        const entry = (providers[m.provider] ??= { baseUrl: m.baseUrl, api: m.api, apiKey: "$QFA_PI_KEY", models: [] });
        if (!entry.models.some((x) => (x as { id: string }).id === m.id)) {
          const { provider: _p, baseUrl: _b, api: _a, ...rest } = m;
          entry.models.push({ ...rest, name: m.id });
        }
      }
      return out({ providers });
    }
    default:
      console.log("usage: qfa submit|ls|show|events|cancel|retry|health|route ... (see src/cli.ts)");
      process.exitCode = cmd ? 1 : 0;
  }
}

main().catch((err) => {
  console.error(String(err?.message ?? err));
  process.exitCode = 1;
});
