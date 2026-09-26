// complexity router: picks the cheapest tier that is likely to finish the task, and the ladder a
// failed attempt climbs.
//
//   small   qwen3.6-35b-a3b, thinking off   lookups, one-file edits, summaries, shell chores
//   medium  qwen3.8-27b, effort medium      ordinary coding, multi-step changes with tests
//   large   qwen3.8-27b, effort xhigh       debugging, design, refactors, anything already failed
//
// two stages. a lexical score in [0, 1] decides the clear cases for free; for the unsure middle
// band the small model grades the task in one short non-thinking call (about 100 ms on the box).
// the judge can only move a task by one tier from the heuristic, so a confused judge can never
// send an obviously hard task to the small model or the reverse.

import type { TierName } from "./config.ts";

export const LADDER: TierName[] = ["small", "medium", "large"];

export interface RouteDecision {
  tier: TierName;
  score: number;
  source: "forced" | "heuristic" | "judge" | "judge-failed" | "escalation";
  reasons: string[];
}

const HARD: Array<[RegExp, number, string]> = [
  [/\b(architect|architecture|design (a|an|the)|system design)\b/i, 0.35, "design"],
  [/\b(refactor|restructure|rewrite|migrat(e|ion)|port (it|this|the)|overhaul)\b/i, 0.3, "refactor"],
  [/\b(debug|root cause|race condition|deadlock|flaky|intermittent|memory leak|segfault|heisenbug)\b/i, 0.35, "debugging"],
  [/\b(concurren|parallel|distributed|lock-?free|consensus|transaction)/i, 0.2, "concurrency"],
  [/\b(optimi[sz]e|performance|latency|throughput|profil(e|ing)|benchmark)\b/i, 0.2, "performance"],
  [/\b(security|vulnerab|exploit|auth(entication|orization)|crypto)/i, 0.2, "security"],
  [/\b(prove|proof|algorithm|complexity|invariant|formal|theorem)\b/i, 0.25, "reasoning"],
  [/\b(implement|build|create|develop) (a |an |the )?(new )?(feature|service|system|library|engine|compiler|parser|server|api|app)\b/i, 0.3, "build"],
  [/\b(across|throughout) (the )?(code ?base|repo|project|files)\b/i, 0.25, "multi-file"],
  [/\b(failing tests?|tests? (are|is) failing|make (the )?tests? pass|ci is red)\b/i, 0.2, "failing tests"],
  [/\b(kernel|cuda|triton|compiler|interpreter|jit|scheduler|allocator)\b/i, 0.2, "systems"],
  [/\b(investigate|figure out why|why does|analy[sz]e)\b/i, 0.15, "investigation"],
  [/\b(bugs?|broken|crash(es|ing)?|regression|misbehav\w*|wrong results?)\b/i, 0.15, "bug fixing"],
  [/\b(edge cases?|corner cases?|deterministic|thread.?safe)\b/i, 0.1, "rigor"],
  [/\b(plan|step by step|end to end|end-to-end|autonomous(ly)?)\b/i, 0.1, "long horizon"],
];

const EASY: Array<[RegExp, number, string]> = [
  [/\b(rename|typo|spelling|reformat|format (the|this)|lint fix)\b/i, 0.3, "cosmetic"],
  [/^\s*(list|show|print|cat|count|what is|what's|where is|which|how many|find)\b/i, 0.3, "lookup"],
  [/\b(summari[sz]e|tl;?dr|explain briefly|one sentence|short answer)\b/i, 0.25, "summary"],
  [/\b(add a (comment|docstring|log line)|bump (the )?version|update (the )?readme)\b/i, 0.3, "small edit"],
  [/\b(convert|translate) (this|the) (json|yaml|csv|string|date)\b/i, 0.2, "conversion"],
  [/\b(git (status|log|diff)|ls |pwd|disk usage|du -sh)\b/i, 0.2, "shell chore"],
];

/** lexical complexity score in [0, 1]; 0.5 is "no idea" */
export function heuristicScore(prompt: string): { score: number; reasons: string[] } {
  const reasons: string[] = [];
  let s = 0.5;
  for (const [re, w, why] of HARD) if (re.test(prompt)) { s += w; reasons.push(`+${why}`); }
  for (const [re, w, why] of EASY) if (re.test(prompt)) { s -= w; reasons.push(`-${why}`); }
  const words = prompt.trim().split(/\s+/).filter(Boolean).length;
  if (words < 15) { s -= 0.15; reasons.push("-short"); }
  else if (words > 250) { s += 0.2; reasons.push("+long spec"); }
  else if (words > 80) { s += 0.1; reasons.push("+detailed"); }
  const files = new Set(prompt.match(/[\w./-]+\.(ts|tsx|js|py|rs|go|java|c|cc|cpp|h|cu|md|json|yaml|toml|sh)\b/g) ?? []);
  if (files.size >= 4) { s += 0.2; reasons.push(`+${files.size} files`); }
  else if (files.size === 1 && words < 60) { s -= 0.05; reasons.push("-single file"); }
  const bullets = (prompt.match(/^\s*([-*]|\d+[.)])\s+/gm) ?? []).length;
  if (bullets >= 5) { s += 0.15; reasons.push(`+${bullets} steps`); }
  if (/```/.test(prompt) && words > 120) { s += 0.05; reasons.push("+code context"); }
  return { score: Math.max(0, Math.min(1, s)), reasons };
}

export function tierForScore(score: number): TierName {
  if (score < 0.35) return "small";
  if (score < 0.72) return "medium";
  return "large";
}

/** the heuristic is trusted outright away from the tier boundaries */
export function heuristicIsConfident(score: number): boolean {
  return score <= 0.2 || score >= 0.85 || (score >= 0.45 && score <= 0.62);
}

export type Judge = (prompt: string) => Promise<TierName>;

export async function route(prompt: string, opts: { forced?: TierName; judge?: Judge }): Promise<RouteDecision> {
  if (opts.forced) return { tier: opts.forced, score: NaN, source: "forced", reasons: ["forced by task"] };
  const { score, reasons } = heuristicScore(prompt);
  const guess = tierForScore(score);
  // a score built only from length features is a default, not evidence: let the judge decide
  const lexical = reasons.some((r) => !/^[+-](short|detailed|long spec|\d+ files|\d+ steps|code context|single file)$/.test(r));
  if (!opts.judge || (lexical && heuristicIsConfident(score))) return { tier: guess, score, source: "heuristic", reasons };
  try {
    const judged = await opts.judge(prompt);
    const gi = LADDER.indexOf(guess);
    const ji = LADDER.indexOf(judged);
    const clamped = LADDER[Math.max(gi - 1, Math.min(gi + 1, ji))];
    return { tier: clamped, score, source: "judge", reasons: [...reasons, `judge=${judged}`] };
  } catch (err) {
    return { tier: guess, score, source: "judge-failed", reasons: [...reasons, `judge error: ${String(err).slice(0, 80)}`] };
  }
}

/** next tier after a failed attempt; the top tier retries itself */
export function escalate(tier: TierName): TierName {
  const i = LADDER.indexOf(tier);
  return LADDER[Math.min(i + 1, LADDER.length - 1)];
}

export const JUDGE_SYSTEM = `You grade how hard a task is for an autonomous coding agent.
Reply with exactly one word:
SMALL  - a lookup, a one-file edit, a summary, a shell chore; a fast model finishes it in a few steps.
MEDIUM - ordinary coding: a feature or fix touching a few files, running tests, some reasoning.
LARGE  - debugging an unknown failure, design or architecture, a refactor across many files, performance work, tricky algorithms, or anything needing careful long reasoning.`;

export function parseJudgeReply(text: string): TierName {
  const m = text.toUpperCase().match(/\b(SMALL|MEDIUM|LARGE)\b/);
  if (!m) throw new Error(`unparseable judge reply: ${text.slice(0, 60)}`);
  return m[1].toLowerCase() as TierName;
}

/** judge backed by any openai compatible chat endpoint (the small tier by default) */
export function httpJudge(getBase: () => string | undefined, model: string, apiKey: string, timeoutMs = 20000): Judge {
  return async (prompt: string) => {
    const base = getBase();
    if (!base) throw new Error("judge endpoint unavailable");
    const ctrl = AbortSignal.timeout(timeoutMs);
    const r = await fetch(`${base.replace(/\/+$/, "")}/v1/chat/completions`, {
      method: "POST",
      signal: ctrl,
      headers: { "content-type": "application/json", authorization: `Bearer ${apiKey}` },
      body: JSON.stringify({
        model,
        messages: [
          { role: "system", content: JUDGE_SYSTEM },
          { role: "user", content: `Task:\n"""\n${prompt.slice(0, 6000)}\n"""\nOne word:` },
        ],
        max_tokens: 4,
        temperature: 0,
        chat_template_kwargs: { enable_thinking: false },
      }),
    });
    if (!r.ok) throw new Error(`judge http ${r.status}`);
    const body = (await r.json()) as { choices?: Array<{ message?: { content?: string } }> };
    return parseJudgeReply(body.choices?.[0]?.message?.content ?? "");
  };
}
