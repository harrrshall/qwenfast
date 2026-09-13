import { resolveBase } from "@/lib/upstream";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * The engine's request envelope, for the chat page.
 *
 * The page needs to know the context window and the message cap so it can keep a conversation
 * inside them by trimming its own history, rather than discovering them by being refused. It
 * cannot read `/v1/limits` itself — that endpoint needs a key, and the demo key lives only in
 * this function's environment — so this route is the keyed hop, exactly like `/api/chat`.
 *
 * Every field is a deployment constant, so it is cached for a minute: a page load during a
 * restart gets the fallback rather than a stall, and a config change is picked up within the
 * minute without a redeploy.
 */

const CACHE_MS = 60_000;

/**
 * What to assume when the engine will not say. Deliberately *conservative* — smaller than any
 * configuration this endpoint has ever run — because the cost of guessing too low is a slightly
 * shorter history, and the cost of guessing too high is the 400 this whole change exists to
 * remove.
 */
const FALLBACK_LIMITS = {
  max_context_length: 4096,
  max_prompt_tokens: 3072,
  default_max_tokens: 1024,
  max_output_tokens: 4096,
  min_completion_tokens: 256,
  max_messages: 128,
  max_tokens_is_clamped: true,
  stale: true,
};

type Cached = { at: number; body: Record<string, unknown> };
declare global {
  // eslint-disable-next-line no-var
  var __qwenfastLimits: Cached | undefined;
}

export async function GET(): Promise<Response> {
  const cached = globalThis.__qwenfastLimits;
  if (cached && Date.now() - cached.at < CACHE_MS) {
    return json(cached.body);
  }

  const base = await resolveBase();
  const key = process.env.QWENFAST_DEMO_KEY;
  if (!base || !key) return json(FALLBACK_LIMITS);

  try {
    const res = await fetch(`${base}/v1/limits`, {
      headers: { Authorization: `Bearer ${key}` },
      cache: "no-store",
      signal: AbortSignal.timeout(4_000),
    });
    if (!res.ok) return json(FALLBACK_LIMITS);
    const body = (await res.json()) as Record<string, unknown>;
    // An older engine without /v1/limits would 404 above; one *with* it but missing the field
    // we depend on is still unusable, so treat that as no answer too.
    if (typeof body.max_context_length !== "number") return json(FALLBACK_LIMITS);
    globalThis.__qwenfastLimits = { at: Date.now(), body };
    return json(body);
  } catch {
    return json(FALLBACK_LIMITS);
  }
}

function json(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
  });
}
