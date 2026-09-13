import { MODEL, resolveBase, invalidateBase, warmUpstream } from "@/lib/upstream";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 300;

type Role = "user" | "assistant" | "system";
type Msg = { role: Role; content: string };

function fail(message: string, status: number, offline = false, code?: string): Response {
  return new Response(JSON.stringify({ error: message, offline, code: code ?? null }), {
    status,
    headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
  });
}

/**
 * Pull the engine's own explanation out of an upstream failure.
 *
 * The engine answers every refusal in OpenAI's envelope
 * (`{"error": {message, type, param, code}}`), and that message is the useful one — "Your
 * messages came to 5,102 tokens, which leaves 0 for a reply" tells the user what to do, where
 * "Engine returned 400." tells them nothing. Falls back to the generic line only when the body
 * is not what we expect.
 */
async function upstreamError(
  res: Response,
  fallback: string,
): Promise<{ message: string; code: string | null }> {
  try {
    const j = await res.json();
    const err = j?.error;
    if (err && typeof err.message === "string" && err.message) {
      return { message: err.message, code: typeof err.code === "string" ? err.code : null };
    }
  } catch {
    /* not JSON, or already consumed */
  }
  return { message: fallback, code: null };
}

/**
 * "Offline" is a distinct, expected state, not an error: while a benchmark owns
 * the GPU nothing is bound to the engine's port and Cloudflare answers 502 for
 * the origin. The UI says so plainly and keeps polling instead of showing a
 * scary status code.
 */
const OFFLINE_LINE = "Engine offline — still warming up.";

export async function POST(req: Request): Promise<Response> {
  let body: { messages?: Msg[]; temperature?: number };
  try {
    body = await req.json();
  } catch {
    return fail("Malformed request.", 400);
  }

  const messages = (body.messages ?? [])
    .filter((m) => m && typeof m.content === "string" && m.content.length > 0)
    .map((m) => ({ role: m.role, content: m.content }));

  if (messages.length === 0) return fail("No message to send.", 400);

  const base = await resolveBase();
  if (!base) return fail(OFFLINE_LINE, 503, true);

  const payload = {
    model: MODEL,
    messages,
    stream: true,
    stream_options: { include_usage: true },
    max_tokens: 1024,
    // Greedy by default: the engine's speculative decoding (MTP, k=3) runs only
    // for greedy requests, and greedy is what the head-to-head benchmarks measure.
    temperature: typeof body.temperature === "number" ? body.temperature : 0,
    chat_template_kwargs: { enable_thinking: false },
  };

  // The public endpoint requires a key; the demo page must keep working without one, so the
  // proxy presents a dedicated `demo` key from a Vercel env var. That key is metered separately
  // in /admin/usage, which keeps "what the demo page cost" and "what the public API cost"
  // distinguishable — they are billed to the same GPU either way.
  const upstreamHeaders: Record<string, string> = {
    "Content-Type": "application/json",
    Accept: "text/event-stream",
  };
  const demoKey = process.env.QWENFAST_DEMO_KEY;
  if (demoKey) upstreamHeaders.Authorization = `Bearer ${demoKey}`;
  // So the engine's per-IP concurrency cap sees the browser, not this function's egress IP:
  // every demo visitor otherwise shares one bucket and the cap misfires on the whole page.
  const fwd = req.headers.get("x-forwarded-for");
  if (fwd) upstreamHeaders["X-Forwarded-For"] = fwd;

  const t0 = performance.now();
  let res: Response;
  try {
    res = await fetch(`${base}/v1/chat/completions`, {
      method: "POST",
      headers: upstreamHeaders,
      body: JSON.stringify(payload),
      // Abort upstream generation when the browser goes away (stop button /
      // navigation), so a cancelled turn stops costing GPU time immediately.
      signal: req.signal,
      cache: "no-store",
    });
  } catch {
    invalidateBase();
    return fail(OFFLINE_LINE, 503, true);
  }

  // Time to upstream response headers: handshake (if the pool was cold) plus
  // the engine's queue wait. The client subtracts this from its own TTFT to
  // separate network cost from engine cost.
  const upstreamMs = Math.round(performance.now() - t0);

  if (!res.ok || !res.body) {
    // 502/504 from the edge means the origin process is not there — the engine went away
    // between our probe and this request. Re-probe next time.
    if (res.status === 502 || res.status === 504) {
      invalidateBase();
      return fail(OFFLINE_LINE, 503, true);
    }
    // 503 now has a second meaning: the server's own admission guard is full (a public
    // endpoint on one GPU *will* hit this). That is "busy", not "offline" — the endpoint is
    // healthy and re-probing it would be wrong.
    // 400 is the one the user can actually act on (a conversation past the context window,
    // too many messages, a bad field), so it carries the engine's own wording and its
    // machine-readable `code` — the page uses the code to trim and retry rather than to
    // print a dead end.
    if (res.status === 400) {
      const { message, code } = await upstreamError(res, "The engine rejected that request.");
      return fail(message, 400, false, code ?? undefined);
    }
    if (res.status === 503) {
      const { message, code } = await upstreamError(
        res,
        "Engine is at capacity right now — try again in a few seconds.",
      );
      return fail(message, 503, false, code ?? undefined);
    }
    if (res.status === 429) {
      const { message, code } = await upstreamError(
        res,
        "Demo rate limit reached — try again in a minute.",
      );
      return fail(message, 429, false, code ?? undefined);
    }
    if (res.status === 401) {
      return fail("Demo key rejected by the engine.", 502);
    }
    const { message, code } = await upstreamError(res, `Engine returned ${res.status}.`);
    return fail(message, 502, false, code ?? undefined);
  }

  return new Response(res.body, {
    status: 200,
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
      "X-Upstream-Ms": String(upstreamMs),
      "Server-Timing": `upstream;dur=${upstreamMs}`,
    },
  });
}

/** Health + endpoint resolution + connection prewarm. */
export async function GET(): Promise<Response> {
  const t0 = performance.now();
  const { ok, base } = await warmUpstream();
  return new Response(JSON.stringify({ ok, base, ms: Math.round(performance.now() - t0) }), {
    status: 200,
    headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
  });
}
