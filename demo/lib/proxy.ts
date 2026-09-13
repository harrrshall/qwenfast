import { resolveBase, invalidateBase } from "@/lib/upstream";

/**
 * Transparent OpenAI-API proxy: `/v1/*` on the demo is forwarded to the engine.
 *
 * "Transparent" is the whole contract. An OpenAI SDK pointed at this base URL must not be able
 * to tell it is talking to a proxy, so:
 *
 *  - the client's `Authorization` header is forwarded *verbatim* and this layer holds no key of
 *    its own — auth, rate limits, and metering are all decided by the engine (`server/auth.py`),
 *    which is the only place that can see the whole picture across the direct and proxied paths;
 *  - the upstream status code and body are returned unchanged, including the error JSON, so a
 *    429 stays a 429 with its `Retry-After` intact and the SDK's own backoff does the right thing.
 *    This matters because the engine answers every refusal in OpenAI's
 *    `{"error": {message, type, param, code}}` envelope (`server/errors.py`), and rewriting or
 *    re-wrapping it here would undo exactly the fix that lets an SDK say
 *    `context_length_exceeded` instead of "Error code: 400";
 *  - the body is piped, not buffered, so SSE streams arrive token-by-token.
 *
 * The one thing it does add is `lib/upstream.ts`'s pooled dispatcher and endpoint discovery:
 * ~150 ms of TLS handshake per request saved (measured: 322 ms unpooled vs 175 ms pooled), and
 * the app follows the engine if it lands on another candidate endpoint. Callers who want the last
 * ~40-80 ms of proxy hop back can use the engine URL directly; docs/api.md documents both.
 */

export const PROXY_TIMEOUT_MS = 600_000;

/**
 * How long a *stream* may go without producing a byte before this proxy gives up on it.
 *
 * A serverless function that pipes an upstream body has no opinion about how long the pipe
 * stays silent: if the engine wedges mid-generation the socket stays open, the browser's
 * `reader.read()` never resolves, and the tab spins until the platform's own 300 s ceiling.
 * The engine's `--request-timeout` covers its own wedges; this covers the ones between here
 * and there (a reclaimed spot instance, a dropped Cloudflare tunnel), which the engine cannot
 * see and the client cannot distinguish from "still thinking".
 *
 * 60 s is well above any legitimate gap — TTFT is ~1 s and inter-token latency ~10 ms even at
 * full load, and a queued request behind a full admission cap is refused with a 503 rather than
 * held silently.
 */
export const IDLE_STREAM_TIMEOUT_MS = Number(
  process.env.QWENFAST_IDLE_STREAM_MS || 60_000,
);

function errorBody(message: string, type: string, code: string | null = null) {
  // OpenAI's error envelope, so client SDKs surface a sensible `.error.message`.
  return JSON.stringify({ error: { message, type, param: null, code } });
}

const OFFLINE = errorBody(
  "The qwenfast engine is offline (the server is down or restarting). Retry in a minute.",
  "service_unavailable",
  "engine_offline",
);

const STALLED = errorBody(
  `The engine stopped sending data for ${IDLE_STREAM_TIMEOUT_MS / 1000}s and the stream was ` +
    `closed. Retry — this is usually a restarting engine, not a bad request.`,
  "server_error",
  "upstream_stalled",
);

/** Headers worth carrying back to the client; everything else is hop-by-hop noise. */
const PASS_BACK = [
  "content-type",
  "retry-after",
  "x-request-id",
  "cache-control",
  "x-accel-buffering",
  // What the engine actually resolved `max_tokens` to after the per-key cap and the
  // context clamp — the signal a client needs to know its request was trimmed, not refused.
  "x-qwenfast-max-tokens",
];

/**
 * Wrap an upstream body so a silent stream is closed cleanly instead of hanging forever.
 *
 * For SSE the termination is *in-band*: a final `data: {"error": …}` frame followed by
 * `data: [DONE]`, which is what an OpenAI-shaped client already knows how to read. Closing the
 * stream with no frame at all would be indistinguishable from a normal end, and the caller
 * would treat a truncated answer as a complete one.
 */
function withIdleTimeout(
  body: ReadableStream<Uint8Array>,
  ms: number,
  isSse: boolean,
): ReadableStream<Uint8Array> {
  const reader = body.getReader();
  const encoder = new TextEncoder();

  return new ReadableStream<Uint8Array>({
    async pull(controller) {
      let timer: ReturnType<typeof setTimeout> | undefined;
      try {
        const timeout = new Promise<"timeout">((resolve) => {
          timer = setTimeout(() => resolve("timeout"), ms);
        });
        const result = await Promise.race([reader.read(), timeout]);

        if (result === "timeout") {
          void reader.cancel("idle timeout").catch(() => undefined);
          if (isSse) {
            controller.enqueue(encoder.encode(`data: ${STALLED}\n\ndata: [DONE]\n\n`));
          }
          controller.close();
          return;
        }

        const { done, value } = result;
        if (done) {
          controller.close();
          return;
        }
        if (value) controller.enqueue(value);
      } catch {
        // An upstream that errors mid-body is the same story as one that stalls: say so
        // in-band rather than leaving the client to guess from a truncated stream.
        if (isSse) {
          try {
            controller.enqueue(encoder.encode(`data: ${STALLED}\n\ndata: [DONE]\n\n`));
          } catch {
            /* controller already closed */
          }
        }
        controller.close();
      } finally {
        if (timer) clearTimeout(timer);
      }
    },
    cancel(reason) {
      void reader.cancel(reason).catch(() => undefined);
    },
  });
}

export async function proxy(req: Request, path: string): Promise<Response> {
  const base = await resolveBase();
  if (!base) {
    return new Response(OFFLINE, {
      status: 503,
      headers: { "Content-Type": "application/json", "Retry-After": "30", "Cache-Control": "no-store" },
    });
  }

  const auth = req.headers.get("authorization");
  const headers: Record<string, string> = {
    Accept: req.headers.get("accept") ?? "application/json",
  };
  if (auth) headers.Authorization = auth;

  // Carry the caller's address through so the engine's per-IP concurrency cap sees the real
  // client rather than this function's egress IP — without it every request from the Vercel
  // proxy shares one bucket, and the per-IP cap becomes a second, much stricter global cap.
  // Appended to any existing chain, per the `X-Forwarded-For` convention.
  const forwarded = req.headers.get("x-forwarded-for");
  if (forwarded) headers["X-Forwarded-For"] = forwarded;

  const method = req.method.toUpperCase();
  let body: string | undefined;
  if (method === "POST" || method === "PUT" || method === "PATCH") {
    body = await req.text();
    headers["Content-Type"] = req.headers.get("content-type") ?? "application/json";
  }

  const url = new URL(req.url);
  const target = `${base}${path}${url.search}`;

  let res: Response;
  try {
    res = await fetch(target, {
      method,
      headers,
      body,
      // Cancel upstream generation when the client hangs up, so an abandoned stream stops
      // costing GPU seconds immediately.
      signal: req.signal,
      cache: "no-store",
    });
  } catch {
    invalidateBase();
    return new Response(OFFLINE, {
      status: 503,
      headers: { "Content-Type": "application/json", "Retry-After": "30", "Cache-Control": "no-store" },
    });
  }

  // 502/503/504 from the edge means nothing is listening at the origin: re-probe next time so a
  // restarted server (or one that moved endpoints) is picked up within seconds.
  if (res.status === 502 || res.status === 504) {
    invalidateBase();
    return new Response(OFFLINE, {
      status: 503,
      headers: { "Content-Type": "application/json", "Retry-After": "30", "Cache-Control": "no-store" },
    });
  }

  const out = new Headers();
  for (const name of PASS_BACK) {
    const v = res.headers.get(name);
    if (v) out.set(name, v);
  }
  if (!out.has("cache-control")) out.set("Cache-Control", "no-store");

  const isSse = Boolean(res.headers.get("content-type")?.includes("text/event-stream"));
  if (isSse) {
    out.set("X-Accel-Buffering", "no");
    out.set("Connection", "keep-alive");
  }

  // Error bodies are small and already in the right shape: pass them straight through, so the
  // status code, the `Retry-After` and the `error.code` all survive the hop unchanged.
  const stream =
    res.body && (isSse || res.ok)
      ? withIdleTimeout(res.body, IDLE_STREAM_TIMEOUT_MS, isSse)
      : res.body;

  return new Response(stream, { status: res.status, headers: out });
}
