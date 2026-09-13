import { Agent, setGlobalDispatcher } from "undici";

/**
 * Serving-path plumbing: connection pooling, endpoint discovery, health.
 *
 * Latency note. A cold TCP and TLS handshake to a remote engine costs on the
 * order of 150 ms, paid on every request unless the socket is pooled, and
 * serverless pools nothing by default. It lands directly on time to first
 * token, so `setGlobalDispatcher` below makes the platform `fetch()` reuse
 * sockets for as long as the instance stays warm.
 */

const KEEPALIVE_MS = 10 * 60_000;

declare global {
  // eslint-disable-next-line no-var
  var __qwenfastDispatcher: Agent | undefined;
}

if (!globalThis.__qwenfastDispatcher) {
  const agent = new Agent({
    keepAliveTimeout: KEEPALIVE_MS,
    keepAliveMaxTimeout: KEEPALIVE_MS,
    connections: 64,
    pipelining: 0,
    headersTimeout: 120_000,
    bodyTimeout: 600_000,
  });
  setGlobalDispatcher(agent);
  globalThis.__qwenfastDispatcher = agent;
}

export const MODEL = "qwen3.8-27b";

const DEFAULT_BASE_URL = "http://127.0.0.1:8000";

const strip = (u: string) => u.replace(/\/+$/, "");

/** Preferred endpoint: whatever QWENFAST_BASE_URL says. */
export function baseUrl(): string {
  return strip(process.env.QWENFAST_BASE_URL || DEFAULT_BASE_URL);
}

/**
 * Candidate endpoints: QWENFAST_BASE_URLS (comma-separated) when set, otherwise
 * the single preferred base url. Every candidate is probed concurrently and the
 * first one that answers is used.
 */
function candidates(): string[] {
  const extra = process.env.QWENFAST_BASE_URLS;
  if (extra) return extra.split(",").map((s) => strip(s.trim())).filter(Boolean);
  return [baseUrl()];
}

type Resolved = { base: string | null; at: number };
declare global {
  // eslint-disable-next-line no-var
  var __qwenfastResolved: Resolved | undefined;
}

// Cache the winner briefly. Short TTLs so that when the benchmark finishes and
// the demo server binds :8000, the app picks it up within seconds without a
// redeploy or a page reload.
const OK_TTL_MS = 30_000;
const FAIL_TTL_MS = 3_000;

async function probe(base: string, timeoutMs: number): Promise<boolean> {
  try {
    const r = await fetch(`${base}/health`, {
      cache: "no-store",
      signal: AbortSignal.timeout(timeoutMs),
    });
    await r.arrayBuffer().catch(() => undefined);
    return r.ok;
  } catch {
    return false;
  }
}

/**
 * Returns a healthy engine base URL, or null when the engine is down.
 * `null` means "not serving yet" — which is the normal state while a benchmark
 * owns the GPU — and callers should report that as offline, not as an error.
 */
export async function resolveBase(timeoutMs = 4_000): Promise<string | null> {
  const cached = globalThis.__qwenfastResolved;
  const ttl = cached?.base ? OK_TTL_MS : FAIL_TTL_MS;
  if (cached && Date.now() - cached.at < ttl) return cached.base;

  const list = candidates();

  // Probe every candidate concurrently rather than preferred-then-siblings. The
  // sequential version cost two serial round trips (~167 ms) whenever the engine
  // was down, which is exactly when /api/metrics polls hardest; concurrently it
  // costs one (~60 ms). The two extra requests in the healthy case are a 502 from
  // the edge and only happen once per cache period.
  const results = await Promise.all(
    list.map(async (b) => ((await probe(b, timeoutMs)) ? b : null)),
  );
  // Candidate order is preference order, so the preferred endpoint still wins
  // whenever more than one answers.
  const win = results.find((b): b is string => b !== null) ?? null;

  globalThis.__qwenfastResolved = { base: win, at: Date.now() };
  return win;
}

/** Forget the cached endpoint so the next call re-probes immediately. */
export function invalidateBase(): void {
  globalThis.__qwenfastResolved = undefined;
}

/**
 * Opens a pooled connection (and resolves the endpoint) before the user sends
 * anything, so the expensive part of the first real request is already done.
 */
export async function warmUpstream(): Promise<{ ok: boolean; base: string | null }> {
  const base = await resolveBase();
  return { ok: base !== null, base };
}
