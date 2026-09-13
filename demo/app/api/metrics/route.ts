import { resolveBase } from "@/lib/upstream";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * Engine-side efficiency, read straight from qwenfast's Prometheus endpoint
 * (server/metrics.py). This is what the engine itself believes about its
 * throughput, as opposed to what the browser can time from the outside.
 */

function parseProm(text: string) {
  const gauges = new Map<string, number>();
  const sums = new Map<string, number>();
  const counts = new Map<string, number>();

  for (const line of text.split("\n")) {
    if (!line || line.startsWith("#")) continue;
    const sp = line.lastIndexOf(" ");
    if (sp === -1) continue;
    const name = line.slice(0, sp).trim();
    const value = Number(line.slice(sp + 1));
    if (!Number.isFinite(value)) continue;

    if (name.endsWith("_sum")) sums.set(name.slice(0, -4), value);
    else if (name.endsWith("_count")) counts.set(name.slice(0, -6), value);
    else if (!name.includes("{")) gauges.set(name, value);
  }

  // Histogram mean, in ms. sum/count is exact — no bucket interpolation needed.
  const meanMs = (base: string): number | null => {
    const s = sums.get(base);
    const c = counts.get(base);
    return s != null && c != null && c > 0 ? (s / c) * 1000 : null;
  };

  const g = (n: string) => gauges.get(n) ?? null;
  const kvUsed = g("qwenfast:kv_pages_used");
  const kvTotal = g("qwenfast:kv_pages_total");
  const tpotMs = meanMs("qwenfast:time_per_output_token_seconds");

  return {
    running: g("qwenfast:num_requests_running"),
    waiting: g("qwenfast:num_requests_waiting"),
    // Lifetime average (generated tokens / uptime), not an instantaneous rate.
    lifetimeTps: g("qwenfast:tokens_per_second"),
    generatedTotal: g("qwenfast:generation_tokens_total"),
    // Per-stream steady-state rate implied by mean inter-token latency: this is
    // the number a single user actually experiences.
    perStreamTps: tpotMs && tpotMs > 0 ? 1000 / tpotMs : null,
    ttftMs: meanMs("qwenfast:time_to_first_token_seconds"),
    tpotMs,
    specAcceptLength: g("qwenfast:spec_accept_length"),
    specAcceptRate: g("qwenfast:spec_acceptance_rate"),
    kvFrac: kvUsed != null && kvTotal ? kvUsed / kvTotal : null,
    uptimeS: g("qwenfast:uptime_seconds"),
  };
}

export async function GET(): Promise<Response> {
  const headers = { "Content-Type": "application/json", "Cache-Control": "no-store" };
  try {
    const base = await resolveBase();
    if (!base) return new Response(JSON.stringify({ ok: false }), { status: 200, headers });
    const res = await fetch(`${base}/metrics`, {
      cache: "no-store",
      signal: AbortSignal.timeout(6_000),
    });
    if (!res.ok) {
      return new Response(JSON.stringify({ ok: false }), { status: 200, headers });
    }
    return new Response(JSON.stringify({ ok: true, ...parseProm(await res.text()) }), {
      status: 200,
      headers,
    });
  } catch {
    return new Response(JSON.stringify({ ok: false }), { status: 200, headers });
  }
}
