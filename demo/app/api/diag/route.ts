import { baseUrl, resolveBase, invalidateBase } from "@/lib/upstream";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * Serving-path diagnostics: which region the function runs in, which engine
 * endpoint discovery picked, and what a round trip costs cold vs pooled.
 * Used to choose `regions` in vercel.json empirically. Works even while the
 * engine is down, because TCP+TLS to the Cloudflare edge completes regardless
 * of whether anything is listening at the origin.
 */
export async function GET(): Promise<Response> {
  invalidateBase();
  const t0 = performance.now();
  const resolved = await resolveBase();
  const resolveMs = +(performance.now() - t0).toFixed(1);

  const target = `${resolved ?? baseUrl()}/health`;
  const samples: { ms: number; status: number | string }[] = [];

  for (let i = 0; i < 4; i++) {
    const t = performance.now();
    try {
      const r = await fetch(target, { cache: "no-store", signal: AbortSignal.timeout(10_000) });
      await r.arrayBuffer().catch(() => undefined);
      samples.push({ ms: +(performance.now() - t).toFixed(1), status: r.status });
    } catch {
      samples.push({ ms: +(performance.now() - t).toFixed(1), status: "err" });
    }
  }

  const pooled = samples.slice(1).map((s) => s.ms);
  const pooledMean = pooled.length
    ? +(pooled.reduce((a, b) => a + b, 0) / pooled.length).toFixed(1)
    : null;

  return new Response(
    JSON.stringify(
      {
        region: process.env.VERCEL_REGION ?? "local",
        preferred: baseUrl(),
        resolved,
        engineOnline: resolved !== null,
        resolveMs,
        samples,
        pooledMeanMs: pooledMean,
      },
      null,
      2,
    ),
    { status: 200, headers: { "Content-Type": "application/json", "Cache-Control": "no-store" } },
  );
}
