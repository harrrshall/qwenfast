"use client";

import { useEffect, useRef, useState } from "react";

export type Metrics = {
  ok: boolean;
  perStreamTps?: number | null;
  lifetimeTps?: number | null;
  specAcceptLength?: number | null;
  kvFrac?: number | null;
  running?: number | null;
  waiting?: number | null;
};

/**
 * Polls the engine's own counters. Doubles as the liveness signal the page uses
 * to recover on its own: while a benchmark owns the GPU this stays `false`, and
 * flips to `true` within seconds of the demo server binding its port — no
 * reload, no redeploy.
 */
export function useEngineStatus(active: boolean) {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    let cancelled = false;

    const tick = async () => {
      if (document.visibilityState === "visible") {
        try {
          const r = await fetch("/api/metrics", { cache: "no-store" });
          const j = (await r.json()) as Metrics;
          if (!cancelled) setMetrics(j?.ok ? j : null);
        } catch {
          if (!cancelled) setMetrics(null);
        }
      }
      // Poll faster while streaming, idle back when quiet.
      if (!cancelled) timer.current = setTimeout(tick, active ? 1000 : 5000);
    };

    tick();
    return () => {
      cancelled = true;
      if (timer.current) clearTimeout(timer.current);
    };
  }, [active]);

  return { online: metrics?.ok === true, metrics };
}

export function EngineBar({ metrics }: { metrics: Metrics | null }) {
  if (!metrics?.ok) return null;

  const tps = metrics.perStreamTps ?? metrics.lifetimeTps;
  const bits: string[] = [];
  if (tps) bits.push(`${tps.toFixed(0)} t/s`);
  if (metrics.specAcceptLength) bits.push(`accept ${metrics.specAcceptLength.toFixed(2)}`);
  if (metrics.kvFrac != null) bits.push(`kv ${(metrics.kvFrac * 100).toFixed(0)}%`);
  if (metrics.running != null) bits.push(`${metrics.running} live`);
  if (bits.length === 0) return null;

  return (
    <div className="enginebar" aria-live="off">
      {bits.map((b, i) => (
        <span key={b}>
          {i > 0 && <span className="sep">·</span>}
          {b}
        </span>
      ))}
    </div>
  );
}
