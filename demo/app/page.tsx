"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Markdown } from "./markdown";
import { EngineBar, useEngineStatus } from "./enginebar";
import { trimHistory, type ChatMsg, type Limits } from "@/lib/trim";

type Role = "user" | "assistant";

type Stats = {
  tokens: number;
  ttftMs: number | null;
  /** Proxy→engine time to response headers; the network share of TTFT. */
  netMs: number | null;
  /** Mean inter-token latency, ms, derived from the generation window. */
  itlMs: number | null;
  elapsedMs: number;
  /** True while the count is still an estimate (no `usage` frame yet). */
  approx: boolean;
};

type Turn = {
  id: string;
  role: Role;
  content: string;
  stats?: Stats;
  error?: string;
  /** Engine wasn't serving yet — recoverable, so offer a retry once it is. */
  offline?: boolean;
  /** How many older messages were left out of the request that produced this turn. */
  trimmed?: number;
};

/**
 * The engine's request envelope, fetched once per page load.
 *
 * `null` until it arrives; `trimHistory` treats that as "assume the conservative defaults",
 * so the first message of a session is already trimmed correctly rather than being the one
 * that discovers the limit.
 */
function useLimits(): Limits | null {
  const [limits, setLimits] = useState<Limits | null>(null);
  useEffect(() => {
    let live = true;
    fetch("/api/limits", { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : null))
      .then((j) => {
        if (live && j) setLimits(j as Limits);
      })
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, []);
  return limits;
}

/** Defensively strip <think>…</think>; an unterminated open tag hides the tail. */
function stripThinking(raw: string): string {
  let out = raw.replace(/<think>[\s\S]*?<\/think>/gi, "");
  const open = (out.match(/<think>/gi) || []).length;
  const close = (out.match(/<\/think>/gi) || []).length;
  if (open > close) out = out.slice(0, out.search(/<think>/i));
  return out.replace(/^\s+/, "");
}

/** Decode rate over the generation window (first token → last), excluding TTFT. */
function rate(s: Stats): number | null {
  const window = s.ttftMs === null ? s.elapsedMs : s.elapsedMs - s.ttftMs;
  if (window <= 0 || s.tokens <= 0) return null;
  return (s.tokens / window) * 1000;
}

function StatLine({ s, live, spec }: { s: Stats; live?: boolean; spec?: number | null }) {
  const tps = rate(s);
  const approx = s.approx ? "~" : "";
  const parts: string[] = [];
  if (tps !== null) parts.push(`${approx}${tps.toFixed(1)} t/s`);
  parts.push(`${approx}${s.tokens} tok`);
  if (s.ttftMs !== null) {
    parts.push(
      s.netMs !== null
        ? `TTFT ${Math.round(s.ttftMs)} ms (net ${s.netMs})`
        : `TTFT ${Math.round(s.ttftMs)} ms`,
    );
  }
  if (s.itlMs !== null) parts.push(`ITL ${s.itlMs.toFixed(1)} ms`);
  if (spec) parts.push(`spec x${spec.toFixed(2)}`);
  if (live) parts.push("streaming");

  return (
    <div className="stats">
      {parts.map((p, i) => (
        <span key={p + i}>
          {i > 0 && <span className="sep">·</span>}
          {p}
        </span>
      ))}
    </div>
  );
}

export default function Page() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);

  const taRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const lastWarm = useRef(0);
  const turnsRef = useRef<Turn[]>([]);
  // Mean tokens carried per SSE chunk, calibrated from each response's `usage`. Starts at 1
  // (assume one chunk per token) so the live meter can only ever *under*-read before it has
  // calibrated — never inflate. A server emitting per-token chunks keeps this at 1.0; an
  // older server batching speculative-decode steps drives it toward the accept length.
  const tpcRef = useRef(1);

  const { online, metrics } = useEngineStatus(busy);
  const limits = useLimits();
  // Read inside `streamTurn` without making it a dependency: the callback is created once and
  // must see whatever the limits are at send time, not at mount time.
  const limitsRef = useRef<Limits | null>(null);
  useEffect(() => {
    limitsRef.current = limits;
  }, [limits]);

  useEffect(() => {
    turnsRef.current = turns;
  }, [turns]);

  /**
   * Open a pooled socket (and resolve the endpoint) before the user sends
   * anything: the handshake costs ~150 ms cold, so paying it here means the
   * first message's TTFT is engine time, not connection setup.
   */
  const warm = useCallback(() => {
    const now = Date.now();
    if (now - lastWarm.current < 30_000) return;
    lastWarm.current = now;
    fetch("/api/chat", { method: "GET", cache: "no-store" }).catch(() => undefined);
  }, []);

  useEffect(() => {
    taRef.current?.focus();
    warm();
  }, [warm]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end" });
  }, [turns]);

  const autosize = useCallback(() => {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${ta.scrollHeight}px`;
  }, []);

  const stop = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
  }, []);

  /** Streams one assistant turn for the given conversation prefix. */
  const streamTurn = useCallback(async (history: Turn[]) => {
    const botId = `a${Date.now()}`;

    // Keep the conversation inside the engine's prompt budget by leaving the oldest turns out
    // of the request. The turns stay on screen — this drops them from what is *sent*, not from
    // what the user can read — and the count drives the quiet note under the reply.
    const { messages: outgoing, dropped } = trimHistory(
      history.map((t) => ({ role: t.role, content: t.content }) as ChatMsg),
      limitsRef.current,
    );

    setTurns([...history, { id: botId, role: "assistant", content: "", trimmed: dropped }]);
    setBusy(true);

    const ac = new AbortController();
    abortRef.current = ac;

    const t0 = performance.now();
    let ttft: number | null = null;
    let netMs: number | null = null;
    let chunkTokens = 0;
    let usageTokens: number | null = null;
    let raw = "";
    let lastPaint = 0;

    // Live token count. Counting SSE chunks is exact only when the server emits one chunk
    // per token; under speculative decoding an older server commits 2-4 tokens per chunk and
    // the running meter would under-read by that factor until `usage` lands. Scale by the
    // tokens-per-chunk ratio observed on previous responses, which is 1.0 against a
    // per-token server and self-corrects against a batching one.
    const estimate = () => Math.round(chunkTokens * Math.max(tpcRef.current, 1));

    const patch = (fn: (t: Turn) => Turn) =>
      setTurns((prev) => prev.map((t) => (t.id === botId ? fn(t) : t)));

    const snapshot = (): Stats => {
      const tokens = usageTokens ?? estimate();
      const elapsedMs = performance.now() - t0;
      // Inter-token latency measured across the generation window rather than as the
      // median gap between arrivals. Now that the server emits one SSE frame per token,
      // TCP and the CDN coalesce many frames into a single read, so per-token arrival
      // timestamps collapse onto each other and a median of those gaps reads 0.0 ms.
      // The window average is unaffected by how the bytes were batched in transit.
      const window = ttft === null ? elapsedMs : elapsedMs - ttft;
      return {
        tokens,
        ttftMs: ttft,
        netMs,
        itlMs: tokens > 1 && window > 0 ? window / (tokens - 1) : null,
        elapsedMs,
        approx: usageTokens === null,
      };
    };

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: outgoing }),
        signal: ac.signal,
      });

      const up = res.headers.get("X-Upstream-Ms");
      if (up) netMs = Number(up);

      if (!res.ok || !res.body) {
        // The route forwards the engine's own `error.message` for anything the caller can act
        // on, so this shows "your messages came to N tokens…" rather than a status code.
        let msg = "Engine unavailable.";
        let offline = false;
        try {
          const j = await res.json();
          if (j?.error) msg = String(j.error);
          offline = Boolean(j?.offline);
        } catch {
          /* keep the generic line */
        }
        patch((t) => ({ ...t, error: msg, offline }));
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";

      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });

        let sawFirst = false;

        // SSE frames are separated by a blank line.
        let cut: number;
        while ((cut = buf.indexOf("\n\n")) !== -1) {
          const frame = buf.slice(0, cut);
          buf = buf.slice(cut + 2);

          for (const line of frame.split("\n")) {
            if (!line.startsWith("data:")) continue;
            const data = line.slice(5).trim();
            if (!data || data === "[DONE]") continue;

            let json: any;
            try {
              json = JSON.parse(data);
            } catch {
              continue;
            }

            if (json.usage?.completion_tokens != null) {
              usageTokens = json.usage.completion_tokens;
            }

            const piece: string | undefined = json.choices?.[0]?.delta?.content;
            if (piece) {
              const now = performance.now();
              if (ttft === null) {
                ttft = now - t0;
                sawFirst = true;
              }
              chunkTokens += 1;
              raw += piece;
            }
          }
        }

        // Paint the first token the instant it lands — the throttle below is
        // for sustained streaming and must not inflate perceived TTFT.
        const now = performance.now();
        if (sawFirst || now - lastPaint > 33) {
          lastPaint = now;
          patch((t) => ({ ...t, content: stripThinking(raw), stats: snapshot() }));
        }
      }

      // Re-calibrate tokens-per-chunk from the authoritative count, for the next response's
      // live meter. EMA so one short reply cannot swing it.
      if (usageTokens && usageTokens > 0 && chunkTokens > 0) {
        tpcRef.current = 0.6 * tpcRef.current + 0.4 * (usageTokens / chunkTokens);
      }
      patch((t) => ({ ...t, content: stripThinking(raw), stats: snapshot() }));
    } catch (e: any) {
      if (e?.name === "AbortError") {
        // Stopped by the user: keep whatever arrived.
        patch((t) => ({ ...t, content: stripThinking(raw), stats: snapshot() }));
      } else {
        patch((t) => ({ ...t, error: "Engine offline — still warming up.", offline: true }));
      }
    } finally {
      abortRef.current = null;
      setBusy(false);
      taRef.current?.focus();
    }
  }, []);

  const send = useCallback(() => {
    const text = draft.trim();
    if (!text || busy) return;
    const userTurn: Turn = { id: `u${Date.now()}`, role: "user", content: text };
    setDraft("");
    requestAnimationFrame(autosize);
    void streamTurn([...turnsRef.current, userTurn]);
  }, [draft, busy, autosize, streamTurn]);

  /** Re-run a turn that failed only because the engine wasn't up yet. */
  const retry = useCallback(
    (failedId: string) => {
      if (busy) return;
      const i = turnsRef.current.findIndex((t) => t.id === failedId);
      if (i < 0) return;
      void streamTurn(turnsRef.current.slice(0, i));
    },
    [busy, streamTurn],
  );

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  return (
    <div className="shell">
      <header className="head">
        <h1 className="wordmark">qwenfast</h1>
        <a className="byline" href="https://x.com/HarshalsinghCN" target="_blank" rel="noopener noreferrer">
          @HarshalsinghCN
        </a>
        <div className="headright">
          <EngineBar metrics={metrics} />
          <button
            className="clear"
            onClick={() => {
              stop();
              setTurns([]);
              taRef.current?.focus();
            }}
            disabled={turns.length === 0}
          >
            clear
          </button>
        </div>
      </header>

      <main className="thread">
        {turns.map((t, i) => {
          const streaming = busy && i === turns.length - 1 && t.role === "assistant";
          return (
            <article key={t.id} className={`turn ${t.role}`}>
              <div className="body">
                {t.error ? (
                  <span className="err">
                    {t.error}
                    {t.offline &&
                      (online ? (
                        <button className="retry" onClick={() => retry(t.id)} disabled={busy}>
                          retry
                        </button>
                      ) : (
                        <span className="waiting" />
                      ))}
                  </span>
                ) : t.role === "user" ? (
                  <p>{t.content}</p>
                ) : (
                  <Markdown text={t.content} caret={streaming} />
                )}
              </div>
              {t.role === "assistant" && !!t.trimmed && (
                <div className="trimmed">
                  {t.trimmed} earlier {t.trimmed === 1 ? "message" : "messages"} trimmed to fit
                  the context window
                </div>
              )}
              {t.role === "assistant" && t.stats && !t.error && (
                <StatLine
                  s={t.stats}
                  live={streaming}
                  spec={metrics?.specAcceptLength ?? null}
                />
              )}
            </article>
          );
        })}
        <div ref={endRef} />
      </main>

      <div className="composer">
        <textarea
          ref={taRef}
          rows={1}
          value={draft}
          placeholder="Ask anything"
          onChange={(e) => {
            setDraft(e.target.value);
            autosize();
          }}
          onFocus={warm}
          onKeyDown={onKeyDown}
          spellCheck={false}
          aria-label="Message"
        />
        {busy && (
          <button className="stop" onClick={stop}>
            stop
          </button>
        )}
      </div>

      <footer className="foot">
        World&apos;s fastest inference for Qwen3.8-27B &middot;{" "}
      </footer>
    </div>
  );
}
