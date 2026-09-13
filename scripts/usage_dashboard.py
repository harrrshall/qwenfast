#!/usr/bin/env python3
"""Private usage dashboard for the public qwenfast endpoint. Runs on *your* machine.

    python3 scripts/usage_dashboard.py            # http://127.0.0.1:8787
    python3 scripts/usage_dashboard.py --open     # ...and open a browser
    python3 scripts/usage_dashboard.py --json     # one-shot text summary, no server

Deliberately local-only and stdlib-only:

* **Local-only** because the admin key must never leave this machine, and usage/cost data is the
  operator's, not the public's. The server binds 127.0.0.1 and holds the key in memory; the
  browser talks to this process, this process talks to the server. Nothing about usage is deployed.
* **Stdlib-only** so it runs from a bare `python3` with no venv, on any machine that can reach the
  server.

The page reads `/admin/usage` (aggregates, cost, per-key breakdown, 24 h hourly series) and
`/metrics` (the engine's own instantaneous rates) and refreshes every 5 s.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import ssl
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ADMIN_KEY_FILE = REPO_ROOT / ".secrets" / "admin_key"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"

_SSL_CTX = ssl.create_default_context()


def sibling_urls(base: str) -> list[str]:
    """Candidate base urls for the server. One entry unless a comma-separated list is given."""
    return [b.strip().rstrip("/") for b in base.split(",") if b.strip()]


class Box:
    """Talks to the server, remembering which candidate endpoint answered."""

    def __init__(self, base: str, admin_key: str, timeout: float = 8.0) -> None:
        self.candidates = sibling_urls(base)
        self.base: str | None = None
        self.admin_key = admin_key
        self.timeout = timeout

    def _get(self, base: str, path: str, auth: bool) -> tuple[int, str]:
        req = urllib.request.Request(f"{base}{path}", method="GET")
        if auth:
            req.add_header("Authorization", f"Bearer {self.admin_key}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout, context=_SSL_CTX) as resp:
                return resp.status, resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode("utf-8", "replace")

    def fetch(self, path: str, auth: bool = True) -> tuple[int, str]:
        order = ([self.base] if self.base else []) + [c for c in self.candidates if c != self.base]
        last = (503, '{"error":"no endpoint answered"}')
        for base in order:
            try:
                status, text = self._get(base, path, auth)
            except Exception as exc:  # noqa: BLE001 - offline is an expected state, not an error
                last = (503, json.dumps({"error": f"{exc.__class__.__name__}: {exc}"}))
                continue
            if status < 500:
                self.base = base
                return status, text
            last = (status, text)
        self.base = None
        return last


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>qwenfast usage</title>
<style>
:root{
  --bg:#fbfaf7; --fg:#17171a; --muted:#8c8a83; --faint:#e2ded4; --rule:#eae6dc;
  --code-bg:#f2efe7; --series:#0b7c5f; --good:#0b7c5f; --bad:#a13b2a;
}
@media (prefers-color-scheme:dark){
  :root{ --bg:#111110; --fg:#e9e7e1; --muted:#7e7c76; --faint:#2a2a27; --rule:#232320;
         --code-bg:#1b1b19; --series:#22ad8b; --good:#22ad8b; --bad:#d8735c; }
}
*{box-sizing:border-box}
html,body{margin:0;padding:0;background:var(--bg);color:var(--fg)}
body{font-family:Georgia,"Times New Roman",serif;font-size:17px;line-height:1.6;
     -webkit-font-smoothing:antialiased}
.mono{font-family:ui-monospace,SFMono-Regular,"SF Mono",Menlo,monospace}
.shell{max-width:52rem;margin:0 auto;padding:0 1.5rem 4rem}
header{display:flex;align-items:baseline;justify-content:space-between;gap:1rem;
       padding:2.25rem 0 1.25rem;border-bottom:1px solid var(--rule)}
h1{font-family:ui-monospace,monospace;font-size:.8125rem;letter-spacing:.02em;font-weight:500;margin:0}
.status{font-family:ui-monospace,monospace;font-size:.6875rem;letter-spacing:.04em;color:var(--muted)}
.dot{display:inline-block;width:.5rem;height:.5rem;border-radius:50%;
     background:var(--muted);margin-right:.4rem;vertical-align:baseline}
.dot.up{background:var(--good)} .dot.down{background:var(--bad)}

/* hero */
.hero{padding:2rem 0 .5rem}
.hero .n{font-family:ui-monospace,monospace;font-size:clamp(2.75rem,9vw,4.5rem);
         line-height:1;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.hero .cap{font-family:ui-monospace,monospace;font-size:.6875rem;letter-spacing:.08em;
           text-transform:uppercase;color:var(--muted);margin-top:.6rem}

/* stat tiles */
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(9.5rem,1fr));
       gap:1px;background:var(--rule);border:1px solid var(--rule);margin:2rem 0 0}
.tile{background:var(--bg);padding:.9rem 1rem}
.tile .k{font-family:ui-monospace,monospace;font-size:.625rem;letter-spacing:.07em;
         text-transform:uppercase;color:var(--muted)}
.tile .v{font-family:ui-monospace,monospace;font-size:1.25rem;font-variant-numeric:tabular-nums;
         margin-top:.25rem;letter-spacing:-.01em}
.tile .s{font-family:ui-monospace,monospace;font-size:.6875rem;color:var(--muted);margin-top:.1rem}

section{margin-top:2.75rem}
h2{font-family:ui-monospace,monospace;font-size:.6875rem;letter-spacing:.08em;
   text-transform:uppercase;color:var(--muted);font-weight:500;margin:0 0 .35rem}
.sub{font-size:.875rem;color:var(--muted);margin:0 0 1rem}

/* chart */
.chartwrap{position:relative;overflow-x:auto}
svg{display:block;width:100%;height:auto;overflow:visible}
.grid{stroke:var(--rule);stroke-width:1}
.axis{fill:var(--muted);font-family:ui-monospace,monospace;font-size:9px}
.bar{fill:var(--series)}
.hit{fill:transparent}
.hit:hover + .bar,.bar.on{opacity:.75}
.dlabel{fill:var(--fg);font-family:ui-monospace,monospace;font-size:9px;
        font-variant-numeric:tabular-nums}
.tip{position:absolute;pointer-events:none;opacity:0;transition:opacity .1s;
     background:var(--bg);border:1px solid var(--faint);padding:.4rem .55rem;
     font-family:ui-monospace,monospace;font-size:.6875rem;line-height:1.45;
     white-space:nowrap;box-shadow:0 2px 10px rgba(0,0,0,.09);z-index:5}
.tip .t{color:var(--muted)}

/* table */
table{width:100%;border-collapse:collapse;font-family:ui-monospace,monospace;font-size:.8125rem;
      font-variant-numeric:tabular-nums}
th{text-align:right;font-weight:500;font-size:.625rem;letter-spacing:.07em;text-transform:uppercase;
   color:var(--muted);padding:.4rem .5rem;border-bottom:1px solid var(--rule)}
th:first-child,td:first-child{text-align:left}
td{padding:.45rem .5rem;border-bottom:1px solid var(--rule);text-align:right}
tr:last-child td{border-bottom:0}
.warn{color:var(--bad)}
footer{margin-top:3rem;padding-top:1rem;border-top:1px solid var(--rule);
       font-family:ui-monospace,monospace;font-size:.6875rem;color:var(--muted)}
</style>
</head>
<body>
<div class="shell">
  <header>
    <h1>qwenfast &middot; usage</h1>
    <div class="status"><span class="dot" id="dot"></span><span id="statusText">connecting…</span></div>
  </header>

  <div class="hero">
    <div class="n" id="heroTokens">—</div>
    <div class="cap">total tokens (prompt + generated) &middot; all time</div>
  </div>

  <div class="tiles">
    <div class="tile"><div class="k">requests</div><div class="v" id="tReq">—</div><div class="s" id="tReqSub">&nbsp;</div></div>
    <div class="tile"><div class="k">tok/s now</div><div class="v" id="tTps">—</div><div class="s">trailing 60 s</div></div>
    <div class="tile"><div class="k">per stream</div><div class="v" id="tStream">—</div><div class="s">tok/s, 1 user</div></div>
    <div class="tile"><div class="k">in flight</div><div class="v" id="tInflight">—</div><div class="s" id="tInflightSub">&nbsp;</div></div>
    <div class="tile"><div class="k">uptime</div><div class="v" id="tUptime">—</div><div class="s" id="tUptimeSub">&nbsp;</div></div>
    <div class="tile"><div class="k">gpu cost</div><div class="v" id="tCost">—</div><div class="s" id="tCostSub">&nbsp;</div></div>
    <div class="tile"><div class="k">per 1M tokens</div><div class="v" id="tPerM">—</div><div class="s" id="tPerMSub">&nbsp;</div></div>
    <div class="tile"><div class="k">prompt tok</div><div class="v" id="tPrompt">—</div><div class="s">ingested</div></div>
  </div>

  <section>
    <h2>Total tokens per hour</h2>
    <p class="sub">Last 24 hours, from the engine's own per-request log. Hover a bar for detail.</p>
    <div class="chartwrap">
      <svg id="chart" viewBox="0 0 720 200" role="img" aria-labelledby="chartTitle"></svg>
      <div class="tip" id="tip"></div>
    </div>
  </section>

  <section>
    <h2>By key</h2>
    <p class="sub">Keys are shown by name; key material never leaves the server.</p>
    <table>
      <thead><tr><th>name</th><th>requests</th><th>prompt</th><th>output</th><th>errors</th><th>last seen</th></tr></thead>
      <tbody id="keyRows"><tr><td colspan="6" style="text-align:left;color:var(--muted)">loading…</td></tr></tbody>
    </table>
  </section>

  <section id="errorSection" hidden>
    <h2>Refusals by reason</h2>
    <p class="sub">Why requests were turned away — the breakdown that "57 errors" used to hide.</p>
    <table>
      <thead><tr><th>reason</th><th>all time</th><th>last 24 h</th></tr></thead>
      <tbody id="errorRows"></tbody>
    </table>
  </section>

  <section id="latencySection">
    <h2>Latency (last hour)</h2>
    <p class="sub">Percentiles over completed requests. TPOT excludes prefill, so it is decode speed.</p>
    <table>
      <thead><tr><th></th><th>p50</th><th>p90</th><th>p99</th></tr></thead>
      <tbody id="latencyRows"></tbody>
    </table>
  </section>

  <section id="healthSection">
    <h2>Serving health</h2>
    <table>
      <tbody id="healthRows"></tbody>
    </table>
  </section>

  <section id="querySection" hidden>
    <h2>Recent queries</h2>
    <p class="sub">
      Content logging is on. This view is local to your machine and admin-key only —
      nothing about it is served from the public site. Text is truncated; use
      <code>scripts/export_queries.py</code> for the full rows.
    </p>
    <table>
      <thead><tr><th>when</th><th>key</th><th>ip</th><th>prompt</th><th>reply</th></tr></thead>
      <tbody id="queryRows"></tbody>
    </table>
  </section>

  <section id="abuseSection" hidden>
    <h2>Queries by key / IP</h2>
    <p class="sub">One caller with a disproportionate share is the first thing to look at.</p>
    <table>
      <thead><tr><th>who</th><th>scope</th><th>queries</th><th>prompt tok</th><th>output tok</th></tr></thead>
      <tbody id="abuseRows"></tbody>
    </table>
  </section>

  <footer id="foot">refreshing every 5 s</footer>
</div>

<script>
const $ = (id) => document.getElementById(id);
const nf = new Intl.NumberFormat("en-US");
const compact = (n) => n == null ? "—" :
  n >= 1e9 ? (n/1e9).toFixed(2)+"B" : n >= 1e6 ? (n/1e6).toFixed(2)+"M" :
  n >= 1e4 ? (n/1e3).toFixed(1)+"k" : nf.format(Math.round(n));
const dur = (s) => {
  if (s == null) return "—";
  const h = Math.floor(s/3600), m = Math.floor((s%3600)/60);
  return h >= 24 ? `${Math.floor(h/24)}d ${h%24}h` : h ? `${h}h ${m}m` : `${m}m`;
};
const ago = (ts) => {
  if (!ts) return "never";
  const d = Date.now()/1000 - ts;
  if (d < 60) return Math.round(d)+"s ago";
  if (d < 3600) return Math.round(d/60)+"m ago";
  if (d < 86400) return Math.round(d/3600)+"h ago";
  return Math.round(d/86400)+"d ago";
};

let lastGood = 0;

function setStatus(ok, text){
  $("dot").className = "dot " + (ok ? "up" : "down");
  $("statusText").textContent = text;
}

/* ---- chart ---------------------------------------------------------------
   Single series, so no legend (the heading names it). Bars are anchored to the
   baseline with 4px rounded tops and a 2px gap between neighbours; the grid is
   recessive; only the peak bar carries a direct label. */
function drawChart(hourly){
  const svg = $("chart"), tip = $("tip");
  const W = 720, H = 200, padL = 44, padR = 12, padT = 16, padB = 24;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const tot = h => (h.total_tokens ?? ((h.prompt_tokens||0) + (h.completion_tokens||0)));
  const peakV = Math.max(1, ...hourly.map(tot));
  // Round the axis top to a nice number so the top tick never restates the peak bar's own
  // direct label (they would otherwise always be the same number, twice).
  const pow = Math.pow(10, Math.floor(Math.log10(peakV)));
  const max = Math.ceil(peakV / (pow/2)) * (pow/2);
  const slot = plotW / hourly.length;
  const bw = Math.max(2, slot - 2);           // 2px surface gap between bars
  const y = (v) => padT + plotH - (v / max) * plotH;

  const ticks = [0, max/2, max];
  let out = `<title id="chartTitle">Total tokens per hour, last 24 hours</title>`;
  for (const t of ticks){
    const yy = y(t).toFixed(1);
    out += `<line class="grid" x1="${padL}" x2="${W-padR}" y1="${yy}" y2="${yy}"/>`;
    out += `<text class="axis" x="${padL-8}" y="${yy}" text-anchor="end" dominant-baseline="middle">${compact(t)}</text>`;
  }

  const peak = hourly.reduce((a,b) => tot(b) > tot(a) ? b : a, hourly[0]);
  hourly.forEach((h, i) => {
    const x = padL + i*slot + (slot-bw)/2;
    const v = tot(h);
    const top = y(v);
    const hgt = Math.max(v > 0 ? 2 : 0, padT + plotH - top);
    if (hgt > 0){
      const r = Math.min(4, bw/2, hgt);
      // Rounded top corners only — the data end is rounded, the baseline end is square.
      out += `<path class="bar" d="M${x} ${padT+plotH} L${x} ${top+r} Q${x} ${top} ${x+r} ${top}`
           + ` L${x+bw-r} ${top} Q${x+bw} ${top} ${x+bw} ${top+r} L${x+bw} ${padT+plotH} Z"/>`;
    }
    if (h === peak && v > 0){
      out += `<text class="dlabel" x="${x+bw/2}" y="${top-5}" text-anchor="middle">${compact(v)}</text>`;
    }
    const d = new Date(h.hour*1000);
    if (i % 6 === 0){
      out += `<text class="axis" x="${x+bw/2}" y="${H-8}" text-anchor="middle">`
           + `${String(d.getHours()).padStart(2,"0")}:00</text>`;
    }
    // Hit target spans the full slot height, so hovering anywhere in the column works.
    out += `<rect class="hit" x="${padL+i*slot}" y="${padT}" width="${slot}" height="${plotH}"`
         + ` data-i="${i}"><title></title></rect>`;
  });
  svg.innerHTML = out;

  const bars = [...svg.querySelectorAll(".bar")];
  svg.querySelectorAll(".hit").forEach((hit) => {
    hit.addEventListener("mousemove", (e) => {
      const h = hourly[+hit.dataset.i];
      const d = new Date(h.hour*1000);
      tip.innerHTML = `<div class="t">${d.toLocaleString([], {month:"short", day:"numeric"})} `
        + `${String(d.getHours()).padStart(2,"0")}:00</div>`
        + `<div>${nf.format(tot(h))} total tokens (${nf.format(h.completion_tokens)} generated)</div>`
        + `<div class="t">${nf.format(h.requests)} requests &middot; ${nf.format(h.prompt_tokens)} prompt</div>`;
      const box = svg.parentElement.getBoundingClientRect();
      tip.style.opacity = 1;
      tip.style.left = Math.min(box.width - tip.offsetWidth - 4, Math.max(0, e.clientX - box.left + 12)) + "px";
      tip.style.top  = Math.max(0, e.clientY - box.top - tip.offsetHeight - 10) + "px";
    });
    hit.addEventListener("mouseleave", () => { tip.style.opacity = 0; });
  });
  void bars;
}

function healthRow(k, v){ return `<tr><td>${k}</td><td>${v}</td></tr>`; }

async function tick(){
  let u, m;
  try {
    const [ur, mr] = await Promise.all([fetch("/api/usage"), fetch("/api/metrics")]);
    u = await ur.json();
    m = await mr.json();
    if (!ur.ok || u.error) throw new Error(u.error || ("HTTP " + ur.status));
  } catch (err) {
    setStatus(false, "engine unreachable" + (lastGood ? " · last ok " + ago(lastGood) : ""));
    $("foot").textContent = String(err.message || err);
    return;
  }
  lastGood = Date.now()/1000;
  const healthy = u.engine_health === "ok";
  setStatus(healthy, healthy ? `${u.model} · live` : `unhealthy: ${u.engine_health}`);

  const t = u.totals, c = u.cost, e = u.engine || {};
  $("heroTokens").textContent = nf.format(t.total_tokens ?? (t.prompt_tokens + t.completion_tokens));
  $("tReq").textContent = compact(t.requests);
  $("tReqSub").textContent = t.errors ? `${nf.format(t.errors)} errors` : "no errors";
  $("tTps").textContent = (u.recent_output_tokens_per_second ?? 0).toFixed(0);
  $("tStream").textContent = e.per_stream_tokens_per_second ? e.per_stream_tokens_per_second.toFixed(0) : "—";
  $("tInflight").textContent = (u.capacity?.inflight ?? 0) + (u.capacity?.capacity ? " / " + u.capacity.capacity : "");
  $("tInflightSub").textContent = `peak ${u.capacity?.peak_inflight ?? 0}`;
  $("tUptime").textContent = dur(u.service_uptime_s);
  $("tUptimeSub").textContent = `process ${dur(u.process_uptime_s)}`;
  $("tCost").textContent = "₹" + nf.format(Math.round(c.inr));
  $("tCostSub").textContent = "$" + (c.usd ?? 0).toFixed(2) + ` · ₹${c.gpu_rate_inr_per_hour}/h`;
  $("tPerM").textContent = c.per_1m_total_tokens_inr ? "₹" + nf.format(Math.round(c.per_1m_total_tokens_inr)) : "—";
  $("tPerMSub").textContent = c.per_1m_total_tokens_inr ? "$" + (c.per_1m_total_tokens_inr / (c.inr_per_usd || 87.5)).toFixed(2) + " per 1M total tokens" : "no tokens yet";
  $("tPrompt").textContent = compact(t.prompt_tokens);

  drawChart(u.hourly);

  $("keyRows").innerHTML = (u.per_key.length ? u.per_key : []).map(k =>
    `<tr><td>${k.name}</td><td>${nf.format(k.requests)}</td><td>${compact(k.prompt_tokens)}</td>`
    + `<td>${nf.format((k.prompt_tokens||0) + (k.completion_tokens||0))}</td>`
    + `<td class="${k.errors ? "warn" : ""}">${k.errors || 0}</td><td>${ago(k.last_seen)}</td></tr>`
  ).join("") || `<tr><td colspan="6" style="text-align:left;color:var(--muted)">no traffic yet</td></tr>`;

  $("healthRows").innerHTML = [
    healthRow("engine", u.engine_health),
    healthRow("running / waiting", `${e.num_requests_running ?? "—"} / ${e.num_requests_waiting ?? "—"}`),
    healthRow("mean TTFT", e.mean_ttft_ms ? e.mean_ttft_ms.toFixed(0) + " ms" : "—"),
    healthRow("spec accept length", e.spec_accept_length ? e.spec_accept_length.toFixed(2) : "—"),
    healthRow("KV pages", e.kv_pages_total ? `${nf.format(e.kv_pages_used)} / ${nf.format(e.kv_pages_total)}` : "—"),
    healthRow("429 / 503 refused", `${nf.format(t.rate_limited)} / ${nf.format(t.rejected_overload)}`),
    healthRow("401 refused", nf.format(t.auth_failures)),
    healthRow("keys loaded", (u.keys || []).map(k => k.name).join(", ") || "—"),
    healthRow("keys file", u.key_store?.error ? `<span class="warn">${u.key_store.error}</span>` : "ok"),
    healthRow("usage db", u.storage?.error ? `<span class="warn">${u.storage.error}</span>` : (u.storage?.db_path || "—")),
    healthRow("dropped audit rows", nf.format(t.dropped_audit_rows)),
  ].join("");

  const classes = u.errors_by_class || {};
  const windowed = u.errors_by_class_window || {};
  const names = [...new Set([...Object.keys(classes), ...Object.keys(windowed)])]
    .sort((a, b) => (classes[b] || 0) - (classes[a] || 0));
  $("errorSection").hidden = names.length === 0;
  $("errorRows").innerHTML = names.map(n =>
    `<tr><td>${n}</td><td class="${classes[n] ? "warn" : ""}">${nf.format(classes[n] || 0)}</td>`
    + `<td>${nf.format(windowed[n] || 0)}</td></tr>`
  ).join("");

  const lat = u.latency || {};
  const q = (o, k) => (o && o[k] != null) ? o[k].toFixed(o[k] < 10 ? 1 : 0) : "—";
  $("latencyRows").innerHTML = [
    `<tr><td>TTFT (ms)</td><td>${q(lat.ttft_ms, "p50")}</td><td>${q(lat.ttft_ms, "p90")}</td><td>${q(lat.ttft_ms, "p99")}</td></tr>`,
    `<tr><td>TPOT (ms/token)</td><td>${q(lat.tpot_ms, "p50")}</td><td>${q(lat.tpot_ms, "p90")}</td><td>${q(lat.tpot_ms, "p99")}</td></tr>`,
    `<tr><td>samples</td><td colspan="3" style="text-align:left">${nf.format(lat.samples || 0)} completed requests in the last hour</td></tr>`,
  ].join("");

  renderQueries();
  $("foot").textContent = `refreshed ${new Date().toLocaleTimeString()} · endpoint ${m.base || "?"} · every 5 s`;
}

const clip = (s, n) => {
  if (!s) return "";
  const one = String(s).replace(/\s+/g, " ").trim();
  return one.length > n ? one.slice(0, n) + "…" : one;
};

/** Pull the last message out of a stored messages array, for the table. */
const lastUser = (raw) => {
  try {
    const msgs = JSON.parse(raw || "[]");
    for (let i = msgs.length - 1; i >= 0; i--) {
      if (msgs[i] && msgs[i].role === "user") return msgs[i].content;
    }
    return msgs.length ? msgs[msgs.length - 1].content : "";
  } catch { return raw || ""; }
};

async function renderQueries() {
  let doc;
  try {
    doc = await (await fetch("/api/queries", {cache: "no-store"})).json();
  } catch { return; }
  if (!doc || doc.enabled === false) {
    $("querySection").hidden = true;
    $("abuseSection").hidden = true;
    return;
  }
  const rows = doc.rows || [];
  $("querySection").hidden = false;
  $("queryRows").innerHTML = rows.map(r =>
    `<tr><td>${ago(r.ts)}</td><td>${r.key_name || "—"}</td><td>${r.client_ip || "—"}</td>`
    + `<td style="text-align:left">${esc(clip(lastUser(r.messages_json), 90))}</td>`
    + `<td style="text-align:left">${esc(clip(r.response_text, 90))}</td></tr>`
  ).join("") || `<tr><td colspan="5" style="text-align:left;color:var(--muted)">no queries logged yet</td></tr>`;

  const counts = doc.counts || [];
  $("abuseSection").hidden = counts.length === 0;
  $("abuseRows").innerHTML = counts.map(c =>
    `<tr><td>${esc(c.key_name ?? c.ip ?? "—")}</td><td>${c.scope}</td><td>${nf.format(c.queries)}</td>`
    + `<td>${compact(c.prompt_tokens)}</td><td>${compact(c.completion_tokens)}</td></tr>`
  ).join("");
}

const esc = (s) => String(s ?? "").replace(/[&<>"]/g, ch =>
  ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[ch]));

tick();
setInterval(tick, 5000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    box: Box = None  # type: ignore[assignment]

    def log_message(self, fmt, *args):  # quieter than the default access log
        pass

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # The browser closed a poll connection mid-write (tab reload, refresh
            # race, Ctrl-C). Nothing to recover; stay quiet.
            pass

    def handle(self) -> None:  # also swallow resets raised before/after _send
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/usage":
            status, text = self.box.fetch("/admin/usage")
            self._send(status, text.encode("utf-8"), "application/json")
            return
        if path == "/api/metrics":
            status, text = self.box.fetch("/metrics", auth=False)
            if status != 200:
                self._send(200, json.dumps({"ok": False, "base": self.box.base}).encode(), "application/json")
                return
            self._send(
                200,
                json.dumps({"ok": True, "base": self.box.base, **parse_prom(text)}).encode(),
                "application/json",
            )
            return
        if path == "/api/queries":
            # Content rows never leave this loopback server, and are only fetched at all
            # when the server has content logging on.
            status, text = self.box.fetch("/admin/queries?limit=50")
            if status != 200:
                self._send(200, b'{"enabled": false}', "application/json")
                return
            try:
                doc = json.loads(text)
            except ValueError:
                self._send(200, b'{"enabled": false}', "application/json")
                return
            if doc.get("enabled") is False:
                self._send(200, b'{"enabled": false}', "application/json")
                return
            counts = []
            for scope, param in (("key", "key"), ("ip", "ip")):
                st, body = self.box.fetch(f"/admin/queries?counts={param}")
                if st != 200:
                    continue
                try:
                    for row in json.loads(body).get("counts", [])[:10]:
                        counts.append({**row, "scope": scope})
                except ValueError:
                    continue
            doc["counts"] = counts
            self._send(200, json.dumps(doc).encode("utf-8"), "application/json")
            return
        self._send(404, b'{"error":"not found"}', "application/json")


def parse_prom(text: str) -> dict:
    gauges: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#") or "{" in line:
            continue
        name, _, value = line.rpartition(" ")
        try:
            gauges[name.strip()] = float(value)
        except ValueError:
            continue
    return {
        "running": gauges.get("qwenfast:num_requests_running"),
        "waiting": gauges.get("qwenfast:num_requests_waiting"),
        "generated": gauges.get("qwenfast:generation_tokens_total"),
        "lifetimeTps": gauges.get("qwenfast:tokens_per_second"),
        "uptimeS": gauges.get("qwenfast:uptime_seconds"),
    }


def read_admin_key(path: str | None) -> str:
    candidate = Path(path) if path else DEFAULT_ADMIN_KEY_FILE
    env = os.environ.get("QWENFAST_ADMIN_KEY")
    if env:
        return env.strip()
    if not candidate.exists():
        raise SystemExit(
            f"no admin key: {candidate} not found and QWENFAST_ADMIN_KEY unset.\n"
            "Generate one with: python3 scripts/make_api_keys.py --admin"
        )
    for line in candidate.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    raise SystemExit(f"{candidate} is empty")


class SnapshotBox:
    """Offline stand-in for Box: serves a saved /admin/usage JSON (e.g. a final
    export taken before a server was shut down). /metrics reports offline."""

    def __init__(self, path: str):
        self.path = path
        self.base = f"snapshot:{path}"
        self.candidates = [self.base]
        with open(path, "r", encoding="utf-8") as f:
            self._text = f.read()
        json.loads(self._text)  # validate early

    def fetch(self, path: str, auth: bool = True) -> tuple[int, str]:
        if path == "/admin/usage":
            return 200, self._text
        return 404, "snapshot mode: live metrics unavailable"


def print_summary(box) -> int:
    status, text = box.fetch("/admin/usage")
    if status != 200:
        print(f"error {status}: {text[:400]}", file=sys.stderr)
        return 1
    u = json.loads(text)
    t, c = u["totals"], u["cost"]
    print(f"endpoint          {box.base}")
    print(f"model             {u['model']}  ({u['engine_health']})")
    print(f"requests          {t['requests']:,}  ({t['errors']:,} errors)")
    print(f"total tokens      {t['total_tokens']:,}  (prompt {t['prompt_tokens']:,} + generated {t['completion_tokens']:,})")
    print(f"prompt tokens     {t['prompt_tokens']:,}")
    print(f"tok/s (60 s)      {u['recent_output_tokens_per_second']:.1f}")
    print(f"service uptime    {u['service_uptime_s']/3600:.2f} h")
    print(f"gpu cost          INR {c['inr']:,.0f}  (${c['usd']:,.2f})")
    if c.get("per_1m_total_tokens_inr"):
        print(f"per 1M total tokens INR {c['per_1m_total_tokens_inr']:,.0f}  (${c['per_1m_total_tokens_inr']/c.get('inr_per_usd',87.5):,.2f})")
    print(f"refused           {t['rate_limited']} rate-limited, {t['rejected_overload']} overload, "
          f"{t.get('rejected_concurrency', 0)} concurrency, {t['auth_failures']} auth")
    classes = u.get("errors_by_class") or {}
    if classes:
        print("refusals by reason:")
        for name, n in sorted(classes.items(), key=lambda kv: -kv[1]):
            print(f"  {name:<28} {n:>6,}")
    lat = u.get("latency") or {}
    if lat.get("samples"):
        tt, tp = lat.get("ttft_ms", {}), lat.get("tpot_ms", {})
        fmt = lambda v: f"{v:.0f}" if v is not None else "—"  # noqa: E731
        print(f"latency (1 h, n={lat['samples']:,})")
        print(f"  TTFT ms    p50 {fmt(tt.get('p50'))}  p90 {fmt(tt.get('p90'))}  p99 {fmt(tt.get('p99'))}")
        print(f"  TPOT ms    p50 {fmt(tp.get('p50'))}  p90 {fmt(tp.get('p90'))}  p99 {fmt(tp.get('p99'))}")
    cl = u.get("content_log") or {}
    if cl.get("enabled"):
        print(f"content log       {cl.get('rows', 0):,} rows at {cl.get('db_path')}"
              + (f" (retain {cl['retention_days']}d)" if cl.get("retention_days") else " (kept forever)"))
    print("by key:")
    for k in u["per_key"]:
        print(f"  {k['name']:<16} {k['requests']:>7,} req  {k.get('prompt_tokens',0)+k.get('completion_tokens',0):>12,} total tok  ({k['completion_tokens']:,} generated)")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default=os.environ.get("QWENFAST_BASE_URL", DEFAULT_BASE_URL))
    p.add_argument("--admin-key-file", default=None, help=f"default: {DEFAULT_ADMIN_KEY_FILE}")
    p.add_argument("--host", default="127.0.0.1", help="local bind address (keep it loopback)")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--open", action="store_true", help="open the dashboard in a browser")
    p.add_argument("--json", action="store_true", help="print a one-shot summary instead of serving")
    p.add_argument("--snapshot", default=None, metavar="FILE",
                   help="offline mode: render a saved /admin/usage JSON (e.g. an archived admin-usage.json)")
    args = p.parse_args(argv)

    box = SnapshotBox(args.snapshot) if args.snapshot else Box(args.base_url, read_admin_key(args.admin_key_file))
    if args.json:
        return print_summary(box)

    Handler.box = box
    port = args.port
    for attempt in range(20):  # fall back to the next free port if one is busy
        try:
            server = ThreadingHTTPServer((args.host, port), Handler)
            break
        except OSError as exc:
            if exc.errno not in (48, 98) or attempt == 19:  # EADDRINUSE (mac/linux)
                raise
            port += 1
    if port != args.port:
        print(f"port {args.port} busy, using {port}")
    url = f"http://{args.host}:{port}"
    print(f"qwenfast usage dashboard → {url}")
    print(f"  server: {' | '.join(box.candidates)}")
    print("  snapshot mode (archived data)." if args.snapshot else "  admin key loaded (not shown). Ctrl-C to stop.")
    if args.open:
        import webbrowser

        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
