// backend keeper: knows where the two model servers are, whether they answer, and keeps the gpu
// box alive (or asleep) on jarvislabs.
//
// every tick (20 s):
//   * health check both endpoints (GET /health; no key needed on either server);
//   * with jarvis on: read the box state, resume a paused box when there is work, re-run the
//     idempotent bring-up script when the box runs but a server stays down, and pause the box
//     after `idlePauseMinutes` with nothing queued;
//   * re-discover endpoint urls after a resume (the machine id and urls change).
// tasks never fail because the backend is down: they wait in `waitFor` until it is back, up to
// the task's backend wait budget.

import { execFile, spawn, type ChildProcess } from "node:child_process";
import { request as httpRequest } from "node:http";
import { request as httpsRequest } from "node:https";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import type { Config, EndpointName } from "./config.ts";

export interface EndpointState {
  url?: string;
  healthy: boolean;
  lastHealthyAt?: number;
  lastCheckAt?: number;
  unhealthySince?: number;
  lastError?: string;
  /** consecutive failed probes; one blip of the https proxy must not mark a tier down */
  failures?: number;
  /** tunnel url (http://127.0.0.1:port) while the ssh tunnel is up */
  direct?: string;
  /** which path requests take right now */
  via?: "tunnel" | "proxy";
}

/** `ssh -o ... user@host` -> "user@host" */
export function sshTarget(sshCommand: string | undefined): string | undefined {
  return sshCommand?.split(/\s+/).reverse().find((t) => /^[\w.-]+@[\w.-]+$/.test(t));
}

/** one supervised `ssh -N -L` process forwarding the two server ports to localhost */
export class SshTunnel {
  private child?: ChildProcess;
  private target?: string;
  private startedAt = 0;
  private backoff = 2000;
  private nextStart = 0;
  private readonly forwards: Array<[number, number]>;
  private readonly log: (m: string) => void;

  constructor(forwards: Array<[number, number]>, log: (m: string) => void) {
    this.forwards = forwards;
    this.log = log;
  }

  /** start or keep the tunnel to `target`; restarts it when the target changes or ssh exited */
  ensure(target: string | undefined): void {
    if (!target) return;
    if (this.child && this.target !== target) this.stop();
    if (this.child || Date.now() < this.nextStart) return;
    this.target = target;
    const args = ["-N", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
      "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3", "-o", "ExitOnForwardFailure=yes", "-o", "LogLevel=ERROR"];
    for (const [local, remote] of this.forwards) args.push("-L", `127.0.0.1:${local}:127.0.0.1:${remote}`);
    args.push(target);
    const child = spawn("ssh", args, { stdio: ["ignore", "ignore", "pipe"] });
    let err = "";
    child.stderr?.on("data", (b) => (err = (err + b.toString()).slice(-300)));
    child.on("exit", (code) => {
      if (this.child !== child) return;
      this.child = undefined;
      const lived = Date.now() - this.startedAt;
      this.backoff = lived > 60_000 ? 2000 : Math.min(this.backoff * 2, 60_000);
      this.nextStart = Date.now() + this.backoff;
      this.log(`ssh tunnel to ${target} exited (${code}) after ${Math.round(lived / 1000)}s ${err.trim()}; retry in ${this.backoff / 1000}s`);
    });
    this.child = child;
    this.startedAt = Date.now();
    this.log(`ssh tunnel to ${target} starting`);
  }

  /** running long enough for the forwards to be listening */
  up(): boolean {
    return !!this.child && Date.now() - this.startedAt > 2000;
  }

  stop(): void {
    const c = this.child;
    this.child = undefined;
    c?.kill("SIGTERM");
  }
}

export interface BoxState {
  machineId?: number;
  status?: string;
  lastCheckAt?: number;
  lastAction?: string;
  lastActionAt?: number;
}

type Logger = (msg: string) => void;

const PORTS: Record<EndpointName, number> = { big: 8000, small: 8001 };

export function parseEnvFile(text: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const line of text.split("\n")) {
    const m = line.match(/^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$/);
    if (!m) continue;
    out[m[1]] = m[2].trim().replace(/^(['"])(.*)\1$/, "$2");
  }
  return out;
}

/** GET url on a fresh connection (no keep-alive pool): a pooled socket the https proxy has
 * silently stalled would otherwise hang every probe until its timeout while the server is fine */
export function probe(url: string, timeoutMs: number): Promise<number> {
  return new Promise((resolve, reject) => {
    const req = (url.startsWith("https:") ? httpsRequest : httpRequest)(url, { method: "GET", agent: false, timeout: timeoutMs }, (res) => {
      res.resume();
      res.on("end", () => resolve(res.statusCode ?? 0));
      res.on("error", reject);
    });
    req.on("timeout", () => req.destroy(new Error("health probe timed out")));
    req.on("error", reject);
    req.end();
  });
}

/** endpoints[0] is the template's port 6006; the rest follow http_ports in order */
export function endpointsFromMachine(m: { http_ports?: string; endpoints?: string[] | null }): Partial<Record<EndpointName, string>> {
  const ports = (m.http_ports ?? "").split(",").map((p) => Number(p.trim())).filter(Boolean);
  const eps = m.endpoints ?? [];
  const out: Partial<Record<EndpointName, string>> = {};
  for (const [name, port] of Object.entries(PORTS) as Array<[EndpointName, number]>) {
    const i = ports.indexOf(port);
    if (i >= 0 && eps[i + 1]) out[name] = eps[i + 1];
  }
  return out;
}

export class Backend {
  readonly endpoints: Record<EndpointName, EndpointState> = { big: { healthy: false }, small: { healthy: false } };
  readonly box: BoxState & { sshTarget?: string } = {};
  private readonly tunnel?: SshTunnel;
  private timer?: NodeJS.Timeout;
  private lastDemandAt = Date.now();
  private wakeUntil = 0;
  private jlQueue: Promise<unknown> = Promise.resolve();
  private bringupAt = 0;
  private ticking = false;

  private readonly cfg: Config;
  private readonly log: Logger;
  private readonly demand: () => number;

  constructor(cfg: Config, log: Logger, demand: () => number) {
    this.cfg = cfg;
    this.log = log;
    this.demand = demand;
    for (const n of ["big", "small"] as EndpointName[]) this.endpoints[n].url = cfg.endpoints[n];
    if (cfg.jarvis.enabled && cfg.jarvis.tunnel) {
      const base = cfg.jarvis.tunnelBasePort;
      this.tunnel = new SshTunnel([[base, PORTS.big], [base + 1, PORTS.small]], log);
      this.endpoints.big.direct = `http://127.0.0.1:${base}`;
      this.endpoints.small.direct = `http://127.0.0.1:${base + 1}`;
    }
    if (cfg.jarvis.enabled && existsSync(cfg.jarvis.machineIdFile)) {
      const id = Number(readFileSync(cfg.jarvis.machineIdFile, "utf8").trim());
      if (id) this.box.machineId = id;
    }
  }

  start(): void {
    void this.tick();
    this.timer = setInterval(() => void this.tick(), 20_000);
    this.timer.unref();
  }

  stop(): void {
    if (this.timer) clearInterval(this.timer);
    this.tunnel?.stop();
  }

  url(name: EndpointName): string | undefined {
    const ep = this.endpoints[name];
    return ep.via === "tunnel" && ep.direct ? ep.direct : ep.url;
  }

  healthy(name: EndpointName): boolean {
    return this.endpoints[name].healthy;
  }

  /** resolves true once the endpoint answers, false when the deadline passes or the signal aborts */
  async waitFor(name: EndpointName, deadlineMs: number, signal?: AbortSignal): Promise<boolean> {
    const end = Date.now() + deadlineMs;
    while (!this.endpoints[name].healthy) {
      if (Date.now() > end || signal?.aborted) return false;
      this.lastDemandAt = Date.now();
      await new Promise((r) => setTimeout(r, 5000));
    }
    return true;
  }

  noteDemand(): void {
    this.lastDemandAt = Date.now();
  }

  /** treat the box as wanted for `minutes` even with nothing queued (`qfc up`); resumes a paused box */
  wake(minutes = 15): void {
    this.wakeUntil = Date.now() + minutes * 60_000;
    this.lastDemandAt = Date.now();
    this.box.lastCheckAt = 0; // re-read the box state on the next tick
    void this.tick();
  }

  /** pause the box now (`qfc down`); a new request or task resumes it again */
  async pauseNow(): Promise<string> {
    if (!this.cfg.jarvis.enabled || !this.box.machineId) return "no managed box";
    this.wakeUntil = 0;
    this.lastDemandAt = 0;
    this.action("pausing box on request");
    const out = JSON.parse(await this.jl(["pause", String(this.box.machineId), "--yes", "--json"], 600_000)) as { error?: string };
    if (out.error) throw new Error(out.error);
    this.box.status = "Paused";
    this.tunnel?.stop();
    for (const n of ["big", "small"] as EndpointName[]) this.endpoints[n].healthy = false;
    return "paused";
  }

  snapshot() {
    return { endpoints: this.endpoints, box: this.box, idleMinutes: Math.round((Date.now() - this.lastDemandAt) / 60000) };
  }

  async tick(): Promise<void> {
    if (this.ticking) return;
    this.ticking = true;
    try {
      if (this.demand() > 0) this.lastDemandAt = Date.now();
      if (this.tunnel && this.box.status === "Running") this.tunnel.ensure(this.box.sshTarget);
      await Promise.all((["big", "small"] as EndpointName[]).map((n) => this.check(n)));
      // requests from anything else (a benchmark, another client) keep the box awake too
      if (await this.externalLoad()) this.lastDemandAt = Date.now();
      if (this.cfg.jarvis.enabled && !this.box.machineId) await this.findBoxByName().catch(() => false);
      if (this.cfg.jarvis.enabled && this.box.machineId) await this.manageBox();
    } catch (err) {
      this.log(`backend tick error: ${String(err)}`);
    } finally {
      this.ticking = false;
    }
  }

  /** true when either server reports requests running or waiting (qwenfast and vllm metrics) */
  private async externalLoad(): Promise<boolean> {
    for (const name of ["big", "small"] as EndpointName[]) {
      const base = this.url(name);
      if (!base || !this.endpoints[name].healthy) continue;
      try {
        const r = await fetch(`${base.replace(/\/+$/, "")}/metrics`, { signal: AbortSignal.timeout(5000) });
        const text = await r.text();
        for (const m of text.matchAll(/^(?:qwenfast:num_requests_(?:running|waiting)|vllm:num_requests_(?:running|waiting))(?:\{[^}]*\})?\s+([0-9.e+]+)/gm)) {
          if (Number(m[1]) > 0) return true;
        }
      } catch {
        /* metrics are advisory */
      }
    }
    return false;
  }

  private async check(name: EndpointName): Promise<void> {
    const ep = this.endpoints[name];
    ep.lastCheckAt = Date.now();
    let ok = false;
    // the tunnel first (no response length cap), the https proxy second
    const paths: Array<["tunnel" | "proxy", string | undefined]> = [
      ["tunnel", this.tunnel?.up() ? ep.direct : undefined],
      ["proxy", ep.url],
    ];
    for (const [via, base] of paths) {
      if (!base) continue;
      try {
        const status = await probe(`${base.replace(/\/+$/, "")}/health`, 10_000);
        if (status === 200) {
          if (ep.via !== via) this.log(`endpoint ${name} now via ${via} (${base})`);
          ep.via = via;
          ok = true;
          break;
        }
        ep.lastError = `${via} http ${status}`;
      } catch (err) {
        ep.lastError = `${via}: ${String((err as Error)?.message ?? err).slice(0, 120)}`;
      }
    }
    if (!ep.url && !ep.direct) ep.lastError = "no url";
    if (ok) {
      if (!ep.healthy) this.log(`endpoint ${name} healthy (${ep.url})`);
      ep.healthy = true;
      ep.failures = 0;
      ep.lastHealthyAt = Date.now();
      ep.unhealthySince = undefined;
      ep.lastError = undefined;
    } else {
      ep.failures = (ep.failures ?? 0) + 1;
      if (ep.healthy && ep.failures < 2) return; // wait for a second failure before declaring it down
      if (ep.healthy) this.log(`endpoint ${name} went unhealthy: ${ep.lastError}`);
      ep.healthy = false;
      ep.unhealthySince ??= Date.now();
    }
  }

  // -- jarvislabs box lifecycle -------------------------------------------------------------

  private jlEnv(file: string): NodeJS.ProcessEnv {
    const env = { ...process.env };
    if (existsSync(file)) {
      // jl authenticates from JL_API_KEY when set, else from its stored login (`jl setup`)
      Object.assign(env, parseEnvFile(readFileSync(file, "utf8")));
    }
    env.PATH = `${env.PATH ?? ""}:${process.env.HOME}/.local/bin`;
    return env;
  }

  private jlOnce(args: string[], env: NodeJS.ProcessEnv, timeoutMs: number): Promise<string> {
    return new Promise<string>((resolve, reject) => {
      execFile("jl", args, { env, timeout: timeoutMs, maxBuffer: 8 << 20 }, (err, stdout, stderr) => {
        // --json errors come back on stdout as {"error": ...}; treat them as failures too
        const jsonErr = /^\s*\{\s*"error"/.test(stdout);
        if (err || jsonErr) reject(new Error(`jl ${args[0]} failed: ${String(stderr || stdout || err?.message).slice(0, 300)}`));
        else resolve(stdout);
      });
    });
  }

  /** jl calls are serialized: two lifecycle mutations must never interleave. a failure with the
   * primary credentials is retried once with the backup key (a revoked or expired key must not
   * strand a paused box). */
  private jl(args: string[], timeoutMs = 300_000): Promise<string> {
    const j = this.cfg.jarvis;
    const run = async () => {
      try {
        return await this.jlOnce(args, this.jlEnv(j.envFile), timeoutMs);
      } catch (err) {
        const msg = String((err as Error).message);
        const authLike = /auth|401|403|token|api key|unauthori[sz]ed|login|forbidden/i.test(msg);
        if (!authLike || !existsSync(j.backupEnvFile)) throw err;
        this.log(`jl ${args[0]} failed with the primary key; retrying with the backup key`);
        return this.jlOnce(args, this.jlEnv(j.backupEnvFile), timeoutMs);
      }
    };
    const p = this.jlQueue.then(run, run);
    this.jlQueue = p.catch(() => undefined);
    return p;
  }

  private setMachineId(id: number): void {
    if (this.box.machineId === id) return;
    this.log(`box machine id ${this.box.machineId} -> ${id}`);
    this.box.machineId = id;
    try {
      writeFileSync(this.cfg.jarvis.machineIdFile, `${id}\n`);
    } catch (err) {
      this.log(`could not persist machine id: ${String(err)}`);
    }
  }

  private action(what: string): void {
    this.box.lastAction = what;
    this.box.lastActionAt = Date.now();
    this.log(`box: ${what}`);
  }

  /** the box id changes on every resume; when the stored one is gone, find the box by its name */
  private async findBoxByName(): Promise<boolean> {
    const list = JSON.parse(await this.jl(["list", "--json"], 60_000)) as Array<{ machine_id: number; name?: string; status?: string }>;
    const hit = list.filter((m) => m.name === this.cfg.jarvis.boxName).sort((a, b) => b.machine_id - a.machine_id)[0];
    if (!hit) return false;
    this.setMachineId(hit.machine_id);
    return true;
  }

  private async refreshBox(): Promise<void> {
    let out: string;
    try {
      out = await this.jl(["get", String(this.box.machineId), "--json"], 60_000);
    } catch (err) {
      if (!/not found/i.test(String(err)) || !(await this.findBoxByName())) throw err;
      out = await this.jl(["get", String(this.box.machineId), "--json"], 60_000);
    }
    const m = JSON.parse(out) as { machine_id?: number; status?: string; http_ports?: string; endpoints?: string[] | null; error?: string; ssh_command?: string };
    if (m.error) throw new Error(m.error);
    if (m.machine_id) this.setMachineId(m.machine_id);
    this.box.status = m.status;
    this.box.sshTarget = sshTarget(m.ssh_command) ?? this.box.sshTarget;
    if (m.status !== "Running") this.tunnel?.stop();
    this.box.lastCheckAt = Date.now();
    if (m.status === "Running") {
      const found = endpointsFromMachine(m);
      for (const n of ["big", "small"] as EndpointName[]) {
        if (!this.cfg.endpoints[n] && found[n] && found[n] !== this.endpoints[n].url) {
          this.log(`endpoint ${n} -> ${found[n]}`);
          this.endpoints[n].url = found[n];
        }
      }
    }
  }

  private async manageBox(): Promise<void> {
    const j = this.cfg.jarvis;
    const now = Date.now();
    const anyDown = !this.endpoints.big.healthy || !this.endpoints.small.healthy;
    // the box state is cheap to read; read it often when something is wrong, rarely otherwise
    const every = anyDown ? 60_000 : 600_000;
    if (!this.box.lastCheckAt || now - this.box.lastCheckAt > every) {
      try {
        await this.refreshBox();
      } catch (err) {
        this.log(`box status unavailable: ${String(err)}`);
        return;
      }
    }
    const demand = this.demand() > 0 || now < this.wakeUntil;
    const status = this.box.status ?? "";
    const idleMin = (now - this.lastDemandAt) / 60000;

    if (status === "Paused" && demand && j.autoResume) {
      this.action("resuming paused box for queued work");
      const args = ["resume", String(this.box.machineId), "--http-ports", "8000,8001,8080", "--yes", "--json"];
      if (j.resumeSpot) args.push("--spot");
      try {
        const out = JSON.parse(await this.jl(args, 900_000)) as { machine_id?: number; error?: string };
        if (out.error) throw new Error(out.error);
        if (out.machine_id) this.setMachineId(out.machine_id);
        this.box.lastCheckAt = 0;
        this.bringupAt = 0;
        await this.refreshBox();
        await this.bringup("after resume");
      } catch (err) {
        this.log(`resume failed: ${String(err)}`);
      }
      return;
    }

    if (status === "Running") {
      const bigDown = this.endpoints.big.unhealthySince;
      const downFor = bigDown ? (now - bigDown) / 60000 : 0;
      if (demand && downFor > j.rebootstrapAfterMinutes && now - this.bringupAt > 30 * 60000) {
        await this.bringup(`big endpoint down ${downFor.toFixed(0)} min`);
      }
      if (j.idlePauseMinutes > 0 && !demand && idleMin > j.idlePauseMinutes) {
        this.action(`pausing box after ${idleMin.toFixed(0)} idle minutes`);
        try {
          const out = JSON.parse(await this.jl(["pause", String(this.box.machineId), "--yes", "--json"], 600_000)) as { error?: string };
          if (out.error) throw new Error(out.error);
          this.box.status = "Paused";
          for (const n of ["big", "small"] as EndpointName[]) this.endpoints[n].healthy = false;
        } catch (err) {
          this.log(`pause failed: ${String(err)}`);
        }
      }
    }
  }

  private async bringup(reason: string): Promise<void> {
    this.bringupAt = Date.now();
    this.action(`running bring-up (${reason})`);
    try {
      const out = await this.jl(
        ["run", "--on", String(this.box.machineId), "--no-follow", "--json", "--yes", "--", "sh", this.cfg.jarvis.bringupScript],
        300_000,
      );
      const runId = (out.match(/"run_id"\s*:\s*"([^"]+)"/) ?? [])[1];
      this.log(`bring-up started ${runId ?? ""}`);
    } catch (err) {
      this.log(`bring-up failed to start: ${String(err)}`);
    }
  }
}
