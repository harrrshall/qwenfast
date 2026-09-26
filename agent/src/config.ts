// daemon configuration: defaults, overridden by $QFA_HOME/config.json, overridden by env.

import { existsSync, mkdirSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join, resolve } from "node:path";

export type TierName = "small" | "medium" | "large";
export type EndpointName = "big" | "small";
export type Thinking = "off" | "low" | "medium" | "high";

export interface TierConfig {
  /** which server answers this tier */
  endpoint: EndpointName;
  /** model name sent on the wire (vllm checks it, qwenfast accepts anything) */
  model: string;
  thinking: Thinking;
  contextWindow: number;
  maxTokens: number;
}

export interface Config {
  home: string;
  host: string;
  port: number;
  /** tasks run in parallel; the big tier serves 24 sequences, the small 32 */
  concurrency: number;
  /** default working directory for tasks that do not name one */
  workspace: string;
  apiKeyFile: string;
  /** fixed base urls (no /v1). when unset they are discovered from jarvislabs */
  endpoints: Partial<Record<EndpointName, string>>;
  jarvis: {
    enabled: boolean;
    envFile: string;
    /** second credential set, tried when a jl call fails with the primary one */
    backupEnvFile: string;
    machineIdFile: string;
    /** resume a paused box when work arrives */
    autoResume: boolean;
    /** pause the box after this many idle minutes; 0 keeps it running */
    idlePauseMinutes: number;
    /** resume as spot (half price, can be preempted; the keeper resumes it again) */
    resumeSpot: boolean;
    bringupScript: string;
    /** instance name used to find the box again when its id is unknown or stale */
    boxName: string;
    /** endpoint unhealthy this long while the box runs -> re-run the bring-up script */
    rebootstrapAfterMinutes: number;
    /** reach the servers through an ssh tunnel to the box. the jarvislabs https proxy closes any
     * response after about two minutes, which truncates long thinking turns; it stays the fallback */
    tunnel: boolean;
    /** local ports for the tunnel: base -> big (8000), base + 1 -> small (8001) */
    tunnelBasePort: number;
  };
  tiers: Record<TierName, TierConfig>;
  task: {
    timeoutMinutes: number;
    maxAttempts: number;
    maxTurns: number;
    /** minutes to wait for a backend before an attempt counts as failed */
    backendWaitMinutes: number;
  };
  router: {
    /** ask the small model to grade complexity when the heuristic is unsure */
    llmJudge: boolean;
  };
  log: { maxBytes: number; keep: number };
}

const REPO = resolve(new URL("../..", import.meta.url).pathname);

/** a secret installed next to the daemon (launchd/install.sh copies them out of the repo, because a
 * launchd job may not read ~/Desktop or ~/Documents) wins over the repo's .secrets/ */
function secretPath(home: string, name: string): string {
  const installed = join(home, "secrets", name);
  return existsSync(installed) ? installed : join(REPO, ".secrets", name);
}

export function defaultConfig(home: string): Config {
  return {
    home,
    host: "127.0.0.1",
    port: 7788,
    concurrency: 4,
    workspace: join(home, "workspace"),
    apiKeyFile: secretPath(home, "agent_key"),
    endpoints: {},
    jarvis: {
      enabled: true,
      envFile: secretPath(home, "jarvislabs.env"),
      backupEnvFile: secretPath(home, "jarvislabs.backup.env"),
      machineIdFile: secretPath(home, "agent_box_id"),
      autoResume: true,
      idlePauseMinutes: 60,
      resumeSpot: false,
      bringupScript: "/home/remote_agent_bringup.sh",
      boxName: "qwenfast-agent",
      rebootstrapAfterMinutes: 20,
      tunnel: true,
      tunnelBasePort: 18000,
    },
    tiers: {
      small: { endpoint: "small", model: "qwen3.6-35b-a3b", thinking: "off", contextWindow: 131072, maxTokens: 16384 },
      medium: { endpoint: "big", model: "qwen3.8-27b", thinking: "medium", contextWindow: 131072, maxTokens: 32768 },
      large: { endpoint: "big", model: "qwen3.8-27b", thinking: "high", contextWindow: 131072, maxTokens: 32768 },
    },
    task: { timeoutMinutes: 180, maxAttempts: 4, maxTurns: 400, backendWaitMinutes: 90 },
    router: { llmJudge: true },
    log: { maxBytes: 20 * 1024 * 1024, keep: 5 },
  };
}

function isObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

export function deepMerge<T>(base: T, over: unknown): T {
  if (!isObject(base) || !isObject(over)) return (over === undefined ? base : (over as T));
  const out: Record<string, unknown> = { ...(base as Record<string, unknown>) };
  for (const [k, v] of Object.entries(over)) {
    out[k] = k in out && isObject(out[k]) && isObject(v) ? deepMerge(out[k], v) : v;
  }
  return out as T;
}

export function loadConfig(env: NodeJS.ProcessEnv = process.env): Config {
  const home = resolve(env.QFA_HOME ?? join(homedir(), ".qwenfast-agent"));
  let cfg = defaultConfig(home);
  const file = join(home, "config.json");
  if (existsSync(file)) cfg = deepMerge(cfg, JSON.parse(readFileSync(file, "utf8")));
  if (env.QFA_PORT) cfg.port = Number(env.QFA_PORT);
  if (env.QFA_BIG_URL) cfg.endpoints.big = env.QFA_BIG_URL;
  if (env.QFA_SMALL_URL) cfg.endpoints.small = env.QFA_SMALL_URL;
  if (env.QFA_API_KEY_FILE) cfg.apiKeyFile = env.QFA_API_KEY_FILE;
  if (env.QFA_JARVIS === "0") cfg.jarvis.enabled = false;
  for (const d of [cfg.home, cfg.workspace, join(cfg.home, "tasks"), join(cfg.home, "inbox"), join(cfg.home, "logs")]) {
    mkdirSync(d, { recursive: true });
  }
  return cfg;
}

export function readApiKey(cfg: Config): string {
  if (process.env.QFA_API_KEY) return process.env.QFA_API_KEY;
  return readFileSync(cfg.apiKeyFile, "utf8").trim();
}
