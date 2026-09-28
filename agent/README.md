# qwenfast agent

a long running autonomous coding agent on top of the [pi](https://github.com/earendil-works/pi) sdk, served by qwenfast. it runs on your machine as a background service, takes tasks from an http api or an inbox folder, routes each one to the right model by complexity, and keeps going for days: crashes, restarts, gpu box pauses and preemptions are all recovered without losing work.

## how it fits together

```
your mac                                             jarvislabs h200 (one box)
┌────────────────────────────────────┐               ┌─────────────────────────────────────────┐
│ service  ai.qwenfast.agent         │               │ keeper  remote_agent_stack.sh           │
│  └ daemon (node, pi sdk)           │   https       │  ├ supervisor big   :8000               │
│     ├ router  small│medium│large   │ ───────────▶  │  │   qwenfast  qwen3.8-27b fp8          │
│     ├ runner  pi agent sessions    │               │  │   mtp spec decode, prefix cache      │
│     ├ store   ~/.qwenfast-code     │               │  └ supervisor small :8001               │
│     └ backend health, resume, idle │ ◀── jl cli ── │      vllm  qwen3.6-35b-a3b fp8          │
└────────────────────────────────────┘               └─────────────────────────────────────────┘
```

| tier | model | thinking | used for |
|---|---|---|---|
| small | qwen3.6-35b-a3b (3b active) | off | lookups, one file edits, summaries, shell chores |
| medium | qwen3.8-27b on qwenfast | effort medium | ordinary coding, features with tests |
| large | qwen3.8-27b on qwenfast | effort xhigh | debugging, design, refactors, anything that already failed |

## openai compatible gateway

the daemon also serves `http://127.0.0.1:7788/v1` (`/models`, `/chat/completions`), which [qwen fast code](../code/README.md) and any other openai client can use. it offers `auto` plus one model per tier, maps each to the right server and thinking settings, sends traffic through the tunnel, holds requests while a paused box wakes up, and remembers each conversation's tier across restarts (`gateway.json`). `POST /box/wake` and `POST /box/pause` start or stop the gpu box on request.

## routing

a lexical score decides the clear cases instantly. tasks near a tier boundary are graded by the small model in one short call, and that grade may move the task by one tier at most. a task can also force a tier.

an attempt that ends blocked, fails its `verify` command, errors or times out escalates: small to medium to large, then large retries. the stronger model starts a fresh session with the previous outcome and the verify output in its prompt, and inspects the partial changes left in the working tree.

## reliability

- every task transition is an atomic file write under `~/.qwenfast-code/agent/tasks/<id>/`
- a daemon killed mid task continues the interrupted pi session when it restarts
- launchd or systemd restarts the daemon on crash, login and reboot, and on macos idle sleep is held off while it runs
- pi retries provider errors with backoff, and an unreachable backend puts tasks on hold until it is back
- model traffic goes through a supervised ssh tunnel to the box (`127.0.0.1:18000` and `18001`). the jarvislabs https proxy closes any response after about two minutes, which would cut long thinking turns; it stays as the fallback path
- health probes use a fresh connection and need two failures in a row, so a proxy blip never stalls the queue
- a revoked jarvislabs key falls back to `.secrets/jarvislabs.backup.env`
- the backend keeper resumes a paused or preempted box when work arrives, reruns the idempotent bring up, and rediscovers the new endpoint urls
- after `idlePauseMinutes` with nothing queued the box is paused, so an idle agent costs storage only
- on the box, each server has its own watchdog (health polls, drain, restart with backoff) and a keeper restarts a dead watchdog
- per task wall clock, turn limit and attempt budget, and a heap limit that recycles the daemon

## speed

qwenfast keeps a turn to turn prefix cache for agent conversations. each prompt snapshots the gated deltanet state where it ends, and the next turn of the same conversation resumes from it. at a 75k token context a turn goes from 9.7 s to 0.6 s time to first token. see [engine/qwenfast/runtime/prefix_cache.py](../engine/qwenfast/runtime/prefix_cache.py).

## install

`sh code/install.sh` (see [qwen fast code](../code/README.md)) installs the daemon together with qwen fast code: it runs under launchd on macos and `systemd --user` on linux, keeps its state in `~/.qwenfast-code/agent` and puts `qfa` on your path. to run it from this checkout while developing:

```bash
cd agent && npm ci
QFA_HOME=/tmp/qfa QFA_BIG_URL=http://<gpu host>:8000 QFA_SMALL_URL=http://<gpu host>:8001 QFA_API_KEY=<key> QFA_JARVIS=0 node src/daemon.ts
```

a launchd job may not read `~/Desktop`, `~/Documents` or `~/Downloads`. to let background tasks work in projects there, give `~/.qwenfast-code/toolchain/node/bin/node` full disk access in system settings.

## use

```bash
qfa submit "add a --json flag to the export command and a test for it" --cwd ~/code/app --verify "pytest -q"
qfa submit "how many python files are in this repo?" --cwd ~/code/app --wait
qfa ls running
qfa events <id> 50
qfa cancel <id>
qfa health
```

or drop a file into `~/.qwenfast-code/agent/inbox/`:

```
cwd: /Users/me/code/app
tier: large
verify: make test

find out why the nightly export job leaks file handles and fix it.
```

http api on `127.0.0.1:7788`: `POST /tasks`, `GET /tasks`, `GET /tasks/<id>`, `GET /tasks/<id>/events`, `POST /tasks/<id>/cancel`, `POST /tasks/<id>/retry`, `GET /health`.

## configuration

`~/.qwenfast-code/agent/config.json` overrides the defaults in [src/config.ts](src/config.ts), for example:

```json
{
  "concurrency": 6,
  "jarvis": { "idlePauseMinutes": 0, "resumeSpot": true },
  "task": { "timeoutMinutes": 360, "maxAttempts": 5 },
  "tiers": { "small": { "thinking": "low" } }
}
```

`idlePauseMinutes: 0` keeps the box running. `resumeSpot` resumes at half price with preemption risk, which the keeper recovers from.

## tests

```bash
npm test          # router, store, resume after kill -9, escalation, verify, inbox (no gpu)
npm run typecheck
```

## safety

the agent runs shell commands and edits files with your user's permissions, unattended. point `cwd` at projects under version control and keep secrets out of task working directories.
