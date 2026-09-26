# qwen fast code

an open source coding agent for your terminal, with the opencode interface, running end to end on your mac or linux machine against your own qwen models. one command, `qfc`, opens the tui in any project. every task is routed to the model that fits it, sessions keep running for hours when you close the terminal, and every terminal sees the same sessions.

```
qfc                      open the tui on this project
qfc ~/code/app           open it on another project
qfc run "fix the flaky test in tests/test_io.py"
qfc status               models, gpu box, routing, server
qfc up | qfc down        wake or pause the gpu box
```

## how it fits together

```
your machine                                                     gpu box (jarvislabs h200)
┌──────────────────────────────────────────────────────────┐     ┌───────────────────────────┐
│ qfc tui ──┐                                              │     │ qwenfast   qwen3.8-27b    │
│ qfc tui ──┼─▶ qfc server :4097  ─▶ gateway :7788  ─ssh──▶│─────│ vllm       qwen3.6-35b-a3b│
│ qfc run ──┘   sessions, tools,     router, tunnel,       │     │ watchdogs, keeper          │
│               agents (opencode)    box wake and pause    │     └───────────────────────────┘
└──────────────────────────────────────────────────────────┘
```

- **qfc** is opencode `v1.18.32` with qwen fast code branding and its own directories (`~/.config/qwenfast-code`, `~/.local/share/qwenfast-code`), so it never touches an opencode install. the tui, agents, tools, lsp, mcp, sessions and keybindings are upstream opencode.
- **the server** (`qfc serve`, started for you) owns every session. a tui or `qfc run` attaches to it, so a task keeps going when its terminal closes, and any terminal can open any session again (`qfc -s <id>`, or the session list in the tui).
- **the gateway** is the local daemon from [../agent](../agent/README.md): an openai compatible endpoint that routes, reaches the models through an ssh tunnel (the jarvislabs https proxy cuts responses after two minutes), wakes a paused gpu box when you start working and pauses it after an idle hour.

## models and routing

pick a model with `ctrl+p` or `/models`; the default is **auto**.

| model | what runs | used for |
|---|---|---|
| auto | the router picks one of the three below per conversation | everything |
| qwen3.8-27b-xhigh | qwen3.8-27b, thinking at xhigh effort | debugging, design, refactors, performance |
| qwen3.8-27b | qwen3.8-27b, thinking at medium effort | features, fixes, tests |
| qwen3.6-35b-a3b | qwen3.6-35b-a3b (3b active), no thinking | lookups, small edits, summaries, session titles |

published scores behind the split ([qwen3.8-27b](https://huggingface.co/Qwen/Qwen3.8-27B), [qwen3.6-35b-a3b](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)):

| | qwen3.8-27b | qwen3.6-35b-a3b |
|---|---|---|
| swe-bench pro | 61.7 | 49.5 |
| terminal-bench | 73.0 (2.1) | 51.5 (2.0) |
| livecodebench v6 | 90.3 | 80.4 |
| gpqa | 89.2 | 86.0 |

the small model is close on knowledge and short code and far behind on long, agentic engineering, while activating a ninth of the weights per token. so auto sends chores to it and real engineering to qwen3.8-27b, with full reasoning effort for the hardest problems. how it decides:

- a lexical score of the message you typed (not the attached context) settles clear cases; borderline ones are graded by the small model in one short call, which can move the choice by one tier at most
- a conversation keeps the highest tier it has needed, so a follow up never lands on a weaker model mid task
- a turn that keeps looping on the small model is promoted to qwen3.8-27b

## install

macos (apple silicon or intel) and linux (x64 or arm64). needs `curl`, `git`, `python3`, `unzip`, `tar`, `ssh`, and a c++ toolchain for two native modules (`xcode-select --install` on macos, `build-essential` on debian and ubuntu). the installer checks and prints the exact command when something is missing.

```bash
sh code/install.sh
```

the installer builds everything from pinned sources with a pinned toolchain, identical on both systems:

- [toolchain.env](toolchain.env) pins bun, node, the opencode tag and the qfc version; [scripts/toolchain.sh](scripts/toolchain.sh) downloads exactly those builds for your os and cpu and checks them against the publishers' sha256 lists
- [scripts/build.sh](scripts/build.sh) checks out the pinned opencode tag, applies [patches/apply.py](patches/apply.py) (asserted edits: name, directories, wordmark, user facing text) and compiles one self contained binary
- services: launchd on macos, `systemd --user` on linux, and a small supervisor where neither exists. on macos the server runs from your terminal session so it can open projects under `~/Desktop` and `~/Documents`, which macos keeps from background jobs

gpu backend: with `.secrets/agent_key` and `.secrets/agent_box_id` in this repo (and `jl` logged in, or `.secrets/jarvislabs.backup.env`), the gateway manages the jarvislabs box. for any other pair of openai compatible servers:

```bash
QFC_API_KEY=... QFC_BIG_URL=http://host:8000 QFC_SMALL_URL=http://host:8001 sh code/install.sh
```

`~/.config/qwenfast-code/opencode.json` is written once and then yours to change: agents, permissions, mcp servers, keybinds, themes, all as in the [opencode docs](https://opencode.ai/docs/).

## test

```bash
npm --prefix agent test                                    # gateway, router, daemon (no gpu)
docker build -f code/test/linux.Dockerfile -t qfc-linux .  # clean linux machine
docker run --rm -v <secrets>:/secrets:ro -v ~/.ssh:/home/ubuntu/.ssh:ro qfc-linux sh code/test/linux_e2e.sh
```

## uninstall

```bash
sh ~/.qwenfast-code/services.sh uninstall
rm -rf ~/.qwenfast-code ~/.local/bin/qfc ~/.config/qwenfast-code ~/.local/share/qwenfast-code
```

## license

mit. qwen fast code is built from [opencode](https://github.com/sst/opencode) (mit) and the qwenfast engine in this repository.
