# qwenfast code

an open source coding agent for your terminal. it has the opencode interface, runs end to end on your mac or linux machine, and talks to your own qwen models. one command, `qfc`, opens it in any project. every task goes to the model that fits it, sessions keep running for hours after you close the terminal, and every terminal sees the same sessions.

```
qfc                      open qwenfast code on this project
qfc ~/code/app           open it on another project
qfc run "fix the flaky test in tests/test_io.py"
qfc status               models, gpu box, routing and server
qfc up | qfc down        wake or pause the gpu box
```

## getting started

### 1. check the prerequisites

qwenfast code runs on macos (apple silicon or intel) and linux (x64 or arm64). it needs `curl`, `git`, `python3`, `unzip`, `tar`, `ssh` and a c++ toolchain:

```bash
xcode-select --install                                                          # macos
sudo apt-get install -y curl git python3 unzip xz-utils openssh-client build-essential   # debian, ubuntu
```

### 2. pick a model backend

qwenfast code talks to two openai compatible servers: a big model for real engineering and a small one for quick chores. choose one of three ways to provide them.

**a. a jarvislabs gpu box, managed for you.** install the [jarvislabs cli](https://jarvislabs.ai/docs), log in, and run the setup script. it creates an h200 box, uploads the engine and starts both models. the first start takes 15 to 25 minutes, later ones about two.

```bash
uv tool install jarvislabs && jl setup
sh scripts/jarvis_setup.sh
```

after this the gateway wakes the box when you start working, pauses it after an idle hour and reaches it through an ssh tunnel.

**b. your own gpu machine.** follow [serve the models](../README.md#quick-start) on a linux machine with an h200 or similar. note its address and the key in `~/.qwenfast/secrets/agent_key`.

**c. any openai compatible servers.** any endpoint that serves chat completions with tool calls works. the router then sends large and medium work to the big url and chores to the small url.

### 3. install

from the root of this repository:

```bash
sh code/install.sh                                                     # backend a
QFC_API_KEY=<key> QFC_BIG_URL=http://<host>:8000 QFC_SMALL_URL=http://<host>:8001 sh code/install.sh   # b or c
```

the installer checks the prerequisites, downloads a pinned and checksum verified toolchain, builds the `qfc` binary from source (about a minute) and starts two background services. it links `qfc` and `qfa` into `~/.local/bin`; add that directory to your `PATH` if it is not there yet.

### 4. run it

```bash
qfc status        # both models should say "up"
cd ~/code/app
qfc               # the tui opens on this project
```

type a task and press enter. the session list (`ctrl+p` then "sessions") holds every conversation; close the terminal at any time and open it again with `qfc`. `qfc run "<task>"` runs one task without the tui and returns when it is done.

## how it fits together

```
your machine                                                     gpu server
┌──────────────────────────────────────────────────────────┐     ┌────────────────────────────┐
│ qfc tui ──┐                                              │     │ qwenfast   qwen3.8-27b     │
│ qfc tui ──┼─▶ qfc server :4097 ─▶ gateway :7788 ─── ssh ─┼────▶│ vllm       qwen3.6-35b-a3b │
│ qfc run ──┘   sessions, tools     routing, tunnel,       │     │ watchdogs and keeper       │
│               and agents          gpu wake and pause     │     └────────────────────────────┘
└──────────────────────────────────────────────────────────┘
```

- **qfc** is opencode `v1.18.32` with qwenfast code branding and its own directories (`~/.config/qwenfast-code`, `~/.local/share/qwenfast-code`), so it leaves any opencode install alone. the tui, agents, tools, lsp, mcp, sessions and keybindings are upstream opencode.
- **the server** owns every session. the tui and `qfc run` attach to it, so a task keeps going when its terminal closes and any terminal can open any session again (`qfc -s <id>`).
- **the gateway** is the local daemon from [../agent](../agent/README.md): an openai compatible endpoint on `127.0.0.1:7788` that routes each conversation, reaches the models and manages the gpu box.

## models and routing

pick a model with `ctrl+p` or `/models`. the default is **auto**.

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

the small model is close on knowledge and short code and falls well behind on long agentic engineering, while activating a ninth of the weights per token. auto therefore sends chores to it and engineering to qwen3.8-27b, with full reasoning effort for the hardest problems:

- a lexical score of the message you typed settles clear cases, and the small model grades borderline ones in one short call that can move the choice by one tier at most
- a conversation keeps the highest tier it has needed, so a follow up stays on the stronger model mid task
- a turn that keeps looping on the small model moves up to qwen3.8-27b

## troubleshooting

| symptom | what to do |
|---|---|
| `qfc: command not found` | add `~/.local/bin` to your `PATH` |
| `qfc status` shows a model "down" | `qfc up` wakes a paused box; a fresh box needs up to 25 minutes on its first start |
| the gateway or server does not start | `qfc logs daemon`, `qfc logs server`, then `qfc service restart` |
| a background task cannot read a project under `~/Desktop` or `~/Documents` on macos | give `~/.qwenfast-code/toolchain/node/bin/node` full disk access in system settings |
| `qfc run` waits forever in a script | it reads piped stdin as extra context; add `</dev/null` |
| the build stops with "missing: make c++" | install the c++ toolchain from step 1 and run the installer again |

## build details

- [toolchain.env](toolchain.env) pins bun, node, the opencode tag and the qfc version. [scripts/toolchain.sh](scripts/toolchain.sh) downloads exactly those builds for your os and cpu and checks them against the publishers' sha256 lists.
- [scripts/build.sh](scripts/build.sh) checks out the pinned opencode tag, applies [patches/apply.py](patches/apply.py) (asserted edits for the name, directories, wordmark and user facing text) and compiles one self contained binary.
- services run under launchd on macos, `systemd --user` on linux, and a small supervisor where neither exists. on macos the server starts from your terminal session so it can open projects under `~/Desktop` and `~/Documents`, which macos keeps from background jobs.
- `~/.config/qwenfast-code/opencode.json` is written once and then yours to change: agents, permissions, mcp servers, keybinds and themes, as in the [opencode docs](https://opencode.ai/docs/).

## test

```bash
npm --prefix agent test                                    # gateway, router, daemon (no gpu)
docker build -f code/test/linux.Dockerfile -t qfc-linux .  # a clean linux machine
docker run --rm -v <secrets>:/secrets:ro -v ~/.ssh:/home/ubuntu/.ssh:ro qfc-linux sh code/test/linux_e2e.sh
```

## uninstall

```bash
sh ~/.qwenfast-code/services.sh uninstall
rm -rf ~/.qwenfast-code ~/.local/bin/qfc ~/.local/bin/qfa ~/.config/qwenfast-code ~/.local/share/qwenfast-code
```

## license

mit. qwenfast code is built from [opencode](https://github.com/sst/opencode) (mit) and the qwenfast engine in this repository.
