# qwenfast code

a fast inference engine for `qwen3.8-27b` and the coding agent that runs on it.

- **engine**: a from scratch server for the hybrid qwen3.8-27b in pytorch, triton and flashinfer, with an openai compatible api, speculative decoding and a prefix cache built for agent conversations
- **qwenfast code** (`qfc`): the opencode terminal interface running on your mac or linux machine against your own qwen models, routing every conversation to the model that fits it
- **agent daemon**: a local gateway and background task runner that keeps sessions alive for hours, wakes the gpu when you start working and pauses it when you stop

```
your machine                                                     gpu server
┌──────────────────────────────────────────────────────────┐     ┌────────────────────────────┐
│ qfc tui ──┐                                              │     │ qwenfast   qwen3.8-27b     │
│ qfc tui ──┼─▶ qfc server :4097 ─▶ gateway :7788 ─── ssh ─┼────▶│ vllm       qwen3.6-35b-a3b │
│ qfc run ──┘   sessions, tools     routing, tunnel,       │     │ watchdogs and keeper       │
│               and agents          gpu wake and pause     │     └────────────────────────────┘
└──────────────────────────────────────────────────────────┘
```

## quick start

new to qwenfast code? the [getting started guide](code/README.md#getting-started) walks through prerequisites, a model backend, install and a first task in four steps.

**use qwenfast code.** on macos or linux, with a c++ toolchain installed (`xcode-select --install` on macos, `build-essential` on debian and ubuntu):

```bash
git clone https://github.com/harrrshall/qwenfast && cd qwenfast
sh scripts/jarvis_setup.sh        # optional: an h200 box on jarvislabs with both models
sh code/install.sh                # or pass QFC_API_KEY, QFC_BIG_URL and QFC_SMALL_URL for your own servers
qfc status                        # models, routing and server health
qfc                               # open the tui in the current project
```

the installer downloads a pinned, checksum verified toolchain, builds `qfc` from a pinned opencode release and starts two background services. any openai compatible servers work as the backend; the next step serves the intended ones on your own gpu.

**serve the models.** on a linux machine with an h200 (or any gpu with about 140 gb of memory for both models), cuda 13 and python 3.10 or newer:

```bash
python -m venv .venv && . .venv/bin/activate && pip install -r engine/requirements.txt
hf download Qwen/Qwen3.8-27B-FP8 && hf download Qwen/Qwen3.6-35B-A3B-FP8
mkdir -p ~/.qwenfast/secrets && python -c "import secrets; print(secrets.token_urlsafe(32))" > ~/.qwenfast/secrets/agent_key
sh scripts/remote_agent_stack.sh      # qwenfast on :8000, vllm on :8001, both supervised
```

the key in `~/.qwenfast/secrets/agent_key` is the `QFC_API_KEY` for the install above. on jarvislabs, `scripts/jarvis_setup.sh` does all of this for you and writes `.secrets/agent_key` and `.secrets/agent_box_id`, so the gateway manages the box itself: it resumes a paused box when you start working, reaches it through an ssh tunnel and pauses it after an idle hour.

**serve only the engine.**

```bash
PYTHONPATH=engine python -m qwenfast.runtime.serve \
  --model /path/to/Qwen3.8-27B-FP8 --served-model-name qwen3.8-27b \
  --preset fastest --spec-k 3 --spec-sampling --prefix-cache-entries 16 \
  --mixed-forward --mixed-graphs --overlap --async-scheduling \
  --max-num-seqs 24 --max-model-len 131072 --n-kv-pages 20480 --host 0.0.0.0 --port 8000
```

## routing

qwenfast code defaults to the `auto` model. the gateway routes each conversation to one of three tiers:

| tier | model | thinking | used for |
|---|---|---|---|
| small | qwen3.6-35b-a3b, 3b active | off | lookups, small edits, summaries, session titles |
| medium | qwen3.8-27b | medium effort | features, fixes and tests |
| large | qwen3.8-27b | xhigh effort | debugging, design, refactors, performance |

the split follows the published scores: qwen3.8-27b reaches 61.7 on swe-bench pro and 73.0 on terminal-bench 2.1, and qwen3.6-35b-a3b reaches 49.5 and 51.5 (2.0) while activating a ninth of the weights per token. a lexical score of the typed message decides clear cases and the small model grades borderline ones. a conversation keeps the highest tier it has needed, that memory survives restarts, and a turn that keeps looping on the small tier moves up to medium.

## performance

engine throughput on one h200 against vllm 0.28.0 in its best configuration, 2000 token prompts and 500 output tokens:

| concurrency | vllm out tok/s | qwenfast out tok/s | ratio |
|---|---|---|---|
| 1 | 97 | 137 | 1.41x |
| 8 | 604 | 642 | 1.06x |
| 32 | 1509 | 1035 | 0.69x |
| 256 | 2251 | 1870 | 0.83x |

qwenfast leads at low concurrency, which is where an interactive agent spends its time. vllm leads in aggregate throughput from 32 streams up. quality matches on gsm8k and ifeval. methodology and latency tables are in [docs/benchmarks.md](docs/benchmarks.md).

agent specific gains:

| | before | after |
|---|---|---|
| time to first token, turn at a 75k token context | 9.7 s | 0.6 s |
| decode, sampled thinking, one stream | 65 tok/s | 123 tok/s |

the first comes from the prefix cache. 48 of the 64 layers are gated deltanet, whose recurrent state cannot be sliced at an arbitrary token, so each prompt snapshots that state where it ends and the next turn of the conversation resumes from it. the second comes from speculative sampling: the mtp draft is exact rejection sampled against the target distribution, so sampled output keeps its distribution.

## how the engine works

- triton gated deltanet kernels for recurrent decode, fused causal conv and a fused verify and commit step for speculative decoding, at 86 to 87 percent of peak hbm bandwidth
- fp8 weights with per shape gemm dispatch across marlin, deepgemm, cutlass and flashinfer, chosen from a measured table
- flashinfer paged attention with separate page pools for kv and recurrent state, one cuda graph per batch bucket covering the whole decode step
- continuous batching with chunked prefill, a mixed prefill and decode step and swap based preemption
- an openai compatible server with streaming, thinking modes, qwen3 coder xml and hermes tool calls, prometheus metrics and keyed rate limits

the walkthrough is in [docs/architecture.md](docs/architecture.md) and the api in [docs/api.md](docs/api.md).

## repository

| path | content |
|---|---|
| [engine/](engine/qwenfast) | the inference engine: kernels, gemm dispatch, attention, runtime, server |
| [code/](code/README.md) | qwenfast code: patches, pinned toolchain, build, installer, services |
| [agent/](agent/README.md) | gateway, router, task runner and gpu box keeper, plus a terminal-bench adapter |
| [scripts/](scripts/README.md) | gpu server bring up, supervisors and operator tools |
| [benchmarks/](benchmarks) and [evals/](evals) | serving load harness, results, gsm8k and ifeval gate |
| [docs/](docs) | architecture, api and benchmarks |
| [demo/](demo) | next.js streaming chat demo |

## tests

```bash
pytest                          # engine and server, cpu only; gpu tests skip themselves
npm --prefix agent ci && npm --prefix agent test      # gateway, router, daemon against a mock server
docker build -f code/test/linux.Dockerfile -t qfc-linux .    # clean linux install of qwenfast code
```

## license

mit, see [license](LICENSE). qwenfast code is built from [opencode](https://github.com/sst/opencode) (mit). files under `engine/reference/` come from the qwen and hugging face transformers teams under apache 2.0.
