# scripts

helpers for bringing up a gpu box, running the server under a supervisor, driving benchmark sweeps and operating a keyed public endpoint. the `remote_*.sh` scripts expect the box layout they create: weights under `/home/hf`, a virtualenv at `/home/venv_vllm`, the engine mirrored to `/home/engine` and results under `/home/qwenfast-results`. adjust the paths at the top of each script for a different layout.

## box setup

| script | what it does |
|---|---|
| `remote_bootstrap_box.sh` | idempotent bring up of a fresh cuda box: results directory, pinned virtualenv (torch, vllm, flashinfer, triton, fla), cuda 13 toolchain inside the venv, weights download if the download script is present |
| `remote_download_weights.sh` | downloads checkpoints into the hugging face cache, `MODELS` overrides the default fp8 and bf16 `qwen3.8-27b` pair |
| `remote_install_vllm.sh` | creates the virtualenv with vllm alone, for baseline runs |
| `remote_gpu_free.sh` | stops any running server or benchmark and waits until gpu memory is released |

## serving

| script | what it does |
|---|---|
| `remote_demo_server.sh` | starts the engine in the foreground with the fastest serving configuration |
| `remote_public_server.sh` | supervised public endpoint: reads api keys from `/home/.secrets`, polls `/health`, restarts the server with backoff, drains in flight streams on stop. configuration through env vars (`MAX_NUM_SEQS`, `MAX_MODEL_LEN`, `MAX_STREAMS_PER_KEY`, `MAX_STREAMS_PER_IP`, `REQUEST_TIMEOUT`, `DRAIN_TIMEOUT`, `LOG_CONTENT`) |
| `supervisor_lock.sh` | single holder lock used by the supervisor so two copies never fight for the same port. usable as a library or as a command |
| `smoke_test.sh` | correctness and single stream speed check against a running server (`BASE` and `MODEL` env vars) |

## agent backend

the two model backend for the long running agent in [agent/](../agent/README.md): qwenfast serving qwen3.8-27b on port 8000 and vllm serving qwen3.6-35b-a3b on port 8001, on one h200.

| script | what it does |
|---|---|
| `remote_agent_bringup.sh` | fresh or resumed box to serving: downloads only the two fp8 checkpoints, builds the venv, then becomes the stack keeper. idempotent |
| `remote_agent_stack.sh` | the keeper: starts the big tier first so its pools are allocated, then the small tier, and restarts a supervisor that dies. runs on a jarvislabs box as is, and on any other gpu machine from this checkout (engine, active venv, hugging face cache, key in `~/.qwenfast/secrets/agent_key`). sizes come from env vars (`BIG_SEQS`, `BIG_CTX`, `BIG_KV_PAGES`, `BIG_PREFIX_ENTRIES`, `SMALL_GPU_UTIL`) |
| `remote_supervise.sh` | per server watchdog: health polls, drain, restart with backoff, log rotation, heartbeat json. kills only its own process group, so two servers can share the gpu |
| `remote_agent_redeploy_big.sh` | swaps in a freshly uploaded engine and restarts only the big tier |

## benchmarks

| script | what it does |
|---|---|
| `remote_vllm_baseline.sh` | vllm server on the fp8 checkpoint with the baseline settings |
| `remote_run_config.sh` | one vllm configuration end to end: restart the server with extra flags, wait for readiness, run the standard sweep |
| `remote_bench_sweep.sh` | the standard sweep with `benchmarks/bench_serve.py` against the server on port 8000 |
| `remote_vllm_eval.sh` | gsm8k and ifeval on vllm's best configuration |
| `step_trace_report.py` | slices an engine step trace (`--step-trace-out`) by concurrency level and joins it with a sweep result |
| `step_profile_report.py` | prints and diffs the per step host phase attribution written by `--step-profile-out` |

## operating a keyed endpoint

| script | what it does |
|---|---|
| `make_api_keys.py` | generates, lists, rotates and revokes api keys in `.secrets/api_keys.json` plus the admin and demo keys. never prints key material unless asked for one key with `--show` |
| `usage_dashboard.py` | local only usage dashboard over `/admin/usage` and `/metrics`, stdlib only |
| `export_queries.py` | pulls the optional content log off the box for abuse review, into the gitignored `exports/` directory |

`.secrets/` is gitignored. keep keys and cloud credentials there and nowhere else.
