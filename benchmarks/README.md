# benchmarks

serving load harness for openai-compatible inference servers. one script sweeps concurrency levels against `/v1/completions` or `/v1/chat/completions`, records per-request latency, and writes a json result that the other scripts merge and annotate. every qwenfast and vllm number in [`docs/benchmarks.md`](../docs/benchmarks.md) was produced with it.

## files

| file | purpose |
|---|---|
| `bench_serve.py` | async load generator. sweeps a list of concurrency levels, prints a markdown table, writes a result json |
| `summarize.py` | merges many result json files into one markdown comparison table |
| `cost.py` | per-million-token cost from a sustained throughput and an hourly gpu rate; can annotate a result json in place |
| `mock_server.py` | tiny aiohttp server that streams fake tokens at a fixed rate, for exercising `bench_serve.py` without a gpu |
| `results/` | curated result files, see below |

dependencies: `aiohttp`, `numpy`, and `transformers` if you pass `--tokenizer`.

## the standard sweep

```bash
cd benchmarks
python bench_serve.py \
  --base-url http://localhost:8000/v1 \
  --model qwen3.8-27b \
  --tokenizer /path/to/Qwen3.8-27B-FP8 \
  --concurrency 1,8,32,64,128,256 \
  --input-len 2000 --output-len 500 \
  --dataset random --nvidia-smi \
  --out results/bench-<tag>.json --tag <tag>
```

run it once per server configuration, each with its own `--tag`, then merge:

```bash
python summarize.py results/bench-*.json --out /tmp/results.md
```

for chat endpoints with a thinking mode you want disabled:

```bash
python bench_serve.py ... --endpoint chat \
  --extra-body '{"chat_template_kwargs":{"enable_thinking":false}}'
```

## flags

| flag | default | meaning |
|---|---|---|
| `--base-url` | required | server base url ending in `/v1` |
| `--model` | required | model name sent in the request body |
| `--api-key` | `EMPTY` | bearer token if the server checks one |
| `--concurrency` | `1,16,64,128,256,512` | comma-separated levels to sweep |
| `--num-prompts` | `max(concurrency * 8, 64)` | requests per level |
| `--input-len` | `2000` | target prompt length in tokens |
| `--output-len` | `500` | output tokens per request |
| `--dataset` | `random` | `random` or `sharegpt-like` |
| `--tokenizer` | none | hugging face tokenizer name or path, used to build exact-length prompts |
| `--endpoint` | `completions` | `completions` or `chat` |
| `--extra-body` | `{}` | json merged into every request body |
| `--stream` / `--no-stream` | stream | streaming responses |
| `--warmup` | `3` | discarded warmup requests per level |
| `--seed` | `0` | prompt rng seed |
| `--nvidia-smi` | off | sample gpu utilization, memory and power once per second during each level |
| `--no-metrics` | off | skip scraping `/metrics` before and after each level |
| `--request-timeout` | `600` | per-request timeout in seconds |
| `--ready-timeout` | `0` | wait up to this many seconds for `/v1/models` and a healthy `/health` before starting |
| `--no-health-check` | off | do not poll `/health` between levels |
| `--continue-on-error` | off | keep sweeping after a level in which every request failed |
| `--out` | required | result json path |
| `--tag` | `run` | label stored in the result and used by `summarize.py` |

## methodology

- closed loop. each level runs `--num-prompts` requests through an `asyncio.Semaphore(concurrency)`, so exactly `concurrency` requests are in flight at all times. this finds the server's saturation point at each level.
- warmup. `--warmup` requests are sent at the same concurrency before the timed run and discarded, absorbing graph capture, connection setup and jit costs.
- fixed prompts. the `random` dataset draws token ids from a seeded rng and decodes them with the tokenizer, so prompts are identical across runs and close to `--input-len` tokens. without a tokenizer it falls back to random ascii words. `sharegpt-like` cycles through a small set of realistic prompts padded to `--input-len`.
- fixed output length. every request sets `ignore_eos: true` and `temperature: 0`, so each request produces exactly `--output-len` tokens and throughput is comparable across configurations.
- exact token counts. streaming requests set `stream_options.include_usage`; servers that honour it report exact prompt and completion token counts, and the harness falls back to counting content chunks otherwise.
- liveness. the sweep checks `/health` before it starts and between levels, and aborts with exit code 2 when the server reports a dead engine or when every warmup request at a level fails.

## metrics

- `ttft_s`: request start to first non-empty streamed token.
- `tpot_s`: per request, `(e2e - ttft) / (completion_tokens - 1)`; mean, p50 and p99 across requests.
- `itl_s`: gap between consecutive streamed tokens, pooled across all requests in a level.
- `e2e_s`: request start to stream end.
- `output_tok_per_s`: completion tokens over the level's wall time.
- `total_tok_per_s`: prompt plus completion tokens over the level's wall time.
- `request_throughput_per_s`: completed requests over the level's wall time.
- `gpu`: mean and max of `utilization_gpu_pct`, `memory_used_mib` and `power_draw_w` when `--nvidia-smi` is set.
- `metrics_before` / `metrics_after`: prometheus values scraped from `/metrics` (`gpu_cache_usage_perc`, `num_requests_waiting`, `num_requests_running`, `num_requests_swapped`) when the server exposes them.

## output format

```json
{
  "metadata": {"timestamp_utc": "...", "tag": "...", "args": {...}, "aborted": null},
  "levels": [
    {
      "concurrency": 64, "num_prompts": 512, "num_success": 512, "num_errors": 0,
      "started_unix": 0.0, "ended_unix": 0.0, "duration_s": 12.3,
      "output_tok_per_s": 4100.2, "total_tok_per_s": 20700.5, "request_throughput_per_s": 41.6,
      "ttft_s": {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0},
      "tpot_s": {"mean": 0.0, "p50": 0.0, "p99": 0.0},
      "itl_s": {"mean": 0.0, "p50": 0.0, "p99": 0.0},
      "e2e_s": {"mean": 0.0, "p50": 0.0, "p99": 0.0},
      "prompt_tokens_total": 0, "completion_tokens_total": 0,
      "metrics_before": {}, "metrics_after": {}, "gpu": {}
    }
  ]
}
```

`started_unix` and `ended_unix` bound the measured window of each level, so an engine-side per-second trace can be sliced to exactly that level.

## cost

`cost.py` turns a throughput into a per-million-token cost from an hourly gpu rate. `--gpu` picks a named rate from the table in the script, `--rate-inr` supplies your own hourly rate.

```bash
python cost.py --tok-per-s 4200 --gpu h200_on_demand
python cost.py --tok-per-s 4200 --rate-inr 250
python cost.py --annotate results/bench-<tag>.json --gpu h200_on_demand   # adds level["cost"]
```

`summarize.py` accepts the same `--gpu` and `--rate-inr` flags and adds a cost column when either is given.

## self-test without a gpu

```bash
python mock_server.py --port 9999 &
python bench_serve.py --base-url http://localhost:9999/v1 --model mock \
  --concurrency 1,4 --num-prompts 8 --input-len 32 --output-len 16 \
  --out /tmp/mock_result.json --tag mock-selftest
```

`mock_server.py` accepts `--host`, `--port`, `--ttft-ms` and `--tok-interval-ms`. the self-test exercises both endpoints, streaming and non-streaming responses, `/metrics` scraping and the stats pipeline in under a second.

## results

`results/` holds the runs cited in [`docs/benchmarks.md`](../docs/benchmarks.md):

| file | run |
|---|---|
| `bench-qwenfast-final.json` | qwenfast, final configuration, the headline sweep |
| `bench-t1-deepgemm.json` | vllm 0.28.0, fp8 + deepgemm, its best configuration |
| `bench-t1-deepgemm-noprefix.json` | same with prefix caching off |
| `bench-t1-bf16-weights.json` | vllm, bf16 weights |
| `bench-t1-bt2k.json` | vllm, 2048-token prefill chunks |
| `bench-t1-noprefixcache.json` | vllm, prefix caching off, no deepgemm |
| `bench-vllm028-fp8-baseline.json` | vllm, fp8 default environment |
| `eval-qwenfast-final.json`, `eval-vllm-deepgemm.json` | quality gate runs from [`evals/`](../evals/README.md) |
