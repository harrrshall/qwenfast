# qwenfast

a custom inference engine for `qwen3.8-27b`, written from scratch in pytorch, triton and flashinfer, and benchmarked head to head against vllm on a single h200.

qwen3.8-27b is a hybrid model: 48 gated deltanet layers, 16 gated gqa attention layers and a multi token prediction head, shipped as an fp8 checkpoint. qwenfast serves it through an openai compatible api with continuous batching, cuda graph decode, chunked prefill and speculative decoding.

## results

served throughput on one h200, 2000 token prompts, 500 output tokens, same checkpoint and prompts for both engines. vllm 0.28.0 runs its best configuration (fp8 weights with deepgemm). qwenfast runs `--preset fastest --spec-k 3 --spec-max-batch 16 --mixed-forward --mixed-graphs --overlap --overlap-min-fill 0.75 --prefill-chunk-tokens 8192 --detok-workers 0 --async-scheduling`.

| concurrency | vllm out tok/s | qwenfast out tok/s | ratio |
|---|---|---|---|
| 1 | 97 | 137 | 1.41x |
| 8 | 604 | 642 | 1.06x |
| 32 | 1509 | 1035 | 0.69x |
| 64 | 1914 | 1448 | 0.76x |
| 128 | 2201 | 1740 | 0.79x |
| 256 | 2251 | 1870 | 0.83x |

qwenfast leads at low concurrency, where speculative decoding amortizes the weight bandwidth floor, and has a lower median time to first token at 1, 8, 32 and 256 streams. vllm leads in aggregate throughput from 32 concurrent streams up, where the qwenfast serving path spends its time in host side scheduling of prefill. every kernel measured in isolation is faster in qwenfast. output quality matches on gsm8k and ifeval.

full methodology, latency tables and quality numbers: [docs/benchmarks.md](docs/benchmarks.md).

## what is inside

- custom triton gated deltanet kernels: pool indexed recurrent decode, fused causal conv and a fused verify and commit kernel for speculative decoding, at 86 to 87 percent of peak hbm bandwidth.
- fused fp8 weights with per shape gemm dispatch across marlin, deepgemm, cutlass and flashinfer backends, picked from a measured priority table.
- flashinfer paged attention with independent page pools for the kv cache and the ssm state.
- one cuda graph per batch size bucket that captures the entire decode step, sampler included.
- continuous batching scheduler with chunked prefill, a mixed prefill plus decode step and swap based preemption.
- mtp speculative decoding with the checkpoint's own draft head and a statistical acceptance gate.
- openai compatible server: streaming, thinking on and off, hermes tool calls, prometheus metrics, api keys with rate limits and usage metering.

architecture walkthrough: [docs/architecture.md](docs/architecture.md). api reference: [docs/api.md](docs/api.md).

## quick start

requirements: a cuda gpu with enough memory for the fp8 checkpoint (an h200 was used), python 3.10 or newer, and the packages in `engine/requirements.txt`.

```bash
pip install -r engine/requirements.txt
hf download Qwen/Qwen3.8-27B-FP8
```

run the server:

```bash
PYTHONPATH=engine python -m qwenfast.runtime.serve \
  --model /path/to/Qwen3.8-27B-FP8 --served-model-name qwen3.8-27b \
  --preset fastest --spec-k 3 --spec-max-batch 16 \
  --mixed-forward --mixed-graphs --overlap --overlap-min-fill 0.75 \
  --prefill-chunk-tokens 8192 --detok-workers 0 --async-scheduling \
  --host 0.0.0.0 --port 8000
```

query it with any openai client:

```python
from openai import OpenAI
client = OpenAI(api_key="none", base_url="http://localhost:8000/v1")
r = client.chat.completions.create(model="qwen3.8-27b",
                                   messages=[{"role": "user", "content": "hi"}],
                                   temperature=0)
print(r.choices[0].message.content)
```

speculative decoding engages on greedy requests, so send `temperature: 0` for the fastest single stream.

## benchmarks and tests

```bash
# serving load sweep against any openai compatible server
python benchmarks/bench_serve.py --base-url http://localhost:8000/v1 --model qwen3.8-27b \
  --concurrency 1,8,32,64,128,256 --input-len 2000 --output-len 500 --dataset random \
  --out results/my-run.json --tag my-run

# quality gate (gsm8k 200, ifeval 50)
python evals/run_eval.py --base-url http://localhost:8000/v1 --model qwen3.8-27b

# cpu tests, no gpu needed (gpu tests skip automatically)
pytest
```

## repository layout

| path | content |
|---|---|
| `engine/qwenfast/` | the engine: `kernels_gdn/`, `gemm/`, `attn/`, `runtime/`, `server/` |
| `engine/reference/` | upstream hugging face config and modeling files used for parity checks |
| `benchmarks/` | serving load harness and the result files behind the tables above |
| `evals/` | gsm8k and ifeval quality gate |
| `kernels/microbench/` | standalone kernel microbenchmarks |
| `demo/` | next.js streaming chat demo that proxies to the engine |
| `scripts/` | gpu box setup, server supervisor and operator tools |
| `docs/` | architecture, benchmarks and api reference |

## license

mit, see [license](LICENSE). the files under `engine/reference/` are from the qwen and hugging face transformers teams under the apache 2.0 license.
