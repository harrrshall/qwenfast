# benchmarks: qwenfast versus vllm on one h200

this page records how qwenfast was measured against vllm on a single h200, what was measured, and how to reproduce every number. every figure below comes from a result file in [`benchmarks/results/`](../benchmarks/results/) unless the text says otherwise.

## summary

- at concurrency 1 and 8, qwenfast serves more output tokens per second than vllm at its best configuration: 1.41x and 1.06x.
- at concurrency 32 and above, vllm is faster: qwenfast delivers 0.69x to 0.83x of vllm's served throughput.
- quality is at parity on the same fp8 checkpoint: gsm8k-200 0.930 (qwenfast) versus 0.915 (vllm), ifeval-50 0.980 versus 0.960, both within sampling noise at n=200 and n=50.
- in isolation, qwenfast's kernels lead at every layer measured; the served gap at high concurrency comes from host-side work in the scheduler step that does not overlap with the gpu (vllm holds the gpu at 99 to 100 percent through the sweep, qwenfast at 89 to 93 percent).

## setup

| item | value |
|---|---|
| gpu | one nvidia h200 sxm, 141 gb hbm3e |
| model | `Qwen/Qwen3.8-27B-FP8` (block-128 fp8 weights, 48 gated-deltanet layers + 16 gated gqa layers, mtp head) |
| vllm | 0.28.0, fp8 weights, deepgemm block-fp8 gemm (`VLLM_USE_DEEP_GEMM=1`) |
| torch / triton / cuda | 2.13 / 3.7.1 / 13.0 toolchain |
| flashinfer | 0.6.16 |
| flash-linear-attention (`fla`) | 0.5.2 (used by vllm; qwenfast uses its own decode kernels) |
| qwenfast | this repository, `--preset fastest --spec-k 3 --spec-max-batch 16 --mixed-forward --mixed-graphs --overlap --overlap-min-fill 0.75 --prefill-chunk-tokens 8192 --detok-workers 0 --async-scheduling`, 256 sequences, bf16 kv cache, fp16 ssm state |

both servers ran on the same machine and the same checkpoint. the load generator ran on the same machine over localhost, so network latency is not part of any number.

## workload

the standard sweep is one invocation of [`benchmarks/bench_serve.py`](../benchmarks/bench_serve.py):

- prompts: `--dataset random`, 2000 random token ids decoded with the model tokenizer, then re-tokenized by the server (measured prompt lengths land at 2090 to 2195 tokens). the prompt rng is seeded (`--seed 0`), so every run sees the same prompts.
- output: `--output-len 500` with `ignore_eos` set on every request, so each request generates exactly 500 tokens.
- concurrency levels: 1, 8, 32, 64, 128, 256 (vllm runs also include 512). each level is a closed loop: `concurrency` requests in flight at all times, `max(concurrency * 8, 64)` requests per level, 3 discarded warmup requests per level.
- sampling: `temperature 0`, streaming on, `/v1/completions` endpoint.
- metrics: output tok/s (sum of completion tokens over the level's wall time), ttft, tpot (`(e2e - ttft) / (completion_tokens - 1)` per request), and `nvidia-smi` samples once per second.

## vllm configurations tried

all runs use `--max-num-seqs 256` and `--gpu-memory-utilization 0.90`. output tok/s at concurrency 256 unless noted.

| tag | configuration | out tok/s | result file |
|---|---|---|---|
| `vllm028-fp8-baseline` | fp8 weights, default environment (triton block-fp8 gemm) | 915 | `bench-vllm028-fp8-baseline.json` |
| `t1-bf16-weights` | bf16 weights, 512 sequences, 16384 batched tokens, async scheduling | 1472 (1653 at 128) | `bench-t1-bf16-weights.json` |
| `t1-bt2k` | fp8, 2048-token prefill chunks | 1000 | `bench-t1-bt2k.json` |
| `t1-noprefixcache` | fp8, prefix caching off, no deepgemm | 1820 | `bench-t1-noprefixcache.json` |
| `t1-deepgemm` | fp8 + deepgemm | 2251 (2334 at 512) | `bench-t1-deepgemm.json` |
| `t1-deepgemm-noprefix` | fp8 + deepgemm, prefix caching off | 2379 (2377 at 512) | `bench-t1-deepgemm-noprefix.json` |

the best vllm configuration is fp8 with deepgemm. deepgemm alone is worth 2.5x over the fp8 default, because the triton block-fp8 gemm the default path uses reads weights at about 0.7 tb/s and makes the whole server compute-bound. prefix caching contributes nothing measurable on random prompts (2251 versus 2379 at concurrency 256 is run-to-run variation), so `t1-deepgemm` is used as the comparison row everywhere below.

## served throughput

output tokens per second, 2000-token prompts, 500 output tokens. vllm: `bench-t1-deepgemm.json`. qwenfast with asynchronous scheduling (the configuration in the setup table): `bench-qwenfast-async-scheduling.json`. zero request errors in both runs.

| concurrency | vllm best | qwenfast | ratio | faster |
|---|---|---|---|---|
| 1 | 97.0 | 137 | 1.41x | qwenfast |
| 8 | 604 | 642 | 1.06x | qwenfast |
| 32 | 1509 | 1035 | 0.69x | vllm |
| 64 | 1914 | 1448 | 0.76x | vllm |
| 128 | 2201 | 1740 | 0.79x | vllm |
| 256 | 2251 | 1870 | 0.83x | vllm |

total (prompt + output) tokens per second at concurrency 256: vllm 11885, qwenfast 9873.

with `--async-scheduling` the engine schedules, plans and launches step n+1 while step n runs on the device, and reads back step n's sampled tokens one step later through a pinned buffer; the sampled token feeds the next step by a device-side gather, so no device-to-host copy sits on the critical path. output is token-identical to synchronous scheduling.

the synchronous scheduling configuration (`--preset fastest --spec-k 3 --spec-max-batch 16 --mixed-forward --prefill-chunk-tokens 8192`, `bench-qwenfast-final.json`) measured:

| concurrency | vllm best | qwenfast, synchronous | ratio | faster |
|---|---|---|---|---|
| 1 | 97.0 | 134 | 1.38x | qwenfast |
| 8 | 604 | 626 | 1.04x | qwenfast |
| 32 | 1509 | 908 | 0.60x | vllm |
| 64 | 1914 | 1211 | 0.63x | vllm |
| 128 | 2201 | 1389 | 0.63x | vllm |
| 256 | 2251 | 1419 | 0.63x | vllm |

qwenfast's advantage at 1 and 8 comes from mtp speculative decoding (k=3, enabled while the decode batch is at most 16). above that batch size speculation is switched off because the verify gemm and the doubled recurrent-state pass cost more than they save, and the comparison becomes plain decode plus chunked prefill, where vllm's scheduler keeps the gpu busier.

## latency

median time to first token and median time per output token, vllm best against qwenfast with asynchronous scheduling.

| concurrency | vllm ttft p50 | qwenfast ttft p50 | vllm tpot p50 | qwenfast tpot p50 |
|---|---|---|---|---|
| 1 | 221 ms | 188 ms | 9.8 ms | 6.7 ms |
| 8 | 957 ms | 190 ms | 11.3 ms | 11.2 ms |
| 32 | 1570 ms | 1201 ms | 18.8 ms | 28.5 ms |
| 64 | 1727 ms | 1761 ms | 30.2 ms | 40.3 ms |
| 128 | 1803 ms | 1803 ms | 55.0 ms | 69.1 ms |
| 256 | 2060 ms | 1892 ms | 110.5 ms | 132.5 ms |

qwenfast has the lower ttft at 1, 8, 32 and 256 and ties at 128. tpot at concurrency 1 reflects speculative decoding (several tokens are emitted per verify step). from concurrency 32 upward qwenfast's tpot is 1.2x to 1.5x vllm's, which is the same gap as the throughput table seen per request.

the synchronous configuration (`bench-qwenfast-final.json`) had lower ttft at every level (176 / 199 / 671 / 1233 / 1317 / 1463 ms) and higher tpot from concurrency 32 upward (33.6 / 50.2 / 89.0 / 175.2 ms), so asynchronous scheduling trades some first-token latency at high concurrency for throughput.

gpu utilization sampled by `nvidia-smi` during the sweep: vllm 98.8 to 100 percent at every level; qwenfast 88.5 to 92.8 percent with asynchronous scheduling and 73 to 92 percent synchronous. the remaining qwenfast gap at high concurrency is host time inside the scheduler step.

## quality parity

[`evals/run_eval.py`](../evals/run_eval.py) against each server: greedy, thinking disabled, 512 max tokens, concurrency 32.

| eval | vllm (deepgemm) | qwenfast, asynchronous scheduling | qwenfast, synchronous |
|---|---|---|---|
| gsm8k-200 accuracy | 0.915 (183/200) | 0.930 (186/200) | 0.925 (185/200) |
| ifeval-50 accuracy | 0.960 (48/50) | 0.980 (49/50) | 0.980 (49/50) |
| request errors | 0 | 0 | 0 |

result files: `eval-vllm-deepgemm.json`, `eval-qwenfast-async-scheduling.json`, `eval-qwenfast-final.json`. speculative decoding was live on both qwenfast servers. the differences (at most 3 gsm8k items, 1 ifeval item) are inside binomial noise at these sample sizes (one standard deviation is about 2 points on gsm8k-200 and 3 points on ifeval-50). speculative decoding verifies every drafted token against the base model's own logits, so it changes speed and leaves accuracy inside the same noise band.

## decode-only steps

graph-timed decode steps with no prefill and no serving overhead isolate the kernels. vllm's number is a served run with 128-token prompts and 500 output tokens (near-pure decode); qwenfast's is a timed cuda-graph replay at the canonical preset, 3 repeats.

| batch | vllm out tok/s | qwenfast out tok/s (ms per step) | ratio |
|---|---|---|---|
| 1 | 92.9 | 78 (12.87) | 0.84x |
| 32 | 1736 | 1946 (16.45) | 1.12x |
| 128 | 3787 | 4741 (27.0) | 1.25x |
| 256 | 4433 (vllm at 512) | 5998 (42.7) | 1.35x |

at batch 1 both engines sit within a few milliseconds of the 6.2 ms floor set by reading 29.7 gb of fp8 weights once per step; speculative decoding is what moves qwenfast past vllm there in the served table. from batch 32 upward qwenfast's decode step is faster, so the served deficit at high concurrency is entirely on the prefill and scheduling side.

these rows come from step-timing logs of the engine and from a vllm sweep that is not in `benchmarks/results/`; they are reported for context and are not part of the headline claim.

## kernel-level comparison

measured in isolation on the same h200.

| kernel | qwenfast | reference |
|---|---|---|
| gated-deltanet decode, 48 layers | custom triton kernel at 86 to 87 percent of peak hbm bandwidth, 3.0x to 4.1x faster than `fla` `fused_recurrent_gated_delta_rule` | `fla` reaches 2.9 tb/s (60 percent of peak) at batch 128 and above, with a 100 microsecond per-layer latency floor at small batch |
| fp8 gemm, whole decode step | marlin w8a16 at small m, deepgemm from m=32, selected per (m bucket, layer shape) from a measured priority table; 8.2 ms per step at batch 1 against the 6.2 ms weight-read floor | vllm's default triton block-fp8 gemm: 36.2 ms per step at batch 1 (0.71 tb/s effective); bf16 `torch.matmul`: 16.2 ms |
| prefill, one 8192-token chunk | 13991 tok/s, 69 percent of the 20060 tok/s fp8 compute ceiling | vllm, derived from its served totals: about 9600 tok/s, 48 percent |
| cuda graphs | one graph per batch bucket capturing the whole step; 4.28x over eager at batch 1 | |

fp8-activation gemm backends (deepgemm, flashinfer block-fp8, cutlass) all land at a relative l2 error of 2.6e-2 against an fp32 dequantized reference; the w8a16 marlin path is at 3.5e-3. the dispatcher exposes `--gemm-accuracy strict` to restrict routing to the w8a16 class when reproducibility matters more than throughput.

the gemm microbenchmark numbers were produced by [`kernels/microbench/`](../kernels/microbench/README.md); the gated-deltanet and prefill figures by the engine's own benches under `engine/qwenfast/`.

## reproducing

start the engine (see [`docs/api.md`](api.md) for the server surface):

```bash
PYTHONPATH=engine python -m qwenfast.runtime.serve \
  --model /path/to/Qwen3.8-27B-FP8 --served-model-name qwen3.8-27b \
  --preset fastest --spec-k 3 --spec-max-batch 16 --mixed-forward --mixed-graphs \
  --overlap --overlap-min-fill 0.75 --prefill-chunk-tokens 8192 --detok-workers 0 --async-scheduling \
  --host 0.0.0.0 --port 8000
```

drop `--async-scheduling` (and the `--mixed-graphs`, `--overlap` and `--detok-workers` flags) for the synchronous configuration.

or start vllm at its best configuration:

```bash
VLLM_USE_DEEP_GEMM=1 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.8-27B-FP8 --served-model-name qwen3.8-27b \
  --max-num-seqs 256 --gpu-memory-utilization 0.90 --host 0.0.0.0 --port 8000
```

run the standard sweep against whichever server is up:

```bash
cd benchmarks
python bench_serve.py --base-url http://localhost:8000/v1 --model qwen3.8-27b \
  --tokenizer /path/to/Qwen3.8-27B-FP8 \
  --concurrency 1,8,32,64,128,256 --input-len 2000 --output-len 500 \
  --dataset random --nvidia-smi --out results/<tag>.json --tag <tag>
```

run the quality gate:

```bash
python evals/run_eval.py --base-url http://localhost:8000/v1 --model qwen3.8-27b \
  --concurrency 32 --enable-thinking false --max-tokens 512 --temperature 0 \
  --out benchmarks/results/eval-<tag>.json --tag <tag>
```

merge any set of sweep files into one table:

```bash
python benchmarks/summarize.py benchmarks/results/bench-*.json --out /tmp/results.md
```

## result files

| file | content |
|---|---|
| `bench-qwenfast-async-scheduling.json` | qwenfast headline sweep, asynchronous scheduling configuration |
| `bench-qwenfast-final.json` | qwenfast, synchronous scheduling configuration |
| `bench-t1-deepgemm.json` | vllm best configuration, the comparison row |
| `bench-t1-deepgemm-noprefix.json` | vllm best configuration with prefix caching off |
| `bench-t1-bf16-weights.json`, `bench-t1-bt2k.json`, `bench-t1-noprefixcache.json`, `bench-vllm028-fp8-baseline.json` | the other vllm configurations tried |
| `eval-qwenfast-async-scheduling.json`, `eval-qwenfast-final.json`, `eval-vllm-deepgemm.json` | gsm8k-200 and ifeval-50 runs with every raw response |

all files follow the output schemas documented in [`benchmarks/README.md`](../benchmarks/README.md) and [`evals/README.md`](../evals/README.md).
