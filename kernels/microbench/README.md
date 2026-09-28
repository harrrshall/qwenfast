# kernel microbenchmarks

standalone microbenchmarks for the kernels behind qwen3.8-27b's hybrid architecture: the gated-deltanet recurrence in the 48 linear-attention layers, the paged attention in the 16 full-attention layers, and every weight gemm shape in the model. each script times one kernel family in isolation with cuda events, writes json and a markdown table, and keeps going when an optional library is missing. the gemm rows in [`docs/benchmarks.md`](../../docs/benchmarks.md) come from this suite.

all shapes are taken from `engine/reference/config-Qwen3.8-27B-FP8.json` and cross-checked against `engine/reference/modeling_qwen3_5.py`; `common.py` holds the constants and the derivation of each one.

## files

| file | benchmarks |
|---|---|
| `common.py` | shared shapes, cuda-event timing, json and markdown output, the `safe_run` wrapper. not runnable on its own |
| `gdn_decode_bench.py` | one decode step of the gated-deltanet recurrence: `fla` `fused_recurrent_gated_delta_rule` with fp32, bf16 and fp16 state, a pure-torch reference at small batch for correctness, `causal_conv1d_update`, and a best-effort `cuLA` probe. batch sweep 1 to 512 |
| `gdn_prefill_bench.py` | chunked prefill with `fla` `chunk_gated_delta_rule` (torch fallback if `fla` is absent) over sequence lengths 512, 2048 and 8192 at batch 1. reports ms, tflop/s from an analytical flop count, and the hbm-bound floor |
| `attn_decode_bench.py` | flashinfer paged-kv batch decode for the attention layers (24 query heads, 4 kv heads, head dim 256): batch sweep x context 2048 and 8192 x kv dtype fp16 and fp8 |
| `gemm_bench.py` | every distinct weight gemm shape (gated-deltanet in and out projections, attention q/k/v/o, mlp gate/up/down, lm_head) x batch sweep x bf16, fp8 per-tensor, fp8 block-128 (vllm triton), fp8 cutlass (vllm), plus a whole-model per-decode-step total |
| `merge_results.py` | combines the four outputs into one `microbench-<ts>.json` and `.md`. pure file io |
| `run_all.sh` | runs the four scripts and the merge |
| `setup_remote.sh` | installs `flash-linear-attention`, `causal-conv1d` and (best effort) `cuLA` into a venv |

## running

quick smoke run of each script:

```sh
cd kernels/microbench
python gdn_decode_bench.py  --out /tmp/gdn_decode  --warmup 5 --iters 20 --batches 1,32,256
python gdn_prefill_bench.py --out /tmp/gdn_prefill --warmup 3 --iters 5  --seq-lens 512,2048
python attn_decode_bench.py --out /tmp/attn_decode --warmup 5 --iters 20 --batches 1,32,256
python gemm_bench.py        --out /tmp/gemm        --warmup 5 --iters 20 --batches 1,32,256
```

full sweep (20 warmup and 100 timed iterations for decode, attention and gemm; 10 and 30 for prefill):

```sh
MICROBENCH_OUT_DIR=/path/to/results VENV=/path/to/venv ./run_all.sh
```

`run_all.sh` reads `MICROBENCH_OUT_DIR`, `VENV`, `WARMUP`, `ITERS`, `PREFILL_WARMUP` and `PREFILL_ITERS` from the environment. `setup_remote.sh` reads `VENV`, `LOG_DIR`, `CULA_SRC_DIR` and `MAX_JOBS`. the defaults inside both scripts point at the gpu box layout used by [`scripts/`](../../scripts/), so set the variables explicitly elsewhere.

common flags on every script: `--out` (base path, writes `<out>.json` and `<out>.md`), `--warmup`, `--iters`, `--device` (default `cuda:0`), `--gpu-bw-gbps` (default 4800, the h200 peak used for efficiency percentages), `--batches` (comma-separated override of the batch sweep), `--seed`.

script-specific flags:

- `gdn_decode_bench.py`: `--torch-ref-max-batch`, `--verify-batch`, `--skip-cula`
- `gdn_prefill_bench.py`: `--seq-lens`, `--prefill-batch`, `--chunk-size` (default 64), `--skip-torch-fallback`
- `attn_decode_bench.py`: `--contexts`, `--page-size` (default 16), `--kv-dtypes` (default `float16,float8_e4m3fn`), `--use-tensor-cores` / `--no-use-tensor-cores`
- `gemm_bench.py`: `--include-alt-mlp` (also time the unfused gate and up projections), `--skip-cutlass`

## output

`run_all.sh` writes, under `MICROBENCH_OUT_DIR`:

```
microbench-<ts>-gdn_decode_bench.json   / .md
microbench-<ts>-gdn_prefill_bench.json  / .md
microbench-<ts>-attn_decode_bench.json  / .md
microbench-<ts>-gemm_bench.json         / .md
microbench-<ts>.json                    / .md   (combined, from merge_results.py)
```

each per-script json holds an `env` block (torch, cuda, gpu name, library versions) and one entry per variant and shape. a variant entry is either a timing record (`mean_us`, `min_us`, `max_us`, percentiles, `n_iters`, plus achieved gb/s or tflop/s and the efficiency against `--gpu-bw-gbps`) or, when the kernel or its library is unavailable, `{"status": "error", "error": "...", "traceback": "..."}`. the markdown file is the same data as tables.

## robustness contract

every variant runs inside `common.safe_run`, which catches all exceptions and stores the error in that variant's slot. a missing optional package (`fla`, `causal_conv1d`, `flashinfer`, `vllm`, `cula`) never aborts a sweep; the script still writes its json and exits 0. the only non-zero exit is cuda being unavailable.

`cuLA` and the vllm cutlass path get one more tier: they are probed against a short list of plausible entrypoints and report `"status": "unavailable"` with the attempts when none matched.

## reading the numbers

- decode is bandwidth-bound. `gdn_decode_bench.py` reports achieved gb/s against the state traffic `2 x 144 mib x batch x 48 layers`; `gemm_bench.py` reports effective tb/s of weight read per shape and the whole-model per-step total, to be compared with the 4.8 tb/s peak and the 6.2 ms weight-read floor.
- prefill is compute-bound. `gdn_prefill_bench.py` reports achieved tflop/s against the analytical flop count of the chunked delta rule (`common.gdn_chunk_algorithmic_flops`), and the hbm floor at each sequence length.
- the sum of the gated-deltanet, attention and gemm rows at one batch size is a lower bound on a real decode step: this suite times eager kernel calls with no cuda graphs, so small-batch rows include launch overhead that a graphed server hides, and no row includes scheduler or python time.

## caveats

- the prefill flop count is analytical and implementation-invariant for the same algorithm; a fused kernel that uses a different chunk size or skips the ut transform will read as a different tflop/s.
- `cuLA` exposes kda and lightning attention entrypoints whose gate parameterization differs from gated-deltanet; a successful probe is labelled as a caveat and should be numerically checked before its speed is compared.
- `gemm_bench.py`'s fp8 block-128 and cutlass variants call vllm internals (`vllm.model_executor.layers.quantization.utils.fp8_utils.w8a8_triton_block_scaled_mm`, `vllm._custom_ops`) that can change signature between vllm releases; the suite was written against vllm 0.28.0.
- `attn_decode_bench.py`'s fp8 kv path uses an implicit scale of 1, which measures the storage bandwidth and kernel path correctly and is not a numerics test.

## checking the suite without a gpu

```sh
python3 -m py_compile common.py gdn_decode_bench.py gdn_prefill_bench.py \
    attn_decode_bench.py gemm_bench.py merge_results.py
python3 merge_results.py --out /tmp/merge_test
```

`merge_results.py` with no inputs exercises the merge and report path and records every section as missing.
