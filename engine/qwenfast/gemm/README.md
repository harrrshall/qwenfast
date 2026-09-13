# qwenfast.gemm

fused fp8 weights, the per-m-bucket gemm dispatch table, and the autotune cache for `qwen3.8-27b`. the package reads checkpoints through `qwenfast.weights` and never dequantizes a weight on the hot path.

the package imports with no torch or cuda present; every backend that touches an optional dependency (`vllm`, deepgemm, flashinfer) is guarded, and `linear()` degrades to a working fallback when one is missing.

## files

| file | owns |
|---|---|
| `fused_weights.py` | `FP8Tensor`, the per-layer weight containers, `build_fused_weights`, `save_fused` / `load_fused`, `quantize_bf16_to_fp8_block128` |
| `dispatch.py` | `linear`, `resolve_backend`, the registered backends, priority profiles, `M_BUCKETS` / `m_bucket`, accuracy modes, weight-cache policy |
| `autotune.py` | times every backend per `(N, K, m bucket)` on the gpu and persists the winner to `autotune_cache/sm<XX>.json` |
| `bench_gemm.py` | graph-timed sweep over every model shape x m bucket x backend; emits priority tables |
| `gemm_numerics.py` | per-backend accuracy against an fp32 dequant reference; argmax-flip proxy through the real `lm_head` |
| `mixed_gemm_probe.py` | large-m probes (deepgemm configs, split/pad/quant experiments) for the mixed step |
| `tests/test_gemm.py` | cpu tests for fusion, dispatch and save/load; gpu tests for every backend against a bf16 reference |

## fused weight layout

the checkpoint's block-128 scale grid concatenates cleanly along the output dimension whenever the fused output dimension is a multiple of 128:

| fused tensor | composition | shape | scale |
|---|---|---|---|
| `in_proj_qkvz` (gdn) | `in_proj_qkv` (10240) + `in_proj_z` (6144) | `[16384, 5120]` fp8 | `[128, 40]` |
| `in_proj_ba` (gdn) | `in_proj_b` (48) + `in_proj_a` (48) | `[96, 5120]` bf16 | none |
| `qkv_proj` (attention) | `q_proj` (12288) + `k_proj` (1024) + `v_proj` (1024) | `[14336, 5120]` fp8 | `[112, 40]` |
| `gate_up_proj` (mlp) | `gate_proj` (17408) + `up_proj` (17408) | `[34816, 5120]` fp8 | `[272, 40]` |

`out_proj`, `o_proj` and `down_proj` pass through unfused as `FP8Tensor`s. the mtp head reuses the attention and mlp helpers. `embed_tokens` and `lm_head` stay bf16; `quantize_lm_head=True` adds an fp8 `lm_head_fp8` for experiments.

`FP8Tensor` holds `weight` (`[N, K]` `float8_e4m3fn`) and `scale_inv` (`[N/128, K/128]`), dequantizing as `weight[n, k] * scale_inv[n // 128, k // 128]`. backend-specific repacked copies are cached on the tensor lazily (`_marlin_cache`, `_deepgemm_cache`, `_pertensor_cache`, `_machete_cache`) and never persisted.

```python
from qwenfast.gemm import build_fused_weights, save_fused, load_fused

fw = build_fused_weights("/path/to/Qwen3.8-27B-FP8", device="cuda:0")
print(fw.nbytes())
save_fused(fw, "/path/to/fused-cache")          # one safetensors file plus json metadata
fw = load_fused("/path/to/fused-cache", device="cuda:0")
```

`FusedModelWeights` carries `embed_tokens`, `lm_head`, `final_norm`, per-layer `layernorms`, `gdn`, `attn`, `mlp` dicts keyed by layer index, and an optional `mtp`.

## dispatch

`linear(x, w, *, backend=None, sm_version=None, use_autotune=True)` takes `x` of shape `[..., K]` and a fused weight, and returns `[..., N]` in `x.dtype`. the backend to try first is resolved as:

1. `backend=` if given;
2. the autotune cache entry for `(sm_version, N, K, m_bucket(M))`;
3. the top of the active priority profile for that m bucket (and, under `v9`, that shape class).

if the chosen backend raises, the remaining entries of the priority list are tried in order. `bf16_dequant` is always last and always works. a plain bf16 weight only ever resolves `bf16_native`. `resolve_backend(x, w)` returns the name `linear` would actually use, which is what the runtime memoizes per m bucket.

registered backends, by activation precision:

| backend | kernel | activations |
|---|---|---|
| `bf16_native` | `F.linear` on a never-quantized bf16 weight | bf16 |
| `vllm_marlin_fp8_w8a16` | vllm marlin, gptq-repacked fp8 weight | bf16 |
| `machete_w8a16` | vllm machete (cutlass mixed input), int8-requantized weight with 128-group scales | bf16 |
| `deepgemm` | `fp8_gemm_nt` with tma-aligned block scales and 1x128 activation groups | fp8 |
| `flashinfer_fp8_blockscale` | `flashinfer.gemm.fp8_blockscale_gemm_sm90`, native 128x128 scales, no repack | fp8 |
| `vllm_block_fp8_cutlass` | vllm `cutlass_scaled_mm` with block scales | fp8 |
| `vllm_block_fp8_triton` | vllm's triton block-scaled kernel | fp8 |
| `scaled_mm_pertensor` | `torch._scaled_mm`, one scale per tensor | fp8 |
| `vllm_cutlass_fp8_pertensor` | vllm `cutlass_scaled_mm` with per-tensor scales | fp8 |
| `bf16_dequant` | dequantize inside the call, then `F.linear` | bf16 |

`available_backends()` lists them; `m_bucket(m)` rounds up into `M_BUCKETS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 3072, 4096, 8192]`.

### priority profiles

`BACKEND_PRIORITY_PROFILES` holds four named cold-start tables, selected with `set_backend_priority_profile(name)` (`--gemm-priority` on the server):

| profile | shape |
|---|---|
| `v8` (default) | marlin first at m of 1 to 16, deepgemm first from 32 to 512, `vllm_cutlass_fp8_pertensor` first from 1024 up |
| `v9` | `v8` plus a per-shape overlay (`V9_BACKEND_PRIORITY_BY_SHAPE_AND_M`) for the buckets a per-shape sweep measured; `priority_for(m, n, k)` consults it first |
| `v7` | `v8` at m of 512 and below, with every larger bucket clamped to the m=512 answer |
| `v4` | marlin at 32 and below, `flashinfer_fp8_blockscale` from 64 to 256, `scaled_mm_pertensor` at 512 |

every profile is a total order over every registered backend at every bucket, which is what keeps the fallback chain exhaustive. `priority_for_m(m)` is the shape-agnostic query.

`rows_per_sequence(n)` is a scope for the speculative verify pass: a `B*(k+1)`-row window routes like the `B`-row decode step it replaces, so the two paths resolve the same kernel.

### accuracy modes

`BACKEND_REL_L2` records each backend's relative l2 error against an fp32 dequantization of the block-128 weight, measured by `gemm_numerics.py` on random block-scaled weights and confirmed on real checkpoint tensors. the errors are flat in m and shape and fall in three tiers: w8a16 backends near bf16's own rounding (`bf16_dequant` 2.4e-3, marlin 3.5e-3, machete 7.4e-3), w8a8 block-scaled backends at 2.7e-2 (the e4m3 activation mantissa), and per-tensor w8a8 at 3.9e-2.

`set_gemm_accuracy("strict")` (`--gemm-accuracy strict`) filters every priority list to backends at or below `STRICT_REL_L2_MAX = 5e-3` and ignores speed-ranked cache entries that fail it. `"fast"` (the default) ranks on speed only. `gemm_accuracy(mode)` is the scoped form.

### weight-cache policy

marlin, machete, deepgemm and the two per-tensor backends each memoize a repacked copy of the weight, about the size of the fp8 tensor it shadows. `set_weight_cache_policy` (`--gemm-weight-cache`) controls how many such copies a weight may hold: `multi` (any), `single` (one slot, the serving default), `none`. `repack_cache_bytes(w)` and `free_repack_caches(w)` expose the accounting the memory planner uses. under `single`, which backend claims the slot is decided by whichever bucket warms up first; the server's `--gemm-cache-owner {decode,prefill}` picks.

## autotune

```bash
PYTHONPATH=engine python -m qwenfast.gemm.autotune --device cuda:0 --graph-n-capture 20
```

times every registered backend at every `(N, K)` in `MODEL_GEMM_SHAPES` and every bucket in `M_BUCKETS` under cuda-graph replay, and writes `autotune_cache/sm<major*10+minor>.json` keyed `"{N}x{K}x{M}"` with the winner and all timings. `linear(..., use_autotune=True)` reads it lazily. `--out-dir` relocates the cache; `invalidate_cache()` clears the in-process copy.

## benchmarks

```bash
# speed: every model shape x m bucket x backend, graph-timed, incremental jsonl
PYTHONPATH=engine python -m qwenfast.gemm.bench_gemm --out gemm_sweep --device cuda:0 \
  --m-buckets 1,8,32,64,128,256,512 --backends deepgemm,vllm_marlin_fp8_w8a16 --emit-priority

# accuracy: relative l2 per backend against the fp32 reference, no timing
PYTHONPATH=engine python -m qwenfast.gemm.gemm_numerics --out gemm_numerics --device cuda:0 \
  --real-weights /path/to/Qwen3.8-27B-FP8 --lm-head-dir /path/to/Qwen3.8-27B-FP8
```

`bench_gemm.py` reports `eager_us`, `graph_us` and `launch_overhead_us` per cell, flags backends that cannot be captured, times the activation-quantization step on its own at small m (`quant_graph_us`), and with `--emit-priority` / `--emit-shape-priority` writes ranked tables ready to compare against `dispatch.py`. `--shapes` isolates one shape, which is useful for backends that jit per shape. `mixed_gemm_probe.py` runs the large-m experiments (`--probe deepgemm-config,deepgemm-knobs,split,pad,quant`).

## tests

```bash
python engine/qwenfast/gemm/tests/test_gemm.py
# or
python -m unittest discover -s engine/qwenfast/gemm/tests -v
```

cpu coverage: fp8 row fusion equals dequantize-then-concatenate, block alignment is enforced, bf16 quantize round trip, the bf16 fallback path, the column-major operand layouts the cutlass and `_scaled_mm` kernels require, every priority profile (`TestPriorityForM`, `TestBackendPriorityProfiles`), `M_BUCKETS` and `m_bucket`, the priority-table derivation, the weight-cache policy, and `save_fused` / `load_fused`. on a gpu, `TestDispatchBackendsOnGPU` and `TestLargeMBackendsOnGPU` compare every backend against a bf16 reference at real shapes and `TestGraphCapturabilityOnGPU` probes capture per backend.
