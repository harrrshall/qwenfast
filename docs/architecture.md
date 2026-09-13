# architecture

how the qwenfast engine turns an http request into tokens. the code lives in [`engine/qwenfast/`](../engine/qwenfast/README.md); this page follows one request through it and then explains the kernels, the memory layout and the bandwidth model that shaped them.

## overview

```
  client ──http──> server/app.py ──> AsyncEngine ──> runtime/scheduler.py ──> one step
   (openai json)   tokenize, auth,    (thread or       admit, plan, pad,        │
                   stream sse         zmq process)     harvest, book-keep       │
                                                                                v
                        ┌───────────────────────────────────────────────────────────┐
                        │ FusedQwenForCausalLM            device buffers, 2 pools  │
                        │  embed -> 64 layers -> norm -> lm_head -> sampler        │
                        │   48 x GatedDeltaNet   conv_update + gdn_decode_step     │
                        │   16 x gated gqa       flashinfer paged attention        │
                        │   64 x mlp             fused fp8 gemm + swiglu           │
                        │  replayed as one cuda graph per batch-size bucket        │
                        └───────────────────────────────────────────────────────────┘
```

## the model

`qwen3.8-27b` is a hybrid. per `config.json` (flattened into `weights.QwenFastConfig`):

| part | shape |
|---|---|
| hidden / mlp / vocab | 5120 / 17408 / 248320, untied `lm_head` |
| layers | 64: 48 gated-deltanet (linear attention) + 16 gated gqa, every 4th layer |
| gated-deltanet | 16 key heads x 128, 48 value heads x 128 (3 value heads per key head), depthwise causal conv of width 4, recurrent state `[48, 128, 128]` per layer |
| gated gqa | 24 query heads x 256, 4 kv heads x 256, qk-rmsnorm, partial rope over the first 64 of 256 dims, sigmoid output gate whose projection is fused into `q_proj` |
| mtp head | one extra full-attention decoder layer plus `fc([embed ; hidden]) -> hidden`, sharing `embed_tokens` and `lm_head` with the main model |

the fp8 checkpoint (`Qwen/Qwen3.8-27B-FP8`) stores every linear as `float8_e4m3fn` `[N, K]` plus a `weight_scale_inv` grid of 128 x 128 blocks. `in_proj_a`, `in_proj_b`, `embed_tokens`, `lm_head` and `mtp.fc` stay bf16. every linear dimension is a multiple of 128, which is what makes the weight fusion below legal.

text-only mrope degenerates to standard rope: the three position rows are identical for text, so the interleave is the identity. the reference implementation in `model.py` is the correctness oracle for everything else in the package.

## the request path

1. `server/app.py` (fastapi + uvicorn) validates the openai request, applies the tokenizer's chat template, resolves thinking mode and sampling defaults, and calls `AsyncEngine.add_request(request_id, prompt_token_ids, SamplingParams)`. it streams `StepOutput`s back as sse, detokenizing incrementally and parsing `</think>` and hermes tool calls on the fly.
2. `runtime/engine.py::QwenFastEngine` implements that interface. it owns one engine thread running `Scheduler.step()` in a loop and bridges results to asyncio queues. with `--engine-process` the loop runs in a separate process (`runtime/engine_core.py`) and the http side talks to it over zeromq: one message per step for the whole batch, so json encoding and detokenization never share a gil with kernel launches.
3. `runtime/scheduler.py::Scheduler` decides what the step is: a chunked prefill, a graphed decode step, a speculative step, or a mixed prefill+decode step. it fills the fixed `DeviceBuffers` (`input_ids`, `positions`, `slot_ids`, sampling params) from host bookkeeping, runs the flashinfer plan, and hands over to the model.
4. `runtime/graphs.py::GraphedDecoder.step(batch, slots)` rounds the batch up to a bucket and replays that bucket's cuda graph. the graph contains `FusedQwenForCausalLM.decode_forward` and the graph-safe sampler, so one replay produces the next token for every row.
5. the scheduler harvests `out_tokens`, appends them to each request, evaluates stop conditions, and emits `StepEvent`s that the engine turns into `StepOutput`s.

## fused fp8 weights and gemm dispatch

`gemm/fused_weights.py::build_fused_weights` reads a safetensors snapshot and concatenates projections along the output dimension without dequantizing anything:

| fused tensor | composition | shape |
|---|---|---|
| `in_proj_qkvz` | `in_proj_qkv` + `in_proj_z` | `[16384, 5120]` fp8 |
| `in_proj_ba` | `in_proj_b` + `in_proj_a` | `[96, 5120]` bf16 |
| `qkv_proj` | `q_proj` + `k_proj` + `v_proj` | `[14336, 5120]` fp8 |
| `gate_up_proj` | `gate_proj` + `up_proj` | `[34816, 5120]` fp8 |

`out_proj`, `o_proj` and `down_proj` pass through as single `FP8Tensor`s. `save_fused` / `load_fused` cache the result in one safetensors file so a server boots from it directly (`--fused-cache`).

`gemm/dispatch.py::linear(x, w)` picks a kernel per call. ten backends are registered: `bf16_native`, `bf16_dequant`, `vllm_block_fp8_triton`, `vllm_block_fp8_cutlass`, `scaled_mm_pertensor`, `vllm_cutlass_fp8_pertensor`, `deepgemm`, `vllm_marlin_fp8_w8a16`, `machete_w8a16`, `flashinfer_fp8_blockscale`. the winner depends on m, the number of activation rows: a weight-only w8a16 kernel (marlin) wins at m of 1 to 16, a native w8a8 kernel (deepgemm) wins from 32 to 512, and per-tensor cutlass wins at prefill scale. the resolution order is an explicit `backend=`, then the autotune cache for `(sm, N, K, m_bucket(M))`, then the active priority profile (`--gemm-priority`, default `v8`); a backend that raises falls through to the next entry, and `bf16_dequant` is always last. `runtime/fused_model.py::ResolvedLinear` memoizes the resolved backend per m bucket so a captured graph never re-resolves.

two process-wide policies sit on top. `--gemm-accuracy strict` restricts the order to backends whose measured relative error clears `STRICT_REL_L2_MAX`; w8a8 kernels quantize activations to e4m3 and sit around ten times the error of the w8a16 ones. `--gemm-weight-cache single` lets each weight keep one backend's repacked copy, because every repack cache is another full copy of the fp8 linears.

## gated-deltanet kernels

`kernels_gdn/` owns the token mixer of the 48 linear-attention layers. the recurrent state lives in one pool, `[n_slots, 48, 48, 128, 128]`, slot-major so one sequence's state is one contiguous span (144 mib in fp32, 72 mib in fp16), and the conv ring in `[n_slots, 48, 3, 10240]` bf16. a `slot_ids[B]` device tensor maps batch rows to pool rows.

the hand-written triton kernels exist because a decode step must move exactly one read and one write of that state per sequence, and the upstream `fla` kernel cannot index a pool: through a slot pool it costs a gather, the kernel, and a scatter. the kernels in `kernels_gdn/triton_kernels.py`:

- `_gdn_decode_kernel`: one program per (value-tile, value head, sequence). it resolves `slot_ids[i]` inside the kernel, loads its `[128, BV]` state tile once, runs `S *= exp(g); kv = S^T k; d = (v - kv) beta; S += k (x) d; o = S^T q` in fp32 registers, and stores the tile once. q/k l2-norm, the `1/sqrt(128)` scale and, with `A_log=`/`dt_bias=`, the gate itself are computed in-kernel. a negative slot id skips the row, which is how padded graph rows are handled.
- `_conv_update_kernel`: the depthwise causal conv and the ring shift in one launch, over a width-major ring so a decode step reads all channels at one tap contiguously. `_conv_prefill_kernel` is the token-major varlen prefill conv.
- `_gdn_window_kernel`: the same body over a window of `k+1` tokens with a second register tile that snapshots the state after the accepted prefix `m`. `COMMIT=False` is a pure verify (one read, no write); `COMMIT=True` with `m` is the fused verify-and-commit (one read, one write, the same traffic as plain decode); `MASK_PAST_M` zeroes `g`/`beta` past `m` so a commit replay needs no mask tensor. `m` is a device tensor and never syncs to the host.

prefill goes through `fla`'s chunked kernel (`gdn_prefill_chunked`, `fla` then `torch`), with `fla_static.py` supplying precomputed chunk index tensors so the call is cuda-graph capturable. every entry point in `kernels_gdn/api.py` takes `backend=` in `{auto, triton, fla, torch}` and falls back automatically; all state math accumulates in fp32 whatever `--ssm-state-dtype` stores.

## attention and the two pools

`attn/kv_pool.py::PagedKVPool` stores k/v for the 16 attention layers and the mtp layer in `[17, n_pages, 2, page_size, 4, 256]`, flashinfer's nhd page layout with a leading layer axis, in bf16 or fp8 with a per-(layer, page, k/v, head) scale. `page_size` is a config field (16 by default).

the kv pool and the ssm slot pool are independent allocators with independent granularities: a slot is request-granular and needs no paging, a page is 16 tokens. `max_num_seqs` is literally the number of ssm slots, and the same slot id indexes both pools for a request.

`attn/flashinfer_attn.py` wraps `BatchDecodeWithPagedKVCacheWrapper` and `BatchPrefillWithPagedKVCacheWrapper` as a `plan()` / `run()` pair: `plan` is host-side and runs before every replay, `run` is a pure kernel launch and is what the graph captures. the decode wrapper is built with `use_tensor_cores=True`, which the 24:4 head ratio at head dim 256 requires. `append_kv` is a pure scatter (graph-safe) and `truncate` is a pointer move, which is all speculative rollback needs on the attention side. page-table updates are staged on a host mirror and flushed as one batched copy per admission pass.

## cuda-graph decode buckets

`runtime/graphs.py` captures one graph per bucket in `(1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)`, all sharing one `graph_pool_handle`. each graph holds the whole step, embed gather through sampler, against fixed device buffers; the host fills those buffers and runs `plan_decode` before each replay. rows above the live batch point at a scratch slot. the sampler (`sample_tokens`) is greedy, temperature, top-p and top-k over the top `sampler_candidates` logits with a gumbel-max draw, written as fixed-shape tensor ops with no host sync so it captures cleanly. `--no-graphs` runs the identical step body eagerly; in the served configuration the graphed step is about 4.3x faster at batch 1.

## continuous batching

`runtime/scheduler.py` runs a single-threaded loop with a three-state request machine:

```
WAITING (no slot) --admit--> WAITING (slot, prefilling) --prefill done--> DECODING --stop--> DONE
DECODING --page exhaustion--> preempted, requeued at the front
```

- admission: a request gets a slot and pages when a slot is free and `free_kv_pages >= ceil(len(prompt)/page_size) + ceil(max_tokens/page_size)`. the second term reserves room for the request's own decode tail.
- chunked prefill: a prefill step packs waiting requests varlen (no padding) into at most `--max-num-batched-tokens` tokens (8192 by default), threads gdn state through the slot pool with `cu_seqlens`, and writes k/v straight into pages. `--prefill-decode-ratio` (default 4) bounds how many decode steps run between prefill opportunities; `--prefill-chunk-tokens` caps a chunk below the budget to bound tail latency.
- preemption: a hybrid model cannot recompute ssm state by replaying a kv window, so preemption swaps. `Scheduler._preempt` copies the victim's ssm rows, conv rows and committed kv to host memory, frees its slot and pages, and requeues it at the front; `_restore_swapped` copies them back and resumes decoding with no re-prefill. the victim is the most recently admitted request.

## mtp speculative decoding

with `--spec-k k` (1 to 3) the checkpoint's own mtp head drafts. `runtime/spec_decode.py::SpecDecoder` runs one speculative step as:

1. draft `k` tokens with `k` sequential mtp passes, each consuming the embedding of the previous token and the previous hidden state.
2. verify all `k+1` window positions in one main-model forward over the packed `[B*(k+1), hidden]` batch. gdn layers use the window kernel with `commit=False`; attention layers use a multi-token causal paged call; k/v for the whole window is written speculatively.
3. accept greedily while draft `j` equals `argmax(logits[j-1])`; `m = accepted + 1` is a device int32 tensor.
4. commit with `gdn_commit` (the window kernel in `MASK_PAST_M` mode, writing `S_m` once); the conv ring moves to `concat(state, x)[m : m+3]`; kv rolls back by a pointer move at the next step.
5. carry `hidden[m-1]` as the next draft's input.

nothing about `m` reaches the host inside the step, so the speculative step has its own cuda graphs per bucket. it is greedy-only (`SpecConfig.greedy_only`): a batch containing a sampled request takes the plain decode step. `--spec-max-batch` (default 16) bounds the batch it is used at, because weight-read amortization is worth most when the gemms are nearly idle.

correctness gate. `runtime/bench_spec.py` holds the speculative path to a statistical bar because bit identity with plain decode is unattainable by construction: a kernel that consumes `k+1` tokens per sequence cannot be the same arithmetic as one that consumes one, and a 128-row gemm is tiled differently from a 32-row one. the gate has four parts, each of which fails on a bug and passes on arithmetic noise: decode and verify must resolve the same gemm backend at every `k`; two plain greedy runs must be token-identical; a `k = 0` verify window over the same context must match the plain forward at the top-1 level, with every disagreement inside a top-2 margin bounded by the measured logit noise; and for every `k` the number of diverging prompts must not exceed the `k = 0` control's count by more than a fixed slack. `SpecConfig(k=0)` is that control: the full verify path with the drafting removed.

## the mixed prefill+decode step

a separate-step design pays the weight read once per decode step and once per prefill chunk. with `--mixed-forward` the scheduler runs one forward over `[prefill chunk tokens ‖ one row per running sequence]` (`Scheduler._run_mixed_step`, `FusedQwenForCausalLM.mixed_forward`), so the decode rows ride the chunk's gemms. running requests are placed first, so a fresh prompt never starves a decoding one, and the speculative step keeps priority whenever it is eligible; the two features partition the batch axis.

`--mixed-graphs` captures that step. the varlen segmentation changes every step, and only two kernels read it (the fla chunk kernel and the varlen conv), so `runtime/mixed_graphs.py` pads the step to a fixed `(prefill_chunk_tokens, decode bucket)` shape: extra prefill segments and extra decode rows land on the scratch slot, `reset_scratch_state` zeroes it before every step, and `fla_static` feeds the chunk kernel static index buffers so the whole step is one graph. `--mixed-graph-holes` is the alternative capture with one eager hole per gdn layer. `--overlap` runs the prefill half and the decode half as two graphs on two streams.

## engine process and asynchronous scheduling

`--engine-process` moves the model, scheduler and cuda context into their own process (`runtime/engine_core.py::run_core`) with `EngineCoreClient` implementing `AsyncEngine` on the http side. the two directions are zeromq push/pull over unix sockets; the core sends one `MSG_STEP` per step with every finished row, and a send that would block is parked and retried after the next step so the loop never waits on http.

`--async-scheduling` overlaps the host half of step `N+1` with the device half of step `N`. `_step_body` launches without synchronizing: sampled tokens are published to a device buffer, one d2h into pinned memory is issued, an event is recorded. the next step feeds the previous token by a device-side gather (`Request.pending_src`), so no readback sits on the critical path; the host commits step `N` one step late (`_commit`), and a request whose eos arrived at step `N` has one extra row computed and discarded. speculative steps commit a device-resident number of tokens, so the pipeline drains before one and refills after.

## the bandwidth model

every decode step reads all the weights once and every sequence's recurrent state once in and once out. with `B` sequences at context `ctx` on an h200 (4.8 tb/s), from the config alone:

```
step_ms(B, ctx) ~= 6.19            weights: 29.7 GB of fp8 read once per step
                 + B * 0.0629      ssm state: 2 x 144 MiB per sequence (fp32)
                 + B * 6.83e-6 * ctx   kv read: 64 KiB per token (bf16)
```

two regimes follow:

- batch 1 is weight-bandwidth-bound. the 6.19 ms weight read is a floor no kernel can beat, and the rest of the step is small next to it. the only lever that changes the constant term is speculative decoding, which amortizes one weight read over up to `k+1` emitted tokens per sequence. this is why `--spec-k` is the low-concurrency configuration.
- large batch is ssm-state-bandwidth-bound. past roughly 100 sequences the `B * 0.0629` term overtakes the weight term, and the asymptote is `1 / 0.0629 ms` of about 13k tokens per second for fp32 state, about 22k for fp16. every extra pass over the state is a linear tax on that asymptote, which is the whole reason the decode kernel is pool-indexed and moves the state exactly once each way. `--ssm-state-dtype fp16` halves the term and the slot pool, and is the serving default. measured with the triton kernel at batch 32 and above, the fitted memory rate is 86 to 87 percent of peak hbm bandwidth.

prefill is a different bound: about 98 percent of its flops are the linear gemms, so it is compute-bound and the 16 attention layers are a small share of it.

## memory planner

`runtime/serve.py::plan_memory` sums every pool from the config and the flags before a weight loads: weights, gemm repack caches, kv pool, ssm and conv state, flashinfer workspaces, graph memory, prefill headroom, and the speculative buffers. the server refuses to start above `--gpu-memory-utilization` of free hbm and checks the plan again against `cudaMemGetInfo` after graph capture, before it takes traffic.
