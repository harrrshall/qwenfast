# qwenfast runtime

`qwenfast.runtime` is everything between "weights are in hbm" and "tokens come out": the fused-weight model with its two forwards, cuda graph capture per batch bucket, the continuous-batching scheduler, mtp speculative decoding, the `AsyncEngine` implementation the http server talks to, and the `python -m qwenfast.runtime.serve` entry point. it also holds the offline benchmarks and profilers that measure a step from the inside.

import it with `PYTHONPATH=engine`. `import qwenfast.runtime` needs only `torch`; every gpu dependency (`flashinfer`, `triton`, `fla`) is imported at its point of use, so the package and its cpu tests load on a machine with no cuda.

## file map

| file | owns |
| --- | --- |
| `__init__.py` | lazy re-exports: `RuntimeConfig`, `SpecConfig`, `GRAPH_BUCKETS`, `build_engine`, `build_async_engine` |
| `fused_model.py` | `FusedQwenForCausalLM`: fused weights, gdn kernels, paged kv, `RuntimeConfig`, `DeviceBuffers`, `PrefillBatch`, `MixedBatch`, `decode_forward`, `prefill_forward`, `mixed_forward`, `FusedMTPHead` |
| `graphs.py` | `GraphedDecoder`: cuda graph capture per bucket over one shared mempool, and `sample_tokens`, the graph-safe sampler |
| `mixed_graphs.py` | `MixedGraphRunner`, `MixedPadSpec`, `pad_mixed_step`: the padded, captured mixed prefill+decode step |
| `scheduler.py` | `Scheduler`, `SlotManager`, `Request`, `GenParams`, `StepEvent`: admission, chunked prefill, decode, preemption, async scheduling |
| `spec_decode.py` | `SpecConfig`, `SpecDecoder`, `build_spec_decoder`: mtp draft, windowed verify, device-side accept and commit |
| `engine.py` | `QwenFastEngine` (the `AsyncEngine`), `build_engine`, `build_async_engine`, `EngineComponents` |
| `engine_core.py` | `EngineCoreClient` and `run_core`: the engine loop in its own process over zeromq |
| `serve.py` | the cli, the serving defaults, the memory plan, `add_runtime_args`, `build_engine_from_args` |
| `preset.py` | `CANONICAL_FAST`, `CANONICAL_BENCH`, `--preset fastest`, `format_resolved_config` |
| `step_trace.py` | `StepTrace` (loop wall-clock split) and `StepProfiler` (per-step host and device phases) |
| `profile_step.py` | one decode step, opened up: resolved gemm backends, host vs device, kernel table, ablations |
| `profile_serving.py` | the scheduler driven as a load test drives it, with a step ledger and prefill-chunk components |
| `prefill_attrib.py` | additive, sync-free per-component timing of one prefill chunk via cuda events |
| `bench_runtime.py` | offline decode and prefill tok/s against the analytic bandwidth ceiling |
| `bench_spec.py` | speculative decoding correctness gate on the real model plus a graph-timed sweep |
| `bench_overlap.py` | does a graphed decode step hide under a graphed prefill chunk on a second stream |
| `fused_ops/` | `gdn_gate.py` and `swiglu.py`: one-launch triton kernels behind `--fused-ops-backend triton` |
| `tests/` | cpu test suite on a tiny random-weight model, see below |

## the forward abi

`FusedQwenForCausalLM` is the stateless-per-step half of the runtime. it owns the weights and the two pools (the ssm state pool and the paged kv pool, indexed by the same slot id) and exposes three forwards.

`decode_forward(buf, batch)` runs one token for each of `batch` slots. it reads `input_ids`, `positions` and `slot_ids` out of `DeviceBuffers`, updates the ssm state and the kv pages in place, and writes `buf.logits[:batch]`. it has no host sync, no data-dependent control flow and no value-dependent allocation, which is what makes it capturable.

`prefill_forward(batch)` takes a `PrefillBatch`: a packed varlen chunk (`token_ids`, `positions`, per-token `slot_ids`, `cu_seqlens`, no padding) that threads gdn state through the slot pool and writes k/v straight into the pages. the mlp is tiled at `mlp_tile_tokens` so a large chunk never materialises the full gate/up activation. prefill is eager by default.

`mixed_forward(batch)` takes a `MixedBatch`, whose token axis is `[prefill chunk tokens, one row per running sequence]`, so the decode rows ride the chunk's gemms. it is enabled with `--mixed-forward`.

`DeviceBuffers` holds the fixed device tensors a decode step reads: `input_ids`, `positions`, `slot_ids`, the flashinfer paged layout (`kv_indptr`, `kv_indices`, `kv_last_page`, `seq_lens`), the per-row sampling parameters (`temperature`, `top_p`, `top_k`, `presence_penalty`, `repetition_penalty`), `out_tokens` and `logits`. nothing is reallocated after construction; each buffer has a pinned host staging tensor and `upload()` copies the named ones to the device before a step.

one slot id addresses both pools. `PagedKVPool` has its own slot free list; the runtime hands every slot to it once at construction (`_claim_all_kv_slots`) so `scheduler.SlotManager` is the single allocator and the two pools can never disagree about which request owns a row. the decode path appends kv through `append_kv_graph_safe`, a pure scatter with no host-side validity check, because the scheduler already guarantees capacity before every step.

numerics follow the reference model and are asserted against it in `tests/test_runtime.py`: rmsnorm accumulates in fp32 and scales by `1 + weight`, the gated rmsnorm after gdn scales by the plain weight then multiplies by `silu(gate)`, q/k are l2-normalised inside the gdn kernel, and attention applies q/k norm before rope, partial rope over the first 64 of 256 dims, and a sigmoid output gate before `o_proj`.

## cuda graph buckets

`GraphedDecoder` captures the whole decode step, embedding gather through 64 layers, final norm, `lm_head` and sampler, once per bucket in `RuntimeConfig.graph_buckets`:

```
1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512
```

`buckets_for()` truncates that table at `max_num_seqs`. a batch of `b` rows replays the smallest bucket at or above `b`; padding rows point at the model's scratch slot. all buckets share one `torch.cuda.graph_pool_handle()`, so graph memory is roughly one pool, whatever the bucket count.

what is captured and what is not is the one correctness-critical split in `graphs.py`:

- captured: `decode_forward` and `sample_tokens`.
- outside the graph, before every replay: `AttentionRunner.plan_decode` (host-side flashinfer planning) and the fill of `input_ids`, `positions` and `slot_ids` from the scheduler's own bookkeeping. the graph never carries a sampled token into the next step's input on its own, because under continuous batching the row that holds a given slot can change between two steps.

`sample_tokens` is a fused greedy, temperature, top-p and top-k sampler built from straight-line tensor ops: no `.item()`, no `torch.multinomial`. it restricts each row to the top `sampler_candidates` logits (default 2048) before any sort, then draws with gumbel-max, and selects greedy per row with `torch.where(temperature <= 0, ...)`.

`--no-graphs` (or a non-cuda device) makes `capture()` a no-op and `step()` always runs the same `_run_step` body eagerly. capture requires the flashinfer attention backend; the torch attention fallback does its own host sync and `capture()` refuses it with a clear error.

## scheduler

`Scheduler` is single-threaded, pure engine-loop logic: no asyncio, no http, no tokenizer. each request is `WAITING` (with or without a slot), `DECODING` or `DONE`.

admission. a request gets an ssm slot and reserved kv pages when a slot is free and `free_kv_pages >= ceil(len(prompt)/page_size) + ceil(max_tokens/page_size)`. the second term reserves room for the request's own decode tail so it rarely needs preemption later. `context_length_error` is the one helper both the http layer and the engine use to reject `prompt + max_tokens > max_model_len`, so the advertised and the enforced limit are the same number.

steps. by default one `step()` runs either a packed varlen prefill chunk of up to `max_num_batched_tokens` tokens across as many waiting requests as fit, or one decode step over every running request. `prefill_decode_ratio` (default 4) bounds how many decode steps run between prefill opportunities and trades ttft against tpot. `prefill_chunk_tokens` caps a single chunk below the batched-token budget, which is a tpot knob.

mixed and overlapped steps. with `--mixed-forward`, a step with both waiting and running requests runs one forward over the chunk and every decode row (`_run_mixed_step`), and the ratio counter has no job. `--mixed-graphs` pads that step to a fixed `(prefill_chunk_tokens, decode bucket)` shape and replays it through `MixedGraphRunner`; padding segments and rows live on the scratch slot, whose state `reset_scratch_state` zeroes before every step. `--overlap` runs the prefill graph and the ordinary decode graph on two cuda streams at once, each with its own mempool and flashinfer workspace (`_run_overlap_step`).

preemption swaps state to host memory: a hybrid model's ssm state cannot be rebuilt cheaply by replaying the prompt. when a running request needs a page and none is free, `_ensure_capacity_with_preemption` picks the last-admitted victim, `_preempt` copies its ssm and conv state rows and its committed kv to host tensors, frees the slot and pages, and requeues it at the front of the waiting queue. `_restore_swapped` reverses that on re-admission without re-running prefill.

the scheduler never spins. `step()` sets `last_step_progressed` explicitly, and the engine loop sleeps briefly whenever a step made no progress, so a request that cannot be admitted yet waits without pinning a core.

## engine process and async scheduling

`QwenFastEngine` implements the `AsyncEngine` contract from [`../server/engine_api.py`](../server/engine_api.py): `add_request(request_id, prompt_token_ids, sampling_params)` returns an async iterator of `StepOutput`, `abort(request_id)` cancels, `get_stats()` returns an `EngineStats` with ttft and tpot histograms. `start()` calls `prepare()` (warmup, graph capture for the decoder, the spec decoder and the mixed runner, then the post-capture memory gate) and launches the loop on a daemon thread; asyncio talks to that thread through thread-safe queues. a fatal error in the loop releases every waiter and `health()` reports it, so `/health` returns 503 and clients never hang on a dead engine.

`--engine-process` moves the loop into a second interpreter. `EngineCoreClient` (the `AsyncEngine` the http process sees) spawns `engine_core.run_core`, which builds the same `QwenFastEngine` through `build_engine_for_core`, runs `prepare()`, and drives `engine.scheduler` directly. the two processes talk over zeromq ipc sockets with one message per step carrying the whole batch; a send that would block is parked and retried after the next step. the http side keeps tokenisation, sse, detokenisation and json to itself, and the engine loop never waits for its gil. `--engine-stats-interval` sets how often the core pushes a `/metrics` snapshot, `--engine-start-timeout` bounds weight loading and capture, and `--engine-idle-poll-ms` is the socket wait when the scheduler has no work.

`--async-scheduling` changes what a step is. the host launches step n, then commits step n-1: it waits on that step's cuda event, reads its tokens out of pinned memory, applies stop conditions and builds events while the device is already running step n. the token sampled by step n is an input of step n+1, so it is fed by a device-side gather (`Request.pending_src`), and the host learns each token one step late. the cost is at most one computed-and-discarded token per request after eos and a step of reporting latency on ttft and tpot. a speculative step commits a device-resident, variable number of tokens, so the pipeline drains before one and refills after it.

## speculative decoding

`SpecDecoder` implements mtp speculative decoding with draft length `k` (`--spec-k 1`, `2` or `3`). one speculative step for `b` sequences:

1. draft `k` sequential mtp steps from the last committed token and the main model's previous post-norm hidden state.
2. verify with one main-model forward over the whole `[b, k+1]` window. gdn layers use the window kernel without committing state and cache their per-position inputs; attention layers use a multi-token causal paged call.
3. accept greedily: draft `j` is accepted while it equals `argmax(logits[j-1])`. the accepted count `m` is a device tensor and never leaves the device inside the step.
4. commit a second pass over the cached gdn inputs with `m` (`gdn_commit`), and move the conv ring.
5. carry the hidden state at `m-1` for the next draft.

nothing is undone on rejection: ssm and conv state are never written speculatively, and kv written for the whole window is rolled back by a pointer move, since the next window starts at the committed length and overwrites it.

`SpecConfig.greedy_only` is on, so the emitted stream is bit-identical to plain greedy decoding. a step is speculative only when every live request has `temperature <= 0` and the live batch is at most `--spec-max-batch` (default 16); otherwise the plain graphed step runs, and `SpecDecoder` captures graphs only for buckets up to that cap (`serve.spec_buckets_for`). `SpecConfig.prefill_mtp` runs the mtp head over every prefill chunk so the draft head has real context. `tests/test_spec_decode.py` asserts the equivalence on the tiny model; `bench_spec.py` asserts it on the real checkpoint before timing anything.

## presets

`preset.CANONICAL_FAST` is the one canonical runtime configuration: the measured-fastest `RuntimeConfig` knobs, and the configuration the server ships. `serve.M1_DEFAULTS` must equal it on every shared key, and `tests/test_preset.py::TestPresetMatchesServe` is that assertion, so the configuration that is benchmarked is the configuration that is served. `RuntimeConfig`'s own dataclass defaults stay conservative reference values (fp32 ssm state, torch norms, torch fused ops, `gemm_weight_cache="multi"`).

`--preset fastest` on any cli (`serve`, `bench_runtime`, `bench_spec`, `profile_step`, `profile_serving`, `bench_overlap`) fills in every knob left at its cli default from the preset; an explicitly passed flag always wins, so `--preset fastest --gemm-backend deepgemm` is an a/b arm. `CANONICAL_BENCH` holds the measurement knobs (bucket rule, steps, warmup, repeats, context), and `format_resolved_config` prints the resolved configuration, with its deviations from the preset, before a benchmark's first number. `canonical_fast_config(**overrides)` builds a `RuntimeConfig` at the preset from python.

## serving

```bash
PYTHONPATH=engine python -m qwenfast.runtime.serve \
    --model /path/to/Qwen3.8-27B-FP8 --port 8000 --api-key secret
```

`python -m qwenfast.server --engine qwenfast` is the same program: `server/cli.py` imports `add_runtime_args` and `build_engine_from_args` from this module, so the two entry points cannot drift. startup prints a memory plan (weights, gemm repack caches, kv pool, ssm and conv state, flashinfer workspaces, device buffers, graph pool, cuda context, prefill activations) and refuses to launch when it exceeds `--gpu-memory-utilization` of free hbm, with a message that names the knobs to lower. after graph capture the measured allocation is compared with the plan (`--memory-plan-tolerance`).

pool geometry is derived from `--max-model-len` and `--max-num-seqs` through `bench_runtime.derive_pool_sizes`, the same rule the offline benchmark uses, so a served number and an offline number are sized alike.

the important flags (`--help` lists all of them, with the public api and reliability options from [`../server/config.py`](../server/config.py)):

| flag | default | meaning |
| --- | --- | --- |
| `--model` | required | weights directory (a snapshot dir or a fused-weights cache) |
| `--tokenizer` | `--model` | tokenizer dir or repo id |
| `--served-model-name` | `--model` | name reported by `/v1/models` |
| `--host`, `--port` | `0.0.0.0`, `8000` | bind address |
| `--api-key` | none | require `Authorization: Bearer <key>` on `/v1/*` |
| `--default-max-tokens` | `512` | `max_tokens` when a request omits it |
| `--max-num-seqs` | `256` | ssm slots, and the largest decode batch |
| `--max-model-len` | `2752` | prompt plus completion cap; drives the rotary table and the kv page count |
| `--max-num-batched-tokens` | `8192` | chunked-prefill token budget per prefill step |
| `--prefill-chunk-tokens` | `0` | cap one chunk below the budget (0 = off); a tpot knob |
| `--prefill-decode-ratio` | `4` | decode steps between prefill opportunities |
| `--page-size` | `16` | kv page size in tokens |
| `--n-kv-pages`, `--max-pages-per-seq`, `--kv-pages-slack` | derived, derived, `64` | override the derived kv geometry |
| `--dtype` | `bf16` | activation dtype |
| `--ssm-state-dtype` | `fp16` | ssm state pool dtype (`fp32` or `fp16`) |
| `--kv-cache-dtype` | `bf16` | kv pool dtype (`bf16` or `fp8`) |
| `--attn-backend` | `auto` | `flashinfer` or `torch`; graphs require flashinfer |
| `--gdn-backend` | `auto` | `torch`, `fla` or `triton` |
| `--gdn-chunk-size` | `32` | fla chunk kernel block size (`16`, `32`, `64`) |
| `--norm-backend` | `triton` | rmsnorm kernels, `torch` or `triton` |
| `--fused-ops-backend` | `triton` | fused gdn-gate and swiglu kernels, `torch` or `triton` |
| `--gemm-backend` | none | pin one gemm backend; the default dispatches per m-bucket |
| `--gemm-priority` | `v8` | gemm backend priority table (`v9`, `v8`, `v7`, `v4`) |
| `--gemm-accuracy` | `fast` | `strict` admits only backends within the strict error bound |
| `--gemm-weight-cache` | `single` | permanent repacked weight copies: `multi`, `single` or `none` |
| `--gemm-cache-owner` | `decode` | which m-bucket claims the single repack cache slot |
| `--prefill-gemm-backend` | none | force one backend for prefill-shaped gemms |
| `--mlp-tile-tokens` | `2048` | prefill mlp tile |
| `--conv-prefill-layout` | `token_major` | prefill depthwise conv path, `token_major` or `channel_major` |
| `--mixed-forward` | off | one forward per step over prefill chunk plus decode rows |
| `--mixed-graphs` | off | capture the mixed step at a fixed padded shape; needs `--mixed-forward` |
| `--mixed-graph-segments`, `--mixed-graph-min-bucket`, `--mixed-graph-holes` | `8`, `32`, off | shape of the graphed mixed step |
| `--overlap` | off | prefill and decode graphs on two streams; needs `--mixed-forward --mixed-graphs` |
| `--overlap-min-fill`, `--overlap-decode-priority` | `0.75`, `0` | eager fallback threshold and decode stream priority |
| `--async-scheduling` | off | launch step n+1 while step n runs, harvest one step late |
| `--spec-k` | `0` | mtp draft length; `0` disables speculative decoding entirely |
| `--spec-max-batch` | `16` | largest live batch that runs the speculative step |
| `--enable-mtp` | off | load the mtp head (implied by `--spec-k`) |
| `--no-graphs` | off | eager decode, debug only |
| `--sampler-candidates` | `2048` | top-k/top-p candidate pool per row |
| `--attn-workspace-mb` | `512` | flashinfer workspace |
| `--fused-cache` | none | directory holding a pre-built `fused_weights.safetensors` |
| `--eos-token-id` | tokenizer's | override the eos id |
| `--gpu-memory-utilization` | `0.94` | fraction of free hbm the memory plan may use |
| `--skip-memory-check` | off | print the plan without enforcing it |
| `--engine-process` | off | run the engine loop in a separate process over zeromq |
| `--engine-stats-interval`, `--engine-start-timeout`, `--engine-idle-poll-ms` | `0.25`, `2400`, `1` | engine-process tuning |
| `--detok-workers` | `4` | detokenizer threads in the http process |
| `--http-keep-alive-timeout` | `300` | seconds an idle http connection stays open |
| `--step-trace-out` | none | write the loop's wall-clock split to json on shutdown |
| `--step-profile-out` | none | write a per-step host and device attribution to json on shutdown |
| `--step-profile-steps`, `--step-profile-warmup`, `--step-profile-min-rows` | `400`, `200`, `0` | window for `--step-profile-out` |
| `--preset` | none | `fastest` fills unset knobs from `CANONICAL_FAST` |

fused weights are rebuilt from the checkpoint on every start unless `--fused-cache` points at a directory holding `fused_weights.safetensors`. build one once with `qwenfast.gemm.fused_weights.build_fused_weights` and `save_fused`.

## profiling tools

all of these need the real checkpoint and a gpu, and all import cleanly on a cpu machine.

| tool | question it answers |
| --- | --- |
| `python -m qwenfast.runtime.bench_runtime --model ... --out ...` | decode ms/step and tok/s per batch size at a fixed context, with and without graphs, fp32 vs fp16 state, against the bandwidth ceiling. `--batch`, `--buckets`, `--bucket-rule {batches,serve}`, `--graphs {both,on,off}`, `--prefill` |
| `python -m qwenfast.runtime.profile_step --model ... --batch 1 32` | where one decode step goes: resolved gemm backend per linear, host plan vs graph replay, kernel table from `torch.profiler`, ablation deltas |
| `python -m qwenfast.runtime.profile_serving --model ... --concurrency 32 256 --duration 60` | the real scheduler under a closed-loop load: a per-step ledger labelled prefill or decode, prefill-chunk components and ablations, host counters, a gil-contention probe |
| `python -m qwenfast.runtime.prefill_attrib` (used from `profile_serving`) | additive per-component milliseconds of one prefill chunk from cuda event pairs, one sync at the end |
| `python -m qwenfast.runtime.bench_spec --model ... --out-json ... --out-md ...` | greedy equivalence of the speculative step on real prompts, then ms/step, accepted tokens per step and effective tok/s over `--batch` x `--k` |
| `python -m qwenfast.runtime.bench_overlap --model ...` | prefill graph alone, decode graph alone, serial and overlapped, and the dilation the overlap costs the prefill half |
| `serve --step-trace-out` | the loop's wall clock split into `step`, `drain`, `emit`, `gap` and `idle`, with a per-second series |
| `serve --step-profile-out` | per-step phases (`admit`, `collect`, `build`, `plan`, `launch`, `sync`, `harvest`, `book`) plus cuda-event device idle and kernel time |

`bench_runtime` fakes the context by reserving pages and setting `seq_len`, because decode cost depends on kv traffic; `bench_spec` really prefills its prompts, because acceptance rate depends on kv content. every bench prints the resolved configuration and its deviations from `CANONICAL_FAST` before its first number.

## tests

the suite runs on cpu with a tiny random-weight model (hidden 64, two gdn layers and one attention layer, a vocab of 32 or 64) through the exact production code path with `attn_backend="torch"`, `gdn_backend="torch"` and graphs off:

```bash
PYTHONPATH=engine python -m pytest engine/qwenfast/runtime/tests
```

| file | covers |
| --- | --- |
| `test_runtime.py` | fused-model parity against `qwenfast.model`, the sampler, slot and page allocation, chunked prefill, preemption, the asyncio engine; graph capture tests run when cuda is available |
| `test_serving.py` | http to scheduler to model through `server/app.py` with a character-level tokenizer |
| `test_serving_path.py` | the varlen prefill conv, no host syncs in prefill, the prefill chunk cap, gemm cache ownership, gdn chunk-size invariance, the profiler pieces |
| `test_spec_decode.py` | speculative greedy output is identical to plain greedy for `b` in `{1, 4}` and `k` in `{1, 2, 3}` |
| `test_mixed_forward.py` | a mixed step equals a prefill chunk followed by a decode step, at the model and scheduler level |
| `test_mixed_graphs.py` | padding a mixed step to a fixed shape changes only its duration |
| `test_overlap.py` | an overlapped step emits exactly what the two separate steps emit |
| `test_async_scheduling.py` | synchronous and asynchronous scheduling produce identical tokens |
| `test_engine_process.py` | a real second process over zeromq produces the same tokens as the in-process thread |
| `test_admission.py` | the o(1) admission bookkeeping stays in sync with the page table |
| `test_preset.py` | `CANONICAL_FAST` equals the serving defaults on every shared key |
