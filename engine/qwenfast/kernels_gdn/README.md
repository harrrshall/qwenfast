# qwenfast.kernels_gdn

the token mixer of `qwen3.8-27b`'s 48 gated-deltanet layers: recurrent decode, the depthwise causal conv, chunked varlen prefill, and the speculative verify-and-commit path. three interchangeable backends sit behind one api: hand-written `triton` kernels for this exact shape, the `fla` (flash-linear-attention) kernels, and a pure-`torch` reference that runs on cpu.

the package never imports `qwenfast.model`; the tests pull in the reference model as an independent oracle.

## why a custom kernel

a decode step must read and write each sequence's recurrent state exactly once. that is 144 mib per sequence in fp32 (72 mib in fp16) across the 48 layers, and past a batch of about 100 it is the term that bounds throughput (see [docs/architecture.md](../../../docs/architecture.md)). upstream `fla` cannot index a slot pool, so using it from one costs a gather, the kernel and a scatter, three passes for one. the triton decode kernel resolves the slot inside the kernel and moves the state once each way; the window kernel does the same for a `k+1` token speculative window.

## files

| file | owns |
|---|---|
| `api.py` | the public entry points, backend resolution and fallback |
| `state.py` | pool allocation, per-layer views, conv-ring layout helpers, gather/scatter |
| `shapes.py` | the model constants (`NUM_GDN_LAYERS`, `NUM_V_HEADS`, `CONV_DIM`, ...) and byte accounting |
| `triton_kernels.py` | `_gdn_decode_kernel`, `_gdn_window_kernel`, `_conv_update_kernel`, `_conv_prefill_kernel` |
| `triton_ops.py` | launch wrappers, tiling tables, variant selection, validation |
| `torch_ops.py` | the reference implementation of every op, plus `prenormalize_qk` |
| `fla_ops.py` | `fla` recurrent and chunked kernels through the pool |
| `fla_static.py` | the `fla` chunk kernel with precomputed index tensors, for cuda-graph capture |
| `bench_kernels_gdn.py` | gpu microbenchmark with roofline fit and tuning sweeps |
| `tests/test_kernels_gdn.py`, `tests/test_fla_static.py` | cpu tests, plus gpu tests that skip without cuda |

## layout contract

```
recurrent state pool : [n_slots, 48 layers, 48 v-heads, 128, 128]   fp32 | fp16 | bf16
                       3 MiB per layer per slot -> 144 MiB per slot in fp32
conv state pool      : [n_slots, 48 layers, 3, 10240]               bf16, width-major
                       60 KiB per layer per slot
```

- slot-major: one slot is one contiguous span, so host swap for preemption is a single copy.
- a per-layer view is `state.layer_state(pool, layer_idx)`, shape `[n_slots, 48, 128, 128]`; it is a strided view and every kernel reads `stride(0)` explicitly, so nothing is copied.
- the slot index is a `slot_ids[B]` int32 device tensor. the batch is never compacted and a captured graph sees fixed pointers.
- the state is `[..., K, V]` (k-major), matching `fla`'s default `state_v_first=False`.
- the conv ring is width-major (`[..., W-1, C]`) so a decode step reads all channels at one tap contiguously. `alloc_conv_state_pool(..., layout="channel_major")` gives the reference layout; both are auto-detected from the shape, and `prepare_conv_weight(w)` reorients a `[C, W]` weight once at load.
- 16 key heads feed 48 value heads: value head `hv` uses key head `hv // 3`. the canonical q/k input has 16 heads; 48-head input is accepted and collapsed.

## api

```python
from qwenfast.kernels_gdn import api, state

pool  = state.alloc_state_pool(n_slots=512, dtype="fp16", device="cuda")
cpool = state.alloc_conv_state_pool(n_slots=512, device="cuda")
lyr   = state.layer_state(pool, layer_idx)         # [512, 48, 128, 128]
clyr  = state.layer_conv_state(cpool, layer_idx)   # [512, 3, 10240]
```

| entry point | contract |
|---|---|
| `gdn_decode_step(q, k, v, g, beta, state_pool, slot_ids, *, backend, out, A_log, dt_bias, qk)` | `q, k [B, 1, 16, 128]`, `v [B, 1, 48, 128]`, `g, beta [B, 1, 48]`; returns `[B, 1, 48, 128]`; pool updated in place |
| `gdn_decode_multi(...)` | same over `T > 1` tokens in one pass |
| `gdn_prefill_chunked(q, k, v, g, beta, cu_seqlens, initial_state, *, state_pool, slot_ids, output_final_state, chunk_size, backend, chunk_indices, chunk_offsets)` | packed varlen (`B == 1`, `cu_seqlens [N+1]`); returns `(out, final_state)`; scatters into the pool when given `slot_ids` |
| `causal_conv_update(x, conv_state_pool, slot_ids, w, *, bias, activation, backend)` | `x [B, C]` or `[B, C, 1]`, `w [C, 4]`; ring updated in place |
| `causal_conv_prefill(x, w, *, cu_seqlens, conv_state_pool, slot_ids, initial_state, ...)` | `x [B, C, T]` |
| `causal_conv_prefill_varlen(...)` | the packed form with the triton token-major kernel |
| `gdn_verify(...)` | outputs for all `k+1` window tokens, no state write |
| `gdn_commit(..., m)` | advance the state to `S_m` from the same inputs |
| `gdn_verify_and_commit(..., m, *, method)` | both; `method` in `{auto, two_phase, fused}` |
| `causal_conv_verify_and_commit(x, ..., m)` | conv over the window, ring committed at `m` |

conventions:

- `q` and `k` arrive un-normalized. the l2 norm (eps 1e-6 inside the sqrt) and the `1/sqrt(128)` scale happen in the kernel.
- `g` is `-exp(A_log) * softplus(a + dt_bias)` in log space and `beta` is already sigmoided. pass `A_log=` and `dt_bias=` and the kernel reads `g`/`beta` as the raw `a`/`b` projections and computes the gate itself; only the triton backend fuses this, the others apply it eagerly with identical numerics.
- pass `qk=` from `torch_ops.prenormalize_qk(q, k)` to take the prenormalized path, which leaves the kernel two reductions per program in place of five.
- all accumulation is fp32 whatever the pool dtype; `--ssm-state-dtype` only changes what is stored.
- `m` is a `[B]` int32 device tensor and never syncs to the host, so a speculative step stays inside one cuda graph.

### backends

`backend=` in `{"auto", "torch", "fla", "triton"}`, resolved per op:

| op | preference |
|---|---|
| decode, verify, commit | `triton`, then `fla`, then `torch` |
| prefill | `fla`, then `torch` |
| conv update, varlen conv prefill | `triton`, then `torch` |
| conv prefill (dense) | `torch` |

`available_backends()` reports what can run and why not. an explicit backend that is unavailable raises; an explicit backend with no implementation for that op falls through. `set_default_backend` / `get_default_backend` change the process default.

## the triton kernels

decode. grid `(cdiv(128, BV), 48, B)`: one program per (value tile, value head, slot). each program resolves `slot_ids[i]`, loads `q`/`k` for key head `hv // 3`, loads its `[128, BV]` state tile once, computes `S *= exp(g); kv = S^T k; d = (v - kv) beta; S += k (x) d; o = S^T q` in fp32 registers, and stores the tile once. the reduction runs over the k axis so the v axis stays contiguous. loads and stores carry `evict_first`, since the state has no reuse within a step. a negative slot id skips the row with the state untouched, which is how padded graph rows are handled.

window. `_gdn_window_kernel` is the same body with a `T`-step loop and a second register tile `S_commit`, refreshed while `t + 1 <= m[i]`. `COMMIT=False` is a pure verify (one read, no writes); `COMMIT=True` with `m=None` is multi-token decode; `COMMIT=True` with `m` is the fused verify-and-commit (one read, one write); `MASK_PAST_M=True` zeroes `g`/`beta` past `m` in-kernel, so a commit replay needs no mask tensor.

conv. `_conv_update_kernel` does the depthwise conv and the ring shift in one pass with `WIDTH` as a `constexpr`. `_conv_prefill_kernel` parallelizes over tokens for the prefill shape (a few sequences, thousands of tokens each), with `_conv_prefill_state_kernel` writing the final ring.

variants. three orthogonal compile-time switches exist on the decode and window kernels: `packed` (16-bit state moved as int32 words), `hoist` (alignment hints and pointer math lifted out of the loop), `sched` (state load issued first). `DECODE_VARIANT_TABLE` and `WINDOW_VARIANT_TABLE` pick per batch and dtype; `packed_sched` is used for 16-bit state at batch 512 and above, everything else runs `base`. `TestGpuPackedDecode` asserts the adopted variants are bit-identical to `base`.

### tuning knobs

`triton_ops.pick_decode_tiling` reads measured tables (`DECODE_TABLE`, `WINDOW_TABLE`, `CONV_TABLE`) keyed by batch and state itemsize, with an analytic fallback for shapes never swept. every knob is overridable from the environment:

| variable | default | effect |
|---|---|---|
| `QWENFAST_GDN_DECODE_BV`, `QWENFAST_GDN_DECODE_WARPS` | 0 (table) | tile width over the v axis and warps per program |
| `QWENFAST_GDN_DECODE_EVICT` | 1 | `evict_first` on the state load and store |
| `QWENFAST_GDN_DECODE_VARIANT` | table | pin a variant, e.g. `packed_sched` |
| `QWENFAST_GDN_WINDOW_BV`, `QWENFAST_GDN_WINDOW_WARPS`, `QWENFAST_GDN_WINDOW_VARIANT` | table | the same for the window kernel |
| `QWENFAST_GDN_CONV_BC`, `QWENFAST_GDN_CONV_WARPS` | table | channels per program and warps for the conv update |
| `QWENFAST_GDN_CONVP_BT`, `QWENFAST_GDN_CONVP_BC`, `QWENFAST_GDN_CONVP_WARPS` | 32, 128, 4 | the prefill conv tile |

`triton_ops.VALIDATE` (default `True`) runs per-call shape and stride checks; turn it off only behind a validated graph capture.

## speculative verify-and-commit

`gdn_verify_and_commit` computes outputs for all `n = k+1` draft tokens from the committed state and writes back only `S_m`, the state after the accepted prefix. there is no per-draft snapshot, which would cost `(k+1) x 144 MiB` per sequence.

- `method="two_phase"`: two stock-kernel passes on any backend. phase a runs the window with no state write. phase b runs it again with `g` and `beta` zeroed for `t >= m`, which makes the recurrence the identity past `m`, so the final state is exactly `S_m`. one and a half times the traffic of plain decode.
- `method="fused"`: the triton window kernel keeps `S_m` in a second register tile and stores it once. the same traffic as plain decode.

the fused kernel needs `m` at launch, and a chain verifier only knows `m` after the last layer's logits. `gdn_verify` and `gdn_commit` are exposed separately so the engine can verify all 48 layers, sample `m`, and commit with the window kernel in `MASK_PAST_M` mode; the fused method serves a deferred-commit design that replays the accepted prefix at the head of the next window. conv rollback is a gather: `causal_conv_verify_and_commit` sets the ring to `concat(state, x)[m : m+3]`.

## graph-capturable prefill

`fla`'s chunk kernel builds its varlen index tensors on the host with a pageable copy and memoizes them on tensor identity, so it cannot be captured directly. `fla_static.build_chunk_meta` computes `chunk_indices` and `chunk_offsets` on the host from the segment lengths, `max_chunk_rows` bounds the row count for a fixed `(chunk_tokens, n_segments)`, and `chunk_gdn_static` passes the buffers straight into `chunk_gated_delta_rule_fwd` inside a `static_index_scope` that replaces `fla`'s memoized helpers for the duration of the call. unused rows duplicate the last real row, which is idempotent because the four sub-kernels are pure per-chunk functions with no atomics. this is what lets the mixed prefill+decode step run as one cuda graph.

## tolerances

| comparison | atol |
|---|---|
| any backend vs the reference model, fp32 state | `2e-4` |
| triton or fla vs torch, fp16 state | `2e-3` |
| triton or fla vs torch, bf16 state | `2e-2` |
| bf16 activations (conv) | `1e-2` |
| fla chunk path | `2e-4 + 4e-3 * max|ref|` |

the chunk kernel runs its intra-chunk matmuls in `tl.dot` at tf32 precision, so its state error scales with the state magnitude; `TRITON_F32_DEFAULT=ieee` forces ieee fp32 and pulls it under the absolute bound. decode does no matmul and accumulates in plain fp32 registers. the fp16 and bf16 rows compare against the torch backend at the same pool dtype, so they test the kernel; whether a reduced-precision state is good enough for the model is a question for `evals/`.

## tests

```bash
python -m pytest engine/qwenfast/kernels_gdn/tests -q
# or, stdlib only
python engine/qwenfast/kernels_gdn/tests/test_kernels_gdn.py
python engine/qwenfast/kernels_gdn/tests/test_fla_static.py
```

cpu coverage: the torch backend against the reference model for recurrent, chunked, conv update and conv prefill; chunked equals sequential; the pool layout byte for byte and the layer view as a view; slot gather with permuted and gapped ids; the 16-head and 48-head input forms; varlen prefill equals per-sequence prefill; state threading across split prefill calls; verify-and-commit equals sequential decode for every `m` in `0..k` on both methods; conv verify-and-commit; both conv ring layouts; gate-in-kernel equals precomputed `g`/`beta`; the prenormalized path; a program-by-program cpu simulation of the triton decode kernel's indexing; the tiling heuristic; variant selection and the packed state view; backend dispatch and fallback; and the static chunk index arithmetic.

on a gpu: fla and triton decode against the oracle at fp32, fp16 and bf16 state, negative-slot skip, the multi-token window, the triton conv across both layouts, chunked prefill, verify-and-commit for every `m` on every backend and method, `TestGpuStateDtypeDrift` (2048 steps of fp16 next to fp32 state, asserting the recurrence damps rounding), `TestGpuCudaGraph` (captured decode and verify-and-commit replay correctly, including a changed `m` without recapture), and the packed variants against `base`.

## benchmark

```bash
PYTHONPATH=engine python -m qwenfast.kernels_gdn.bench_kernels_gdn --out gdn_bench \
  --batches 1,8,32,64,128,256,512 --state-dtypes fp32,fp16,bf16 \
  --backends triton,fla,torch --spec-k 3 --ctx 2048 --iters 100 --warmup 20
```

writes `<out>.json` and `<out>.md`. per backend and state dtype it reports one decode step of one layer (`mean_us`, `p50_us`, `p99_us`, achieved gb/s against the ideal one-read-one-write traffic, the extrapolated 48-layer ms), the conv update, the chunked prefill at `--prefill-tokens`, and both verify-and-commit methods. graph-replayed timing is the default (`--no-graph` for eager only). the report fits `t/B = bytes/BW + c + e(dtype)` by least squares over batches at or above `--roofline-min-batch` and prints the speculative-decoding economics table (microseconds per accepted token against plain decode).

useful switches: `--sweep-tuning --sweep-op {decode,window,conv} --tuning-batches 1,8,64,256` re-derives the `(BV, num_warps)` tables on the current gpu and reports registers and spills straight off the compiled kernel; `--variants all` (or e.g. `packed+sched`) compares the kernel variants with a roofline fit per variant; `--prenorm` benches the prenormalized path; `--conv-layouts` benches both ring layouts; `--sections decode,conv,spec,prefill` selects benchmark families. no optional dependency can make it raise; a missing backend records its error string.
