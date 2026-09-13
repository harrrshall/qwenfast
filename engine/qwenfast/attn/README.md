# qwenfast.attn

paged kv cache, flashinfer decode and prefill wrappers, the fused pre- and post-attention ops, and the rope table for the 16 gated gqa layers and the mtp layer of `qwen3.8-27b`. everything downstream of the q/k/v projections lives here; the projections themselves are gemms owned by [`qwenfast.gemm`](../gemm/README.md).

importing the package never requires cuda or flashinfer; every gpu dependency is guarded at the point of use (`flashinfer_attn.HAS_FLASHINFER`, `HAS_FA3`, `HAS_TRITON`).

## files

| file | owns |
|---|---|
| `kv_pool.py` | `KVPoolConfig`, `PageAllocator`, `PagedKVPool`: bf16/fp8 paged storage, page table, `append_kv`, `truncate` |
| `rope.py` | `RotaryTable`, the precomputed cos/sin table; re-exports `apply_rotary_pos_emb`, `rotate_half`, `interleave_mrope` from `qwenfast.model` |
| `flashinfer_attn.py` | `FlashInferDecodeAttention`, `FlashInferPrefillAttention`, the fused qk-norm+rope and output-gate ops, torch and flash-attention-3 fallbacks |
| `fused_qk_rope.py` | `qk_norm_rope`, one triton launch for qk-rmsnorm plus partial rope, used by the prefill and mixed paths |
| `bench_attn.py` | gpu decode/prefill microbenchmark, json out |
| `tests/test_attn.py`, `tests/test_fused_qk_rope.py` | cpu tests, plus gpu tests that skip without cuda |

## layout

```
kv     [n_layers=17, n_pages, 2, page_size, num_kv_heads=4, head_dim=256]   bf16 | float8_e4m3fn
scale  [n_layers, n_pages, 2, num_kv_heads]                                  fp32, fp8 only
page_table [max_seqs, max_pages_per_seq]                                     int32, -1 = unmapped
seq_len    [max_seqs]                                                        int32, committed tokens
```

this is flashinfer's `"NHD"` per-page layout with a leading layer axis, so `wrapper.run()` consumes `pool.kv[layer]` with no transposition. `page_size` is a `KVPoolConfig` field (16 by default). the kv pool and the ssm slot pool are independent allocators with independent granularities; the runtime uses one slot id to index both.

## public api

`PagedKVPool(KVPoolConfig(...))`:

| method | role |
|---|---|
| `alloc_slot()` / `free_slot(slot)` | claim or release a per-sequence page-table row |
| `ensure_capacity(slot, num_tokens)` | host-side page allocation; runs outside any graph |
| `append_kv(layer, slot_ids, positions, k, v)` | pure scatter into pages; graph-safe once pages exist |
| `truncate(slot, new_len)` | o(1) rollback: moves `seq_len`, touches no data |
| `reclaim_trailing_pages(slot)` | frees pages past `seq_len`; host-side, optional |
| `build_flashinfer_indices(slot_ids, seq_lens=None, staged=False)` | the `(kv_indptr, kv_indices, kv_last_page_len, seq_lens)` tuple `plan` wants, from the host page-table mirror |
| `defer_page_table_writes()` / `flush_page_table()` | batch all page-table updates of one admission pass into one device copy |
| `calibrate_page_scale(...)` / `calibrate_uniform_scale(...)` | set fp8 scales from a sample before steady-state writes |
| `gather_dense(layer, slot)` | dense k/v for a slot; the reference path and the tests |
| `pages_needed(n)`, `pages_allocated(slot)`, `num_free_pages()`, `nbytes()` | accounting |

`flashinfer_attn`:

| symbol | role |
|---|---|
| `FlashInferDecodeAttention(workspace, num_qo_heads, num_kv_heads, head_dim, page_size, max_batch_size, max_pages, kv_dtype, use_cuda_graph=True, use_tensor_cores=True)` | one instance per graph bucket; `plan(kv_indptr, kv_indices, kv_last_page_len)` outside the graph, `run(q, pool.kv[layer])` inside |
| `FlashInferPrefillAttention(...)` | varlen causal prefill into the paged pool; with `qo_indptr = decode_qo_indptr(B)` it also serves as a decode backend |
| `fused_qk_norm_rope(q, k, q_norm_w, k_norm_w, cos, sin)` | per-head qk-rmsnorm with `(1 + weight)`, then rope on the first 64 dims |
| `split_q_gate(qg, num_heads, head_dim)`, `apply_output_gate(out, gate)` | split the fused q/gate projection; sigmoid gate before `o_proj` |
| `rms_norm_head`, `torch_fallback_decode`, `torch_fallback_prefill` | reference ops, cpu-runnable |
| `fa3_varlen_prefill`, `fa3_decode_with_kvcache` | flash-attention-3 paths when `vllm_flash_attn` or `flash_attn_interface` is importable |
| `pad_indptr_to_bucket(indptr, bucket)` | pad a csr indptr to a bucket while keeping it monotonic |
| `DECODE_BACKENDS` | `("flashinfer_decode_tc", "flashinfer_prefill_as_decode", "fa3_kvcache", "torch")` |

`rope.RotaryTable(device=...)` builds a `[262144, 64]` bf16 cos/sin table once; `lookup(positions)` is a gather, graph-safe. `fused_qk_rope.qk_norm_rope(..., use_fused=True)` runs the triton kernel; the default is a pass-through to the eager op, which is what the graphed decode path uses.

```python
import torch
from qwenfast.attn import KVPoolConfig, PagedKVPool, RotaryTable
from qwenfast.attn.flashinfer_attn import FlashInferDecodeAttention, fused_qk_norm_rope, apply_output_gate

pool = PagedKVPool(KVPoolConfig(n_pages=200_000, max_seqs=512, max_pages_per_seq=1024,
                                dtype="bf16", device="cuda:0"))
table = RotaryTable(device="cuda:0")
slot = pool.alloc_slot()
pool.ensure_capacity(slot, num_tokens=1)                 # host side
cos, sin = table.lookup(positions)
q, k = fused_qk_norm_rope(q_raw, k_raw, q_norm_w, k_norm_w, cos, sin)
pool.append_kv(layer, slot_ids, positions, k, v)         # graph-safe

workspace = torch.empty(128 << 20, dtype=torch.uint8, device="cuda:0")
decode = FlashInferDecodeAttention(workspace, num_qo_heads=24, num_kv_heads=4, head_dim=256,
                                   page_size=16, max_batch_size=512, max_pages=200_000,
                                   kv_dtype=pool.storage_dtype, device="cuda:0")
kv_indptr, kv_indices, kv_last_page_len, _ = pool.build_flashinfer_indices(live_slots)
decode.plan(kv_indptr, kv_indices, kv_last_page_len)     # outside the graph
out = decode.run(q, pool.kv[layer])                       # inside the graph
gated = apply_output_gate(out.reshape(B, -1), gate)
```

## design notes

- plan/run split. flashinfer's `plan` does host-driven dispatch and cannot be captured; `run` is a kernel launch. the wrappers hold persistent `indptr`/`indices`/`last_page_len` buffers sized for their bucket, copy the live (padded) arrays into them in `plan`, and replay `run`.
- tensor cores are required. at 24 query heads over 4 kv heads (group size 6) with head dim 256, flashinfer's non-tensor-core decode kernel is unsupported, so `FlashInferDecodeAttention` defaults to `use_tensor_cores=True`. `flashinfer_prefill_as_decode` and `fa3_kvcache` are the alternative decode paths, and `bench_attn.py` measures all three.
- the page table is host-mirrored. allocation is pure host bookkeeping, so `build_flashinfer_indices` never reads the device table, `pages_allocated` is an o(1) counter, and page-table writes are staged into pinned buffers and flushed once per admission pass. `build_flashinfer_indices(staged=True)` returns host-side `indptr`/`last_page_len` from a pinned ring, which is what `plan` wants anyway.
- rollback is a pointer move. the speculative step appends k/v for the whole `k+1` window, then `truncate(slot, committed + m)`; the next append overwrites the invisible tail. no requantization, no data movement.
- fp8 scales are not recomputed on append: rescaling a page that already holds tokens is neither o(1) nor graph-safe. calibrate a page before steady-state writes, or use `calibrate_uniform_scale` for one scale per layer, which is what the flashinfer `k_scale`/`v_scale` arguments take.
- numerics follow `qwenfast.model.Attention`: qk-rmsnorm with `(1 + weight)` on the full head dim before rope, rope only on the first 64 of 256 dims, sigmoid output gate before `o_proj`, no extra float casts.
- `QWENFAST_FI_HOST_PLAN=0` disables the host-side decode plan fast path and restores the device round trip.

## tests

```bash
python engine/qwenfast/attn/tests/test_attn.py
python engine/qwenfast/attn/tests/test_fused_qk_rope.py
# or
python -m unittest discover -s engine/qwenfast/attn/tests -v
```

cpu coverage: the page allocator, bf16 and fp8 pool round trips, `truncate` and later overwrite, `reclaim_trailing_pages`, rope and `RotaryTable` against a from-scratch reference and `model.RotaryEmbedding`, qk-rmsnorm and its `(1 + weight)` convention, and the full pipeline (project, fused qk-norm+rope, paged append, torch attention, gate, `o_proj`) against `model.py`'s dense `Attention` for a varlen prefill batch and for a decode step after a prefill. `TestFlashInferGPU`, `TestFA3GPU` and `TestTritonKernelGPU` run the real kernels against the torch fallbacks on a gpu and skip elsewhere.

## benchmark

```bash
PYTHONPATH=engine python -m qwenfast.attn.bench_attn --quick --out attn_quick.json
PYTHONPATH=engine python -m qwenfast.attn.bench_attn --out attn_bench.json \
  --batches 1,8,32,64,128,256,512 --contexts 2048,8192 --kv-dtypes bf16,fp8 \
  --prefill-tokens 512,2048,8192 --page-size 16 \
  --backends flashinfer_decode_tc,flashinfer_prefill_as_decode,fa3_kvcache
```

decode cells report microseconds per step, achieved gb/s and the extrapolated ms for 16 layers; prefill cells report tokens per second with kv append timed separately. every cell is exception-safe and bounded by `--cell-timeout-s` (60 s by default), so a missing backend records an error and a hang records a timeout. a jit warm-up phase runs first and is reported apart from the timed cells. `--quick` samples batches 1, 32, 128 and 512 at one context. add `torch` to `--backends` for the reference path.
