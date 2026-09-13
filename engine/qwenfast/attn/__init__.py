"""qwenfast attention / paged-KV subpackage.

See ``docs/architecture.md`` for the overall design, and this package's
``README.md`` for remote (GPU) usage.

Modules:

* ``kv_pool`` -- paged KV pool (bf16/fp8), page allocator, per-sequence page
  table, ``append_kv``/``truncate`` (rollback).
* ``rope`` -- precomputed cos/sin table + the reference ``apply_rotary_pos_emb``
  (re-exported from ``qwenfast.model``, which is verified bit-identical to
  ``engine/reference/modeling_qwen3_5.py``).
* ``flashinfer_attn`` -- FlashInfer decode/prefill wrappers (CUDA-graph-aware),
  fused QK-RMSNorm+RoPE / output-gate ops, and the torch/FA3 fallback backends.

Importing this package never requires CUDA or flashinfer -- every GPU-only
dependency is guarded at the point of use (see ``flashinfer_attn.HAS_*`` flags).
"""

from .kv_pool import KVPoolConfig, PageAllocator, PagedKVPool
from .rope import RotaryTable, apply_rotary_pos_emb, interleave_mrope, rotate_half

__all__ = [
    "KVPoolConfig",
    "PageAllocator",
    "PagedKVPool",
    "RotaryTable",
    "apply_rotary_pos_emb",
    "rotate_half",
    "interleave_mrope",
]
