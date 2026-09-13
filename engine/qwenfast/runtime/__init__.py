"""``qwenfast.runtime``: the model runtime.

Owns everything between "weights are in HBM" and "tokens come out":

======================  ====================================================
``fused_model``         the model: fused weights + GDN kernels + paged KV /
                        FlashInfer, decode and varlen chunked prefill over
                        the device-buffer ABI
``graphs``              CUDA-graph capture per batch-size bucket, shared
                        mempool, graph-safe sampler, ``--no-graphs`` path
``scheduler``           continuous batching: slots, pages, chunked
                        prefill budget, prefill/decode interleave, preemption
``spec_decode``         MTP speculative decoding: k-step draft loop,
                        windowed verify + device-side acceptance, the GDN
                        verify-and-commit kernel, its own per-bucket graphs
``engine``              the ``AsyncEngine`` contract implementation
``serve``               ``python -m qwenfast.runtime.serve``: the OpenAI
                        server wired to this engine
``bench_runtime``       offline decode/prefill benchmark
``bench_spec``          spec-decode correctness gate + timed sweep
======================  ====================================================

Nothing here is imported at package-import time beyond the light
``RuntimeConfig`` dataclass, so ``import qwenfast.runtime`` works on a laptop
with no CUDA (and, apart from ``torch`` itself, no optional dependency at
all).  Every GPU-only import (``flashinfer``, ``triton``, ``fla``, ``vllm``)
is guarded at its point of use.
"""

from __future__ import annotations

__all__ = [
    "RuntimeConfig",
    "SpecConfig",
    "GRAPH_BUCKETS",
    "build_engine",
    "build_async_engine",
]

GRAPH_BUCKETS = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)


def __getattr__(name):  # lazy re-exports; keeps `import qwenfast.runtime` torch-free-ish
    if name == "RuntimeConfig":
        from .fused_model import RuntimeConfig

        return RuntimeConfig
    if name == "SpecConfig":
        from .spec_decode import SpecConfig

        return SpecConfig
    if name == "build_engine":
        from .engine import build_engine

        return build_engine
    if name == "build_async_engine":
        from .engine import build_async_engine

        return build_async_engine
    raise AttributeError(name)
