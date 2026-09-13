"""qwenfast.gemm: GEMM and quantization.

Fused weight builder, FP8/bf16 GEMM dispatch table, and the M-bucketed
autotune cache. See ``docs/architecture.md`` and this package's ``README.md``.

Import-safe with no torch/CUDA available: nothing here touches torch at
module scope.
"""

from .fused_weights import (
    FP8Tensor,
    AttnFusedWeights,
    FusedModelWeights,
    GDNFusedWeights,
    MLPFusedWeights,
    MTPFusedWeights,
    build_fused_weights,
    fuse_bf16_rows,
    fuse_fp8_rows,
    load_fused,
    quantize_bf16_to_fp8_block128,
    save_fused,
)
from .dispatch import linear, available_backends, m_bucket, M_BUCKETS

__all__ = [
    "FP8Tensor",
    "AttnFusedWeights",
    "FusedModelWeights",
    "GDNFusedWeights",
    "MLPFusedWeights",
    "MTPFusedWeights",
    "build_fused_weights",
    "fuse_bf16_rows",
    "fuse_fp8_rows",
    "load_fused",
    "quantize_bf16_to_fp8_block128",
    "save_fused",
    "linear",
    "available_backends",
    "m_bucket",
    "M_BUCKETS",
]
