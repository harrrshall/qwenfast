"""qwenfast — a custom inference engine for Qwen/Qwen3.8-27B.

The top-level modules (``model``, ``weights``) are a minimal pure-PyTorch
reference implementation used as the correctness oracle for the optimised
runtime.  See ``docs/architecture.md``.
"""

from .weights import (
    QwenFastConfig,
    SafetensorsStore,
    dequant_block128,
    load_into_model,
    resolve_snapshot,
)
from .model import (
    HAS_FLA,
    Attention,
    DecoderLayer,
    GatedDeltaNet,
    Generator,
    HybridCache,
    MLP,
    MTPHead,
    QwenFastForCausalLM,
    QwenFastModel,
    RMSNorm,
    RMSNormGated,
    RotaryEmbedding,
    apply_rotary_pos_emb,
    causal_conv1d,
    gated_delta_rule,
    interleave_mrope,
    l2norm,
    rotate_half,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)

__version__ = "0.0.1-m0"

__all__ = [
    "__version__",
    "HAS_FLA",
    "QwenFastConfig",
    "SafetensorsStore",
    "dequant_block128",
    "load_into_model",
    "resolve_snapshot",
    "QwenFastForCausalLM",
    "QwenFastModel",
    "Generator",
    "HybridCache",
    "MTPHead",
    "GatedDeltaNet",
    "Attention",
    "MLP",
    "DecoderLayer",
    "RMSNorm",
    "RMSNormGated",
    "RotaryEmbedding",
    "apply_rotary_pos_emb",
    "rotate_half",
    "interleave_mrope",
    "l2norm",
    "causal_conv1d",
    "gated_delta_rule",
    "torch_chunk_gated_delta_rule",
    "torch_recurrent_gated_delta_rule",
]
