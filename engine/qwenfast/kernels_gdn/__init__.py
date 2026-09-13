"""``qwenfast.kernels_gdn``: Gated-DeltaNet kernels for Qwen3.8-27B.

Owns the 48 ``linear_attention`` layers' token mixer: chunked varlen prefill,
recurrent decode, the depthwise causal conv, and the speculative
verify-and-commit path.

Three interchangeable backends behind one frozen API:

``torch``   pure-PyTorch reference, CPU-runnable, the CI oracle
``fla``     ``flash-linear-attention`` >= 0.5.2 Triton kernels
``triton``  hand-written kernels for *this* shape: one program per
            (slot, v-head, dv-block), the ``[128, 128]`` state loaded and
            stored exactly once, slot-indexed straight out of the pool

Quick start::

    from qwenfast.kernels_gdn import api, state

    pool = state.alloc_state_pool(n_slots=512, dtype="fp16", device="cuda")
    lyr  = state.layer_state(pool, layer_idx=0)      # [512, 48, 128, 128] view
    out  = api.gdn_decode_step(q, k, v, g, beta, lyr, slot_ids)

This subpackage never imports ``qwenfast.model``: the reference model is only pulled
in by the tests, so that the two implementations stay genuinely independent.
"""

from __future__ import annotations

from . import shapes, state
from .api import (
    BACKENDS,
    available_backends,
    causal_conv_prefill,
    causal_conv_prefill_varlen,
    causal_conv_update,
    causal_conv_verify_and_commit,
    gdn_commit,
    gdn_decode_multi,
    gdn_decode_step,
    gdn_prefill_chunked,
    gdn_verify,
    gdn_verify_and_commit,
    get_default_backend,
    resolve_backend,
    set_default_backend,
)

__all__ = [
    "shapes",
    "state",
    "BACKENDS",
    "available_backends",
    "set_default_backend",
    "get_default_backend",
    "resolve_backend",
    "gdn_decode_step",
    "gdn_decode_multi",
    "gdn_prefill_chunked",
    "causal_conv_update",
    "causal_conv_prefill",
    "causal_conv_prefill_varlen",
    "causal_conv_verify_and_commit",
    "gdn_verify",
    "gdn_commit",
    "gdn_verify_and_commit",
]
