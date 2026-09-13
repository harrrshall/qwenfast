"""``python -m qwenfast.runtime.serve`` -- the OpenAI server wired to the real
``qwenfast`` runtime.

    python -m qwenfast.runtime.serve --model /path/to/Qwen3.8-27B-FP8 --port 8000

This module owns three things and nothing else:

1. **The default runtime configuration**, which is the best-known measured
   config: CUDA graphs on, ``fp16`` SSM
   state, FlashInfer attention, Triton norms, per-M-bucket GEMM dispatch.
   Every one of those differs from ``RuntimeConfig``'s own dataclass default,
   which is deliberately the conservative/reference value; the *serving*
   defaults are the measured-fastest ones. See :data:`M1_DEFAULTS`.
2. **Pool geometry derived from ``--max-model-len`` and ``--max-num-seqs``**
   instead of ``RuntimeConfig``'s flat ``n_kv_pages=8192`` (which is 131,072
   tokens total -- less than one 64-way ctx-2048 batch). Uses
   ``bench_runtime.derive_pool_sizes`` so the server and the offline bench
   size their pools by the same rule. See :func:`plan_memory` for the byte
   accounting this implies and :func:`main` for the startup report.
3. **The wiring**, which is exactly ``server/cli.py``'s ``--engine mock``
   path with a different engine object: ``runtime.engine.build_async_engine``
   in place of ``MockEngine``, into the same ``server/app.py::create_app``.

``server/cli.py --engine qwenfast`` now delegates here (:func:`add_runtime_args`
/ :func:`build_engine_from_args` are the two entry points it imports), so the
two commands are the same program with the same flags.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, Optional, Tuple

from .bench_runtime import derive_pool_sizes
from .engine import build_async_engine
from .fused_model import RuntimeConfig
from .preset import add_preset_arg, apply_preset
from .spec_decode import SpecConfig

# --------------------------------------------------------------------------- #
# 1. defaults
# --------------------------------------------------------------------------- #
#: The best-known serving configuration (the measured final decode numbers),
#: which is *not* the same as ``RuntimeConfig``'s dataclass defaults:
#:
#: * ``ssm_state_dtype="fp16"`` -- a wash at B<=8, 5-17% faster from B=32 up,
#:   and it halves the 144 MiB/slot state pool, which is the capacity lever.
#:   Measured state drift after 2048 steps: 2.5e-4 relative.
#: * ``norm_backend="triton"`` -- B=1 15.375 -> 12.638 ms (-17.8%), B=32
#:   22.629 -> 18.407 ms (-18.7%), launches/step 3,472 -> 1,945.
#: * ``attn_backend="auto"`` -> flashinfer, which CUDA-graph capture *requires*
#:   (``graphs.GraphedDecoder.capture`` refuses anything else).
#: * ``kv_cache_dtype="bf16"`` -- fp8 KV is wired (k/v scales are forwarded
#:   to FlashInfer) but not yet validated end to end. bf16 is what the
#:   reference decode numbers were measured with, and ``--kv-cache-dtype fp8``
#:   switches it.
#: * ``gemm_backend=None`` -- do NOT pin one. dispatch resolves per M-bucket
#:   (marlin for M<=32, flashinfer_fp8_blockscale for 64-256, scaled_mm at
#:   512); pinning bucket 1's winner costs 5x at M=256.
M1_DEFAULTS: Dict[str, Any] = {
    "dtype": "bf16",
    "ssm_state_dtype": "fp16",
    "kv_cache_dtype": "bf16",
    "page_size": 16,
    "gdn_backend": "auto",
    "gemm_backend": None,
    "attn_backend": "auto",
    "norm_backend": "triton",
    #: The fused GDN-gate/SwiGLU Triton kernels behind
    #: `RuntimeConfig.fused_ops_backend` are the faster path (B=1 12.638 ->
    #: 12.352 ms, B=128 28.677 -> 27.708 ms, launches/step 1,945 -> 1,689).
    #: Without this key the server would silently run `RuntimeConfig`'s
    #: conservative `"torch"` default, a measured serving regression.
    "fused_ops_backend": "triton",
    #: gemm_v7 unchanged at M<=512, plus
    #: the four prefill-scale buckets (1024/2048/4096/8192) v7 silently
    #: clamped to the M=512 answer.
    "gemm_priority": "v8",
    "use_cuda_graphs": True,
    "sampler_candidates": 2048,
    "attn_workspace_mb": 512,
    #: "multi" (the dataclass default, kept for the offline
    #: microbenchmarks) lets two GEMM backends each memoise a full repacked copy
    #: of all 23 GiB of FP8 linears. Serving pins "single".
    "gemm_weight_cache": "single",
    #: "fast" == the historical speed-only ranking, kept as the serving
    #: default until "strict" (W8A16 only, relL2 <= 5e-3) is priced end to
    #: end. Note what "fast" means for a SERVER
    #: specifically: every prefill chunk resolves at M-bucket 512 and every
    #: decode batch of 64+ at M >= 64, so a server spends most of its GEMM
    #: time in the 2.6e-2 class while its small-batch decode steps run at
    #: 2.7e-3, a split whose quality impact has not been evaluated.
    "gemm_accuracy": "fast",
    #: Bounds the fp32 working set of the torch conv-prefill fallback
    #: independently of --max-num-batched-tokens.
    "conv_prefill_tile_tokens": 2048,
    #: The Triton token-major prefill conv. Made the serving default because
    #: the path it replaces is the largest non-GEMM term in a prefill chunk
    #: and prefill is ~77% of wall clock at conc 256; `channel_major` is the
    #: one-flag rollback.
    "conv_prefill_layout": "token_major",
    #: None == m_bucket dispatch (which clamps at 512 and
    #: has never been measured at a prefill M). Only set this to a backend
    #: that owns no repack cache unless --gemm-weight-cache is `multi`.
    "prefill_gemm_backend": None,
    #: 0 == chunk at --max-num-batched-tokens. A TPOT knob, not a throughput
    #: knob -- see the RuntimeConfig field's docstring. Under `--mixed-forward`
    #: it is *the* p99 knob.
    "prefill_chunk_tokens": 0,
    #: One forward per step over [prefill chunk ‖ decode rows] instead of
    #: alternating the two, so the decode rows ride the chunk's GEMMs.
    #: False == "a step is never mixed". The projected win is at concurrency >= 64 only; at conc 1-8 a mixed step is
    #: strictly worse than the graphed decode step it replaces, which is why
    #: this is a flag rather than a rewrite.
    "mixed_forward": False,
    #: CUDA-graph the mixed step: pad it to a fixed
    #: (prefill_chunk_tokens, decode bucket) shape and replay it as ~49 graph
    #: segments with one eager hole per GDN layer, instead of ~4,000 eager
    #: launches. Measured motivation: at conc 1024 an eager mixed step is
    #: 153 ms where its own GEMM+state work is ~91, and the ~60 ms difference
    #: is host launch latency. Requires --mixed-forward; off by default.
    "mixed_graphs": False,
    #: Prefill plan rows per graphed mixed step (>= 2). At most
    #: `n - 1` real segments; the chunk budget is `prefill_chunk_tokens - n`.
    "mixed_graph_segments": 8,
    "mixed_graph_holes": False,
    "mixed_graph_min_bucket": 32,
    #: Run the prefill chunk and the decode step of one step
    #: **concurrently on two CUDA streams**, each its own CUDA graph, instead
    #: of fusing them into one row-concatenated forward. Measured (conc 256,
    #: real 2,139-token context): the second stream is worth 1-9 % of
    #: a decode step, and *not fusing* is worth 7 % at chunk 8,192 and 23 % at
    #: chunk 1,024 -- the fused mixed step runs its decode rows through
    #: FlashInfer's paged **prefill** kernel, which at 2 k context is ~2x the
    #: paged decode kernel. Requires --mixed-forward --mixed-graphs; off by
    #: default until it is priced end to end.
    "overlap_streams": False,
    #: Below this fraction of the graph's fixed chunk, an
    #: overlapped step runs its prefill half eagerly instead of padding to the
    #: graph shape. 0.0 = always graphed (which spends most of a conc-32 step
    #: computing padding).
    "overlap_min_fill": 0.75,
    #: Asynchronous step scheduling: the host prepares and launches step N+1
    #: while the device runs step N, and reads step N's tokens back one step
    #: later. Off by default.
    "async_scheduling": False,
    #: `FusedMLP` splits a prefill chunk into tiles of
    #: this many tokens, so at the default an 8,192-token chunk runs the two
    #: largest GEMMs in the model as four M=2,048 tiles plus a `torch.cat` of
    #: the four outputs, x 64 layers -- i.e. the chunk's M never reaches the
    #: bucket-8192 row of the priority table. `plan_memory`'s prefill term
    #: already budgets the *untiled* `[T, 2I]` intermediate, so raising this to
    #: the chunk size spends memory the plan has already reserved. Kept at
    #: 2048 (the historical value) until the change is priced.
    "mlp_tile_tokens": 2048,
    #: "decode" == the historical behaviour (marlin claims
    #: the single repack-cache slot at warmup bucket 1 and every larger M is
    #: thereafter denied its own winner). "prefill" hands the slot to the
    #: large-M winner instead. Still exactly one cache either way, so the
    #: memory plan is byte-identical; defaulted off pending measurement.
    "gemm_cache_owner": "decode",
    #: fla's GDN chunk-kernel `BT`. 32, not the library's own 64, on a
    #: real-model A/B (`64=565ms 32=550ms 16=559ms`, ~2.7% faster) plus a
    #: CPU parity test that
    #: the torch-backend math is chunk-size-invariant on the tiny model
    #: (`test_serving_path.py::TestGdnChunkSizeParity`) -- fla itself is
    #: GPU-only, so that CPU test is what "safe to flip" means here; the real
    #: model's fla-32-vs-64 logit parity is still to be confirmed.
    "gdn_chunk_size": 32,
}
#: Backends that memoise a permanent repacked copy of every fp8 linear
#: (~23 GiB on this checkpoint). Forcing one of these for prefill while a
#: *different* one is already cached runs the server out of memory, so
#: `build_runtime_config` refuses the combination rather than letting the
#: first prefill chunk discover it. Mirrors
#: `gemm.dispatch._CACHE_ATTR_BY_BACKEND`.
REPACK_CACHE_BACKENDS = (
    "vllm_marlin_fp8_w8a16",
    "scaled_mm_pertensor",
    "vllm_cutlass_fp8_pertensor",
    "deepgemm",
    "machete_w8a16",
)

#: Serving-shape defaults, sized for the serving sweep (2000-in / 500-out,
#: concurrency up to 256). ``max_model_len`` must cover prompt + completion
#: with slack, because it bounds *both* the rotary table
#: (``FusedQwenForCausalLM.max_context_len`` -- an out-of-range position is a
#: device-side assert) and ``max_pages_per_seq``.
DEFAULT_MAX_NUM_SEQS = 256
#: **Not** 2000+500+slack: ``bench_serve.py --dataset
#: random --input-len 2000`` does not send 2000-token prompts. It draws 2000
#: random ids, *decodes* them to text, and the server re-encodes that text --
#: which round-trips to 2090-2195 tokens on Qwen3.8's 248k vocab (measured
#: over all 928 prompts of the seed-0 sweep). The vLLM baseline sees the same
#: prompts without issue only because it runs ``--max-model-len 65536``.
#: 2752 = 2195 (measured worst case) + 500 out + 57. Do **not** "fix" this by
#: shortening ``--input-len``: that would change the workload out from under
#: the vLLM comparison.
DEFAULT_MAX_MODEL_LEN = 2752
DEFAULT_MAX_NUM_BATCHED_TOKENS = 8192  # matches the vLLM baseline

# Qwen3.8-27B shape constants (from the checkpoint's config.json), used for the
# *pre-load* memory estimate. Overridden from the checkpoint's own
# ``config.json`` when it is readable, so the plan is right for whatever model
# is actually being served (and for the tiny CPU test model).
_ARCH = {
    "n_attn_layers": 16,
    "n_gdn_layers": 48,
    "num_kv_heads": 4,
    "head_dim": 256,
    "gdn_v_heads": 48,
    "gdn_head_k": 128,
    "gdn_head_v": 128,
    "conv_dim": 10240,
    "conv_width": 4,
    # -- the weight and
    #    prefill-activation terms need the dense shapes too.
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "vocab_size": 248320,
    "num_attention_heads": 24,
    "gdn_z_dim": 6144,  # linear_num_value_heads * linear_value_head_dim
}
_ELEM_BYTES = {"fp8": 1, "bf16": 2, "fp16": 2, "fp32": 4}
_GIB = 1024 ** 3
_MIB = 1024 ** 2

# --------------------------------------------------------------------------- #
# Measured per-process allocations. Each is a real allocation that a single
# flat "graphs/workspace/logits" slack line would hide.
# --------------------------------------------------------------------------- #
#: FlashInfer's own per-wrapper int workspace (`_int_workspace_buffer`),
#: allocated once per *persistent* (use_cuda_graph=True) decode wrapper, i.e.
#: once per graph bucket, plus one for the prefill wrapper.
_FI_INT_WORKSPACE_BYTES = 8 * _MIB
#: The CUDA-graph private mempool, shared across all buckets by
#: ``graphs.GraphedDecoder.capture`` (one ``graph_pool_handle()``). Measured
#: 782 MiB over 13 buckets on the H200; budgeted at 1 GiB.
_GRAPH_POOL_BYTES = 1024 * _MIB
#: CUDA context + cuBLAS/cuDNN handles + the driver's own allocations: the
#: gap between ``torch.cuda.memory_allocated()`` and "this process has N in
#: use". Measured 0.88 GiB; budgeted at 1 GiB.
_CUDA_CONTEXT_BYTES = 1024 * _MIB
#: Marlin's permuted scales are fp16 over a 128-element group, i.e. 2 bytes
#: per 128 weight bytes on top of the repacked qweight (which is byte-for-byte
#: the size of the FP8 weight it replaces).
_MARLIN_SCALE_RATIO = 2.0 / 128.0
#: Allocator churn factor on the prefill working set: the terms below are the
#: live tensors at the peak, but the caching allocator also holds freed blocks
#: of the previous layer's shapes. 2x is what fits a measured OOM trace.
_PREFILL_CHURN = 2.0
#: The speculative verify window caches, per GDN layer and
#: per window token, the conv input + k + v + g + beta so the commit can be
#: rolled back -- ~36 KiB/token/layer. It is allocated inside the captured
#: graph, so it is *resident* for the life of the SpecDecoder, once per
#: captured bucket shape (the pool reuses blocks between captures, so the
#: largest bucket is what sizes it).
_SPEC_GDN_CACHE_BYTES_PER_TOKEN_LAYER = 36.0 * 1024.0


# --------------------------------------------------------------------------- #
# 2. memory plan
# --------------------------------------------------------------------------- #
def weight_bytes_for(arch: Optional[Dict[str, int]] = None) -> Dict[str, float]:
    """FP8 checkpoint byte inventory, derived from the shapes (not a constant).

    Mirrors ``gemm.fused_weights``' fused layout exactly -- every entry here
    is one tensor that builder actually creates:

    ``mlp``  gate_up ``[2I, H]`` + down ``[H, I]`` fp8, every layer.
    ``attn`` qkv ``[(2 Hq + 2 Hkv) d, H]`` + o ``[H, Hq d]`` fp8, attention
             layers. Note the **2x on the Q rows**: Qwen3.8's ``q_proj`` is
             ``num_attention_heads * head_dim * 2`` wide because it emits the
             attention output gate alongside Q (verified against the
             checkpoint's ``[12288, 5120]``). Getting this wrong is a 0.5 GiB
             error on the 27B, and it is the shape behind 280 MiB fp32
             dequant allocations.
    ``gdn``  in_proj_qkvz ``[conv_dim + z, H]`` + out_proj ``[H, z]`` fp8,
             plus the bf16 ``in_proj_ba``/``conv1d`` tails.
    ``mtp``  one extra attn+mlp layer plus a bf16 ``fc`` ``[H, 2H]``.
    ``embed_lm_head`` two bf16 ``[V, H]`` matrices.

    ``linear_fp8`` is the subtotal the GEMM dispatcher can hang a repacked
    copy on.
    """
    a = dict(_ARCH)
    if arch:
        a.update(arch)
    h, i, v = a["hidden_size"], a["intermediate_size"], a["vocab_size"]
    n_attn, n_gdn = a["n_attn_layers"], a["n_gdn_layers"]
    n_layers = n_attn + n_gdn
    d, hq, hkv = a["head_dim"], a["num_attention_heads"], a["num_kv_heads"]
    z = a["gdn_z_dim"]

    def fp8(n: int, k: int) -> float:
        # weight bytes + the block-128 fp32 scale grid
        return float(n) * k + 4.0 * math.ceil(n / 128) * math.ceil(k / 128)

    mlp_one = fp8(2 * i, h) + fp8(h, i)
    attn_one = fp8((2 * hq + 2 * hkv) * d, h) + fp8(h, hq * d)
    gdn_one = fp8(a["conv_dim"] + z, h) + fp8(h, z)
    gdn_tail = 2.0 * (2 * a["gdn_v_heads"] * h + a["conv_dim"] * a["conv_width"])

    mlp = n_layers * mlp_one
    attn = n_attn * attn_one
    gdn = n_gdn * (gdn_one + gdn_tail)
    embed_lm_head = 2.0 * v * h * 2  # bf16 embed_tokens + bf16 lm_head
    # `from_pretrained(include_mtp=True)` always *builds* the MTP head, so its
    # bytes are resident whether or not `--enable-mtp` executes it.
    mtp = attn_one + mlp_one + 2.0 * h * 2 * h + 4.0 * h * 2
    norms = n_layers * 2.0 * h * 2

    linear_fp8 = mlp + attn + n_gdn * gdn_one + attn_one + mlp_one
    return {
        "mlp": mlp,
        "attn": attn,
        "gdn": gdn,
        "embed_lm_head": embed_lm_head,
        "mtp": mtp,
        "total": mlp + attn + gdn + embed_lm_head + mtp + norms,
        "linear_fp8": linear_fp8,
    }


def plan_memory(
    *,
    max_num_seqs: int,
    max_model_len: int,
    page_size: int,
    n_kv_pages: int,
    kv_cache_dtype: str,
    ssm_state_dtype: str,
    dtype: str = "bf16",
    enable_mtp: bool = False,
    has_mtp: bool = True,
    arch: Optional[Dict[str, int]] = None,
    max_pages_per_seq: Optional[int] = None,
    max_num_batched_tokens: int = DEFAULT_MAX_NUM_BATCHED_TOKENS,
    conv_prefill_tile_tokens: int = 2048,
    n_graph_buckets: int = 13,
    max_batch: Optional[int] = None,
    attn_workspace_mb: int = 512,
    gemm_weight_cache: str = "single",
    use_cuda_graphs: bool = True,
    weight_bytes: Optional[float] = None,
    linear_fp8_bytes: Optional[float] = None,
    spec_window: int = 0,
    spec_max_batch: Optional[int] = None,
    mixed_forward: bool = False,
    mixed_graphs: bool = False,
    mixed_graph_chunk: int = 0,
    mixed_graph_holes: bool = False,
    mixed_graph_segments: int = 8,
    gdn_chunk_size: int = 64,
    mixed_graph_buckets: int = 0,
) -> Dict[str, float]:
    """Byte accounting for one server configuration, in GiB.

    A plan that sums weights + KV + SSM + conv and adds a flat slack line
    for "graphs/workspace/logits" can be off by tens of GiB: the dominant
    hidden term is whole extra copies of the FP8 linear weights memoised by
    different GEMM backends (``gemm/dispatch.py``'s ``_marlin_cache`` at the
    decode buckets, ``_pertensor_cache`` at prefill's M-bucket 512).

    Every line below is arithmetic on a shape something actually allocates:

    * **weights** -- :func:`weight_bytes_for`, derived from the config.
    * **repack** -- ``k x linear_fp8`` where ``k`` is the number of permanent
      repacked weight copies the ``gemm_weight_cache`` policy permits: 0 for
      ``"none"``, 1 for ``"single"`` (the serving default: marlin at M<=32),
      2 for ``"multi"`` (marlin *and* per-tensor fp8 -- the historical,
      unbounded behaviour). Marlin's qweight is byte-for-byte the size of the
      FP8 weight it replaces, plus fp16 group-128 scales.

      ``machete_w8a16`` does **not** add a term here. Its
      cache is the same size class (int8 qweight, byte-for-byte the fp8
      weight, plus bf16 ``[K/128, N]`` group scales = 1.016x, within a
      rounding of ``_MARLIN_SCALE_RATIO``'s 1.016x) and it takes its own
      cache *slot* in ``gemm/dispatch.py``, so under ``"single"`` a weight
      holds marlin's copy **or** machete's, never both -- which is exactly
      why the new backend is a candidate to *replace* marlin rather than to
      be added alongside it. Under ``"multi"`` a run that resolves three
      different cache-holding backends would exceed this estimate; that is
      the pre-existing hazard ``"multi"`` already carries and the reason
      serving does not use it.
    * **KV pool** -- ``n_kv_pages x page_size x n_kv_layers x 2 x
      num_kv_heads x head_dim`` elements; ``n_kv_layers`` is 17 (16 attention
      + 1 MTP), so one token is ``17 x 4 x 256 x 2 = 34,816`` elements =
      **34 KiB fp8 / 68 KiB bf16**. Plus the fp8 scale grid, the
      ``[max_seqs, max_pages_per_seq]`` page table and ``seq_len``.
    * **SSM state** -- ``(max_num_seqs+1) x 48 x 48 x 128 x 128`` =
      **144 MiB/slot fp32, 72 MiB/slot fp16**; ``+1`` is the scratch slot.
    * **conv state** -- ``(max_num_seqs+1) x 48 x conv_dim x (W-1)`` act-dtype.
    * **workspace** -- FlashInfer's shared float workspace
      (``--attn-workspace-mb``) plus its *per-wrapper* 8 MiB int workspace and
      ``max_pages`` index buffer, one persistent wrapper **per graph bucket**.
    * **buffers** -- ``DeviceBuffers``: the ``[max_batch, vocab]`` fp32 logits
      buffer dominates (256 x 248,320 x 4 = 254 MB).
    * **graph pool** -- the single shared CUDA-graph mempool (measured 782 MiB
      across 13 buckets; budgeted 1 GiB).
    * **context** -- CUDA context + cuBLAS/cuDNN handles, the gap between
      ``memory_allocated()`` and the driver's "in use" (measured 0.88 GiB).
    * **spec** -- ``spec_window`` is the speculative window width
      ``n = k + 1`` (0 when no :class:`~.spec_decode.SpecDecoder` is built).
      A speculative step runs ``spec_max_batch * n`` token rows through the
      GDN layers and must cache
      ~36 KiB per token per layer to be able to roll the conv ring back;
      that cache is allocated inside the captured graph and is therefore
      resident, not transient. Plus the slot-indexed
      ``h_prev`` carry (sized off the full ``max_num_seqs``, not
      ``spec_max_batch``: any slot can carry a spec-eligible request).
      ``spec_max_batch`` defaults to ``max_batch`` when not given (the
      offline bench's usage: no cap, so the widest captured bucket sizes
      it). The server also builds this term when ``--spec-k > 0``, sized off
      ``--spec-max-batch`` -- the *server's* spec graphs are only captured up
      to that batch (see ``_spec_buckets_for``), which is smaller than the
      full serving ``max_batch`` and keeps this term from over-charging the
      plan for spec buckets nothing will ever run.
    * **prefill** -- the *bounded* transient working set of one
      ``max_num_batched_tokens`` prefill chunk. This is the term that must
      stay resident-free: everything in it is per-chunk, not per-request.

      Under ``mixed_forward`` a step's activations are
      ``M = max_num_batched_tokens + max_batch`` rows wide, not
      ``max_num_batched_tokens`` -- the decode rows are appended to the chunk
      and every GEMM/norm/MLP intermediate grows with them. Measured at +2.6%
      of this term at the serving geometry (8,192 + 256) and +10.6% at chunk
      2,048 / conc 512 (``tests/test_mixed_forward.py::TestMixedMemoryPlan``).
      Neither equals ``max_batch / max_num_batched_tokens``, because the term
      is affine in the token count, not proportional: the fp32 conv tile and
      the per-chunk logits rows do not scale with it. Small either way, and
      here anyway: a term that is *silently* light is how a plan drifts tens
      of GiB from reality, and this one grows with ``max_batch`` while the
      chunk cap does not.
      The per-chunk logits row is unchanged: a mixed step runs ``lm_head`` on
      ``N_pre + B_dec`` rows, and every one of those holds a distinct slot, so
      the count is still bounded by ``max_num_seqs``.

    ``steady_gib`` is everything except ``prefill_gib`` -- i.e. what
    :func:`measure_allocation` should see right after graph capture, and what
    the post-capture gate compares against. ``total_gib`` adds the prefill
    headroom and is what the pre-flight budget check uses.
    """
    a = dict(_ARCH)
    if arch:
        a.update(arch)
    kv_elem = _ELEM_BYTES[kv_cache_dtype]
    state_elem = _ELEM_BYTES[ssm_state_dtype]
    act_elem = _ELEM_BYTES[dtype]
    if max_batch is None:
        max_batch = max_num_seqs

    # -- 1. weights + the repacked copies ---------------------------------- #
    winv = weight_bytes_for(a)
    w_bytes = float(winv["total"] if weight_bytes is None else weight_bytes)
    lin_bytes = float(winv["linear_fp8"] if linear_fp8_bytes is None else linear_fp8_bytes)
    n_caches = {"none": 0, "single": 1, "multi": 2}[gemm_weight_cache]
    repack_bytes = n_caches * lin_bytes * (1.0 + _MARLIN_SCALE_RATIO)
    if not has_mtp and weight_bytes is None:
        w_bytes -= float(winv["mtp"])

    # -- 2. KV pool --------------------------------------------------------- #
    # +1 MTP KV layer: `FusedQwenForCausalLM.__init__` sizes the pool as
    # `len(attention_layers) + (1 if fused.mtp is not None else 0)`, and the
    # 27B checkpoint always ships an MTP head, so `has_mtp` defaults True. The
    # tiny CPU test model has none, which is why this is a parameter and not a
    # constant `+ 1` -- getting it wrong is a 2x error on the KV pool.
    n_kv_layers = a["n_attn_layers"] + (1 if has_mtp else 0)
    # `_make_kv_pool` re-types a "bf16" pool to the *activation* dtype when the
    # two differ (`KVPoolConfig.dtype` only accepts bf16/fp8, so the fp32 CPU parity path gets its fp32 pool by re-allocating
    # `pool.kv`). A plan that ignored that would be 2x light on `--dtype fp32`.
    kv_store_elem = act_elem if (kv_cache_dtype == "bf16" and act_elem != 2) else kv_elem
    kv_bytes_per_token = n_kv_layers * a["num_kv_heads"] * a["head_dim"] * 2 * kv_store_elem
    kv_bytes = float(n_kv_pages) * page_size * kv_bytes_per_token
    if kv_cache_dtype == "fp8":  # per-page/per-head fp32 scales
        kv_bytes += n_kv_layers * n_kv_pages * 2 * a["num_kv_heads"] * 4
    rows = max_num_seqs + 1  # + scratch slot
    if max_pages_per_seq is None:
        max_pages_per_seq = math.ceil((max_model_len + 1) / page_size)
    kv_bytes += rows * max_pages_per_seq * 4 + rows * 4  # page_table + seq_len

    # -- 3. recurrent state -------------------------------------------------- #
    ssm_bytes_per_slot = (
        a["n_gdn_layers"] * a["gdn_v_heads"] * a["gdn_head_k"] * a["gdn_head_v"] * state_elem
    )
    conv_bytes_per_slot = a["n_gdn_layers"] * a["conv_dim"] * (a["conv_width"] - 1) * act_elem
    ssm_bytes = float(rows) * ssm_bytes_per_slot
    conv_bytes = float(rows) * conv_bytes_per_slot

    # -- 4. FlashInfer workspaces ------------------------------------------- #
    n_wrappers = (n_graph_buckets if use_cuda_graphs else 1) + 1  # + prefill wrapper
    workspace_bytes = float(attn_workspace_mb) * _MIB
    workspace_bytes += n_wrappers * (_FI_INT_WORKSPACE_BYTES + n_kv_pages * 4 + 8 * max_batch)

    # -- 5. DeviceBuffers + rotary ------------------------------------------ #
    buffer_bytes = float(max_batch) * a["vocab_size"] * 4  # fp32 logits
    buffer_bytes += max_batch * (4 * 8 + 4 * 5) + n_kv_pages * 4  # int32/fp32 [B] vectors
    buffer_bytes += 2.0 * max_model_len * a["head_dim"] * act_elem  # rotary cos/sin (over-counts)

    graph_bytes = float(_GRAPH_POOL_BYTES) if use_cuda_graphs else 0.0
    context_bytes = float(_CUDA_CONTEXT_BYTES)

    # -- 5b. the speculative window (0 when spec decoding is off) ------------ #
    spec_bytes = 0.0
    if spec_window and spec_window > 1:
        spec_batch = float(spec_max_batch if spec_max_batch is not None else max_batch)
        spec_bytes = (
            spec_batch * spec_window * a["n_gdn_layers"]
            * _SPEC_GDN_CACHE_BYTES_PER_TOKEN_LAYER
        )
        # `SpecDecoder.h_prev` [n_slots + 1, hidden] + the [Bmax, n] int32
        # window/pos/slot staging tensors.
        spec_bytes += rows * a["hidden_size"] * act_elem
        spec_bytes += spec_batch * spec_window * 4 * 4

    # -- 6. the bounded prefill working set --------------------------------- #
    # A mixed step's forward is `chunk tokens + one row per running
    # sequence` wide. `max_batch` is the widest that can be, and
    # is already what the plan uses for every other per-row term.
    t = float(max_num_batched_tokens) + (float(max_batch) if mixed_forward else 0.0)
    h, i = a["hidden_size"], a["intermediate_size"]
    tile = conv_prefill_tile_tokens or max_num_batched_tokens

    def _working_set(n_tokens: float) -> float:
        """The live activation set of one prefill-shaped forward, in bytes."""
        return (
            n_tokens * 2 * i * act_elem                             # gate_up output [T, 2I]
            + n_tokens * i * act_elem                               # SwiGLU output  [T, I]
            + 2 * n_tokens * (a["conv_dim"] + a["gdn_z_dim"]) * act_elem  # in_proj_qkvz in/out
            + n_tokens * (2 * a["num_attention_heads"] + 2 * a["num_kv_heads"])
            * a["head_dim"] * act_elem                              # attn qkv+gate [T, 14336]
            + 2 * a["conv_dim"] * (tile + a["conv_width"]) * 4      # conv fp32 tile in/out
            + n_tokens * a["gdn_z_dim"] * 4                         # GDN core fp32
            + n_tokens * h * act_elem * 2                           # residual stream + norm
            + float(max_num_seqs) * a["vocab_size"] * 4             # per-chunk prefill logits
        )

    prefill_bytes = _working_set(t) * _PREFILL_CHURN

    # -- 6b. the captured mixed step ------------------------------------------ #
    # A graphed mixed step does not *churn* its activations -- they are
    # allocated once, inside the graph's private mempool, and reused by every
    # replay. So this is the same working set as above at the graphed shape
    # (`chunk + max_batch` rows), without the churn factor, plus the two
    # process-lifetime allocations the runner adds: one shared GDN-prefill hole
    # buffer (`[chunk, HV, V]`, one for all 48 layers -- see
    # `MixedGraphRunner._break`) and one graph-mode FlashInfer prefill wrapper
    # per decode bucket.
    mixed_graph_bytes = 0.0
    if mixed_forward and mixed_graphs and use_cuda_graphs:
        c = float(mixed_graph_chunk or max_num_batched_tokens)
        mixed_graph_bytes = _working_set(c + float(max_batch))
        if mixed_graph_holes:
            # The shared GDN-prefill hole buffer, `[chunk, HV, V]`, one
            # for all 48 layers (`MixedGraphRunner._break`).
            mixed_graph_bytes += c * a["gdn_z_dim"] * act_elem
        else:
            # One-graph mode: no hole buffer -- the fla call is inside the
            # graph, so its output is an ordinary graph-pool activation already
            # counted by `_working_set`. What the one-graph mode *does* add is
            # fla's `h` scratch, `[1, NT, HV, K, V]` in the activation dtype,
            # where `NT` is the static chunk-index row count
            # (`fla_static.max_chunk_rows`). It is allocated and freed inside
            # each GDN layer's call, so the mempool holds **one** of them, not
            # 48 -- but it is a big one (36 MiB at chunk 1,024, 61 at 2,048)
            # and it exists only because of the capture, so it is named here
            # rather than left to `_working_set`'s slack.
            nt = float((int(c) - mixed_graph_segments) // max(gdn_chunk_size, 1)
                       + mixed_graph_segments)
            mixed_graph_bytes += (
                nt * a["gdn_v_heads"] * a["gdn_head_k"] * a["gdn_head_v"] * act_elem
            )
            # The four static segmentation tensors are int32 and tiny
            # (`n_segments+1` twice, `NT x 2`, `n_segments`): under 1 KiB
            # in total, below this plan's resolution. Not modelled.
        # A graphed mixed step is captured for the buckets at or above
        # `--mixed-graph-min-bucket`, not the whole decode ladder -- 4 of 14 at
        # conc 256 -- and each one costs a graph-mode FlashInfer prefill
        # wrapper. `0` means "the decode ladder", which is what a caller that
        # has not been told the floor should assume.
        mixed_graph_bytes += (mixed_graph_buckets or n_graph_buckets) * (
            _FI_INT_WORKSPACE_BYTES + n_kv_pages * 4 + 8 * max_batch
        )

    steady = (
        w_bytes + repack_bytes + kv_bytes + ssm_bytes + conv_bytes
        + workspace_bytes + buffer_bytes + graph_bytes + context_bytes
        + spec_bytes + mixed_graph_bytes
    )
    return {
        "spec_gib": spec_bytes / _GIB,
        "weights_gib": w_bytes / _GIB,
        "repack_gib": repack_bytes / _GIB,
        "kv_gib": kv_bytes / _GIB,
        "ssm_gib": ssm_bytes / _GIB,
        "conv_gib": conv_bytes / _GIB,
        "workspace_gib": workspace_bytes / _GIB,
        "buffers_gib": buffer_bytes / _GIB,
        "graph_gib": graph_bytes / _GIB,
        "mixed_graph_gib": mixed_graph_bytes / _GIB,
        "context_gib": context_bytes / _GIB,
        "prefill_gib": prefill_bytes / _GIB,
        "steady_gib": steady / _GIB,
        "total_gib": (steady + prefill_bytes) / _GIB,
        "linear_fp8_gib": lin_bytes / _GIB,
        "n_repack_caches": float(n_caches),
        "kv_kib_per_token": kv_bytes_per_token / 1024.0,
        "ssm_mib_per_slot": (ssm_bytes_per_slot + conv_bytes_per_slot) / (1024.0 ** 2),
        "kv_tokens_capacity": float(n_kv_pages * page_size),
        "kv_tokens_needed": float(max_num_seqs * max_model_len),
    }


def arch_from_checkpoint(model_dir: str) -> Optional[Dict[str, int]]:
    """Read the shape constants out of the checkpoint's ``config.json`` so the
    pre-load estimate is right for whatever model is actually being served
    (and silently falls back to the Qwen3.8-27B constants if anything is off)."""
    path = os.path.join(model_dir, "config.json")
    try:
        with open(path) as fh:
            cfg = json.load(fh)
    except Exception:
        return None
    # Qwen3.8-27B is a VL checkpoint: the text stack lives under `text_config`
    # Reading the top level only would silently fall through to the
    # hard-coded `_ARCH` every time.
    if isinstance(cfg.get("text_config"), dict):
        cfg = {**cfg, **cfg["text_config"]}
    try:
        layer_types = cfg.get("layer_types") or []
        n_attn = sum(1 for t in layer_types if t == "full_attention")
        n_gdn = sum(1 for t in layer_types if t == "linear_attention")
        if not n_attn or not n_gdn:
            return None
        return {
            "n_attn_layers": n_attn,
            "n_gdn_layers": n_gdn,
            "num_kv_heads": int(cfg["num_key_value_heads"]),
            "head_dim": int(cfg["head_dim"]),
            "gdn_v_heads": int(cfg["linear_num_value_heads"]),
            "gdn_head_k": int(cfg["linear_key_head_dim"]),
            "gdn_head_v": int(cfg["linear_value_head_dim"]),
            "conv_dim": int(
                cfg.get("conv_dim")
                or cfg["linear_num_key_heads"] * cfg["linear_key_head_dim"] * 2
                + cfg["linear_num_value_heads"] * cfg["linear_value_head_dim"]
            ),
            "conv_width": int(cfg["linear_conv_kernel_dim"]),
            # -- dense shapes, for the weight and prefill-activation terms --
            "hidden_size": int(cfg["hidden_size"]),
            "intermediate_size": int(cfg["intermediate_size"]),
            "vocab_size": int(cfg["vocab_size"]),
            "num_attention_heads": int(cfg["num_attention_heads"]),
            "gdn_z_dim": int(
                cfg["linear_num_value_heads"] * cfg["linear_value_head_dim"]
            ),
        }
    except Exception:
        return None


def format_memory_plan(plan: Dict[str, float], rt: RuntimeConfig, max_model_len: int) -> str:
    n = int(plan["n_repack_caches"])
    lines = [
        "[serve] memory plan (pre-load estimate, GiB)",
        f"[serve]   weights (fp8 + MTP)      {plan['weights_gib']:8.2f}",
        f"[serve]   gemm repack caches       {plan['repack_gib']:8.2f}"
        f"   ({n} x {plan['linear_fp8_gib']:.2f} GiB of fp8 linears,"
        f" --gemm-weight-cache {rt.gemm_weight_cache})",
        f"[serve]   KV pool  ({rt.kv_cache_dtype:>4})          {plan['kv_gib']:8.2f}"
        f"   ({rt.n_kv_pages} pages x {rt.page_size} tok"
        f" @ {plan['kv_kib_per_token']:.0f} KiB/token)",
        f"[serve]   SSM state ({rt.ssm_state_dtype:>4})        {plan['ssm_gib']:8.2f}"
        f"   ({rt.max_num_seqs}+1 slots @ {plan['ssm_mib_per_slot']:.1f} MiB/slot incl. conv)",
        f"[serve]   conv state ({rt.dtype:>4})       {plan['conv_gib']:8.2f}",
        f"[serve]   flashinfer workspaces    {plan['workspace_gib']:8.2f}",
        f"[serve]   device buffers/logits    {plan['buffers_gib']:8.2f}",
        f"[serve]   cuda-graph pool          {plan['graph_gib']:8.2f}",
        f"[serve]   cuda context/handles     {plan['context_gib']:8.2f}",
    ]
    if plan.get("spec_gib"):
        lines.append(
            f"[serve]   spec window cache        {plan['spec_gib']:8.2f}"
            f"   (36 KiB/token/layer)"
        )
    if plan.get("mixed_graph_gib"):
        lines.append(
            f"[serve]   mixed-step graphs        {plan['mixed_graph_gib']:8.2f}"
            f"   (chunk "
            f"{rt.prefill_chunk_tokens or rt.max_num_batched_tokens} + {rt.max_num_seqs} rows)"
        )
    lines += [
        f"[serve]   {'-' * 40}",
        f"[serve]   steady state             {plan['steady_gib']:8.2f}   (measured after capture)",
        f"[serve]   prefill headroom         {plan['prefill_gib']:8.2f}"
        f"   (bounded by --max-num-batched-tokens {rt.max_num_batched_tokens}"
        + (f" + {rt.max_num_seqs} mixed decode rows)" if rt.mixed_forward else ")"),
        f"[serve]   total                    {plan['total_gib']:8.2f}",
        f"[serve]   KV capacity {int(plan['kv_tokens_capacity']):,} tokens"
        f" vs {int(plan['kv_tokens_needed']):,} needed"
        f" ({rt.max_num_seqs} seqs x {max_model_len} ctx)",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 2b. measurement -- the other half of "matches within 5%"
# --------------------------------------------------------------------------- #
def measure_allocation(device: str) -> Optional[Dict[str, float]]:
    """What the device *actually* holds, in GiB. ``None`` off CUDA.

    ``in_use`` is ``total - free`` from ``cudaMemGetInfo``: it counts the CUDA
    context, cuBLAS handles and CUDA-graph private pools that
    ``memory_allocated()`` does not, and it is the number in the OOM message
    ("this process has 139.78 GiB in use") that the plan has to match."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        dev = torch.device(device)
        free, total = torch.cuda.mem_get_info(dev)
        return {
            "in_use_gib": (total - free) / _GIB,
            "allocated_gib": torch.cuda.memory_allocated(dev) / _GIB,
            "reserved_gib": torch.cuda.memory_reserved(dev) / _GIB,
            "free_gib": free / _GIB,
            "total_gib": total / _GIB,
        }
    except Exception:
        return None


def format_measured(measured: Dict[str, float], plan: Dict[str, float]) -> str:
    steady = plan["steady_gib"]
    err = (measured["in_use_gib"] - steady) / steady * 100.0 if steady else 0.0
    return (
        f"[serve] measured after capture: in-use {measured['in_use_gib']:.2f} GiB "
        f"(torch allocated {measured['allocated_gib']:.2f}, reserved "
        f"{measured['reserved_gib']:.2f}) vs plan steady {steady:.2f} GiB "
        f"-> {err:+.1f}%"
    )


# --------------------------------------------------------------------------- #
# 3. argument parsing
# --------------------------------------------------------------------------- #
def add_runtime_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the ``--engine qwenfast`` runtime knobs. Shared with
    ``server/cli.py`` so both entry points expose the same surface."""
    rt = p.add_argument_group("qwenfast runtime options (defaults = best measured serving config)")
    rt.add_argument("--device", default="cuda:0")
    rt.add_argument("--dtype", default=M1_DEFAULTS["dtype"], choices=["bf16", "fp16", "fp32"])
    rt.add_argument("--ssm-state-dtype", default=M1_DEFAULTS["ssm_state_dtype"], choices=["fp32", "fp16"])
    rt.add_argument("--kv-cache-dtype", default=M1_DEFAULTS["kv_cache_dtype"], choices=["bf16", "fp8"])
    rt.add_argument("--page-size", type=int, default=M1_DEFAULTS["page_size"])
    rt.add_argument("--max-num-seqs", type=int, default=DEFAULT_MAX_NUM_SEQS,
                    help="== n_ssm_slots; each slot costs ~75 MiB fp16 / ~147 MiB fp32")
    rt.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN,
                    help="prompt+completion cap; drives max_pages_per_seq and n_kv_pages")
    rt.add_argument("--n-kv-pages", type=int, default=None,
                    help="override the derived KV page count (default: max_num_seqs x ceil((ctx+1)/page_size))")
    rt.add_argument("--max-pages-per-seq", type=int, default=None, help="override the derived per-sequence page cap")
    rt.add_argument("--kv-pages-slack", type=int, default=64, help="spare pages above the derived requirement")
    rt.add_argument("--max-num-batched-tokens", type=int, default=DEFAULT_MAX_NUM_BATCHED_TOKENS,
                    help="chunked-prefill token budget per prefill step")
    rt.add_argument("--prefill-decode-ratio", type=int, default=4,
                    help="decode steps between prefill opportunities; higher = better TPOT, worse TTFT")
    rt.add_argument("--gdn-backend", default=M1_DEFAULTS["gdn_backend"], choices=["auto", "torch", "fla", "triton"])
    rt.add_argument("--gemm-backend", default=M1_DEFAULTS["gemm_backend"],
                    help="pin one GEMM backend; default None = per-M-bucket dispatch (recommended)")
    rt.add_argument("--gemm-weight-cache", default=M1_DEFAULTS["gemm_weight_cache"],
                    choices=["multi", "single", "none"],
                    help="permanent repacked weight copies per weight. 'multi' is the "
                         "unbounded behaviour and costs 2x the fp8 linears (46 GiB on 27B); 'single' "
                         "keeps marlin for decode and pushes prefill onto a cache-free backend; "
                         "'none' frees the marlin copy too (~23 GiB) for ~0.5 ms/step at M=1")
    rt.add_argument("--gemm-accuracy", default=M1_DEFAULTS["gemm_accuracy"],
                    choices=["fast", "strict"],
                    help="accuracy bar for GEMM backend selection. 'fast' ranks on "
                         "graph-timed speed alone -- which routes every prefill chunk and every "
                         "decode batch >=64 to an fp8-ACTIVATION kernel at relL2 2.6e-2 against "
                         "an exact fp32 reference, vs 2.7e-3 below that. 'strict' "
                         "admits only backends measured <= dispatch.STRICT_REL_L2_MAX (5e-3), "
                         "i.e. W8A16 only; costs ~2x GEMM time at M>=128 today")
    rt.add_argument("--conv-prefill-layout", default=M1_DEFAULTS["conv_prefill_layout"],
                    choices=["token_major", "channel_major"],
                    help="prefill depthwise-conv path. token_major = the "
                         "Triton varlen kernel over [T, C]; channel_major = the older "
                         "transpose + fp32 F.conv1d fallback, tiled at "
                         "--conv-prefill-tile-tokens")
    rt.add_argument("--prefill-gemm-backend", default=M1_DEFAULTS["prefill_gemm_backend"],
                    help="force one gemm backend for prefill-shaped GEMMs only. "
                         "A backend that owns a repack cache needs either "
                         "--gemm-weight-cache multi or --gemm-cache-owner prefill, else it "
                         "would allocate a *second* 23 GiB copy and run out of memory")
    rt.add_argument("--gemm-cache-owner", default=M1_DEFAULTS["gemm_cache_owner"],
                    choices=["decode", "prefill"],
                    help="which M-bucket claims the one repack-cache slot allowed by "
                         "--gemm-weight-cache single. 'decode' = marlin at "
                         "bucket 1, the historical behaviour; 'prefill' = the large-M "
                         "winner, which costs ~3-7%% at B=1 and buys it back at every "
                         "prefill chunk and every decode batch >= 32")
    rt.add_argument("--gdn-chunk-size", type=int, default=M1_DEFAULTS["gdn_chunk_size"],
                    choices=[16, 32, 64],
                    help="fla's GDN chunk-kernel BT. 32 (default) measured "
                         "2.7%% faster than fla's own 64 at the real 8,192-token prefill "
                         "shape; refused by fla builds that bake BT into the kernel's "
                         "autotune config (kernels_gdn.fla_ops.supports_chunk_size())")
    rt.add_argument("--prefill-chunk-tokens", type=int, default=M1_DEFAULTS["prefill_chunk_tokens"],
                    help="cap one prefill chunk below --max-num-batched-tokens (0 = off). "
                         "A TPOT knob, not a throughput knob")
    rt.add_argument("--mixed-forward", action="store_true",
                    default=M1_DEFAULTS["mixed_forward"],
                    help="run ONE forward per step over [prefill chunk || every running "
                         "decode row] instead of alternating a prefill step and a decode "
                         "step. The decode rows then ride the chunk's GEMMs: at "
                         "conc 256 the separate-step design is bounded at ~2,120 out tok/s "
                         "and the mixed one at ~2,650. Eager (no CUDA graph for that step); "
                         "pair with --prefill-chunk-tokens to bound TPOT p99.")
    rt.add_argument("--no-mixed-forward", dest="mixed_forward", action="store_false",
                    help="force the 'a step is never mixed' behaviour (the default)")
    rt.add_argument("--mixed-graphs", action="store_true",
                    default=M1_DEFAULTS["mixed_graphs"],
                    help="CUDA-graph the mixed step. Pads every mixed step to "
                         "(--prefill-chunk-tokens, decode bucket) and replays it as ~49 "
                         "captured segments with one eager hole per GDN layer (the fla chunk "
                         "kernel, whose index prep is host-built and cannot be captured). "
                         "Removes the ~60 ms/step of host launch latency measured "
                         "at chunk 1024. Requires --mixed-forward.")
    rt.add_argument("--no-mixed-graphs", dest="mixed_graphs", action="store_false",
                    help="run the mixed step eagerly (the default, and the one-flag rollback)")
    rt.add_argument("--mixed-graph-holes", action="store_true",
                    default=M1_DEFAULTS["mixed_graph_holes"],
                    help="rollback: capture the mixed step as ~49 graph segments "
                         "with one eager hole per GDN layer instead of ONE graph. "
                         "The default (--no-mixed-graph-holes) drives fla "
                         "with precomputed chunk_indices/chunk_offsets and pins the conv "
                         "grid, which leaves nothing in the step that reads the "
                         "segmentation on the host. Use this only to A/B the two, or on a "
                         "build whose fla cannot take the index tensors.")
    rt.add_argument("--no-mixed-graph-holes", dest="mixed_graph_holes",
                    action="store_false",
                    help="capture the mixed step as one graph (the default)")
    rt.add_argument("--mixed-graph-min-bucket", type=int,
                    default=M1_DEFAULTS["mixed_graph_min_bucket"],
                    help="smallest decode-row bucket a graphed mixed step is captured "
                         "for. Buckets below it are dropped and their "
                         "steps pad up, which costs padding rows and saves whole "
                         "graphs -- 4 captures instead of 14 at conc 256. 1 == keep "
                         "the whole ladder.")
    rt.add_argument("--overlap", dest="overlap_streams", action="store_true",
                    default=M1_DEFAULTS["overlap_streams"],
                    help="run a step's prefill chunk and its decode "
                         "rows as two CUDA graphs on two streams instead of one fused "
                         "forward. The prefill graph is captured for exactly one "
                         "(padding) decode row, the decode graph is the ordinary one, "
                         "and they get separate graph mempools and separate FlashInfer "
                         "workspaces because they are in flight at the same time. "
                         "Requires --mixed-forward --mixed-graphs.")
    rt.add_argument("--no-overlap", dest="overlap_streams", action="store_false",
                    help="fuse the two halves into one forward (the default)")
    rt.add_argument("--overlap-min-fill", type=float,
                    default=M1_DEFAULTS["overlap_min_fill"],
                    help="smallest fraction of --prefill-chunk-tokens a step's real "
                         "prefill tokens may fill before its prefill half runs eagerly "
                         "instead of padding to the graph's fixed shape. "
                         "0 = always graphed.")
    rt.add_argument("--async-scheduling", dest="async_scheduling",
                    action="store_true",
                    default=M1_DEFAULTS["async_scheduling"],
                    help="schedule step N+1 on the host while step "
                         "N runs on the device, and harvest step N's tokens one step "
                         "later through a pinned buffer and a CUDA event (vLLM's "
                         "'async scheduling'). The sampled token of step N is fed "
                         "into step N+1 by a device-side gather, so no D2H sits on "
                         "the critical path; the host learns each token one step "
                         "late, which costs at most one extra generated-and-"
                         "discarded token per request after EOS. Speculative steps "
                         "drain the pipeline (they commit a device-resident number "
                         "of tokens), so this changes nothing at conc <= "
                         "--spec-max-batch.")
    rt.add_argument("--no-async-scheduling", dest="async_scheduling",
                    action="store_false",
                    help="harvest every step's tokens inside the step (the default)")
    rt.add_argument("--overlap-decode-priority", type=int, default=0,
                    help="CUDA stream priority for the decode half of an overlapped "
                         "step (0 = default, -1 = high)")
    rt.add_argument("--mixed-graph-segments", type=int,
                    default=M1_DEFAULTS["mixed_graph_segments"],
                    help="prefill plan rows in a graphed mixed step; at most n-1 real "
                         "segments, and the chunk budget becomes chunk_tokens - n")
    rt.add_argument("--mlp-tile-tokens", type=int, default=M1_DEFAULTS["mlp_tile_tokens"],
                    help="prefill MLP tile. The chunk's MLP GEMMs run at "
                         "M=min(this, chunk tokens); the default 2048 means an 8192-token "
                         "chunk never routes them on the M=8192 bucket. Raising it to "
                         "--max-num-batched-tokens costs no extra plan memory.")
    rt.add_argument("--conv-prefill-tile-tokens", type=int, default=M1_DEFAULTS["conv_prefill_tile_tokens"],
                    help="token tile for the fp32 depthwise conv during prefill; 0 = untiled")
    rt.add_argument("--attn-backend", default=M1_DEFAULTS["attn_backend"], choices=["auto", "flashinfer", "torch"])
    rt.add_argument("--norm-backend", default=M1_DEFAULTS["norm_backend"], choices=["torch", "triton"])
    rt.add_argument("--fused-ops-backend", default=M1_DEFAULTS["fused_ops_backend"],
                    choices=["torch", "triton"],
                    help="fused GDN-gate/SwiGLU Triton kernels. 'triton' is the "
                         "measured-faster path and the canonical preset's value; 'torch' is the "
                         "unfused rollback.")
    rt.add_argument("--gemm-priority", default=M1_DEFAULTS["gemm_priority"], choices=["v9", "v8", "v7", "v4"],
                    help="GEMM cold-start priority table. 'v8' (default) = v7 "
                         "unchanged at M<=512 plus real prefill-scale buckets (1024/2048/4096/8192); "
                         "'v7' = the graph-timed gemm_v7 sweep (deepgemm rank 1 "
                         "at M in [32,512], clamps every M>512 to the M=512 answer); 'v4' = the "
                         "older table (marlin <=32, flashinfer 64-256, scaled_mm at 512).")
    rt.add_argument("--sampler-candidates", type=int, default=M1_DEFAULTS["sampler_candidates"])
    rt.add_argument("--attn-workspace-mb", type=int, default=M1_DEFAULTS["attn_workspace_mb"])
    rt.add_argument("--no-graphs", action="store_true", help="disable CUDA-graph capture (4.3x slower; debug only)")
    rt.add_argument("--enable-mtp", action="store_true")
    rt.add_argument("--mtp-hidden-first", action="store_true")
    rt.add_argument("--spec-k", type=int, default=0, choices=[0, 1, 2, 3],
                    help="MTP speculative-decode draft length; "
                         "0 (default) = spec decoding off entirely, byte-for-byte the plain "
                         "serving path. k>=1 implies --enable-mtp (forced on) and builds a "
                         "SpecDecoder; a head-to-head measured k=3 winning decisively "
                         "at B<=8 (150 vs 80 tok/s at B=1) and losing at B>=128 -- see "
                         "--spec-max-batch. Greedy (temperature<=0) requests use it; sampling "
                         "requests fall back to the plain step for that step regardless "
                         "(SpecConfig.greedy_only).")
    rt.add_argument("--spec-max-batch", type=int, default=16,
                    help="Per-step policy, not a build-time one: a decode step whose "
                         "*live* batch is <= this many sequences runs the speculative step when "
                         "--spec-k > 0 and every request in it is greedy; above it, the plain "
                         "graphed step runs instead, even for an all-greedy batch (spec "
                         "wins at B<=8, loses at B>=128 -- the doubled GDN window pass and the "
                         "M=B*(k+1) verify GEMM dominate there). Ignored when --spec-k is 0.")
    rt.add_argument("--fused-cache", default=None, help="dir holding a pre-built fused_weights.safetensors")
    rt.add_argument("--eos-token-id", type=int, default=None,
                    help="override the EOS id (default: the tokenizer's, else the model config's)")
    rt.add_argument("--gpu-memory-utilization", type=float, default=0.94,
                    help="preflight guard: refuse to start if the memory plan exceeds this fraction of free HBM")
    rt.add_argument("--skip-memory-check", action="store_true", help="print the memory plan but do not enforce it")
    rt.add_argument("--memory-plan-tolerance", type=float, default=0.05,
                    help="warn when the measured post-capture allocation differs from the plan's "
                         "steady state by more than this fraction (the plan is only useful if it "
                         "tracks measurement)")
    return p


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m qwenfast.runtime.serve",
        description="OpenAI-compatible server for qwenfast, backed by the real GPU runtime.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", required=True, help="weights directory (a snapshot dir, or a fused-weights cache)")
    p.add_argument("--tokenizer", default=None, help="tokenizer dir/repo id (default: --model)")
    p.add_argument("--served-model-name", default=None, help="name reported by /v1/models (default: --model)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--api-key", default=None, help="if set, require `Authorization: Bearer <key>` on /v1/*")
    p.add_argument("--default-max-tokens", type=int, default=512)
    p.add_argument("--detok-workers", type=int, default=4)
    p.add_argument("--log-level", default="info")
    # -- the engine loop in its own process -----------------------------------
    p.add_argument(
        "--engine-process", dest="engine_process", action="store_true", default=False,
        help="run the engine loop (model, scheduler, CUDA) in a separate "
             "process and talk to it over ZeroMQ, vLLM-style, instead of on a "
             "thread of the uvicorn interpreter. The in-process "
             "loop competes for the GIL with every event-loop operation on the "
             "serving side, which accounts for the gap between served and "
             "offline throughput.",
    )
    p.add_argument(
        "--no-engine-process", dest="engine_process", action="store_false",
        help="run the engine loop on an in-process thread (the default).",
    )
    p.add_argument("--engine-stats-interval", type=float, default=0.25,
                   help="how often the engine core pushes a /metrics snapshot "
                        "to the HTTP process (--engine-process only). A scrape "
                        "reads the last one, so it never enters the loop's "
                        "critical path.")
    p.add_argument("--engine-start-timeout", type=float, default=2400.0,
                   help="seconds to wait for the engine core to load weights "
                        "and capture graphs before failing startup")
    p.add_argument("--engine-idle-poll-ms", type=int, default=1,
                   help="how long the engine core waits on its input socket "
                        "when the scheduler has no work (the in-process loop's "
                        "0.5 ms sleep); a poll wakes the instant a request lands")
    p.add_argument("--step-trace-out", default=None,
                   help="write the engine loop's wall-clock split (step / "
                        "drain / emit / gap / idle, plus a per-second series) "
                        "to this JSON on shutdown. Works in both "
                        "modes; costs five perf_counter() calls per step.")
    p.add_argument("--step-profile-out", default=None,
                   help="write a per-step host+device attribution (admit / "
                        "collect / build / plan / launch / sync / harvest / "
                        "book, plus CUDA-event device idle and kernel time) to "
                        "this JSON on shutdown. Bounded to "
                        "--step-profile-steps steps after --step-profile-warmup, "
                        "so it is a window, not a permanent cost.")
    p.add_argument("--step-profile-steps", type=int, default=400,
                   help="steps to record for --step-profile-out")
    p.add_argument("--step-profile-min-rows", type=int, default=0,
                   help="only profile steps with at least this many decode rows. "
                        "A served run gates, warms up and ramps at "
                        "batch 1-8 before it reaches the level under test, so a "
                        "window counted in raw steps lands in the ramp; 200 at "
                        "conc 256 pins it to the level.")
    p.add_argument("--step-profile-warmup", type=int, default=200,
                   help="steps to skip before --step-profile-out starts recording "
                        "(a served sweep ramps; the first steps of a level are "
                        "not the level)")
    # uvicorn's default is **5 seconds**, and at conc 256 that is shorter than
    # the gap between two requests on one pooled client connection: the server
    # closes the socket, the client picks it out of its pool and writes into
    # it, and aiohttp reports "Server disconnected" or "Can not write request
    # body" while the engine itself logs nothing. Mixing makes it *more* likely
    # because a mixed step finishes every running request's token at once, so
    # completions -- and therefore idle sockets -- come in lumps.
    #
    # 300 s is longer than any plausible idle gap in the sweep (whose worst
    # measured end-to-end request is 166 s) and costs nothing but a held file
    # descriptor. It is a measurement artefact being removed, not a
    # performance knob: an error in a benchmark that is not the engine's is
    # worse than no benchmark.
    from ..server.config import add_public_api_args

    add_public_api_args(p)
    p.add_argument("--http-keep-alive-timeout", type=float, default=300.0,
                   help="seconds an idle HTTP connection is kept open "
                        "(uvicorn's default 5 s drops pooled client connections "
                        "at high concurrency)")
    add_runtime_args(p)
    # `serve.py`'s own CLI defaults already equal `preset.CANONICAL_FAST`
    # on every shared key (`tests/test_preset.py::TestPresetMatchesServe`), so
    # `--preset fastest` is a no-op against the defaults here -- it exists so a
    # launch script can say what config it means without the two ever being
    # able to disagree, the same contract `bench_runtime`/`bench_spec`/
    # `profile_step` already give it.
    add_preset_arg(p)
    return p


def _resolved_gdn_chunk_size(requested: int) -> int:
    """The preset default (32) needs fla's `chunk_size=` kwarg
    (`kernels_gdn.fla_ops.supports_chunk_size()`); an older fla bakes `BT`
    into the kernel's autotune config and `fla_ops.chunk_gdn` *raises* on
    anything but 64 (deliberately -- it never silently drops the keyword).
    That is the right behaviour for an explicit bench flag and the wrong one
    for a shipped preset default: a `--preset fastest` server should start on
    any host, not crash on its first prefill chunk because that host's fla is
    older. Falling back here (printed, not silent) only ever changes speed,
    never the model's numbers -- `chunk_size` is a pure implementation detail
    of the chunked-scan recurrence (`test_serving_path.py
    ::TestGdnChunkSizeParity` pins the torch-backend math invariant across
    16/32/64) -- so silently downgrading it is safe in a way a silently
    downgraded correctness knob would not be.
    """
    from ..kernels_gdn import fla_ops, shapes as gdn_shapes

    if requested == gdn_shapes.DEFAULT_CHUNK_SIZE or fla_ops.supports_chunk_size():
        return requested
    print(
        f"[serve] --gdn-chunk-size {requested} needs a newer fla "
        f"(kernels_gdn.fla_ops.supports_chunk_size() is False on this host) -- "
        f"falling back to {gdn_shapes.DEFAULT_CHUNK_SIZE}"
    )
    return gdn_shapes.DEFAULT_CHUNK_SIZE


def _pages_per_seq_floor(derived: int, args) -> int:
    """``max_pages_per_seq``, raised if a graphed mixed step needs more.

    A graphed mixed step pads up to ``--prefill-chunk-tokens`` tokens onto the
    scratch slot, and the KV page table is
    ``[max_seqs, max_pages_per_seq]``, so that slot cannot hold the padding
    unless the table is at least ``ceil(chunk / page_size) + 1`` wide. The
    derived value comes from ``--max-model-len`` (173 at 2,752 / page 16) and
    is smaller than a 4,096-token chunk needs; without the floor, graphed
    chunk-4,096 and chunk-8,192 steps fail with
    ``slot 256 needs 256 pages (> max_pages_per_seq=173)``.

    Only ever raises, and only under ``--mixed-forward --mixed-graphs``, so
    every other configuration is byte-identical. The cost is
    ``max_num_seqs * extra * 4`` bytes of page table -- 88 KiB at 264 slots and
    chunk 8,192.
    """
    if not (getattr(args, "mixed_forward", False) and getattr(args, "mixed_graphs", False)):
        return int(derived)
    from .mixed_graphs import scratch_page_floor

    chunk = getattr(args, "prefill_chunk_tokens", 0) or getattr(
        args, "max_num_batched_tokens", 0
    )
    return max(
        int(derived),
        scratch_page_floor(chunk, args.page_size, getattr(args, "max_model_len", 0) or 0)
        + 1,
    )


def runtime_config_from_args(args: argparse.Namespace) -> RuntimeConfig:
    """``argparse.Namespace`` -> :class:`RuntimeConfig`, with the pool geometry
    derived from ``--max-model-len``/``--max-num-seqs`` unless overridden.

    ``derive_pool_sizes`` is ``bench_runtime``'s, deliberately: the offline
    decode bench and the server must size their pools by the same rule or a
    served number is not comparable to an offline one.
    """
    geom = derive_pool_sizes(
        args.max_num_seqs, args.max_model_len, args.page_size, slack_pages=args.kv_pages_slack
    )
    pgb = getattr(args, "prefill_gemm_backend", None)
    if (
        pgb in REPACK_CACHE_BACKENDS
        and getattr(args, "gemm_weight_cache", "multi") != "multi"
        # `--gemm-cache-owner prefill` makes this backend the *first* claimant
        # rather than a second one, so it takes the one slot legitimately and
        # the decode buckets fall to a cache-free backend. Still one cache.
        and getattr(args, "gemm_cache_owner", "decode") != "prefill"
    ):
        raise SystemExit(
            f"--prefill-gemm-backend {pgb} owns a weight repack cache "
            f"(~23 GiB on this checkpoint) and --gemm-weight-cache is "
            f"{args.gemm_weight_cache!r}. Forcing it would allocate a *second* "
            f"repacked copy on the first prefill chunk and run out of memory. "
            f"Use a cache-free backend (flashinfer_fp8_blockscale, "
            f"vllm_block_fp8_cutlass, vllm_block_fp8_triton), or pass "
            f"--gemm-weight-cache multi and re-check the printed memory plan."
        )
    return RuntimeConfig(
        device=args.device,
        dtype=args.dtype,
        ssm_state_dtype=args.ssm_state_dtype,
        kv_cache_dtype=args.kv_cache_dtype,
        page_size=args.page_size,
        max_num_seqs=args.max_num_seqs,
        n_kv_pages=args.n_kv_pages if args.n_kv_pages is not None else geom["n_kv_pages"],
        # Raised if `--mixed-graphs` needs a wider page table for the
        # scratch slot than `--max-model-len` implies. Explicit
        # `--max-pages-per-seq` still wins, and is then checked at capture.
        max_pages_per_seq=_pages_per_seq_floor(
            args.max_pages_per_seq if args.max_pages_per_seq is not None
            else geom["max_pages_per_seq"],
            args,
        ),
        max_num_batched_tokens=args.max_num_batched_tokens,
        prefill_decode_ratio=args.prefill_decode_ratio,
        max_model_len=args.max_model_len,
        gdn_backend=args.gdn_backend,
        gemm_backend=args.gemm_backend,
        gemm_weight_cache=args.gemm_weight_cache,
        gemm_accuracy=args.gemm_accuracy,
        conv_prefill_tile_tokens=args.conv_prefill_tile_tokens,
        conv_prefill_layout=args.conv_prefill_layout,
        mlp_tile_tokens=getattr(args, "mlp_tile_tokens", M1_DEFAULTS["mlp_tile_tokens"]),
        prefill_gemm_backend=args.prefill_gemm_backend,
        prefill_chunk_tokens=args.prefill_chunk_tokens,
        mixed_forward=getattr(args, "mixed_forward", M1_DEFAULTS["mixed_forward"]),
        mixed_graphs=getattr(args, "mixed_graphs", M1_DEFAULTS["mixed_graphs"]),
        mixed_graph_segments=getattr(
            args, "mixed_graph_segments", M1_DEFAULTS["mixed_graph_segments"]
        ),
        mixed_graph_holes=getattr(
            args, "mixed_graph_holes", M1_DEFAULTS["mixed_graph_holes"]
        ),
        mixed_graph_min_bucket=getattr(
            args, "mixed_graph_min_bucket", M1_DEFAULTS["mixed_graph_min_bucket"]
        ),
        overlap_streams=getattr(args, "overlap_streams",
                                M1_DEFAULTS["overlap_streams"]),
        overlap_decode_priority=getattr(args, "overlap_decode_priority", 0),
        overlap_min_fill=getattr(args, "overlap_min_fill",
                                 M1_DEFAULTS["overlap_min_fill"]),
        async_scheduling=getattr(args, "async_scheduling",
                                 M1_DEFAULTS["async_scheduling"]),
        gemm_cache_owner=args.gemm_cache_owner,
        gdn_chunk_size=_resolved_gdn_chunk_size(
            getattr(args, "gdn_chunk_size", M1_DEFAULTS["gdn_chunk_size"])
        ),
        attn_backend=args.attn_backend,
        norm_backend=args.norm_backend,
        fused_ops_backend=getattr(args, "fused_ops_backend", M1_DEFAULTS["fused_ops_backend"]),
        gemm_priority=getattr(args, "gemm_priority", M1_DEFAULTS["gemm_priority"]),
        sampler_candidates=args.sampler_candidates,
        attn_workspace_mb=args.attn_workspace_mb,
        use_cuda_graphs=not args.no_graphs,
        # Spec decoding needs the MTP head loaded regardless of
        # whether `--enable-mtp` was also passed -- `SpecDecoder.__init__`
        # raises if `model.mtp is None`, and that would
        # otherwise be a confusing crash three minutes into weight loading
        # rather than a config the user can read back.
        enable_mtp=args.enable_mtp or getattr(args, "spec_k", 0) > 0,
        mtp_hidden_first=args.mtp_hidden_first,
    )


# --------------------------------------------------------------------------- #
# 4. engine construction
# --------------------------------------------------------------------------- #
def load_tokenizer(model: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "transformers is required to load the tokenizer (`pip install transformers`)."
        ) from exc
    return AutoTokenizer.from_pretrained(model, trust_remote_code=True)


def _free_hbm_gib(device: str) -> Optional[float]:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info(torch.device(device))
        return free / _GIB
    except Exception:
        return None


def spec_buckets_for(rt: RuntimeConfig, spec_max_batch: int) -> Tuple[int, ...]:
    """Which of ``rt``'s graph buckets a server's ``SpecDecoder`` should
    capture.

    ``Scheduler._use_spec`` never routes a step to speculation above
    ``--spec-max-batch`` live sequences (spec wins at B<=8, loses at
    B>=128), so capturing spec graphs for the server's *full* bucket table --
    up to 512 -- would pay capture time and, more to the point,
    :func:`plan_memory`'s resident spec-window-cache bytes for shapes
    nothing ever replays. This caps the captured set at
    ``spec_max_batch``, always keeping at least one bucket (so a degenerate
    ``spec_max_batch`` still gets *something* to capture/warm up against) and
    always covering ``spec_max_batch`` itself (appended if no bucket lands on
    it exactly), so ``SpecDecoder.bucket_for(spec_max_batch)`` -- the largest
    batch spec can ever actually be asked to run -- resolves to a captured
    graph rather than falling through to the eager per-step path.
    """
    cap = max(int(spec_max_batch), 1)
    buckets = tuple(b for b in rt.buckets_for() if b <= cap)
    if not buckets or buckets[-1] < cap:
        buckets = buckets + (cap,)
    return buckets


def plan_memory_for(
    args: argparse.Namespace,
    rt: RuntimeConfig,
    *,
    weight_bytes: Optional[float] = None,
    linear_fp8_bytes: Optional[float] = None,
) -> Dict[str, float]:
    """:func:`plan_memory` fed from the resolved CLI args + runtime config.

    One call site for the argument mapping so the startup report, the
    post-capture gate and the tests can never disagree about what was planned.
    ``weight_bytes``/``linear_fp8_bytes`` let the post-capture gate replace the
    two config-derived estimates with what the loaded model really holds.
    """
    buckets = rt.buckets_for()
    # `spec_window` (n = k+1) is 0 -- the plain serving plan, byte for
    # byte -- unless `--spec-k` is set; `spec_max_batch` is the *widest spec*
    # bucket (see `spec_buckets_for`), not the server's own `max_batch`.
    spec_k = int(getattr(args, "spec_k", 0) or 0)
    spec_window = spec_k + 1 if spec_k > 0 else 0
    spec_max_batch = (
        spec_buckets_for(rt, getattr(args, "spec_max_batch", 16) or 1)[-1] if spec_window else None
    )
    return plan_memory(
        weight_bytes=weight_bytes,
        linear_fp8_bytes=linear_fp8_bytes,
        max_num_seqs=rt.max_num_seqs,
        max_model_len=args.max_model_len,
        page_size=rt.page_size,
        n_kv_pages=rt.n_kv_pages,
        max_pages_per_seq=rt.max_pages_per_seq,
        kv_cache_dtype=rt.kv_cache_dtype,
        ssm_state_dtype=rt.ssm_state_dtype,
        dtype=rt.dtype,
        enable_mtp=rt.enable_mtp,
        arch=arch_from_checkpoint(args.model),
        max_num_batched_tokens=rt.max_num_batched_tokens,
        conv_prefill_tile_tokens=rt.conv_prefill_tile_tokens,
        n_graph_buckets=len(buckets),
        mixed_graph_buckets=len(rt.mixed_buckets_for()) if (
            rt.mixed_graph_buckets or rt.overlap_streams
        ) else (sum(1 for b in buckets if b >= rt.mixed_graph_min_bucket) or 1),
        max_batch=buckets[-1] if rt.use_cuda_graphs else rt.max_num_seqs,
        attn_workspace_mb=rt.attn_workspace_mb,
        gemm_weight_cache=rt.gemm_weight_cache,
        use_cuda_graphs=rt.use_cuda_graphs,
        spec_window=spec_window,
        spec_max_batch=spec_max_batch,
        mixed_forward=rt.mixed_forward,
        mixed_graphs=rt.mixed_graphs,
        mixed_graph_chunk=(rt.prefill_chunk_tokens or rt.max_num_batched_tokens),
        mixed_graph_holes=rt.mixed_graph_holes,
        mixed_graph_segments=rt.mixed_graph_segments,
        gdn_chunk_size=rt.gdn_chunk_size,
    )


def _post_capture_check(engine, rt: RuntimeConfig, plan: Dict[str, float],
                        budget: Optional[float], args: argparse.Namespace,
                        *, verbose: bool = True) -> None:
    """Compare the *measured* post-capture footprint against plan and budget.

    Two independent judgements, deliberately not conflated:

    1. **Accuracy** -- does the plan track reality within
       ``--memory-plan-tolerance``? A miss here is a bug in :func:`plan_memory`
       and is reported loudly, but it does not by itself stop the server.
    2. **Safety** -- is ``measured + prefill headroom`` inside the budget? A
       miss here *does* stop the server, because the alternative is READY,
       then a CUDA OOM seconds later on the first real prompt, with the engine
       thread dead and the HTTP server still answering.
    """
    measured = measure_allocation(rt.device)
    if measured is None:
        return

    # Re-plan against the tensors that actually exist. The pre-load plan has to
    # guess the weight and repack byte counts from config.json; here the model
    # is loaded and warmed, so both are readable, and the accuracy check below
    # is then a statement about the *plan's structure* (did it miss a term?)
    # rather than about how well its shape constants were guessed.
    try:
        pools = engine.model.pool_nbytes()
        plan = dict(plan)
        plan.update(
            plan_memory_for(
                args, rt,
                weight_bytes=float(pools["weights"]),
                linear_fp8_bytes=float(pools["repack_caches"]) / (1.0 + _MARLIN_SCALE_RATIO)
                if pools["repack_caches"]
                else None,
            )
        )
        if verbose:
            parts = "  ".join(f"{k} {v / _GIB:.2f}" for k, v in sorted(pools.items()))
            print(f"[serve] measured pools (GiB): {parts}", flush=True)
    except Exception:  # pragma: no cover -- introspection only; keep the plan
        pass

    if verbose:
        print(format_measured(measured, plan), flush=True)

    steady = plan["steady_gib"]
    if steady > 0:
        err = abs(measured["in_use_gib"] - steady) / steady
        if err > args.memory_plan_tolerance:
            print(
                f"[serve] WARNING: memory plan is off by {err * 100:.1f}% "
                f"(> {args.memory_plan_tolerance * 100:.0f}% tolerance): planned "
                f"{steady:.2f} GiB steady, measured {measured['in_use_gib']:.2f} GiB. "
                f"serve.plan_memory is missing a term.",
                flush=True,
            )

    if budget is None or args.skip_memory_check:
        return
    needed = measured["in_use_gib"] + plan["prefill_gib"]
    if needed > budget:
        raise RuntimeError(
            f"post-capture memory check failed: {measured['in_use_gib']:.2f} GiB is in use after "
            f"graph capture and one prefill chunk needs up to {plan['prefill_gib']:.2f} GiB more "
            f"({needed:.2f} GiB), over the {budget:.2f} GiB budget. Refusing to serve -- this "
            f"configuration would run out of memory on its first request. Lower "
            f"--max-num-seqs / --max-model-len / --max-num-batched-tokens, or set "
            f"--gemm-weight-cache none (frees {plan['linear_fp8_gib']:.1f} GiB)."
        )


def build_engine_from_args(args: argparse.Namespace, tokenizer=None, *, verbose: bool = True):
    """Build the :class:`~.engine.QwenFastEngine` for ``args``.

    Prints the memory plan and (unless ``--skip-memory-check``) refuses to
    start when it does not fit ``--gpu-memory-utilization`` of free HBM --
    an up-front, readable failure instead of a CUDA OOM 40 GiB into weight
    loading, or (worse) an OOM three minutes later during graph capture.
    """
    rt = runtime_config_from_args(args)
    plan = plan_memory_for(args, rt)
    if verbose:
        print(format_memory_plan(plan, rt, args.max_model_len), flush=True)

    free_gib = _free_hbm_gib(rt.device)
    budget = None
    if free_gib is not None:
        budget = free_gib * args.gpu_memory_utilization
        if verbose:
            print(
                f"[serve] free HBM {free_gib:.1f} GiB, budget {budget:.1f} GiB "
                f"(--gpu-memory-utilization {args.gpu_memory_utilization})",
                flush=True,
            )
        if plan["total_gib"] > budget and not args.skip_memory_check:
            raise SystemExit(
                f"error: memory plan needs {plan['total_gib']:.1f} GiB "
                f"({plan['steady_gib']:.1f} steady + {plan['prefill_gib']:.1f} prefill) but only "
                f"{budget:.1f} GiB is budgeted ({free_gib:.1f} GiB free x "
                f"{args.gpu_memory_utilization}). Lower --max-num-seqs (each slot is "
                f"{plan['ssm_mib_per_slot']:.0f} MiB) or --max-model-len (each token is "
                f"{plan['kv_kib_per_token']:.0f} KiB of KV), switch --kv-cache-dtype fp8 / "
                f"--ssm-state-dtype fp16, drop --gemm-weight-cache to 'none' (frees "
                f"{plan['linear_fp8_gib']:.1f} GiB), or pass --skip-memory-check."
            )

    # `--spec-k 0` (the default) builds no
    # SpecDecoder at all -- `spec=None` is the exact plain serving path,
    # byte for byte, not a k=0 SpecConfig (the k=0 control build is
    # for the correctness gate, not production: it is strictly slower than
    # plain decode). `spec_buckets_for` caps what the *server's* SpecDecoder
    # captures at `--spec-max-batch` -- see its docstring for why capturing
    # the server's full (up to 512) bucket table would be wasted capture time
    # and resident memory for shapes `Scheduler._use_spec` never routes to.
    spec_cfg: Optional[SpecConfig] = None
    spec_max_batch: Optional[int] = None
    if args.spec_k > 0:
        spec_max_batch = args.spec_max_batch
        spec_cfg = SpecConfig(k=args.spec_k, buckets=spec_buckets_for(rt, spec_max_batch))

    engine = build_async_engine(
        args.model, rt=rt, spec=spec_cfg, spec_max_batch=spec_max_batch, fused_cache=args.fused_cache
    )

    # Before `engine.start()` -> `decoder.warmup()`, because the
    # point is to resolve the prefill bucket *first*: after warmup has run
    # bucket 1, marlin owns the only cache slot and this would be a no-op.
    if rt.gemm_cache_owner == "prefill":
        engine.model.claim_gemm_cache(verbose=verbose)

    # -- the post-capture gate ------------------------------------------------
    # The pre-load plan is an estimate; graph capture is where the real number
    # lands, and it can be tens of GiB over the estimate. `QwenFastEngine.start()` calls this right after
    # `capture()`, before the HTTP layer accepts anything, so an over-budget
    # config fails at startup instead of on the first 2000-token prompt.
    engine.memory_plan = plan
    engine.memory_budget_gib = None if args.skip_memory_check else budget
    engine.memory_plan_tolerance = args.memory_plan_tolerance
    engine.post_capture_check = lambda: _post_capture_check(
        engine, rt, plan, budget, args, verbose=verbose
    )

    eos = args.eos_token_id
    if eos is None and tokenizer is not None:
        eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        # The tokenizer's EOS (`<|im_end|>` for the chat template) is the one
        # the served model actually stops on; `config.eos_token_id` can be the
        # base `<|endoftext|>`. `QwenFastEngine` reads this attribute on every
        # `add_request`, so setting it post-construction is enough -- and it
        # keeps `engine.py` (shared with the offline benches) free of a
        # serving-only kwarg.
        engine.eos_token_id = int(eos)
    if verbose:
        print(
            f"[serve] eos_token_id={engine.eos_token_id} graphs={'on' if rt.use_cuda_graphs else 'off'} "
            f"norm={rt.norm_backend} attn={rt.attn_backend} gdn={rt.gdn_backend} "
            f"ssm_state={rt.ssm_state_dtype} kv={rt.kv_cache_dtype} "
            f"max_num_seqs={rt.max_num_seqs} max_num_batched_tokens={rt.max_num_batched_tokens} "
            f"prefill_decode_ratio={rt.prefill_decode_ratio} "
            f"prefill_chunk_tokens={rt.prefill_chunk_tokens or rt.max_num_batched_tokens} "
            f"mixed_forward={'on' if rt.mixed_forward else 'off'} "
            f"mixed_graphs={'on' if rt.mixed_graphs else 'off'}"
            f"{'(holes)' if rt.mixed_graphs and rt.mixed_graph_holes else ''} "
            f"mlp_tile={rt.mlp_tile_tokens} "
            f"conv_prefill={rt.conv_prefill_layout} "
            f"prefill_gemm={rt.prefill_gemm_backend or 'dispatch'} "
            f"gemm_cache_owner={rt.gemm_cache_owner} "
            f"n_kv_pages={rt.n_kv_pages} max_pages_per_seq={rt.max_pages_per_seq} "
            # The *enforced* limit, which is min(rotary rows, page-table
            # geometry) and not necessarily --max-model-len. This line exists so the number that 400s a request is visible
            # in the server log without reading the source.
            f"max_context_len={engine.max_context_len} "
            # spec_k=0 is the exact plain serving path (no SpecDecoder
            # built at all); spec_buckets is the *capped* set `spec_buckets_for`
            # chose, printed so a startup log states exactly what got captured
            # rather than leaving it to be inferred from --spec-max-batch.
            f"spec_k={args.spec_k} spec_max_batch={spec_max_batch} "
            f"spec_buckets={list(spec_cfg.buckets) if spec_cfg is not None else None}",
            flush=True,
        )
    return engine


def build_engine_for_core(args: dict, eos_token_id: Optional[int] = None):
    """The ``module:function`` builder ``--engine-process`` hands the core.

    Runs **in the child**, so everything that touches CUDA -- the free-HBM
    probe, weight loading, graph capture, the post-capture memory gate --
    happens there and the HTTP process never creates a CUDA context.  Takes a
    plain dict (``vars(args)``) because that is what ``multiprocessing``
    spawn can pickle without dragging argparse internals across.

    ``eos_token_id`` is resolved from the tokenizer in the *parent*, so the
    child does not load a second copy of a 248k-entry tokenizer just to read
    one integer off it.
    """
    ns = argparse.Namespace(**args)
    engine = build_engine_from_args(ns, tokenizer=None, verbose=True)
    if eos_token_id is not None:
        engine.eos_token_id = int(eos_token_id)
    if getattr(ns, "step_profile_out", None):
        # The step profile is a *scheduler*-side instrument, so it goes
        # wherever the scheduler lives -- here, in the core, not in the HTTP
        # process that has no CUDA context.
        from .step_trace import StepProfiler

        engine.step_profiler = StepProfiler(
            path=ns.step_profile_out,
            max_steps=getattr(ns, "step_profile_steps", 400),
            warmup_steps=getattr(ns, "step_profile_warmup", 200),
            min_rows=getattr(ns, "step_profile_min_rows", 0),
            device=engine.model.device,
            meta={"mode": "engine-process",
                  "async_scheduling": bool(getattr(ns, "async_scheduling", False))},
        )
        engine.scheduler.profiler = engine.step_profiler
    return engine


def build_engine_client_from_args(args: argparse.Namespace, tokenizer=None):
    """The ``--engine-process`` counterpart of :func:`build_engine_from_args`.

    Returns an :class:`~.engine_core.EngineCoreClient`, which satisfies the
    same ``AsyncEngine`` contract (plus ``max_context_len`` /
    ``context_length_error`` / ``health``), so ``server/app.py`` is unchanged.
    Nothing is loaded here: the model appears when ``app``'s lifespan calls
    ``engine.start()``, which spawns the core and blocks until it is ready.
    """
    from .engine_core import EngineCoreClient

    eos = args.eos_token_id
    if eos is None and tokenizer is not None:
        eos = getattr(tokenizer, "eos_token_id", None)
    return EngineCoreClient(
        builder="qwenfast.runtime.serve:build_engine_for_core",
        builder_kwargs={"args": vars(args), "eos_token_id": eos},
        start_timeout_s=args.engine_start_timeout,
        stats_interval_s=args.engine_stats_interval,
        idle_poll_ms=args.engine_idle_poll_ms,
        step_trace_out=args.step_trace_out,
        eos_token_id=eos,
        trace_meta={
            "prefill_chunk_tokens": args.prefill_chunk_tokens,
            "max_num_seqs": args.max_num_seqs,
            "overlap": bool(args.overlap_streams),
            "spec_k": args.spec_k,
            "detok_workers": args.detok_workers,
        },
    )


def main(argv: Optional[list] = None) -> None:
    # Imported here, not at module scope: `plan_memory` is now also the memory
    # gate for the *offline* benches (`bench_spec`), which run in the
    # same venv but must not need fastapi/uvicorn to compute a byte budget.
    import uvicorn

    from ..server.app import create_app
    from ..server.config import describe_public_config, public_api_kwargs

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    # `--preset fastest` fills in only knobs left at their CLI default (an
    # explicitly-passed flag always wins); a no-op today since M1_DEFAULTS
    # already equals CANONICAL_FAST, kept for parity with the other CLIs and
    # so a launch script's `--preset fastest --spec-k 3 ...` says what it means.
    args = apply_preset(args, parser, argv=argv)
    tokenizer = load_tokenizer(args.tokenizer or args.model)
    model_name = args.served_model_name or args.model

    if args.engine_process:
        # The model is built in the child, by `engine.start()` inside
        # uvicorn's lifespan -- so a core that cannot start (OOM, the
        # post-capture gate) fails the lifespan and exits the process with a
        # non-zero status, rather than leaving a live HTTP port in front of a
        # dead engine. That is the same contract `/health` gives at run time.
        engine = build_engine_client_from_args(args, tokenizer)
        print("[serve] engine mode: separate process (--engine-process)", flush=True)
    else:
        engine = build_engine_from_args(args, tokenizer)
        if args.step_trace_out:
            from .step_trace import StepTrace

            engine.step_trace = StepTrace(
                args.step_trace_out,
                meta={
                    "mode": "in-process",
                    "pid": os.getpid(),
                    "prefill_chunk_tokens": args.prefill_chunk_tokens,
                    "max_num_seqs": args.max_num_seqs,
                    "overlap": bool(args.overlap_streams),
                    "spec_k": args.spec_k,
                    "detok_workers": args.detok_workers,
                    "async_scheduling": bool(getattr(args, "async_scheduling", False)),
                },
            )
        if getattr(args, "step_profile_out", None):
            from .step_trace import StepProfiler

            engine.step_profiler = StepProfiler(
                path=args.step_profile_out,
                max_steps=args.step_profile_steps,
                warmup_steps=args.step_profile_warmup,
                min_rows=args.step_profile_min_rows,
                device=engine.model.device,
                meta={
                    "prefill_chunk_tokens": args.prefill_chunk_tokens,
                    "max_num_seqs": args.max_num_seqs,
                    "overlap": bool(args.overlap_streams),
                    "spec_k": args.spec_k,
                    "async_scheduling": bool(getattr(args, "async_scheduling", False)),
                },
            )
            engine.scheduler.profiler = engine.step_profiler
    public = public_api_kwargs(args)
    app = create_app(
        engine,
        tokenizer,
        model_name=model_name,
        api_key=args.api_key,
        default_max_tokens=args.default_max_tokens,
        detok_workers=args.detok_workers,
        **public,
    )
    print(describe_public_config(public, app.state.key_store.names()), flush=True)
    # `timeout_graceful_shutdown` is the *hard* half of the drain:
    # `create_app`'s SIGTERM handler flips the server to refusing new work at
    # once, and this bounds how long uvicorn then waits for the SSE streams
    # already in flight before exiting anyway.
    uvicorn.run(
        app, host=args.host, port=args.port, log_level=args.log_level, http="h11",
        timeout_keep_alive=int(args.http_keep_alive_timeout),
        timeout_graceful_shutdown=int(getattr(args, "drain_timeout", 0) or 0) or None,
    )


__all__ = [
    "M1_DEFAULTS",
    "arch_from_checkpoint",
    "add_runtime_args",
    "build_arg_parser",
    "build_engine_client_from_args",
    "build_engine_for_core",
    "build_engine_from_args",
    "format_measured",
    "format_memory_plan",
    "load_tokenizer",
    "main",
    "measure_allocation",
    "plan_memory",
    "plan_memory_for",
    "runtime_config_from_args",
    "spec_buckets_for",
    "weight_bytes_for",
]

if __name__ == "__main__":
    main()
