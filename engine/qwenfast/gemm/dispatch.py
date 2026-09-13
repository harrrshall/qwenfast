"""GEMM dispatch table.

``linear(x, w) -> bf16`` for every fused weight kind produced by
``fused_weights.py``: ``w`` is either an :class:`~fused_weights.FP8Tensor`
(block-128 quantized) or a plain bf16 ``torch.Tensor`` (``in_proj_ba``,
``conv1d``, norms, ...). The dispatcher picks a backend keyed on
``(sm_version, M, N, K)`` via the M-bucketed autotune cache in
``autotune.py``, falling back to a static priority order when no autotune
entry exists, and falling further down that order on any backend failure
(missing package, unsupported shape, ...) so ``linear()`` never hard-fails
on a host that is missing an optional dependency.

## Why there are several FP8 backends

FP8 blockwise GEMM is not unconditionally faster than bf16 (CUTLASS #2923 /
FlashInfer #2146). On H200 (torch 2.13 cu13, vLLM 0.28) a whole-model
per-decode-step sweep measured ``vllm_block_fp8_triton`` 2.1x **slower** than
plain bf16 ``torch.matmul`` at every M from 1 to 512 (36.2 ms vs 16.2 ms per
step). The backends below exist so that routing is decided by measurement per
shape and M rather than assumed:

  flashinfer_fp8_blockscale: ``flashinfer.gemm.fp8_blockscale_gemm_sm90``,
                            bf16 activations in, FP8 weight. **Not W8A16**:
                            it quantizes the activations internally and
                            measures relL2 2.6e-2 against an exact dequant
                            reference where marlin measures 2.7e-3. See the
                            ACCURACY WARNING on ``_flashinfer_fp8_blockscale``
                            below. Reads the checkpoint's native 128x128
                            block scale directly. FlashInfer's own docstring:
                            "SwapAB kernel is automatically used when M < 32",
                            so it already carries a small-M/batch-1
                            optimization without a bespoke Triton kernel.
                            SM90 (Hopper) only.
  vllm_marlin_fp8_w8a16:    ``vllm._custom_ops.marlin_gemm`` (the unified
                            Marlin op, ``b_q_type=scalar_types.float8_e4m3fn``)
                            fed a GPTQ-repacked/permuted copy of the weight,
                            ported from vLLM's
                            ``marlin_utils_fp8.prepare_fp8_layer_for_marlin``
                            + ``apply_fp8_marlin_linear`` (that pair is
                            layer/nn.Module-shaped; ported here to operate on
                            a bare ``FP8Tensor``). W8A16 (bf16 in, weight-only
                            fp8). vLLM describes this path as existing for
                            "GPUs that lack FP8 hardware support", so it is
                            not expected to win on hardware with native FP8
                            everywhere, but a benchmark-driven dispatch table
                            lets the autotune sweep decide rather than assume.
  machete_w8a16:            vLLM's Machete kernel
                            (``_custom_ops.machete_mm`` / ``machete_prepack_B``,
                            CUTLASS mixed-input, Hopper-only), the second true
                            W8A16 path: bf16 activations in, weight
                            dequantized inside the kernel, **zero activation
                            error**. It exists because marlin, the only other
                            W8A16 backend, does not scale in M (8.2 ms/step at
                            M=1 -> 88.8 at 512), which is why the dispatcher
                            hands every batch >= 64 to a 10x-less-accurate
                            kernel.
                            **Caveat:** vLLM 0.28.0's Machete has NO compiled
                            fp8 b_type (``machete_supported_schedules(bf16,
                            float8_e4m3fn, ...)`` returns an empty list; only
                            ``uint4b8``/``uint8b128`` are instantiated), so this
                            backend re-quantizes the weight to int8 with
                            per-(128-K-group, output-channel) scales. That costs
                            relL2 ~7.0e-3 against the fp32-dequant reference vs
                            marlin's 2.7e-3: still 3.7x *better* than every
                            fp8-activation backend, but not marlin's equal. See
                            ``_machete_repacked`` for the arithmetic and
                            ``BACKEND_REL_L2`` for how routing uses it.
  deepgemm:                 weight repack via
                            ``fp8_utils.deepgemm_post_process_fp8_weight_block``
                            (TMA-aligned scale layout) + activation quant via
                            ``fp8_utils.per_token_group_quant_fp8(...,
                            column_major_scales=True, tma_aligned_scales=True)``
                            + ``vllm.utils.deep_gemm.fp8_gemm_nt``, matching
                            vLLM's real call site
                            (``model_executor/kernels/linear/scaled_mm/
                            deep_gemm.py``). Auto-sets ``CUDA_HOME`` to the
                            cu13 nvcc wheel in the venv (``.../nvidia/cu13``)
                            if unset, so the JIT does not need a hand-exported
                            env var.

Other backends:

  bf16_dequant:           dequantize (if FP8) then ``F.linear``. Always
                          available (pure torch). Materializes the full bf16
                          weight *inside every call* (not cached), so it does
                          *more* total HBM traffic than a baseline with bf16
                          weights already resident (read fp8 + write bf16 +
                          read bf16, vs. just read bf16). It is a correctness
                          fallback, not a competitive backend.
  vllm_block_fp8_triton:  vLLM's own block-scaled kernel:
                          ``vllm.model_executor.layers.quantization.utils
                          .fp8_utils.w8a8_triton_block_scaled_mm``, fed by
                          ``per_token_group_quant_fp8`` (1x128 activation
                          groups, matching the checkpoint's weight grid).
                          **Measured 2.1x slower than bf16** on H200; kept
                          registered (autotune may still prefer it at a
                          shape/M that has not been swept) but demoted in the
                          default priority order.
  vllm_block_fp8_cutlass: vLLM's CUTLASS block-scaled ``scaled_mm``
                          (``vllm._custom_ops.cutlass_scaled_mm``, which
                          natively supports 1x128 / 128x128 "group"
                          broadcast scales; see the vLLM 0.28.0 note below).
  scaled_mm_pertensor:    ``torch._scaled_mm`` with ONE scale for the whole
                          tensor, re-derived from the block scales (max
                          over blocks). Documented accuracy hit: this
                          collapses 128x128-granular scaling to a single
                          number, so it is only appropriate for wide,
                          roughly-uniform-magnitude weights (e.g. lm_head).
                          Layout: cuBLASLt's ``torch._scaled_mm`` requires
                          operand A row-major and operand B column-major, so
                          the weight is passed as a transposed view (``.t()``
                          with **no** ``.contiguous()``). A transpose of a
                          contiguous ``[N, K]`` tensor is already exactly
                          column-major ``[K, N]``; ``.contiguous()`` after it
                          would re-materialize row-major layout and hit
                          "Only multiplication of row-major and column-major
                          matrices is supported by cuBLASLt".

Note on the "CUTLASS path" name: ``apply_w8a8_block_fp8_linear`` does not
exist in vLLM 0.28.0. The real block-FP8 CUTLASS surface in
0.28.0 is ``vllm.model_executor.kernels.linear.init_fp8_linear_kernel(...)
.apply_weights(layer, x, bias)``, a heavyweight API that expects a live
``nn.Module`` layer (registered ``weight``/``weight_scale`` parameters, a
``MMLinearLayerConfig`` describing TP partitioning, etc.), which is not a fit
for a raw-tensor ``linear(x, w)`` call. The raw tensor-in/tensor-out primitive
that *is* stable and used underneath both the Triton and CUTLASS code paths is
``vllm._custom_ops.cutlass_scaled_mm(a, b, scale_a, scale_b, out_dtype,
bias=None)``, whose docstring explicitly documents 1x128 / 128x128 "group"
broadcast scaling support; that is what ``vllm_block_fp8_cutlass`` below
calls.

This module must import cleanly with no torch/CUDA available -- every torch,
vllm, or flashinfer import lives inside a function body.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

from .fused_weights import BLOCK, FP8Tensor

try:  # module-level so lazily imported backends never hit a NameError on torch
    import torch
except ImportError:  # pragma: no cover
    if TYPE_CHECKING:
        import torch

WeightT = Union["torch.Tensor", FP8Tensor]
BackendFn = Callable[["torch.Tensor", WeightT], "torch.Tensor"]

# M buckets the autotune cache and bench_gemm.py sweep over: the standard
# qwenfast decode batch buckets (a subset of the CUDA-graph bucket list, plus
# 1), extended with prefill-scale buckets 1024, 1536, 2048, 3072, 4096, 8192.
#
# Why this matters: `m_bucket` rounds *up* to the next entry here and clamps
# at the top (`for b in M_BUCKETS: if m <= b: return b`). Without the
# prefill-scale entries every prefill chunk (M up to `max_num_batched_tokens`,
# 8,192 by default) would resolve on the bucket measured at M=512, a table
# never validated above it. `ResolvedLinear.__call__`'s non-forced path
# (`runtime/fused_model.py`) is generic in `m_bucket`, so this list alone is
# what gives a real prefill chunk its own routing decision.
#
# 1536 and 3072 are not cosmetic: the mixed step's M is
# `prefill_chunk + decode_rows` (1,024 + 128/256 = 1,152/1,280; 2,048 + 256 =
# 2,304), and without them each of those would round up to 2048/4096, i.e. the
# M this engine's every-step-mixing configuration actually runs would be
# routed on a table measured 1.6-1.8x above it. They make `m_bucket` *finer
# exactly where the mixed step lands* and change nothing elsewhere: for the
# v4/v7/v8 profiles each new bucket's row is a copy of the row the value used
# to round up to (1536 -> 2048, 3072 -> 4096), so routing under those profiles
# is byte-identical. Only `v9`, which is measured per (shape, bucket),
# distinguishes them.
M_BUCKETS: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 3072, 4096, 8192]

# Cold-start order (used only until the autotune cache has an entry for a
# given (sm_version, N, K, M-bucket)). It is M-bucket-aware because no single
# backend wins at every M:
#
#   vllm_marlin_fp8_w8a16 is best at low M but compute-bound: a W8A16
#   weight-only-dequant kernel does not scale to large M the way a native W8A8
#   GEMM does (vLLM's own comment on this kernel says it exists for GPUs
#   *without* native FP8 support), so its step time inverts past M=64.
#   vllm_block_fp8_triton is slower than bf16 at every M.
#
# Timing methodology: eager timing (each call individually launched and timed)
# is dominated at per-layer shapes by ~55 us/call of CPU launch overhead, not
# GPU compute, and can rank backends differently than they perform inside the
# CUDA graph the real decode step always runs in. Every table in this module is
# therefore graph-timed (`bench_gemm.py`'s `cuda_graph_time_fn`/`graph_us`).
# `bench_gemm.py --emit-priority` produces a ready-to-compare
# `DERIVED_BACKEND_PRIORITY_BY_M_BUCKET` from a run's `graph_us`; diff it
# against the tables below and update deliberately, the same way
# `autotune.py`'s cache is consulted at runtime rather than baked in.
# ---------------------------------------------------------------------------
# GRAPH-TIMED priority table for the ``"v4"`` profile, derived by
# `bench_gemm.py --emit-priority`.
#
# These are **whole-model per-decode-step** milliseconds measured under CUDA
# *graph replay*, the metric that matches the real engine (eager timing
# overstated everything by 2-3x at small M). Weight-read floor
# is 6.19 ms (29.71 GB FP8 / 4.8 TB/s).
#
#   M | marlin | flashinfer | CUTLASS blk | scaled_mm | vllm triton | bf16 deq
#   --+--------+------------+-------------+-----------+-------------+---------
#    1|  8.22  |   8.70     |    9.89     |  11.49    |   31.54     | 208.5
#    8|  8.26  |   8.81     |   13.42     |  11.69    |   24.64     | 208.1
#   32|  9.98  |  10.35     |   13.62     |  11.82    |   24.61     | 208.3
#   64| 13.88  |   9.09     |   13.82     |  12.35    |   25.02     | 208.8
#  128| 23.37  |  10.34     |   13.64     |  12.79    |   28.25     | 209.5
#  256| 44.84  |  15.88     |   17.84     |  16.75    |   39.61     | 213.5
#  512| 88.83  |  30.88     |   32.85     |  28.08    |   71.52     | 229.1
#
# So: marlin (W8A16) wins M<=32, flashinfer_fp8_blockscale wins 64..256,
# scaled_mm_pertensor wins at 512, and marlin *inverts* past M=64
# (44.8 ms at 256, 5x its own M=1 time), which is why this is per-bucket and
# not one global order.
#
# Ranking policy, and the two things the raw ranking does not say:
#  1. **Every registered backend appears in every bucket**: the total-order
#     contract `tests/test_gemm.py::TestPriorityForM
#     ::test_every_registered_backend_appears_in_every_bucket` asserts, and
#     what makes `linear()`'s fallback chain exhaustive. The sweep behind this
#     table only produced `graph_us` for six backends; the others are appended
#     at the **tail**, immediately above the bf16 fallbacks (`bf16_dequant`
#     must stay last: it is the always-works correctness fallback, at
#     208 ms/step a backend of last resort in the literal sense):
#       * `deepgemm`: when `is_deep_gemm_supported()` is False it raises
#         `RuntimeError("deepgemm backend unavailable")` and the chain moves
#         on at zero cost.
#       * `vllm_cutlass_fp8_pertensor`: had no `complete` row in that sweep.
#       * `machete_w8a16`: not timed by that sweep.
#     Unmeasured is not the same as slow, but an unmeasured backend must
#     never outrank a measured one, so the tail is where they go until a
#     sweep gives them a number.
#  2. `vllm_block_fp8_cutlass` keeps its measured rank. Its timing predates
#     the scale-layout fixes documented on `_vllm_block_fp8_cutlass`, so its
#     rank among the fallbacks is provisional; it is never #1 in any bucket,
#     so nothing selected by this table depends on it.
#
# Buckets 2, 4 and 16 had no measured row of their own; they inherit the
# next measured bucket up (8, 8, 32 respectively), all of which have the
# identical ordering, so the inheritance is a no-op in practice.
_UNMEASURED_TAIL: List[str] = ["vllm_cutlass_fp8_pertensor", "deepgemm", "machete_w8a16"]
#: The last two entries of every priority list, in this order, always.
#:
#: ``bf16_native`` is a *shape* fallback, not a speed one: it is the only entry
#: that can serve a plain-bf16 (never-quantized) fused weight, and it raises
#: immediately for an ``FP8Tensor``, so putting it here costs an fp8 weight one
#: ``isinstance`` check on a path that has already exhausted eight real
#: backends. ``bf16_dequant`` stays strictly last -- see its own docstring for
#: why 208 ms/step makes it a backend of last resort *for fp8 weights*.
_BF16_TAIL: List[str] = ["bf16_native", "bf16_dequant"]
_MARLIN_FIRST: List[str] = [           # measured best at M in {1, 8, 32}
    "vllm_marlin_fp8_w8a16",
    "flashinfer_fp8_blockscale",
    "vllm_block_fp8_cutlass",
    "scaled_mm_pertensor",
    "vllm_block_fp8_triton",
] + _UNMEASURED_TAIL + _BF16_TAIL
_FLASHINFER_FIRST: List[str] = [       # measured best at M in {64, 128, 256}
    "flashinfer_fp8_blockscale",
    "scaled_mm_pertensor",
    "vllm_block_fp8_cutlass",
    "vllm_marlin_fp8_w8a16",
    "vllm_block_fp8_triton",
] + _UNMEASURED_TAIL + _BF16_TAIL
_SCALED_MM_FIRST: List[str] = [        # measured best at M = 512
    "scaled_mm_pertensor",
    "flashinfer_fp8_blockscale",
    "vllm_block_fp8_cutlass",
    "vllm_block_fp8_triton",
    "vllm_marlin_fp8_w8a16",
] + _UNMEASURED_TAIL + _BF16_TAIL
# marlin degrades monotonically past M=32 (9.98 -> 13.88 -> 23.37 -> 44.84);
# flashinfer takes over at 64. This is the measured crossover, not a guess.
_HIGH_M_THRESHOLD = 64
#: The first graph-timed table, kept verbatim as the ``"v4"`` rollback
#: profile. **No longer the default**: see ``BACKEND_PRIORITY_PROFILES`` below.
V4_BACKEND_PRIORITY_BY_M_BUCKET: Dict[int, List[str]] = {
    1: _MARLIN_FIRST,
    2: _MARLIN_FIRST,
    4: _MARLIN_FIRST,
    8: _MARLIN_FIRST,
    16: _MARLIN_FIRST,
    32: _MARLIN_FIRST,
    64: _FLASHINFER_FIRST,
    128: _FLASHINFER_FIRST,
    256: _FLASHINFER_FIRST,
    512: _SCALED_MM_FIRST,
    # This table predates the prefill-scale buckets, so every bucket above
    # 512 reuses the 512 answer verbatim, faithfully reproducing the
    # clamp-at-512 behavior this rollback profile always had. It keeps
    # `priority_for_m` total without asserting anything new.
    1024: _SCALED_MM_FIRST,
    # The mixed-step buckets 1536/3072 carry the row the value used to round
    # up to, so `priority_for_m` under this rollback profile is unchanged.
    1536: _SCALED_MM_FIRST,
    2048: _SCALED_MM_FIRST,
    3072: _SCALED_MM_FIRST,
    4096: _SCALED_MM_FIRST,
    8192: _SCALED_MM_FIRST,
}

# --------------------------------------------------------------------------- #
# gemm_v7 graph-timed table
# --------------------------------------------------------------------------- #
# Emitted by `bench_gemm.py --emit-priority`. Same metric as the v4 table
# above (**whole-model per-decode-step ms under CUDA-graph replay**), but this
# is the first sweep in which `deepgemm` produced a `complete` row, and it
# changes the answer at every bucket from 32 up:
#
#   M | deepgemm | marlin | flashinfer | scaled_mm | CUTLASS blk | machete
#   --+----------+--------+------------+-----------+-------------+--------
#    1|   8.50   |  8.23  |    8.79    |   11.47   |    9.35     |  9.62
#   32|   8.31   | 10.00  |   10.42    |   11.73   |   13.19     | 10.82
#   64|   8.28   | 13.99  |    9.12    |   12.29   |   13.38     | 12.97
#  128|   9.68   | 23.98  |   10.66    |   12.81   |    13.22    | 14.41
#  256|  14.13   | 45.37  |   16.82    |   16.83   |    17.54    | 22.59
#  512|  26.65   | 89.98  |   31.19    |   27.99   |    32.50    | 43.52
#
# Two things this says, and one it does not:
#  * **DeepGEMM is rank 1 at every M >= 32**, by 20% at 32/64 and by 10-20%
#    at 128-512, and unlike marlin it does not invert. That is what makes it
#    the first pick here.
#  * **marlin still wins M=1** (8.23 vs 8.50, a 3% edge), so buckets 1..16
#    are unchanged. Batch-1 latency is this engine's sharpest constraint and a
#    3% regression there to gain nothing is not a trade worth making.
#  * It does **not** say the *runtime* is faster with deepgemm first. An
#    end-to-end `--gemm-backend deepgemm` run measured B=1 19.77 / B=32
#    31.97 / B=128 38.50 ms/step, worse than the default at B=1 and B=128.
#    Two per-call overheads (`_per_token_group_quant_fp8`'s silent
#    wrong-layout fallback and `_deepgemm`'s per-call failing import, both
#    fixed below) were candidate causes. **If an end-to-end run does not
#    reproduce the kernel-level win, roll back with
#    ``set_backend_priority_profile("v4")`` (or `--gemm-priority v4`)**;
#    that is exactly why the old table is kept rather than deleted.
#
# NOTE (measured, not theoretical): under ``--gemm-weight-cache single``,
# the *serving* default, this reordering is close to a no-op for
# decode. Warmup resolves bucket 1 first, marlin populates `_marlin_cache`,
# and `_cache_policy_allows` then refuses `deepgemm` (a different cache slot)
# for that weight at every larger bucket, so the chain falls through to
# `flashinfer_fp8_blockscale`, which repacks nothing. See
# `_policy_filtered`/`resolve_backend(..., reasons=...)`: the rejection is
# reported instead of silent.
_V7_MARLIN_FIRST: List[str] = [        # M in {1, 2, 4, 8, 16}: marlin 8.23
    "vllm_marlin_fp8_w8a16",
    "deepgemm",
    "flashinfer_fp8_blockscale",
    "vllm_block_fp8_cutlass",
    "machete_w8a16",
    "vllm_cutlass_fp8_pertensor",
    "scaled_mm_pertensor",
    "vllm_block_fp8_triton",
] + _BF16_TAIL
_V7_DEEPGEMM_32: List[str] = [         # M = 32: deepgemm 8.31 vs marlin 10.00
    "deepgemm",
    "vllm_marlin_fp8_w8a16",
    "flashinfer_fp8_blockscale",
    "machete_w8a16",
    "scaled_mm_pertensor",
    "vllm_block_fp8_cutlass",
    "vllm_cutlass_fp8_pertensor",
    "vllm_block_fp8_triton",
] + _BF16_TAIL
_V7_DEEPGEMM_64: List[str] = [         # M = 64: deepgemm 8.28, marlin inverts
    "deepgemm",
    "flashinfer_fp8_blockscale",
    "vllm_cutlass_fp8_pertensor",
    "scaled_mm_pertensor",
    "machete_w8a16",
    "vllm_block_fp8_cutlass",
    "vllm_marlin_fp8_w8a16",
    "vllm_block_fp8_triton",
] + _BF16_TAIL
_V7_DEEPGEMM_128: List[str] = [        # M in {128, 256}
    "deepgemm",
    "flashinfer_fp8_blockscale",
    "scaled_mm_pertensor",
    "vllm_block_fp8_cutlass",
    "vllm_cutlass_fp8_pertensor",
    "machete_w8a16",
    "vllm_marlin_fp8_w8a16",
    "vllm_block_fp8_triton",
] + _BF16_TAIL
_V7_DEEPGEMM_512: List[str] = [        # M = 512 (every prefill chunk)
    "deepgemm",
    "scaled_mm_pertensor",
    "vllm_cutlass_fp8_pertensor",
    "flashinfer_fp8_blockscale",
    "vllm_block_fp8_cutlass",
    "machete_w8a16",
    "vllm_marlin_fp8_w8a16",
    "vllm_block_fp8_triton",
] + _BF16_TAIL
V7_BACKEND_PRIORITY_BY_M_BUCKET: Dict[int, List[str]] = {
    1: _V7_MARLIN_FIRST,
    2: _V7_MARLIN_FIRST,
    4: _V7_MARLIN_FIRST,
    8: _V7_MARLIN_FIRST,
    16: _V7_MARLIN_FIRST,
    32: _V7_DEEPGEMM_32,
    64: _V7_DEEPGEMM_64,
    128: _V7_DEEPGEMM_128,
    256: _V7_DEEPGEMM_128,
    512: _V7_DEEPGEMM_512,
    # v7 predates the prefill-scale buckets, so its answer above 512 was
    # always "whatever M=512 measured", silently (the gap the v8 table below
    # closes). Recorded explicitly here so `priority_for_m` stays total under
    # the rollback v7 profile too, without changing what v7 has ever meant.
    1024: _V7_DEEPGEMM_512,
    1536: _V7_DEEPGEMM_512,   # mixed-step bucket; v7 clamps everything >512 anyway
    2048: _V7_DEEPGEMM_512,
    3072: _V7_DEEPGEMM_512,   # mixed-step bucket
    4096: _V7_DEEPGEMM_512,
    8192: _V7_DEEPGEMM_512,
}

# --------------------------------------------------------------------------- #
# gemm_v8: prefill-scale buckets
# --------------------------------------------------------------------------- #
# Without prefill-scale buckets `m_bucket` clamped at 512, so every prefill
# chunk (up to `RuntimeConfig.max_num_batched_tokens`, 8,192 tokens by
# default) resolved on the table measured at M=512, never validated above it.
# `M_BUCKETS` now carries 1024..8192 so a real chunk gets its own bucket; this
# table is the per-bucket answer for it.
#
# M=2048 IS measured, not a guess (the v7 sweep: whole-model
# per-decode-step-equivalent, CUDA-graph-replayed ms, 4 fused model shapes):
#
#   backend                      | graph_ms | rank
#   vllm_cutlass_fp8_pertensor   |  93.35   |  1   <- NOT deepgemm
#   scaled_mm_pertensor          |  98.02   |  2
#   deepgemm                     | 101.62   |  3
#   flashinfer_fp8_blockscale    | 107.18   |  4
#   vllm_block_fp8_cutlass       | 109.92   |  5
#   machete_w8a16                | 160.17   |  6
#   vllm_block_fp8_triton        | 218.74   |  7
#   bf16_dequant                 | 336.96   |  -   (always last regardless -- _BF16_TAIL)
#   vllm_marlin_fp8_w8a16        | 356.37   |  8   <- WORSE than dequant-every-call
#
# Two things this says. First, at real chunk scale the per-tensor CUTLASS path,
# not deepgemm (the M=32..512 champion), is fastest; routing a chunk through
# the M=512 table (deepgemm first) would leave ~8% on the table without being
# *wrong*. Second, marlin, the M<=16 champion, is the single worst backend
# measured at this scale, slower than dequantizing the whole weight to bf16 on
# every call; a strict prefill chunk on marlin would cost more than the whole
# chunk's budget.
#
# M in {1024, 4096, 8192} are NOT measured in this profile. They use the
# M=2048 order for the five fp8-activation backends: prefill GEMM is
# compute-bound (~98.7% of a chunk's FLOPs), so ranking by arithmetic
# intensity is a property of the *shape*, not a fine-grained function of M
# once M is this large, the same flat-in-M behavior every backend's *accuracy*
# shows. This is an extrapolation, not a measurement, and it is treated as
# one: marlin/machete/vllm_block_fp8_triton are pushed to a slow tail ahead of
# `_BF16_TAIL` **not** because they are assumed to keep losing (this module's
# own policy: "unmeasured is not the same as slow, but must never outrank a
# measured backend") but because they are already measured 3-14x slower than
# the fp8-activation group at BOTH M=512 and M=2048. Demoting an
# already-measured-uncompetitive backend costs nothing, since nothing here
# depends on the guess being right, only on it not being *promoted* over an
# untested claim. The v9 profile below replaces the extrapolated rows with
# per-shape measurements.
_V8_FP8_ACTIVATION_LARGE_M: List[str] = [   # measured at M=2048; extrapolated to 1024/4096/8192
    "vllm_cutlass_fp8_pertensor",
    "scaled_mm_pertensor",
    "deepgemm",
    "flashinfer_fp8_blockscale",
    "vllm_block_fp8_cutlass",
]
_V8_SLOW_TAIL_LARGE_M: List[str] = [        # measured uncompetitive at M=512 AND M=2048 alike
    "machete_w8a16",
    "vllm_block_fp8_triton",
    "vllm_marlin_fp8_w8a16",
]
_V8_LARGE_M: List[str] = _V8_FP8_ACTIVATION_LARGE_M + _V8_SLOW_TAIL_LARGE_M + _BF16_TAIL

V8_BACKEND_PRIORITY_BY_M_BUCKET: Dict[int, List[str]] = dict(V7_BACKEND_PRIORITY_BY_M_BUCKET)
V8_BACKEND_PRIORITY_BY_M_BUCKET.update({
    1024: _V8_LARGE_M,
    # v8 has one order for every prefill bucket, so the mixed-step buckets
    # 1536/3072 get that same order and `m_bucket`'s refinement is a no-op
    # under v8, which is exactly what makes a v8-vs-v9 A/B a test of the
    # *table*, not of the bucket list.
    1536: _V8_LARGE_M,
    2048: _V8_LARGE_M,   # the only one of the four that is actually measured
    3072: _V8_LARGE_M,
    4096: _V8_LARGE_M,
    8192: _V8_LARGE_M,
})


# --------------------------------------------------------------------------- #
# gemm_v9: per-(shape, M) routing for the mixed step
# --------------------------------------------------------------------------- #
# **What is wrong with every table above, stated precisely.** All of them are
# keyed on M alone. But `bench_gemm.py --emit-priority`, which produced them,
# ranks backends by a **whole-model per-decode-step** total -- a `count`-weighted
# sum over the six fused shapes. So a bucket's "winner" is the backend that wins
# the *sum*, and the sum at prefill scale is dominated by `mlp_gate_up_proj`
# ([34816, 5120], x64) and `mlp_down_proj` ([5120, 17408], x64), which together
# are ~70% of a step's GEMM FLOPs. A backend that loses those two by 5% and wins
# `gdn_out_proj` ([5120, 6144], x48) by 40% is ranked last and never runs
# anywhere. That is the structural reason a per-M table leaves money on the
# table, and it is why this profile is keyed on **(shape, M)**, not M.
#
# The second thing v9 fixes is the bucket list itself: see `M_BUCKETS` above.
# The mixed step at chunk 1,024 + 128/256 rows presents M = 1,152/1,280 and at
# chunk 2,048 + 256 rows M = 2,304 -- values that rounded up to 2048 and 4096
# under the old list, i.e. routed on a table measured 1.6-1.8x above them.
#
# **Shape classes.** Routing keys on the literal fused ``(N, K)``, because that
# (with M) is the entire input to a GEMM's tile/config choice; the class names
# below exist so the table reads like the model rather than like a shape sheet.
# Note `gdn_out_proj` and `attn_o_proj` are the *same* GEMM ([5120, 6144]) --
# one entry serves both, which is a property of this checkpoint, not a
# simplification.
SHAPE_CLASS_BY_NK: Dict[Tuple[int, int], str] = {
    (34816, 5120): "mlp_gate_up",        # fused gate_proj + up_proj, x64
    (5120, 17408): "mlp_down",           # x64
    (16384, 5120): "gdn_in_proj_qkvz",   # fused in_proj_qkv + in_proj_z, x48
    (5120, 6144): "out_proj",            # gdn_out_proj (x48) AND attn_o_proj (x16)
    (14336, 5120): "attn_qkv",           # fused q/k/v, x16
    (248320, 5120): "lm_head",           # x1
    (96, 5120): "gdn_in_proj_ba",        # x48, bf16 in the checkpoint
}


def shape_class(n: Optional[int], k: Optional[int]) -> Optional[str]:
    """The routing class for a fused ``(N, K)``, or ``None`` if unknown.

    ``None`` is the honest answer for a shape this project has never measured
    (a different checkpoint, a test's toy weight, the MTP head's own copies):
    the caller then falls back to the M-only table, which is what every profile
    before v9 would have done anyway. An unmeasured shape must never pick up a
    measured shape's answer by accident."""
    if n is None or k is None:
        return None
    return SHAPE_CLASS_BY_NK.get((int(n), int(k)))


#: ``{shape_class: {m_bucket: order}}``: the v9 overrides. A ``(class,
#: bucket)`` that is absent falls through to ``V9_BACKEND_PRIORITY_BY_M_BUCKET``
#: below, so this table only has to carry the cells a sweep actually measured.
#: Emitted by `bench_gemm.py --emit-shape-priority` (per-cell `graph_us`, not
#: the whole-model total; that difference is the whole point of this profile).
V9_BACKEND_PRIORITY_BY_SHAPE_AND_M: Dict[str, Dict[int, List[str]]] = {
    # x64, 34816x5120 -- with mlp_down, ~70% of a step's GEMM FLOPs.
    #: Bucket 2048 is the one that matters in a served step >= 2048 tokens: the
    #: MLP is tiled at `RuntimeConfig.mlp_tile_tokens` (2048), so these two
    #: shapes never see the mixed step's M above that.
    'mlp_gate_up': {
        1536: ['vllm_cutlass_fp8_pertensor', 'deepgemm', 'scaled_mm_pertensor', 'flashinfer_fp8_blockscale', 'vllm_block_fp8_cutlass'],
        2048: ['deepgemm', 'scaled_mm_pertensor', 'vllm_cutlass_fp8_pertensor', 'flashinfer_fp8_blockscale', 'vllm_block_fp8_cutlass'],
        3072: ['vllm_cutlass_fp8_pertensor', 'scaled_mm_pertensor', 'deepgemm', 'flashinfer_fp8_blockscale', 'vllm_block_fp8_cutlass'],
        8192: ['deepgemm', 'scaled_mm_pertensor', 'flashinfer_fp8_blockscale', 'vllm_cutlass_fp8_pertensor', 'vllm_block_fp8_cutlass'],
    },
    # x64, 5120x17408 -- tiled at mlp_tile_tokens like mlp_gate_up
    'mlp_down': {
        1536: ['deepgemm', 'scaled_mm_pertensor', 'vllm_cutlass_fp8_pertensor', 'vllm_block_fp8_cutlass', 'flashinfer_fp8_blockscale'],
        2048: ['vllm_cutlass_fp8_pertensor', 'scaled_mm_pertensor', 'flashinfer_fp8_blockscale', 'vllm_block_fp8_cutlass', 'deepgemm'],
        3072: ['deepgemm', 'scaled_mm_pertensor', 'vllm_cutlass_fp8_pertensor', 'vllm_block_fp8_cutlass', 'flashinfer_fp8_blockscale'],
        8192: ['deepgemm', 'scaled_mm_pertensor', 'flashinfer_fp8_blockscale', 'vllm_cutlass_fp8_pertensor', 'vllm_block_fp8_cutlass'],
    },
    # x48, 16384x5120 -- untiled, so it DOES see the mixed step's M
    'gdn_in_proj_qkvz': {
        1536: ['deepgemm', 'vllm_cutlass_fp8_pertensor', 'scaled_mm_pertensor', 'flashinfer_fp8_blockscale', 'vllm_block_fp8_cutlass'],
        2048: ['deepgemm', 'vllm_cutlass_fp8_pertensor', 'scaled_mm_pertensor', 'flashinfer_fp8_blockscale', 'vllm_block_fp8_cutlass'],
        3072: ['vllm_cutlass_fp8_pertensor', 'deepgemm', 'scaled_mm_pertensor', 'flashinfer_fp8_blockscale', 'vllm_block_fp8_cutlass'],
        8192: ['deepgemm', 'scaled_mm_pertensor', 'vllm_cutlass_fp8_pertensor', 'flashinfer_fp8_blockscale', 'vllm_block_fp8_cutlass'],
    },
    # x48 (gdn_out_proj) + x16 (attn_o_proj), 5120x6144 -- one GEMM, two call sites
    'out_proj': {
        1536: ['deepgemm', 'vllm_cutlass_fp8_pertensor', 'scaled_mm_pertensor', 'flashinfer_fp8_blockscale', 'vllm_block_fp8_cutlass'],
        2048: ['vllm_cutlass_fp8_pertensor', 'scaled_mm_pertensor', 'flashinfer_fp8_blockscale', 'vllm_block_fp8_cutlass', 'deepgemm'],
        3072: ['deepgemm', 'scaled_mm_pertensor', 'vllm_cutlass_fp8_pertensor', 'flashinfer_fp8_blockscale', 'vllm_block_fp8_cutlass'],
        8192: ['deepgemm', 'scaled_mm_pertensor', 'flashinfer_fp8_blockscale', 'vllm_cutlass_fp8_pertensor', 'vllm_block_fp8_cutlass'],
    },
    # x16, 14336x5120
    'attn_qkv': {
        1536: ['vllm_cutlass_fp8_pertensor', 'scaled_mm_pertensor', 'deepgemm', 'vllm_block_fp8_cutlass', 'flashinfer_fp8_blockscale'],
        2048: ['vllm_cutlass_fp8_pertensor', 'deepgemm', 'scaled_mm_pertensor', 'flashinfer_fp8_blockscale', 'vllm_block_fp8_cutlass'],
        3072: ['deepgemm', 'vllm_cutlass_fp8_pertensor', 'scaled_mm_pertensor', 'flashinfer_fp8_blockscale', 'vllm_block_fp8_cutlass'],
        8192: ['deepgemm', 'scaled_mm_pertensor', 'vllm_cutlass_fp8_pertensor', 'flashinfer_fp8_blockscale', 'vllm_block_fp8_cutlass'],
    },
    # x1, 248320x5120 -- runs at M = the step's logits rows, not at the
    #: chunk's M; kept because the sweep measured it and a row costs nothing
    'lm_head': {
        1536: ['vllm_cutlass_fp8_pertensor', 'deepgemm', 'scaled_mm_pertensor', 'vllm_block_fp8_cutlass', 'flashinfer_fp8_blockscale'],
        2048: ['vllm_cutlass_fp8_pertensor', 'deepgemm', 'scaled_mm_pertensor', 'vllm_block_fp8_cutlass', 'flashinfer_fp8_blockscale'],
        3072: ['vllm_cutlass_fp8_pertensor', 'deepgemm', 'scaled_mm_pertensor', 'vllm_block_fp8_cutlass', 'flashinfer_fp8_blockscale'],
        8192: ['deepgemm', 'scaled_mm_pertensor', 'flashinfer_fp8_blockscale', 'vllm_cutlass_fp8_pertensor', 'vllm_block_fp8_cutlass'],
    },
}
# What that table says, and the two things it does NOT say:
#
#  * The winner changes with the shape at the same M. At bucket 1536:
#    `deepgemm` wins mlp_down / gdn_in_proj_qkvz / out_proj, and
#    `vllm_cutlass_fp8_pertensor` wins mlp_gate_up / attn_qkv / lm_head. A
#    single-backend-per-M table cannot express that, which is why this profile
#    exists, but the *size* of the win is small (0.0-2.8% of a step's GEMM
#    time over the best single backend). Per-shape routing is real, and it is
#    not the lever.
#  * **`flashinfer_fp8_blockscale` is 4th or 5th of five at every shape and
#    every bucket**, by 9.9-16.5% aggregate, and it is exactly what the
#    served configuration runs, because it is the only fast backend that
#    repacks nothing and therefore the only one `--gemm-weight-cache single`
#    leaves reachable once marlin has claimed each weight's one cache slot at
#    M=1. That, not tiling, is the reachable GEMM lever, and it is a *policy*
#    question (`--gemm-cache-owner`), not a table one.
#  * It does NOT say a GEMM is less efficient at the mixed step's M. Measured
#    aggregate best-achievable rate is 1,120 / 1,120 / 1,175 / 1,149 / 1,128 /
#    1,147 TFLOP/s at M = 1152 / 1280 / 1536 / 2304 / 4352 / 8448: flat to
#    within 5% across a 7.3x range of M. A 1,024-row GEMM is not intrinsically
#    a worse GEMM.
#
# M=4352 was benched and ranks slightly differently from M=8448
# (`vllm_cutlass_fp8_pertensor` edges deepgemm by 0.05% at the aggregate), but
# both land in bucket 8192 and the emitter keys a bucket on the largest M in it,
# so bucket 8192 carries M=8448's order. The difference is under 0.1% and a
# seventh bucket is not worth its cost.

#: v9's shape-agnostic fallback: v8's table, verbatim. Two consequences worth
#: stating. (1) A shape with no v9 row routes exactly as it does today, so
#: switching the profile can only change a cell this project has measured per
#: shape. (2) `priority_for_m(m)` -- the shapeless API every older caller and
#: test uses -- keeps returning a total order over every registered backend
#: under v9 too.
V9_BACKEND_PRIORITY_BY_M_BUCKET: Dict[int, List[str]] = dict(V8_BACKEND_PRIORITY_BY_M_BUCKET)


#: Named, swappable cold-start tables. ``"v8"`` is the default (buckets <=512
#: byte-identical to v7; 1024-8192 are new). ``"v7"``/``"v4"`` are rollbacks;
#: `TestBackendPriorityProfiles` covers switching between them. ``"v9"`` adds a
#: *second*, shape-keyed layer on top of its M-only entry here; see
#: `BACKEND_PRIORITY_SHAPE_PROFILES`.
BACKEND_PRIORITY_PROFILES: Dict[str, Dict[int, List[str]]] = {
    "v9": V9_BACKEND_PRIORITY_BY_M_BUCKET,
    "v8": V8_BACKEND_PRIORITY_BY_M_BUCKET,
    "v7": V7_BACKEND_PRIORITY_BY_M_BUCKET,
    "v4": V4_BACKEND_PRIORITY_BY_M_BUCKET,
}
#: The shape-keyed overlay, per profile. Every profile before v9 has an empty
#: overlay, which is what makes them literally unchanged rather than merely
#: intended-to-be-unchanged: `priority_for` consults this dict first and finds
#: nothing, then falls through to `BACKEND_PRIORITY_PROFILES` exactly as
#: `priority_for_m` always did.
BACKEND_PRIORITY_SHAPE_PROFILES: Dict[str, Dict[str, Dict[int, List[str]]]] = {
    "v9": V9_BACKEND_PRIORITY_BY_SHAPE_AND_M,
    "v8": {},
    "v7": {},
    "v4": {},
}
_priority_profile: str = "v8"
DEFAULT_BACKEND_PRIORITY_BY_M_BUCKET: Dict[int, List[str]] = V8_BACKEND_PRIORITY_BY_M_BUCKET
DEFAULT_BACKEND_PRIORITY_BY_SHAPE_AND_M: Dict[str, Dict[int, List[str]]] = {}


def set_backend_priority_profile(name: str) -> str:
    """Select the cold-start priority table; returns the previous profile name.

    Process-wide for the same reason the accuracy mode and the weight-cache
    policy are: the choice must be identical at every call site of a weight
    shared between them, and constant across a CUDA-graph capture."""
    global _priority_profile, DEFAULT_BACKEND_PRIORITY_BY_M_BUCKET
    global DEFAULT_BACKEND_PRIORITY_BY_SHAPE_AND_M
    global DEFAULT_BACKEND_PRIORITY, _LOW_M_PRIORITY, _HIGH_M_PRIORITY
    if name not in BACKEND_PRIORITY_PROFILES:
        raise ValueError(
            f"gemm priority profile must be one of {tuple(BACKEND_PRIORITY_PROFILES)}, got {name!r}"
        )
    prev, _priority_profile = _priority_profile, name
    table = BACKEND_PRIORITY_PROFILES[name]
    DEFAULT_BACKEND_PRIORITY_BY_M_BUCKET = table
    DEFAULT_BACKEND_PRIORITY_BY_SHAPE_AND_M = BACKEND_PRIORITY_SHAPE_PROFILES.get(name, {})
    DEFAULT_BACKEND_PRIORITY = table[1]
    _LOW_M_PRIORITY = table[1]
    _HIGH_M_PRIORITY = table[128]
    return prev


def get_backend_priority_profile() -> str:
    return _priority_profile


# Backward-compatible flat default (e.g. for callers/tests that just want
# "a" reasonable order, not per-M tuning): the low-M list, since batch-1
# latency is this engine's sharpest constraint.
DEFAULT_BACKEND_PRIORITY: List[str] = _V7_MARLIN_FIRST
# kept as aliases so older imports/tests keep resolving
_LOW_M_PRIORITY = _V7_MARLIN_FIRST
_HIGH_M_PRIORITY = _V7_DEEPGEMM_128


def priority_for_m(m: int) -> List[str]:
    """The cold-start backend try-order for a given (unbucketed) M.

    ``M`` is interpreted **per sequence** (see ``rows_per_sequence`` below):
    outside that scope this is the identity, inside it a ``B*n``-row verify
    window routes exactly like the ``B``-row decode step it replaces.

    Under ``gemm_accuracy == "strict"`` the order is additionally filtered to
    the backends whose measured ``relL2`` clears ``STRICT_REL_L2_MAX`` -- see
    the accuracy block below. ``"fast"`` (the default) is the identity, so
    nothing about the historical behaviour changes unless a caller opts in.
    """
    return priority_for(m)


def priority_for(m: int, n: Optional[int] = None, k: Optional[int] = None) -> List[str]:
    """The cold-start try-order for ``(M, N, K)`` -- the shape-aware form.

    Resolution, in order:

      1. the active profile's **shape overlay** for ``(shape_class(N, K),
         m_bucket(M))``, if it has one -- this is the only thing v9 adds and
         the only way a routing decision can differ between two shapes at the
         same M;
      2. the profile's M-only table for that bucket -- what every profile
         before v9 has always returned, and what v9 returns for any shape or
         bucket its sweep did not cover.

    A shape overlay is allowed to be *partial* (a short list, e.g. only the
    five fp8-activation backends a large-M sweep actually benches). The rest of
    the M-only order is appended behind it, deduplicated, so the return value is
    still a **total order over every registered backend** -- which is what makes
    `linear()`'s fallback chain exhaustive, and what
    `test_every_registered_backend_appears_in_every_bucket` asserts. An overlay
    can therefore only ever *promote* what it measured; it can never delete a
    fallback.

    ``priority_for_m(m)`` is this function with the shape unknown.
    """
    bucket = m_bucket(sequence_m(m))
    base = DEFAULT_BACKEND_PRIORITY_BY_M_BUCKET[bucket]
    cls = shape_class(n, k)
    if cls is not None:
        head = DEFAULT_BACKEND_PRIORITY_BY_SHAPE_AND_M.get(cls, {}).get(bucket)
        if head:
            seen = set()
            order: List[str] = []
            for name in list(head) + list(base):
                if name in seen:
                    continue
                seen.add(name)
                order.append(name)
            return _accuracy_filtered(order)
    return _accuracy_filtered(base)


# --------------------------------------------------------------------------- #
# accuracy-aware routing
# --------------------------------------------------------------------------- #
# `DEFAULT_BACKEND_PRIORITY_BY_M_BUCKET` above was ranked on **speed alone**.
# Measured against an exact reference, the backend it picks for every M >= 64
# is an order of magnitude less accurate than the one it picks below 64,
# because the two are not the same *class* of kernel.
#
# There are exactly two classes here, and the split is structural, not a
# tuning difference:
#
#   W8A16 -- activations stay bf16; only the weight is fp8/int8 and it is
#            dequantized inside the kernel. The error floor is bf16's own
#            representation error on the activations, measured 2.7e-3 across
#            every shape and every M on this checkpoint.
#   W8A8  -- the kernel quantizes the activations to fp8-e4m3 first. e4m3 has
#            3 mantissa bits, i.e. ~6% relative ulp / ~3.6e-2 rms relative
#            error per element, and *no* scale granularity fixes that (the
#            error is mantissa-bound, not range-bound). Measured 2.6e-2 for
#            every fp8-activation backend regardless of 1x128 vs per-tensor
#            scaling. This is a floor for the class, not an implementation
#            defect: `deepgemm` and `flashinfer_fp8_blockscale` land on the
#            same number because they are literally the same kernel family.
#
# So an "accuracy-aware priority" is not a re-ranking, it is a *class* choice,
# and the table below records the measured number for each backend rather than
# its class alone so the threshold can be moved with evidence.
#
# ``relL2`` here is ``||y_backend - y_ref||_2 / ||y_ref||_2`` where ``y_ref``
# is an exact **fp32 dequant** of the block-128 fp8 checkpoint weight times an
# fp32 copy of the activation -- i.e. it measures a backend's error against
# what this engine's own weights *mean*, not against the original bf16 model
# (the checkpoint's own fp8 quantization is a separate, larger ~3.6e-2 term
# that every backend here shares and none of them can fix).
#
# Every number below is the **worst** cell measured by `gemm_numerics.py` over
# {4 fused shapes x random weights} u {3 real checkpoint tensors} x
# M in {1, 32, 64, 128, 256, 512, 2048}: worst, not mean, because this table
# gates a threshold test and a conservative bound is the only useful kind.
#
# The three-tier structure is the whole finding, and it is *flat in M and in
# shape* -- the spread within a tier is ~5%, the gap between tiers is 3-10x:
BACKEND_REL_L2: Dict[str, float] = {
    # --- exact: no quantization anywhere in the path ---------------------- #
    "bf16_native": 0.0,               # Only reachable for a weight that was
                                      # never quantized (`in_proj_ba`, `mtp.fc`,
                                      # `lm_head`): `F.linear(x_bf16, w_bf16)`, i.e.
                                      # bit-identical to what the checkpoint stores.
                                      # 0.0 is not an approximation -- there is no
                                      # quantization step for it to be an error of.
    # --- W8A16, bf16 activations: the floor is bf16's own 2.3e-3 ---------- #
    "bf16_dequant": 2.4e-3,           # 2.28e-3..2.36e-3. The reference, modulo bf16 acts.
    "vllm_marlin_fp8_w8a16": 3.5e-3,  # 2.28e-3..3.46e-3 (the top of the range is
                                      # M>=128, where `use_fp32_reduce` turns off)
    "machete_w8a16": 7.4e-3,          # 7.17e-3..7.34e-3 -- ALL of it the int8
                                      # re-quantization this vLLM's Machete forces
                                      # (no fp8 b_type); see `_machete_repacked`
    # --- W8A8, fp8-e4m3 activations: mantissa-bound, ~2.6e-2, unfixable --- #
    "flashinfer_fp8_blockscale": 2.7e-2,  # 2.47e-2..2.69e-2
    "deepgemm": 2.7e-2,               # 2.47e-2..2.69e-2 -- IDENTICAL to flashinfer,
                                      # which it should be: flashinfer's blockscale
                                      # GEMM is DeepGEMM-backed. 1x128
                                      # activation scaling does not move it
                                      # toward the W8A16 class, because
                                      # e4m3's 3 mantissa bits set the error,
                                      # not the scale's dynamic range.
    "vllm_block_fp8_cutlass": 2.7e-2,  # 2.47e-2..2.69e-2, only after the
                                      # scale_b layout fix; it measured 0.10
                                      # (random) / **1.27** (real weights) before
    "vllm_block_fp8_triton": 2.7e-2,  # 2.47e-2..2.69e-2
    # --- W8A8 + per-TENSOR weight scale: strictly worse than block-scaled - #
    "scaled_mm_pertensor": 3.9e-2,    # 3.75e-2..3.88e-2. This is what M-bucket
                                      # 512 -- i.e. every prefill chunk -- runs on.
    "vllm_cutlass_fp8_pertensor": 3.9e-2,  # 3.77e-2..3.88e-2
}
#: Structural class of each backend, independent of any measurement: does the
#: kernel quantize the *activation*? ``"bf16"`` == W8A16, ``"fp8"`` == W8A8.
#: This is what ``"strict"`` really means; ``BACKEND_REL_L2`` is the evidence.
BACKEND_ACT_DTYPE: Dict[str, str] = {
    "bf16_native": "bf16",
    "bf16_dequant": "bf16",
    "vllm_marlin_fp8_w8a16": "bf16",
    "machete_w8a16": "bf16",
    "flashinfer_fp8_blockscale": "fp8",
    "deepgemm": "fp8",
    "vllm_block_fp8_cutlass": "fp8",
    "vllm_block_fp8_triton": "fp8",
    "scaled_mm_pertensor": "fp8",
    "vllm_cutlass_fp8_pertensor": "fp8",
}
#: The bar a backend must clear in ``"strict"`` mode.
#:
#: **Why 5e-3.** It is set just above the measured W8A16 floor (2.7e-3, the
#: cost of bf16 activations, which no backend here beats) and an order
#: of magnitude below the W8A8 class (2.6e-2), so it separates the two classes
#: with a wide margin on both sides rather than slicing through either. It is
#: deliberately *not* a quality target derived from an eval (no eval in this
#: tree covers the split); it is the "same-arithmetic-as-the-decode-step" bar
#: that speculative decoding needs.
#:
#: **What 2.6e-2 costs, as far as anything here has measured.** Two separate
#: facts, and they say different things:
#:   * *Determinism*: pushed through this model's **real** bf16 ``lm_head``
#:     over 4,096 hidden states, a relative perturbation of
#:     the final hidden state flips the greedy argmax at:
#:
#:         1e-3   0.29%  (1 in 341)      5e-3   1.05%  (1 in  95)
#:         2.7e-3 0.61%  (1 in 164)      7e-3   1.66%  (1 in  60)
#:         1e-2   2.44%  (1 in  41)      2.6e-2 6.23%  (1 in  16)
#:
#:     i.e. the W8A8 class changes one greedy token in 16, the W8A16 class one
#:     in 164, and the threshold sits at one in 95. At the W8A8 rate a
#:     speculative verify-vs-decode equivalence check fails on most prompts
#:     within the first dozen or so tokens. (A synthetic Gaussian logit row
#:     underestimates these rates by ~2x, because it lacks the real lm_head's
#:     near-tie density.)
#:   * *Quality*: **unmeasured.** A flipped near-tie is not by itself a worse
#:     token: the checkpoint's own fp8 quantization already disagrees with
#:     the bf16 model on 1.9% of argmaxes at unchanged eval
#:     scores. Nothing in this tree has run GSM8K/IFEval with the M>=64 path
#:     pinned to each class, so the honest statement is "we know it changes
#:     the output, we do not know that it degrades it". Until such an eval
#:     exists this threshold governs *reproducibility*, which is
#:     a real requirement on its own (spec decode, and any A/B whose decode
#:     and prefill must agree).
STRICT_REL_L2_MAX: float = 5e-3
GEMM_ACCURACY_MODES: Tuple[str, ...] = ("fast", "strict")
_gemm_accuracy: str = "fast"


def set_gemm_accuracy(mode: str) -> str:
    """Set the process-wide GEMM accuracy mode; returns the previous one.

    ``"fast"`` (default, unchanged behaviour): rank on speed only.
    ``"strict"``: only backends with ``BACKEND_REL_L2 <= STRICT_REL_L2_MAX``
    may be resolved. Process-wide for the same reason the weight-cache policy
    is: the decision has to be identical at every call site of a weight that
    is shared across them, and it must be constant across a CUDA-graph
    capture."""
    global _gemm_accuracy
    if mode not in GEMM_ACCURACY_MODES:
        raise ValueError(f"gemm accuracy must be one of {GEMM_ACCURACY_MODES}, got {mode!r}")
    prev, _gemm_accuracy = _gemm_accuracy, mode
    return prev


def get_gemm_accuracy() -> str:
    return _gemm_accuracy


@contextmanager
def gemm_accuracy(mode: str):
    """Scoped :func:`set_gemm_accuracy` (tests, one-off comparisons)."""
    prev = set_gemm_accuracy(mode)
    try:
        yield
    finally:
        set_gemm_accuracy(prev)


def backend_rel_l2(name: str) -> float:
    """Measured relL2 for ``name``. Unknown backends are treated as *failing*
    the strict bar (``inf``) -- a backend nobody has measured must never be
    admitted by a mode whose entire purpose is a measured accuracy bound."""
    return BACKEND_REL_L2.get(name, float("inf"))


def accurate_backends(threshold: Optional[float] = None) -> List[str]:
    """Registered backends whose measured relL2 clears ``threshold``."""
    thr = STRICT_REL_L2_MAX if threshold is None else threshold
    return [n for n in _BACKENDS if backend_rel_l2(n) <= thr]


def _accuracy_allows(name: str) -> bool:
    """May backend ``name`` be *resolved* at all under the current mode?"""
    return _gemm_accuracy == "fast" or backend_rel_l2(name) <= STRICT_REL_L2_MAX


def _accuracy_filtered(order: List[str]) -> List[str]:
    """Apply the current accuracy mode to a try-order.

    ``bf16_dequant`` is always appended: it is both the most accurate backend
    available (it is the reference, modulo bf16 activations) and the one that
    always works, so it keeps :func:`linear`'s fallback chain non-empty in
    every mode. It is last, so it only ever runs if everything above it fails."""
    if _gemm_accuracy == "fast":
        return order
    keep = [n for n in order if backend_rel_l2(n) <= STRICT_REL_L2_MAX]
    if "bf16_dequant" not in keep:
        keep.append("bf16_dequant")
    return keep


_BACKENDS: Dict[str, BackendFn] = {}


def register(name: str):
    def deco(fn: BackendFn) -> BackendFn:
        _BACKENDS[name] = fn
        return fn

    return deco


def available_backends() -> List[str]:
    return list(_BACKENDS.keys())


def m_bucket(m: int) -> int:
    """Round ``m`` up to the next autotune/bench bucket (clamps at the top)."""
    for b in M_BUCKETS:
        if m <= b:
            return b
    return M_BUCKETS[-1]


# --------------------------------------------------------------------------- #
# rows-per-sequence scope
# --------------------------------------------------------------------------- #
# Every routing decision in this module is keyed on **M, the number of rows in
# the activation**, on the assumption that M is the batch size: that is what
# ``DEFAULT_BACKEND_PRIORITY_BY_M_BUCKET`` was measured against, and it is why
# marlin (W8A16) is picked at M<=32 and ``flashinfer_fp8_blockscale`` at M>=64.
#
# A speculative-decoding *verify* pass breaks that assumption. It is standing in
# for a decode step of ``B`` sequences, but it packs ``n = k+1`` token rows per
# sequence, so it presents ``M = B*n`` to every linear. At B=32/k=3 that is
# M=128, which crosses the M=64 threshold and silently swaps **256 of the
# model's 305 linears** from marlin to ``flashinfer_fp8_blockscale``.
#
# That is not a tuning difference, it is a *precision* difference. Measured
# against an exact fp32 dequant reference on the real ``[14336, 5120]`` /
# ``[5120, 17408]`` shapes:
#
#     vllm_marlin_fp8_w8a16       relL2 = 2.7e-3     (true W8A16: bf16 acts)
#     flashinfer_fp8_blockscale   relL2 = 2.6e-2     (10x worse -- it quantizes
#                                                     the activations to fp8,
#                                                     contrary to its docstring)
#     marlin vs flashinfer        relL2 = 2.6e-2
#
# A 2.6e-2 relative perturbation of the logits flips a greedy argmax roughly
# once every 38 tokens over a 248,320 vocabulary, which is enough to make
# speculative greedy output diverge from non-speculative greedy output on most
# prompts within a dozen or so tokens.
#
# So: a forward that packs several token rows per sequence must resolve its
# backend at the **sequence count**, not the row count. ``rows_per_sequence(n)``
# declares that multiplicity for the duration of a forward; ``sequence_m(M)``
# divides it back out. Both are pure host-side Python: the value is constant
# for the whole of a captured region, so a CUDA-graph capture bakes in the same
# kernel choice every replay uses.
_ROWS_PER_SEQUENCE = 1


def rows_per_sequence_now() -> int:
    """How many activation rows the current forward packs per *sequence*."""
    return _ROWS_PER_SEQUENCE


def set_rows_per_sequence(n: int) -> int:
    """Set the multiplicity; returns the previous value (for manual restore)."""
    global _ROWS_PER_SEQUENCE
    prev = _ROWS_PER_SEQUENCE
    _ROWS_PER_SEQUENCE = max(1, int(n))
    return prev


@contextmanager
def rows_per_sequence(n: int):
    """Scope in which every M-keyed routing decision divides ``M`` by ``n``.

    Use it around a forward that packs ``n`` token rows per sequence (a
    speculative verify window) so it runs the *same* kernels as the plain
    decode step of the same batch that it must reproduce.
    """
    prev = set_rows_per_sequence(n)
    try:
        yield
    finally:
        set_rows_per_sequence(prev)


def sequence_m(m: int) -> int:
    """``m`` re-expressed as a per-sequence row count (>= 1)."""
    return max(1, int(m) // _ROWS_PER_SEQUENCE)


# --------------------------------------------------------------------------- #
# weight-repack cache policy
# --------------------------------------------------------------------------- #
# Several backends below keep a **permanent, per-weight, full-size**
# repacked copy of the weight on the device, memoised on the ``FP8Tensor``
# itself so the repack is paid once rather than once per step:
#
#   vllm_marlin_fp8_w8a16      -> `_marlin_cache`     (~1.00x the FP8 weight)
#   scaled_mm_pertensor        -> `_pertensor_cache`  (~1.00x)
#   vllm_cutlass_fp8_pertensor -> `_pertensor_cache`  (same slot as above)
#   deepgemm                   -> `_deepgemm_cache`   (~1.00x)
#
# That memoisation is correct and is a real speedup, but it is *invisible to
# every memory plan in this tree*, and the M-bucketed dispatch means a single
# server naturally touches more than one of them: warmup resolves the decode
# buckets (M<=32 -> marlin) and then the first prefill chunk resolves M-bucket
# 512 (-> scaled_mm_pertensor). On Qwen3.8-27B that is 23.0 GiB of FP8 linear
# weights duplicated *twice*: 46 GiB of unaccounted device memory, enough to
# OOM a loaded server. See `repack_cache_bytes`.
#
# The policy below bounds it. It is a *process-wide* setting because the caches
# live on the weight objects, which are shared by every call site:
#
#   "multi"  -- historical behaviour: any backend may materialise its cache.
#               Offline microbenchmarks (bench_gemm.py) want this; they sweep
#               one shape at a time and care about kernel time, not footprint.
#   "single" -- at most ONE repack cache per weight. Once a weight has a cache,
#               backends that would allocate a *different* one are skipped in
#               resolution, so the next-best cache-free backend is used instead.
#               This is the serving default (runtime.serve): it keeps marlin at
#               the decode buckets (the M=1 latency win) and pushes
#               prefill onto `flashinfer_fp8_blockscale`, which repacks nothing.
#   "none"   -- no backend may materialise a repack cache. Frees the marlin copy
#               too (~23 GiB on 27B) at the cost of ~0.5 ms/step at M=1.
_CACHE_ATTR_BY_BACKEND: Dict[str, str] = {
    "vllm_marlin_fp8_w8a16": "_marlin_cache",
    "scaled_mm_pertensor": "_pertensor_cache",
    "vllm_cutlass_fp8_pertensor": "_pertensor_cache",
    "deepgemm": "_deepgemm_cache",
    # Same size class as marlin's (1 B/elt qweight + bf16 [K/128, N]
    # group scales == 1.016x the fp8 weight), and it takes its OWN slot rather
    # than sharing marlin's, which is what makes `--gemm-weight-cache single`
    # do the right thing automatically: a weight may hold marlin's copy *or*
    # machete's, never both, so adding this backend does not add a third
    # 23 GiB to the memory plan. See `repack_cache_bytes`.
    "machete_w8a16": "_machete_cache",
}
_CACHE_ATTRS: Tuple[str, ...] = (
    "_marlin_cache", "_pertensor_cache", "_deepgemm_cache", "_machete_cache",
)
WEIGHT_CACHE_POLICIES: Tuple[str, ...] = ("multi", "single", "none")
_weight_cache_policy: str = "multi"


def set_weight_cache_policy(policy: str) -> str:
    """Set the process-wide repack-cache policy; returns the previous one."""
    global _weight_cache_policy
    if policy not in WEIGHT_CACHE_POLICIES:
        raise ValueError(
            f"weight cache policy must be one of {WEIGHT_CACHE_POLICIES}, got {policy!r}"
        )
    prev, _weight_cache_policy = _weight_cache_policy, policy
    return prev


def get_weight_cache_policy() -> str:
    return _weight_cache_policy


def repack_cache_bytes(w: WeightT) -> int:
    """Device bytes currently held by ``w``'s backend repack caches.

    The number every memory plan needs and none of them had. Counts the
    populated cache slots only, so it is 0 before warmup and stable after."""
    total = 0
    for attr in _CACHE_ATTRS:
        cached = getattr(w, attr, None)
        if cached is None:
            continue
        for t in cached:
            if hasattr(t, "numel") and hasattr(t, "element_size"):
                total += int(t.numel()) * int(t.element_size())
    return total


def free_repack_caches(w: WeightT) -> int:
    """Drop every repack cache on ``w``; returns the bytes released.

    Not called on the hot path -- this is for a caller that has already
    resolved its backends and wants the other backends' copies back (or a
    test that wants a clean slate)."""
    freed = repack_cache_bytes(w)
    for attr in _CACHE_ATTRS:
        if getattr(w, attr, None) is not None:
            setattr(w, attr, None)
    return freed


def _cache_policy_allows(name: str, w: WeightT) -> bool:
    """May backend ``name`` be *resolved* for weight ``w`` under the policy?

    ``"single"`` only blocks a backend whose cache slot is not the one this
    weight already filled -- re-using an already-populated slot is free."""
    attr = _CACHE_ATTR_BY_BACKEND.get(name)
    if attr is None or _weight_cache_policy == "multi":
        return True
    if _weight_cache_policy == "none":
        return getattr(w, attr, None) is not None
    # "single"
    if getattr(w, attr, None) is not None:
        return True
    return not any(getattr(w, other, None) is not None for other in _CACHE_ATTRS)


def _policy_filtered(order: List[str], w: WeightT) -> List[str]:
    if _weight_cache_policy == "multi":
        return order
    keep = [name for name in order if _cache_policy_allows(name, w)]
    # `bf16_dequant` allocates no persistent cache and always works, so the
    # chain can never be emptied by the filter; assert that invariant cheaply.
    return keep or list(order)


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
_ptgq_fp8: Optional[Tuple[Optional[Callable], frozenset]] = None


def _per_token_group_quant_fp8_entry() -> Tuple[Optional[Callable], frozenset]:
    """Resolve vLLM's ``per_token_group_quant_fp8`` **once**, with the set of
    keyword arguments this build actually accepts.

    Memoized because :func:`_deepgemm` and the two block-fp8 backends call this
    on every one of the model's ~256 fp8 linears, every step, where a
    ``from vllm... import`` inside a try on each call is measurable overhead."""
    global _ptgq_fp8
    if _ptgq_fp8 is not None:
        return _ptgq_fp8
    import inspect

    try:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            per_token_group_quant_fp8,
        )
    except Exception:  # noqa: BLE001 -- vLLM missing/renamed: use the torch path
        _ptgq_fp8 = (None, frozenset())
        return _ptgq_fp8
    try:
        params = inspect.signature(per_token_group_quant_fp8).parameters
        accepted = frozenset(
            name
            for name, p in params.items()
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        )
        if any(p.kind is p.VAR_KEYWORD for p in params.values()):
            accepted = frozenset()  # **kwargs: cannot tell, assume it handles them
            _ptgq_fp8 = (per_token_group_quant_fp8, accepted)
            return _ptgq_fp8
    except (TypeError, ValueError):  # C extension with no introspectable signature
        accepted = frozenset()
    _ptgq_fp8 = (per_token_group_quant_fp8, accepted)
    return _ptgq_fp8


def _per_token_group_quant_fp8(x: "torch.Tensor", block: int = 128, **kwargs):
    """``vllm``'s per-token-group (1x128) activation quantizer, with a pure
    torch fallback for the plain row-major-fp32-scale case.

    This is on `deepgemm`'s hot path, and the obvious shape for it::

        try:
            from vllm... import per_token_group_quant_fp8
            return per_token_group_quant_fp8(x, block, **kwargs)
        except Exception:
            <pure-torch, row-major fp32 scales, kwargs ignored>

    hides three separate problems, all of them invisible from the outside:

    1. **The fallback ignores the layout kwargs.** ``_deepgemm`` asks for
       ``column_major_scales=True, tma_aligned_scales=True, use_ue8m0=...``
       because ``fp8_gemm_nt`` reads the scale buffer against TMA-aligned
       column-major strides. A row-major fp32 ``[M, K/128]`` answer is not
       slower, it is *wrong*: bounded, finite, no assertion, every element
       multiplied by the wrong block's scale. A ``TypeError`` from an
       unsupported kwarg after a vLLM version bump would silently switch
       DeepGEMM from "correct" to "quietly garbage" with no log line anywhere.
    2. **`except Exception` also swallows real kernel failures** (an OOM, a
       CUDA error, an assert inside the Triton quantizer) and answers them
       with ~9 extra eager kernels' worth of torch ops instead. At 256 fp8
       linears/step that is ~2,300 extra launches per decode step hidden
       behind a bare except.
    3. The ``from vllm...`` import would run on every call.

    So the entry point is resolved once, and the two failure modes are handled
    differently on purpose, because they are not the same risk:

    * **vLLM's function exists but does not accept a kwarg the caller passed.**
      This is the dangerous one, because it is what a version bump produces and
      because a real fp8 kernel *is* about to consume the result. It raises,
      which the dispatcher's fallback chain turns into "try the next backend":
      the correct outcome, and a visible one.
    * **vLLM is not importable at all.** Then there is no fp8 GEMM on this host
      to consume the scales in the first place: ``_deepgemm`` raises at
      ``_deepgemm_available()`` long before reaching here, and every other fp8
      backend imports vLLM directly. The only callers that get this far are the
      CPU test suite and ``bench_gemm``'s quant-only microbenchmark, both of
      which want the torch quantizer. So the fallback stays, but it *honours*
      what it can rather than silently dropping it:
      ``column_major_scales`` (a real stride change) and ``use_ue8m0``
      (round each scale up to a power of two, which is what e8m0 means) are
      implemented; ``tma_aligned_scales`` is a padding/alignment hint for a TMA
      kernel that by construction is not present here, and is the one thing
      ignored, noisily, once."""
    import torch

    fn, accepted = _per_token_group_quant_fp8_entry()
    if fn is not None:
        unsupported = [k for k in kwargs if accepted and k not in accepted]
        if unsupported:
            raise RuntimeError(
                f"per_token_group_quant_fp8 in this vLLM build does not accept "
                f"{unsupported} (accepts {sorted(accepted)}); refusing to fall back to a "
                f"layout the caller did not ask for -- see this function's docstring"
            )
        return fn(x, block, **kwargs)

    column_major = bool(kwargs.pop("column_major_scales", False))
    use_ue8m0 = bool(kwargs.pop("use_ue8m0", False))
    tma_aligned = bool(kwargs.pop("tma_aligned_scales", False))
    if kwargs:
        raise RuntimeError(
            f"vllm's per_token_group_quant_fp8 is unavailable and the pure-torch fallback "
            f"does not implement {sorted(kwargs)}"
        )
    if tma_aligned:
        _warn_once(
            "per_token_group_quant_fp8: vllm is unavailable, so the pure-torch fallback is "
            "running and `tma_aligned_scales=True` is being IGNORED (it is a padding hint for "
            "a TMA kernel that cannot be present on a host without vllm). Correct for the CPU "
            "test/microbench callers this path exists for; never reachable from a real fp8 GEMM."
        )
    FP8_MAX = 448.0
    m, k = x.shape
    groups = k // block
    xg = x.float().view(m, groups, block)
    amax = xg.abs().amax(dim=-1).clamp(min=1e-8)
    scale = amax / FP8_MAX
    if use_ue8m0:
        # e8m0 scales are powers of two, no mantissa. Round *up* so the
        # quantized values still fit in e4m3's range.
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    xq = (xg / scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).view(m, k)
    if column_major:
        # What `.t().contiguous().t()` means and why it is not a no-op: the
        # caller asked for stride(0) == 1. `.contiguous()` *after* the second
        # transpose would undo it.
        scale = scale.t().contiguous().t()
    return xq, scale


_warned: set = set()


def _warn_once(msg: str) -> None:
    if msg in _warned:
        return
    _warned.add(msg)
    import warnings

    warnings.warn(msg, RuntimeWarning, stacklevel=3)


def _fix_cuda_home_link_layout(cuda_home: str) -> None:
    """Best-effort: repair two link-layout gaps that break FlashInfer's
    ``fp8_blockscale_gemm_sm90`` JIT link step (``cannot find -lcudart``)
    even when ``nvcc`` itself works. The pip-installed
    ``nvidia-cuda-nvcc``/``nvidia-cu13`` wheel layout ships versioned shared
    objects (``libcudart.so.13``) and a ``lib64`` (not ``lib``) directory,
    but the linker invoked by the JIT looks for unversioned
    ``-lcudart``/``-lcublas``/``-lcublasLt`` under ``lib``. This creates the
    missing links (``libcudart.so -> libcudart.so.13`` + cublas/cublasLt,
    ``lib -> lib64``) so a fresh venv does not need a manual step. Every
    operation is wrapped individually and silently skipped on failure
    (read-only filesystem, no matching versioned file, already-correct
    layout, ...): this is a convenience, never a requirement for
    ``_ensure_deepgemm_cuda_home`` to succeed."""
    import glob
    import os

    lib64 = os.path.join(cuda_home, "lib64")
    lib = os.path.join(cuda_home, "lib")
    try:
        if os.path.isdir(lib64) and not os.path.exists(lib):
            os.symlink(lib64, lib)
    except Exception:
        pass

    for lib_dir in (lib, lib64):
        if not os.path.isdir(lib_dir):
            continue
        for base_name in ("libcudart.so", "libcublas.so", "libcublasLt.so"):
            target = os.path.join(lib_dir, base_name)
            if os.path.exists(target):
                continue
            versioned = sorted(glob.glob(os.path.join(lib_dir, base_name + ".*")))
            if versioned:
                try:
                    os.symlink(os.path.basename(versioned[0]), target)
                except Exception:
                    pass


def _ensure_deepgemm_cuda_home() -> None:
    """Best-effort: if ``CUDA_HOME`` isn't set, point it at the cu13 nvcc
    wheel bundled in this venv, so DeepGEMM's (and FlashInfer's
    DeepGEMM-backed ``fp8_blockscale_gemm_sm90``) JIT compiler matches
    torch's cu13 build instead of falling back to a stale or absent system
    nvcc (an nvcc/torch CUDA-version mismatch breaks the JIT). The wheel is
    typically importable as ``nvidia.cu13`` (e.g.
    ``<venv>/lib/python3.10/site-packages/nvidia/cu13``) rather than
    ``nvidia.cuda_nvcc``; this probes both module names plus a raw
    ``sys.path`` filesystem scan so it is not tied to either layout. Also
    runs :func:`_fix_cuda_home_link_layout` on whichever ``CUDA_HOME`` ends
    up set (found or pre-existing); see that function's docstring for the
    ``cannot find -lcudart`` JIT-link failure it repairs."""
    import glob
    import os
    import sys

    if os.environ.get("CUDA_HOME"):
        _fix_cuda_home_link_layout(os.environ["CUDA_HOME"])
        return

    candidates: List[str] = []
    for modname in ("nvidia.cu13", "nvidia.cuda_nvcc"):
        try:
            mod = __import__(modname, fromlist=["_"])
            candidates.append(os.path.dirname(os.path.abspath(mod.__file__)))
        except Exception:
            pass
    for base in sys.path:
        candidates.extend(glob.glob(os.path.join(base, "nvidia", "cu13")))
        candidates.extend(glob.glob(os.path.join(base, "nvidia", "cuda_nvcc")))

    for cand in candidates:
        nvcc = os.path.join(cand, "bin", "nvcc")
        if os.path.exists(nvcc):
            os.environ["CUDA_HOME"] = cand
            os.environ["PATH"] = os.path.join(cand, "bin") + os.pathsep + os.environ.get("PATH", "")
            _fix_cuda_home_link_layout(cand)
            return


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #
@register("bf16_native")
def _bf16_native(x: "torch.Tensor", w: WeightT) -> "torch.Tensor":
    """``F.linear`` on a weight that was **never quantized**.

    This exists to split one name into two, because the two halves are three
    orders of magnitude apart and the engine's resolved-backend table could
    not otherwise tell them apart:

      * ``bf16_dequant`` on an ``FP8Tensor`` materializes the full bf16 weight
        *inside every call*: 208 ms/step whole-model, a correctness fallback
        and a serious regression if the engine ever lands on it silently.
      * ``bf16_dequant`` on a **plain bf16 tensor** does nothing of the sort:
        `w.dtype == x.dtype`, so the "dequant" is a no-op and the call is a
        single `F.linear` on a weight that is already resident in the dtype the
        GEMM wants. It is not a fallback, it is *the* right kernel, and it is
        the only one that can run at all.

    Without this backend, 49 of the model's 305 linears report `bf16_dequant`
    in every M-bucket in both accuracy modes, which reads as "49 layers fell
    off a cliff" and is in fact "49 layers are bf16 in the checkpoint". They
    are the 48 GDN ``in_proj_ba`` ``[96, 5120]`` weights (N=96 is not a
    multiple of 128, so `fused_weights.py` deliberately keeps them
    unquantized) plus the bf16 ``lm_head`` ``[248320, 5120]``. Their combined
    weight traffic is 2.59 GB/step (47.2 MB for all 48 `in_proj_ba` + 2.54 GB
    for `lm_head`), i.e. ~0.6 ms at 4.3 TB/s, essentially all of it `lm_head`;
    see `TestBf16NativeIsNotTheDequantFallback` for the arithmetic pinned as a
    test. That is why quantizing `lm_head` (`quantize_lm_head`) is the only
    one of the two worth anything.

    With this backend registered, ``bf16_dequant`` appearing in a resolved
    table means what it always should have meant: **an fp8 weight found no fp8
    backend**, which is a defect. `runtime/bench_runtime.py::
    check_resolved_backends` asserts exactly that."""
    if isinstance(w, FP8Tensor):
        raise TypeError(
            "bf16_native requires a plain (never-quantized) bf16 weight; this is an "
            "FP8Tensor -- use one of the fp8 backends, or bf16_dequant as a last resort"
        )
    import torch.nn.functional as F

    return F.linear(x, w if w.dtype == x.dtype else w.to(x.dtype))


@register("bf16_dequant")
def _bf16_dequant(x: "torch.Tensor", w: WeightT) -> "torch.Tensor":
    import torch.nn.functional as F

    if isinstance(w, FP8Tensor):
        wd = w.dequant(out_dtype=x.dtype)
    else:
        wd = w if w.dtype == x.dtype else w.to(x.dtype)
    return F.linear(x, wd)


@register("vllm_block_fp8_triton")
def _vllm_block_fp8_triton(x: "torch.Tensor", w: WeightT) -> "torch.Tensor":
    if not isinstance(w, FP8Tensor):
        raise TypeError("vllm_block_fp8_triton requires an FP8Tensor weight")
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        w8a8_triton_block_scaled_mm,
    )

    x_fp8, x_scale = _per_token_group_quant_fp8(x, 128)
    return w8a8_triton_block_scaled_mm(
        x_fp8, w.weight, x_scale, w.scale_inv.float(), [128, 128], output_dtype=x.dtype
    )


@register("vllm_block_fp8_cutlass")
def _vllm_block_fp8_cutlass(x: "torch.Tensor", w: WeightT) -> "torch.Tensor":
    """Block-scaled FP8 via ``vllm._custom_ops.cutlass_scaled_mm``.

    The layout contract follows vLLM 0.28.0's own production consumer of this
    op (``vllm/model_executor/kernels/linear/scaled_mm/cutlass.py``'s
    ``CutlassFp8BlockScaledMMKernel``), which builds its activation quantizer
    as ``QuantFP8(..., use_ue8m0=False, column_major_scales=True)`` (no
    ``tma_aligned_scales``: that flag is DeepGEMM-specific, see the
    ``deepgemm`` backend below) and calls
    ``ops.cutlass_scaled_mm(A, B.T, out_dtype=..., scale_a=As, scale_b=Bs.T)``.
    Every operand except ``A`` must be **column-major**, and each one fails in
    its own way if it is not:

      * ``b`` (the weight): transposing a contiguous ``[N, K]`` weight gives a
        column-major ``[K, N]`` view (``stride(0) == 1``) for free, but
        ``.contiguous()`` on it forces row-major layout back
        (``stride(0) == N``). ``scaled_mm_entry.cu``
        (``csrc/libtorch_stable/quantization/w8a8/cutlass/``) rejects that
        with ``STD_TORCH_CHECK(b.stride(0) == 1);  // Column-major``, a check
        with no message string, so the raised error is empty.
      * ``scale_a`` (the activation scale): must come from
        ``per_token_group_quant_fp8(..., column_major_scales=True)``. A
        row-major ``[M, K/128]`` scale still satisfies the documented
        ``scale_a.shape * [1, 128] == a.shape`` broadcast rule, so nothing
        asserts or crashes, but the raw CUDA kernel underneath
        ``torch.ops._C.cutlass_scaled_mm`` reads the buffer assuming the
        column-major layout vLLM's own producer always uses, so every element
        is multiplied by the wrong block's scale: a silent, bounded error of
        ~20% relL2 on some shapes.
      * ``scale_b`` (the weight scale): vLLM passes ``Bs.T``, a *transposed
        view* of the ``[N/128, K/128]`` scale, i.e. column-major
        ``[K/128, N/128]``. ``.t().contiguous()`` re-materializes row-major
        and gives the same silent class of error, measured at relL2 0.10 on
        random weights and **1.27** on the real checkpoint's
        ``layers.0.mlp.down_proj``, against 2.6e-2 for every other
        fp8-activation backend. A loose fixed threshold (e.g. 0.15) lets the
        0.10 case pass; comparing backends to each other exposes it.

    This is a *fallback* backend (never rank 1 in any M-bucket), but it sits
    high in every bucket's fallback chain, so a layout error here would turn a
    host where the faster backends fail into fluent-looking garbage.
    """
    if not isinstance(w, FP8Tensor):
        raise TypeError("vllm_block_fp8_cutlass requires an FP8Tensor weight")
    from vllm import _custom_ops as ops

    # column_major_scales=True: the flag vLLM's own
    # CutlassFp8BlockScaledMMKernel passes to QuantFP8/per_token_group_quant_fp8
    # for this op (see the docstring). use_ue8m0 is left at its default
    # (False); tma_aligned_scales is NOT passed (DeepGEMM-only).
    x_fp8, x_scale = _per_token_group_quant_fp8(x, 128, column_major_scales=True)
    # cutlass_scaled_mm computes (scale_a * a) @ (scale_b * b) with "group"
    # broadcast: scale_a.shape * [1, 128] == a.shape, scale_b.shape *
    # [128, 128] == b.shape. `a` is x [M, K]; `b` must be [K, N], i.e. the
    # weight transposed, and MUST stay column-major (scaled_mm_entry.cu
    # checks `b.stride(0) == 1`), so no `.contiguous()` on `b`.
    # `scale_b` likewise: vLLM's own caller passes `Bs.T`, a transposed VIEW of
    # the [N/128, K/128] scale, so the kernel reads it column-major.
    # `.contiguous()` here is a silent 1.27-relL2 error, not a harmless
    # densification.
    b = w.weight.t()  # [K, N], column-major by construction -- do not force contiguous
    scale_b = w.scale_inv.float().t()  # [K/128, N/128] column-major VIEW -- no .contiguous()
    return ops.cutlass_scaled_mm(x_fp8, b, x_scale, scale_b, x.dtype)


def _scaled_fp8_quant_pertensor(x: "torch.Tensor"):
    """Dynamic per-tensor FP8 quantization via vLLM's single fused CUDA
    kernel (``torch.ops._C.dynamic_scaled_fp8_quant``, wrapped by
    ``vllm._custom_ops.scaled_fp8_quant``), with a slow-but-correct pure
    torch fallback. **Perf note**: computing the per-tensor scale with
    separate eager ops (``.abs()``, ``.amax()``, ``.clamp()``, ``.to()``,
    ``/``, ``.clamp()``, ``.to(fp8)``) costs one CUDA kernel launch each. At
    ~257 GEMM invocations per decode step, ~7 extra kernel launches/call x
    ~10-15us launch overhead each is ~20-27ms of pure launch overhead, enough
    to make a per-tensor backend slower than the bf16 baseline even though
    the GEMM itself is tiny and bandwidth-cheap at these shapes.
    ``scaled_fp8_quant`` does the reduction + quantize in one (or two) fused
    kernel launches instead."""
    try:
        from vllm import _custom_ops as ops

        return ops.scaled_fp8_quant(x, scale=None)  # dynamic per-tensor
    except Exception:
        import torch

        FP8_MAX = 448.0
        xf = x.float()
        amax = xf.abs().amax().clamp(min=1e-8)
        scale = (amax / FP8_MAX).to(torch.float32).reshape(1)
        xq = (xf / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        return xq, scale


@register("scaled_mm_pertensor")
def _scaled_mm_pertensor(x: "torch.Tensor", w: WeightT) -> "torch.Tensor":
    """``torch._scaled_mm`` with a single scale for the whole weight tensor
    and a single scale for the whole activation tile, re-derived from the
    block-128 scales. **Documented accuracy hit**: collapsing a 128x128
    grid to one number can be a large relative error on a weight whose
    per-block magnitude varies a lot; this exists as a big-M / cuBLASLt-class
    fallback ("torch._scaled_mm / cuBLASLt for large M"), not as a default.
    Best used on wide, roughly-uniform-magnitude weights such as lm_head.

    Layout: cuBLASLt's scaled_mm requires operand A row-major, operand B
    column-major. ``x_fp8`` (produced by ``scaled_fp8_quant``, which writes
    into a fresh contiguous output tensor) is row-major, correct as-is. The
    weight side must be transposed to ``[K, N]`` *and stay column-major*:
    do **not** call ``.contiguous()`` after ``.t()``: that re-materializes
    row-major layout and reintroduces "Only multiplication of row-major and
    column-major matrices is supported by cuBLASLt". A transpose
    of an already-contiguous ``[N, K]`` tensor is, by construction, exactly
    column-major ``[K, N]``: no copy needed, just don't undo it.

    Perf: activation *and* weight-side per-tensor quantization both go
    through ``_scaled_fp8_quant_pertensor`` (one fused kernel launch each)
    instead of a chain of eager ops; see that helper's docstring for the
    launch-overhead reasoning.
    """
    import torch

    if not hasattr(torch, "_scaled_mm"):
        raise RuntimeError("this torch build has no torch._scaled_mm")

    if isinstance(w, FP8Tensor):
        if w._pertensor_cache is None:
            wd = w.dequant(out_dtype=torch.bfloat16)  # one-time, amortized across all calls
            wq, wscale = _scaled_fp8_quant_pertensor(wd)
            # NOTE: `.t()` only -- see the layout note in this function's
            # docstring. `wq` is a fresh contiguous [N, K] tensor here, so
            # `wq.t()` is exactly column-major [K, N] with no copy.
            w._pertensor_cache = (wq.t(), wscale)
        w_fp8_t, w_scale = w._pertensor_cache
    else:
        wb = w if w.dtype in (torch.bfloat16, torch.float16) else w.to(torch.bfloat16)
        wq, w_scale = _scaled_fp8_quant_pertensor(wb)
        w_fp8_t = wq.t()  # same layout note as above -- no `.contiguous()`

    x_fp8, x_scale = _scaled_fp8_quant_pertensor(x)  # row-major, correct as-is

    return torch._scaled_mm(x_fp8, w_fp8_t, scale_a=x_scale, scale_b=w_scale, out_dtype=x.dtype)


@register("vllm_cutlass_fp8_pertensor")
def _vllm_cutlass_fp8_pertensor(x: "torch.Tensor", w: WeightT) -> "torch.Tensor":
    """CUTLASS ``cutlass_scaled_mm`` with per-tensor (not block) scales, a
    candidate for faster FP8 GEMV at M=1. Same underlying op as
    ``vllm_block_fp8_cutlass`` (so it follows that backend's column-major
    weight layout), but with `[1,1]`/`[1]` per-tensor scales instead of the
    1x128/128x128 block-scale tensors. ``cutlass_scaled_mm``'s docstring
    documents ordinary NumPy-style broadcasting for extent-1 scales, so this
    is a legal, supported call shape, and CUTLASS's own kernel selection may
    pick a different (small-M-tuned) tile config for a plain per-tensor scale
    than for the block-scale path. Same documented accuracy hit as
    ``scaled_mm_pertensor`` (single scale per tensor, not per 128x128 block)."""
    if not isinstance(w, FP8Tensor):
        raise TypeError("vllm_cutlass_fp8_pertensor requires an FP8Tensor weight")
    import torch

    from vllm import _custom_ops as ops

    if w._pertensor_cache is None:
        wd = w.dequant(out_dtype=torch.bfloat16)
        wq, wscale = _scaled_fp8_quant_pertensor(wd)
        w._pertensor_cache = (wq.t(), wscale)  # column-major, no `.contiguous()` -- see above
    w_fp8_t, w_scale = w._pertensor_cache

    x_fp8, x_scale = _scaled_fp8_quant_pertensor(x)
    return ops.cutlass_scaled_mm(x_fp8, w_fp8_t, x_scale, w_scale, x.dtype)


_deepgemm_availability: Optional[Tuple[bool, str]] = None


def _ensure_deepgemm_use_env() -> None:
    """Force ``VLLM_USE_DEEP_GEMM`` on before DeepGEMM availability is probed.

    ``vllm.utils.deep_gemm.is_deep_gemm_supported()`` is::

        envs.VLLM_USE_DEEP_GEMM and has_deep_gemm() and is_supported_arch

    ``envs.VLLM_USE_DEEP_GEMM`` is a *lazy* per-access property
    (``vllm/envs.py``'s ``environment_variables`` dict:
    ``"VLLM_USE_DEEP_GEMM": lambda: bool(int(os.getenv("VLLM_USE_DEEP_GEMM",
    "1")))``, default **on**), not a hardware/toolchain gate. A shell
    environment that exports ``VLLM_USE_DEEP_GEMM=0`` (commonly done to keep a
    stock vLLM server from JIT-compiling DeepGEMM against a mismatched CUDA
    toolchain) makes ``is_deep_gemm_supported()`` resolve False for this
    dispatcher too, even though nothing is wrong with the kernel.

    Because the lookup is lazy (not resolved or cached at import time),
    setting ``os.environ["VLLM_USE_DEEP_GEMM"] = "1"`` in-process, even after
    ``import vllm.envs``, before the first call flips it to ``True``, so the
    override is safe to do here rather than in every launch script. The other
    two terms: ``has_deep_gemm()`` checks ``_has_module("deep_gemm") or
    _has_module("vllm.third_party.deep_gemm")``, and vLLM 0.28.0 vendors its
    own copy at ``vllm.third_party.deep_gemm`` (compiled ``_C*.so`` present),
    so the standalone ``deep_gemm`` PyPI package does not need to be
    installed. H200 is ``sm_90`` (Hopper,
    ``torch.cuda.get_device_capability(0) == (9, 0)``), so
    ``current_platform.support_deep_gemm()`` (the third AND-term) is True.

    **Ordering hazard**: ``is_deep_gemm_supported()`` is itself
    ``@functools.cache``d inside vLLM: once *anything* in the process
    calls it, the result is fixed for the process's lifetime regardless of
    later env changes. This function must therefore run before any other code
    path could have already resolved it False (and
    :func:`_deepgemm_available` also clears that cache). Launch scripts that
    want DeepGEMM should export ``VLLM_USE_DEEP_GEMM=1`` as well, so the env
    is correct from process start, not just from this in-process override.

    Only sets the var if it is unset or exactly ``"0"``; an explicit non-"0"
    override some caller set on purpose is left alone."""
    import os

    if os.environ.get("VLLM_USE_DEEP_GEMM", "0") == "0":
        os.environ["VLLM_USE_DEEP_GEMM"] = "1"


def _deepgemm_available() -> Tuple[bool, str]:
    """Resolve-time, memoized DeepGEMM availability check (see
    :func:`_ensure_deepgemm_use_env` for the env gate it corrects).
    Re-doing the ``vllm.utils.deep_gemm`` import and
    ``is_deep_gemm_supported()`` probe on *every call* would make every caller
    (the dispatcher's fallback chain, every shape/M cell in
    ``TestDispatchBackendsOnGPU``) pay that cost again instead of getting a
    single clear answer resolved once per process. Computed once and cached at
    module scope; never raises itself: always returns ``(available, reason)``
    so callers (this module's ``_deepgemm`` and any future dispatcher-side
    pre-filtering) get a clear, immediate answer."""
    global _deepgemm_availability
    if _deepgemm_availability is not None:
        return _deepgemm_availability

    try:
        _ensure_deepgemm_use_env()
        _ensure_deepgemm_cuda_home()
        from vllm.utils import deep_gemm as _dg

        is_deep_gemm_supported = _dg.is_deep_gemm_supported
        # **The `functools.cache` ordering hazard is real, and it fires.** A
        # cold process can report `deepgemm` unavailable while the same host in
        # the same env answers `is_deep_gemm_supported() -> True` when that is
        # the first thing the process asks. The difference is purely who got
        # there first: `is_deep_gemm_supported` is `@functools.cache`d, and
        # importing `fp8_utils` (which `vllm_block_fp8_triton` and
        # `vllm_block_fp8_cutlass` both do, and which can happen before
        # `deepgemm` is probed) calls it while `VLLM_USE_DEEP_GEMM` may still
        # be "0", freezing False for the life of the process before
        # `_ensure_deepgemm_use_env` above ever runs.
        #
        # So the env fix alone is not sufficient: the cache has to be dropped
        # so the (now-corrected) env is actually re-read. `cache_clear` is the
        # documented `functools.cache` API and this is the only place in the
        # tree that touches it; clearing it is safe because the function is a
        # pure read of env + platform, so re-evaluating it can only ever give
        # a *more* correct answer.
        if hasattr(is_deep_gemm_supported, "cache_clear"):
            is_deep_gemm_supported.cache_clear()
        if is_deep_gemm_supported():
            _deepgemm_availability = (True, "")
        else:
            _deepgemm_availability = (
                False,
                "vllm.utils.deep_gemm.is_deep_gemm_supported() is False on this host "
                "even with VLLM_USE_DEEP_GEMM=1 forced, likely a real hardware/"
                "toolchain gate (see _ensure_deepgemm_use_env docstring "
                "for the env-var gate this already rules out)",
            )
    except Exception as exc:  # noqa: BLE001 -- import/probe itself can fail
        _deepgemm_availability = (False, f"{type(exc).__name__}: {exc}")
    return _deepgemm_availability


_deepgemm_entrypoints: Optional[Tuple[Callable, Optional[Callable]]] = None


def _deepgemm_entry() -> Tuple[Callable, Optional[Callable]]:
    """``(fp8_gemm_nt, should_use_deepgemm_for_fp8_linear_or_None)``, resolved
    **once** per process.

    ``_deepgemm`` runs on every one of the model's 256 fp8 linears, every
    step. Resolving these per call would cost:

      * ``from vllm.utils.deep_gemm import fp8_gemm_nt``: a `sys.modules`
        hit, but still a full import statement plus attribute lookup; and
      * ``try: from ...fp8_utils import should_use_deepgemm_for_fp8_linear
        except ImportError:``, where that name **does not exist** in vLLM
        0.28.0. The steady state would be *raising, propagating and catching
        a Python ImportError from the import machinery* once per linear per
        step: the single most expensive way to compute a constant.

    Neither cost shows up under CUDA-graph *replay*, so graph-timed kernel
    benchmarks look clean. Both are paid in full during every eager warmup,
    every capture, every `resolve_backend` probe, every non-graphed
    (`--no-graphs`) step, and the whole of the prefill path (`prefill_forward`
    is never captured). Resolved once, both become a tuple unpack."""
    global _deepgemm_entrypoints
    if _deepgemm_entrypoints is not None:
        return _deepgemm_entrypoints
    from vllm.utils.deep_gemm import fp8_gemm_nt

    # A defensive restatement of a gate our shapes already satisfy (every
    # fused N, K here is a multiple of 128). Absent in vLLM 0.28.0; `None`
    # means "use the inlined gate", which is what that helper checks anyway.
    try:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            should_use_deepgemm_for_fp8_linear,
        )
    except ImportError:
        should_use_deepgemm_for_fp8_linear = None  # type: ignore[assignment]
    _deepgemm_entrypoints = (fp8_gemm_nt, should_use_deepgemm_for_fp8_linear)
    return _deepgemm_entrypoints


def _deepgemm_repacked(w: FP8Tensor) -> Tuple["torch.Tensor", "torch.Tensor", bool]:
    """One-time DeepGEMM weight repack + TMA-aligned scale layout, cached on
    the ``FP8Tensor``. Ports vLLM's real call site
    (``model_executor/kernels/linear/scaled_mm/deep_gemm.py
    ::process_weights_after_loading``) from an ``nn.Module`` layer onto a
    bare weight/scale pair."""
    if w._deepgemm_cache is not None:
        return w._deepgemm_cache

    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        deepgemm_post_process_fp8_weight_block,
    )
    from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used

    use_e8m0 = is_deep_gemm_e8m0_used()
    # `deepgemm_post_process_fp8_weight_block` mutates its `wq` argument
    # in place when use_e8m0=True (via requant_weight_ue8m0_inplace) -- clone
    # so we never touch the canonical checkpoint tensor.
    wq = w.weight.clone()
    ws = w.scale_inv.float().clone()
    dg_weight, dg_scale = deepgemm_post_process_fp8_weight_block(
        wq=wq, ws=ws, quant_block_shape=(BLOCK, BLOCK), use_e8m0=use_e8m0
    )
    w._deepgemm_cache = (dg_weight, dg_scale, use_e8m0)
    return w._deepgemm_cache


@register("deepgemm")
def _deepgemm(x: "torch.Tensor", w: WeightT) -> "torch.Tensor":
    """Real DeepGEMM integration via vLLM's own wrapper
    (``vllm.utils.deep_gemm.fp8_gemm_nt``), matching vLLM's production call
    site (``model_executor/kernels/linear/scaled_mm/deep_gemm.py``):
    one-time weight repack (``deepgemm_post_process_fp8_weight_block``,
    cached on the ``FP8Tensor``) + per-call TMA-aligned/column-major
    per-token-group activation quant + ``fp8_gemm_nt``. Requires DeepGEMM's
    CUDA extension to be importable/JIT-able on this host.

    Availability (``VLLM_USE_DEEP_GEMM`` env fix + ``CUDA_HOME`` setup +
    ``is_deep_gemm_supported()``) is resolved once via the memoized
    :func:`_deepgemm_available`; see :func:`_ensure_deepgemm_use_env` for why
    the env var has to be forced and for the ``functools.cache`` ordering
    hazard involved. If it resolves unavailable (e.g. a genuinely unsupported
    GPU), this backend raises immediately (before touching the weight repack
    or activation quant), the dispatcher's fallback chain (``linear()``) moves
    on to the next backend in ``priority_for_m(m)``, and
    ``TestDispatchBackendsOnGPU``'s ``_run_backend`` catches the raise and
    calls ``self.skipTest(...)``, so the GPU test suite reports this as a
    skip, not a failure.

    **Small-M path**: this DeepGEMM build has no dedicated small-M / SWAP_AB
    entry point for our shapes analogous to ``flashinfer_fp8_blockscale``'s
    auto-SwapAB under M<32. In the vendored source
    (``vllm/third_party/deep_gemm/include/...``) ``swap_ab`` appears only
    inside the ``sm100_*`` (Blackwell) epilogue files; H200 is ``sm_90``
    (Hopper), so that path never compiles/dispatches here. The other
    small-M-shaped candidates, ``fp8_m_grouped_gemm_nt_masked`` /
    ``m_grouped_fp8_gemm_nt_contiguous``, are DeepGEMM's *grouped* GEMM entry
    points for MoE expert routing (masked/contiguous multi-expert batching);
    this model (Qwen3.8-27B, hybrid GDN+attention) has no MoE layers, so
    there is nothing to group and these do not apply. ``fp8_gemm_nt`` is
    DeepGEMM's single dense-GEMM entry point on Hopper; any small-M
    tile-config selection happens inside the compiled kernel/JIT, not via a
    separate Python-level call this dispatcher needs to choose between.
    ``should_use_deepgemm_for_fp8_linear``'s shape gate (``N % 64 == 0``,
    ``K % 128 == 0``, bf16 output) is applied below as a defensive check;
    every qwenfast fused shape already satisfies it (N, K are multiples of
    128), so this is expected to be a no-op, not a filter that changes
    behavior."""
    if not isinstance(w, FP8Tensor):
        raise TypeError("deepgemm requires an FP8Tensor weight")
    import torch

    available, reason = _deepgemm_available()
    if not available:
        raise RuntimeError(f"deepgemm backend unavailable: {reason}")

    fp8_gemm_nt, shape_gate = _deepgemm_entry()

    n, k = w.weight.shape
    ok = (
        bool(shape_gate(x.dtype, (n, k), supports_deep_gemm=True))
        if shape_gate is not None
        else ((n % 64 == 0) and (k % 128 == 0) and x.dtype is torch.bfloat16)
    )
    if not ok:
        raise RuntimeError(
            f"deepgemm backend unavailable: shape gate rejected N={n} K={k} "
            f"dtype={x.dtype} (needs N%64==0, K%128==0, bf16 output)"
        )

    dg_weight, dg_scale, use_e8m0 = _deepgemm_repacked(w)
    x_fp8, x_scale = _per_token_group_quant_fp8(
        x, BLOCK, column_major_scales=True, tma_aligned_scales=True, use_ue8m0=use_e8m0
    )
    m = x.shape[0]
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    fp8_gemm_nt((x_fp8, x_scale), (dg_weight, dg_scale), out, is_deep_gemm_e8m0_used=use_e8m0)
    return out


def _marlin_repacked(w: FP8Tensor, n: int, k: int) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """One-time GPTQ-style repack + scale permutation for the unified Marlin
    kernel, cached on the ``FP8Tensor``. Ported directly from vLLM's
    ``marlin_utils_fp8.prepare_fp8_layer_for_marlin`` (which operates on a
    live ``nn.Module`` layer via ``getattr``/``replace_parameter``) onto a
    bare ``(weight [N,K], scale_inv [N/128,K/128])`` pair -- the
    ``size_k_first=False`` branch of that function, since our fused weights
    are always stored ``[N, K]``. Every qwenfast fused-GEMM (N, K) is a
    multiple of 128 in both dims, which is also a multiple of
    Marlin's 64/128 tile requirement, so the ``padded_n == n`` / ``padded_k
    == k`` case always applies here -- the padding helpers below are still
    called (for parity with the upstream function and future shapes) but
    should be no-ops for every shape this engine actually uses."""
    if w._marlin_cache is not None:
        return w._marlin_cache

    import torch

    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_pad_qweight,
        marlin_pad_scales,
        marlin_padded_nk,
        marlin_permute_scales,
        marlin_make_workspace_new,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
        fp8_fused_exponent_bias_into_scales,
        pack_fp8_to_int32,
    )

    group_size = BLOCK  # weight_block_size[1]
    block_n = BLOCK  # weight_block_size[0]
    device = w.weight.device
    padded_n, padded_k = marlin_padded_nk(n, k, group_size)

    # WEIGHT: [N, K] fp8 -> GPTQ-packed int32 [K // 4, N] -> padded -> Marlin repack.
    qweight = pack_fp8_to_int32(w.weight, size_k_first=False)  # [N, K // 4]
    qweight = qweight.T.contiguous()  # [K // 4, N]
    qweight = marlin_pad_qweight(qweight, n, k, padded_n, padded_k)
    perm = torch.empty(0, dtype=torch.int, device=device)
    marlin_qweight = ops.gptq_marlin_repack(
        b_q_weight=qweight, perm=perm, size_k=padded_k, size_n=padded_n, num_bits=8
    )

    # SCALES: [N/128, K/128] -> [K/128, N] group-wise -> padded -> permuted -> bias-fused (W8A16).
    scales = w.scale_inv.to(torch.bfloat16).T.contiguous()  # [K/128, N/128]
    scales = scales.repeat_interleave(block_n, dim=1)[:, :n]  # [K/128, N]
    scales = marlin_pad_scales(scales, n, k, padded_n, padded_k, group_size)
    marlin_scales = marlin_permute_scales(s=scales, size_k=padded_k, size_n=padded_n, group_size=group_size)
    marlin_scales = fp8_fused_exponent_bias_into_scales(marlin_scales)  # input_dtype != fp8 (we're W8A16)

    workspace = marlin_make_workspace_new(device)
    w._marlin_cache = (marlin_qweight, marlin_scales, workspace)
    return w._marlin_cache


@register("vllm_marlin_fp8_w8a16")
def _vllm_marlin_fp8(x: "torch.Tensor", w: WeightT) -> "torch.Tensor":
    """Weight-only FP8 (W8A16) via vLLM's unified Marlin op. Activations stay
    bf16 (no quantization step: Marlin dequantizes the packed weight
    on-the-fly inside the kernel), i.e. a Marlin-style path that reads FP8
    weights against bf16 activations.

    Best of every backend at low M, but compute-bound and degrading badly past
    M=64 (graph-timed whole-model step: 8.2 ms at M=1, 23.4 ms at M=128,
    88.8 ms at M=512), which is *architecturally expected* for a
    weight-only-dequant kernel (vLLM's own log message: it exists for "GPUs
    that lack FP8 hardware support"). ``DEFAULT_BACKEND_PRIORITY_BY_M_BUCKET``
    already routes larger M to other kernels first, keeping this one at the
    top only for small M, so the tuning below is a secondary lever on top of
    that routing decision, not a fix for the high-M cliff itself.

    Tuning knobs on ``vllm._custom_ops.marlin_gemm`` /
    ``apply_fp8_marlin_linear``:
      - ``is_zp_float``: INT4-with-float-zero-point flag, not applicable to
        FP8 (always ``False`` internally for this path); nothing to tune.
      - ``use_atomic_add``: already auto-selected per-call inside
        ``apply_fp8_marlin_linear`` via ``should_use_atomic_add_reduce(m, n,
        k, device, dtype)``; not exposed as a knob to override, and
        shouldn't be (it's shape-driven).
      - ``use_fp32_reduce`` (default ``True`` via
        ``marlin_utils.USE_FP32_REDUCE_DEFAULT``): IS a real lever, so
        exposed here as M-aware: ``False`` (cheaper, lower-precision
        reduction) once M crosses into the already-compute-bound regime
        (``m >= 128``, matching the measured cliff), ``True`` (full
        precision) below that, where accuracy matters more and the extra
        reduction cost is not yet dominant.
      - A dedicated FP8 GEMV/W8A16 Triton kernel: vLLM 0.28.0 has no
        ``fp8_gemv``/``w8a16`` Triton kernel. Marlin and the CUTLASS/cuBLASLt
        per-tensor paths (this module's ``vllm_cutlass_fp8_pertensor`` /
        ``scaled_mm_pertensor``) are the real candidates for the M=1 case;
        there is no fourth option to add.
    """
    if not isinstance(w, FP8Tensor):
        raise TypeError("vllm_marlin_fp8_w8a16 requires an FP8Tensor weight")
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
        apply_fp8_marlin_linear,
    )

    n, k = w.weight.shape
    m = x.shape[0]
    marlin_qweight, marlin_scales, workspace = _marlin_repacked(w, n, k)
    return apply_fp8_marlin_linear(
        input=x,
        weight=marlin_qweight,
        weight_scale=marlin_scales,
        workspace=workspace,
        size_n=n,
        size_k=k,
        bias=None,
        input_dtype=None,  # bf16 activations -> W8A16 path
        # M-aware (see docstring), but keyed on the *sequence* count, not the
        # raw row count: a speculative verify window packs n=k+1 rows per
        # sequence, and flipping this flag between the decode step and the
        # verify pass that must reproduce it is a real numerical difference
        # (measured: relL2 2.7e-3 -> 3.4e-3 at M=128).
        # `sequence_m` is the identity outside a `rows_per_sequence` scope.
        use_fp32_reduce=(sequence_m(m) < 128),
    )


# --------------------------------------------------------------------------- #
# Machete (CUTLASS Hopper mixed-input W8A16)
# --------------------------------------------------------------------------- #
#: Machete's group-scale axis is K, and 128 is both a supported group size and
#: exactly the checkpoint's own K-block, so one machete group sits entirely
#: inside one checkpoint scale block. (64 is also supported for bf16
#: activations and buys ~8% less requant error -- not enough to change the
#: class; 128 keeps the group boundary aligned with the checkpoint's.)
_MACHETE_GROUP = BLOCK
#: Output rows (N) re-quantized at a time in `_machete_repacked`. Bounds the
#: one-time repack's transient at ~12 B x chunk x K independently of N; see
#: that function for why an unchunked repack is a 15 GiB spike on lm_head.
_MACHETE_REPACK_CHUNK_N = 4096


def _machete_scalar_type():
    """``uint8b128``: signed int8 stored biased by +128.

    **Why not fp8.** ``machete_supported_schedules(bf16, float8_e4m3fn, ...)``
    returns an EMPTY list: vLLM 0.28.0's Machete is only instantiated for
    ``uint4b8`` and ``uint8b128``, and
    ``query_machete_supported_quant_types(zero_points=False)`` agrees. There is
    no fp8 b_type schedule to call, so a Machete path over an fp8 checkpoint
    *must* re-quantize. That is a real, measured accuracy cost (see
    ``_machete_repacked``), not a layout detail."""
    from vllm.scalar_type import scalar_types

    return scalar_types.uint8b128


def _machete_repacked(w: FP8Tensor, n: int, k: int) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """One-time int8 re-quantization + Machete prepack, cached on the ``FP8Tensor``.

    Ported from vLLM's ``kernels/linear/mixed_precision/machete.py``
    (``MacheteLinearKernel.process_weights_after_loading``, which operates on a
    live ``nn.Module``) onto a bare ``FP8Tensor``, with one extra step upstream
    of it that vLLM never needs: **the re-quantization**.

    ### The re-quantization, stated plainly

    Every other backend in this module is a pure re-layout: the numbers the
    kernel multiplies are bit-for-bit the checkpoint's fp8 codes. This one is
    not. Machete has no fp8 b_type (see ``_machete_scalar_type``), so the fp8
    weight is dequantized exactly (fp32) and re-quantized to int8 with a
    per-``(K-group of 128, output channel)`` scale:

        scale[n, kg] = amax_k |W[n, kg*128 : (kg+1)*128]| / 127
        q[n, k]      = round(W[n, k] / scale[n, k // 128])   in [-128, 127]

    Note the scale grid is *finer* than the checkpoint's own: the checkpoint
    shares one scale across a 128x128 block, this shares one across 128 K
    values of a **single** output channel. So all 128 values in a machete group
    carry the same checkpoint block scale, and the re-quantization is exactly
    "rescale the fp8 codes of one column-segment onto a uniform 127-level
    grid".

    **Measured cost of that step** (N=256/K=512 random Gaussian, and
    confirmed on the real shapes): the re-quantized weight is
    ``relL2 6.5e-3`` from the exact fp8 dequant, and the resulting
    GEMM is ``relL2 ~7.0e-3`` from the fp32 reference, against marlin's
    2.7e-3 on the same input. Against the *re-quantized* weight the GEMM is
    2.8e-3, i.e. the kernel itself is exactly as good as marlin and the entire
    gap is the int8 step. That is arithmetic, not tuning: int8 with an amax
    scale over 128 roughly-Gaussian values has an rms relative error of
    ``(amax/127)/sqrt(12)/sigma ~= (2.9/127)/3.46 = 6.6e-3``, which is what was
    measured. A finer group (64) moves it to ~6.1e-3 and nothing moves it to
    the 2.7e-3 class.

    So this backend sits **between** the two classes: 3.7x more accurate than
    every fp8-activation backend, 2.6x less accurate than marlin. Whether that
    clears ``STRICT_REL_L2_MAX`` is a threshold decision documented there, not
    something this function should paper over.

    ### Memory

    ``qweight`` is int8-packed-into-int32, i.e. byte-for-byte the size of the
    fp8 weight it replaces; ``group scales`` are bf16 ``[K/128, N]`` = 1.6%
    more. Same size class as ``_marlin_cache``, and it takes its own cache slot
    so ``--gemm-weight-cache single`` admits marlin **or** machete, never both
    (still exactly one repack copy per weight)."""
    if w._machete_cache is not None:
        return w._machete_cache

    import torch

    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        pack_quantized_values_into_int32,
    )

    g = _MACHETE_GROUP
    if k % g != 0 or n % 128 != 0 or k % 64 != 0:
        raise RuntimeError(
            f"machete_w8a16: unsupported shape N={n} K={k} "
            f"(needs K % {g} == 0, K % 64 == 0, N % 128 == 0)"
        )
    b_type = _machete_scalar_type()

    # Exact fp32 dequant of the block-128 fp8 weight, then int8 re-quant on a
    # [N, K/g] scale grid, then Machete's int32 packing.
    #
    # **Chunked over N, and it has to be.** The naive form materializes fp32
    # [N, K] + int32 [N, K] + int32 [K, N] simultaneously = 12 bytes per weight
    # element. On `mlp_gate_up_proj` that is 2.1 GiB of transient (tolerable);
    # on an fp8 `lm_head` ([248320, 5120], the optional quantized-lm_head path)
    # it is **15 GiB**, more than the whole prefill working set, and would
    # OOM a loaded server outright, because this runs at warmup, i.e. after
    # the weights are already resident. Chunking bounds it
    # at 12 bytes x `chunk` x K (~855 MiB at the default) regardless of N.
    # The packing is along K, so N-chunks are independent and concatenate; the
    # prepack below then runs once on the assembled [K/4, N], which is only as
    # large as the fp8 weight it replaces.
    chunk = max(128, (_MACHETE_REPACK_CHUNK_N // 128) * 128)
    packed_parts: List["torch.Tensor"] = []
    scale_parts: List["torch.Tensor"] = []
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        # Dequantize only this N-slice (slicing a full `w.dequant()` would
        # materialize the very fp32 [N, K] this chunking exists to avoid).
        # `chunk` and every fused N are multiples of 128, so the scale rows
        # line up exactly with the slice.
        wf = (
            w.weight[lo:hi].float()
            * w.scale_inv[lo // BLOCK: hi // BLOCK]
            .repeat_interleave(BLOCK, 0)
            .repeat_interleave(BLOCK, 1)[:, :k]
        ).view(hi - lo, k // g, g)
        s = (wf.abs().amax(dim=2) / 127.0).clamp_min(torch.finfo(torch.float32).tiny)
        q = torch.round(wf / s.unsqueeze(-1)).clamp_(-128, 127).to(torch.int32).view(hi - lo, k)
        del wf
        # Machete's `w_q` contract: {input_dim = 0, output_dim = 1,
        # packed_dim = 0} -> int32 [K // 4, N] holding uint8b128-BIASED codes
        # (vLLM's own `quantize_weights` does `w_q += quant_type.bias` for any
        # `uintNbM` type, so the packer must see [0, 255], not [-128, 127]).
        q_kn = q.t().contiguous() + b_type.bias  # [K, chunk]
        del q
        packed_parts.append(pack_quantized_values_into_int32(q_kn, b_type, packed_dim=0))
        scale_parts.append(s)
        del q_kn, s
    packed = torch.cat(packed_parts, dim=1)  # [K//4, N]
    scale = torch.cat(scale_parts, dim=0)    # [N, K/g]
    del packed_parts, scale_parts

    # `machete_prepack_B` wants B **column-major** (upstream writes that as
    # `x.data.t().contiguous().t()`: a transpose of a contiguous [N, K//4],
    # i.e. exactly a column-major [K//4, N] view -- the same idiom
    # `scaled_mm_pertensor` and `_vllm_block_fp8_cutlass` both document).
    qweight = ops.machete_prepack_B(
        packed.t().contiguous().t(),
        a_type=torch.bfloat16,
        b_type=b_type,
        group_scales_type=torch.bfloat16,
    )
    del packed
    # `w_s` contract: {input_dim = 0, output_dim = 1} -> [K/g, N], act dtype.
    gscales = scale.t().contiguous().to(torch.bfloat16)  # [K/g, N]
    w._machete_cache = (qweight, gscales)
    return w._machete_cache


@register("machete_w8a16")
def _machete_w8a16(x: "torch.Tensor", w: WeightT) -> "torch.Tensor":
    """W8A16 via vLLM's Machete kernel (CUTLASS, Hopper-only, min capability 90).

    Same *class* as ``vllm_marlin_fp8_w8a16`` -- bf16 activations in, weight
    dequantized inside the kernel, no activation quantization anywhere -- but a
    different implementation family: Machete is a TMA/warp-specialized CUTLASS
    mixed-input GEMM, and it exists in vLLM precisely because Marlin's
    throughput falls off a cliff at large M. Graph-timed on this model, marlin
    is 8.22 ms/step at M=1 but 23.37 at M=128 and 88.83 at M=512, against
    ``flashinfer_fp8_blockscale``'s 10.34 / 30.88, which is why speed-ranked
    tables hand every M>=64 to a *less accurate* kernel. Machete is the
    candidate that could make "accurate" and "scales with M" the same choice.

    **Two caveats, both real:**

    1. It re-quantizes the weight to int8 (``_machete_repacked``): ~7.0e-3
       relL2 vs the fp32-dequant reference, against marlin's 2.7e-3. It is not
       a drop-in numerical equal of marlin.
    2. The sweep behind the v4 table did not time it, so it sits in
       ``_UNMEASURED_TAIL`` there, per this module's standing policy:
       "unmeasured is not the same as slow, but must never outrank a measured
       backend". Later profiles rank it on measured graph timings.

    Bias/zero-points: none (our weights are symmetric int8, ``uint8b128`` is
    the signed-with-bias type, so ``b_group_zeros`` stays ``None`` and
    ``b_channel_scales``/``a_token_scales`` -- Machete's W4A8 path -- are unused).
    """
    if not isinstance(w, FP8Tensor):
        raise TypeError("machete_w8a16 requires an FP8Tensor weight")
    import torch

    from vllm import _custom_ops as ops

    n, k = w.weight.shape
    qweight, gscales = _machete_repacked(w, n, k)
    return ops.machete_mm(
        a=x if x.dtype == torch.bfloat16 else x.to(torch.bfloat16),
        b_q=qweight,
        b_type=_machete_scalar_type(),
        out_type=torch.bfloat16,
        b_group_scales=gscales,
        b_group_size=_MACHETE_GROUP,
    ).to(x.dtype)


@register("flashinfer_fp8_blockscale")
def _flashinfer_fp8_blockscale(x: "torch.Tensor", w: WeightT) -> "torch.Tensor":
    """``flashinfer.gemm.fp8_blockscale_gemm_sm90``, bf16 activations in + FP8
    weight with our checkpoint's native 128x128 block scale passed straight
    through (its docstring explicitly documents that scale shape as one of
    the two granularities it accepts).

    **ACCURACY WARNING.** Despite taking bf16 activations, this is **not**
    weight-only FP8 (W8A16). Measured against an exact fp32 dequant
    reference on the real checkpoint's shapes (``[14336, 5120]`` and
    ``[5120, 17408]``), at M in {32, 64, 96, 128}:

        vllm_marlin_fp8_w8a16       relL2 = 2.7e-3   (genuinely W8A16)
        flashinfer_fp8_blockscale   relL2 = 2.6e-2   (10x worse)

    An order-of-magnitude gap that is flat in M is not a tiling effect: it is
    what per-token fp8-e4m3 quantization of the *activations* costs. The kernel
    is DeepGEMM-backed and DeepGEMM's blockscale GEMM is W8A8; ``input_scale=
    None`` means "quantize the input for me", not "skip quantization".

    Consequences, neither of which this function can fix on its own:
      * A speed-ranked priority table can route **every M >= 64** here, so the
        engine runs at ~2.6e-2 GEMM error at batch >= 64 while running at
        2.7e-3 below it. ``BACKEND_REL_L2`` and ``gemm_accuracy("strict")``
        exist to make that trade explicit.
      * A ``B*(k+1)``-row speculative verify window can cross a bucket
        threshold that the ``B``-row decode step it has to reproduce does not,
        breaking verify-vs-decode equivalence; see ``rows_per_sequence``
        above.

    Per its own docstring: "SwapAB kernel is automatically used when M < 32
    (threshold)", so it already covers the low-M/batch-1 case without a
    hand-rolled Triton GEMV. SM90 (Hopper) only; raises on any other
    architecture (checked internally via ``_match_sm_version``)."""
    if not isinstance(w, FP8Tensor):
        raise TypeError("flashinfer_fp8_blockscale requires an FP8Tensor weight")
    _ensure_deepgemm_cuda_home()  # this kernel is DeepGEMM-backed per its own docstring
    from flashinfer.gemm import fp8_blockscale_gemm_sm90

    return fp8_blockscale_gemm_sm90(
        x, w.weight, input_scale=None, weight_scale=w.scale_inv.float(), out_dtype=x.dtype
    )


# --------------------------------------------------------------------------- #
# top-level dispatch
# --------------------------------------------------------------------------- #
def _default_backend(w: WeightT, m: int, n: Optional[int] = None,
                     k: Optional[int] = None) -> str:
    """The static (no-autotune-entry) first pick: the top of ``priority_for(m,
    n, k)`` for FP8 weights (M-bucket- **and**, under v9, shape-aware -- e.g.
    marlin at M=1, deepgemm at M>=32), or ``bf16_native`` for plain bf16
    weights (nothing else applies -- see :func:`_bf16_native`).

    ``n``/``k`` default to ``None`` so an older caller that only has ``(w, m)``
    still gets the shape-agnostic answer rather than a ``TypeError``."""
    return priority_for(m, n, k)[0] if isinstance(w, FP8Tensor) else "bf16_native"


#: Backends that can serve a weight the checkpoint stores in bf16. Order
#: matters: ``bf16_native`` is a single `F.linear`, ``bf16_dequant`` is the
#: identical call behind one extra `isinstance`/dtype branch, kept only so the
#: chain is never empty if `bf16_native` is somehow deregistered.
_BF16_WEIGHT_ORDER: List[str] = ["bf16_native", "bf16_dequant"]


def _shape_of(x: "torch.Tensor", w: WeightT) -> Tuple[int, int, int]:
    k = x.shape[-1]
    m = 1
    for d in x.shape[:-1]:
        m *= d
    n = w.weight.shape[0] if isinstance(w, FP8Tensor) else w.shape[0]
    return m, n, k


def resolve_backend(
    x: "torch.Tensor",
    w: WeightT,
    *,
    sm_version: Optional[int] = None,
    use_autotune: bool = True,
    autotune_cache_dir: Optional[str] = None,
    reasons: Optional[Dict[str, str]] = None,
) -> str:
    """Name of the backend :func:`linear` *would actually use* for ``(x, w)``.

    The point of this function is that it calls each candidate's registered
    implementation **directly** and returns the first that does not raise.
    Callers that memoise a backend (``runtime.fused_model.ResolvedLinear``)
    must not do that by calling :func:`linear` with an explicit ``backend=``
    and assuming it was honoured: ``linear`` catches a failing backend and
    silently walks the rest of the priority order, so a probe built on it
    reports "backend X worked" for any X whenever *some* backend works.
    A probe like that can pin a slow fallback such as
    ``vllm_block_fp8_triton`` (31.5 ms/step whole-model at M=1 against
    marlin's 8.2) for every linear layer.

    Resolution order is the same as :func:`linear`'s: autotune-cache winner
    for ``(sm_version, N, K, m_bucket(M))`` first, then
    ``priority_for_m(M)``.

    ``reasons``: optional dict, filled in with ``{backend: why it was not
    chosen}`` for every candidate ahead of the winner -- a raised exception's
    ``Type: message``, or the weight-cache policy that removed it before it was
    ever tried. Purely diagnostic (nothing on the hot path reads it), and the
    only way to tell "deepgemm is unavailable on this host" apart from
    "deepgemm was policy-filtered because marlin already owns this weight's one
    cache slot" -- two very different facts that used to look identical from
    the outside. ``ResolvedLinear`` passes one in during warmup so
    ``bench_runtime.check_resolved_backends`` can print it.
    """
    m, n, k = _shape_of(x, w)
    x2 = x.reshape(-1, k)
    order: List[str] = []
    if use_autotune:
        from . import autotune

        # per-sequence M: see `rows_per_sequence`. Identity outside that scope.
        cached = autotune.lookup(sm_version, n, k, sequence_m(m), cache_dir=autotune_cache_dir)
        # The autotune cache is ranked on speed alone (autotune.py times, it
        # does not measure error), so a cached winner has to clear the same
        # accuracy bar as the static table or "strict" would be silently
        # bypassed by any host that has run a sweep.
        if cached and _accuracy_allows(cached):
            order.append(cached)
    order += [b for b in priority_for(m, n, k) if b not in order]
    if not isinstance(w, FP8Tensor):
        # nothing but the plain-bf16 path applies to a never-quantized weight
        order = list(_BF16_WEIGHT_ORDER)
    # Drop candidates that would allocate a second permanent repacked copy of
    # this weight (see `_cache_policy_allows`). This is the *resolution* filter,
    # so a backend skipped here is never pinned by `ResolvedLinear` either.
    #
    # Record *why*, not just *that*. Under `--gemm-weight-cache single` this
    # filter can remove a table's rank-1 pick (e.g. `deepgemm`) from every
    # bucket above the one warmup resolved first, which is a 20% GEMM decision
    # made invisibly; see `V7_BACKEND_PRIORITY_BY_M_BUCKET`'s note.
    filtered = _policy_filtered(order, w)
    if reasons is not None:
        for name in order:
            if name not in filtered:
                reasons[name] = (
                    f"skipped by weight-cache policy {_weight_cache_policy!r} "
                    f"(would allocate a second repack cache)"
                )
    order = filtered
    seen = set()
    last_err: Optional[BaseException] = None
    for name in order:
        if name in seen or name not in _BACKENDS:
            continue
        seen.add(name)
        try:
            _BACKENDS[name](x2, w)
            return name
        except Exception as exc:  # noqa: BLE001 -- probing on purpose
            last_err = exc
            if reasons is not None:
                reasons[name] = f"{type(exc).__name__}: {exc}"
            continue
    raise RuntimeError(
        f"gemm.dispatch.resolve_backend: no backend works for M={m} N={n} K={k}; "
        f"last error: {last_err}"
    )


def linear(
    x: "torch.Tensor",
    w: WeightT,
    *,
    backend: Optional[str] = None,
    sm_version: Optional[int] = None,
    use_autotune: bool = True,
    autotune_cache_dir: Optional[str] = None,
) -> "torch.Tensor":
    """``x`` is ``[..., K]`` bf16 (or any float dtype); ``w`` is a fused
    weight (``FP8Tensor`` or plain ``torch.Tensor``). Returns ``[..., N]`` in
    ``x.dtype``.

    Resolution order for the backend to try first:
      1. ``backend`` if explicitly passed.
      2. the autotune cache's winner for ``(sm_version, N, K, m_bucket(M))``,
         if ``use_autotune`` and an entry exists.
      3. the static default for this weight kind.
    If the chosen backend raises (missing/incompatible dependency,
    unsupported shape, ...), the remaining backends in
    ``priority_for_m(M)`` (an M-bucket-aware order -- see
    ``DEFAULT_BACKEND_PRIORITY_BY_M_BUCKET``'s docstring for the measured
    numbers behind it) are tried in order; ``bf16_dequant`` is always last
    and always works, so ``linear()`` only raises if even that fails (e.g.
    a completely malformed input).
    """
    m, n, k = _shape_of(x, w)
    orig_shape = x.shape
    x2 = x.reshape(-1, k)

    chosen = backend
    if chosen is None and use_autotune:
        from . import autotune

        # per-sequence M: see `rows_per_sequence`. Identity outside that scope.
        chosen = autotune.lookup(sm_version, n, k, sequence_m(m), cache_dir=autotune_cache_dir)
        if chosen is not None and not _accuracy_allows(chosen):
            chosen = None  # speed-ranked cache entry, fails the strict bar
    if chosen is None:
        chosen = _default_backend(w, m, n, k)

    if not isinstance(w, FP8Tensor):
        # A never-quantized bf16 weight has exactly two candidates and every
        # fp8 backend rejects it on an `isinstance` check. Walking the full
        # 8-deep priority order for it (e.g. under an explicit
        # `backend="deepgemm"` from `RuntimeConfig.gemm_backend` or
        # `prefill_gemm_scope`) would throw and catch 8 Python exceptions per
        # call for each of the 49 such linears in this model, every step,
        # purely to arrive at the one answer that was ever possible. An
        # explicit `backend=` that names an fp8 kernel is not an instruction
        # that can be honoured here, so it is dropped rather than walked.
        order = ([chosen] if chosen in _BF16_WEIGHT_ORDER else []) + [
            b for b in _BF16_WEIGHT_ORDER if b != chosen
        ]
    else:
        order = [chosen] + [b for b in priority_for(m, n, k) if b != chosen]
    if backend is None:
        # An explicit `backend=` is an instruction, not a preference (it is how
        # `ResolvedLinear` replays an already-resolved, already-policy-filtered
        # choice), so only the fallback-chain form is filtered here.
        order = _policy_filtered(order, w)
    seen = set()
    last_err: Optional[BaseException] = None
    for name in order:
        if name in seen or name not in _BACKENDS:
            continue
        seen.add(name)
        try:
            out = _BACKENDS[name](x2, w)
            return out.reshape(*orig_shape[:-1], n)
        except Exception as exc:  # noqa: BLE001 -- deliberate: try the next backend
            last_err = exc
            continue
    raise RuntimeError(f"gemm.dispatch.linear: every backend failed; last error: {last_err}")


__all__ = [
    "WeightT",
    "BackendFn",
    "M_BUCKETS",
    "DEFAULT_BACKEND_PRIORITY",
    "DEFAULT_BACKEND_PRIORITY_BY_M_BUCKET",
    "V4_BACKEND_PRIORITY_BY_M_BUCKET",
    "V7_BACKEND_PRIORITY_BY_M_BUCKET",
    "V8_BACKEND_PRIORITY_BY_M_BUCKET",
    "BACKEND_PRIORITY_PROFILES",
    "set_backend_priority_profile",
    "get_backend_priority_profile",
    "priority_for_m",
    "priority_for",
    "shape_class",
    "SHAPE_CLASS_BY_NK",
    "V9_BACKEND_PRIORITY_BY_M_BUCKET",
    "V9_BACKEND_PRIORITY_BY_SHAPE_AND_M",
    "BACKEND_PRIORITY_SHAPE_PROFILES",
    "register",
    "available_backends",
    "m_bucket",
    "rows_per_sequence",
    "rows_per_sequence_now",
    "set_rows_per_sequence",
    "sequence_m",
    "linear",
    "resolve_backend",
    "WEIGHT_CACHE_POLICIES",
    "set_weight_cache_policy",
    "get_weight_cache_policy",
    "repack_cache_bytes",
    "free_repack_caches",
    "BACKEND_REL_L2",
    "BACKEND_ACT_DTYPE",
    "STRICT_REL_L2_MAX",
    "GEMM_ACCURACY_MODES",
    "set_gemm_accuracy",
    "get_gemm_accuracy",
    "gemm_accuracy",
    "backend_rel_l2",
    "accurate_backends",
]
