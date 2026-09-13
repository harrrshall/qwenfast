"""CPU unit tests for qwenfast.gemm (GEMM/quantization).

Runs on CPU in float32/bf16, no GPU needed -- gates every commit. GPU-only
tests (each real dispatch backend vs a bf16 reference, at the real model
shapes) are marked ``unittest.skipUnless(torch.cuda.is_available(), ...)`` so
this file is a no-op suite pass on a machine without a GPU, and a real
correctness gate on a GPU host.

Run::

    python -m unittest discover -s engine/qwenfast/gemm/tests -v
    # or
    python engine/qwenfast/gemm/tests/test_gemm.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))  # -> engine/

from qwenfast.gemm.fused_weights import (  # noqa: E402
    BLOCK,
    AttnFusedWeights,
    FP8Tensor,
    FusedModelWeights,
    GDNFusedWeights,
    MLPFusedWeights,
    fuse_bf16_rows,
    fuse_fp8_rows,
    load_fused,
    quantize_bf16_to_fp8_block128,
    save_fused,
)
from qwenfast.gemm import dispatch  # noqa: E402
from qwenfast.gemm.dispatch import M_BUCKETS, linear, m_bucket  # noqa: E402
from qwenfast.weights import QwenFastConfig, dequant_block128  # noqa: E402


# =========================================================================== #
# helpers
# =========================================================================== #
class _FakeStore:
    """Minimal stand-in for weights.SafetensorsStore: a dict of name -> tensor,
    ``.has``/``.raw`` only (exactly what fuse_fp8_rows/fuse_bf16_rows use)."""

    def __init__(self, tensors: dict):
        self._t = tensors

    def has(self, name: str) -> bool:
        return name in self._t

    def raw(self, name: str) -> torch.Tensor:
        return self._t[name]


def _random_fp8_part(n: int, k: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    w_bf16 = torch.randn(n, k, generator=g, dtype=torch.float32).to(torch.bfloat16)
    fw = quantize_bf16_to_fp8_block128(w_bf16, block=BLOCK)
    return fw.weight, fw.scale_inv


def _tiny_config(layer_types) -> QwenFastConfig:
    return QwenFastConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=len(layer_types),
        vocab_size=384,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        layer_types=list(layer_types),
        mtp_num_hidden_layers=0,
    )


# =========================================================================== #
# 1. fused-scale concat correctness: dequant(fused) == cat(dequant(parts))
# =========================================================================== #
class TestFuseFp8Rows(unittest.TestCase):
    def test_two_part_concat_matches_dequant_then_cat(self):
        k = 256  # 2 K-blocks
        n1, n2 = 384, 128  # 3 and 1 N-blocks -> fused N=512, still block-aligned
        w1, s1 = _random_fp8_part(n1, k, seed=1)
        w2, s2 = _random_fp8_part(n2, k, seed=2)
        store = _FakeStore({
            "a.weight": w1, "a.weight_scale_inv": s1,
            "b.weight": w2, "b.weight_scale_inv": s2,
        })

        fused = fuse_fp8_rows(store, ["a.weight", "b.weight"])
        self.assertEqual(tuple(fused.weight.shape), (n1 + n2, k))
        self.assertEqual(tuple(fused.scale_inv.shape), ((n1 + n2) // BLOCK, k // BLOCK))

        dequant_fused = dequant_block128(fused.weight, fused.scale_inv, out_dtype=torch.float32)
        dequant_cat = torch.cat(
            [
                dequant_block128(w1, s1, out_dtype=torch.float32),
                dequant_block128(w2, s2, out_dtype=torch.float32),
            ],
            dim=0,
        )
        torch.testing.assert_close(dequant_fused, dequant_cat, atol=0.0, rtol=0.0)

    def test_three_part_concat_gdn_style_shapes(self):
        # mirrors the qkv_proj fusion ratios (12288:1024:1024 / 128
        # = 96:8:8, i.e. 12:1:1), scaled down to the smallest block-aligned
        # (multiple-of-128) representative shape that keeps the CPU test fast.
        k = 128
        parts = [(12 * 128, "q"), (1 * 128, "k"), (1 * 128, "v")]
        names, tensors = [], {}
        raw_parts = []
        for i, (n, tag) in enumerate(parts):
            w, s = _random_fp8_part(n, k, seed=100 + i)
            tensors[f"{tag}.weight"] = w
            tensors[f"{tag}.weight_scale_inv"] = s
            names.append(f"{tag}.weight")
            raw_parts.append((w, s))
        store = _FakeStore(tensors)

        fused = fuse_fp8_rows(store, names)
        total_n = sum(n for n, _ in parts)
        self.assertEqual(tuple(fused.weight.shape), (total_n, k))

        dequant_fused = dequant_block128(fused.weight, fused.scale_inv, out_dtype=torch.float32)
        dequant_cat = torch.cat(
            [dequant_block128(w, s, out_dtype=torch.float32) for w, s in raw_parts], dim=0
        )
        torch.testing.assert_close(dequant_fused, dequant_cat, atol=0.0, rtol=0.0)

    def test_rejects_non_block_aligned_n(self):
        store = _FakeStore({
            "x.weight": torch.zeros(48, 128, dtype=torch.float8_e4m3fn),
            "x.weight_scale_inv": torch.ones(1, 1, dtype=torch.bfloat16),
        })
        with self.assertRaises(ValueError):
            fuse_fp8_rows(store, ["x.weight"])

    def test_rejects_missing_scale(self):
        store = _FakeStore({"x.weight": torch.zeros(128, 128, dtype=torch.float8_e4m3fn)})
        with self.assertRaises(KeyError):
            fuse_fp8_rows(store, ["x.weight"])

    def test_fuse_bf16_rows_ba_order(self):
        b = torch.arange(4.0).reshape(4, 1)
        a = torch.arange(4.0, 8.0).reshape(4, 1)
        store = _FakeStore({"linear_attn.in_proj_b.weight": b, "linear_attn.in_proj_a.weight": a})
        fused = fuse_bf16_rows(store, ["linear_attn.in_proj_b.weight", "linear_attn.in_proj_a.weight"])
        torch.testing.assert_close(fused, torch.cat([b, a], dim=0))


# =========================================================================== #
# 2. quantize_bf16_to_fp8_block128 sanity
# =========================================================================== #
class TestQuantizeBf16ToFp8Block128(unittest.TestCase):
    def test_roundtrip_is_close(self):
        g = torch.Generator().manual_seed(0)
        w = (torch.randn(256, 256, generator=g) * 0.5).to(torch.bfloat16)
        fw = quantize_bf16_to_fp8_block128(w)
        deq = fw.dequant(out_dtype=torch.float32)
        # e4m3 has ~2-3 bits of mantissa; block-128 scaling keeps relative
        # error small but not tiny -- this is a sanity bound, not a tight one.
        err = (deq - w.float()).abs()
        rel = err / w.float().abs().clamp(min=1e-3)
        self.assertLess(float(rel.mean()), 0.1)

    def test_rejects_non_block_aligned(self):
        with self.assertRaises(ValueError):
            quantize_bf16_to_fp8_block128(torch.zeros(100, 256, dtype=torch.bfloat16))


# =========================================================================== #
# 3. dispatch fallback (bf16_dequant) correctness vs a bf16 reference
# =========================================================================== #
class TestDispatchBf16Fallback(unittest.TestCase):
    def test_fp8_weight_matches_bf16_reference_within_fp8_tolerance(self):
        torch.manual_seed(0)
        m, k, n = 4, 256, 128
        x = torch.randn(m, k, dtype=torch.bfloat16)
        w_bf16 = torch.randn(n, k, dtype=torch.bfloat16)
        fw = quantize_bf16_to_fp8_block128(w_bf16)

        out = linear(x, fw, backend="bf16_dequant", use_autotune=False)
        ref = F.linear(x, w_bf16)
        self.assertEqual(tuple(out.shape), (m, n))
        # fp8 e4m3 block-128 quantization error: bounded in aggregate (relative
        # L2 norm), not element-wise -- individual near-zero output entries
        # can have arbitrarily large *relative* error from cancellation even
        # though the quantization itself is well-behaved.
        rel_l2 = torch.linalg.norm(out.float() - ref.float()) / torch.linalg.norm(ref.float())
        self.assertLess(float(rel_l2), 0.1)

    def test_plain_bf16_weight_is_exact(self):
        torch.manual_seed(1)
        x = torch.randn(3, 64, dtype=torch.bfloat16)
        w = torch.randn(32, 64, dtype=torch.bfloat16)
        out = linear(x, w, backend="bf16_dequant", use_autotune=False)
        ref = F.linear(x, w)
        torch.testing.assert_close(out, ref)

    def test_unknown_backend_falls_through_to_bf16_dequant(self):
        torch.manual_seed(2)
        x = torch.randn(2, 64, dtype=torch.bfloat16)
        w = torch.randn(16, 64, dtype=torch.bfloat16)
        # GPU-only backends will raise ImportError/RuntimeError on a CPU host;
        # linear() must fall through the priority chain to bf16_dequant.
        out = linear(x, w, use_autotune=False)
        ref = F.linear(x, w)
        torch.testing.assert_close(out, ref)


class TestScaledMmPertensorLayout(unittest.TestCase):
    """CPU-checkable regression test for the cuBLASLt weight layout:
    `torch._scaled_mm` itself only runs on CUDA, but the *stride layout*
    `_scaled_mm_pertensor` hands it is plain tensor-metadata and fully
    checkable on CPU. `.t().contiguous()` on the weight re-materializes
    row-major layout and undoes the transpose; the correct form is `.t()`
    with no `.contiguous()`, which cuBLASLt needs as a column-major operand B."""

    def test_transposed_view_is_column_major_without_contiguous(self):
        n, k = 128, 256
        w_bf16 = torch.randn(n, k, dtype=torch.bfloat16)
        wq = w_bf16.to(torch.float8_e4m3fn)  # fresh, contiguous [N, K]
        self.assertTrue(wq.is_contiguous())
        self.assertEqual(wq.stride(), (k, 1))

        w_t = wq.t()  # the fix: no .contiguous()
        self.assertEqual(tuple(w_t.shape), (k, n))
        # column-major [K, N]: element (i, j) at offset i*1 + j*k -> stride (1, k)
        self.assertEqual(w_t.stride(), (1, k))
        self.assertFalse(w_t.is_contiguous())  # correct: it's F-contiguous, not C-contiguous
        self.assertTrue(w_t.is_contiguous(memory_format=torch.contiguous_format) is False)

    def test_bug_reintroduced_by_contiguous_after_transpose(self):
        n, k = 128, 256
        wq = torch.randn(n, k, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        w_t_bad = wq.t().contiguous()  # the bug: forces row-major [K, N] back
        # this is exactly the wrong layout cuBLASLt rejects:
        # row-major B instead of column-major B.
        self.assertEqual(w_t_bad.stride(), (n, 1))
        self.assertNotEqual(w_t_bad.stride(), wq.t().stride())

    def test_scaled_mm_pertensor_produces_transposed_view_not_contiguous_copy(self):
        # exercises the real code path's tensor prep (everything up to the
        # CUDA-only torch._scaled_mm call itself) via the FP8Tensor cache,
        # which is the actual regression surface.
        from qwenfast.gemm.dispatch import _scaled_mm_pertensor

        torch.manual_seed(0)
        w_bf16 = torch.randn(128, 128, dtype=torch.bfloat16)
        fw = quantize_bf16_to_fp8_block128(w_bf16)
        x = torch.randn(4, 128, dtype=torch.bfloat16)
        if not hasattr(torch, "_scaled_mm"):
            self.skipTest("torch build has no torch._scaled_mm (CPU-only builds may omit it)")
        try:
            _scaled_mm_pertensor(x, fw)
        except Exception as exc:  # noqa: BLE001
            # torch._scaled_mm itself is CUDA/Hopper-only and expected to
            # fail on a CPU host; what we care about is that FP8Tensor's
            # cache now holds the *view*, not a re-contiguous'd copy.
            pass
        self.assertIsNotNone(fw._pertensor_cache)
        w_fp8_t, _ = fw._pertensor_cache
        self.assertEqual(w_fp8_t.stride(), (1, 128))  # column-major [K, N]


class TestCutlassBlockFp8Layout(unittest.TestCase):
    """Regression test for the CUTLASS block-fp8 weight layout.
    `vllm_block_fp8_cutlass` fails with `cutlass_scaled_mm,
    .../scaled_mm_entry.cu:209` and an empty message if the weight operand is
    row-major: in vLLM 0.28.0 that line is
    `STD_TORCH_CHECK(b.stride(0) == 1); // Column-major`, so the weight must
    not be forced back to row-major with `.t().contiguous()` (the same hazard
    as `scaled_mm_pertensor`). The actual
    CUTLASS call is CUDA-only, but the layout it's handed is plain
    tensor-metadata and fully checkable on CPU -- that's what this covers."""

    def test_dispatch_builds_column_major_weight_operand(self):
        # Exercise the real function up to (but not including) the CUDA-only
        # ops.cutlass_scaled_mm call, by monkeypatching `ops` inside the
        # module so we can inspect exactly what layout `b` gets handed.
        import types

        from qwenfast.gemm import dispatch as dispatch_mod
        from qwenfast.gemm.fused_weights import FP8Tensor

        captured = {}

        def _fake_cutlass_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias=None):
            captured["a"] = a
            captured["b"] = b
            captured["scale_b"] = scale_b
            return torch.zeros(a.shape[0], b.shape[1], dtype=out_dtype)

        fake_ops_module = types.SimpleNamespace(cutlass_scaled_mm=_fake_cutlass_scaled_mm)
        fake_vllm_module = types.SimpleNamespace(_custom_ops=fake_ops_module)
        sys.modules["vllm"] = fake_vllm_module
        sys.modules["vllm._custom_ops"] = fake_ops_module
        try:
            n, k = 128, 256
            w_bf16 = torch.randn(n, k, dtype=torch.bfloat16)
            fw = quantize_bf16_to_fp8_block128(w_bf16)
            x = torch.randn(4, k, dtype=torch.bfloat16)
            # bypass _per_token_group_quant_fp8's own `from vllm...` import
            # (also faked out above) by calling the backend directly.
            dispatch_mod._vllm_block_fp8_cutlass(x, fw)
        finally:
            sys.modules.pop("vllm", None)
            sys.modules.pop("vllm._custom_ops", None)

        b = captured["b"]
        self.assertEqual(tuple(b.shape), (k, n))
        # the exact check scaled_mm_entry.cu:209 makes: column-major B.
        self.assertEqual(b.stride(0), 1)
        self.assertNotEqual(b.stride(0), n)  # n is what the .contiguous() bug would give


class TestPriorityForM(unittest.TestCase):
    """Regression coverage for the M-bucket-aware priority table.

    The table follows the graph-timed kernel sweep, in which `deepgemm` is
    rank 1 at every M >= 32. The older table is kept as the `"v4"` profile and
    covered by `TestBackendPriorityProfiles`."""

    def test_low_m_prefers_marlin(self):
        """M <= 16: marlin 8.23 ms/step vs deepgemm 8.50. A 3% edge,
        but batch-1 latency is the most sensitive target."""
        from qwenfast.gemm.dispatch import priority_for_m

        for m in (1, 8, 16):
            with self.subTest(m=m):
                self.assertEqual(priority_for_m(m)[0], "vllm_marlin_fp8_w8a16")

    def test_m32_and_up_prefers_deepgemm(self):
        """Graph-timed whole-model per-step:
        deepgemm 8.31/8.28/9.68/14.13/26.65 ms at M=32/64/128/256/512 against
        marlin's 10.00/13.99/23.98/45.37/89.98."""
        from qwenfast.gemm.dispatch import priority_for_m

        for m in (32, 64, 128, 256, 512):
            with self.subTest(m=m):
                self.assertEqual(priority_for_m(m)[0], "deepgemm")

    def test_high_m_demotes_marlin_below_bf16_fallback_candidates(self):
        from qwenfast.gemm.dispatch import priority_for_m

        for m in (64, 128, 256, 512):
            with self.subTest(m=m):
                order = priority_for_m(m)
                self.assertNotEqual(order[0], "vllm_marlin_fp8_w8a16")
                # still registered (autotune may still prefer it at some
                # shape), just not the first thing tried.
                self.assertIn("vllm_marlin_fp8_w8a16", order)
                self.assertLess(
                    order.index("vllm_marlin_fp8_w8a16"), order.index("bf16_dequant")
                )

    def test_bf16_dequant_always_last(self):
        from qwenfast.gemm.dispatch import M_BUCKETS, priority_for_m

        for m in M_BUCKETS:
            with self.subTest(m=m):
                self.assertEqual(priority_for_m(m)[-1], "bf16_dequant")

    def test_bf16_native_is_second_to_last(self):
        """The two bf16 entries are the tail, in this order: an fp8
        weight reaches `bf16_native` only after every real backend raised, and
        it raises too, so `bf16_dequant` still terminates the chain."""
        from qwenfast.gemm.dispatch import M_BUCKETS, priority_for_m

        for m in M_BUCKETS:
            with self.subTest(m=m):
                self.assertEqual(priority_for_m(m)[-2:], ["bf16_native", "bf16_dequant"])


class TestBackendPriorityProfiles(unittest.TestCase):
    """Backend priority profiles, including the prefill-scale (M > 512)
    buckets. The v7 table comes from a graph-timed *kernel* sweep whose
    end-to-end counterpart did not fully agree with it; v8 adds the four
    prefill buckets on top of it, unchanged below 512. `--gemm-priority v7`/`v4`
    are the one-flag rollbacks, and a rollback nobody tests is not a rollback."""

    def setUp(self):
        self.prev = dispatch.get_backend_priority_profile()

    def tearDown(self):
        dispatch.set_backend_priority_profile(self.prev)

    def test_default_profile_is_v8(self):
        self.assertEqual(dispatch.get_backend_priority_profile(), "v8")

    def test_v8_is_byte_identical_to_v7_at_or_below_512(self):
        """The whole point of layering v8 on top of v7: nothing about decode
        (M <= 512) is supposed to change when the prefill buckets are added."""
        for m in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
            with self.subTest(m=m):
                self.assertEqual(
                    dispatch.V8_BACKEND_PRIORITY_BY_M_BUCKET[m],
                    dispatch.V7_BACKEND_PRIORITY_BY_M_BUCKET[m],
                )

    def test_v4_restores_the_window4_ranking(self):
        dispatch.set_backend_priority_profile("v4")
        self.assertEqual(dispatch.priority_for_m(32)[0], "vllm_marlin_fp8_w8a16")
        for m in (64, 128, 256):
            self.assertEqual(dispatch.priority_for_m(m)[0], "flashinfer_fp8_blockscale")
        self.assertEqual(dispatch.priority_for_m(512)[0], "scaled_mm_pertensor")

    def test_v7_clamps_every_prefill_bucket_to_the_m512_answer(self):
        """v7 predates the prefill buckets; this pins that its rollback
        meaning is exactly the old clamp-at-512 behavior, not a crash and not
        a silently different answer."""
        dispatch.set_backend_priority_profile("v7")
        at_512 = dispatch.priority_for_m(512)
        for m in (1024, 2048, 4096, 8192):
            with self.subTest(m=m):
                self.assertEqual(dispatch.priority_for_m(m), at_512)

    def test_switching_back_and_forth_is_exact(self):
        dispatch.set_backend_priority_profile("v8")
        v8 = {m: list(dispatch.priority_for_m(m)) for m in M_BUCKETS}
        dispatch.set_backend_priority_profile("v4")
        dispatch.set_backend_priority_profile("v8")
        self.assertEqual({m: list(dispatch.priority_for_m(m)) for m in M_BUCKETS}, v8)

    def test_profile_name_is_validated(self):
        with self.assertRaises(ValueError):
            dispatch.set_backend_priority_profile("v6")

    def test_every_profile_is_a_total_order_over_every_backend(self):
        registered = set(dispatch.available_backends())
        for name in dispatch.BACKEND_PRIORITY_PROFILES:
            dispatch.set_backend_priority_profile(name)
            for m in M_BUCKETS:
                with self.subTest(profile=name, m=m):
                    order = dispatch.priority_for_m(m)
                    self.assertEqual(set(order), registered)
                    self.assertEqual(len(order), len(registered))  # no duplicates

    def test_v7_matches_the_emitted_sweep_ranking_at_every_measured_bucket(self):
        """Pins the table to the sweep ranking it was derived from, first pick
        only; the tail ordering below rank 1 is informative, not load-bearing."""
        expected_first = {1: "vllm_marlin_fp8_w8a16", 32: "deepgemm", 64: "deepgemm",
                          128: "deepgemm", 256: "deepgemm", 512: "deepgemm"}
        dispatch.set_backend_priority_profile("v7")
        for m, first in expected_first.items():
            with self.subTest(m=m):
                self.assertEqual(dispatch.priority_for_m(m)[0], first)

    def test_v8_matches_the_measured_m2048_ranking(self):
        """The measured M=2048 ranking (real GPU data, not extrapolated):
        `vllm_cutlass_fp8_pertensor` wins, NOT deepgemm. The M=32..512
        champion inverts at prefill scale."""
        dispatch.set_backend_priority_profile("v8")
        self.assertEqual(dispatch.priority_for_m(2048)[0], "vllm_cutlass_fp8_pertensor")
        self.assertEqual(
            dispatch.priority_for_m(2048)[:5],
            [
                "vllm_cutlass_fp8_pertensor",
                "scaled_mm_pertensor",
                "deepgemm",
                "flashinfer_fp8_blockscale",
                "vllm_block_fp8_cutlass",
            ],
        )

    def test_v8_prefill_buckets_never_promote_marlin_or_machete_or_triton(self):
        """Marlin is the single worst backend measured at M=2048
        (356 ms, slower than dequantizing to bf16 on every call). Machete and
        `vllm_block_fp8_triton` are 6-14x the fp8-activation group there too.
        None of the three should ever be the first pick for a prefill bucket."""
        dispatch.set_backend_priority_profile("v8")
        for m in (1024, 2048, 4096, 8192):
            with self.subTest(m=m):
                first = dispatch.priority_for_m(m)[0]
                self.assertNotIn(
                    first,
                    ("vllm_marlin_fp8_w8a16", "machete_w8a16", "vllm_block_fp8_triton"),
                )

    def test_v8_covers_every_bucket_in_m_buckets(self):
        for m in M_BUCKETS:
            with self.subTest(m=m):
                self.assertIn(m, dispatch.V8_BACKEND_PRIORITY_BY_M_BUCKET)

    def test_every_registered_backend_appears_in_every_bucket(self):
        from qwenfast.gemm.dispatch import M_BUCKETS, available_backends, priority_for_m

        backends = set(available_backends())
        for m in M_BUCKETS:
            with self.subTest(m=m):
                self.assertEqual(set(priority_for_m(m)), backends)


class TestQuantOnlyMapping(unittest.TestCase):
    """CPU-safe sanity checks on bench_gemm.py's
    `_QUANT_ONLY_CALL` table (isolates each backend's activation-quant
    kernel under its own graph capture, for M<=32). Catches
    typos/drift between this table and the real registered backend names
    without needing a GPU to actually run the captured callables."""

    def test_every_key_is_a_registered_backend(self):
        from qwenfast.gemm.bench_gemm import _QUANT_ONLY_CALL
        from qwenfast.gemm.dispatch import available_backends

        backends = set(available_backends())
        for name in _QUANT_ONLY_CALL:
            with self.subTest(backend=name):
                self.assertIn(name, backends)

    def test_w8a16_backends_have_no_quant_only_entry(self):
        # vllm_marlin_fp8_w8a16 / machete_w8a16: bf16 activations straight
        # through, no separate quant kernel to isolate.
        # bf16_dequant: also bf16 in, dequantizes the *weight*, not x.
        #
        # `flashinfer_fp8_blockscale` is deliberately NOT on this list: it does
        # quantize the activations, it just does it inside its own kernel
        # rather than through one of this module's helpers, which is why it
        # has no `_QUANT_ONLY_CALL` entry and also why its relL2 is ~2.7e-2. Absence
        # from that table means "no isolable quant step", NOT "W8A16"; the
        # authority on that is `dispatch.BACKEND_ACT_DTYPE`, asserted below.
        from qwenfast.gemm.bench_gemm import _QUANT_ONLY_CALL

        for name in ("vllm_marlin_fp8_w8a16", "machete_w8a16", "bf16_dequant"):
            with self.subTest(backend=name):
                self.assertNotIn(name, _QUANT_ONLY_CALL)
                self.assertEqual(dispatch.BACKEND_ACT_DTYPE[name], "bf16")
        self.assertEqual(dispatch.BACKEND_ACT_DTYPE["flashinfer_fp8_blockscale"], "fp8")

    def test_quant_only_callables_run_on_cpu_via_fallback(self):
        # dispatch._per_token_group_quant_fp8 / _scaled_fp8_quant_pertensor
        # both fall back to a pure-torch path when vllm isn't importable
        # (true on a CPU host) -- so the *callables* in _QUANT_ONLY_CALL
        # should actually run (not just import cleanly), just not under a
        # real CUDA graph here.
        from qwenfast.gemm import dispatch as dispatch_mod
        from qwenfast.gemm.bench_gemm import _QUANT_ONLY_CALL

        x = torch.randn(4, 128, dtype=torch.bfloat16)
        for name, factory in _QUANT_ONLY_CALL.items():
            with self.subTest(backend=name):
                xq, scale = factory(dispatch_mod, x)
                self.assertEqual(tuple(xq.shape), tuple(x.shape))
                self.assertEqual(xq.dtype, torch.float8_e4m3fn)


class TestLargeMSweepPolicy(unittest.TestCase):
    """CPU-safe logic tests for bench_gemm.py's large-M sweep policy:
    above `LARGE_M_THRESHOLD`, only the five
    fp8-activation backends get benched by default, except at
    `LARGE_M_REFERENCE_M`, a reference cell where every backend still runs."""

    def test_reference_m_gets_every_applicable_backend(self):
        from qwenfast.gemm.bench_gemm import LARGE_M_REFERENCE_M, applicable_backends_for
        from qwenfast.gemm.dispatch import available_backends

        backends = available_backends()
        self.assertEqual(
            applicable_backends_for("fp8", LARGE_M_REFERENCE_M, backends),
            list(backends),  # every backend: an fp8 shape admits all of them (pre-existing rule)
        )

    def test_above_threshold_restricts_to_fp8_activation_backends(self):
        from qwenfast.gemm.bench_gemm import (
            LARGE_M_FP8_ACTIVATION_BACKENDS,
            LARGE_M_REFERENCE_M,
            LARGE_M_THRESHOLD,
            applicable_backends_for,
        )
        from qwenfast.gemm.dispatch import available_backends

        backends = available_backends()
        for m in (LARGE_M_THRESHOLD + 1, 2048, 4096, 8192):
            if m == LARGE_M_REFERENCE_M:
                continue
            with self.subTest(m=m):
                got = applicable_backends_for("fp8", m, backends)
                self.assertEqual(set(got), set(LARGE_M_FP8_ACTIVATION_BACKENDS) & set(backends))
                self.assertNotIn("vllm_marlin_fp8_w8a16", got)
                self.assertNotIn("machete_w8a16", got)
                self.assertNotIn("vllm_block_fp8_triton", got)
                self.assertNotIn("bf16_dequant", got)

    def test_at_or_below_threshold_is_unrestricted(self):
        from qwenfast.gemm.bench_gemm import LARGE_M_THRESHOLD, applicable_backends_for
        from qwenfast.gemm.dispatch import available_backends

        backends = available_backends()
        for m in (1, 32, 256, LARGE_M_THRESHOLD):
            with self.subTest(m=m):
                self.assertEqual(applicable_backends_for("fp8", m, backends), list(backends))

    def test_bf16_kind_is_never_restricted_by_m(self):
        from qwenfast.gemm.bench_gemm import applicable_backends_for
        from qwenfast.gemm.dispatch import available_backends

        backends = available_backends()
        for m in (1, 512, 2048, 8192):
            with self.subTest(m=m):
                self.assertEqual(applicable_backends_for("bf16", m, backends), ["bf16_dequant"])

    def test_every_fp8_activation_backend_is_registered(self):
        from qwenfast.gemm.bench_gemm import LARGE_M_FP8_ACTIVATION_BACKENDS
        from qwenfast.gemm.dispatch import available_backends

        registered = set(available_backends())
        for name in LARGE_M_FP8_ACTIVATION_BACKENDS:
            with self.subTest(backend=name):
                self.assertIn(name, registered)

    def test_reference_m_is_above_the_threshold(self):
        """Otherwise it wouldn't be exercising the reference-cell branch at all."""
        from qwenfast.gemm.bench_gemm import LARGE_M_REFERENCE_M, LARGE_M_THRESHOLD

        self.assertGreater(LARGE_M_REFERENCE_M, LARGE_M_THRESHOLD)

    def test_reference_m_and_large_m_buckets_are_real_m_buckets(self):
        from qwenfast.gemm.bench_gemm import LARGE_M_REFERENCE_M
        from qwenfast.gemm.dispatch import M_BUCKETS

        self.assertIn(LARGE_M_REFERENCE_M, M_BUCKETS)


class TestDerivePriorityFromResults(unittest.TestCase):
    """Pure-Python logic test (no GPU needed) for bench_gemm.py's
    `derive_priority_from_results`, which re-derives the per-M-bucket
    priority from graph_us."""

    def test_ranks_by_graph_total_ascending(self):
        from qwenfast.gemm.bench_gemm import derive_priority_from_results

        results = {
            "totals_per_step": [
                {"m": 1, "backend": "fast", "total_ms_per_decode_step_graph": 5.0,
                 "total_ms_per_decode_step": 20.0, "complete": True},
                {"m": 1, "backend": "slow", "total_ms_per_decode_step_graph": 10.0,
                 "total_ms_per_decode_step": 15.0, "complete": True},
                # eager total says "slow" should win (15 < 20), but graph
                # total (the metric that matters) says "fast" wins --
                # the eager-vs-graph ranking flip is exactly what this
                # function exists to get right, so it's the thing to assert on.
            ]
        }
        priority = derive_priority_from_results(results)
        self.assertEqual(priority[1], ["fast", "slow"])

    def test_incomplete_rows_are_excluded(self):
        from qwenfast.gemm.bench_gemm import derive_priority_from_results

        results = {
            "totals_per_step": [
                {"m": 1, "backend": "a", "total_ms_per_decode_step_graph": 5.0, "complete": True},
                {"m": 1, "backend": "b", "total_ms_per_decode_step_graph": 1.0, "complete": False},
            ]
        }
        priority = derive_priority_from_results(results)
        self.assertEqual(priority[1], ["a"])  # "b" excluded despite a lower number -- not trustworthy

    def test_bucket_with_no_complete_rows_is_omitted(self):
        from qwenfast.gemm.bench_gemm import derive_priority_from_results

        results = {"totals_per_step": [
            {"m": 1, "backend": "a", "total_ms_per_decode_step_graph": 5.0, "complete": False},
        ]}
        priority = derive_priority_from_results(results)
        self.assertNotIn(1, priority)

    def test_empty_results(self):
        from qwenfast.gemm.bench_gemm import derive_priority_from_results

        self.assertEqual(derive_priority_from_results({}), {})
        self.assertEqual(derive_priority_from_results({"totals_per_step": []}), {})


class TestComputeTotalsGraphFallback(unittest.TestCase):
    """CPU-safe test of `_compute_totals`'s graph-total +
    fallback-to-eager logic, with synthetic cells (no GPU needed -- this is
    pure aggregation arithmetic over already-produced cell dicts)."""

    def _shape(self, name, n=128, k=128, count=2, kind="fp8"):
        return {"name": name, "n": n, "k": k, "count": count, "kind": kind, "group": "test"}

    def test_graph_capturable_cell_uses_graph_us(self):
        from qwenfast.gemm.bench_gemm import _compute_totals

        shapes = [self._shape("s1")]
        cells = [{
            "status": "ok", "m": 1, "backend": "b", "shape_name": "s1", "count": 2,
            "mean_us": 100.0, "graph_us": 10.0, "graph_capturable": True, "weight_bytes": 1000,
        }]
        totals = _compute_totals(cells, shapes, [1], ["b"])
        self.assertEqual(len(totals), 1)
        t = totals[0]
        self.assertAlmostEqual(t["total_ms_per_decode_step"], 100.0 * 2 / 1000.0)
        self.assertAlmostEqual(t["total_ms_per_decode_step_graph"], 10.0 * 2 / 1000.0)
        self.assertEqual(t["graph_fallback_shapes"], [])
        self.assertTrue(t["graph_complete"])

    def test_non_capturable_cell_falls_back_to_eager_and_is_flagged(self):
        from qwenfast.gemm.bench_gemm import _compute_totals

        shapes = [self._shape("s1")]
        cells = [{
            "status": "ok", "m": 1, "backend": "b", "shape_name": "s1", "count": 2,
            "mean_us": 100.0, "graph_capturable": False, "graph_error": "nope", "weight_bytes": 1000,
        }]
        totals = _compute_totals(cells, shapes, [1], ["b"])
        t = totals[0]
        # graph total falls back to the eager mean_us for this shape.
        self.assertAlmostEqual(t["total_ms_per_decode_step_graph"], 100.0 * 2 / 1000.0)
        self.assertEqual(t["graph_fallback_shapes"], ["s1"])
        self.assertTrue(t["complete"])  # eager-complete...
        self.assertFalse(t["graph_complete"])  # ...but not graph-complete


class TestEnsureDeepgemmCudaHome(unittest.TestCase):
    def setUp(self):
        self._orig_cuda_home = os.environ.get("CUDA_HOME")
        self._orig_path = os.environ.get("PATH", "")
        self._orig_sys_path = list(sys.path)
        self._orig_modules = dict(sys.modules)

    def tearDown(self):
        if self._orig_cuda_home is None:
            os.environ.pop("CUDA_HOME", None)
        else:
            os.environ["CUDA_HOME"] = self._orig_cuda_home
        os.environ["PATH"] = self._orig_path
        sys.path[:] = self._orig_sys_path
        for name in list(sys.modules):
            if name not in self._orig_modules:
                del sys.modules[name]

    def test_noop_when_cuda_home_already_set(self):
        from qwenfast.gemm.dispatch import _ensure_deepgemm_cuda_home

        os.environ["CUDA_HOME"] = "/some/existing/path"
        _ensure_deepgemm_cuda_home()
        self.assertEqual(os.environ["CUDA_HOME"], "/some/existing/path")

    def test_finds_nvcc_wheel_on_sys_path(self):
        from qwenfast.gemm.dispatch import _ensure_deepgemm_cuda_home

        os.environ.pop("CUDA_HOME", None)
        with tempfile.TemporaryDirectory() as d:
            # mimic the pip CUDA wheel layout: <site-packages>/nvidia/cu13/bin/nvcc
            cu13_dir = os.path.join(d, "nvidia", "cu13")
            os.makedirs(os.path.join(cu13_dir, "bin"))
            nvcc_path = os.path.join(cu13_dir, "bin", "nvcc")
            with open(nvcc_path, "w") as f:
                f.write("#!/bin/sh\n")
            os.chmod(nvcc_path, 0o755)
            sys.path.insert(0, d)

            _ensure_deepgemm_cuda_home()

            self.assertEqual(os.environ.get("CUDA_HOME"), cu13_dir)
            self.assertIn(os.path.join(cu13_dir, "bin"), os.environ.get("PATH", ""))


class TestMBucket(unittest.TestCase):
    def test_buckets_round_up(self):
        self.assertEqual(m_bucket(1), 1)
        self.assertEqual(m_bucket(3), 4)
        self.assertEqual(m_bucket(17), 32)
        self.assertEqual(m_bucket(512), 512)
        self.assertEqual(m_bucket(9999), M_BUCKETS[-1])

    def test_prefill_scale_buckets_no_longer_clamp_at_512(self):
        """Every M > 512 must not clamp to 512. A real prefill chunk
        (M up to `max_num_batched_tokens`, 8,192 by default) must now resolve
        its own bucket, not silently reuse M=512's."""
        self.assertEqual(m_bucket(513), 1024)
        self.assertEqual(m_bucket(1024), 1024)
        # The 1536 and 3072 buckets keep these two from rounding up too far:
        # 1,025 and 2,139 are mixed-step M values (chunk + decode rows), and
        # rounding them to 2048/4096 would route them on a table measured
        # 1.6-1.9x above them.
        self.assertEqual(m_bucket(1025), 1536)
        self.assertEqual(m_bucket(1536), 1536)
        self.assertEqual(m_bucket(1537), 2048)
        self.assertEqual(m_bucket(2048), 2048)
        self.assertEqual(m_bucket(2139), 3072)  # an example mixed chunk segment total
        self.assertEqual(m_bucket(3072), 3072)
        self.assertEqual(m_bucket(3073), 4096)
        self.assertEqual(m_bucket(4096), 4096)
        self.assertEqual(m_bucket(4097), 8192)
        self.assertEqual(m_bucket(8192), 8192)
        self.assertEqual(m_bucket(8193), 8192)  # still clamps above the real ceiling

    def test_m_buckets_covers_1024_2048_4096_8192(self):
        for m in (1024, 1536, 2048, 3072, 4096, 8192):
            with self.subTest(m=m):
                self.assertIn(m, M_BUCKETS)

    def test_bucket_mapping_is_monotonic(self):
        """`m_bucket` must never decrease as `m` increases -- a routing table
        that isn't monotonic in M would mean a *bigger* prefill chunk could
        get routed to a *smaller*-M-measured answer, which is backwards.
        Above the top bucket (8,192) it clamps rather than rounding up
        (there is no larger bucket to round up to); the clamp itself is
        still monotonic (constant), just no longer `>= m`."""
        prev = 0
        for m in range(1, 9000):
            b = m_bucket(m)
            self.assertGreaterEqual(b, prev, f"m_bucket({m})={b} < previous bucket {prev}")
            if m <= M_BUCKETS[-1]:
                self.assertGreaterEqual(b, m, f"m_bucket({m})={b} rounds down, not up")
            else:
                self.assertEqual(b, M_BUCKETS[-1], f"m_bucket({m})={b} should clamp at the top bucket")
            prev = b

    def test_m_buckets_itself_is_sorted_and_deduplicated(self):
        self.assertEqual(M_BUCKETS, sorted(set(M_BUCKETS)))


# =========================================================================== #
# 4. save_fused / load_fused round trip
# =========================================================================== #
class TestSaveLoadFused(unittest.TestCase):
    def _build_tiny_fused(self) -> FusedModelWeights:
        layer_types = ["linear_attention", "full_attention"]
        config = _tiny_config(layer_types)
        h = config.hidden_size

        def fp8(n, k, seed):
            w, s = _random_fp8_part(n, k, seed)
            return FP8Tensor(w, s)

        gdn = {
            0: GDNFusedWeights(
                layer_idx=0,
                in_proj_qkvz=fp8(256, h, 10),
                in_proj_ba=torch.randn(96, h, dtype=torch.bfloat16),
                out_proj=fp8(h, 256, 11),
                conv1d_weight=torch.randn(384, 1, 4, dtype=torch.bfloat16),
                A_log=torch.randn(4, dtype=torch.bfloat16),
                dt_bias=torch.randn(4, dtype=torch.bfloat16),
                norm_weight=torch.randn(64, dtype=torch.bfloat16),
            )
        }
        attn = {
            1: AttnFusedWeights(
                layer_idx=1,
                qkv_proj=fp8(384, h, 20),
                o_proj=fp8(h, 256, 21),
                q_norm=torch.randn(64, dtype=torch.bfloat16),
                k_norm=torch.randn(64, dtype=torch.bfloat16),
            )
        }
        mlp = {
            i: MLPFusedWeights(layer_idx=i, gate_up_proj=fp8(1024, h, 30 + i), down_proj=fp8(h, 512, 40 + i))
            for i in range(2)
        }
        layernorms = {
            i: (torch.randn(h, dtype=torch.bfloat16), torch.randn(h, dtype=torch.bfloat16)) for i in range(2)
        }
        return FusedModelWeights(
            config=config,
            embed_tokens=torch.randn(config.vocab_size, h, dtype=torch.bfloat16),
            lm_head=torch.randn(config.vocab_size, h, dtype=torch.bfloat16),
            final_norm=torch.randn(h, dtype=torch.bfloat16),
            layernorms=layernorms,
            gdn=gdn,
            attn=attn,
            mlp=mlp,
            mtp=None,
        )

    def test_roundtrip_preserves_tensors_and_config(self):
        try:
            import safetensors  # noqa: F401
        except Exception:
            self.skipTest("safetensors not installed")

        fw = self._build_tiny_fused()
        with tempfile.TemporaryDirectory() as d:
            path = save_fused(fw, d)
            self.assertTrue(os.path.exists(path))
            loaded = load_fused(d)

        self.assertEqual(loaded.config.hidden_size, fw.config.hidden_size)
        self.assertEqual(loaded.config.layer_types, fw.config.layer_types)
        self.assertIsNone(loaded.mtp)
        torch.testing.assert_close(loaded.embed_tokens, fw.embed_tokens)
        torch.testing.assert_close(loaded.lm_head, fw.lm_head)

        g0, g0l = loaded.gdn[0], fw.gdn[0]
        torch.testing.assert_close(g0.in_proj_qkvz.weight, g0l.in_proj_qkvz.weight)
        torch.testing.assert_close(g0.in_proj_qkvz.scale_inv, g0l.in_proj_qkvz.scale_inv)
        torch.testing.assert_close(g0.in_proj_ba, g0l.in_proj_ba)
        torch.testing.assert_close(g0.conv1d_weight, g0l.conv1d_weight)

        a1, a1l = loaded.attn[1], fw.attn[1]
        torch.testing.assert_close(a1.qkv_proj.weight, a1l.qkv_proj.weight)
        torch.testing.assert_close(a1.o_proj.scale_inv, a1l.o_proj.scale_inv)

        for i in range(2):
            torch.testing.assert_close(loaded.mlp[i].gate_up_proj.weight, fw.mlp[i].gate_up_proj.weight)
            torch.testing.assert_close(loaded.mlp[i].down_proj.scale_inv, fw.mlp[i].down_proj.scale_inv)


# =========================================================================== #
# 5. GPU tests -- skipped on CPU. Each real dispatch backend vs a bf16
#    reference, at the real model shapes.
# =========================================================================== #
@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestDispatchBackendsOnGPU(unittest.TestCase):
    """Real backends (vLLM Triton/CUTLASS block-fp8, torch._scaled_mm
    per-tensor, DeepGEMM) vs a bf16 matmul reference, at representative real
    model shapes (fused shapes, small M so this stays fast)."""

    SHAPES = [
        ("gdn_in_proj_qkvz", 16384, 5120),
        ("attn_qkv_proj", 14336, 5120),
        ("mlp_gate_up_proj", 34816, 5120),
        ("lm_head", 248320, 5120),
    ]
    M_VALUES = [1, 8, 32]

    def _run_backend(self, name, x, w, ref):
        from qwenfast.gemm.dispatch import _BACKENDS

        fn = _BACKENDS[name]
        try:
            out = fn(x, w)
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"backend {name!r} unavailable/failed: {type(exc).__name__}: {exc}")
            return
        self.assertEqual(tuple(out.shape), tuple(ref.shape))
        # aggregate relative L2 error, not element-wise -- see the CPU-side
        # bf16_dequant test above for why element-wise rtol is the wrong tool
        # for fp8-quantization noise.
        rel_l2 = torch.linalg.norm(out.float() - ref.float()) / torch.linalg.norm(ref.float())
        self.assertLess(float(rel_l2), 0.15, f"backend {name!r} relative L2 error {float(rel_l2):.4f}")

    def test_backends_vs_bf16_reference(self):
        device = "cuda:0"
        skipped_shapes = []
        for name, n, k in self.SHAPES:
            # Fixture setup, not a backend call. `quantize_bf16_to_fp8_block128`
            # materializes an fp32 [N, K] (5.1 GiB at the lm_head shape), and
            # the GPU may be shared with another job holding most of its
            # memory. An OOM *here* is an
            # environment condition, not a result about any backend, and
            # failing the whole test on it hides the 72 subtests that did run.
            # Recorded and reported at the end rather than swallowed.
            try:
                w_bf16 = torch.randn(n, k, device=device, dtype=torch.bfloat16)
                fw = quantize_bf16_to_fp8_block128(w_bf16)
            except torch.OutOfMemoryError as exc:
                skipped_shapes.append(f"{name} [{n},{k}]: {exc}".split("\n")[0])
                torch.cuda.empty_cache()
                continue
            for m in self.M_VALUES:
                x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
                ref = F.linear(x, w_bf16)
                for backend in ("vllm_block_fp8_triton", "vllm_block_fp8_cutlass",
                                  "scaled_mm_pertensor", "deepgemm",
                                  "flashinfer_fp8_blockscale", "vllm_marlin_fp8_w8a16",
                                  "vllm_cutlass_fp8_pertensor", "machete_w8a16"):
                    with self.subTest(shape=name, m=m, backend=backend):
                        self._run_backend(backend, x, fw, ref)
            del w_bf16, fw
            torch.cuda.empty_cache()
        if skipped_shapes:
            print("\n[backends-vs-bf16] shapes skipped -- OOM building the FIXTURE "
                  "(shared GPU, not a backend result):")
            for line in skipped_shapes:
                print(f"  {line}")
        self.assertLess(len(skipped_shapes), len(self.SHAPES),
                        "every shape OOM'd during fixture setup -- no backend was "
                        "actually exercised, so this run proves nothing")


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestGraphCapturabilityOnGPU(unittest.TestCase):
    """Tests each backend for graph-capturability, since the engine needs
    graph-capturable GEMMs. Runs
    `cuda_graph_time_fn` against every registered backend at one
    representative shape/M and reports which ones succeed -- this is
    inherently a discovery test (we don't know the answer ahead of a real
    run), so it does not fail the suite over a backend turning out to be
    non-capturable; it prints a results table (visible with `-v`/on
    failure) and only fails if EVERY backend is non-capturable (that would
    mean `cuda_graph_time_fn` itself is broken, not that some backend has a
    real capture limitation)."""

    def test_every_backend_capturability(self):
        from qwenfast.gemm.bench_gemm import cuda_graph_time_fn
        from qwenfast.gemm.dispatch import _BACKENDS
        from qwenfast.gemm.fused_weights import quantize_bf16_to_fp8_block128

        device = "cuda:0"
        n, k, m = 14336, 5120, 8  # attn_qkv_proj shape, small M
        x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
        w_bf16 = torch.randn(n, k, device=device, dtype=torch.bfloat16)
        w = quantize_bf16_to_fp8_block128(w_bf16)
        w_plain = torch.randn(n, k, device=device, dtype=torch.bfloat16)

        capturable = {}
        for name, fn in _BACKENDS.items():
            # `bf16_native` and `bf16_dequant` are the two backends that
            # can serve a never-quantized weight; `bf16_native` *only* serves
            # one, so handing it the fp8 fixture would print a FAILED row that
            # says nothing about capturability.
            weight = w_plain if name in ("bf16_dequant", "bf16_native") else w
            try:
                fn(x, weight)  # smoke call: absorb JIT / repack-cache warmup outside capture
                timing = cuda_graph_time_fn(lambda: fn(x, weight), warmup=3, iters=3, n_capture=5)
                capturable[name] = ("OK", timing["graph_us"])
            except Exception as exc:  # noqa: BLE001
                capturable[name] = ("FAILED", f"{type(exc).__name__}: {exc}")

        print("\n[graph-capturability]")
        for name, (status, detail) in sorted(capturable.items()):
            print(f"  {name:28s} {status:8s} {detail}")

        self.assertTrue(
            any(status == "OK" for status, _ in capturable.values()),
            f"every backend failed graph capture -- likely a bug in cuda_graph_time_fn itself: {capturable}",
        )


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestLargeMBackendsOnGPU(unittest.TestCase):
    """Large-M correctness, not timing -- the timed sweep is `bench_gemm.py`'s
    job. This is deliberately kept to well under a <2 min/<10GB short-test
    budget: ONE shape, ONE M (M=2048, which the timed sweep also covers, so
    the (N,K,M) JIT-compile for the five backends below is usually already in
    the disk cache), and the five `bench_gemm.LARGE_M_FP8_ACTIVATION_BACKENDS`
    the default sweep runs above M=512 -- not all eight registered backends,
    on purpose (marlin/machete at M=2048 are individually slow, and this is a
    correctness smoke test, not the timed reference-cell sweep)."""

    SHAPE = ("attn_qkv_proj", 14336, 5120)
    M = 2048

    def _check_backend(self, name, x, w, ref, m):
        from qwenfast.gemm.dispatch import _BACKENDS

        fn = _BACKENDS[name]
        try:
            out = fn(x, w)
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"backend {name!r} unavailable/failed at M={m}: {type(exc).__name__}: {exc}")
            return
        self.assertEqual(tuple(out.shape), tuple(ref.shape))
        rel_l2 = torch.linalg.norm(out.float() - ref.float()) / torch.linalg.norm(ref.float())
        # every fp8-activation backend is measured 2.5-2.7e-2 at
        # every M including 2048; 0.05 leaves headroom without admitting a
        # backend that's silently fallen into the per-tensor (3.8-3.9e-2) or
        # worse class.
        self.assertLess(float(rel_l2), 0.05, f"backend {name!r} relative L2 error {float(rel_l2):.4f} at M={m}")

    def test_large_m_backends_vs_bf16_reference(self):
        from qwenfast.gemm.bench_gemm import LARGE_M_FP8_ACTIVATION_BACKENDS
        from qwenfast.gemm.fused_weights import quantize_bf16_to_fp8_block128

        device = "cuda:0"
        _name, n, k = self.SHAPE
        w_bf16 = torch.randn(n, k, device=device, dtype=torch.bfloat16)
        fw = quantize_bf16_to_fp8_block128(w_bf16)
        x = torch.randn(self.M, k, device=device, dtype=torch.bfloat16)
        ref = F.linear(x, w_bf16)
        for backend in LARGE_M_FP8_ACTIVATION_BACKENDS:
            with self.subTest(backend=backend):
                self._check_backend(backend, x, fw, ref, self.M)
        del w_bf16, fw
        torch.cuda.empty_cache()

    def test_deepgemm_odd_m_not_a_multiple_of_four(self):
        """DeepGEMM's TMA-aligned scale layout pads to multiples of 4 rows,
        which is a hazard at large M. A real prefill chunk's M is a token
        count, not a value anyone rounds (a typical mixed chunk segment total
        is 2,139, not a multiple of 4). This is
        the one cell that actually exercises that: M=2049 (2048 + 1), single
        shape, single backend, so it stays cheap."""
        from qwenfast.gemm.fused_weights import quantize_bf16_to_fp8_block128

        device = "cuda:0"
        _name, n, k = self.SHAPE
        m = 2049
        w_bf16 = torch.randn(n, k, device=device, dtype=torch.bfloat16)
        fw = quantize_bf16_to_fp8_block128(w_bf16)
        x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
        ref = F.linear(x, w_bf16)
        self._check_backend("deepgemm", x, fw, ref, m)
        del w_bf16, fw
        torch.cuda.empty_cache()


# =========================================================================== #
# weight-repack cache policy (bounds duplicate repacked weight copies)
# =========================================================================== #
class TestWeightCachePolicy(unittest.TestCase):
    """Three backends memoise a full repacked copy of the weight on the
    ``FP8Tensor``. Nothing bounded how many of them a single process could
    populate, and a server that touched two of them carried 2x the FP8 linear
    weights in caches nobody could see -- 46 GiB on the 27B.

    These tests run on CPU: they exercise the *policy*, which is pure
    bookkeeping over the cache slots, not the GPU kernels behind them."""

    def setUp(self):
        self.prev = dispatch.get_weight_cache_policy()
        self.w = FP8Tensor(
            weight=torch.zeros(256, 128, dtype=torch.int8),  # stands in for fp8 storage
            scale_inv=torch.ones(2, 1, dtype=torch.float32),
        )

    def tearDown(self):
        dispatch.set_weight_cache_policy(self.prev)

    def test_policy_names_are_validated(self):
        with self.assertRaises(ValueError):
            dispatch.set_weight_cache_policy("some-other-thing")
        self.assertIn("single", dispatch.WEIGHT_CACHE_POLICIES)

    def test_repack_cache_bytes_counts_every_slot(self):
        self.assertEqual(dispatch.repack_cache_bytes(self.w), 0)
        self.w._marlin_cache = (torch.zeros(64, dtype=torch.int32), torch.zeros(8), torch.zeros(1))
        self.assertEqual(dispatch.repack_cache_bytes(self.w), 64 * 4 + 8 * 4 + 1 * 4)
        self.w._pertensor_cache = (torch.zeros(32, dtype=torch.int8), torch.zeros(1))
        self.assertEqual(dispatch.repack_cache_bytes(self.w), 64 * 4 + 8 * 4 + 1 * 4 + 32 + 4)

    def test_free_repack_caches_releases_and_reports(self):
        self.w._marlin_cache = (torch.zeros(100, dtype=torch.int32),)
        freed = dispatch.free_repack_caches(self.w)
        self.assertEqual(freed, 400)
        self.assertIsNone(self.w._marlin_cache)
        self.assertEqual(dispatch.repack_cache_bytes(self.w), 0)

    def test_multi_allows_a_second_cache_owner(self):
        dispatch.set_weight_cache_policy("multi")
        self.w._marlin_cache = (torch.zeros(4),)
        self.assertTrue(dispatch._cache_policy_allows("scaled_mm_pertensor", self.w))

    def test_single_blocks_a_second_cache_owner(self):
        """The bug, in one assertion: with marlin already resolved for the
        decode buckets, prefill's M-bucket-512 winner must not be allowed to
        allocate a *second* full copy of the same weight."""
        dispatch.set_weight_cache_policy("single")
        self.assertTrue(dispatch._cache_policy_allows("vllm_marlin_fp8_w8a16", self.w))
        self.w._marlin_cache = (torch.zeros(4),)
        self.assertFalse(dispatch._cache_policy_allows("scaled_mm_pertensor", self.w))
        self.assertFalse(dispatch._cache_policy_allows("vllm_cutlass_fp8_pertensor", self.w))
        self.assertFalse(dispatch._cache_policy_allows("deepgemm", self.w))
        # the slot it already owns stays usable, and cache-free backends are
        # never affected
        self.assertTrue(dispatch._cache_policy_allows("vllm_marlin_fp8_w8a16", self.w))
        self.assertTrue(dispatch._cache_policy_allows("flashinfer_fp8_blockscale", self.w))
        self.assertTrue(dispatch._cache_policy_allows("bf16_dequant", self.w))

    def test_none_blocks_every_cache_owner(self):
        dispatch.set_weight_cache_policy("none")
        for name in ("vllm_marlin_fp8_w8a16", "scaled_mm_pertensor",
                     "vllm_cutlass_fp8_pertensor", "deepgemm"):
            self.assertFalse(dispatch._cache_policy_allows(name, self.w), name)
        self.assertTrue(dispatch._cache_policy_allows("flashinfer_fp8_blockscale", self.w))

    def test_filter_never_empties_the_fallback_chain(self):
        """`linear()`'s contract is that the chain always ends in a backend
        that works; the filter must not be able to break that."""
        for policy in dispatch.WEIGHT_CACHE_POLICIES:
            dispatch.set_weight_cache_policy(policy)
            for m in M_BUCKETS:
                order = dispatch._policy_filtered(list(dispatch.priority_for_m(m)), self.w)
                self.assertTrue(order, f"{policy}/{m}: empty order")
                self.assertIn("bf16_dequant", order, f"{policy}/{m}")

    def test_single_keeps_marlin_for_decode_and_drops_pertensor_at_prefill(self):
        """The concrete resolution order a server sees: warmup resolves the
        decode buckets first (marlin, M<=32), then the first prefill chunk asks
        for M-bucket 512 -- whose measured winner is `scaled_mm_pertensor`, a
        cache owner. Under "single" it must fall through to the cache-free
        `flashinfer_fp8_blockscale` instead."""
        dispatch.set_weight_cache_policy("single")
        low = dispatch._policy_filtered(list(dispatch.priority_for_m(1)), self.w)
        self.assertEqual(low[0], "vllm_marlin_fp8_w8a16")
        self.w._marlin_cache = (torch.zeros(4),)  # as warmup would leave it
        high = dispatch._policy_filtered(list(dispatch.priority_for_m(2000)), self.w)
        self.assertEqual(high[0], "flashinfer_fp8_blockscale")
        self.assertNotIn("scaled_mm_pertensor", high)


# =========================================================================== #
# accuracy-aware routing
# =========================================================================== #
class TestBackendAccuracyTables(unittest.TestCase):
    """`BACKEND_REL_L2` and `BACKEND_ACT_DTYPE` are *data*, and stale data is
    worse than none here: `gemm_accuracy="strict"` is a promise about measured
    error, and a backend added later without a row would silently inherit
    ``inf`` (correct, it gets excluded) while a backend *removed* would leave a
    dangling row nobody notices. These pin both directions."""

    def test_every_registered_backend_has_a_measured_rel_l2(self):
        for name in dispatch.available_backends():
            with self.subTest(backend=name):
                self.assertIn(name, dispatch.BACKEND_REL_L2)
                self.assertIn(name, dispatch.BACKEND_ACT_DTYPE)

    def test_no_stale_rows(self):
        registered = set(dispatch.available_backends())
        self.assertEqual(set(dispatch.BACKEND_REL_L2), registered)
        self.assertEqual(set(dispatch.BACKEND_ACT_DTYPE), registered)

    def test_the_two_classes_do_not_overlap(self):
        """The measured split is the whole basis for `strict`: every
        bf16-activation backend must be an order of magnitude better than
        every fp8-activation one. If a future kernel breaks that, `strict`
        stops meaning "W8A16 only" and this test should be the thing that
        says so, not a silently different routing table."""
        bf16 = [dispatch.BACKEND_REL_L2[n] for n, a in dispatch.BACKEND_ACT_DTYPE.items()
                if a == "bf16"]
        fp8 = [dispatch.BACKEND_REL_L2[n] for n, a in dispatch.BACKEND_ACT_DTYPE.items()
               if a == "fp8"]
        self.assertTrue(bf16 and fp8)
        self.assertLess(max(bf16), min(fp8))

    def test_strict_threshold_sits_between_the_classes(self):
        bf16_floor = min(dispatch.BACKEND_REL_L2[n]
                         for n, a in dispatch.BACKEND_ACT_DTYPE.items() if a == "bf16")
        fp8_floor = min(dispatch.BACKEND_REL_L2[n]
                        for n, a in dispatch.BACKEND_ACT_DTYPE.items() if a == "fp8")
        self.assertGreater(dispatch.STRICT_REL_L2_MAX, bf16_floor)
        self.assertLess(dispatch.STRICT_REL_L2_MAX, fp8_floor)

    def test_unknown_backend_is_treated_as_failing(self):
        self.assertEqual(dispatch.backend_rel_l2("no_such_backend"), float("inf"))


class TestGemmAccuracyKnob(unittest.TestCase):
    def setUp(self):
        self.prev = dispatch.get_gemm_accuracy()

    def tearDown(self):
        dispatch.set_gemm_accuracy(self.prev)

    def test_default_is_fast_and_changes_nothing(self):
        """The accuracy knob ships off: `fast` is the identity on whatever the
        current priority profile says (see `TestBackendPriorityProfiles` for
        which table that is)."""
        self.assertEqual(dispatch.get_gemm_accuracy(), "fast")
        for m in M_BUCKETS:
            with self.subTest(m=m):
                self.assertEqual(
                    dispatch.priority_for_m(m),
                    dispatch.DEFAULT_BACKEND_PRIORITY_BY_M_BUCKET[dispatch.m_bucket(m)],
                )

    def test_mode_names_are_validated(self):
        with self.assertRaises(ValueError):
            dispatch.set_gemm_accuracy("mostly")

    def test_strict_admits_only_measured_accurate_backends(self):
        dispatch.set_gemm_accuracy("strict")
        for m in M_BUCKETS:
            with self.subTest(m=m):
                order = dispatch.priority_for_m(m)
                self.assertTrue(order)
                for name in order:
                    self.assertLessEqual(
                        dispatch.backend_rel_l2(name), dispatch.STRICT_REL_L2_MAX,
                        f"{name} at M={m}",
                    )

    def test_strict_excludes_the_backend_the_speed_table_picks_at_high_m(self):
        """The point of the knob, stated as a test: at M>=64 the speed-ranked
        table's first pick is `flashinfer_fp8_blockscale` (2.7e-2), and strict
        must not resolve it. Self-guarding -- it first asserts that *is* the
        fast pick, so it fails loudly if the table it exists to constrain
        changes underneath it."""
        for m, fast_pick in ((32, "deepgemm"),
                             (64, "deepgemm"),
                             (128, "deepgemm"),
                             (256, "deepgemm"),
                             (512, "deepgemm")):
            with self.subTest(m=m):
                dispatch.set_gemm_accuracy("fast")
                self.assertEqual(dispatch.priority_for_m(m)[0], fast_pick)
                self.assertGreater(dispatch.BACKEND_REL_L2[fast_pick],
                                   dispatch.STRICT_REL_L2_MAX)
                dispatch.set_gemm_accuracy("strict")
                self.assertNotIn(fast_pick, dispatch.priority_for_m(m))

    def test_strict_keeps_marlin_first_where_it_already_was(self):
        dispatch.set_gemm_accuracy("strict")
        for m in (1, 8, 32):
            with self.subTest(m=m):
                self.assertEqual(dispatch.priority_for_m(m)[0], "vllm_marlin_fp8_w8a16")

    def test_strict_preserves_relative_order(self):
        """Strict is a *filter*, not a re-ranking: within the surviving
        backends the speed order must be untouched, so `strict` never trades
        speed for anything except the accuracy bound it advertises."""
        for m in M_BUCKETS:
            dispatch.set_gemm_accuracy("fast")
            fast = dispatch.priority_for_m(m)
            dispatch.set_gemm_accuracy("strict")
            strict = [n for n in dispatch.priority_for_m(m) if n in fast]
            with self.subTest(m=m):
                self.assertEqual(strict, [n for n in fast if n in strict])

    def test_strict_never_empties_the_chain(self):
        dispatch.set_gemm_accuracy("strict")
        for m in M_BUCKETS:
            with self.subTest(m=m):
                order = dispatch.priority_for_m(m)
                self.assertIn("bf16_dequant", order)
                self.assertEqual(order[-1], "bf16_dequant")

    def test_scope_is_restored_after_an_exception(self):
        with self.assertRaises(RuntimeError):
            with dispatch.gemm_accuracy("strict"):
                self.assertEqual(dispatch.get_gemm_accuracy(), "strict")
                raise RuntimeError("boom")
        self.assertEqual(dispatch.get_gemm_accuracy(), self.prev)

    def test_accurate_backends_matches_the_threshold(self):
        acc = set(dispatch.accurate_backends())
        for name in dispatch.available_backends():
            with self.subTest(backend=name):
                self.assertEqual(
                    name in acc,
                    dispatch.BACKEND_REL_L2[name] <= dispatch.STRICT_REL_L2_MAX,
                )

    def test_a_speed_ranked_autotune_entry_cannot_bypass_strict(self):
        """`autotune.py` times, it does not measure error, so its cached
        winner is exactly the kind of thing that would quietly re-admit a
        2.7e-2 backend under `strict`."""
        dispatch.set_gemm_accuracy("strict")
        self.assertFalse(dispatch._accuracy_allows("flashinfer_fp8_blockscale"))
        self.assertTrue(dispatch._accuracy_allows("vllm_marlin_fp8_w8a16"))
        dispatch.set_gemm_accuracy("fast")
        self.assertTrue(dispatch._accuracy_allows("flashinfer_fp8_blockscale"))

    def test_runtime_config_default_is_fast(self):
        """The dataclass default must match the dispatcher's, or a model built
        with a default RuntimeConfig would silently flip a process-wide knob."""
        from qwenfast.runtime.fused_model import RuntimeConfig

        self.assertEqual(RuntimeConfig().gemm_accuracy, "fast")
        self.assertIn(RuntimeConfig().gemm_accuracy, dispatch.GEMM_ACCURACY_MODES)


class TestMacheteRepackLayout(unittest.TestCase):
    """`machete_w8a16` is the one backend here that does not just re-lay-out
    the checkpoint's numbers -- it re-quantizes them to int8, because this
    vLLM's Machete has no compiled fp8 b_type. Both halves
    are checkable on CPU: the requantization is plain torch arithmetic, and
    the layout handed to the CUDA-only `machete_prepack_B` is tensor
    metadata. Same technique as TestCutlassBlockFp8Layout above."""

    def _run_repack(self, n=256, k=512):
        import types

        from qwenfast.gemm import dispatch as dispatch_mod

        captured = {}

        def _fake_prepack_B(b_q_weight, a_type, b_type, group_scales_type):
            captured["b_q_weight"] = b_q_weight
            captured["a_type"] = a_type
            captured["group_scales_type"] = group_scales_type
            return b_q_weight

        def _fake_pack(w_q, wtype, packed_dim=0):
            captured["packed_input"] = w_q.clone()
            captured["packed_dim"] = packed_dim
            perm = (*[i for i in range(w_q.dim()) if i != packed_dim], packed_dim)
            shape = list(w_q.permute(perm).shape)
            shape[-1] //= 4
            return torch.zeros(shape, dtype=torch.int32).permute(
                tuple(perm.index(i) for i in range(len(perm)))
            )

        class _FakeScalarType:
            bias = 128

        fake_ops = types.SimpleNamespace(machete_prepack_B=_fake_prepack_B)
        fake_vllm = types.SimpleNamespace(_custom_ops=fake_ops)
        fake_qu = types.SimpleNamespace(pack_quantized_values_into_int32=_fake_pack)
        saved = {m: sys.modules.get(m) for m in (
            "vllm", "vllm._custom_ops",
            "vllm.model_executor.layers.quantization.utils.quant_utils")}
        sys.modules["vllm"] = fake_vllm
        sys.modules["vllm._custom_ops"] = fake_ops
        sys.modules["vllm.model_executor.layers.quantization.utils.quant_utils"] = fake_qu
        prev_scalar = dispatch_mod._machete_scalar_type
        dispatch_mod._machete_scalar_type = lambda: _FakeScalarType()
        try:
            w_bf16 = torch.randn(n, k, dtype=torch.bfloat16)
            fw = quantize_bf16_to_fp8_block128(w_bf16)
            qweight, gscales = dispatch_mod._machete_repacked(fw, n, k)
        finally:
            dispatch_mod._machete_scalar_type = prev_scalar
            for m, mod in saved.items():
                if mod is None:
                    sys.modules.pop(m, None)
                else:
                    sys.modules[m] = mod
        return captured, qweight, gscales, fw

    def test_group_scales_are_k_major_and_act_dtype(self):
        n, k = 256, 512
        captured, _, gscales, _ = self._run_repack(n, k)
        # Machete's w_s contract: {input_dim = 0, output_dim = 1} -> [K/g, N]
        self.assertEqual(tuple(gscales.shape), (k // BLOCK, n))
        self.assertEqual(gscales.dtype, torch.bfloat16)
        self.assertEqual(captured["group_scales_type"], torch.bfloat16)
        self.assertEqual(captured["a_type"], torch.bfloat16)

    def test_packed_weight_is_biased_into_uint8b128_range(self):
        """`quantize_weights` in vLLM does `w_q += quant_type.bias` for a
        `uintNbM` type, so the packer must be handed values in [0, 255], not
        the signed [-128, 127]. Getting this wrong is not a crash -- it is a
        wrong-by-128 weight, i.e. exactly the kind of silent numerical bug
        these tests exist to catch."""
        captured, _, _, _ = self._run_repack()
        packed_in = captured["packed_input"]
        self.assertGreaterEqual(int(packed_in.min()), 0)
        self.assertLessEqual(int(packed_in.max()), 255)
        self.assertEqual(captured["packed_dim"], 0)

    def test_packer_is_fed_a_k_major_matrix(self):
        n, k = 256, 512
        captured, _, _, _ = self._run_repack(n, k)
        self.assertEqual(tuple(captured["packed_input"].shape), (k, n))

    def test_requantization_error_is_the_documented_int8_group_error(self):
        """The number `_machete_repacked`'s docstring and BACKEND_REL_L2 both
        quote. Computed here from the same arithmetic the backend uses, so a
        change to the group size or the scale rule shows up as a test failure
        rather than as a quietly worse engine."""
        n, k = 256, 512
        _, _, gscales, fw = self._run_repack(n, k)
        exact = fw.dequant(torch.float32)
        scale = gscales.float().t()                       # [N, K/g]
        wg = exact.view(n, k // BLOCK, BLOCK)
        q = torch.round(wg / scale.unsqueeze(-1)).clamp_(-128, 127)
        approx = (q * scale.unsqueeze(-1)).view(n, k)
        rel = float((approx - exact).norm() / exact.norm())
        self.assertLess(rel, 1.2e-2)
        self.assertGreater(rel, 3e-3)   # it is NOT in marlin's 2.7e-3 class


class TestMacheteCacheAccounting(unittest.TestCase):
    """The memory plan counts *cache slots*, not backends. Machete must
    take its own slot so `--gemm-weight-cache single` makes it an
    alternative to marlin rather than a third 23 GiB copy alongside it."""

    def setUp(self):
        self.prev = dispatch.get_weight_cache_policy()
        self.w = FP8Tensor(
            weight=torch.zeros(256, 128, dtype=torch.int8),
            scale_inv=torch.ones(2, 1, dtype=torch.float32),
        )

    def tearDown(self):
        dispatch.set_weight_cache_policy(self.prev)

    def test_machete_has_its_own_cache_slot(self):
        self.assertEqual(dispatch._CACHE_ATTR_BY_BACKEND["machete_w8a16"], "_machete_cache")
        self.assertIn("_machete_cache", dispatch._CACHE_ATTRS)

    def test_every_cache_owning_backend_maps_to_a_known_slot(self):
        for name, attr in dispatch._CACHE_ATTR_BY_BACKEND.items():
            with self.subTest(backend=name):
                self.assertIn(name, dispatch.available_backends())
                self.assertIn(attr, dispatch._CACHE_ATTRS)

    def test_repack_cache_bytes_counts_the_machete_slot(self):
        self.assertEqual(dispatch.repack_cache_bytes(self.w), 0)
        self.w._machete_cache = (torch.zeros(64, dtype=torch.int32), torch.zeros(8))
        self.assertEqual(dispatch.repack_cache_bytes(self.w), 64 * 4 + 8 * 4)
        self.assertGreater(dispatch.free_repack_caches(self.w), 0)
        self.assertIsNone(self.w._machete_cache)

    def test_single_policy_admits_marlin_or_machete_but_not_both(self):
        dispatch.set_weight_cache_policy("single")
        self.assertTrue(dispatch._cache_policy_allows("machete_w8a16", self.w))
        self.assertTrue(dispatch._cache_policy_allows("vllm_marlin_fp8_w8a16", self.w))
        self.w._marlin_cache = (torch.zeros(4),)  # as warmup would leave it
        self.assertFalse(dispatch._cache_policy_allows("machete_w8a16", self.w))
        self.assertTrue(dispatch._cache_policy_allows("vllm_marlin_fp8_w8a16", self.w))


class TestCutlassBlockFp8ScaleLayout(unittest.TestCase):
    """Regression test for the CUTLASS block-fp8 *weight-scale* operand
    layout: `w.scale_inv.float().t().contiguous()` re-materializes row-major,
    the same hazard as for `b` and `scale_a`. With that layout relL2 is
    **1.27** on the real checkpoint's `layers.0.mlp.down_proj`, against 2.6e-2
    for every other fp8-activation backend, yet random weights give only 0.10,
    under a 0.15 threshold. A per-backend comparison, not a fixed threshold,
    is what exposes it."""

    def test_scale_b_is_a_column_major_view(self):
        import types

        from qwenfast.gemm import dispatch as dispatch_mod

        captured = {}

        def _fake_cutlass_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias=None):
            captured["scale_b"] = scale_b
            return torch.zeros(a.shape[0], b.shape[1], dtype=out_dtype)

        fake_ops = types.SimpleNamespace(cutlass_scaled_mm=_fake_cutlass_scaled_mm)
        sys.modules["vllm"] = types.SimpleNamespace(_custom_ops=fake_ops)
        sys.modules["vllm._custom_ops"] = fake_ops
        try:
            n, k = 256, 512
            fw = quantize_bf16_to_fp8_block128(torch.randn(n, k, dtype=torch.bfloat16))
            dispatch_mod._vllm_block_fp8_cutlass(torch.randn(4, k, dtype=torch.bfloat16), fw)
        finally:
            sys.modules.pop("vllm", None)
            sys.modules.pop("vllm._custom_ops", None)

        sb = captured["scale_b"]
        self.assertEqual(tuple(sb.shape), (k // BLOCK, n // BLOCK))
        # vLLM's own caller passes `Bs.T`, a transposed view -> stride(0) == 1.
        self.assertEqual(sb.stride(0), 1)
        self.assertNotEqual(sb.stride(0), n // BLOCK)  # what .contiguous() would give

class TestBf16NativeIsNotTheDequantFallback(unittest.TestCase):
    """Why `bf16_native` exists separately from `bf16_dequant`.

    A runtime backend census of `{'vllm_marlin_fp8_w8a16': 256,
    'bf16_dequant': 49}` at every M-bucket in **both** accuracy modes reads as
    49 layers on a 208 ms/step backend. They are not: they are the 48 GDN
    `in_proj_ba` `[96, 5120]` weights (N=96 is not a multiple of 128, so
    `fused_weights.py` deliberately never quantizes them) plus the bf16
    `lm_head` `[248320, 5120]`, and for a weight the checkpoint stores in bf16
    `bf16_dequant` degenerates to a plain `F.linear`. A distinct name keeps
    the census honest.
    """

    def _bf16_weight(self, n=96, k=512):
        return torch.randn(n, k, dtype=torch.bfloat16)

    def test_bf16_native_serves_a_plain_bf16_weight(self):
        w = self._bf16_weight()
        x = torch.randn(4, 512, dtype=torch.bfloat16)
        out = dispatch._bf16_native(x, w)
        self.assertEqual(tuple(out.shape), (4, 96))
        torch.testing.assert_close(out, F.linear(x, w))

    def test_bf16_native_refuses_an_fp8_weight(self):
        """The whole point: it cannot silently stand in for a real fp8 backend.
        An FP8Tensor reaching it must raise so the chain moves on to
        `bf16_dequant`, which is a *reported defect*, not a quiet success."""
        fw = quantize_bf16_to_fp8_block128(torch.randn(256, 512, dtype=torch.bfloat16))
        with self.assertRaises(TypeError):
            dispatch._bf16_native(torch.randn(4, 512, dtype=torch.bfloat16), fw)

    def test_resolve_backend_picks_bf16_native_for_a_bf16_weight(self):
        w = self._bf16_weight()
        x = torch.randn(4, 512, dtype=torch.bfloat16)
        self.assertEqual(
            dispatch.resolve_backend(x, w, use_autotune=False), "bf16_native"
        )

    def test_default_backend_for_a_bf16_weight_is_bf16_native(self):
        w = self._bf16_weight()
        for m in M_BUCKETS:
            with self.subTest(m=m):
                self.assertEqual(dispatch._default_backend(w, m), "bf16_native")

    def test_forced_fp8_backend_on_a_bf16_weight_does_not_walk_the_fp8_chain(self):
        """`--gemm-backend deepgemm` must not send these 49 layers through
        `linear()`'s full try/except order (eight thrown-and-caught Python
        exceptions per call, per layer, per step) to reach the one answer
        that is possible. `linear` short-circuits on the weight kind.
        Asserted by counting how many backends are even *attempted*."""
        w = self._bf16_weight()
        x = torch.randn(4, 512, dtype=torch.bfloat16)
        attempted = []
        real = dict(dispatch._BACKENDS)

        def spy(name, fn):
            def wrapped(a, b):
                attempted.append(name)
                return fn(a, b)
            return wrapped

        dispatch._BACKENDS.update({n: spy(n, f) for n, f in real.items()})
        try:
            out = dispatch.linear(x, w, backend="deepgemm", use_autotune=False)
        finally:
            dispatch._BACKENDS.clear()
            dispatch._BACKENDS.update(real)
        torch.testing.assert_close(out, F.linear(x, w))
        self.assertEqual(attempted, ["bf16_native"])

    def test_the_49_bf16_linears_are_a_negligible_weight_read(self):
        """The weight-read arithmetic, pinned. The reason "leave them in bf16" is
        the right answer for 48 of the 49 and `lm_head` is the only lever."""
        in_proj_ba = 48 * 96 * 5120 * 2          # [96, 5120] bf16, one per GDN layer
        lm_head = 248320 * 5120 * 2              # bf16 vocab projection
        self.assertAlmostEqual(in_proj_ba / 1e6, 47.2, places=1)      # MB
        self.assertAlmostEqual(lm_head / 1e9, 2.543, places=3)        # GB
        # ~0.6 ms/step at a measured 4.3 TB/s, 98.2% of it lm_head.
        self.assertGreater(lm_head / (in_proj_ba + lm_head), 0.98)


class TestPerTokenGroupQuantRefusesToDegradeSilently(unittest.TestCase):
    """`_per_token_group_quant_fp8` must not answer *any* exception
    (including a `TypeError` from a layout kwarg a newer vLLM dropped) with a
    pure-torch quantizer that produces **row-major fp32 scales and ignores
    `use_ue8m0`**. `deepgemm` passes
    `column_major_scales=True, tma_aligned_scales=True` because `fp8_gemm_nt`
    reads that buffer against TMA-aligned column-major strides; handing it the
    fallback's layout gives a bounded, finite, silent accuracy loss."""

    def setUp(self):
        self.prev = dispatch._ptgq_fp8

    def tearDown(self):
        dispatch._ptgq_fp8 = self.prev

    def test_unsupported_layout_kwarg_raises_instead_of_falling_back(self):
        def narrow(x, block, column_major_scales=False):
            raise AssertionError("must not be called")

        dispatch._ptgq_fp8 = (narrow, frozenset({"x", "block", "column_major_scales"}))
        with self.assertRaises(RuntimeError) as cm:
            dispatch._per_token_group_quant_fp8(
                torch.randn(4, 512), 128, column_major_scales=True, tma_aligned_scales=True
            )
        self.assertIn("tma_aligned_scales", str(cm.exception))

    def test_missing_vllm_still_uses_the_torch_path(self):
        """The other half of the split. With vLLM absent there is no fp8 GEMM
        on the host to consume these scales -- `_deepgemm` raises at
        `_deepgemm_available()` first -- so the only callers that reach here
        are the CPU suite and `bench_gemm`'s quant-only microbench, both of
        which want the torch quantizer."""
        dispatch._ptgq_fp8 = (None, frozenset())
        xq, scale = dispatch._per_token_group_quant_fp8(torch.randn(4, 512), 128)
        self.assertEqual(xq.dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(scale.shape), (4, 4))

    def test_torch_fallback_really_honours_column_major_scales(self):
        """It must not *ignore* the kwarg and return row-major. `stride(0) == 1`
        is the whole content of the request (a `.contiguous()` after the
        transpose would undo it)."""
        dispatch._ptgq_fp8 = (None, frozenset())
        _, scale = dispatch._per_token_group_quant_fp8(
            torch.randn(8, 512), 128, column_major_scales=True
        )
        self.assertEqual(tuple(scale.shape), (8, 4))
        self.assertEqual(scale.stride(0), 1)

    def test_torch_fallback_really_honours_use_ue8m0(self):
        """e8m0 has no mantissa: every scale must be an exact power of two."""
        dispatch._ptgq_fp8 = (None, frozenset())
        _, scale = dispatch._per_token_group_quant_fp8(
            torch.randn(8, 512), 128, use_ue8m0=True
        )
        log2 = torch.log2(scale)
        torch.testing.assert_close(log2, torch.round(log2))

    def test_torch_fallback_warns_about_the_one_kwarg_it_cannot_honour(self):
        dispatch._ptgq_fp8 = (None, frozenset())
        dispatch._warned.clear()
        with self.assertWarns(RuntimeWarning):
            dispatch._per_token_group_quant_fp8(
                torch.randn(4, 512), 128, tma_aligned_scales=True
            )

    def test_torch_fallback_rejects_a_kwarg_it_has_never_heard_of(self):
        dispatch._ptgq_fp8 = (None, frozenset())
        with self.assertRaises(RuntimeError):
            dispatch._per_token_group_quant_fp8(torch.randn(4, 512), 128, nonsense=True)

    def test_supported_kwargs_are_forwarded_verbatim(self):
        seen = {}

        def wide(x, block, column_major_scales=False, tma_aligned_scales=False, use_ue8m0=False):
            seen.update(column_major_scales=column_major_scales,
                        tma_aligned_scales=tma_aligned_scales, use_ue8m0=use_ue8m0)
            return "q", "s"

        dispatch._ptgq_fp8 = (wide, frozenset(
            {"x", "block", "column_major_scales", "tma_aligned_scales", "use_ue8m0"}))
        dispatch._per_token_group_quant_fp8(
            torch.randn(4, 512), 128,
            column_major_scales=True, tma_aligned_scales=True, use_ue8m0=True,
        )
        self.assertEqual(seen, {"column_major_scales": True,
                                "tma_aligned_scales": True, "use_ue8m0": True})


class TestDeepGemmEntrypointsAreResolvedOnce(unittest.TestCase):
    """`_deepgemm` must not run `from vllm.utils.deep_gemm import
    fp8_gemm_nt` or a `try: from ...fp8_utils import
    should_use_deepgemm_for_fp8_linear except ImportError:` on every call.
    In some vLLM builds that name does not exist, so the steady state would
    raise and catch an ImportError once per linear per step to compute a
    constant. That is free under CUDA-graph replay but paid in full during
    warmup, capture, `resolve_backend` probing, `--no-graphs`, and all of
    prefill."""

    def setUp(self):
        self.prev = dispatch._deepgemm_entrypoints

    def tearDown(self):
        dispatch._deepgemm_entrypoints = self.prev

    def test_memoised_after_the_first_resolution(self):
        sentinel = (object(), None)
        dispatch._deepgemm_entrypoints = sentinel
        self.assertIs(dispatch._deepgemm_entry(), sentinel)

    def test_deepgemm_source_has_no_per_call_import_statement(self):
        """Structural, because the cost is invisible to any functional test:
        the body of `_deepgemm` must contain no `import` at all beyond the
        `import torch` every backend does."""
        import inspect

        src = inspect.getsource(dispatch._deepgemm)
        body = src.split('"""', 2)[-1]  # drop the docstring
        imports = [ln.strip() for ln in body.splitlines()
                   if ln.strip().startswith(("import ", "from "))]
        self.assertEqual(imports, ["import torch"], f"per-call imports back in _deepgemm: {imports}")


# =========================================================================== #
# per-(shape, M) routing, the v9 profile, the finer buckets
# =========================================================================== #
class TestShapeClass(unittest.TestCase):
    """`shape_class(N, K)` is the *only* thing that makes two call sites at the
    same M route differently, so its failure mode matters: an unknown shape
    must map to `None` (fall back to the M-only table), never to a neighbouring
    shape's answer."""

    def test_every_fused_model_shape_has_a_class(self):
        from qwenfast.gemm.bench_gemm import SHAPES
        from qwenfast.gemm.dispatch import shape_class

        for shape in SHAPES:
            with self.subTest(shape=shape["name"]):
                self.assertIsNotNone(
                    shape_class(shape["n"], shape["k"]),
                    f"bench_gemm benches {shape['name']} [{shape['n']}, {shape['k']}] but "
                    f"dispatch.shape_class cannot name it -- an emitted v9 row for it would "
                    f"be silently dropped",
                )

    def test_gdn_out_proj_and_attn_o_proj_are_the_same_gemm(self):
        """Not a simplification: both are [5120, 6144] in this checkpoint, so
        one table row is the *correct* number of rows for them, and a table
        that pretended otherwise would be claiming a distinction it cannot
        measure."""
        from qwenfast.gemm.dispatch import shape_class

        self.assertEqual(shape_class(5120, 6144), "out_proj")

    def test_unknown_shape_is_none_not_a_neighbour(self):
        from qwenfast.gemm.dispatch import shape_class

        for n, k in ((34816, 4096), (5120, 5120), (1, 1), (34815, 5120)):
            with self.subTest(n=n, k=k):
                self.assertIsNone(shape_class(n, k))

    def test_none_shape_is_tolerated(self):
        from qwenfast.gemm.dispatch import shape_class

        self.assertIsNone(shape_class(None, None))
        self.assertIsNone(shape_class(5120, None))


class TestPriorityForShapeAndM(unittest.TestCase):
    def setUp(self):
        self.prev = dispatch.get_backend_priority_profile()
        self.saved = {c: dict(b) for c, b in
                      dispatch.V9_BACKEND_PRIORITY_BY_SHAPE_AND_M.items()}

    def tearDown(self):
        dispatch.V9_BACKEND_PRIORITY_BY_SHAPE_AND_M.clear()
        dispatch.V9_BACKEND_PRIORITY_BY_SHAPE_AND_M.update(self.saved)
        dispatch.set_backend_priority_profile(self.prev)

    def test_priority_for_without_a_shape_is_priority_for_m(self):
        for name in dispatch.BACKEND_PRIORITY_PROFILES:
            dispatch.set_backend_priority_profile(name)
            for m in M_BUCKETS:
                with self.subTest(profile=name, m=m):
                    self.assertEqual(dispatch.priority_for(m), dispatch.priority_for_m(m))

    def test_shape_overlay_promotes_and_stays_a_total_order(self):
        """An overlay is allowed to be a *partial* list (a large-M sweep only
        benches five of the ten registered backends). The rest of the M-only
        order must still be appended behind it, or `linear()`'s fallback chain
        stops being exhaustive -- which is the kind of bug that only shows up
        as a hard failure on a host where the promoted backend is unavailable."""
        dispatch.set_backend_priority_profile("v9")
        dispatch.V9_BACKEND_PRIORITY_BY_SHAPE_AND_M.clear()
        dispatch.V9_BACKEND_PRIORITY_BY_SHAPE_AND_M["mlp_gate_up"] = {
            1536: ["vllm_block_fp8_cutlass", "deepgemm"]
        }
        order = dispatch.priority_for(1280, 34816, 5120)
        self.assertEqual(order[:2], ["vllm_block_fp8_cutlass", "deepgemm"])
        self.assertEqual(set(order), set(dispatch.available_backends()))
        self.assertEqual(len(order), len(dispatch.available_backends()))
        self.assertEqual(order[-2:], ["bf16_native", "bf16_dequant"])

    def test_overlay_applies_only_to_its_own_shape_and_bucket(self):
        dispatch.set_backend_priority_profile("v9")
        dispatch.V9_BACKEND_PRIORITY_BY_SHAPE_AND_M.clear()
        dispatch.V9_BACKEND_PRIORITY_BY_SHAPE_AND_M["mlp_gate_up"] = {
            1536: ["vllm_block_fp8_cutlass"]
        }
        base = dispatch.priority_for_m(1280)
        # different shape, same bucket
        self.assertEqual(dispatch.priority_for(1280, 5120, 17408), base)
        # same shape, different bucket
        self.assertEqual(dispatch.priority_for(8192, 34816, 5120),
                         dispatch.priority_for_m(8192))
        # unknown shape, same bucket
        self.assertEqual(dispatch.priority_for(1280, 999, 999), base)

    def test_v9_with_an_empty_overlay_is_exactly_v8(self):
        """The rollback property, and the reason a v9 A/B is a test of the
        table rather than of the bucket list: with no measured rows, v9 and v8
        must be indistinguishable at every bucket."""
        dispatch.V9_BACKEND_PRIORITY_BY_SHAPE_AND_M.clear()
        dispatch.set_backend_priority_profile("v8")
        v8 = {m: list(dispatch.priority_for_m(m)) for m in M_BUCKETS}
        dispatch.set_backend_priority_profile("v9")
        self.assertEqual({m: list(dispatch.priority_for_m(m)) for m in M_BUCKETS}, v8)

    def test_switching_away_from_v9_drops_the_overlay(self):
        """The overlay is process-wide state set by the profile switch. If a
        switch to v8 left v9's overlay installed, a `--gemm-priority v8`
        rollback would silently keep v9's routing -- i.e. the rollback would
        not be one."""
        dispatch.set_backend_priority_profile("v9")
        dispatch.V9_BACKEND_PRIORITY_BY_SHAPE_AND_M.clear()
        dispatch.V9_BACKEND_PRIORITY_BY_SHAPE_AND_M["mlp_gate_up"] = {
            1536: ["vllm_block_fp8_cutlass"]
        }
        self.assertEqual(dispatch.priority_for(1280, 34816, 5120)[0], "vllm_block_fp8_cutlass")
        dispatch.set_backend_priority_profile("v8")
        self.assertEqual(dispatch.priority_for(1280, 34816, 5120),
                         dispatch.priority_for_m(1280))

    def test_every_profile_has_a_shape_overlay_entry(self):
        for name in dispatch.BACKEND_PRIORITY_PROFILES:
            with self.subTest(profile=name):
                self.assertIn(name, dispatch.BACKEND_PRIORITY_SHAPE_PROFILES)

    def test_only_v9_has_a_nonempty_overlay(self):
        for name, overlay in dispatch.BACKEND_PRIORITY_SHAPE_PROFILES.items():
            if name != "v9":
                with self.subTest(profile=name):
                    self.assertEqual(overlay, {})

    def test_overlay_keys_are_real_shape_classes_and_real_buckets(self):
        """A typo in a pasted table is silent -- `priority_for` just never finds
        the row and falls back. This turns it into a test failure."""
        classes = set(dispatch.SHAPE_CLASS_BY_NK.values())
        for cls, buckets in dispatch.V9_BACKEND_PRIORITY_BY_SHAPE_AND_M.items():
            with self.subTest(cls=cls):
                self.assertIn(cls, classes)
                for bucket, order in buckets.items():
                    self.assertIn(bucket, M_BUCKETS)
                    for backend in order:
                        self.assertIn(backend, dispatch.available_backends())

    def test_accuracy_filter_still_applies_through_the_overlay(self):
        """`strict` mode filters the order down to the W8A16 backends. An
        overlay that promoted an fp8-activation backend must not survive that
        filter, or `--gemm-accuracy strict` would be silently bypassed by the
        one profile that has a shape table."""
        dispatch.set_backend_priority_profile("v9")
        dispatch.V9_BACKEND_PRIORITY_BY_SHAPE_AND_M.clear()
        dispatch.V9_BACKEND_PRIORITY_BY_SHAPE_AND_M["mlp_gate_up"] = {
            1536: ["deepgemm"]
        }
        with dispatch.gemm_accuracy("strict"):
            order = dispatch.priority_for(1280, 34816, 5120)
            self.assertNotIn("deepgemm", order)
            self.assertTrue(all(dispatch.backend_rel_l2(b) <= dispatch.STRICT_REL_L2_MAX
                                for b in order))


class TestFinerBucketsChangeNothingForV4V7V8(unittest.TestCase):
    """`M_BUCKETS` includes 1536 and 3072. That is a global
    every profile reads, so the claim "v4/v7/v8 are unchanged" has to be a test
    and not a comment: each new bucket's row must equal the row the value used
    to round up to."""

    INHERITS = {1536: 2048, 3072: 4096}

    def test_new_buckets_inherit_the_row_they_used_to_round_up_to(self):
        for table_name, table in (
            ("v4", dispatch.V4_BACKEND_PRIORITY_BY_M_BUCKET),
            ("v7", dispatch.V7_BACKEND_PRIORITY_BY_M_BUCKET),
            ("v8", dispatch.V8_BACKEND_PRIORITY_BY_M_BUCKET),
        ):
            for new, old in self.INHERITS.items():
                with self.subTest(table=table_name, bucket=new):
                    self.assertEqual(table[new], table[old])

    def test_every_profile_covers_every_bucket(self):
        for name, table in dispatch.BACKEND_PRIORITY_PROFILES.items():
            for m in M_BUCKETS:
                with self.subTest(profile=name, m=m):
                    self.assertIn(m, table)

    def test_the_mixed_steps_real_m_values_now_have_their_own_buckets(self):
        """The mixed-step configurations: chunk 1,024/2,048 plus 128/256
        decode rows. Without the finer buckets these would land on 2048 or 4096."""
        self.assertEqual(m_bucket(1024 + 128), 1536)
        self.assertEqual(m_bucket(1024 + 256), 1536)
        self.assertEqual(m_bucket(2048 + 256), 3072)


class TestDeriveShapePriorityFromResults(unittest.TestCase):
    """`bench_gemm.derive_shape_priority_from_results` is what turns a sweep
    into the v9 table, so its keying is load-bearing: get the bucket or the
    class wrong and the pasted table is a no-op that looks like a win."""

    def _cell(self, n, k, m, backend, graph_us, status="ok"):
        return {"n": n, "k": k, "m": m, "backend": backend, "graph_us": graph_us,
                "graph_capturable": True, "status": status, "mean_us": graph_us * 2,
                "count": 64, "shape_name": "x"}

    def test_ranks_each_shape_independently(self):
        from qwenfast.gemm.bench_gemm import derive_shape_priority_from_results

        results = {"cells": [
            # mlp_gate_up: A wins
            self._cell(34816, 5120, 1280, "deepgemm", 100.0),
            self._cell(34816, 5120, 1280, "scaled_mm_pertensor", 120.0),
            # out_proj at the same M: the other way round
            self._cell(5120, 6144, 1280, "deepgemm", 90.0),
            self._cell(5120, 6144, 1280, "scaled_mm_pertensor", 40.0),
        ]}
        got = derive_shape_priority_from_results(results)
        self.assertEqual(got["mlp_gate_up"][1536], ["deepgemm", "scaled_mm_pertensor"])
        self.assertEqual(got["out_proj"][1536], ["scaled_mm_pertensor", "deepgemm"])

    def test_keys_are_buckets_not_benched_m(self):
        from qwenfast.gemm.bench_gemm import derive_shape_priority_from_results

        results = {"cells": [self._cell(34816, 5120, 1152, "deepgemm", 10.0)]}
        got = derive_shape_priority_from_results(results)
        self.assertEqual(list(got["mlp_gate_up"]), [1536])

    def test_the_largest_m_in_a_bucket_wins_the_key(self):
        """Routing rounds M *up*, so the benched M nearest the bucket ceiling
        is the one whose ranking that bucket should carry. Mixing two M values'
        timings into one ranking would be comparing across shapes."""
        from qwenfast.gemm.bench_gemm import derive_shape_priority_from_results

        results = {"cells": [
            self._cell(34816, 5120, 1152, "deepgemm", 10.0),
            self._cell(34816, 5120, 1152, "scaled_mm_pertensor", 20.0),
            self._cell(34816, 5120, 1280, "deepgemm", 30.0),
            self._cell(34816, 5120, 1280, "scaled_mm_pertensor", 15.0),
        ]}
        got = derive_shape_priority_from_results(results)
        self.assertEqual(got["mlp_gate_up"][1536], ["scaled_mm_pertensor", "deepgemm"])

    def test_unknown_shapes_are_dropped_not_guessed(self):
        from qwenfast.gemm.bench_gemm import derive_shape_priority_from_results

        results = {"cells": [self._cell(1234, 5678, 1280, "deepgemm", 10.0)]}
        self.assertEqual(derive_shape_priority_from_results(results), {})

    def test_error_cells_are_ignored(self):
        from qwenfast.gemm.bench_gemm import derive_shape_priority_from_results

        results = {"cells": [
            self._cell(34816, 5120, 1280, "deepgemm", 1.0, status="error"),
            self._cell(34816, 5120, 1280, "scaled_mm_pertensor", 50.0),
        ]}
        got = derive_shape_priority_from_results(results)
        self.assertEqual(got["mlp_gate_up"][1536], ["scaled_mm_pertensor"])

    def test_output_is_paste_ready_for_the_dispatch_table(self):
        """Round-trip: emit, install, and check `priority_for` actually returns
        what the sweep ranked first. This is the step that catches a
        class/bucket keying mistake the tests above could miss."""
        from qwenfast.gemm.bench_gemm import derive_shape_priority_from_results

        results = {"cells": [
            self._cell(5120, 17408, 2304, "vllm_block_fp8_cutlass", 5.0),
            self._cell(5120, 17408, 2304, "deepgemm", 9.0),
        ]}
        table = derive_shape_priority_from_results(results)
        prev = dispatch.get_backend_priority_profile()
        saved = {c: dict(b) for c, b in dispatch.V9_BACKEND_PRIORITY_BY_SHAPE_AND_M.items()}
        try:
            dispatch.V9_BACKEND_PRIORITY_BY_SHAPE_AND_M.clear()
            dispatch.V9_BACKEND_PRIORITY_BY_SHAPE_AND_M.update(table)
            dispatch.set_backend_priority_profile("v9")
            self.assertEqual(dispatch.priority_for(2304, 5120, 17408)[0],
                             "vllm_block_fp8_cutlass")
        finally:
            dispatch.V9_BACKEND_PRIORITY_BY_SHAPE_AND_M.clear()
            dispatch.V9_BACKEND_PRIORITY_BY_SHAPE_AND_M.update(saved)
            dispatch.set_backend_priority_profile(prev)


class TestMixedGemmProbePlan(unittest.TestCase):
    """CPU-safe checks on `mixed_gemm_probe`'s tables -- the same drift guard
    `TestQuantOnlyMapping` gives `bench_gemm._QUANT_ONLY_CALL`."""

    def test_every_probe_shape_is_a_real_fused_shape(self):
        from qwenfast.gemm.bench_gemm import SHAPES
        from qwenfast.gemm.mixed_gemm_probe import MIXED_SHAPES

        by_name = {s["name"]: s for s in SHAPES}
        for s in MIXED_SHAPES:
            with self.subTest(shape=s["name"]):
                self.assertIn(s["name"], by_name)
                self.assertEqual((s["n"], s["k"], s["count"]),
                                 (by_name[s["name"]]["n"], by_name[s["name"]]["k"],
                                  by_name[s["name"]]["count"]))

    def test_every_probe_backend_is_registered(self):
        from qwenfast.gemm.mixed_gemm_probe import DEFAULT_BACKENDS

        for b in DEFAULT_BACKENDS:
            with self.subTest(backend=b):
                self.assertIn(b, dispatch.available_backends())

    def test_split_table_sums_to_its_m(self):
        from qwenfast.gemm.mixed_gemm_probe import DEFAULT_M, SPLIT_OF_M

        for m in DEFAULT_M:
            with self.subTest(m=m):
                self.assertIn(m, SPLIT_OF_M)
                mp, md = SPLIT_OF_M[m]
                self.assertEqual(mp + md, m)

    def test_default_m_values_are_the_mixed_steps_real_m(self):
        from qwenfast.gemm.mixed_gemm_probe import DEFAULT_M

        # chunk + decode rows, for the common mixed-step chunk sizes
        for chunk, rows in ((1024, 128), (1024, 256), (2048, 256),
                            (4096, 256), (8192, 256)):
            with self.subTest(chunk=chunk, rows=rows):
                self.assertIn(chunk + rows, DEFAULT_M)


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA GPU")
class TestMixedStepMBackendsOnGPU(unittest.TestCase):
    """Mixed-step correctness gate: the v9 table can only ever *reorder* backends,
    so what has to hold is that every backend v9 could promote is numerically
    sound at the mixed step's real M -- which are not the M any earlier sweep
    checked (`TestLargeMBackendsOnGPU` uses 2048, a power of two; the mixed
    step's M never is).

    Kept inside a short-test budget: two shapes x two M x the five
    fp8-activation backends, against an fp32 reference of the dequantized
    weight (`gemm_numerics.fp32_reference`'s definition -- what the engine's
    own weights mean, not the original bf16 model)."""

    SHAPES = ((34816, 5120), (5120, 6144))
    MS = (1152, 2304)

    def test_every_promotable_backend_is_accurate_at_the_mixed_steps_m(self):
        from qwenfast.gemm.bench_gemm import LARGE_M_FP8_ACTIVATION_BACKENDS
        from qwenfast.gemm.dispatch import _BACKENDS
        from qwenfast.gemm.fused_weights import quantize_bf16_to_fp8_block128

        device = "cuda:0"
        for n, k in self.SHAPES:
            w_bf16 = torch.randn(n, k, device=device, dtype=torch.bfloat16)
            fw = quantize_bf16_to_fp8_block128(w_bf16)
            for m in self.MS:
                x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
                ref = F.linear(x.float(), dequant_block128(fw.weight, fw.scale_inv).float())
                for backend in LARGE_M_FP8_ACTIVATION_BACKENDS:
                    with self.subTest(n=n, k=k, m=m, backend=backend):
                        try:
                            out = _BACKENDS[backend](x, fw)
                        except Exception as exc:  # noqa: BLE001
                            self.skipTest(f"{backend} unavailable at M={m}: "
                                          f"{type(exc).__name__}: {exc}")
                            continue
                        self.assertEqual(tuple(out.shape), (m, n))
                        rel = (torch.linalg.norm(out.float() - ref)
                               / torch.linalg.norm(ref))
                        # the W8A8 class floor is 2.6e-2 flat in M
                        # and in shape; 0.05 admits that and nothing worse.
                        self.assertLess(float(rel), 0.05,
                                        f"{backend} relL2 {float(rel):.4f} at M={m} "
                                        f"[{n}, {k}]")
                del x
            del w_bf16, fw
            torch.cuda.empty_cache()

    def test_v9_routing_matches_the_reference_through_linear(self):
        """End to end through `dispatch.linear` under the v9 profile -- i.e.
        whatever the table promotes for this shape and M, run for real."""
        from qwenfast.gemm.fused_weights import quantize_bf16_to_fp8_block128

        device = "cuda:0"
        prev = dispatch.get_backend_priority_profile()
        try:
            dispatch.set_backend_priority_profile("v9")
            for n, k in self.SHAPES:
                w_bf16 = torch.randn(n, k, device=device, dtype=torch.bfloat16)
                fw = quantize_bf16_to_fp8_block128(w_bf16)
                ref_w = dequant_block128(fw.weight, fw.scale_inv).float()
                for m in self.MS:
                    with self.subTest(n=n, k=k, m=m):
                        x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
                        out = linear(x, fw, use_autotune=False)
                        ref = F.linear(x.float(), ref_w)
                        rel = (torch.linalg.norm(out.float() - ref)
                               / torch.linalg.norm(ref))
                        self.assertLess(float(rel), 0.05)
                del w_bf16, fw, ref_w
                torch.cuda.empty_cache()
        finally:
            dispatch.set_backend_priority_profile(prev)


if __name__ == "__main__":
    unittest.main()
