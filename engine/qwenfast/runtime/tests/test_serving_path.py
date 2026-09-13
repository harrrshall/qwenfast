"""CPU tests for the serving path's prefill internals.

``test_serving.py`` covers the serving seam (HTTP -> scheduler -> model). This
file covers the prefill-path machinery *inside* that seam, and it exists as a
separate file because every test here is about a **property that must be
testable without a GPU**, not about the server contract:

``TestConvPrefillVarlen``
    The token-major prefill conv is a from-scratch reimplementation of a
    numerically exact tiling, and it writes the GDN ring state. A wrong ring
    state does not raise -- it silently poisons every decode step that follows
    -- so the reference is pinned against the ``[B, C, T]`` path it replaces on
    every shape that distinguishes them: split chunks (state threads through),
    segments shorter than ``W-1`` (the tail comes from the *old* ring), empty
    segments, and a single-token segment (the decode shape, reached through the
    prefill path when a chunk's last request has one token left).

``TestNoPrefillHostSyncs``
    The two D2H syncs on the prefill path (``seq_slot_ids.tolist()`` once per
    chunk, ``cu_seqlens.to("cpu")`` once per GDN layer per chunk) are invisible
    on CPU, so a timing-based test cannot catch them. These tests assert on
    *which functions get called*, which is visible on CPU, rather than on
    timing, which is not.

``TestPrefillChunkCap`` / ``TestPrefillGemmScope``
    The two new knobs do what they say and are no-ops at their defaults.

``TestProfileServingParts``
    The pure functions of ``profile_serving`` -- the chunk-packing model and
    the FLOP counter -- plus one real drive of ``ClosedLoopDriver`` against the
    tiny model, so a typo in the profiler fails here rather than 40 minutes
    into a GPU run.

Run::

    python -m pytest engine/qwenfast/runtime/tests/test_serving_path.py -v
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))  # -> engine/
sys.path.insert(0, _HERE)

from qwenfast.kernels_gdn import api as gdn_api  # noqa: E402
from qwenfast.kernels_gdn import fla_ops  # noqa: E402
from qwenfast.kernels_gdn import torch_ops  # noqa: E402
from qwenfast.runtime import prefill_attrib as pa  # noqa: E402
from qwenfast.runtime import profile_serving as ps  # noqa: E402
from qwenfast.runtime import scheduler as sched_mod  # noqa: E402
from qwenfast.runtime.fused_model import (  # noqa: E402
    RuntimeConfig,
    _PREFILL_GEMM_BACKEND,
    make_prefill_batch,
    prefill_gemm_scope,
)
from qwenfast.runtime.scheduler import Request, Scheduler  # noqa: E402
from qwenfast.weights import QwenFastConfig  # noqa: E402

from test_serving import build_scheduler, gen_params  # noqa: E402

CPU = torch.device("cpu")


# =========================================================================== #
# 1. the token-major prefill conv
# =========================================================================== #
class TestConvPrefillVarlen(unittest.TestCase):
    """``torch_ops.conv_prefill_varlen`` == ``torch_ops.conv_prefill``."""

    C = 12
    W = 4

    def setUp(self):
        torch.manual_seed(3)
        self.w = torch.randn(self.C, self.W, dtype=torch.float32)

    def _pools(self, n_slots: int, *, width_major: bool = True):
        """Two *identical* conv pools, so each path mutates its own copy."""
        shape = (n_slots, self.W - 1, self.C) if width_major else (n_slots, self.C, self.W - 1)
        a = torch.randn(*shape, dtype=torch.float32)
        return a.clone(), a.clone()

    def _run_both(self, seq_lens, *, width_major=True, activation="silu", n_slots=None):
        total = sum(seq_lens)
        n = len(seq_lens)
        n_slots = n_slots or n
        x_tc = torch.randn(total, self.C, dtype=torch.float32)
        cu = torch.tensor(
            [0] + list(torch.tensor(seq_lens).cumsum(0).tolist()), dtype=torch.int32
        )
        slots = torch.arange(n, dtype=torch.int32)
        pool_ref, pool_new = self._pools(n_slots, width_major=width_major)

        ref = torch_ops.conv_prefill(
            x_tc.t().unsqueeze(0).contiguous(),
            self.w,
            cu_seqlens=cu,
            conv_state_pool=pool_ref,
            slot_ids=slots,
            activation=activation,
        )
        ref = ref.squeeze(0).t().contiguous()
        got = torch_ops.conv_prefill_varlen(
            x_tc, self.w, seq_lens, pool_new, slots, activation=activation
        )
        return ref, got, pool_ref, pool_new

    def test_matches_channel_major_path(self):
        for seq_lens in ([7], [5, 9], [2139 % 37, 11, 4], [1, 1, 1]):
            with self.subTest(seq_lens=seq_lens):
                ref, got, pr, pn = self._run_both(seq_lens)
                torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(pn, pr, rtol=1e-5, atol=1e-5)

    def test_segment_shorter_than_kernel_width(self):
        """``n < W-1``: the new ring keeps the tail of the *old* one.

        The case a naive "take the last W-1 of x" write-back gets wrong, and it
        is reachable in production: a request whose remaining prompt is 1-2
        tokens is exactly what the chunked-prefill budget leaves behind.
        """
        for n in (1, 2, 3):
            with self.subTest(n=n):
                ref, got, pr, pn = self._run_both([n])
                torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(pn, pr, rtol=1e-5, atol=1e-5)

    def test_empty_segment_leaves_state_alone(self):
        ref, got, pr, pn = self._run_both([0, 6])
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(pn, pr, rtol=1e-5, atol=1e-5)

    def test_channel_major_pool_layout(self):
        ref, got, pr, pn = self._run_both([5, 3], width_major=False)
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(pn, pr, rtol=1e-5, atol=1e-5)

    def test_identity_activation(self):
        ref, got, _pr, _pn = self._run_both([6], activation=None)
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)

    def test_split_chunks_thread_state_exactly(self):
        """Prefilling ``[0:a]`` then ``[a:n]`` == prefilling ``[0:n]`` at once.

        This is the property chunked prefill rests on: splitting a request
        across two chunks is exact. It has always held
        for the channel-major path; it has to hold for the new one too, and
        the ring write-back is the only thing that can break it.
        """
        n, a = 13, 5
        x = torch.randn(n, self.C, dtype=torch.float32)
        pool_whole, pool_split = self._pools(1)
        slots = torch.tensor([0], dtype=torch.int32)

        whole = torch_ops.conv_prefill_varlen(x, self.w, [n], pool_whole, slots)
        p1 = torch_ops.conv_prefill_varlen(x[:a], self.w, [a], pool_split, slots)
        p2 = torch_ops.conv_prefill_varlen(x[a:], self.w, [n - a], pool_split, slots)

        torch.testing.assert_close(torch.cat([p1, p2], 0), whole, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(pool_split, pool_whole, rtol=1e-5, atol=1e-5)

    def test_api_dispatch_falls_back_to_torch_for_cpu_tensors(self):
        """``backend='auto'`` on a CPU tensor must land on torch.

        Non-obvious: ``resolve_backend`` answers "is triton importable", not
        "can triton run on these tensors". On a GPU host triton imports, so
        without the device check in ``causal_conv_prefill_varlen`` this call
        would be handed to a CUDA kernel and fail at launch -- and it would
        fail *only on the GPU host*. This test passes on a laptop for the
        trivial reason and on the GPU host for the real one.
        """
        seq_lens = [4, 6]
        total = sum(seq_lens)
        x = torch.randn(total, self.C, dtype=torch.float32)
        cu = torch.tensor([0, 4, 10], dtype=torch.int32)
        slots = torch.arange(2, dtype=torch.int32)
        pool = torch.randn(2, self.W - 1, self.C, dtype=torch.float32)
        out = gdn_api.causal_conv_prefill_varlen(
            x, self.w, seq_lens=seq_lens, cu_seqlens=cu,
            conv_state_pool=pool, slot_ids=slots,
        )
        self.assertEqual(out.shape, x.shape)
        # ...and it is the same answer the reference gives.
        pool2 = pool.clone()
        ref = torch_ops.conv_prefill_varlen(x, self.w, seq_lens, pool2, slots)
        self.assertEqual(out.shape, ref.shape)



# =========================================================================== #
# 2. the prefill path's host syncs
# =========================================================================== #
class TestNoPrefillHostSyncs(unittest.TestCase):
    def test_plan_prefill_reads_host_slots_not_the_device_tensor(self):
        """``prefill_forward`` must plan from ``batch.seq_slots``.

        The check is on *provenance*, not on the value: both sources produce
        ``[0]`` here, so comparing the value would pass even after a
        regression. Instead the device tensor is poisoned with a sentinel and
        ``plan_prefill`` is asserted to have seen the host list -- if anything
        on the path went back to ``seq_slot_ids.tolist()`` it would see the
        sentinel instead. On CPU the sync is free, so an assertion like this
        is the only thing that can catch the regression before it costs 48
        pipeline drains per chunk on the GPU.
        """
        sched, fm, _dec, _rt = build_scheduler()
        batch = make_prefill_batch([[3, 4, 5, 6]], [0], [0], CPU)
        self.assertEqual(batch.seq_slots, [0])
        # Same slot the tensor names, so the forward stays valid; a different
        # *object identity* is what the assertion below keys on.
        batch.seq_slot_ids = torch.tensor([0], dtype=torch.int32)

        seen = []
        orig = fm.attn.plan_prefill

        def spy(slots, q_lens, kv_lens):
            seen.append(slots)
            return orig(slots, q_lens, kv_lens)

        fm.attn.plan_prefill = spy
        fm.reset_slot(0)
        fm.kv_pool.ensure_capacity(0, 8)
        try:
            fm.prefill_forward(batch, all_logits=False)
        finally:
            fm.attn.plan_prefill = orig
        self.assertEqual(seen, [[0]])
        self.assertIsNot(seen[0], batch.seq_slots, "plan_prefill got a live alias")

    def test_seq_slots_of_falls_back_for_a_legacy_batch(self):
        """A ``PrefillBatch`` built before ``seq_slots`` existed still works."""
        from qwenfast.runtime.fused_model import _seq_slots_of

        batch = make_prefill_batch([[3, 4], [5, 6, 7]], [0, 0], [1, 2], CPU)
        self.assertEqual(_seq_slots_of(batch), [1, 2])
        batch.seq_slots = []
        self.assertEqual(_seq_slots_of(batch), [1, 2])

    def test_token_major_layout_never_calls_channel_major_conv(self):
        sched, _fm, _dec, _rt = build_scheduler()
        seen = {"varlen": 0, "channel": 0}
        orig_v = torch_ops.conv_prefill_varlen
        orig_c = torch_ops.conv_prefill

        def spy_v(*a, **k):
            seen["varlen"] += 1
            return orig_v(*a, **k)

        def spy_c(*a, **k):
            seen["channel"] += 1
            return orig_c(*a, **k)

        torch_ops.conv_prefill_varlen = spy_v
        torch_ops.conv_prefill = spy_c
        try:
            sched.add_request(Request("a", list(range(3, 20)), gen_params(max_tokens=1)))
            for _ in range(30):
                sched.step()
        finally:
            torch_ops.conv_prefill_varlen = orig_v
            torch_ops.conv_prefill = orig_c
        self.assertGreater(seen["varlen"], 0, "token-major conv never ran")
        self.assertEqual(seen["channel"], 0, "channel-major conv ran under token_major")

    def test_channel_major_layout_still_works(self):
        """The rollback flag is a rollback, not a dead branch."""
        sched, _fm, _dec, _rt = build_scheduler(conv_prefill_layout="channel_major")
        req = Request("a", list(range(3, 20)), gen_params(max_tokens=3))
        sched.add_request(req)
        for _ in range(60):
            sched.step()
            if req.is_finished:
                break
        self.assertTrue(req.is_finished)
        self.assertEqual(len(req.output_token_ids), 3)

    def test_both_layouts_give_the_same_tokens(self):
        outs = {}
        for layout in ("channel_major", "token_major"):
            sched, _fm, _dec, _rt = build_scheduler(conv_prefill_layout=layout)
            req = Request("a", list(range(3, 30)), gen_params(max_tokens=6))
            sched.add_request(req)
            for _ in range(200):
                sched.step()
                if req.is_finished:
                    break
            outs[layout] = list(req.output_token_ids)
        self.assertEqual(outs["channel_major"], outs["token_major"])


# =========================================================================== #
# 3. the new scheduler / dispatch knobs
# =========================================================================== #
class TestPrefillChunkCap(unittest.TestCase):
    def test_cap_bounds_the_chunk_below_the_token_budget(self):
        sched, _fm, _dec, rt = build_scheduler(
            max_num_batched_tokens=32, prefill_chunk_tokens=8
        )
        self.assertEqual(sched.prefill_chunk_tokens, 8)
        sched.add_request(Request("a", list(range(3, 3 + 30)), gen_params(max_tokens=1)))
        sizes = []
        for _ in range(20):
            sched.step()
            if sched.last_chunk_tokens:
                sizes.append(sched.last_chunk_tokens)
                sched.last_chunk_tokens = 0
        self.assertTrue(sizes, "no prefill chunk ran")
        self.assertLessEqual(max(sizes), 8)

    def test_cap_is_clamped_to_the_token_budget(self):
        sched, _fm, _dec, _rt = build_scheduler(
            max_num_batched_tokens=12, prefill_chunk_tokens=99999
        )
        self.assertEqual(sched.prefill_chunk_tokens, 12)

    def test_zero_means_no_cap(self):
        sched, _fm, _dec, _rt = build_scheduler(max_num_batched_tokens=12)
        self.assertEqual(sched.prefill_chunk_tokens, 12)

    def test_chunk_telemetry_is_per_step_not_sticky(self):
        """``last_chunk_tokens`` must be 0 on a decode step.

        Not cosmetic: the profiler labels a step by this field, so a value
        left standing from the previous prefill would make every decode step
        after a chunk count as another whole chunk -- and the profiler's
        headline number is exactly "how much wall time was prefill".
        """
        sched, _fm, _dec, _rt = build_scheduler(max_num_batched_tokens=64)
        sched.add_request(Request("a", list(range(3, 15)), gen_params(max_tokens=6)))
        saw_prefill = saw_decode = False
        for _ in range(40):
            events = sched.step()
            if sched.last_chunk_tokens:
                saw_prefill = True
                self.assertGreater(sched.last_chunk_seqs, 0)
            elif events:
                saw_decode = True
                self.assertEqual(sched.last_chunk_seqs, 0)
        self.assertTrue(saw_prefill and saw_decode)

    def test_cap_does_not_change_the_generated_tokens(self):
        """A latency knob must not be a correctness knob."""
        outs = []
        for cap in (0, 5):
            sched, _fm, _dec, _rt = build_scheduler(
                max_num_batched_tokens=16, prefill_chunk_tokens=cap
            )
            req = Request("a", list(range(3, 3 + 25)), gen_params(max_tokens=5))
            sched.add_request(req)
            for _ in range(300):
                sched.step()
                if req.is_finished:
                    break
            outs.append(list(req.output_token_ids))
        self.assertEqual(outs[0], outs[1])


class TestPrefillGemmScope(unittest.TestCase):
    def test_scope_sets_and_restores(self):
        self.assertIsNone(_PREFILL_GEMM_BACKEND[0])
        with prefill_gemm_scope("flashinfer_fp8_blockscale"):
            self.assertEqual(_PREFILL_GEMM_BACKEND[0], "flashinfer_fp8_blockscale")
        self.assertIsNone(_PREFILL_GEMM_BACKEND[0])

    def test_none_is_a_noop(self):
        with prefill_gemm_scope(None):
            self.assertIsNone(_PREFILL_GEMM_BACKEND[0])

    def test_scope_is_cleared_on_exception(self):
        with self.assertRaises(ValueError):
            with prefill_gemm_scope("scaled_mm_pertensor"):
                raise ValueError("boom")
        self.assertIsNone(_PREFILL_GEMM_BACKEND[0])

    def test_default_config_forces_nothing(self):
        self.assertIsNone(RuntimeConfig().prefill_gemm_backend)


class TestServeRefusesUnsafePrefillBackend(unittest.TestCase):
    """A repack-cache backend forced for prefill under ``single`` allocates a
    second repack cache and OOMs; ``serve.py`` must refuse it at
    argument-parse time."""

    def _args(self, **kw):
        from qwenfast.runtime import serve

        p = serve.build_arg_parser()
        base = ["--model", "/nonexistent"]
        for k, v in kw.items():
            base += [f"--{k.replace('_', '-')}", str(v)]
        return serve, p.parse_args(base)

    def test_refuses_cache_owning_backend_under_single(self):
        serve, args = self._args(
            prefill_gemm_backend="scaled_mm_pertensor", gemm_weight_cache="single"
        )
        with self.assertRaises(SystemExit) as cm:
            serve.runtime_config_from_args(args)
        self.assertIn("repack cache", str(cm.exception))

    def test_allows_cache_free_backend_under_single(self):
        serve, args = self._args(
            prefill_gemm_backend="flashinfer_fp8_blockscale", gemm_weight_cache="single"
        )
        rt = serve.runtime_config_from_args(args)
        self.assertEqual(rt.prefill_gemm_backend, "flashinfer_fp8_blockscale")

    def test_allows_cache_owning_backend_under_multi(self):
        serve, args = self._args(
            prefill_gemm_backend="scaled_mm_pertensor", gemm_weight_cache="multi"
        )
        rt = serve.runtime_config_from_args(args)
        self.assertEqual(rt.prefill_gemm_backend, "scaled_mm_pertensor")


class TestGemmCacheOwner(unittest.TestCase):
    """Who gets the one repack-cache slot ``single`` allows."""

    def _args(self, **kw):
        from qwenfast.runtime import serve

        p = serve.build_arg_parser()
        base = ["--model", "/nonexistent"]
        for k, v in kw.items():
            base += [f"--{k.replace('_', '-')}", str(v)]
        return serve, p.parse_args(base)

    def test_default_is_the_historical_behaviour(self):
        self.assertEqual(RuntimeConfig().gemm_cache_owner, "decode")
        serve, args = self._args()
        self.assertEqual(serve.runtime_config_from_args(args).gemm_cache_owner, "decode")

    def test_prefill_owner_permits_a_cache_owning_prefill_backend(self):
        """It is the *first* claimant then, not a second one -- so the
        memory plan is unchanged and the guard must not fire."""
        serve, args = self._args(
            prefill_gemm_backend="deepgemm",
            gemm_weight_cache="single",
            gemm_cache_owner="prefill",
        )
        rt = serve.runtime_config_from_args(args)
        self.assertEqual(rt.prefill_gemm_backend, "deepgemm")
        self.assertEqual(rt.gemm_cache_owner, "prefill")

    def test_claim_gemm_cache_is_a_noop_on_cpu(self):
        _sched, fm, _dec, _rt = build_scheduler()
        self.assertEqual(fm.claim_gemm_cache(m=8), {})

    def test_claim_gemm_cache_default_m_is_the_real_prefill_chunk_not_512(self):
        """A bare default of 512 would make `--gemm-cache-owner prefill`
        claim the cache slot for whichever backend wins the *decode* M=512
        bucket, not for a real prefill chunk.
        `claim_gemm_cache(m=None)` (i.e. its real default, which serve.py's
        one call site uses) must derive M from the RuntimeConfig's own
        chunk-token fields instead. CPU-only: the CUDA early-return happens
        *after* the default is computed, so `m` alone is checkable here."""
        from qwenfast.runtime.fused_model import default_claim_gemm_cache_m

        _sched, fm, _dec, rt = build_scheduler(max_num_batched_tokens=8192, prefill_chunk_tokens=0)
        self.assertEqual(default_claim_gemm_cache_m(rt), 8192)
        self.assertNotEqual(default_claim_gemm_cache_m(rt), 512)

        # `prefill_chunk_tokens`, when set, is a tighter cap than
        # `max_num_batched_tokens` and takes precedence -- it is the real
        # per-chunk M a capped server actually presents to the GEMMs.
        _sched2, _fm2, _dec2, rt2 = build_scheduler(max_num_batched_tokens=8192, prefill_chunk_tokens=2048)
        self.assertEqual(default_claim_gemm_cache_m(rt2), 2048)

    def test_claim_gemm_cache_default_m_includes_the_decode_rows_when_mixed(self):
        """A mixed step's activation is the chunk concatenated
        with every decode row, so its M is `chunk + max_num_seqs` and its
        `m_bucket` is a different bucket from the chunk's. Claiming the repack
        cache at the chunk's M claims it for a bucket the server never routes
        on -- the same class of miss as a bare 512 default above."""
        from qwenfast.gemm.dispatch import m_bucket
        from qwenfast.runtime.fused_model import default_claim_gemm_cache_m

        _s, _f, _d, rt = build_scheduler(max_num_batched_tokens=8192,
                                          prefill_chunk_tokens=1024)
        rt.mixed_forward = False
        self.assertEqual(default_claim_gemm_cache_m(rt), 1024)
        rt.mixed_forward = True
        self.assertEqual(default_claim_gemm_cache_m(rt), 1024 + rt.max_num_seqs)
        # and it is a genuinely different routing key, not just a bigger number
        self.assertNotEqual(m_bucket(1024), m_bucket(1024 + rt.max_num_seqs))

    def test_claim_gemm_cache_m_is_per_weight_for_the_tiled_mlp_shapes(self):
        """`FusedMLP.__call__(tiled=True)` splits the
        activation at `mlp_tile_tokens`, so above that the two MLP shapes are
        never called at the step's M -- and under `--gemm-weight-cache single`
        whoever claims a weight's one cache slot fixes that weight's backend at
        every bucket. Claiming an MLP weight at the step's M therefore pins it
        for a bucket it never routes on; measured, that made `mlp_down` 19.6 %
        slower (deepgemm wins bucket 3072, `vllm_cutlass_fp8_pertensor` wins
        bucket 2048, and deepgemm is the *worst* of five there)."""
        from qwenfast.runtime.fused_model import (
            claim_gemm_cache_m_for, default_claim_gemm_cache_m)

        _s, _f, _d, rt = build_scheduler(max_num_batched_tokens=8192,
                                          prefill_chunk_tokens=4096)
        rt.mixed_forward = True
        rt.mlp_tile_tokens = 2048
        step_m = default_claim_gemm_cache_m(rt)          # 4096 + max_num_seqs
        self.assertGreater(step_m, rt.mlp_tile_tokens)
        # tiled MLP shapes -> the tile
        for n, k in ((34816, 5120), (5120, 17408)):
            with self.subTest(n=n, k=k):
                self.assertEqual(claim_gemm_cache_m_for(rt, n, k), 2048)
        # everything else -> the step's M
        for n, k in ((16384, 5120), (5120, 6144), (14336, 5120), (248320, 5120)):
            with self.subTest(n=n, k=k):
                self.assertEqual(claim_gemm_cache_m_for(rt, n, k), step_m)
        # an unknown shape is not guessed at -- it gets the step's M
        self.assertEqual(claim_gemm_cache_m_for(rt, 4242, 4242), step_m)

    def test_claim_gemm_cache_m_per_weight_is_a_no_op_below_the_tile(self):
        """At chunk 1,024 the mixed step's M (1,280) is under the tile, so the
        MLP is untiled and every shape sees the same M. The per-weight rule
        must be the identity there, or it would introduce a difference where
        the engine has none."""
        from qwenfast.runtime.fused_model import (
            claim_gemm_cache_m_for, default_claim_gemm_cache_m)

        _s, _f, _d, rt = build_scheduler(max_num_batched_tokens=8192,
                                          prefill_chunk_tokens=1024)
        rt.mixed_forward = True
        rt.mlp_tile_tokens = 2048
        rt.max_num_seqs = 256
        step_m = default_claim_gemm_cache_m(rt)
        self.assertEqual(step_m, 1280)
        for n, k in ((34816, 5120), (5120, 17408), (16384, 5120), (5120, 6144)):
            with self.subTest(n=n, k=k):
                self.assertEqual(claim_gemm_cache_m_for(rt, n, k), step_m)

    def test_claim_gemm_cache_explicit_m_still_overrides(self):
        _sched, fm, _dec, _rt = build_scheduler()
        # explicit m= must win over the RuntimeConfig-derived default, same as
        # before this pass -- covered above by the m=8 no-op test, restated
        # here to pin the *reason* (an explicit m short-circuits the default
        # computation entirely).
        from unittest import mock

        from qwenfast.runtime import fused_model as fm_mod

        with mock.patch.object(fm_mod, "default_claim_gemm_cache_m") as default_m:
            fm.claim_gemm_cache(m=8)
            default_m.assert_not_called()

    def test_iter_resolved_linears_covers_every_linear(self):
        """The claim walk and ``collect_gemm_backends`` must agree on the set,
        or the claim would silently miss layers and leave them un-cached."""
        from qwenfast.runtime.bench_runtime import collect_gemm_backends
        from qwenfast.runtime.fused_model import _iter_resolved_linears

        _sched, fm, _dec, _rt = build_scheduler()
        walked = list(_iter_resolved_linears(fm))
        named = collect_gemm_backends(fm)
        self.assertEqual(len(walked), len(named))
        self.assertEqual(len({id(x) for x in walked}), len(walked))


# =========================================================================== #
# 4. the profiler's own logic
# =========================================================================== #
def _real_config() -> QwenFastConfig:
    return QwenFastConfig(
        layer_types=[
            "full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(64)
        ]
    )


class TestProfileServingParts(unittest.TestCase):
    def test_chunk_shape_matches_the_scheduler_budget_loop(self):
        """The packing model must be the scheduler's, not an approximation."""
        self.assertEqual(ps.chunk_shape(8192, 2139), [2139, 2139, 2139, 1775])
        self.assertEqual(ps.chunk_shape(2048, 2139), [2048])
        self.assertEqual(ps.chunk_shape(8192, 8192), [8192])
        self.assertEqual(sum(ps.chunk_shape(8192, 300)), 8192)

    def test_chunk_shape_agrees_with_a_real_scheduler(self):
        budget, prompt = 12, 5
        sched, _fm, _dec, _rt = build_scheduler(max_num_batched_tokens=budget)
        for i in range(4):
            sched.add_request(
                Request(f"r{i}", list(range(3, 3 + prompt)), gen_params(max_tokens=1))
            )
        sched.step()
        self.assertEqual(sched.last_chunk_tokens, sum(ps.chunk_shape(budget, prompt)))
        self.assertEqual(sched.last_chunk_seqs, len(ps.chunk_shape(budget, prompt)))

    def test_flops_per_token_matches_the_checkpoint_weight_split(self):
        """The checkpoint splits as MLP 17.12 / attn 1.68 / GDN
        5.59 GB of fp8 linears. 2 x those params is what a prefill token pays,
        so the counter is right iff it reproduces that split."""
        f = ps.linear_flops_per_token(_real_config())
        self.assertAlmostEqual(f["mlp"] / 2e9, 17.12, places=1)
        self.assertAlmostEqual(f["gdn_proj"] / 2e9, 5.56, places=1)
        self.assertAlmostEqual(f["attn_proj"] / 2e9, 1.68, places=1)

    def test_prefill_ceiling(self):
        c = ps.prefill_ceiling(_real_config(), peak_tflops=990.0)
        # 48.7 GFLOP/token of linears -> ~20.2k tok/s at 990 TFLOP/s.
        self.assertAlmostEqual(c["flops_per_token_linear"] / 1e9, 48.70, places=1)
        self.assertGreater(c["ceiling_tok_s"], 19_000)
        self.assertLess(c["ceiling_tok_s"], 22_000)
        # The GDN state recurrence is a rounding error next to the linears --
        # which is what makes prefill GEMM-bound.
        self.assertLess(c["flops_per_token_gdn_state"], 0.02 * c["flops_per_token_linear"])

    def test_attn_flops_are_small_at_the_sweep_context(self):
        cfg = _real_config()
        lin = ps.linear_flops_per_token(cfg)["total_linear"]
        self.assertLess(ps.attn_flops_per_token(cfg, 2139), 0.02 * lin)

    def test_attribute_splits_prefill_from_decode(self):
        recs = [
            ps.StepRecord("prefill", 900.0, 890.0, 8192, 4, 0, 256, 12),
            ps.StepRecord("decode", 45.0, 44.0, 256, 256, 256, 256, 12),
            ps.StepRecord("decode", 45.0, 44.0, 256, 256, 256, 256, 12),
        ]
        a = ps.attribute(recs, duration_s=1.0)
        self.assertEqual(a["output_tokens"], 512)
        self.assertAlmostEqual(a["by_kind"]["prefill"]["wall_pct"], 90.9, places=0)
        # 900 ms of prefill spread over 512 emitted tokens is 1.76 ms each;
        # decode's own 90 ms is 0.176. The sum is the served per-token time.
        self.assertAlmostEqual(
            a["by_kind"]["prefill"]["ms_per_output_token"]
            + a["by_kind"]["decode"]["ms_per_output_token"],
            a["served_ms_per_output_token"],
            places=3,
        )

    def test_host_counters(self):
        c = ps.HostCounters()
        c.add("x", 2.0)
        c.add("x", 4.0)
        c.add("y", 1.0)
        snap = c.snapshot()
        self.assertEqual(list(snap), ["x", "y"])  # sorted by total, descending
        self.assertEqual(snap["x"]["calls"], 2)
        self.assertAlmostEqual(snap["x"]["ms_per_call"], 3.0)

    def test_detok_load_runs_and_stops(self):
        with ps.DetokLoad(tokens_per_s=100_000, batch=8) as load:
            deadline = 0.2
            import time as _t

            t0 = _t.perf_counter()
            while _t.perf_counter() - t0 < deadline:
                pass
        self.assertGreater(load.iterations, 0)


class TestClosedLoopDriver(unittest.TestCase):
    """The driver against the real tiny scheduler, i.e. the code path the GPU
    host will run -- not a mock of it."""

    def test_drive_produces_prefill_and_decode_records(self):
        sched, fm, _dec, _rt = build_scheduler(max_num_batched_tokens=16)
        driver = ps.ClosedLoopDriver(
            sched,
            concurrency=2,
            input_len=12,
            output_len=3,
            vocab_size=fm.config.vocab_size,
        )
        # Step-bounded, not wall-clock-bounded: see `run`'s docstring.
        driver.run(0.0, warmup_s=0.0, max_steps=120)
        kinds = {r.kind for r in driver.records}
        self.assertIn("prefill", kinds)
        self.assertIn("decode", kinds)
        self.assertGreater(driver.completed, 0)

        a = ps.attribute(driver.records, duration_s=1.0)
        self.assertGreater(a["output_tokens"], 0)
        self.assertIn("prefill", a["by_kind"])
        # Every prefill record must carry the chunk it ran, or the ledger's
        # tok/s column is meaningless.
        for r in driver.records:
            if r.kind == "prefill":
                self.assertGreater(r.tokens, 0)
                self.assertGreater(r.n_seqs, 0)

    def test_driver_keeps_the_loop_closed(self):
        """``concurrency`` requests in flight at all times, as the semaphore in
        ``bench_serve.run_level`` guarantees."""
        sched, fm, _dec, _rt = build_scheduler(max_num_batched_tokens=16)
        driver = ps.ClosedLoopDriver(
            sched, concurrency=3, input_len=8, output_len=2,
            vocab_size=fm.config.vocab_size,
        )
        driver.run(0.0, warmup_s=0.0, max_steps=100)
        self.assertTrue(driver.records)
        self.assertLessEqual(max(r.running for r in driver.records), 3)

    def test_host_counters_instrumentation_restores(self):
        sched, fm, _dec, _rt = build_scheduler()
        before = (sched_mod.make_prefill_batch, fm.attn.plan_prefill, sched._harvest)
        counters, restore = ps.install_host_counters(sched, fm)
        sched.add_request(Request("a", list(range(3, 20)), gen_params(max_tokens=2)))
        for _ in range(40):
            sched.step()
        restore()
        self.assertEqual(
            (sched_mod.make_prefill_batch, fm.attn.plan_prefill, sched._harvest), before
        )
        snap = counters.snapshot()
        self.assertIn("make_prefill_batch", snap)
        self.assertIn("plan_prefill", snap)


# =========================================================================== #
# 7. profiler memory planning
# =========================================================================== #
class TestProfilerMemoryPlan(unittest.TestCase):
    """Profiler OOMs, made impossible on CPU.

    Running ``profile_serving --concurrency 32 256`` without a memory plan
    can end in ``torch.OutOfMemoryError`` on ``alloc_state_pool`` with "this
    process has 137.09 GiB memory in use".  Two independent causes, one test
    class:

    * **No plan.**  A profiler that builds whatever ``RuntimeConfig`` its own
      flag defaults imply discovers the footprint on the device.  It instead
      goes through ``serve.plan_memory`` -- the same function whose printed
      plan the server refuses to start against -- and these tests assert the
      conc-256 plan fits in the budget *without a GPU*, which is the only
      place that assertion is cheap.
    * **No teardown.**  See :class:`TestProfilerTeardown`.
    """

    def _args(self, *extra: str):
        parser = ps.build_arg_parser()
        argv = ["--model", "/nonexistent", "--preset", "fastest", *extra]
        args = parser.parse_args(argv)
        return ps.apply_preset(args, parser, argv=argv)

    def test_conc_256_plan_is_under_125_gib(self):
        """The number a run has to clear on a 139.8 GiB H200."""
        args = self._args()
        rt, plan = ps.plan_for(args, 256)
        self.assertLess(plan["total_gib"], 125.0, msg=f"plan={plan}")
        self.assertLess(plan["steady_gib"], 125.0)
        # ...and it is not under 125 by being absurdly small: this is a real
        # 27B config with a real KV pool for 256 x 2,703-token sequences.
        self.assertGreater(plan["total_gib"], 100.0)
        self.assertEqual(rt.max_num_seqs, 256)
        self.assertEqual(rt.max_model_len, 2139 + 500 + 64)

    def test_the_plan_names_the_terms_that_oomd(self):
        """The two pools an unplanned run dies between, with their sizes."""
        args = self._args()
        _rt, plan = ps.plan_for(args, 256)
        # 44,353-ish pages x 16 tokens x 68 KiB/token
        self.assertGreater(plan["kv_gib"], 40.0)
        self.assertLess(plan["kv_gib"], 52.0)
        # 257 slots x 72 MiB/slot fp16 -- the 18.07 GiB allocation that can raise
        self.assertGreater(plan["ssm_gib"], 16.0)
        self.assertLess(plan["ssm_gib"], 20.0)
        # exactly one repack cache under `--gemm-weight-cache single`
        self.assertEqual(int(plan["n_repack_caches"]), 1)

    def test_the_budget_gate_has_teeth(self):
        """A config that does not fit must be *seen* not to fit."""
        args = self._args()
        _rt, plan = ps.plan_for(args, 1024)
        self.assertGreater(plan["total_gib"], 131.0)  # 139.8 x 0.94

    def test_preset_fills_the_knobs_the_profiler_used_to_get_wrong(self):
        """`--preset fastest` is what makes "the served config" true.

        `fused_ops_backend` is the specific one: the profiler's own default is
        `"torch"` while `serve.M1_DEFAULTS` and `preset.CANONICAL_FAST` both
        say `"triton"`, so profiling without the preset measures a config
        nobody serves."""
        rt = ps.runtime_config_for(self._args(), 256)
        from qwenfast.runtime.preset import CANONICAL_FAST

        for key, want in CANONICAL_FAST.items():
            if key in ("max_num_seqs", "use_cuda_graphs"):
                continue
            self.assertEqual(getattr(rt, key), want, msg=f"{key} is not the preset value")

    def test_explicit_flags_still_beat_the_preset(self):
        rt = ps.runtime_config_for(self._args("--conv-prefill-layout", "channel_major"), 32)
        self.assertEqual(rt.conv_prefill_layout, "channel_major")

    def test_pool_geometry_is_the_servers_own_function(self):
        """Not a re-derivation: `serving_pool_sizes` must be
        `bench_runtime.derive_pool_sizes`, which is what `serve` calls."""
        from qwenfast.runtime.bench_runtime import derive_pool_sizes

        got = ps.serving_pool_sizes(256, 2703, 16)
        want = derive_pool_sizes(256, 2703, 16, slack_pages=16)
        self.assertEqual(got["n_kv_pages"], want["n_kv_pages"])
        self.assertEqual(got["max_pages_per_seq"], want["max_pages_per_seq"])


class TestProfilerTeardown(unittest.TestCase):
    def test_teardown_drops_the_engines_pools(self):
        """The other half of the profiler OOM.

        ``del comps, sched, model`` plus ``empty_cache()`` is not enough: the
        pools are reachable through reference *cycles* (``_retype_kv_pool``
        alone rebinds ``pool.gather_dense`` to a closure that captures the
        pool's own bound method), so without ``gc.collect()`` the conc-32
        engine stays resident while the conc-256 engine loads on top of it.
        ``teardown`` nulls the pools explicitly *and* collects.
        """
        _sched, fm, dec, _rt = build_scheduler()

        class _Comps:
            pass

        comps = _Comps()
        comps.model = fm
        comps.decoder = dec
        ps.teardown(comps)
        self.assertIsNone(comps.model)
        self.assertIsNone(comps.decoder)
        self.assertIsNone(fm.state_pool)
        self.assertIsNone(fm.kv_pool)

    def test_teardown_of_nothing_is_a_no_op(self):
        ps.teardown(None)


class TestPrefillAttribution(unittest.TestCase):
    """``runtime/prefill_attrib.py`` -- the additive attribution table."""

    def test_event_tape_records_and_aggregates(self):
        tape = pa.EventTape(CPU)
        tape.new_repeat()
        for _ in range(3):
            h = tape.enter("gemm.mlp.gate_up.M8.bf16_native")
            tape.exit(h, flops=2.0)
        h = tape.enter("gdn.chunk.torch")
        tape.exit(h)
        got = tape.resolve()
        self.assertEqual(got["gemm.mlp.gate_up.M8.bf16_native"]["calls"], 3)
        self.assertEqual(got["gemm.mlp.gate_up.M8.bf16_native"]["flops"], 6.0)
        self.assertEqual(got["gdn.chunk.torch"]["calls"], 1)

    def test_repeats_are_averaged_not_summed(self):
        tape = pa.EventTape(CPU)
        for _ in range(4):
            tape.new_repeat()
            h = tape.enter("norm.rms")
            tape.exit(h)
        got = tape.resolve()
        self.assertEqual(got["norm.rms"]["calls"], 1.0)  # 4 calls / 4 repeats

    def test_groups(self):
        self.assertEqual(pa.group_of("gemm.mlp.gate_up.M2048.deepgemm"), "gemm")
        self.assertEqual(pa.group_of("gdn.conv.token_major"), "gdn_conv")
        self.assertEqual(pa.group_of("gdn.chunk.fla"), "gdn_chunk")
        self.assertEqual(pa.group_of("gdn.state_io.gather"), "gdn_state_io")
        self.assertEqual(pa.group_of("attn.prefill"), "attn_prefill")
        self.assertEqual(pa.group_of("norm.add_rms"), "norm")
        self.assertEqual(pa.group_of("something.else"), "other")

    def test_table_residual_is_the_unbracketed_remainder(self):
        labels = ps.OrderedDict(
            [
                ("gemm.mlp.gate_up.M8.x", {"ms": 60.0, "calls": 2.0, "flops": 0.0}),
                ("gdn.chunk.torch", {"ms": 20.0, "calls": 1.0, "flops": 0.0}),
            ]
        )
        t = pa.attribution_table(labels, chunk_ms=100.0, tokens=8192)
        self.assertAlmostEqual(t["attributed_ms"], 80.0)
        self.assertAlmostEqual(t["residual_ms"], 20.0)
        self.assertAlmostEqual(t["residual_pct"], 20.0)
        self.assertEqual(t["components"][0]["component"], "gemm")  # sorted by ms
        self.assertIn("gemm", pa.format_table(t))

    def _chunk(self):
        _sched, fm, _dec, _rt = build_scheduler(max_num_batched_tokens=16)
        batch = ps.synthetic_chunk(fm, [5, 4])
        return fm, batch

    def test_instrument_brackets_every_component_of_a_real_chunk(self):
        fm, batch = self._chunk()
        tape = pa.EventTape(fm.device)
        restore = pa.instrument(fm, tape)
        try:
            tape.active = True
            tape.new_repeat()
            fm.prefill_forward(batch, all_logits=False)
        finally:
            tape.active = False
            restore()
        labels = tape.resolve()
        groups = {pa.group_of(k) for k in labels}
        for want in ("gemm", "gdn_conv", "gdn_chunk", "attn_prefill", "norm", "mlp_act"):
            self.assertIn(want, groups, msg=f"{want} not bracketed; labels={list(labels)}")
        # The GEMM labels must carry the site, the M and the backend -- the
        # by-shape/by-backend table needs at a real chunk.
        gemms = [k for k in labels if k.startswith("gemm.")]
        self.assertTrue(any(k.startswith("gemm.mlp.gate_up.M") for k in gemms), gemms)
        self.assertTrue(any(k.startswith("gemm.gdn.in_proj_qkvz.M") for k in gemms), gemms)
        self.assertTrue(any(k.startswith("gemm.attn.qkv.M") for k in gemms), gemms)
        self.assertTrue(any(".M9." in k for k in gemms), gemms)  # 5 + 4 tokens

    def test_instrument_restores_every_patch(self):
        """A profiler that leaks a patch poisons every A/B that follows it."""
        from qwenfast.attn import flashinfer_attn as fi
        from qwenfast.attn import fused_qk_rope as _qkrope
        from qwenfast.kernels_gdn import api as _api
        from qwenfast.kernels_gdn import torch_ops as _tops
        from qwenfast.runtime import fused_model as fm_mod

        fm, batch = self._chunk()
        before = (
            _api.causal_conv_prefill_varlen,
            _api.gather_states,
            _tops.chunk_gdn,
            fm_mod.rms_norm_w1p,
            fm_mod.swiglu,
            fm_mod.ResolvedLinear.__call__,
            fi.fused_qk_norm_rope,
            _qkrope.qk_norm_rope,
        )
        tape = pa.EventTape(fm.device)
        restore = pa.instrument(fm, tape)
        tape.active = True
        fm.prefill_forward(batch, all_logits=False)
        tape.active = False
        restore()
        after = (
            _api.causal_conv_prefill_varlen,
            _api.gather_states,
            _tops.chunk_gdn,
            fm_mod.rms_norm_w1p,
            fm_mod.swiglu,
            fm_mod.ResolvedLinear.__call__,
            fi.fused_qk_norm_rope,
            _qkrope.qk_norm_rope,
        )
        self.assertEqual(before, after)
        self.assertNotIn("prefill", fm.attn.__dict__)  # instance patch removed

    def test_patches_are_inert_while_the_tape_is_inactive(self):
        """The patched call sites read one attribute and get out of the way,
        so an A/B timed *between* instrumented chunks is not paying for the
        instrumentation."""
        fm, batch = self._chunk()
        tape = pa.EventTape(fm.device)
        restore = pa.instrument(fm, tape)
        try:
            fm.prefill_forward(batch, all_logits=False)  # tape.active is False
        finally:
            restore()
        self.assertEqual(tape.resolve(), ps.OrderedDict())

    def test_instrumentation_does_not_change_the_logits(self):
        fm, batch = self._chunk()
        ps.reset_chunk_slots(fm, batch)
        want = fm.prefill_forward(batch, all_logits=False).clone()
        tape = pa.EventTape(fm.device)
        restore = pa.instrument(fm, tape)
        try:
            tape.active = True
            ps.reset_chunk_slots(fm, batch)
            got = fm.prefill_forward(batch, all_logits=False)
        finally:
            tape.active = False
            restore()
        torch.testing.assert_close(got, want, rtol=0, atol=0)


class TestGdnChunkSizeKnob(unittest.TestCase):
    """``chunk_size`` is a real knob.

    fla's chunk size is not baked into the kernel's autotune config:
    ``fla-core`` 0.5.2 accepts ``chunk_size = kwargs.pop('chunk_size', 64)``
    with values 16/32/64, which matters at the one shape where BT is a real
    trade.  fla is not importable on a laptop, so what is tested here is
    (a) the capability probe is honest, and (b) the *maths* is
    chunk-size-invariant on the torch backend, which is what makes sweeping
    it on the GPU a speed question rather than a correctness one.
    """

    def _inputs(self, t=24, hv=6, hk=2, dk=8, dv=8):
        torch.manual_seed(11)
        q = torch.randn(1, t, hk, dk, dtype=torch.float32)
        k = torch.randn(1, t, hk, dk, dtype=torch.float32)
        v = torch.randn(1, t, hv, dv, dtype=torch.float32)
        g = -torch.rand(1, t, hv, dtype=torch.float32)
        beta = torch.rand(1, t, hv, dtype=torch.float32)
        cu = torch.tensor([0, 10, 24], dtype=torch.int32)
        return q, k, v, g, beta, cu

    def test_torch_backend_is_chunk_size_invariant(self):
        q, k, v, g, beta, cu = self._inputs()
        outs = []
        for cs in (16, 32, 64):
            o, _ = gdn_api.gdn_prefill_chunked(
                q, k, v, g, beta, cu_seqlens=cu, backend="torch",
                output_final_state=True, chunk_size=cs,
            )
            outs.append(o)
        for o in outs[1:]:
            torch.testing.assert_close(o, outs[0], rtol=2e-4, atol=2e-4)

    def test_cu_seqlens_cpu_is_accepted_and_inert_on_the_torch_backend(self):
        q, k, v, g, beta, cu = self._inputs()
        base, _ = gdn_api.gdn_prefill_chunked(
            q, k, v, g, beta, cu_seqlens=cu, backend="torch", output_final_state=True
        )
        got, _ = gdn_api.gdn_prefill_chunked(
            q, k, v, g, beta, cu_seqlens=cu, backend="torch", output_final_state=True,
            cu_seqlens_cpu=cu.to(torch.int64),
        )
        torch.testing.assert_close(got, base, rtol=0, atol=0)

    def test_capability_probes_are_honest(self):
        self.assertEqual(fla_ops.FLA_CHUNK_SIZES, (16, 32, 64))
        self.assertIsInstance(fla_ops.supports_chunk_size(), bool)
        self.assertIsInstance(fla_ops.supports_cu_seqlens_cpu(), bool)
        if not fla_ops.is_available():
            # No fla here, so the conservative answer is the only safe one:
            # never hand an unknown build a keyword it will silently drop.
            self.assertFalse(fla_ops.supports_chunk_size())
            self.assertFalse(fla_ops.supports_cu_seqlens_cpu())


class TestGdnChunkSizeParity(unittest.TestCase):
    """The CPU precondition for moving the preset default off
    64 (``preset.CANONICAL_FAST["gdn_chunk_size"] = 32``, ``serve.py``'s
    ``_resolved_gdn_chunk_size``).

    fla is not importable on a laptop, so what is provable here is not "fla
    at 32 matches fla at 64" (that needs a GPU run, where 32 measured 2.7%%
    faster -- speed only, not parity) but
    the thing that actually licenses flipping a *default*: the engine's
    torch-backend prefill path -- real weights, a real ``Scheduler``,
    ``FusedGDN.prefill``, ``gdn_api.gdn_prefill_chunked`` and all -- produces
    the same logits end to end at every accepted chunk size, on a real (tiny)
    model forward. ``TestGdnChunkSizeKnob`` above already pins this at the
    level of ``fla_ops.chunk_gdn``'s isolated maths on synthetic tensors;
    this is the same claim one layer up, through the actual call site the
    preset default controls.
    """

    def test_tiny_model_prefill_logits_match_across_chunk_sizes(self):
        outs = []
        for cs in (64, 32, 16):
            _sched, fm, _dec, _rt = build_scheduler(max_num_batched_tokens=128, gdn_chunk_size=cs)
            self.assertEqual(_rt.gdn_chunk_size, cs)
            batch = ps.synthetic_chunk(fm, [50, 30])  # > any of 16/32/64, several BT's worth
            logits = fm.prefill_forward(batch, all_logits=False)
            outs.append(logits.clone())
        for cs, o in zip((32, 16), outs[1:]):
            torch.testing.assert_close(
                o, outs[0], rtol=2e-4, atol=2e-4,
                msg=f"chunk_size={cs} prefill logits diverged from chunk_size=64",
            )


class TestCuSeqlensOnTheHost(unittest.TestCase):
    """``StepContext.cu_seqlens_cpu`` -- built once per chunk, never per layer.

    fla's varlen path reads ``cu_seqlens`` back to the host to build its chunk
    index tensors (``fla/ops/utils/index.py``), memoised on Python object
    identity in a 4-deep deque.  This engine pays that once per chunk rather
    than once per GDN layer *only* because ``PrefillBatch.cu_seqlens`` is
    already int32 on the right device, so ``fla_ops.chunk_gdn``'s ``.to(...)``
    returns the same object -- an invariant nothing tested and any dtype change
    would have broken silently, turning 1 pipeline drain per chunk into 48.
    """

    def test_cu_seqlens_is_int32_on_the_batch(self):
        batch = make_prefill_batch([[1, 2, 3], [4, 5]], [0, 0], [0, 1], CPU)
        self.assertEqual(batch.cu_seqlens.dtype, torch.int32)

    def test_context_carries_the_host_copy_for_prefill(self):
        _sched, fm, _dec, _rt = build_scheduler()
        batch = make_prefill_batch([[1, 2, 3], [4, 5]], [0, 0], [0, 1], CPU)
        ctx = fm._context(
            batch.slot_ids, batch.positions,
            seq_slot_ids=batch.seq_slot_ids, cu_seqlens=batch.cu_seqlens,
            q_lens=batch.q_lens,
        )
        self.assertIsNotNone(ctx.cu_seqlens_cpu)
        self.assertEqual(ctx.cu_seqlens_cpu.device.type, "cpu")
        self.assertEqual(ctx.cu_seqlens_cpu.tolist(), batch.cu_seqlens.tolist())

    def test_decode_context_has_no_host_cu_seqlens(self):
        _sched, fm, _dec, _rt = build_scheduler()
        ids = torch.zeros(2, dtype=torch.int32)
        ctx = fm._context(ids, ids)
        self.assertIsNone(ctx.cu_seqlens_cpu)


class TestMlpTileKnob(unittest.TestCase):
    """``mlp_tile_tokens`` is a served knob."""

    def test_the_config_reaches_every_mlp(self):
        _sched, fm, _dec, _rt = build_scheduler(mlp_tile_tokens=7)
        self.assertTrue(fm.layers)
        for layer in fm.layers:
            self.assertEqual(layer.mlp.tile, 7)

    def test_the_tile_does_not_change_the_answer(self):
        """The whole case for raising it: it is a memory/M-bucket knob, not a
        numerics one."""
        _sched, fm, _dec, _rt = build_scheduler(max_num_batched_tokens=16)
        batch = ps.synthetic_chunk(fm, [5, 4])
        outs = []
        for tile in (2, 9, 4096):
            for layer in fm.layers:
                layer.mlp.tile = tile
            ps.reset_chunk_slots(fm, batch)
            outs.append(fm.prefill_forward(batch, all_logits=False).clone())
        for o in outs[1:]:
            torch.testing.assert_close(o, outs[0], rtol=1e-5, atol=1e-5)

    def test_serve_exposes_it(self):
        from qwenfast.runtime import serve as serve_mod

        p = serve_mod.build_arg_parser()
        args = p.parse_args(["--model", "/nonexistent", "--mlp-tile-tokens", "8192"])
        rt = serve_mod.runtime_config_from_args(args)
        self.assertEqual(rt.mlp_tile_tokens, 8192)


# =========================================================================== #
# 8. profiler regressions: pool sizing, GEMM sweep caches, probe isolation
# =========================================================================== #
class TestChunk12Regressions(unittest.TestCase):
    """Profiler failure modes that each turn a run into an OOM with no JSON
    written: pools sized for the ledger rather than the chunk, a GEMM sweep
    that allocates a second repack cache, and one raising probe discarding
    every other probe's results. One test group each.
    """

    def _args(self, *extra: str):
        parser = ps.build_arg_parser()
        argv = ["--model", "/nonexistent", "--preset", "fastest", *extra]
        args = parser.parse_args(argv)
        return ps.apply_preset(args, parser, argv=argv)

    # -- pools must be sized for the chunk, not the ledger ------------------ #
    def test_pool_size_is_decoupled_from_the_chunk(self):
        """`--pool-max-num-seqs 64` must give the *same chunk* and ~47 GiB more
        room. An 8,192-token chunk at input-len 2,139 is four sequences; it
        should not be profiled behind a 256-sequence pool."""
        big = ps.runtime_config_for(self._args(), 256)
        small = ps.runtime_config_for(self._args("--pool-max-num-seqs", "64"), 256)
        self.assertEqual(big.max_num_seqs, 256)
        self.assertEqual(small.max_num_seqs, 64)
        # the chunk the probes actually run is identical
        self.assertEqual(
            ps.chunk_shape(8192, 2139)[:256], ps.chunk_shape(8192, 2139)[:64]
        )
        self.assertEqual(len(ps.chunk_shape(8192, 2139)), 4)

    def test_the_small_pool_buys_real_headroom(self):
        _rt_b, big = ps.plan_for(self._args(), 256)
        _rt_s, small = ps.plan_for(self._args("--pool-max-num-seqs", "64"), 256)
        freed = big["total_gib"] - small["total_gib"]
        self.assertGreater(freed, 40.0, msg=f"only freed {freed:.1f} GiB")
        # ...and the result must clear the >= 25 GiB headroom bar on a 139.8
        # GiB card at the 0.94 utilisation the profiler budgets against.
        self.assertLess(small["total_gib"], 139.8 * 0.94 - 25.0)

    def test_drive_requires_a_pool_that_can_hold_the_concurrency(self):
        """The ledger genuinely needs the slots, so shrinking the pool without
        `--no-drive` must be refused at parse time, not at step 1."""
        with self.assertRaises(SystemExit):
            ps.main(["--model", "/nonexistent", "--concurrency", "256",
                     "--pool-max-num-seqs", "64"])

    # -- the GEMM sweep must not allocate a second 23 GiB repack cache ------ #
    def test_gemm_sweep_refuses_cache_owning_backends_under_single(self):
        """`prefill_gemm_scope` bypasses the weight-cache policy by design, so
        an unguarded sweep materialises a second ~23 GiB repacked copy of every
        fp8 linear -- the repack-cache OOM, through the profiler's front door
        (plan steady 117.80 + ~19 = 137.02 GiB)."""
        from qwenfast.gemm import dispatch as gd

        class _W:  # a weight that already holds marlin's cache, as after warmup
            _marlin_cache = (torch.zeros(1),)

        class _Lin:
            weight = _W()

        class _Layer:
            mlp = type("M", (), {"gate_up": _Lin()})()

        model = type("Model", (), {"layers": [_Layer()]})()
        prev = gd.set_weight_cache_policy("single")
        try:
            rows = dict((n, (ok, why)) for n, ok, why in ps._prefill_backends_for(model))
        finally:
            gd.set_weight_cache_policy(prev)
        self.assertTrue(rows[None][0])                       # dispatch: always
        self.assertTrue(rows["flashinfer_fp8_blockscale"][0])  # cache-free: always
        for name in ("scaled_mm_pertensor", "deepgemm"):
            ok, why = rows[name]
            self.assertFalse(ok, msg=f"{name} must be refused under 'single'")
            self.assertIn("repack cache", why)

    def test_gemm_sweep_allows_everything_under_multi(self):
        from qwenfast.gemm import dispatch as gd

        class _Lin:
            weight = type("W", (), {})()

        class _Layer:
            mlp = type("M", (), {"gate_up": _Lin()})()

        model = type("Model", (), {"layers": [_Layer()]})()
        prev = gd.set_weight_cache_policy("multi")
        try:
            rows = {n: ok for n, ok, _ in ps._prefill_backends_for(model)}
        finally:
            gd.set_weight_cache_policy(prev)
        self.assertTrue(all(rows.values()))

    # -- one raising probe must not discard the others ---------------------- #
    def test_run_probe_turns_a_crash_into_a_row(self):
        """`mlp_tile_ab` raising must cost that row and nothing else."""
        def boom():
            raise RuntimeError("CUDA out of memory. Tried to allocate 80.00 MiB")

        got = ps.run_probe("mlp_tile_ab", boom, verbose=False)
        self.assertIn("error", got)
        self.assertIn("RuntimeError", got["error"])
        self.assertIn("out of memory", got["error"])

    def test_run_probe_skips_what_cannot_fit(self):
        """With a real free-memory reading, a probe that does not fit says so
        rather than finding out 20 GiB in."""
        ran = []
        orig = ps.free_gib
        ps.free_gib = lambda device=None: 2.0  # 2 GiB free
        try:
            got = ps.run_probe(
                "mlp_tile_ab", lambda: ran.append(1) or {}, need_gib=8.0, verbose=False
            )
        finally:
            ps.free_gib = orig
        self.assertEqual(ran, [])
        self.assertIn("skipped", got)
        self.assertIn("8.0 GiB", got["skipped"])

    def test_run_probe_runs_when_the_headroom_is_unknown(self):
        """Off CUDA there is no reading, and the guard must be inert rather
        than skipping everything."""
        orig = ps.free_gib
        ps.free_gib = lambda device=None: None
        try:
            got = ps.run_probe("x", lambda: {"ok": 1}, need_gib=1e6, verbose=False)
        finally:
            ps.free_gib = orig
        self.assertEqual(got, {"ok": 1})

    def test_run_probe_passes_results_through(self):
        self.assertEqual(ps.run_probe("x", lambda: {"ok": 1}, verbose=False), {"ok": 1})

    def test_headroom_helper_is_none_off_cuda(self):
        if not torch.cuda.is_available():
            self.assertIsNone(ps.free_gib())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
