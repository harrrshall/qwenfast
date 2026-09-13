"""CPU/GPU unit tests for MTP speculative decoding.

The correctness criterion for speculative decoding ("greedy output
**bit-identical** to non-speculative greedy decoding"),
checked here as :class:`TestSpecGreedyEquivalence`:

    for B in {1, 4} and k in {1, 2, 3}, >= 64 generated tokens per sequence,
    the token stream produced by ``Scheduler(..., spec=SpecDecoder(k))`` is
    **exactly** the token stream produced by the same scheduler with no spec
    decoder at all -- same prompts, same weights, same greedy rule.

That is the only test that can catch the whole class of speculative-decoding
bugs at once (a wrong acceptance rule, an off-by-one in the window positions, a
state committed one token too far, a KV tail that was not rolled back): every
one of them shows up as a diverging token, usually within a dozen steps.

Everything runs on the tiny synthetic config from ``test_runtime.py`` (hidden
64, 2 Gated-DeltaNet + 1 full-attention layer + 1 MTP layer, vocab 32), so the
whole file is seconds, not minutes, and needs no checkpoint.  ``DEVICE`` follows
``test_runtime``: ``cuda:0`` when present (which also exercises the real Triton
GDN window kernel via ``TestSpecGpuBackends``), CPU otherwise.

Run::

    python engine/qwenfast/runtime/tests/test_spec_decode.py -v
    pytest engine/qwenfast/runtime/tests/test_spec_decode.py -v
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)  # -> test_runtime fixtures
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))  # -> engine/

if torch.cuda.is_available():
    # same rationale as test_runtime.py: TF32's rounding is not invariant to
    # GEMM tiling, and this file compares a B*n-row window GEMM against B-row
    # decode GEMMs of the same linear algebra.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

from test_runtime import DEVICE, base_rt, build_pieces  # noqa: E402
from test_runtime import build_m0 as _build_m0_uncached  # noqa: E402

_M0_CACHE = {}


def build_m0(**kw):
    """Memoized ``test_runtime.build_m0``.

    Every test here builds the same tiny model twice (once for the reference
    decode, once for the speculative one) and the weights are *shared* on
    purpose -- only the pools differ. Rebuilding it per call was ~a third of
    this file's wall time on the gpu host, and this suite has to stay short
    because it runs on shared GPU time.
    """
    key = tuple(sorted((k, str(v)) for k, v in kw.items()))
    if key not in _M0_CACHE:
        _M0_CACHE[key] = _build_m0_uncached(**kw)
    return _M0_CACHE[key]

from qwenfast.gemm import dispatch as gemm_dispatch  # noqa: E402
from qwenfast.model import QwenFastForCausalLM  # noqa: E402
from qwenfast.runtime.fused_model import ResolvedLinear  # noqa: E402
from qwenfast.runtime.scheduler import GenParams, Request, Scheduler  # noqa: E402
from qwenfast.runtime.spec_decode import SpecConfig, SpecDecoder, build_spec_decoder  # noqa: E402
from qwenfast.weights import QwenFastConfig  # noqa: E402


# =========================================================================== #
# fixtures
# =========================================================================== #
def spec_rt(**overrides):
    """``base_rt`` with room for a 64+-token decode at B=4 and an MTP head."""
    kwargs = dict(
        enable_mtp=True,
        page_size=4,
        n_kv_pages=1024,
        max_pages_per_seq=64,
        max_model_len=256,
        max_num_seqs=8,
        max_num_batched_tokens=128,
    )
    kwargs.update(overrides)
    return base_rt(**kwargs)


def make_scheduler(m0, rt, spec_cfg=None, spec_max_batch=None):
    comps = build_pieces(rt, m0)
    spec = build_spec_decoder(comps.model, comps.buf, comps.rt, spec_cfg) if spec_cfg else None
    sched = Scheduler(comps.model, comps.decoder, comps.rt, spec=spec, spec_max_batch=spec_max_batch)
    return comps, sched, spec


def generate(sched, prompts, max_tokens, *, eos=None, ignore_eos=True, stop_ids=()):
    reqs = []
    for i, p in enumerate(prompts):
        r = Request(
            request_id=f"r{i}",
            prompt_token_ids=list(p),
            params=GenParams(
                temperature=0.0,
                max_tokens=max_tokens,
                ignore_eos=ignore_eos,
                eos_token_id=eos,
                stop_token_ids=tuple(stop_ids),
            ),
        )
        sched.add_request(r)
        reqs.append(r)
    guard = 0
    while sched.has_work() and guard < 20000:
        sched.step()
        guard += 1
    if sched.has_work():  # pragma: no cover - would be a livelock bug
        raise AssertionError("scheduler did not drain")
    return reqs


PROMPTS = [
    [3, 5, 7, 9, 11, 2, 4, 6],
    [1, 2, 3, 4, 5, 6, 7, 8],
    [10, 12, 14, 16, 18, 20, 22, 24],
    [31, 29, 27, 25, 23, 21, 19, 17],
]


# =========================================================================== #
# 1. the acceptance criterion
# =========================================================================== #
class TestSpecGreedyEquivalence(unittest.TestCase):
    """Spec-decoded greedy output == plain greedy output, token for token."""

    def _run_pair(self, k: int, b: int, max_tokens: int = 64, **rt_kw):
        m0, _cfg = build_m0(seed=11, with_mtp=True)
        prompts = PROMPTS[:b]

        rt_ref = spec_rt(**rt_kw)
        _c1, sched_ref, _ = make_scheduler(m0, rt_ref)
        ref = generate(sched_ref, prompts, max_tokens)

        rt_spec = spec_rt(**rt_kw)
        _c2, sched_spec, spec = make_scheduler(m0, rt_spec, SpecConfig(k=k))
        got = generate(sched_spec, prompts, max_tokens)

        for i, (a, c) in enumerate(zip(ref, got)):
            self.assertEqual(
                len(a.output_token_ids), max_tokens, f"reference seq {i} short"
            )
            self.assertEqual(
                a.output_token_ids,
                c.output_token_ids,
                f"k={k} B={b} seq {i}: spec output diverged from greedy\n"
                f"  ref  = {a.output_token_ids}\n"
                f"  spec = {c.output_token_ids}",
            )
        return spec

    def test_b1(self):
        for k in (1, 2, 3):
            with self.subTest(k=k):
                self._run_pair(k, 1)

    def test_b4(self):
        for k in (1, 2, 3):
            with self.subTest(k=k):
                self._run_pair(k, 4)

    def test_accept_length_is_reported(self):
        spec = self._run_pair(2, 4)
        st = spec.stats()
        self.assertGreater(st["spec_steps"], 0)
        # every step emits at least the bonus token, and never more than k+1
        self.assertGreaterEqual(st["spec_accept_length"], 1.0)
        self.assertLessEqual(st["spec_accept_length"], 3.0)
        self.assertGreaterEqual(st["spec_acceptance_rate"], 0.0)
        self.assertLessEqual(st["spec_acceptance_rate"], 1.0)


# =========================================================================== #
# 2. state equality
# =========================================================================== #
class TestSpecStateParity(unittest.TestCase):
    """After N tokens, the SSM/conv/KV state must equal what sequential
    decoding of the *same accepted tokens* would have left behind.

    This is the half of correctness the token stream cannot prove on its own:
    a commit that ran one token too far still emits the right tokens for a
    while (the drafts were right, after all) and only diverges later.
    """

    def test_state_matches_sequential_decode(self):
        k, b, max_tokens = 3, 2, 48
        m0, _cfg = build_m0(seed=13, with_mtp=True)
        prompts = PROMPTS[:b]

        c_ref, sched_ref, _ = make_scheduler(m0, spec_rt())
        ref = generate(sched_ref, prompts, max_tokens)

        c_spec, sched_spec, _spec = make_scheduler(m0, spec_rt(), SpecConfig(k=k))
        got = generate(sched_spec, prompts, max_tokens)

        for i in range(b):
            self.assertEqual(ref[i].output_token_ids, got[i].output_token_ids)

        # `_finish` frees pages but never touches the SSM/conv rows, and slot
        # assignment is deterministic for an identical request order, so the
        # two runs' slot 0..b-1 hold the same sequences.
        for slot in range(b):
            torch.testing.assert_close(
                c_spec.model.state_pool[slot].float().cpu(),
                c_ref.model.state_pool[slot].float().cpu(),
                rtol=2e-4,
                atol=2e-4,
                msg=f"SSM state diverged on slot {slot}",
            )
            torch.testing.assert_close(
                c_spec.model.conv_pool[slot].float().cpu(),
                c_ref.model.conv_pool[slot].float().cpu(),
                rtol=2e-4,
                atol=2e-4,
                msg=f"conv state diverged on slot {slot}",
            )

    def test_kv_matches_sequential_decode(self):
        """The committed KV prefix must match too -- the rejected tail is
        allowed to differ (it is overwritten, never read)."""
        k, b, max_tokens = 2, 1, 32
        m0, _cfg = build_m0(seed=14, with_mtp=True)
        prompts = PROMPTS[:b]
        # `-1`: the *last* generated token never gets a KV entry -- it is the
        # step's output and would only be written when it is fed back as the
        # next step's input, which never happens because the request finishes.
        # Both runs therefore end at num_computed_tokens = prompt + max_tokens
        # - 1, and position `prompt + max_tokens - 1` is an untouched (zeroed)
        # page slot in the reference while the speculative run legitimately
        # left a rejected draft's K/V there. That tail is exactly what "the
        # rejected tail may differ" means; comparing it would be asserting the
        # opposite of the design.
        n_cmp = len(prompts[0]) + max_tokens - 1

        def kv_after(spec_cfg):
            comps, sched, _ = make_scheduler(m0, spec_rt(), spec_cfg)
            # `Scheduler._finish` returns the request's pages to the allocator,
            # so the cache has to be read on the way out rather than after
            # `generate` returns. Only the *main* attention layers are compared:
            # the MTP layer's cache is a draft-side artefact.
            n_main = len(comps.model.attn_layer_indices)
            snap = {}
            orig_finish = sched._finish

            def finish(req, reason):
                snap[req.request_id] = [
                    tuple(t.float().cpu().clone()
                          for t in comps.model.kv_pool.gather_dense(layer, req.slot, n_cmp))
                    for layer in range(n_main)
                ]
                orig_finish(req, reason)

            sched._finish = finish
            reqs = generate(sched, prompts, max_tokens)
            return snap, [r.output_token_ids for r in reqs], [r.num_computed_tokens for r in reqs]

        ref, ref_toks, ref_len = kv_after(None)
        got, got_toks, got_len = kv_after(SpecConfig(k=k))
        self.assertEqual(ref_toks, got_toks)
        for rl, gl in zip(ref_len, got_len):
            self.assertGreaterEqual(gl, n_cmp, "spec committed fewer tokens than it emitted")
            self.assertGreaterEqual(rl, n_cmp)

        for rid, layers in ref.items():
            for layer, ((kr, vr), (ks, vs)) in enumerate(zip(layers, got[rid])):
                torch.testing.assert_close(
                    ks, kr, rtol=2e-3, atol=2e-3,
                    msg=f"K diverged on attention layer {layer} ({rid})",
                )
                torch.testing.assert_close(
                    vs, vr, rtol=2e-3, atol=2e-3,
                    msg=f"V diverged on attention layer {layer} ({rid})",
                )


# =========================================================================== #
# 3. stop conditions inside the window
# =========================================================================== #
class TestStopInsideWindow(unittest.TestCase):
    def test_eos_inside_window_truncates(self):
        """A step can produce k+1 tokens with EOS in the middle; the tokens
        after it must never be emitted, and the request must finish 'stop'."""
        m0, _cfg = build_m0(seed=15, with_mtp=True)
        prompts = PROMPTS[:1]

        # first find what greedy actually generates, then make one of those
        # tokens the EOS so it is guaranteed to land inside a window.
        _c, sched, _ = make_scheduler(m0, spec_rt())
        ref = generate(sched, prompts, 24)
        stream = ref[0].output_token_ids
        eos = stream[9]
        expected = stream[: stream.index(eos) + 1]

        _c2, sched2, _ = make_scheduler(m0, spec_rt(), SpecConfig(k=3))
        got = generate(sched2, prompts, 24, eos=eos, ignore_eos=False)
        self.assertEqual(got[0].output_token_ids, expected)
        self.assertEqual(got[0].finish_reason, "stop")
        self.assertNotIn(eos, got[0].output_token_ids[:-1])

    def test_max_tokens_not_a_multiple_of_the_window(self):
        m0, _cfg = build_m0(seed=16, with_mtp=True)
        for max_tokens in (1, 5, 7, 10):
            with self.subTest(max_tokens=max_tokens):
                _c, sched, _ = make_scheduler(m0, spec_rt())
                ref = generate(sched, PROMPTS[:2], max_tokens)
                _c2, sched2, _ = make_scheduler(m0, spec_rt(), SpecConfig(k=3))
                got = generate(sched2, PROMPTS[:2], max_tokens)
                for a, c in zip(ref, got):
                    self.assertEqual(len(c.output_token_ids), max_tokens)
                    self.assertEqual(a.output_token_ids, c.output_token_ids)
                    self.assertEqual(c.finish_reason, "length")


# =========================================================================== #
# 4. acceptance-rule unit test
# =========================================================================== #
class TestAcceptanceRule(unittest.TestCase):
    def _spec(self, k):
        m0, _cfg = build_m0(seed=17, with_mtp=True)
        comps = build_pieces(spec_rt(), m0)
        return build_spec_decoder(comps.model, comps.buf, comps.rt, SpecConfig(k=k))

    def test_prefix_only_acceptance(self):
        """A draft that matches *after* a mismatch must not be accepted: the
        rule is a cumulative product over the prefix, not a per-position OR."""
        k = 3
        spec = self._spec(k)
        n, vocab = spec.n, spec.model.config.vocab_size
        b = 2
        # sequence 0: drafts [5, 6, 7], target argmax [5, 9, 7] -> accept 1
        # sequence 1: drafts [1, 2, 3], target argmax [1, 2, 3] -> accept 3
        drafts = torch.tensor([[5, 6, 7], [1, 2, 3]], dtype=torch.int32, device=DEVICE)
        tgt = torch.tensor([[5, 9, 7, 4], [1, 2, 3, 8]], dtype=torch.long, device=DEVICE)
        spec.window_tokens[:b, 1:] = drafts
        logits = torch.zeros(b * n, vocab, device=DEVICE)
        logits.scatter_(1, tgt.reshape(-1, 1), 10.0)

        spec._accept(b, logits)
        self.assertEqual(spec.m[:b].tolist(), [2, 4])
        self.assertEqual(
            spec.out_window[:b].tolist(),
            [[5, 9, -1, -1], [1, 2, 3, 8]],
        )

    def test_no_acceptance_gives_one_token(self):
        spec = self._spec(2)
        n, vocab = spec.n, spec.model.config.vocab_size
        spec.window_tokens[:1, 1:] = torch.tensor([[5, 6]], dtype=torch.int32, device=DEVICE)
        tgt = torch.tensor([[9, 9, 9]], dtype=torch.long, device=DEVICE)
        logits = torch.zeros(n, vocab, device=DEVICE)
        logits.scatter_(1, tgt.reshape(-1, 1), 10.0)
        spec._accept(1, logits)
        self.assertEqual(spec.m[:1].tolist(), [1])
        self.assertEqual(spec.out_window[:1].tolist(), [[9, -1, -1]])


# =========================================================================== #
# 5. routing / fallback
# =========================================================================== #
class TestSpecRouting(unittest.TestCase):
    def test_sampling_requests_fall_back_to_plain_decode(self):
        m0, _cfg = build_m0(seed=18, with_mtp=True)
        _c, sched, spec = make_scheduler(m0, spec_rt(), SpecConfig(k=2))
        self.assertTrue(spec.eligible([0.0, 0.0]))
        self.assertFalse(spec.eligible([0.0, 0.7]))

        r = Request(
            request_id="s0",
            prompt_token_ids=list(PROMPTS[0]),
            params=GenParams(temperature=0.8, max_tokens=8, ignore_eos=True),
        )
        sched.add_request(r)
        guard = 0
        while sched.has_work() and guard < 500:
            sched.step()
            guard += 1
        self.assertEqual(len(r.output_token_ids), 8)
        self.assertEqual(spec.steps, 0, "a sampling request must not take the spec path")

    def test_greedy_only_flag_off_admits_sampling(self):
        m0, _cfg = build_m0(seed=19, with_mtp=True)
        _c, _sched, spec = make_scheduler(m0, spec_rt(), SpecConfig(k=2, greedy_only=False))
        self.assertTrue(spec.eligible([0.0, 0.7]))

    def test_requires_mtp_head(self):
        m0, _cfg = build_m0(seed=20, with_mtp=False)
        comps = build_pieces(spec_rt(enable_mtp=False), m0)
        with self.assertRaises(RuntimeError):
            SpecDecoder(comps.model, comps.buf, comps.rt, SpecConfig(k=2))


# =========================================================================== #
# 6. GPU-only: real kernels + CUDA-graph capture
# =========================================================================== #
class TestSpecGpuBackends(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
    def test_triton_window_kernel_matches_greedy(self):  # pragma: no cover - box only
        """Same equivalence check, but through the *Triton* window kernel
        (``gdn_verify`` + ``gdn_commit``) instead of the torch reference."""
        m0, _cfg = build_m0(seed=21, with_mtp=True)
        ref_rt = spec_rt(gdn_backend="torch")
        _c, sched_ref, _ = make_scheduler(m0, ref_rt)
        ref = generate(sched_ref, PROMPTS[:2], 48)

        spec_rt_ = spec_rt(gdn_backend="triton")
        _c2, sched_spec, _ = make_scheduler(m0, spec_rt_, SpecConfig(k=3))
        got = generate(sched_spec, PROMPTS[:2], 48)
        for a, c in zip(ref, got):
            self.assertEqual(a.output_token_ids, c.output_token_ids)

    @unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
    def test_graph_replay_matches_eager(self):  # pragma: no cover - box only
        """Capture the whole speculative step (k drafts + verify + accept +
        commit) per bucket and replay it: same tokens as the eager path.

        bf16 + flashinfer + head_dim 64, exactly as
        ``test_runtime.TestCudaGraphCapture`` -- FlashInfer has no fp32 KV
        kernels and rejects a 1-tile QK dimension.
        """
        m0, _cfg = build_m0(
            seed=22, with_mtp=True, dtype=torch.bfloat16,
            device=torch.device("cuda:0"), head_dim=64,
        )
        eager_rt = spec_rt(
            device="cuda:0", dtype="bf16", attn_backend="auto",
            use_cuda_graphs=False, page_size=16,
        )
        _c, sched_eager, _ = make_scheduler(m0, eager_rt, SpecConfig(k=2))
        eager = generate(sched_eager, PROMPTS[:2], 32)

        graph_rt = spec_rt(
            device="cuda:0", dtype="bf16", attn_backend="auto",
            use_cuda_graphs=True, page_size=16, graph_buckets=(1, 2, 4),
        )
        comps, sched_graph, spec = make_scheduler(m0, graph_rt, SpecConfig(k=2))
        comps.decoder.warmup()
        comps.decoder.capture()
        spec.warmup()
        spec.capture(pool_handle=comps.decoder._pool)
        graphed = generate(sched_graph, PROMPTS[:2], 32)

        for a, c in zip(eager, graphed):
            self.assertEqual(a.output_token_ids, c.output_token_ids)


# =========================================================================== #
# 7. verify-pass GEMM routing must match the decode step
# =========================================================================== #
class TestSpecGemmRouting(unittest.TestCase):
    """The verify pass must resolve the *decode step's* GEMM backend.

    Getting this wrong makes the real model diverge from plain greedy at
    every k with identical first-divergence indices: the signature of a
    k-independent, deterministic difference rather than noise.

    **The GEMM routing table these tests read is owned by the dispatcher**, and
    it can change: the current table makes ``deepgemm`` the winner at bucket 32
    as well as 64, which removes the M=64 threshold this hazard originally
    straddled. The invariant below is unchanged and still required -- the
    M=128 -> 512 and 512 -> up boundaries are still there for larger buckets,
    ``use_fp32_reduce`` still flips at 128, and a future table can reintroduce a
    threshold anywhere -- but the *non-vacuity* half is now derived from the
    live table instead of pinned to M=64.

    ``ResolvedLinear`` memoises its GEMM backend per ``gemm_dispatch
    .m_bucket(M)``, where ``M`` is the *row* count of the activation. A
    speculative verify pass packs ``n = k+1`` token rows per sequence, so at
    ``bucket=32`` it presented M = 64/96/128 instead of 32 -- across
    ``gemm.dispatch``'s measured M=64 threshold, which swaps 256 of the real
    model's 305 linears from ``vllm_marlin_fp8_w8a16`` to
    ``flashinfer_fp8_blockscale``. Measured on the real model: those
    two kernels differ by relL2 2.6e-2 on the real shapes, which flips a greedy
    argmax roughly once every 38 tokens over a 248,320 vocabulary.

    Neither of these tests needs FP8 weights or a GPU: the first pins the
    routing rule directly, the second pins the *mechanism* -- which m-buckets a
    speculative step is allowed to touch -- and both fail if the verify pass routes per row.
    """

    BUCKETS = (1, 8, 32, 128)
    KS = (1, 2, 3)

    def test_routing_rule_is_per_sequence_not_per_row(self):
        """The invariant, plus a non-vacuity guard that survives table changes.

        A test which cannot fail is not a test, but hard-coding one threshold
        (``priority_for_m(32) != priority_for_m(64)``) makes the guard brittle
        to a routing table this test does not own; the current table has no
        threshold at M=64 at all.

        So it checks the invariant on the live table, and then keeps itself
        honest in whichever of the two regimes the table is in: if any threshold
        the spec path can straddle still exists, assert it really does change the
        answer without the scope (the old guard, re-derived rather than pinned);
        if the table is uniform over that range, assert the *mechanism* instead,
        because that is all that is left to protect.
        """
        # -- 1. the invariant itself, on whatever table is deployed ---------- #
        for bucket in self.BUCKETS:
            decode = gemm_dispatch.priority_for_m(bucket)
            for k in self.KS:
                n = k + 1
                with gemm_dispatch.rows_per_sequence(n):
                    verify = gemm_dispatch.priority_for_m(bucket * n)
                self.assertEqual(
                    verify, decode,
                    f"verify window (bucket={bucket}, k={k}, {bucket * n} rows) routes "
                    f"to {verify[0]} where the decode step it reproduces routes to "
                    f"{decode[0]}",
                )

        # -- 2. the mechanism, which holds no matter what the table says ----- #
        for bucket in self.BUCKETS:
            for k in self.KS:
                n = k + 1
                with gemm_dispatch.rows_per_sequence(n):
                    self.assertEqual(gemm_dispatch.sequence_m(bucket * n), bucket)
                    self.assertEqual(
                        gemm_dispatch.m_bucket(gemm_dispatch.sequence_m(bucket * n)),
                        gemm_dispatch.m_bucket(bucket),
                    )

        # -- 3. non-vacuity, re-derived from the live table ------------------ #
        straddled = [
            (b, b * (k + 1))
            for b in self.BUCKETS
            for k in self.KS
            if gemm_dispatch.priority_for_m(b)[0]
            != gemm_dispatch.priority_for_m(b * (k + 1))[0]
        ]
        if straddled:
            b, m = straddled[0]
            self.assertNotEqual(
                gemm_dispatch.priority_for_m(b)[0],
                gemm_dispatch.priority_for_m(m)[0],
                "internal inconsistency in this test's own scan",
            )
        else:
            # No threshold in the spec path's range today, so part 1 is
            # trivially satisfied and part 2 is the whole content of the test.
            # Assert the scope is a real, observable transformation rather than
            # letting a no-op dispatcher make this test pass vacuously.
            with gemm_dispatch.rows_per_sequence(4):
                self.assertEqual(gemm_dispatch.rows_per_sequence_now(), 4)
                self.assertEqual(gemm_dispatch.sequence_m(512), 128)
            self.assertEqual(gemm_dispatch.sequence_m(512), 512)

    def test_scope_is_restored_after_an_exception(self):
        self.assertEqual(gemm_dispatch.rows_per_sequence_now(), 1)
        with self.assertRaises(ValueError):
            with gemm_dispatch.rows_per_sequence(4):
                self.assertEqual(gemm_dispatch.rows_per_sequence_now(), 4)
                raise ValueError("boom")
        self.assertEqual(gemm_dispatch.rows_per_sequence_now(), 1)

    @staticmethod
    def _bucket_keys(comps) -> set:
        """Every ``m_bucket`` key any ``ResolvedLinear`` in the main stack has
        memoised a backend for.  ``ResolvedLinear`` uses ``__slots__``, so walk
        the (plain) mixer/MLP objects that hold them."""
        keys = set()
        holders = []
        for layer in comps.model.layers:
            holders += [layer.mixer, layer.mlp]
        for holder in holders:
            for attr in vars(holder).values():
                if isinstance(attr, ResolvedLinear):
                    keys |= set(attr.resolved_backends().keys())
        keys |= set(comps.model.lm_head.resolved_backends().keys())
        return keys

    def test_a_spec_step_touches_only_the_decode_m_bucket(self):
        """End-to-end: after a speculative step at batch ``b``, no
        ``ResolvedLinear`` has memoised a backend for a bucket the plain decode
        step at the same ``b`` would not have used.

        This is the assertion that catches per-row routing: without the
        ``rows_per_sequence`` scope the k=3 run adds ``m_bucket(b*4)``, a *different* key, and
        on the real (FP8) checkpoint that key resolves to a different kernel.
        """
        b, max_tokens = 4, 12
        m0, _cfg = build_m0(seed=23, with_mtp=True)

        c_ref, sched_ref, _ = make_scheduler(m0, spec_rt())
        generate(sched_ref, PROMPTS[:b], max_tokens)
        decode_keys = self._bucket_keys(c_ref)
        self.assertTrue(decode_keys, "no ResolvedLinear resolved anything -- test is broken")

        # Under speculation the requests emit 1..k+1 tokens per step and so
        # finish at *different* steps, which legitimately shrinks the live batch
        # (and its graph bucket) to 1..b. Those smaller buckets are allowed; a
        # bucket ABOVE m_bucket(b) is the bug.
        allowed = decode_keys | {gemm_dispatch.m_bucket(x) for x in range(1, b + 1)}

        for k in (1, 2, 3):
            with self.subTest(k=k):
                c_spec, sched_spec, _ = make_scheduler(m0, spec_rt(), SpecConfig(k=k))
                generate(sched_spec, PROMPTS[:b], max_tokens)
                spec_keys = self._bucket_keys(c_spec)
                self.assertTrue(
                    spec_keys <= allowed,
                    f"k={k}: the speculative step resolved GEMM backends for "
                    f"m_bucket(s) {sorted(spec_keys - allowed)} that no plain decode "
                    f"step at this batch ever uses ({sorted(allowed)}). On an FP8 "
                    f"checkpoint that is a different kernel with a different "
                    f"activation precision.",
                )


# =========================================================================== #
# 8. real-model-*shaped* equivalence  (the shapes the tiny config was missing)
# =========================================================================== #
def real_shaped_config() -> QwenFastConfig:
    """A tiny model with Qwen3.8-27B's **structural** shape parameters.

    ``tiny_config`` (test_runtime) is small in every dimension at once, so it
    misses four things the real checkpoint has that could hide a shape-dependent
    bug:

    ==================== ============ ============ ==============================
    parameter            tiny_config  here         real model
    ==================== ============ ============ ==============================
    GQA group size       2 (2/1)      **6** (24/4) 6 (24/4)
    GVA group size       2 (4/2)      **3** (12/4) 3 (48/16)
    conv kernel width    4            4            4
    partial rotary       0.25         0.25         0.25 (64 of 256 dims)
    full-attn interval   1 of 3       **1 of 4**   1 of 4 (16 of 64)
    chunked prefill      never        **3 chunks** yes (2000-token prompts)
    ==================== ============ ============ ==============================

    Chunked prefill is the important addition: the gate's own prompts are
    50-250 tokens against a 8192-token budget, so the gate alone never runs the
    speculative path across a prefill chunk boundary at all -- ``on_prefill``'s
    ``h_prev`` hand-off between chunks was untested end to end.
    """
    return QwenFastConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        vocab_size=64,
        num_attention_heads=24,      # GQA group 6, as shipped
        num_key_value_heads=4,
        head_dim=64,                 # 4 QK tiles: FlashInfer-instantiable
        attn_output_gate=True,
        full_attention_interval=4,
        linear_num_value_heads=12,   # GVA group 3, as shipped
        linear_num_key_heads=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,    # window n<=4 is <= the conv kernel width
        rope_theta=1e5,
        partial_rotary_factor=0.25,  # rotate 16 of 64 dims, pass the rest through
        max_position_embeddings=512,
        layer_types=[
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
        ],
        mtp_num_hidden_layers=1,
        mtp_use_dedicated_embeddings=False,
        bos_token_id=0,
        eos_token_id=1,
        tie_word_embeddings=False,
    )


_REAL_SHAPED_CACHE = {}


def build_real_shaped(seed: int = 31):
    key = seed
    if key not in _REAL_SHAPED_CACHE:
        torch.manual_seed(seed)
        cfg = real_shaped_config()
        model = QwenFastForCausalLM(cfg, with_mtp=True, mtp_hidden_first=False)
        for p in model.parameters():
            p.data.normal_(0, 0.02)
        model = model.to(DEVICE).eval()
        _REAL_SHAPED_CACHE[key] = (model, cfg)
    return _REAL_SHAPED_CACHE[key]


def real_shaped_rt(**overrides):
    """Deliberately small ``max_num_batched_tokens``: the prompts below are 60
    tokens, so every request is prefilled in **three** chunks and the
    speculative decode steps interleave with the remaining chunks
    (``prefill_decode_ratio``), which is exactly the hand-off the gate's short
    prompts never exercised."""
    kwargs = dict(
        dtype="fp32",
        ssm_state_dtype="fp32",
        enable_mtp=True,
        page_size=16,
        n_kv_pages=512,
        max_pages_per_seq=32,
        max_model_len=512,
        max_num_seqs=8,
        max_num_batched_tokens=24,   # < the 60-token prompts => chunked prefill
        mlp_tile_tokens=16,
        gdn_backend="torch",
        attn_backend="torch",
        norm_backend="torch",
        use_cuda_graphs=False,
    )
    kwargs.update(overrides)
    return base_rt(**kwargs)


LONG_PROMPTS = [
    [(3 + 7 * i) % 61 + 1 for i in range(60)],
    [(11 + 5 * i) % 59 + 2 for i in range(60)],
    [(29 + 3 * i) % 53 + 3 for i in range(60)],
]


class TestRealShapedSpecEquivalence(unittest.TestCase):
    """Greedy equivalence at the real model's shape parameters, over a prompt
    long enough to be prefilled in several chunks."""

    def _pair(self, k: int, b: int, max_tokens: int = 40):
        m0, _cfg = build_real_shaped()
        prompts = LONG_PROMPTS[:b]

        _c1, sched_ref, _ = make_scheduler(m0, real_shaped_rt())
        ref = generate(sched_ref, prompts, max_tokens)

        _c2, sched_spec, spec = make_scheduler(m0, real_shaped_rt(), SpecConfig(k=k))
        got = generate(sched_spec, prompts, max_tokens)

        for i, (a, c) in enumerate(zip(ref, got)):
            self.assertEqual(len(a.output_token_ids), max_tokens, f"reference seq {i} short")
            self.assertEqual(
                a.output_token_ids, c.output_token_ids,
                f"real-shaped k={k} B={b} seq {i}: spec output diverged\n"
                f"  ref  = {a.output_token_ids}\n"
                f"  spec = {c.output_token_ids}",
            )
        return spec

    def test_chunked_prefill_b1(self):
        for k in (1, 2, 3):
            with self.subTest(k=k):
                self._pair(k, 1)

    def test_chunked_prefill_b3(self):
        for k in (1, 2, 3):
            with self.subTest(k=k):
                self._pair(k, 3)

    def test_prompt_really_was_chunked(self):
        """Guard the guard: if `max_num_batched_tokens` ever grows past the
        prompt length this whole class silently stops testing chunked prefill."""
        rt = real_shaped_rt()
        self.assertLess(
            rt.max_num_batched_tokens, len(LONG_PROMPTS[0]),
            "prompts no longer exceed the prefill budget -- chunked prefill is "
            "not being exercised",
        )

    def test_window_state_parity_at_real_shape(self):
        """SSM + conv state after a chunk-prefilled, speculatively decoded run
        equals sequential decoding's, at GVA 3 / conv width 4."""
        k, b, max_tokens = 3, 2, 32
        m0, _cfg = build_real_shaped()
        c_ref, sched_ref, _ = make_scheduler(m0, real_shaped_rt())
        ref = generate(sched_ref, LONG_PROMPTS[:b], max_tokens)
        c_spec, sched_spec, _ = make_scheduler(m0, real_shaped_rt(), SpecConfig(k=k))
        got = generate(sched_spec, LONG_PROMPTS[:b], max_tokens)
        for i in range(b):
            self.assertEqual(ref[i].output_token_ids, got[i].output_token_ids)
        for slot in range(b):
            torch.testing.assert_close(
                c_spec.model.state_pool[slot].float().cpu(),
                c_ref.model.state_pool[slot].float().cpu(),
                rtol=2e-4, atol=2e-4, msg=f"SSM state diverged on slot {slot}",
            )
            torch.testing.assert_close(
                c_spec.model.conv_pool[slot].float().cpu(),
                c_ref.model.conv_pool[slot].float().cpu(),
                rtol=2e-4, atol=2e-4, msg=f"conv state diverged on slot {slot}",
            )


# =========================================================================== #
# the bench's memory plan -- CPU only, no model, no CUDA
# =========================================================================== #
class TestBenchSpecMemoryPlan(unittest.TestCase):
    """``bench_spec``'s byte budget, checkable on a laptop.

    An over-budget configuration OOMs on an otherwise free H200 only after
    minutes spent loading 27.9 GiB of weights, yet every byte of that failure
    is decidable before the run: it is arithmetic on the pool shapes, and
    ``serve.plan_memory`` already does that arithmetic for the server. These
    tests are that plan applied to ``bench_spec``'s own geometry, so an
    over-budget configuration fails in CI rather than on the gpu host.

    ``arch=None`` here (no checkpoint on the test machine) so the plan falls
    back to ``serve._ARCH`` -- which *is* the Qwen3.8-27B production geometry,
    which is exactly what these bounds are about.
    """

    #: H200 SXM: 141 GB = 139.80 GiB usable, as reported by CUDA's OOM message
    #: ("GPU 0 has a total capacity of 139.80 GiB").
    H200_GIB = 139.80

    def _plan(self, argv):
        from qwenfast.runtime import bench_spec

        args = bench_spec.build_arg_parser().parse_args(["--model", "/nonexistent"] + argv)
        self.budget = args.gpu_memory_utilization * self.H200_GIB
        return bench_spec.plan_for_args(args)

    def test_default_geometry_is_under_125_gib(self):
        """The default configuration must fit with real margin."""
        rt, plan, span, buckets = self._plan([])
        self.assertLess(
            plan["total_gib"], 125.0,
            f"bench_spec's default plan is {plan['total_gib']:.1f} GiB: "
            f"{plan}",
        )
        # ... and inside the same budget `main` enforces.
        self.assertLess(plan["total_gib"], self.budget)
        # the KV pool must actually hold the sweep it is sized for
        self.assertGreaterEqual(plan["kv_tokens_capacity"], max(rt.max_num_seqs, 1) * span)

    def test_defaults_are_the_measured_fastest_config(self):
        """The 'no spec' baseline's knobs."""
        rt, _plan, _span, buckets = self._plan([])
        self.assertEqual(rt.norm_backend, "triton")       # fastest measured
        self.assertEqual(rt.fused_ops_backend, "triton")  # fused SwiGLU + GDN gate
        self.assertEqual(rt.kv_cache_dtype, "bf16")       # default KV dtype
        self.assertEqual(rt.ssm_state_dtype, "fp32")      # the gate needs it
        self.assertTrue(rt.use_cuda_graphs)
        # the server's pin, not RuntimeConfig's "multi"
        self.assertEqual(rt.gemm_weight_cache, "single")
        # B=256 is a separate guarded invocation, not a default cell
        self.assertNotIn(256, buckets)

    def test_the_window_that_oomed_is_now_refused(self):
        """Non-vacuity: a geometry known to OOM on an H200.

        ``--batch 1 8 32 128 256`` with ``RuntimeConfig``'s ``"multi"`` weight
        cache does not fit, and the plan for it must say so -- if this ever
        passes, the guard in ``main`` would let that run start."""
        _rt, plan, _span, _b = self._plan(
            ["--batch", "1", "8", "32", "128", "256", "--gemm-weight-cache", "multi"]
        )
        self.assertGreater(
            plan["total_gib"], self.H200_GIB,
            "the configuration that OOM'd must be visibly over the card",
        )
        # and the single knob that is worth the most of it
        _rt, single, _span, _b = self._plan(
            ["--batch", "1", "8", "32", "128", "256", "--gemm-weight-cache", "single"]
        )
        self.assertAlmostEqual(
            plan["repack_gib"] - single["repack_gib"], single["repack_gib"], places=3
        )
        self.assertGreater(plan["repack_gib"] - single["repack_gib"], 23.0)

    def test_guarded_b256_cell_fits_at_fp16_state(self):
        """The separate B=256 invocation.

        fp32 state does *not* fit beside a 256-slot pool and the repack cache;
        fp16 does, at the cost of making the gate advisory -- which is why it
        is a separate invocation with ``--skip-correctness`` rather than a
        cell of the gated sweep."""
        _rt, fp32, _s, _b = self._plan(["--batch", "256"])
        _rt, fp16, _s, _b = self._plan(["--batch", "256", "--ssm-state-dtype", "fp16"])
        self.assertGreater(fp32["total_gib"], self.budget)
        self.assertLess(fp16["total_gib"], self.budget)
        self.assertAlmostEqual(fp32["ssm_gib"], 2 * fp16["ssm_gib"], places=3)

    def test_spec_window_cache_is_planned(self):
        """The spec window cache: ~36 KiB/token/layer, ~0.9 GB at B=128 / k=3.

        It is allocated inside the captured graph and therefore resident, so a
        plan that omitted it would be ~1 GiB light at k=3 and ~2 GiB at B=256.
        """
        _rt, plan, _s, _b = self._plan([])
        self.assertAlmostEqual(plan["spec_gib"], 0.9, delta=0.15)
        _rt, k1, _s, _b = self._plan(["--k", "1"])
        # n = k+1, so the term must halve from k=3 (n=4) to k=1 (n=2)
        self.assertLess(k1["spec_gib"], plan["spec_gib"] * 0.6)

    def test_kv_pool_is_sized_from_the_bench_span_not_max_model_len(self):
        """The bench's sequences are ctx + what it generates. Sizing the pool
        from ``--max-model-len`` (2752, the server's) would be +26% of KV for
        tokens no sequence here ever reaches."""
        _rt, plan, span, _b = self._plan([])
        self.assertEqual(span, 2048 + 128)  # ctx + max(gate tokens, sweep budget)
        _rt, wide, _s, _b = self._plan(["--ctx", "2624"])  # span -> 2752
        self.assertGreater(wide["kv_gib"], plan["kv_gib"] * 1.2)


# =========================================================================== #
# 9. the k=0 path control and the correctness-gate helpers
# =========================================================================== #
class TestSpecPathControl(unittest.TestCase):
    """``SpecConfig(k=0)``: a one-position verify window, no drafting.

    This is the control a divergence diagnosis rests on.  It runs the
    plain decode step's *contract* -- one token per sequence per step, ``m``
    always 1 -- through the verify path's *kernels* (``FusedGDN.window`` and
    the torch conv rather than ``FusedGDN.decode`` and ``conv_update``,
    ``SpecAttentionRunner.window`` rather than ``AttentionRunner.decode``,
    packed-window GEMM shapes rather than ``[B, hidden]``).  On the tiny model
    every one of those pairs is the same torch arithmetic, so the streams must
    be *identical* here; on the real FP8 checkpoint the same construction is
    what separates "the verify path rounds differently" from "speculation is
    buggy".
    """

    def test_k0_is_legal_and_negative_is_not(self):
        self.assertEqual(SpecConfig(k=0).n, 1)
        with self.assertRaises(ValueError):
            SpecConfig(k=-1)
        with self.assertRaises(ValueError):
            SpecConfig(k=9)

    def test_k0_matches_plain_greedy_and_emits_one_token_per_step(self):
        m0, _cfg = build_m0(seed=11, with_mtp=True)
        prompts = PROMPTS[:2]
        _c1, sched_ref, _ = make_scheduler(m0, spec_rt())
        ref = generate(sched_ref, prompts, 48)

        _c2, sched_k0, spec = make_scheduler(m0, spec_rt(), SpecConfig(k=0))
        got = generate(sched_k0, prompts, 48)

        for i, (a, b) in enumerate(zip(ref, got)):
            self.assertEqual(a.output_token_ids, b.output_token_ids,
                             f"k=0 control diverged from plain greedy on seq {i}")
        st = spec.stats()
        # a k=0 step commits exactly the bonus token, never more
        self.assertAlmostEqual(st["spec_accept_length"], 1.0, places=6)
        self.assertEqual(st["spec_acceptance_rate"], 0.0)

    def test_keep_window_logits_records_the_verify_pass(self):
        """The logit-parity probe reads ``SpecDecoder.window_logits``; this
        pins that the buffer exists, is the right shape, and actually holds the
        verify pass's logits rather than a stale zero."""
        m0, _cfg = build_m0(seed=11, with_mtp=True)
        comps, sched, spec = make_scheduler(
            m0, spec_rt(), SpecConfig(k=0, keep_window_logits=True)
        )
        self.assertIsNotNone(spec.window_logits)
        self.assertEqual(spec.window_logits.shape[-1], comps.model.config.vocab_size)
        reqs = generate(sched, PROMPTS[:2], 8)
        lg = spec.window_logits[:2]
        self.assertTrue(bool(lg.abs().sum() > 0), "window_logits was never written")
        # both sequences finish on the same (last) step, so the buffer still
        # holds that step's verify logits: its argmax is what each emitted.
        self.assertEqual(
            [int(x) for x in lg.argmax(-1)],
            [r.output_token_ids[-1] for r in reqs],
        )

    def test_no_window_logits_buffer_by_default(self):
        """The buffer is 508 MiB at bucket 128/k=3; nothing in production reads
        it, so it must not be allocated unless a diagnostic asks."""
        m0, _cfg = build_m0(seed=11, with_mtp=True)
        _c, _s, spec = make_scheduler(m0, spec_rt(), SpecConfig(k=1))
        self.assertIsNone(spec.window_logits)


class TestGateV2Helpers(unittest.TestCase):
    """The pure helpers behind the correctness gate's verdict.

    They carry the whole difference between "a rollback bug" and "bf16 rounding":
    ``_locate`` says whether a divergence landed at window position 0 of a step
    whose predecessor rejected drafts (rollback) or scattered across offsets
    (arithmetic).  Without it, a report can only say "prompt 8 diverged at
    token 10".
    """

    def test_first_divergence(self):
        from qwenfast.runtime.bench_spec import _first_divergence

        self.assertIsNone(_first_divergence([1, 2, 3], [1, 2, 3]))
        self.assertEqual(_first_divergence([1, 2, 3], [1, 9, 3]), 1)
        self.assertEqual(_first_divergence([1, 2, 3], [1, 2]), 2)

    def test_locate_maps_a_token_index_into_the_window_structure(self):
        from qwenfast.runtime.bench_spec import _locate

        # k=2 (n=3): steps emitted 3, 1, 2, 3 tokens -> boundaries 0, 3, 4, 6
        emits = [3, 1, 2, 3]
        self.assertEqual(_locate(emits, 0, 3)["step"], 0)
        self.assertEqual(_locate(emits, 2, 3)["offset_in_window"], 2)
        got = _locate(emits, 3, 3)
        self.assertEqual((got["step"], got["offset_in_window"]), (1, 0))
        self.assertTrue(got["step_rejected"])          # m=1 < n=3
        self.assertFalse(got["prev_step_rejected"])    # step 0 accepted all 3
        got = _locate(emits, 4, 3)
        self.assertEqual((got["step"], got["offset_in_window"]), (2, 0))
        self.assertTrue(got["prev_step_rejected"])     # step 1 emitted 1 of 3
        self.assertEqual(_locate(emits, 999, 3)["step"], -1)

    def test_gate_thresholds_are_named_and_documented(self):
        """The three numbers the verdict turns on are module constants, not
        literals buried in an expression -- a gate whose threshold cannot be
        cited is not a gate."""
        from qwenfast.runtime import bench_spec as bs

        self.assertGreater(bs.MARGIN_SLACK, 1.0)
        self.assertGreaterEqual(bs.SPEC_DIVERGENCE_SLACK, 0)
        self.assertLess(bs.PATH_DISAGREEMENT_MAX, 0.05)


# =========================================================================== #
# 10. server-integration switching
#     `--spec-k`/`--spec-max-batch`: a per-*step* policy, not a build-time
#     one -- the same scheduler serves both shapes and switches every call as
#     the running batch crosses the threshold.
# =========================================================================== #
class TestSpecMaxBatchPolicy(unittest.TestCase):
    """``Scheduler(spec_max_batch=N)``: spec runs only while the live decode
    batch is <= N, even though every request is greedy and ``spec`` is
    configured -- the serving policy (spec wins at B<=8, loses at
    B>=128), exposed as ``--spec-max-batch`` on the server."""

    def test_batch_above_threshold_never_takes_the_spec_path(self):
        m0, _cfg = build_m0(seed=41, with_mtp=True)
        prompts = PROMPTS[:4]
        _c, sched, spec = make_scheduler(m0, spec_rt(), SpecConfig(k=2), spec_max_batch=1)
        got = generate(sched, prompts, 16)
        self.assertEqual(spec.steps, 0, "batch=4 > spec_max_batch=1 must never spec-decode")

        _c2, sched_ref, _ = make_scheduler(m0, spec_rt())
        ref = generate(sched_ref, prompts, 16)
        for a, c in zip(ref, got):
            self.assertEqual(a.output_token_ids, c.output_token_ids)

    def test_batch_at_or_below_threshold_takes_the_spec_path(self):
        m0, _cfg = build_m0(seed=42, with_mtp=True)
        prompts = PROMPTS[:2]
        _c, sched, spec = make_scheduler(m0, spec_rt(), SpecConfig(k=2), spec_max_batch=2)
        generate(sched, prompts, 16)
        self.assertGreater(spec.steps, 0, "batch=2 <= spec_max_batch=2 must spec-decode")

    def test_default_is_unbounded(self):
        """``spec_max_batch=None`` (the direct-``SpecDecoder``-test default,
        and every caller that does not set it) must reproduce the exact pre-existing
        behaviour: spec runs at any batch ``spec.eligible`` allows."""
        m0, _cfg = build_m0(seed=46, with_mtp=True)
        _c, sched, spec = make_scheduler(m0, spec_rt(), SpecConfig(k=2))
        self.assertIsNone(sched.spec_max_batch)
        generate(sched, PROMPTS[:4], 16)
        self.assertGreater(spec.steps, 0)

    def test_toggling_across_the_threshold_mid_stream_matches_plain_greedy(self):
        """The policy's whole point: batch crosses the threshold *while
        requests are live*, not only at admission. Two short requests finish
        early and drop the live batch from 4 (> threshold=2, plain) to 2
        (<= threshold, spec) for the two survivors -- exercising
        ``on_plain_step`` maintaining state through the plain stretch and the
        spec path resuming correctly from it, end to end through the real
        scheduler loop (the correctness gate's own criterion for what
        "correct" means here: the emitted stream must equal plain greedy's --
        checked below with :func:`bench_spec._first_divergence` so a failure
        names the exact token rather than a full-list diff).
        """
        from qwenfast.runtime.bench_spec import _first_divergence

        m0, _cfg = build_m0(seed=43, with_mtp=True)
        prompts = PROMPTS[:4]
        max_tokens = [6, 6, 40, 40]  # two finish fast: live batch 4 -> 2

        def run(spec_cfg, spec_max_batch=None):
            _comps, sched, spec = make_scheduler(m0, spec_rt(), spec_cfg, spec_max_batch)
            reqs = []
            for i, (p, mt) in enumerate(zip(prompts, max_tokens)):
                r = Request(
                    request_id=f"r{i}", prompt_token_ids=list(p),
                    params=GenParams(temperature=0.0, max_tokens=mt, ignore_eos=True),
                )
                sched.add_request(r)
                reqs.append(r)
            guard = 0
            while sched.has_work() and guard < 20000:
                sched.step()
                guard += 1
            if sched.has_work():  # pragma: no cover - would be a livelock bug
                raise AssertionError("scheduler did not drain")
            return reqs, spec

        ref, _ = run(None)
        got, spec = run(SpecConfig(k=2), spec_max_batch=2)

        for i, (a, c) in enumerate(zip(ref, got)):
            self.assertEqual(len(a.output_token_ids), max_tokens[i])
            div = _first_divergence(a.output_token_ids, c.output_token_ids)
            self.assertIsNone(
                div,
                f"seq {i}: toggling across --spec-max-batch diverged from plain greedy "
                f"at token {div}\n  ref = {a.output_token_ids}\n  got = {c.output_token_ids}",
            )
        # the two survivors must actually have used the spec path once the
        # live batch dropped to 2 -- otherwise this test could not tell "the
        # threshold correctly gated the transition" from "spec never ran".
        self.assertGreater(spec.steps, 0)

    def test_mixed_sampling_batch_fallback_still_respects_the_cap(self):
        """The pre-existing veto (any sampling request forces that step
        plain) and the batch-size cap are independent and
        compose: a batch under the cap but with a sampling request must still
        fall back, and `on_plain_step` must not choke on it either."""
        m0, _cfg = build_m0(seed=47, with_mtp=True)
        _c, sched, spec = make_scheduler(m0, spec_rt(), SpecConfig(k=2), spec_max_batch=8)
        r0 = Request(
            request_id="g0", prompt_token_ids=list(PROMPTS[0]),
            params=GenParams(temperature=0.0, max_tokens=10, ignore_eos=True),
        )
        r1 = Request(
            request_id="s0", prompt_token_ids=list(PROMPTS[1]),
            params=GenParams(temperature=0.8, max_tokens=10, ignore_eos=True),
        )
        sched.add_request(r0)
        sched.add_request(r1)
        guard = 0
        while sched.has_work() and guard < 5000:
            sched.step()
            guard += 1
        self.assertEqual(len(r0.output_token_ids), 10)
        self.assertEqual(len(r1.output_token_ids), 10)
        self.assertEqual(spec.steps, 0, "a sampling request in the batch must veto spec")


class TestSpecOnPlainStepHook(unittest.TestCase):
    """``SpecDecoder.on_plain_step`` in isolation: it must move ``h_prev``
    (and, transitively, keep the MTP KV layer populated) across a plain
    stretch, and correctness must never depend on it having run (the verify
    pass reads only the target model's own logits over
    the real committed context, never the MTP layer or ``h_prev``)."""

    def _run_n_plain_steps(self, disable_hook: bool, n: int = 12):
        m0, _cfg = build_m0(seed=45, with_mtp=True)
        # spec_max_batch=0: every step -- even a lone request -- takes the
        # plain path, so `n` steps is `n` calls to `on_plain_step` (or, in
        # the disabled arm, `n` skipped opportunities).
        comps, sched, spec = make_scheduler(m0, spec_rt(), SpecConfig(k=2), spec_max_batch=0)
        if disable_hook:
            spec.on_plain_step = lambda *a, **kw: None  # type: ignore[method-assign]
        reqs = [
            Request(
                request_id=f"r{i}", prompt_token_ids=list(p),
                params=GenParams(temperature=0.0, max_tokens=50, ignore_eos=True),
            )
            for i, p in enumerate(PROMPTS[:2])
        ]
        for r in reqs:
            sched.add_request(r)
        for _ in range(n):
            sched.step()
        return reqs, spec, comps

    def test_hook_advances_h_prev_and_a_disabled_hook_leaves_it_frozen(self):
        reqs_with, spec_with, _c1 = self._run_n_plain_steps(disable_hook=False)
        reqs_without, spec_without, _c2 = self._run_n_plain_steps(disable_hook=True)

        slot_with = reqs_with[0].slot
        slot_without = reqs_without[0].slot
        self.assertIsNotNone(slot_with)
        self.assertIsNotNone(slot_without)

        h_with = spec_with.h_prev[slot_with].clone()
        h_without = spec_without.h_prev[slot_without].clone()
        # `on_prefill` sets both to the same starting value (identical
        # prompts, identical weights, identical seed); only the hook moves it
        # afterwards, so a disabled hook must leave it exactly where prefill
        # put it while 12 real plain-decode steps ran.
        self.assertFalse(
            torch.allclose(h_with, h_without),
            "on_plain_step must change h_prev across a plain stretch; a disabled "
            "hook must leave it exactly where on_prefill set it",
        )
        self.assertTrue(bool(torch.isfinite(h_with).all()))

    def test_disabling_the_hook_never_breaks_correctness(self):
        """The safety property that matters: whether or not
        the hook ran, resuming spec after a plain stretch must still
        reproduce plain greedy exactly (only accept length, not the emitted
        stream, may be worse without it -- the same invariant, re-pinned
        here at the unit level rather than only end to end)."""
        from qwenfast.runtime.bench_spec import _first_divergence

        m0, _cfg = build_m0(seed=48, with_mtp=True)
        prompts = PROMPTS[:2]
        max_tokens = 32
        _c0, sched_ref, _ = make_scheduler(m0, spec_rt())
        ref = generate(sched_ref, prompts, max_tokens)

        def run(disable_hook):
            _comps, sched, spec = make_scheduler(m0, spec_rt(), SpecConfig(k=3), spec_max_batch=2)
            if disable_hook:
                spec.on_plain_step = lambda *a, **kw: None  # type: ignore[method-assign]
            reqs = [
                Request(
                    request_id=f"r{i}", prompt_token_ids=list(p),
                    params=GenParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True),
                )
                for i, p in enumerate(prompts)
            ]
            for r in reqs:
                sched.add_request(r)
            # force several plain-path steps, then let spec take over --
            # simulates "batch was above the cap, then dropped" without
            # needing extra staggered requests.
            sched.spec_max_batch = 0
            for _ in range(6):
                sched.step()
            sched.spec_max_batch = 2
            guard = 0
            while sched.has_work() and guard < 20000:
                sched.step()
                guard += 1
            return reqs

        for disable_hook in (False, True):
            with self.subTest(disable_hook=disable_hook):
                got = run(disable_hook)
                for i in range(2):
                    div = _first_divergence(ref[i].output_token_ids, got[i].output_token_ids)
                    self.assertIsNone(
                        div,
                        f"seq {i} (disable_hook={disable_hook}) diverged at token {div}",
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
