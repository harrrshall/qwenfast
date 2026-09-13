"""CPU unit tests for qwenfast.runtime.

Everything here runs on CPU with a tiny synthetic config (hidden 64, 2
Gated-DeltaNet + 1 full-attention layer, vocab 32) so it gates every commit
without a GPU, per the same convention as ``kernels_gdn/tests``,
``gemm/tests``, ``attn/tests``. Two kinds of coverage:

* **Fused-model parity**: build a tiny dense reference (``qwenfast.model``)
  module with random weights, fuse it with :func:`~qwenfast.runtime.fused_model
  .fused_weights_from_module` (the exact code path ``from_m0_module`` uses),
  and assert the fused model's prefill/decode logits agree with the reference's dense
  forward -- the same assertion ``verify_vs_hf.py`` makes against HF, one
  level down the stack. No FP8 is involved (every tensor here is a plain
  ``torch.Tensor``, so ``gemm.dispatch.linear`` always takes the
  ``bf16_dequant`` path), which is exactly what "bf16, no fp8" means for
  this test: the *precision path*, not necessarily the activation dtype --
  both a bf16-weights/bf16-activations run and a tighter fp32 run are
  covered. (That path is named ``bf16_native`` rather than
  ``bf16_dequant`` -- same single ``F.linear``, but a name that does not
  collide with the 208 ms/step fp8-dequant fallback. See
  ``TestResolvedLinearBackends``.)
* **Scheduler / sampler**: pure Python + CPU-tensor logic (admission,
  chunked-prefill budget accounting, preemption under page pressure, abort,
  completion, sampler determinism) exercised end to end through the exact
  same ``Scheduler``/``GraphedDecoder``/``QwenFastEngine`` classes the GPU
  path uses, just with ``use_cuda_graphs=False`` and the tiny model.

GPU-only tests (CUDA-graph capture/replay parity) are marked
``unittest.skipUnless(torch.cuda.is_available(), ...)`` -- meant for the
remote GPU host (see ``README.md``).

Run::

    python -m unittest discover -s engine/qwenfast/runtime/tests -v
    # or
    python engine/qwenfast/runtime/tests/test_runtime.py
    # or (pytest also collects unittest.TestCase)
    pytest engine/qwenfast/runtime/tests/test_runtime.py -v
"""

from __future__ import annotations

import asyncio
import os
import sys
import unittest

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))  # -> engine/

# The reference-vs-fused parity tests below compare against a tight (1e-2 / 0.05)
# absolute tolerance derived on CPU, where "fp32" really is IEEE fp32. On a
# CUDA device, `torch.matmul`/`F.linear` default to TF32 for fp32 inputs
# (~10-bit mantissa, not 23), and -- critically -- TF32's rounding pattern
# is *not* invariant to how a GEMM is tiled: the reference's four separate per-
# projection GEMMs and the fused model's one wide concatenated GEMM are the
# same linear algebra but different tile shapes, so TF32 can round them to
# measurably different results even though true fp32 (associative up to
# ~1e-6) would not. Disabling TF32 for this test module only (never touched
# in the production RuntimeConfig(dtype="bf16") path, which never runs a
# fp32 matmul in the first place) removes that confound so a real numerics
# bug isn't masked by -- or mistaken for -- a precision artifact. No-op on
# CPU (these flags only affect CUDA/cuDNN kernels).
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

# Everything below builds both the dense reference and the fused model on the
# same device -- DEVICE is cuda:0 whenever CUDA is present, cpu otherwise.
# Two hazards follow from *not* doing this:
#
# 1. `qwenfast.model`'s GDN path auto-selects `fla` (`use_fla=True` default)
#    whenever the `fla` package is importable (`HAS_FLA`, a module-level
#    constant set at import time -- independent of what device any given
#    tensor happens to be on). On a GPU host with `fla` installed, a
#    reference model/tensors left on CPU by accident still route through
#    `fla`'s Triton kernels, which then fail on CPU tensors with "Pointer
#    argument cannot be accessed from Triton (cpu tensor?)". Building the
#    reference on DEVICE fixes this for real GPU runs; `use_fla=False`
#    (passed explicitly at every reference forward call below) fixes it
#    unconditionally and *also* keeps the reference on the same "torch"
#    numerics path as the fused model's `gdn_backend="torch"` (these tests
#    deliberately never depend on `fla`/Triton's own correctness, which the
#    GDN kernel tests cover).
# 2. A reference model left on CPU makes `from_m0_module` fuse a model out
#    of CPU weight tensors while every pool/buffer (built straight from a
#    `cuda:0` `RuntimeConfig`) is CUDA -- a device-mismatch inside the first
#    op that combines the two ("Expected all tensors to be on the same
#    device"). `build_m0` therefore moves the model to DEVICE before
#    returning, and every test's own tensor construction
#    (`make_prefill_batch`, `DeviceBuffers`, the reference-forward input
#    tensors) uses DEVICE too, rather than a hardcoded `torch.device("cpu")`.
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from qwenfast.model import QwenFastForCausalLM  # noqa: E402
from qwenfast.weights import QwenFastConfig  # noqa: E402

from qwenfast.runtime.fused_model import (  # noqa: E402
    DeviceBuffers,
    FusedQwenForCausalLM,
    RuntimeConfig,
    make_prefill_batch,
)
from qwenfast.runtime.graphs import GraphedDecoder, sample_tokens  # noqa: E402
from qwenfast.runtime.scheduler import GenParams, Request, Scheduler, SlotManager  # noqa: E402
from qwenfast.runtime.engine import EngineComponents, QwenFastEngine  # noqa: E402
from qwenfast.server.engine_api import SamplingParams  # noqa: E402


# =========================================================================== #
# shared fixtures
# =========================================================================== #
def tiny_config(head_dim: int = 16) -> QwenFastConfig:
    """The CPU-test model.

    ``head_dim`` is a parameter because FlashInfer cannot run the default.
    16 is the cheapest head dim that exercises every code path on CPU, but
    FlashInfer's tensor-core decode kernel (which is mandatory at our real
    GQA group size -- see ``attn/flashinfer_attn.py``) tiles the head
    dimension in units of 16 and rejects a 1-tile QK dimension outright:

        FlashInfer Internal Error: Invalid configuration :
        NUM_MMA_Q=1 NUM_MMA_D_QK=1 NUM_MMA_D_VO=1 NUM_MMA_KV=8 ...

    So the GPU-only
    CUDA-graph test builds its model with ``head_dim=64`` -- 4 QK tiles,
    a configuration FlashInfer actually instantiates -- while every CPU
    test keeps the cheaper 16.  ``head_dim`` is independent of
    ``hidden_size`` here (``q_proj`` is ``[num_heads * head_dim * 2,
    hidden]``), so nothing else in the config has to move.
    """
    return QwenFastConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        vocab_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=head_dim,
        attn_output_gate=True,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        rope_theta=1e5,
        partial_rotary_factor=0.25,
        max_position_embeddings=128,
        layer_types=["linear_attention", "linear_attention", "full_attention"],
        mtp_num_hidden_layers=1,
        mtp_use_dedicated_embeddings=False,
        bos_token_id=0,
        eos_token_id=1,
        tie_word_embeddings=False,
    )


def build_m0(seed: int = 0, with_mtp: bool = False, dtype: torch.dtype = torch.float32,
             device: "torch.device | None" = None, head_dim: int = 16):
    torch.manual_seed(seed)
    cfg = tiny_config(head_dim=head_dim)
    model = QwenFastForCausalLM(cfg, with_mtp=with_mtp, mtp_hidden_first=False)
    for p in model.parameters():
        p.data.normal_(0, 0.02)
    if dtype is not torch.float32:
        model = model.to(dtype)
    model = model.to(device or DEVICE)
    model.eval()
    return model, cfg


def base_rt(**overrides) -> RuntimeConfig:
    kwargs = dict(
        device=str(DEVICE),
        dtype="fp32",
        ssm_state_dtype="fp32",
        kv_cache_dtype="bf16",
        page_size=4,
        max_num_seqs=8,
        n_kv_pages=64,
        max_pages_per_seq=32,
        max_num_batched_tokens=64,
        mlp_tile_tokens=32,
        max_model_len=128,
        gdn_backend="torch",
        attn_backend="torch",
        norm_backend="torch",
        use_cuda_graphs=False,
    )
    kwargs.update(overrides)
    return RuntimeConfig(**kwargs)


def build_pieces(rt: RuntimeConfig, model) -> EngineComponents:
    fm = FusedQwenForCausalLM.from_m0_module(model, rt)
    buf = DeviceBuffers(
        max_batch=rt.max_num_seqs, vocab_size=fm.config.vocab_size, max_pages=rt.n_kv_pages, device=DEVICE
    )
    dec = GraphedDecoder(fm, buf, rt)
    return EngineComponents(model=fm, buf=buf, decoder=dec, rt=rt)


# =========================================================================== #
# 1. fused-model parity vs the dense reference
# =========================================================================== #
class TestFusedModelParity(unittest.TestCase):
    def test_prefill_logits_match_m0_fp32(self):
        model, cfg = build_m0(seed=1)
        rt = base_rt(dtype="fp32")
        fm = FusedQwenForCausalLM.from_m0_module(model, rt)

        torch.manual_seed(2)
        ids = [3, 5, 7, 9, 11, 2, 4]
        with torch.no_grad():
            m0_logits = model(torch.tensor([ids], dtype=torch.long, device=DEVICE), use_fla=False)[0]

        slot = 0
        fm.reset_slot(slot)
        fm.kv_pool.ensure_capacity(slot, len(ids))
        batch = make_prefill_batch([ids], [0], [slot], DEVICE)
        with torch.no_grad():
            fused_logits = fm.prefill_forward(batch, all_logits=True)

        diff = (m0_logits.float() - fused_logits.float()).abs()
        self.assertLess(float(diff.max()), 1e-2, f"max abs diff {float(diff.max())}")
        self.assertTrue(torch.equal(m0_logits.argmax(-1), fused_logits.argmax(-1)))

    def test_decode_logits_match_m0_fp32(self):
        model, cfg = build_m0(seed=3)
        rt = base_rt(dtype="fp32")
        fm = FusedQwenForCausalLM.from_m0_module(model, rt)

        ids = [1, 4, 6, 8]
        slot = 0
        fm.reset_slot(slot)
        fm.kv_pool.ensure_capacity(slot, len(ids))
        batch = make_prefill_batch([ids], [0], [slot], DEVICE)
        with torch.no_grad():
            fused_prefill_logits = fm.prefill_forward(batch, all_logits=True)
        next_tok = int(fused_prefill_logits[-1].argmax().item())

        buf = DeviceBuffers(max_batch=4, vocab_size=cfg.vocab_size, max_pages=64, device=DEVICE)
        buf.host["input_ids"][:1] = torch.tensor([next_tok], dtype=torch.int32)
        buf.host["positions"][:1] = torch.tensor([len(ids)], dtype=torch.int32)
        buf.host["slot_ids"][:1] = torch.tensor([slot], dtype=torch.int32)
        buf.upload(["input_ids", "positions", "slot_ids"])
        fm.kv_pool.ensure_capacity(slot, len(ids) + 1)
        fm.attn.plan_decode([slot], 1)
        with torch.no_grad():
            decode_logits = fm.decode_forward(buf, 1)

        full_ids = ids + [next_tok]
        with torch.no_grad():
            m0_logits2 = model(torch.tensor([full_ids], dtype=torch.long, device=DEVICE), use_fla=False)[0]

        diff = (m0_logits2[-1].float() - decode_logits[0].float()).abs()
        self.assertLess(float(diff.max()), 1e-2)
        self.assertEqual(int(m0_logits2[-1].argmax()), int(decode_logits[0].argmax()))

    def test_prefill_logits_match_m0_bf16_no_fp8(self):
        """The bf16 parity check: bf16 weights (never FP8Tensor
        -- from_m0_module always produces plain tensors), max diff < 0.05."""
        model, cfg = build_m0(seed=4, dtype=torch.bfloat16)
        rt = base_rt(dtype="bf16")
        fm = FusedQwenForCausalLM.from_m0_module(model, rt)

        from qwenfast.gemm.fused_weights import FP8Tensor

        for layer in fm.layers:
            for lin in (
                [layer.mlp.gate_up, layer.mlp.down]
                + (
                    [layer.mixer.in_proj_qkvz, layer.mixer.in_proj_ba, layer.mixer.out_proj]
                    if hasattr(layer.mixer, "in_proj_qkvz")
                    else [layer.mixer.qkv, layer.mixer.o_proj]
                )
            ):
                self.assertNotIsInstance(lin.weight, FP8Tensor)

        ids = [2, 6, 10, 14, 3]
        with torch.no_grad():
            m0_logits = model(torch.tensor([ids], dtype=torch.long, device=DEVICE), use_fla=False)[0]

        slot = 0
        fm.reset_slot(slot)
        fm.kv_pool.ensure_capacity(slot, len(ids))
        batch = make_prefill_batch([ids], [0], [slot], DEVICE)
        with torch.no_grad():
            fused_logits = fm.prefill_forward(batch, all_logits=True)

        diff = (m0_logits.float() - fused_logits.float()).abs()
        self.assertLess(float(diff.max()), 0.05, f"max abs diff {float(diff.max())} (parity tolerance is < 0.05)")

    def test_conv_weight_is_width_major(self):
        """kernels_gdn/state.py::prepare_conv_weight: the conv
        weight should be re-strided so channel is the contiguous axis
        (stride(0) == 1), matching the width-major conv state pool default
        (kernels_gdn.state.alloc_conv_state_pool's default `layout=
        "width_major"`). Value-identical either way -- this only checks the
        performance-relevant memory layout, not correctness of the numbers
        (that's covered by the parity tests above, which already exercise
        this code path)."""
        model, cfg = build_m0(seed=6)
        rt = base_rt(dtype="fp32")
        fm = FusedQwenForCausalLM.from_m0_module(model, rt)
        gdn_layers = [layer.mixer for layer in fm.layers if hasattr(layer.mixer, "conv_w")]
        self.assertTrue(gdn_layers, "expected at least one GDN layer in the tiny config")
        for mixer in gdn_layers:
            self.assertEqual(mixer.conv_w.stride(0), 1, f"conv_w strides {mixer.conv_w.stride()}")

    def test_mtp_forward_both_orders_run(self):
        model, cfg = build_m0(seed=5, with_mtp=True)
        rt = base_rt(dtype="fp32", enable_mtp=True)
        fm = FusedQwenForCausalLM.from_m0_module(model, rt)
        self.assertIsNotNone(fm.mtp)

        ids = [1, 2, 3, 4, 5, 6]
        slot = 0
        fm.reset_slot(slot)
        fm.kv_pool.ensure_capacity(slot, len(ids))
        batch = make_prefill_batch([ids], [0], [slot], DEVICE)
        with torch.no_grad():
            _logits, hidden = fm.prefill_forward(batch, all_logits=True, return_hidden=True)

        next_ids = ids[1:]
        mtp_batch = make_prefill_batch([next_ids], [1], [slot], DEVICE)
        for hidden_first in (False, True):
            with torch.no_grad():
                out = fm.mtp_forward(
                    torch.tensor(next_ids, dtype=torch.int32, device=DEVICE), hidden[:-1], mtp_batch, hidden_first=hidden_first
                )
            self.assertEqual(tuple(out.shape), (len(next_ids), cfg.vocab_size))


# =========================================================================== #
# 2. sampler
# =========================================================================== #
class TestResolvedLinearBackends(unittest.TestCase):
    """The engine's resolved-backend table is the main artefact for diagnosing
    which linear path each layer actually takes, and is easy to misread;
    these are the invariants that make it readable."""

    def setUp(self):
        self.model, _ = build_m0(seed=11)
        self.fm = FusedQwenForCausalLM.from_m0_module(self.model, base_rt(dtype="bf16"))

    def _resolve_everything(self):
        """Run one eager decode-shaped step so every ResolvedLinear resolves,
        the way `GraphedDecoder.warmup` does before capture."""
        ids = [1, 4, 6, 8]
        slot = 0
        self.fm.reset_slot(slot)
        self.fm.kv_pool.ensure_capacity(slot, len(ids) + 1)
        batch = make_prefill_batch([ids], [0], [slot], DEVICE)
        with torch.no_grad():
            self.fm.prefill_forward(batch, all_logits=False)

    def test_a_plain_bf16_model_resolves_to_bf16_native_not_bf16_dequant(self):
        """The tiny model is entirely unquantized, so every linear is the
        `in_proj_ba`/`lm_head` case: a single `F.linear`, correctly named."""
        from qwenfast.runtime.bench_runtime import collect_gemm_backends, summarize_backends

        self._resolve_everything()
        summary = summarize_backends(collect_gemm_backends(self.fm))
        self.assertTrue(summary, "nothing resolved -- the probe did not run")
        for bucket, counts in summary.items():
            with self.subTest(bucket=bucket):
                self.assertNotIn("bf16_dequant", counts)
                self.assertEqual(set(counts), {"bf16_native"})

    def test_check_resolved_backends_is_clean_for_this_model(self):
        from qwenfast.runtime.bench_runtime import check_resolved_backends, collect_gemm_backends

        self._resolve_everything()
        warnings = check_resolved_backends(
            self.fm, collect_gemm_backends(self.fm), verbose=False
        )
        self.assertEqual(warnings, [])

    def test_check_resolved_backends_flags_a_suspect_backend(self):
        """Self-guarding: the check is only worth having if it fires. Force one
        layer onto `bf16_dequant` and assert it is reported by name."""
        from qwenfast.runtime.bench_runtime import check_resolved_backends

        warnings = check_resolved_backends(
            self.fm, {"layer0.mlp.down": {"32": "bf16_dequant"}}, verbose=False
        )
        self.assertEqual(len(warnings), 1)
        self.assertIn("layer0.mlp.down", warnings[0])
        self.assertIn("bf16_dequant", warnings[0])
        self.assertIn("208 ms/step", warnings[0])

    def test_a_forced_fp8_backend_is_not_reported_for_a_bf16_weight(self):
        """`--gemm-backend deepgemm` on this model used to make every
        `ResolvedLinear` report `{0: 'deepgemm'}` -- including the 49 layers
        that provably cannot run it. It now reports what will actually run."""
        from qwenfast.runtime.bench_runtime import collect_gemm_backends, summarize_backends

        fm = FusedQwenForCausalLM.from_m0_module(
            self.model, base_rt(dtype="bf16", gemm_backend="deepgemm")
        )
        summary = summarize_backends(collect_gemm_backends(fm))
        for bucket, counts in summary.items():
            with self.subTest(bucket=bucket):
                self.assertNotIn("deepgemm", counts)
                self.assertEqual(set(counts), {"bf16_native"})

    def test_forced_ignored_records_what_was_asked_for(self):
        fm = FusedQwenForCausalLM.from_m0_module(
            self.model, base_rt(dtype="bf16", gemm_backend="deepgemm")
        )
        lin = fm.lm_head
        self.assertEqual(lin.forced, "bf16_native")
        self.assertEqual(lin.forced_ignored, "deepgemm")
        self.assertFalse(lin.is_fp8)

    def test_rejection_reasons_are_recorded_during_resolution(self):
        """`resolve_backend`'s `reasons` out-param is what tells "deepgemm is
        unavailable on this host" apart from "deepgemm was policy-filtered
        because marlin owns this weight's one cache slot" -- two facts that
        otherwise look identical from outside."""
        self._resolve_everything()
        reasons = self.fm.lm_head.rejection_reasons()
        self.assertTrue(reasons, "no bucket resolved")
        # bf16_native is rank 1 for a bf16 weight, so nothing precedes it.
        for bucket, why in reasons.items():
            with self.subTest(bucket=bucket):
                self.assertNotIn("bf16_native", why)


class TestSampler(unittest.TestCase):
    def test_greedy_is_argmax(self):
        torch.manual_seed(0)
        logits = torch.randn(6, 32)
        temp = torch.zeros(6)
        top_p = torch.ones(6)
        top_k = torch.zeros(6)
        tok = sample_tokens(logits, temp, top_p, top_k)
        self.assertTrue(torch.equal(tok.long(), logits.argmax(-1)))

    def test_deterministic_with_generator(self):
        torch.manual_seed(0)
        logits = torch.randn(8, 32)
        temp = torch.full((8,), 1.0)
        top_p = torch.full((8,), 0.9)
        top_k = torch.full((8,), 5.0)

        g1 = torch.Generator(device="cpu")
        g1.manual_seed(123)
        out1 = sample_tokens(logits, temp, top_p, top_k, generator=g1)

        g2 = torch.Generator(device="cpu")
        g2.manual_seed(123)
        out2 = sample_tokens(logits, temp, top_p, top_k, generator=g2)

        self.assertTrue(torch.equal(out1, out2))

    def test_top_k_restricts_to_top_k_support(self):
        torch.manual_seed(1)
        logits = torch.randn(1, 32)
        top1 = torch.topk(logits[0], 3).indices
        temp = torch.tensor([1.0])
        top_p = torch.tensor([1.0])
        top_k = torch.tensor([3.0])
        g = torch.Generator(device="cpu")
        g.manual_seed(0)
        for _ in range(20):
            tok = sample_tokens(logits, temp, top_p, top_k, generator=g)
            self.assertIn(int(tok.item()), top1.tolist())

    def test_disabled_top_k_is_le_zero(self):
        logits = torch.randn(4, 16)
        temp = torch.ones(4)
        top_p = torch.ones(4)
        top_k = torch.tensor([0.0, -1.0, -5.0, 0.0])
        # should not raise, and should not crash the top-k masking path
        tok = sample_tokens(logits, temp, top_p, top_k)
        self.assertEqual(tok.shape, (4,))


# =========================================================================== #
# 3. slot manager
# =========================================================================== #
class TestSlotManager(unittest.TestCase):
    def test_alloc_free_roundtrip(self):
        sm = SlotManager(4)
        self.assertEqual(sm.num_free, 4)
        a = sm.alloc()
        b = sm.alloc()
        self.assertNotEqual(a, b)
        self.assertEqual(sm.num_free, 2)
        sm.free(a)
        self.assertEqual(sm.num_free, 3)

    def test_exhaustion_raises(self):
        sm = SlotManager(1)
        sm.alloc()
        with self.assertRaises(RuntimeError):
            sm.alloc()


# =========================================================================== #
# 4. scheduler
# =========================================================================== #
class TestScheduler(unittest.TestCase):
    def setUp(self):
        self.model, self.cfg = build_m0(seed=7)

    def _sched(self, **rt_overrides) -> Scheduler:
        rt = base_rt(**rt_overrides)
        comps = build_pieces(rt, self.model)
        return Scheduler(comps.model, comps.decoder, rt)

    def test_admission_and_completion(self):
        sched = self._sched(max_num_batched_tokens=16, prefill_decode_ratio=2)
        r1 = Request("r1", [3, 4, 5, 6, 7], GenParams(temperature=0.0, max_tokens=4, eos_token_id=999))
        r2 = Request("r2", [1, 2, 3], GenParams(temperature=0.0, max_tokens=3, eos_token_id=999))
        sched.add_request(r1)
        sched.add_request(r2)

        steps = 0
        while sched.has_work() and steps < 100:
            sched.step()
            steps += 1

        self.assertTrue(r1.is_finished and r2.is_finished)
        self.assertEqual(len(r1.output_token_ids), 4)
        self.assertEqual(len(r2.output_token_ids), 3)
        self.assertEqual(r1.finish_reason, "length")
        # both slots should be returned to the free pool
        self.assertEqual(sched.slots.num_free, sched.model.n_slots)

    def test_chunked_prefill_accounting(self):
        sched = self._sched(max_num_batched_tokens=3)
        prompt = list(range(10))  # needs >= 4 chunks of budget 3
        r = Request("long", prompt, GenParams(temperature=0.0, max_tokens=2, eos_token_id=999))
        sched.add_request(r)

        partial_chunks = 0
        steps = 0
        while sched.has_work() and steps < 50:
            before = r.num_computed_tokens
            sched.step()
            steps += 1
            if before < r.num_computed_tokens < len(prompt):
                partial_chunks += 1

        self.assertTrue(r.is_finished)
        self.assertEqual(len(r.output_token_ids), 2)
        self.assertGreaterEqual(partial_chunks, 2, "a 10-token prompt at budget=3 must take multiple chunks")

    def test_preemption_under_page_pressure(self):
        # page_size=2, 7 pages total (6 usable beyond the model's 1-page
        # scratch slot): both prompts' pages fit, but the two requests'
        # *full* lifetime usage (5 + 4 = 9 pages) exceeds the 6 usable pages,
        # so decode-time growth must preempt one of them and both must still
        # complete correctly once pages free up.
        sched = self._sched(page_size=2, n_kv_pages=7, max_pages_per_seq=8, max_num_seqs=4, max_num_batched_tokens=32)
        r1 = Request("a", [1, 2, 3], GenParams(temperature=0.0, max_tokens=6, eos_token_id=999))
        r2 = Request("b", [4, 5], GenParams(temperature=0.0, max_tokens=6, eos_token_id=999))
        sched.add_request(r1)
        sched.add_request(r2)

        steps = 0
        saw_preemption = False
        while sched.has_work() and steps < 200:
            sched.step()
            steps += 1
            if sched._swapped:
                saw_preemption = True

        self.assertTrue(saw_preemption, "this page budget is deliberately too tight to avoid preemption")
        self.assertTrue(r1.is_finished and r2.is_finished)
        self.assertEqual(len(r1.output_token_ids), 6)
        self.assertEqual(len(r2.output_token_ids), 6)
        self.assertEqual(sched._swapped, {})

    def test_abort_waiting_request(self):
        sched = self._sched()
        r1 = Request("keep", [1, 2, 3], GenParams(temperature=0.0, max_tokens=1000, eos_token_id=999))
        r2 = Request("drop", [4, 5], GenParams(temperature=0.0, max_tokens=5, eos_token_id=999))
        sched.add_request(r1)
        sched.abort("keep")  # abort before it's even been admitted
        sched.add_request(r2)

        steps = 0
        while r2 not in sched.running.values() and not r2.is_finished and steps < 20:
            sched.step()
            steps += 1
        # process_aborts runs at the top of every step()
        sched.step()

        self.assertTrue(r1.is_finished)
        self.assertEqual(r1.finish_reason, "abort")

    def test_abort_running_request(self):
        sched = self._sched()
        r1 = Request("a", [1, 2, 3], GenParams(temperature=0.0, max_tokens=50, eos_token_id=999))
        sched.add_request(r1)
        sched.step()  # prefill -> admits + generates first token
        self.assertTrue(r1.slot is not None)
        sched.abort("a")
        sched.step()  # processes the abort
        self.assertTrue(r1.is_finished)
        self.assertEqual(r1.finish_reason, "abort")
        self.assertEqual(sched.slots.num_free, sched.model.n_slots)

    def test_max_tokens_zero_finishes_without_generating(self):
        sched = self._sched()
        r = Request("z", [1, 2, 3], GenParams(temperature=0.0, max_tokens=0, eos_token_id=999))
        sched.add_request(r)
        steps = 0
        while sched.has_work() and steps < 20:
            sched.step()
            steps += 1
        self.assertTrue(r.is_finished)
        self.assertEqual(r.output_token_ids, [])
        self.assertEqual(r.finish_reason, "length")

    def test_stop_token_ids(self):
        sched = self._sched()
        # First sample a token deterministically with a huge max_tokens, then
        # use that exact token as a stop id for a second identical request to
        # verify stop_token_ids is honoured (deterministic since temperature=0).
        probe = Request("probe", [1, 2, 3], GenParams(temperature=0.0, max_tokens=1, eos_token_id=999))
        sched.add_request(probe)
        while not probe.is_finished:
            sched.step()
        first_tok = probe.output_token_ids[0]

        r = Request(
            "stopper", [1, 2, 3],
            GenParams(temperature=0.0, max_tokens=50, stop_token_ids=(first_tok,), eos_token_id=999),
        )
        sched.add_request(r)
        steps = 0
        while sched.has_work() and steps < 50:
            sched.step()
            steps += 1
        self.assertTrue(r.is_finished)
        self.assertEqual(r.finish_reason, "stop")
        self.assertEqual(r.output_token_ids[-1], first_tok)


# =========================================================================== #
# 5. graphed decoder (eager path on CPU)
# =========================================================================== #
class TestGraphedDecoderEager(unittest.TestCase):
    def test_graphs_disabled_on_cpu(self):
        # Explicitly force device="cpu" -- `base_rt()`'s default is DEVICE,
        # which is cuda:0 on a host with CUDA (see the DEVICE comment above),
        # so relying on the default here would silently flip this test's
        # premise from "runs on cpu" to "runs on cuda" on a GPU host.
        model, cfg = build_m0(seed=9, device=torch.device("cpu"))
        rt = base_rt(device="cpu", use_cuda_graphs=True)  # requested, but CPU can't capture
        comps = build_pieces(rt, model)
        self.assertFalse(comps.decoder.graphs_enabled)

    @unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
    def test_graphs_enabled_on_cuda(self):  # pragma: no cover - GPU host only
        # Mirror of test_graphs_disabled_on_cpu: on a CUDA host with
        # use_cuda_graphs=True and a real bucket table, graphs_enabled must
        # be True. attn_backend="auto" (-> flashinfer) since building the
        # model is enough to observe this property -- capture() itself
        # requires the graph-safe backend (see TestCudaGraphCapture) but
        # `graphs_enabled` only reflects device/config, not the backend.
        model, cfg = build_m0(seed=9)
        rt = base_rt(device="cuda:0", use_cuda_graphs=True, attn_backend="auto")
        comps = build_pieces(rt, model)
        self.assertTrue(comps.decoder.graphs_enabled)

    def test_bucket_for_rounds_up(self):
        model, cfg = build_m0(seed=10)
        rt = base_rt(max_num_seqs=8, graph_buckets=(1, 2, 4, 8))
        comps = build_pieces(rt, model)
        self.assertEqual(comps.decoder.bucket_for(1), 1)
        self.assertEqual(comps.decoder.bucket_for(3), 4)
        self.assertEqual(comps.decoder.bucket_for(8), 8)

    def test_eager_step_shapes(self):
        model, cfg = build_m0(seed=11)
        rt = base_rt()
        comps = build_pieces(rt, model)
        fm, buf, dec = comps.model, comps.buf, comps.decoder

        slot = 0
        fm.reset_slot(slot)
        fm.kv_pool.ensure_capacity(slot, 2)
        fm.kv_pool.seq_len[slot] = 1
        bucket = dec.bucket_for(1)
        pad = bucket - 1
        slots = [slot] + [fm.scratch_slot] * pad
        buf.host["input_ids"][:bucket] = torch.zeros(bucket, dtype=torch.int32)
        buf.host["positions"][:bucket] = torch.tensor([1] + [0] * pad, dtype=torch.int32)
        buf.host["slot_ids"][:bucket] = torch.tensor(slots, dtype=torch.int32)
        buf.host["temperature"][:bucket] = torch.ones(bucket)
        buf.host["top_p"][:bucket] = torch.ones(bucket)
        buf.host["top_k"][:bucket] = torch.zeros(bucket)
        buf.upload(["input_ids", "positions", "slot_ids", "temperature", "top_p", "top_k"])

        out = dec.step(1, slots)
        self.assertEqual(tuple(out.shape), (bucket,))


# =========================================================================== #
# 6. QwenFastEngine (asyncio bridge)
# =========================================================================== #
class TestQwenFastEngine(unittest.IsolatedAsyncioTestCase):
    async def _engine(self, **rt_overrides) -> QwenFastEngine:
        model, cfg = build_m0(seed=13)
        rt = base_rt(**rt_overrides)
        comps = build_pieces(rt, model)
        engine = QwenFastEngine(comps.model, comps.decoder, comps.rt, eos_token_id=999, capture_graphs=False)
        await engine.start()
        self.addAsyncCleanup(engine.shutdown)
        return engine

    async def test_single_request(self):
        engine = await self._engine()
        params = SamplingParams(temperature=0.0, max_tokens=5)
        tokens = []
        finished_flags = []
        async for step in engine.add_request("r1", [1, 2, 3, 4], params):
            tokens.extend(step.new_token_ids)
            finished_flags.append(step.finished)
        self.assertEqual(len(tokens), 5)
        self.assertEqual(finished_flags[-1], True)

    async def test_concurrent_requests(self):
        engine = await self._engine()

        async def run(rid, prompt):
            out = []
            async for step in engine.add_request(rid, prompt, SamplingParams(temperature=0.0, max_tokens=4)):
                out.extend(step.new_token_ids)
            return out

        r1, r2 = await asyncio.gather(run("a", [5, 6, 7]), run("b", [8, 9]))
        self.assertEqual(len(r1), 4)
        self.assertEqual(len(r2), 4)

        stats = engine.get_stats()
        self.assertEqual(stats.num_requests_running, 0)
        self.assertEqual(stats.num_requests_waiting, 0)
        self.assertEqual(stats.generation_tokens_total, 8)

    async def test_abort_mid_stream(self):
        engine = await self._engine()
        gen = engine.add_request("c", [1, 2, 3], SamplingParams(temperature=0.0, max_tokens=20))
        got = []
        async for step in gen:
            got.append(step)
            if len(got) == 2:
                await engine.abort("c")
        self.assertTrue(got[-1].finished)
        self.assertEqual(got[-1].finish_reason, "abort")
        self.assertLess(len(got), 20)


# =========================================================================== #
# 7. GPU-only (skipped on CPU-only hosts)
# =========================================================================== #
class TestTritonNorms(unittest.TestCase):
    """`--norm-backend triton` must be numerically indistinguishable from the
    torch path, at every shape the model actually uses.

    These kernels exist purely to cut kernel *launches* (~1,600 of a decode
    step's 3,922 launches were RMSNorm/RMSNormGated), so the only thing that
    could make them a bad trade is a numerics regression -- and RMSNormGated
    is exactly where the easy bug lives ("plain `weight`, NOT
    `1 + weight`"). Shapes: `[B, 5120]` for the two residual
    norms (hidden size) and `[B*48, 128]` for the gated GDN output norm.

    Tolerance: these ops *output* bf16, whose ULP at magnitude ~1 is
    2**-7 = 0.0078, so a last-bit disagreement between two mathematically
    equivalent orderings measures 0.0078 -- far above any "tight" rtol.
    Asserting rtol=2e-3 therefore fails on *correct* code (observed: 2
    elements of 655,360, at exactly 0.0078). So each check is a pair: an
    elementwise bound of one bf16 ULP, **and** a mean-absolute-error bound
    ~80x tighter -- which a real numerics bug (e.g. the numerics-contract
    `1 + weight` vs plain `weight` trap) blows through immediately, while
    last-bit rounding cannot.
    """

    BF16_ULP = 2.0 ** -7

    def assert_norm_close(self, got, ref, *, mae_max=1e-4):
        got_f, ref_f = got.float(), ref.float()
        torch.testing.assert_close(got_f, ref_f, rtol=self.BF16_ULP, atol=self.BF16_ULP)
        mae = float((got_f - ref_f).abs().mean())
        self.assertLess(mae, mae_max, f"mean abs error {mae:.3e} exceeds {mae_max:.3e}")

    @unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
    def test_triton_rms_norm_matches_torch(self):  # pragma: no cover - GPU only
        from qwenfast.runtime.fused_model import (
            HAS_TRITON, norm_weight_1p, rms_norm_w1p, triton_rms_norm,
        )

        if not HAS_TRITON:
            self.skipTest("triton not importable")
        torch.manual_seed(0)
        for rows, cols, dt in ((1, 5120, torch.bfloat16), (32, 5120, torch.bfloat16),
                               (48, 128, torch.bfloat16), (7, 5120, torch.float32)):
            with self.subTest(rows=rows, cols=cols, dtype=dt):
                x = torch.randn(rows, cols, device="cuda:0", dtype=dt)
                w = torch.randn(cols, device="cuda:0", dtype=dt) * 0.1
                w1p = norm_weight_1p(w)
                ref = rms_norm_w1p(x, w1p, 1e-6)
                got = triton_rms_norm(x, w1p, 1e-6)
                self.assert_norm_close(got, ref)

    @unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
    def test_triton_add_rms_norm_matches_torch(self):  # pragma: no cover - GPU only
        from qwenfast.runtime.fused_model import (
            HAS_TRITON, add_rms_norm_w1p, norm_weight_1p, triton_add_rms_norm,
        )

        if not HAS_TRITON:
            self.skipTest("triton not importable")
        torch.manual_seed(1)
        for rows, cols in ((1, 5120), (32, 5120), (128, 5120)):
            with self.subTest(rows=rows, cols=cols):
                x = torch.randn(rows, cols, device="cuda:0", dtype=torch.bfloat16)
                r = torch.randn(rows, cols, device="cuda:0", dtype=torch.bfloat16)
                w1p = norm_weight_1p(torch.randn(cols, device="cuda:0", dtype=torch.bfloat16) * 0.1)
                ref_res, ref_out = add_rms_norm_w1p(r, x, w1p, 1e-6)
                got_res, got_out = triton_add_rms_norm(r, x, w1p, 1e-6)
                self.assert_norm_close(got_res, ref_res)
                self.assert_norm_close(got_out, ref_out)

    @unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
    def test_triton_rms_norm_gated_matches_torch(self):  # pragma: no cover - GPU only
        from qwenfast.runtime.fused_model import (
            HAS_TRITON, rms_norm_gated, triton_rms_norm_gated,
        )

        if not HAS_TRITON:
            self.skipTest("triton not importable")
        torch.manual_seed(2)
        for rows, cols in ((48, 128), (32 * 48, 128), (128 * 48, 128)):
            with self.subTest(rows=rows, cols=cols):
                x = torch.randn(rows, cols, device="cuda:0", dtype=torch.bfloat16)
                gate = torch.randn(rows, cols, device="cuda:0", dtype=torch.bfloat16)
                w = torch.randn(cols, device="cuda:0", dtype=torch.bfloat16) * 0.1
                ref = rms_norm_gated(x, gate, w, 1e-6)
                got = triton_rms_norm_gated(x, gate, w, 1e-6)
                self.assert_norm_close(got, ref, mae_max=3e-4)

    @unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
    def test_triton_norm_backend_matches_torch_end_to_end(self):  # pragma: no cover - GPU only
        """The whole prefill forward, torch norms vs triton norms."""
        from qwenfast.runtime.fused_model import HAS_TRITON

        if not HAS_TRITON:
            self.skipTest("triton not importable")
        model, cfg = build_m0(seed=9, dtype=torch.bfloat16, device=torch.device("cuda:0"),
                              head_dim=64)
        ids = [3, 5, 7, 9, 11, 2, 4]
        outs = []
        for backend in ("torch", "triton"):
            rt = base_rt(device="cuda:0", dtype="bf16", attn_backend="torch",
                         norm_backend=backend)
            fm = FusedQwenForCausalLM.from_m0_module(model, rt)
            fm.reset_slot(0)
            fm.kv_pool.ensure_capacity(0, len(ids))
            batch = make_prefill_batch([ids], [0], [0], torch.device("cuda:0"))
            with torch.no_grad():
                outs.append(fm.prefill_forward(batch, all_logits=True).float())
        torch.testing.assert_close(outs[1], outs[0], rtol=2e-2, atol=2e-2)
        self.assertTrue(torch.equal(outs[0].argmax(-1), outs[1].argmax(-1)))


class TestFusedOps(unittest.TestCase):
    """Launch-reduction kernels: ``fused_ops/swiglu.py``
    and ``fused_ops/gdn_gate.py`` vs their eager references, at every shape
    the model actually uses, plus one end-to-end check
    (``fused_ops_backend="triton"`` vs ``"torch"``) mirroring
    ``TestTritonNorms.test_triton_norm_backend_matches_torch_end_to_end``.

    Same bf16-ULP + MAE tolerance convention as the Triton norms above --
    these are also bf16-output kernels compared against a mathematically
    equivalent but differently-ordered eager sequence. One difference: the
    norm kernels' MAE bound (1e-4) leans on RMSNorm's mean-reduction, which
    makes independent per-element rounding differences partially cancel in
    the *average*. ``swiglu``/the GDN gate have no such reduction -- every
    output element's rounding error is independent -- so their measured MAE
    sits closer to a fraction of one full ULP (observed up to ~4.7e-4 on
    the H200, still comfortably inside a half-ULP=0.0039 budget) rather than
    the norm kernels' ~1e-4; ``mae_max=6e-4`` below is that looser, but
    still far-from-a-full-ULP, bound.
    """

    BF16_ULP = 2.0 ** -7

    def assert_close_bf16(self, got, ref, *, mae_max=1e-4):
        got_f, ref_f = got.float(), ref.float()
        torch.testing.assert_close(got_f, ref_f, rtol=self.BF16_ULP, atol=self.BF16_ULP)
        mae = float((got_f - ref_f).abs().mean())
        self.assertLess(mae, mae_max, f"mean abs error {mae:.3e} exceeds {mae_max:.3e}")

    @unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
    def test_triton_swiglu_matches_reference(self):  # pragma: no cover - GPU only
        from qwenfast.runtime.fused_ops.swiglu import HAS_TRITON, swiglu_reference, triton_swiglu

        if not HAS_TRITON:
            self.skipTest("triton not importable")
        torch.manual_seed(0)
        # 128: tiny_config's intermediate_size. 17408: the real model's
        # -- both must round-trip through the 2D tile grid.
        for rows, inter, dt in ((1, 128, torch.bfloat16), (32, 128, torch.bfloat16),
                                 (7, 17408, torch.bfloat16), (1, 17408, torch.float32)):
            with self.subTest(rows=rows, inter=inter, dtype=dt):
                gate_up = torch.randn(rows, 2 * inter, device="cuda:0", dtype=dt)
                ref = swiglu_reference(gate_up)
                got = triton_swiglu(gate_up)
                self.assert_close_bf16(got, ref, mae_max=6e-4)

    @unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
    def test_triton_gdn_gate_matches_reference(self):  # pragma: no cover - GPU only
        from qwenfast.runtime.fused_ops.gdn_gate import HAS_TRITON, gdn_gate_reference, triton_gdn_gate

        if not HAS_TRITON:
            self.skipTest("triton not importable")
        torch.manual_seed(1)
        # 4: tiny_config's linear_num_value_heads. 48: the real model's.
        for rows, h in ((1, 4), (32, 4), (7, 48), (128, 48)):
            with self.subTest(rows=rows, h=h):
                ba = torch.randn(rows, 2 * h, device="cuda:0", dtype=torch.bfloat16)
                neg_exp_A = -torch.rand(h, device="cuda:0", dtype=torch.float32) * 3.0
                dt_bias_f = torch.randn(h, device="cuda:0", dtype=torch.float32) * 0.1
                ref_beta, ref_g = gdn_gate_reference(ba, neg_exp_A, dt_bias_f, h)
                got_beta, got_g = triton_gdn_gate(ba, neg_exp_A, dt_bias_f)
                self.assert_close_bf16(got_beta, ref_beta)
                # g is fp32 end to end on both paths -- a tight direct bound,
                # not the bf16-rounding bound above.
                torch.testing.assert_close(got_g.float(), ref_g.float(), rtol=1e-5, atol=1e-5)

    @unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
    def test_fused_ops_backend_matches_torch_end_to_end(self):  # pragma: no cover - GPU only
        """The whole prefill forward, torch fused-ops vs triton fused-ops --
        exercises both kernels together through FusedMLP and FusedGDN."""
        from qwenfast.runtime.fused_ops.swiglu import HAS_TRITON

        if not HAS_TRITON:
            self.skipTest("triton not importable")
        model, cfg = build_m0(seed=25, dtype=torch.bfloat16, device=torch.device("cuda:0"),
                              head_dim=64)
        ids = [4, 8, 12, 16, 3, 9]
        outs = []
        for backend in ("torch", "triton"):
            rt = base_rt(device="cuda:0", dtype="bf16", attn_backend="torch",
                         fused_ops_backend=backend)
            fm = FusedQwenForCausalLM.from_m0_module(model, rt)
            fm.reset_slot(0)
            fm.kv_pool.ensure_capacity(0, len(ids))
            batch = make_prefill_batch([ids], [0], [0], torch.device("cuda:0"))
            with torch.no_grad():
                outs.append(fm.prefill_forward(batch, all_logits=True).float())
        torch.testing.assert_close(outs[1], outs[0], rtol=2e-2, atol=2e-2)
        self.assertTrue(torch.equal(outs[0].argmax(-1), outs[1].argmax(-1)))


class TestFP8KVCache(unittest.TestCase):
    """``RuntimeConfig(kv_cache_dtype="fp8")`` end to end.

    ``attn_backend="torch"`` throughout -- the CPU-safe fallback path
    (``AttentionRunner.decode``/``.prefill`` -> ``torch_fallback_decode`` /
    ``_torch_paged_prefill``, both reading through ``PagedKVPool
    .gather_dense``, which dequantizes fp8 identically on CPU and CUDA) --
    so these gate on every commit, no GPU required. The CUDA-graph-safety
    check is GPU-only (:class:`TestFP8KVCacheGraphSafety` below).
    """

    def test_calibrate_kv_scales_is_uniform_and_noop_on_bf16(self):
        model, cfg = build_m0(seed=20)
        rt_bf16 = base_rt(dtype="fp32", kv_cache_dtype="bf16")
        fm_bf16 = FusedQwenForCausalLM.from_m0_module(model, rt_bf16)
        self.assertEqual(fm_bf16.calibrate_kv_scales([[1, 2, 3]]), {})

        rt = base_rt(dtype="fp32", kv_cache_dtype="fp8")
        fm = FusedQwenForCausalLM.from_m0_module(model, rt)
        prompts = [[1, 2, 3, 4, 5], [6, 7, 8]]
        scales = fm.calibrate_kv_scales(prompts)
        # tiny_config() has exactly one "full_attention" layer -> one kv_layer
        self.assertEqual(len(scales), 1)
        for layer, (k_scale, v_scale) in scales.items():
            self.assertGreater(k_scale, 0.0)
            self.assertGreater(v_scale, 0.0)
            # uniform: every page/head of that layer's K (and V) shares the
            # one calibrated scale (calibrate_uniform_scale's contract)
            self.assertTrue(bool((fm.kv_pool.scale[layer, :, 0, :] == k_scale).all()))
            self.assertTrue(bool((fm.kv_pool.scale[layer, :, 1, :] == v_scale).all()))
            self.assertEqual(fm.attn.kv_scales[layer], (k_scale, v_scale))

        # calibration must not leak: every slot it borrowed is fully reset
        for slot in range(len(prompts)):
            self.assertEqual(int(fm.kv_pool.seq_len[slot]), 0)
            self.assertEqual(fm.kv_pool.pages_allocated(slot), 0)
            self.assertTrue(bool((fm.state_pool[slot] == 0).all()))

    def test_calibrate_kv_scales_sequential_chunking_matches_single_batch(self):
        """More calibration prompts than slots must not
        raise -- it must chunk sequentially and land on the same scale a
        single wide batch would have produced (this is what
        ``profile_step``/``bench_runtime`` hit with ``max_num_seqs=1``:
        "4 calibration prompts > 1 available slots")."""
        model, cfg = build_m0(seed=22)
        prompts = [[1, 2, 3, 4, 5], [6, 7, 8], [9, 10, 11, 12], [13, 14]]

        rt_wide = base_rt(dtype="fp32", kv_cache_dtype="fp8", max_num_seqs=8)
        fm_wide = FusedQwenForCausalLM.from_m0_module(model, rt_wide)
        scales_wide = fm_wide.calibrate_kv_scales(prompts)
        self.assertEqual(len(scales_wide), 1)

        # max_num_seqs=1: a single slot, fewer slots than prompts -- must
        # calibrate in 4 sequential one-prompt chunks instead of raising.
        rt_narrow = base_rt(dtype="fp32", kv_cache_dtype="fp8", max_num_seqs=1)
        fm_narrow = FusedQwenForCausalLM.from_m0_module(model, rt_narrow)
        scales_narrow = fm_narrow.calibrate_kv_scales(prompts)

        # Relative, not exact: the wide path packs all four prompts into one
        # varlen prefill (T=14) and the narrow path runs four (T=5,3,4,2), so
        # every linear reduces over a different GEMM tiling. The claim under
        # test is "chunking lands on the same scale", not "fp32 reassociates
        # identically" -- exact equality here failed on torch 2.13/CPU at a
        # relative 3e-7 while the calibration itself
        # was correct. 1e-5 is still four orders tighter than the 10% headroom
        # `calibrate_kv_scales` applies on top of the amax.
        self.assertEqual(scales_wide.keys(), scales_narrow.keys())
        for layer, (kw, vw) in scales_wide.items():
            kn, vn = scales_narrow[layer]
            self.assertAlmostEqual(kw, kn, delta=abs(kw) * 1e-5)
            self.assertAlmostEqual(vw, vn, delta=abs(vw) * 1e-5)

        # no residue on the one slot it borrowed and reused 4 times
        self.assertEqual(int(fm_narrow.kv_pool.seq_len[0]), 0)
        self.assertEqual(fm_narrow.kv_pool.pages_allocated(0), 0)

    def test_calibrate_kv_scales_explicit_slots_still_bounded_by_n_slots(self):
        model, cfg = build_m0(seed=23)
        rt = base_rt(dtype="fp32", kv_cache_dtype="fp8", max_num_seqs=2)
        fm = FusedQwenForCausalLM.from_m0_module(model, rt)
        with self.assertRaises(ValueError):
            fm.calibrate_kv_scales([[1, 2], [3, 4], [5, 6]], slots=[0, 1, 2])

    def test_fp8_kv_decode_matches_bf16_within_tolerance(self):
        """The literal task requirement: fp8-KV decode logits within
        tolerance of bf16-KV, on the tiny model."""
        model, cfg = build_m0(seed=21, dtype=torch.bfloat16)
        ids = [2, 5, 9, 12, 4, 7]
        calib_prompts = [[3, 6, 10, 1, 4], [8, 2, 5, 9]]

        def run(kv_dtype):
            rt = base_rt(dtype="bf16", kv_cache_dtype=kv_dtype)
            fm = FusedQwenForCausalLM.from_m0_module(model, rt)
            if kv_dtype == "fp8":
                fm.calibrate_kv_scales(calib_prompts)
            slot = 0
            fm.reset_slot(slot)
            fm.kv_pool.ensure_capacity(slot, len(ids) + 1)
            batch = make_prefill_batch([ids], [0], [slot], DEVICE)
            with torch.no_grad():
                prefill_logits = fm.prefill_forward(batch, all_logits=True)
            next_tok = int(prefill_logits[-1].argmax())

            buf = DeviceBuffers(max_batch=1, vocab_size=cfg.vocab_size,
                                 max_pages=fm.kv_pool.cfg.n_pages, device=DEVICE)
            buf.host["input_ids"][:1] = torch.tensor([next_tok], dtype=torch.int32)
            buf.host["positions"][:1] = torch.tensor([len(ids)], dtype=torch.int32)
            buf.host["slot_ids"][:1] = torch.tensor([slot], dtype=torch.int32)
            buf.upload(["input_ids", "positions", "slot_ids"])
            fm.attn.plan_decode([slot], 1)
            with torch.no_grad():
                decode_logits = fm.decode_forward(buf, 1)
            return decode_logits[0].float().cpu()

        bf16_logits = run("bf16")
        fp8_logits = run("fp8")
        diff = (bf16_logits - fp8_logits).abs()
        # e4m3 KV quantization noise, not a numerics bug -- the same shape of
        # tolerance accepted for FP8-dequant *weights* (max Δlogit
        # 1.75, argmax identical) on the real 27B/248k-vocab model; this is
        # the much smaller tiny model/vocab, so the bound is tighter.
        self.assertLess(float(diff.max()), 1.0, f"max abs diff {float(diff.max())}")
        self.assertEqual(int(bf16_logits.argmax()), int(fp8_logits.argmax()))

    def test_fp8_kv_prefill_and_decode_append_do_not_crash_uncalibrated(self):
        """Uncalibrated fp8 KV (scale defaults to 1.0) must still run
        prefill-append -> decode-append -> plan/run end to end without
        crashing -- the dtype path itself, independent of calibration
        accuracy (which the previous test covers)."""
        model, cfg = build_m0(seed=22)
        rt = base_rt(dtype="fp32", kv_cache_dtype="fp8")
        fm = FusedQwenForCausalLM.from_m0_module(model, rt)
        ids = [1, 3, 5, 7]
        slot = 0
        fm.reset_slot(slot)
        fm.kv_pool.ensure_capacity(slot, len(ids) + 2)
        batch = make_prefill_batch([ids], [0], [slot], DEVICE)
        with torch.no_grad():
            logits = fm.prefill_forward(batch, all_logits=True)
        self.assertEqual(tuple(logits.shape), (len(ids), cfg.vocab_size))

        next_tok = int(logits[-1].argmax())
        buf = DeviceBuffers(max_batch=1, vocab_size=cfg.vocab_size,
                             max_pages=fm.kv_pool.cfg.n_pages, device=DEVICE)
        buf.host["input_ids"][:1] = torch.tensor([next_tok], dtype=torch.int32)
        buf.host["positions"][:1] = torch.tensor([len(ids)], dtype=torch.int32)
        buf.host["slot_ids"][:1] = torch.tensor([slot], dtype=torch.int32)
        buf.upload(["input_ids", "positions", "slot_ids"])
        fm.attn.plan_decode([slot], 1)
        with torch.no_grad():
            decode_logits = fm.decode_forward(buf, 1)
        self.assertEqual(tuple(decode_logits.shape), (1, cfg.vocab_size))


class TestFP8KVCacheGraphSafety(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
    def test_fp8_kv_graph_replay_matches_eager(self):  # pragma: no cover - GPU host only
        """CUDA-graph-safe fp8 KV: calibrate (before capture) -> capture ->
        replay must match an identical uncaptured (eager) run, mirroring
        ``TestCudaGraphCapture.test_graph_replay_matches_eager`` but with
        ``kv_cache_dtype='fp8'``. Calibration must run before capture (see
        ``calibrate_kv_scales``'s docstring): the k_scale/v_scale floats get
        baked into the captured kernel launch, not read from a device
        tensor at replay time."""
        model, cfg = build_m0(seed=24, dtype=torch.bfloat16, device=torch.device("cuda:0"),
                              head_dim=64)
        calib_prompts = [[3, 6, 10, 1, 4, 2], [8, 2, 5, 9, 1, 7]]

        rt = base_rt(device="cuda:0", dtype="bf16", use_cuda_graphs=True,
                     attn_backend="auto", kv_cache_dtype="fp8")
        comps = build_pieces(rt, model)
        fm, buf, dec = comps.model, comps.buf, comps.decoder
        scales = fm.calibrate_kv_scales(calib_prompts)
        self.assertTrue(scales)
        dec.warmup()
        dec.capture()

        slot = 0
        fm.reset_slot(slot)
        fm.kv_pool.ensure_capacity(slot, 2)
        bucket = dec.bucket_for(1)
        pad = bucket - 1
        slots = [slot] + [fm.scratch_slot] * pad
        buf.host["input_ids"][:bucket].zero_()
        buf.host["positions"][:bucket] = torch.tensor([1] + [0] * pad, dtype=torch.int32)
        buf.host["slot_ids"][:bucket] = torch.tensor(slots, dtype=torch.int32)
        buf.host["temperature"][:bucket].fill_(0.0)
        buf.upload(["input_ids", "positions", "slot_ids", "temperature"])

        out_graph = dec.step(1, slots).clone()

        rt2 = base_rt(device="cuda:0", dtype="bf16", use_cuda_graphs=False,
                      attn_backend="auto", kv_cache_dtype="fp8")
        comps2 = build_pieces(rt2, model)
        fm2, buf2, dec2 = comps2.model, comps2.buf, comps2.decoder
        fm2.calibrate_kv_scales(calib_prompts)
        fm2.reset_slot(slot)
        fm2.kv_pool.ensure_capacity(slot, 2)
        buf2.host["input_ids"][:1].zero_()
        buf2.host["positions"][:1] = torch.tensor([1], dtype=torch.int32)
        buf2.host["slot_ids"][:1] = torch.tensor([slot], dtype=torch.int32)
        buf2.host["temperature"][:1].fill_(0.0)
        buf2.upload(["input_ids", "positions", "slot_ids", "temperature"])
        out_eager = dec2.step(1, [slot])

        torch.testing.assert_close(out_graph[:1].cpu(), out_eager[:1].cpu())


class TestCudaGraphCapture(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
    def test_graph_replay_matches_eager(self):  # pragma: no cover - GPU host only
        # attn_backend="auto" (-> "flashinfer" where it's importable) rather
        # than base_rt()'s usual forced "torch": capture requires the
        # graph-safe backend. The torch/eager fallback is *not*
        # graph-capturable at all (attn's torch_fallback_decode does a host
        # sync via `slot_ids.tolist()`) -- GraphedDecoder.capture() raises a
        # clear error rather than a cryptic low-level CUDA error three calls
        # deep. gdn_backend stays "torch" -- graph-safe
        # (kernels_gdn.state.gather_states/scatter_states are pure
        # index_select/index_copy_, no host sync) and keeps this test
        # independent of GDN kernel correctness.
        # bf16 model + bf16 KV: FlashInfer has no fp32 KV kernels (KeyError:
        # torch.float32 in get_batch_prefill_uri), and the fp32 test path
        # re-types the pool to fp32 (_retype_kv_pool).
        # head_dim=64, not the CPU tests' 16: FlashInfer's tensor-core decode
        # kernel rejects a 1-tile QK dimension ("Invalid configuration :
        # NUM_MMA_Q=1 NUM_MMA_D_QK=1 ..."). See tiny_config().
        model, cfg = build_m0(seed=17, dtype=torch.bfloat16, device=torch.device("cuda:0"),
                              head_dim=64)
        rt = base_rt(device="cuda:0", dtype="bf16", use_cuda_graphs=True, attn_backend="auto")
        comps = build_pieces(rt, model)
        fm, buf, dec = comps.model, comps.buf, comps.decoder
        dec.warmup()
        dec.capture()

        slot = 0
        fm.reset_slot(slot)
        fm.kv_pool.ensure_capacity(slot, 2)
        bucket = dec.bucket_for(1)
        pad = bucket - 1
        slots = [slot] + [fm.scratch_slot] * pad
        buf.host["input_ids"][:bucket].zero_()
        buf.host["positions"][:bucket] = torch.tensor([1] + [0] * pad, dtype=torch.int32)
        buf.host["slot_ids"][:bucket] = torch.tensor(slots, dtype=torch.int32)
        buf.host["temperature"][:bucket].fill_(0.0)
        buf.upload(["input_ids", "positions", "slot_ids", "temperature"])

        out_graph = dec.step(1, slots).clone()

        rt2 = base_rt(device="cuda:0", dtype="bf16", use_cuda_graphs=False, attn_backend="auto")
        comps2 = build_pieces(rt2, model)
        fm2, buf2, dec2 = comps2.model, comps2.buf, comps2.decoder
        fm2.reset_slot(slot)
        fm2.kv_pool.ensure_capacity(slot, 2)
        buf2.host["input_ids"][:1].zero_()
        buf2.host["positions"][:1] = torch.tensor([1], dtype=torch.int32)
        buf2.host["slot_ids"][:1] = torch.tensor([slot], dtype=torch.int32)
        buf2.host["temperature"][:1].fill_(0.0)
        buf2.upload(["input_ids", "positions", "slot_ids", "temperature"])
        out_eager = dec2.step(1, [slot])

        torch.testing.assert_close(out_graph[:1].cpu(), out_eager[:1].cpu())


if __name__ == "__main__":
    unittest.main(verbosity=2)
