"""Tests for ``attn.fused_qk_rope``.

CPU tests (always run, no GPU needed): the dispatcher (``qk_norm_rope``)
falls back to the exact eager reference whenever the Triton path cannot run
-- including the case where Triton imports but the tensors are on CPU
("Triton importable" is not "tensors on CUDA") -- and ``norm_weight_1p``
matches the inline ``(1 + weight)`` convention every other norm site in this
tree uses.

GPU tests (skipped automatically when CUDA or Triton aren't available): the
Triton kernel itself, at the real prefill shape (``head_dim=256``,
``rotary_dim=64``), against ``flashinfer_attn.fused_qk_norm_rope``, the
eager reference it replaces on the prefill path.

Run::

    python engine/qwenfast/attn/tests/test_fused_qk_rope.py
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))  # -> engine/

from qwenfast.attn import fused_qk_rope as qkrope  # noqa: E402
from qwenfast.attn.flashinfer_attn import fused_qk_norm_rope as eager_ref  # noqa: E402
from qwenfast.attn.rope import RotaryTable  # noqa: E402

HAS_CUDA = torch.cuda.is_available()

# production shape (`weights.QwenFastConfig`): 24 q-heads, 4 kv
# heads, head_dim 256, rotary_dim 64.
HEAD_DIM = 256
ROTARY_DIM = 64
NUM_Q_HEADS = 24
NUM_KV_HEADS = 4


def _random_inputs(t: int, seed: int = 0, dtype=torch.float32, device="cpu"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(t, NUM_Q_HEADS, HEAD_DIM, generator=g).to(dtype=dtype, device=device)
    k = torch.randn(t, NUM_KV_HEADS, HEAD_DIM, generator=g).to(dtype=dtype, device=device)
    q_w = torch.randn(HEAD_DIM, generator=g).to(dtype=torch.float32, device=device) * 0.1
    k_w = torch.randn(HEAD_DIM, generator=g).to(dtype=torch.float32, device=device) * 0.1
    # cos/sin dtype must match q/k's dtype, exactly as production builds it
    # (`FusedModel.__init__`: `self.rotary = RotaryTable(..., dtype=self.dtype,
    # ...)` where `self.dtype` is the model's runtime compute dtype -- the
    # same dtype `q`/`k` come out of `qkv_proj` in). Hardcoding this table to
    # fp32 regardless of `dtype` (the pre-fix version of this helper) silently
    # changes the eager reference's *output* dtype too: `apply_rotary_pos_emb`
    # multiplies a bf16 `q_rot` by an fp32 `cos`, and torch's type promotion
    # upconverts the whole rotated half to fp32 -- a mismatch against the
    # Triton kernel (which always returns `q`/`k`'s own input dtype, matching
    # what `FlashInferPrefillAttention`/`FlashInferDecodeAttention` require via
    # their `q_dtype` default of bf16) that is an artifact of this test's own
    # table construction, not a real dtype-contract bug in either kernel.
    table = RotaryTable(max_positions=max(t, 8), rotary_dim=ROTARY_DIM, theta=1e7, dtype=dtype, device=device)
    positions = torch.arange(t, device=device)
    cos, sin = table.lookup(positions)
    return q, k, q_w, k_w, cos, sin


# =========================================================================== #
# CPU: the dispatcher's fallback / safety logic
# =========================================================================== #
class TestDispatcherCPU(unittest.TestCase):
    def test_use_fused_false_matches_eager_reference_bit_identical(self):
        q, k, q_w, k_w, cos, sin = _random_inputs(9, dtype=torch.float32)
        got_q, got_k = qkrope.qk_norm_rope(q, k, q_w, k_w, cos, sin, eps=1e-6, use_fused=False)
        want_q, want_k = eager_ref(q, k, q_w, k_w, cos, sin, eps=1e-6)
        torch.testing.assert_close(got_q, want_q, rtol=0, atol=0)
        torch.testing.assert_close(got_k, want_k, rtol=0, atol=0)

    def test_use_fused_true_on_cpu_tensors_still_falls_back_to_eager(self):
        """'Triton imports' is
        not 'the tensors are on CUDA'. `use_fused=True` on CPU tensors must
        never reach the Triton kernel (it would hand it CPU pointers), and
        must produce exactly the eager answer -- not silently do nothing."""
        q, k, q_w, k_w, cos, sin = _random_inputs(6, dtype=torch.float32)
        got_q, got_k = qkrope.qk_norm_rope(q, k, q_w, k_w, cos, sin, eps=1e-6, use_fused=True)
        want_q, want_k = eager_ref(q, k, q_w, k_w, cos, sin, eps=1e-6)
        torch.testing.assert_close(got_q, want_q, rtol=0, atol=0)
        torch.testing.assert_close(got_k, want_k, rtol=0, atol=0)

    def test_norm_weight_1p_matches_inline_convention(self):
        w = torch.randn(HEAD_DIM) * 0.3
        got = qkrope.norm_weight_1p(w)
        want = 1.0 + w.float()
        torch.testing.assert_close(got, want, rtol=0, atol=0)
        self.assertEqual(got.dtype, torch.float32)

    def test_dispatcher_default_is_not_fused(self):
        """The default (`use_fused` unset) must be the old behaviour byte for
        byte -- this is what makes the change at the `decode`/`window` call
        sites in `FusedAttention._qkv` a no-op."""
        q, k, q_w, k_w, cos, sin = _random_inputs(4, dtype=torch.float32)
        got_q, got_k = qkrope.qk_norm_rope(q, k, q_w, k_w, cos, sin, eps=1e-6)
        want_q, want_k = eager_ref(q, k, q_w, k_w, cos, sin, eps=1e-6)
        torch.testing.assert_close(got_q, want_q, rtol=0, atol=0)
        torch.testing.assert_close(got_k, want_k, rtol=0, atol=0)


# =========================================================================== #
# GPU: the Triton kernel itself
# =========================================================================== #
@unittest.skipUnless(HAS_CUDA and qkrope.HAS_TRITON, "requires CUDA + triton")
class TestTritonKernelGPU(unittest.TestCase):
    def _check(self, t: int, seed: int = 0):
        device = "cuda"
        q, k, q_w, k_w, cos, sin = _random_inputs(t, seed=seed, dtype=torch.bfloat16, device=device)
        got_q, got_k = qkrope.fused_qk_norm_rope_triton(q, k, q_w, k_w, cos, sin, eps=1e-6)
        want_q, want_k = eager_ref(q, k, q_w, k_w, cos, sin, eps=1e-6)
        # bf16 eps ~ 0.0078; rtol/atol generous enough to absorb the reduction
        # order difference between the kernel's fp32 accumulate and torch's,
        # tight enough to catch a wrong rotate-half or a wrong weight lane.
        torch.testing.assert_close(got_q, want_q, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(got_k, want_k, rtol=2e-2, atol=2e-2)

    def test_matches_eager_reference_prefill_shape(self):
        self._check(t=37)

    def test_matches_eager_reference_single_token(self):
        self._check(t=1)

    def test_matches_eager_reference_large_chunk(self):
        self._check(t=8192, seed=11)

    def test_matches_eager_reference_on_noncontiguous_qkv_split(self):
        """Production `q`/`k` are non-contiguous views sliced out of the
        fused `qkv_proj` output (`FusedAttention._qkv`'s `fused.split(...)`),
        not freshly allocated contiguous tensors. The kernel takes explicit
        strides for exactly this reason; this test exercises that path."""
        device = "cuda"
        t = 17
        q_size = NUM_Q_HEADS * HEAD_DIM
        kv_size = NUM_KV_HEADS * HEAD_DIM
        fused = torch.randn(t, q_size + 2 * kv_size, dtype=torch.bfloat16, device=device)
        qg, k, _v = fused.split([q_size, kv_size, kv_size], dim=-1)
        q = qg.view(t, NUM_Q_HEADS, HEAD_DIM)
        k = k.view(t, NUM_KV_HEADS, HEAD_DIM)
        self.assertFalse(k.is_contiguous())  # the shape under test
        q_w = (torch.randn(HEAD_DIM, device=device) * 0.1).float()
        k_w = (torch.randn(HEAD_DIM, device=device) * 0.1).float()
        # cos/sin dtype matches q/k's bf16 dtype -- see `_random_inputs`'s
        # comment above for why this must not be fp32.
        table = RotaryTable(max_positions=32, rotary_dim=ROTARY_DIM, theta=1e7, dtype=torch.bfloat16, device=device)
        positions = torch.arange(t, device=device)
        cos, sin = table.lookup(positions)

        got_q, got_k = qkrope.fused_qk_norm_rope_triton(q, k, q_w, k_w, cos, sin, eps=1e-6)
        want_q, want_k = eager_ref(q, k, q_w, k_w, cos, sin, eps=1e-6)
        torch.testing.assert_close(got_q, want_q, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(got_k, want_k, rtol=2e-2, atol=2e-2)

    def test_dispatcher_use_fused_true_on_cuda_routes_to_triton(self):
        device = "cuda"
        q, k, q_w, k_w, cos, sin = _random_inputs(5, dtype=torch.bfloat16, device=device)
        calls = []
        real = qkrope.fused_qk_norm_rope_triton
        qkrope.fused_qk_norm_rope_triton = lambda *a, **kw: (calls.append(1), real(*a, **kw))[1]
        try:
            qkrope.qk_norm_rope(q, k, q_w, k_w, cos, sin, eps=1e-6, use_fused=True)
        finally:
            qkrope.fused_qk_norm_rope_triton = real
        self.assertEqual(len(calls), 1)

    def test_dispatcher_use_fused_false_on_cuda_does_not_route_to_triton(self):
        device = "cuda"
        q, k, q_w, k_w, cos, sin = _random_inputs(5, dtype=torch.bfloat16, device=device)
        calls = []
        real = qkrope.fused_qk_norm_rope_triton
        qkrope.fused_qk_norm_rope_triton = lambda *a, **kw: (calls.append(1), real(*a, **kw))[1]
        try:
            qkrope.qk_norm_rope(q, k, q_w, k_w, cos, sin, eps=1e-6, use_fused=False)
        finally:
            qkrope.fused_qk_norm_rope_triton = real
        self.assertEqual(len(calls), 0)


if __name__ == "__main__":
    unittest.main()
