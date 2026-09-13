"""Tests for the GDN kernels (``qwenfast.kernels_gdn``).

Two tiers:

* **CPU** (always run) — the ``torch`` backend against the reference oracle
  (``qwenfast.model.torch_{chunk,recurrent}_gated_delta_rule`` and
  ``causal_conv1d``), plus every layout/semantics property that does not need a
  GPU: slot gather, varlen prefill, state threading, GVA equivalence, and
  verify-and-commit equivalence with sequential decode for **every** accept
  length ``m in 0..k``.
* **GPU** (skipped without CUDA) — the ``fla`` and ``triton`` backends against
  the same oracle at 2e-4 for an fp32 state pool, plus the documented fp16/bf16
  state tolerances.

Run::

    python -m unittest discover -s engine/qwenfast/kernels_gdn/tests -v
    python engine/qwenfast/kernels_gdn/tests/test_kernels_gdn.py

Tolerances
----------
=====================================  =======  ==============================
comparison                             atol     rationale
=====================================  =======  ==============================
any backend vs oracle, **fp32** state  2e-4     the API contract
triton/fla vs torch, **fp16** state    2e-3     fp16 has 11 mantissa bits; the
                                                state round-trips through it
                                                once per step, so the error
                                                floor is ~2^-11 * |S| ~ 5e-4
                                                and 4x that is comfortable
triton/fla vs torch, **bf16** state    2e-2     8 mantissa bits => ~2^-8 * |S|
fp16 pool vs fp32 pool (informational) --       reported, not asserted; this is
                                                the number the eval gate for
                                                half-precision state decides on
=====================================  =======  ==============================
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))  # -> engine/

from qwenfast.kernels_gdn import api, shapes, state as st  # noqa: E402
from qwenfast.kernels_gdn import fla_ops, torch_ops, triton_ops  # noqa: E402
from qwenfast.model import (  # noqa: E402  (the reference oracle)
    causal_conv1d as oracle_conv1d,
    torch_chunk_gated_delta_rule as oracle_chunk,
    torch_recurrent_gated_delta_rule as oracle_recurrent,
)

HAS_CUDA = torch.cuda.is_available()

ATOL_FP32 = 2e-4
ATOL_FP16_STATE = 2e-3
ATOL_BF16_STATE = 2e-2

# fla's *chunk* kernel does its intra-chunk matmuls on tensor cores.  Triton's
# `tl.dot` defaults to `input_precision="tf32"` for fp32 operands, i.e. 10
# mantissa bits (unit roundoff 4.9e-4), and the chunk recurrence accumulates
# that across chunk boundaries.  Measured on the H200 at T=130 (3 chunks):
# final-state maxdiff 8.9e-4 against |state|max = 0.80, i.e. 1.1e-3 relative =
# 2.3x TF32 ULP.  The *outputs* passed at 2e-4 only because |out|max is 10x
# smaller.  So the chunk path gets a **relative** tolerance; 4e-3 is ~8x TF32
# ULP, still orders of magnitude tighter than any real bug.
#   Confirm the diagnosis on a GPU host with:  TRITON_F32_DEFAULT=ieee pytest ...
#   which should pull the diff back under ATOL_FP32.
RTOL_CHUNK_TF32 = 4e-3

HV = shapes.NUM_V_HEADS  # 48
HK = shapes.NUM_K_HEADS  # 16
DK = shapes.HEAD_K_DIM  # 128
DV = shapes.HEAD_V_DIM  # 128
REP = shapes.GVA_GROUP  # 3


# =========================================================================== #
# input construction (mirrors kernels/microbench/gdn_decode_bench.py)
# =========================================================================== #
def make_inputs(b, t, device="cpu", seed=0, dtype=torch.float32,
                hv=HV, hk=HK, dk=DK, dv=DV):
    """Post-conv GDN inputs: q/k at ``hk`` heads, v/g/beta at ``hv``."""
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def rn(*shape):
        return torch.randn(*shape, generator=gen).to(device=device, dtype=torch.float32)

    q = rn(b, t, hk, dk)
    k = rn(b, t, hk, dk)
    v = rn(b, t, hv, dv).to(dtype)
    a_log = torch.log(torch.empty(hv).uniform_(0.01, 16, generator=gen)).to(device)
    dt_bias = torch.ones(hv, device=device)
    a_raw = rn(b, t, hv)
    b_raw = rn(b, t, hv)
    beta = b_raw.sigmoid()
    g = -a_log.float().exp() * torch.nn.functional.softplus(a_raw.float() + dt_bias)
    return dict(q=q, k=k, v=v, g=g, beta=beta)


def make_state(b, device="cpu", seed=1, dtype=torch.float32, hv=HV, dk=DK, dv=DV):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    s = torch.randn(b, hv, dk, dv, generator=gen) * 0.02
    return s.to(device=device, dtype=dtype)


def expand(x):
    """q/k at 16 heads -> 48, the form the reference oracle wants."""
    return x.repeat_interleave(REP, dim=2)


def maxdiff(a, b):
    return (a.float() - b.float()).abs().max().item()


def scaled_tol(ref, atol=ATOL_FP32, rtol=0.0):
    """``atol + rtol * |ref|_max`` — a magnitude-aware bound."""
    return atol + rtol * ref.float().abs().max().item()


def make_state_on(b, device, seed=1, dtype=torch.float32, hv=HV, dk=DK, dv=DV):
    """:func:`make_state` but generated **on the device**.

    ``make_state`` seeds a CPU generator, which is the right thing for the
    small reproducible cases but costs seconds and 1.6 GB of host RAM at
    B=512 (402 M elements).  The packed-path tests only need *a* state, not
    the same state as the CPU tier."""
    gen = torch.Generator(device=device).manual_seed(seed)
    s = torch.randn(b, hv, dk, dv, generator=gen, device=device) * 0.02
    return s.to(dtype)


class pinned_variant:
    """Pin ``cfg['variant']`` (see ``triton_ops.parse_variant``) for a block.

    The variant is a *constexpr*, so it has to be pinned around the launch, not
    just around the wrapper call."""

    def __init__(self, cfg, variant):
        self.cfg, self.variant = cfg, variant

    def __enter__(self):
        self.saved = self.cfg.get("variant", "")
        self.cfg["variant"] = self.variant
        return self

    def __exit__(self, *exc):
        self.cfg["variant"] = self.saved
        return False


# =========================================================================== #
# CPU: torch backend vs the reference oracle
# =========================================================================== #
class TestTorchBackendVsOracle(unittest.TestCase):
    def test_recurrent_matches_oracle(self):
        inp = make_inputs(2, 4, seed=3)
        s0 = make_state(2, seed=4)
        o_ref, s_ref = oracle_recurrent(
            expand(inp["q"]), expand(inp["k"]), inp["v"], inp["g"], inp["beta"],
            initial_state=s0.clone(), output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        o, s = torch_ops.recurrent_gdn(**inp, initial_state=s0.clone())
        self.assertLess(maxdiff(o, o_ref), ATOL_FP32)
        self.assertLess(maxdiff(s, s_ref), ATOL_FP32)

    def test_chunk_matches_oracle(self):
        inp = make_inputs(2, 130, seed=5)  # 130 = 2 chunks + 2 -> exercises padding
        s0 = make_state(2, seed=6)
        o_ref, s_ref = oracle_chunk(
            expand(inp["q"]), expand(inp["k"]), inp["v"], inp["g"], inp["beta"],
            chunk_size=64, initial_state=s0.clone(), output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        o, s = torch_ops.chunk_gdn(**inp, initial_state=s0.clone())
        self.assertLess(maxdiff(o, o_ref), ATOL_FP32)
        self.assertLess(maxdiff(s, s_ref), ATOL_FP32)

    def test_chunk_matches_recurrent(self):
        """The chunked and sequential forms are the same function."""
        inp = make_inputs(1, 70, seed=7)
        s0 = make_state(1, seed=8)
        o_c, s_c = torch_ops.chunk_gdn(**inp, initial_state=s0.clone())
        o_r, s_r = torch_ops.recurrent_gdn(**inp, initial_state=s0.clone())
        self.assertLess(maxdiff(o_c, o_r), ATOL_FP32)
        self.assertLess(maxdiff(s_c, s_r), ATOL_FP32)

    def test_conv_update_matches_oracle(self):
        c, w = 512, shapes.CONV_KERNEL
        b = 3
        x = torch.randn(b, c)
        weight = torch.randn(c, w) * 0.1
        pool = torch.randn(5, c, w - 1)
        slots = torch.tensor([4, 0, 2], dtype=torch.int32)

        ref_state = pool[slots.long()].clone()
        ref = oracle_conv1d(x.unsqueeze(-1), ref_state, weight)[..., 0]

        pool2 = pool.clone()
        got = api.causal_conv_update(x, pool2, slots, weight, backend="torch")
        self.assertLess(maxdiff(got, ref), ATOL_FP32)
        self.assertLess(maxdiff(pool2[slots.long()], ref_state), ATOL_FP32)
        # untouched slots stay untouched
        for s in (1, 3):
            self.assertEqual(maxdiff(pool2[s], pool[s]), 0.0)

    def test_conv_prefill_matches_oracle(self):
        c, w, t = 256, shapes.CONV_KERNEL, 17
        b = 2
        x = torch.randn(b, c, t)
        weight = torch.randn(c, w) * 0.1
        pool = torch.randn(4, c, w - 1)
        slots = torch.tensor([3, 1], dtype=torch.int32)

        ref_state = pool[slots.long()].clone()
        ref = oracle_conv1d(x, ref_state, weight)

        pool2 = pool.clone()
        got = api.causal_conv_prefill(
            x, weight, conv_state_pool=pool2, slot_ids=slots
        )
        self.assertLess(maxdiff(got, ref), ATOL_FP32)
        self.assertLess(maxdiff(pool2[slots.long()], ref_state), ATOL_FP32)

    def test_conv_prefill_tiling_is_exact(self):
        """``tile_tokens`` bounds the fp32 working set of the torch conv
        fallback (an untiled 8192-token chunk materialises
        671 MiB of fp32 per GDN layer, enough to OOM a loaded server). A causal depthwise conv only needs the previous ``W-1``
        inputs as left context, so tiling must be *exact*, not approximate --
        including at a tile smaller than the kernel width, and including the
        state written back to the pool."""
        c, w, t = 64, shapes.CONV_KERNEL, 37
        x = torch.randn(2, c, t)
        weight = torch.randn(c, w) * 0.1
        pool = torch.randn(4, c, w - 1)
        slots = torch.tensor([3, 1], dtype=torch.int32)

        pool_ref = pool.clone()
        ref = api.causal_conv_prefill(x, weight, conv_state_pool=pool_ref, slot_ids=slots)
        for tile in (1, 2, 3, 8, 16, t - 1, t, t + 5):
            with self.subTest(tile=tile):
                pool_t = pool.clone()
                got = api.causal_conv_prefill(
                    x, weight, conv_state_pool=pool_t, slot_ids=slots, tile_tokens=tile
                )
                self.assertLess(maxdiff(got, ref), ATOL_FP32)
                self.assertLess(maxdiff(pool_t, pool_ref), ATOL_FP32)

    def test_conv_prefill_tiling_is_exact_varlen(self):
        """Same, on the packed-varlen path the scheduler actually uses."""
        c, w = 64, shapes.CONV_KERNEL
        lens = [11, 1, 20, 5]
        t = sum(lens)
        cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32)
        x = torch.randn(1, c, t)
        weight = torch.randn(c, w) * 0.1
        pool = torch.randn(6, c, w - 1)
        slots = torch.tensor([4, 0, 2, 5], dtype=torch.int32)

        pool_ref = pool.clone()
        ref = api.causal_conv_prefill(
            x, weight, cu_seqlens=cu, conv_state_pool=pool_ref, slot_ids=slots
        )
        for tile in (1, 4, 7, 64):
            with self.subTest(tile=tile):
                pool_t = pool.clone()
                got = api.causal_conv_prefill(
                    x, weight, cu_seqlens=cu, conv_state_pool=pool_t,
                    slot_ids=slots, tile_tokens=tile,
                )
                self.assertLess(maxdiff(got, ref), ATOL_FP32)
                self.assertLess(maxdiff(pool_t, pool_ref), ATOL_FP32)


# =========================================================================== #
# CPU: pool semantics
# =========================================================================== #
class TestStatePool(unittest.TestCase):
    def test_pool_layout_matches_design_doc(self):
        """3.00 MiB/layer/slot fp32, 144 MiB/slot, 72 MiB at fp16."""
        self.assertEqual(shapes.state_bytes_per_slot_per_layer(4), 3 * 1024 * 1024)
        self.assertEqual(
            shapes.state_bytes_per_slot_per_layer(4) * shapes.NUM_GDN_LAYERS,
            144 * 1024 * 1024,
        )
        self.assertEqual(
            shapes.conv_state_bytes_per_slot_per_layer(2), 60 * 1024
        )
        pool = st.alloc_state_pool(2, dtype="fp16")
        self.assertEqual(tuple(pool.shape), (2, 48, 48, 128, 128))
        _, total, per_slot = st.describe_pool(pool)
        self.assertEqual(per_slot, 72 * 1024 * 1024)
        self.assertEqual(total, 2 * per_slot)

    def test_layer_view_is_a_view_not_a_copy(self):
        pool = st.alloc_state_pool(3, n_layers=4, n_v_heads=2, head_k=8, head_v=8)
        lyr = st.layer_state(pool, 2)
        self.assertEqual(tuple(lyr.shape), (3, 2, 8, 8))
        lyr[1] = 5.0
        self.assertEqual(pool[1, 2].max().item(), 5.0)
        self.assertEqual(pool[1, 1].max().item(), 0.0)
        # the slot stride is the *whole slot*, which is what the kernels need
        self.assertEqual(lyr.stride(0), 4 * 2 * 8 * 8)

    def test_decode_step_only_touches_named_slots(self):
        n_slots, b = 6, 3
        pool = torch.zeros(n_slots, HV, DK, DV)
        pool.copy_(torch.randn_like(pool) * 0.02)
        before = pool.clone()
        slots = torch.tensor([5, 0, 3], dtype=torch.int32)  # permuted + gapped
        inp = make_inputs(b, 1, seed=11)
        api.gdn_decode_step(**inp, state_pool=pool, slot_ids=slots, backend="torch")
        for s in (1, 2, 4):
            self.assertEqual(maxdiff(pool[s], before[s]), 0.0)
        for s in (5, 0, 3):
            self.assertGreater(maxdiff(pool[s], before[s]), 0.0)

    def test_decode_step_slot_gather_matches_per_row(self):
        """Permuted slot ids give the same answer as running each row alone."""
        n_slots, b = 8, 4
        pool = torch.randn(n_slots, HV, DK, DV) * 0.02
        slots = torch.tensor([7, 2, 5, 1], dtype=torch.int32)
        inp = make_inputs(b, 1, seed=12)

        pool_a = pool.clone()
        out_a = api.gdn_decode_step(
            **inp, state_pool=pool_a, slot_ids=slots, backend="torch"
        )

        for i, s in enumerate(slots.tolist()):
            pool_b = pool.clone()
            one = {kk: vv[i : i + 1] for kk, vv in inp.items()}
            out_b = api.gdn_decode_step(
                **one,
                state_pool=pool_b,
                slot_ids=torch.tensor([s], dtype=torch.int32),
                backend="torch",
            )
            self.assertLess(maxdiff(out_a[i : i + 1], out_b), ATOL_FP32)
            self.assertLess(maxdiff(pool_a[s], pool_b[s]), ATOL_FP32)

    def test_decode_step_matches_oracle_through_pool(self):
        b = 2
        inp = make_inputs(b, 1, seed=13)
        s0 = make_state(b, seed=14)
        o_ref, s_ref = oracle_recurrent(
            expand(inp["q"]), expand(inp["k"]), inp["v"], inp["g"], inp["beta"],
            initial_state=s0.clone(), output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        pool = torch.zeros(4, HV, DK, DV)
        slots = torch.tensor([3, 1], dtype=torch.int32)
        pool[slots.long()] = s0
        o = api.gdn_decode_step(
            **inp, state_pool=pool, slot_ids=slots, backend="torch"
        )
        self.assertLess(maxdiff(o, o_ref), ATOL_FP32)
        self.assertLess(maxdiff(pool[slots.long()], s_ref), ATOL_FP32)

    def test_gva_expanded_and_collapsed_inputs_agree(self):
        b = 2
        inp = make_inputs(b, 3, seed=15)
        s0 = make_state(b, seed=16)
        o_a, s_a = torch_ops.recurrent_gdn(**inp, initial_state=s0.clone())
        o_b, s_b = torch_ops.recurrent_gdn(
            expand(inp["q"]), expand(inp["k"]), inp["v"], inp["g"], inp["beta"],
            initial_state=s0.clone(),
        )
        self.assertEqual(maxdiff(o_a, o_b), 0.0)
        self.assertEqual(maxdiff(s_a, s_b), 0.0)

    def test_gva_head_mapping_is_floor_divide(self):
        """v-head hv reads k-head hv//3 — the mapping baked into the Triton
        kernel (``i_h = i_hv // GVA``) and into fla's kernel.  Easiest thing in
        the whole package to get backwards, so pin it."""
        x = torch.randn(1, 1, HK, DK)
        xe = expand(x)
        self.assertEqual(xe.shape[2], HV)
        for hv in range(HV):
            self.assertTrue(torch.equal(xe[0, 0, hv], x[0, 0, hv // REP]))

    def test_triton_decode_program_simulation(self):
        """Transcribe the Triton decode kernel program-by-program on CPU.

        This is not the kernel — it cannot catch a Triton API misuse — but it
        does check the part that is pure indexing logic (slot resolution, GVA
        head mapping, the [K, BV] tile decomposition over the value dim) against
        the reference.  A tiling bug shows up here, not on the GPU host.
        """
        b, n_slots, bv = 3, 5, 32
        inp = make_inputs(b, 1, seed=19)
        pool = torch.randn(n_slots, HV, DK, DV) * 0.02
        slots = torch.tensor([4, 1, 3], dtype=torch.int32)

        ref_pool = pool.clone()
        ref_out = api.gdn_decode_step(
            **inp, state_pool=ref_pool, slot_ids=slots, backend="torch"
        )

        sim_pool = pool.clone()
        sim_out = torch.zeros(b, 1, HV, DV)
        scale = DK ** -0.5
        gva = HV // HK
        for i_n in range(b):                        # program_id(0)
            slot = int(slots[i_n])
            for i_hv in range(HV):                  # program_id(1)
                i_h = i_hv // gva
                q = torch_ops.l2norm(inp["q"][i_n, 0, i_h].float()) * scale
                k = torch_ops.l2norm(inp["k"][i_n, 0, i_h].float())
                g = float(inp["g"][i_n, 0, i_hv])
                bt = float(inp["beta"][i_n, 0, i_hv])
                for i_v in range(0, DV, bv):        # program_id(2)
                    sl = slice(i_v, i_v + bv)
                    qk = (q * k).sum()          # hoisted, as in the kernel
                    s = sim_pool[slot, i_hv, :, sl].float() * torch.exp(
                        torch.tensor(g)
                    )
                    # the two reductions read the same S' and are independent
                    kv = (s * k[:, None]).sum(0)
                    qs = (s * q[:, None]).sum(0)
                    d = (inp["v"][i_n, 0, i_hv, sl].float() - kv) * bt
                    sim_out[i_n, 0, i_hv, sl] = qs + qk * d
                    sim_pool[slot, i_hv, :, sl] = s + k[:, None] * d[None, :]
        self.assertLess(maxdiff(sim_out, ref_out), ATOL_FP32)
        self.assertLess(maxdiff(sim_pool, ref_pool), ATOL_FP32)

    def test_fp16_pool_round_trip(self):
        """fp16 state: math stays fp32, only storage rounds. Drift reported."""
        b = 2
        inp = make_inputs(b, 1, seed=17)
        s0 = make_state(b, seed=18)
        pool32 = s0.clone()
        pool16 = s0.to(torch.float16)
        slots = torch.arange(b, dtype=torch.int32)
        o32 = api.gdn_decode_step(
            **inp, state_pool=pool32, slot_ids=slots, backend="torch"
        )
        o16 = api.gdn_decode_step(
            **inp, state_pool=pool16, slot_ids=slots, backend="torch"
        )
        self.assertEqual(pool16.dtype, torch.float16)
        drift_o = maxdiff(o32, o16)
        drift_s = maxdiff(pool32, pool16.float())
        # informational, but must be in the right ballpark (not broken, not exact)
        self.assertLess(drift_o, ATOL_FP16_STATE)
        self.assertLess(drift_s, ATOL_FP16_STATE)
        self.assertGreater(drift_s, 0.0)
        print(f"\n  [fp16 state drift] out={drift_o:.2e} state={drift_s:.2e}")


class TestConvLayouts(unittest.TestCase):
    """The conv ring pool may be width-major [n, W-1, C] (the default, and the
    only one that coalesces in the decode kernel) or channel-major [n, C, W-1]
    (the reference model's layout).  Both must give bit-comparable answers."""

    def _pools(self, n_slots, c, w):
        cw = torch.randn(n_slots, c, w - 1)          # channel-major
        return cw, cw.transpose(1, 2).contiguous()   # width-major

    def test_update_agrees_across_layouts(self):
        c, w, b, n = 512, shapes.CONV_KERNEL, 3, 6
        x = torch.randn(b, c)
        weight = torch.randn(c, w) * 0.1
        p_cw, p_wc = self._pools(n, c, w)
        slots = torch.tensor([5, 0, 2], dtype=torch.int32)

        a_cw, a_wc = p_cw.clone(), p_wc.clone()
        y1 = api.causal_conv_update(x, a_cw, slots, weight, backend="torch")
        y2 = api.causal_conv_update(x, a_wc, slots, weight, backend="torch")
        self.assertEqual(maxdiff(y1, y2), 0.0)
        self.assertEqual(maxdiff(a_cw, a_wc.transpose(1, 2)), 0.0)
        # ... and both match the reference oracle
        ref_state = p_cw[slots.long()].clone()
        ref = oracle_conv1d(x.unsqueeze(-1), ref_state, weight)[..., 0]
        self.assertLess(maxdiff(y1, ref), ATOL_FP32)
        self.assertLess(maxdiff(a_cw[slots.long()], ref_state), ATOL_FP32)

    def test_prefill_and_spec_agree_across_layouts(self):
        c, w, b, n, t = 256, shapes.CONV_KERNEL, 2, 4, 9
        x = torch.randn(b, c, t)
        weight = torch.randn(c, w) * 0.1
        p_cw, p_wc = self._pools(n, c, w)
        slots = torch.tensor([3, 1], dtype=torch.int32)

        a_cw, a_wc = p_cw.clone(), p_wc.clone()
        y1 = api.causal_conv_prefill(x, weight, conv_state_pool=a_cw, slot_ids=slots)
        y2 = api.causal_conv_prefill(x, weight, conv_state_pool=a_wc, slot_ids=slots)
        self.assertEqual(maxdiff(y1, y2), 0.0)
        self.assertEqual(maxdiff(a_cw, a_wc.transpose(1, 2)), 0.0)

        m = torch.tensor([1, 3], dtype=torch.int32)
        b_cw, b_wc = p_cw.clone(), p_wc.clone()
        z1 = api.causal_conv_verify_and_commit(x[:, :, :4], b_cw, slots, weight, m)
        z2 = api.causal_conv_verify_and_commit(x[:, :, :4], b_wc, slots, weight, m)
        self.assertEqual(maxdiff(z1, z2), 0.0)
        self.assertEqual(maxdiff(b_cw, b_wc.transpose(1, 2)), 0.0)

    def test_alloc_and_strides(self):
        c, w = shapes.CONV_DIM, shapes.CONV_KERNEL
        wc = st.alloc_conv_state_pool(2, n_layers=3, layout="width_major")
        cw = st.alloc_conv_state_pool(2, n_layers=3, layout="channel_major")
        self.assertEqual(tuple(wc.shape), (2, 3, w - 1, c))
        self.assertEqual(tuple(cw.shape), (2, 3, c, w - 1))
        self.assertTrue(st.conv_pool_is_width_major(wc, c))
        self.assertFalse(st.conv_pool_is_width_major(cw, c))
        # 60 KiB/layer/slot either way
        self.assertEqual(wc.numel() * 2 // (2 * 3), 60 * 1024)
        if triton_ops.is_available() or True:  # pure-python stride math
            self.assertEqual(
                triton_ops.conv_state_strides(st.layer_conv_state(wc, 1), c), (1, c)
            )
            self.assertEqual(
                triton_ops.conv_state_strides(st.layer_conv_state(cw, 1), c), (w - 1, 1)
            )

    def test_prepare_conv_weight_is_width_major_but_same_values(self):
        w = torch.randn(shapes.CONV_DIM, shapes.CONV_KERNEL)
        wt = st.prepare_conv_weight(w)
        self.assertTrue(torch.equal(w, wt))
        self.assertEqual(wt.stride(0), 1)
        self.assertEqual(wt.stride(1), shapes.CONV_DIM)
        self.assertEqual(triton_ops.conv_weight_strides(wt), (1, shapes.CONV_DIM))


class TestGateInKernel(unittest.TestCase):
    """``A_log``/``dt_bias`` let the kernel fold in
    ``g = -exp(A_log)*softplus(a+dt_bias)`` and ``beta = sigmoid(b)``, saving
    two elementwise launches per layer (numerics unchanged)."""

    def _raw(self, b, t, seed=71):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        a_raw = torch.randn(b, t, HV, generator=gen)
        b_raw = torch.randn(b, t, HV, generator=gen)
        A_log = torch.log(torch.empty(HV).uniform_(0.01, 16, generator=gen))
        dt_bias = torch.ones(HV)
        return a_raw, b_raw, A_log, dt_bias

    def test_apply_gate_matches_model_py_formula(self):
        a_raw, b_raw, A_log, dt_bias = self._raw(2, 3)
        g, beta = torch_ops.apply_gate(a_raw, b_raw, A_log, dt_bias)
        ref_g = -A_log.float().exp() * torch.nn.functional.softplus(
            a_raw.float() + dt_bias.float()
        )
        self.assertEqual(maxdiff(g, ref_g), 0.0)
        self.assertEqual(maxdiff(beta, b_raw.sigmoid()), 0.0)

    def test_decode_step_gate_in_kernel_matches_precomputed(self):
        b = 3
        inp = make_inputs(b, 1, seed=72)
        a_raw, b_raw, A_log, dt_bias = self._raw(b, 1)
        g, beta = torch_ops.apply_gate(a_raw, b_raw, A_log, dt_bias)
        s0 = make_state(b, seed=73)
        slots = torch.arange(b, dtype=torch.int32)

        p1 = s0.clone()
        o1 = api.gdn_decode_step(
            inp["q"], inp["k"], inp["v"], g, beta, p1, slots, backend="torch"
        )
        p2 = s0.clone()
        o2 = api.gdn_decode_step(
            inp["q"], inp["k"], inp["v"], a_raw, b_raw, p2, slots,
            backend="torch", A_log=A_log, dt_bias=dt_bias,
        )
        self.assertEqual(maxdiff(o1, o2), 0.0)
        self.assertEqual(maxdiff(p1, p2), 0.0)

    def test_verify_and_commit_gate_in_kernel(self):
        b, n = 2, 4
        inp = make_inputs(b, n, seed=74)
        a_raw, b_raw, A_log, dt_bias = self._raw(b, n)
        g, beta = torch_ops.apply_gate(a_raw, b_raw, A_log, dt_bias)
        s0 = make_state(b, seed=75)
        slots = torch.arange(b, dtype=torch.int32)
        m = torch.tensor([1, 3], dtype=torch.int32)
        for method in ("two_phase", "fused"):
            p1, p2 = s0.clone(), s0.clone()
            o1 = api.gdn_verify_and_commit(
                inp["q"], inp["k"], inp["v"], g, beta, p1, slots, m,
                method=method, backend="torch",
            )
            o2 = api.gdn_verify_and_commit(
                inp["q"], inp["k"], inp["v"], a_raw, b_raw, p2, slots, m,
                method=method, backend="torch", A_log=A_log, dt_bias=dt_bias,
            )
            self.assertEqual(maxdiff(o1, o2), 0.0, method)
            self.assertEqual(maxdiff(p1, p2), 0.0, method)


class TestTiling(unittest.TestCase):
    """The measured tiling tables.

    Every assertion here is a *measurement*, not a model.  Reasoning about
    occupancy and CTA counts has repeatedly been beaten by the sweep,
    so these tests exist to stop the next plausible-sounding edit:

    * decode: few warps, ~200 registers, ~25% occupancy beats 100% occupancy by
      25-40% — the K-axis reduction binds, not occupancy;
    * window: BV collapses to **8** at B>=64 (16 dv-blocks per head), far below
      the register-budget argument's (32, 2);
    * conv: B=1 wants the **largest** BC (2048 = 5 CTAs), not the most CTAs.
    """

    DECODE = {  # itemsize -> {batch: (BV, num_warps)}
        4: {1: (16, 1), 2: (32, 1), 8: (32, 1), 16: (64, 2), 64: (64, 2),
            256: (64, 2), 512: (64, 2)},
        2: {1: (16, 1), 2: (64, 1), 8: (64, 1), 16: (64, 2), 64: (64, 2),
            256: (64, 2), 512: (64, 2)},
    }
    WINDOW = {1: (16, 1), 8: (16, 1), 64: (8, 1), 256: (8, 1)}
    CONV = {1: (2048, 8), 8: (2048, 8), 64: (1024, 8), 128: (1024, 8),
            256: (512, 4), 512: (512, 4)}

    def test_decode_table_matches_measured_winners(self):
        for itemsize, table in self.DECODE.items():
            for b, want in table.items():
                got = triton_ops.pick_decode_tiling(b, HV, DV, itemsize=itemsize)
                self.assertEqual(got, want, f"itemsize={itemsize} B={b}")

    def test_window_table_matches_measured_winners(self):
        for itemsize in (4, 2):
            for b, want in self.WINDOW.items():
                got = triton_ops.pick_decode_tiling(
                    b, HV, DV, triton_ops.WINDOW_TUNING, tiles_factor=4,
                    itemsize=itemsize,
                )
                self.assertEqual(got, want, f"itemsize={itemsize} B={b}")

    def test_conv_table_matches_measured_winners(self):
        for b, want in self.CONV.items():
            self.assertEqual(triton_ops.pick_conv_tiling(b, shapes.CONV_DIM), want,
                             f"B={b}")

    # (itemsize, batch) entries knowingly above the 255-register cap.  fp16
    # B=2..15 uses (64, 1) = 256 tile elements/thread and *does* spill (50
    # spills measured) — it still won at B=8 (7.4 us) because at that batch
    # there is spare bandwidth to hide the spill traffic.  Anything not listed
    # here must stay inside the budget.
    KNOWN_SPILLERS = {(2, 2), (2, 8)}

    def test_only_known_entries_exceed_the_register_budget(self):
        """A thread caps at 255 registers, so >255 tile elements per thread
        spills to local memory.  That is sometimes still the fastest choice, but
        it must be a deliberate, listed one — not something a future edit
        introduces silently."""
        for itemsize in (4, 2):
            for b in (1, 2, 8, 16, 64, 256, 512):
                bv, nw = triton_ops.pick_decode_tiling(b, HV, DV, itemsize=itemsize)
                elems = DK * bv / (32 * nw)
                if (itemsize, b) in self.KNOWN_SPILLERS:
                    self.assertGreater(elems, 255, f"decode B={b}: no longer spills, "
                                                    "drop it from KNOWN_SPILLERS")
                else:
                    self.assertLessEqual(elems, 255, f"decode itemsize={itemsize} B={b}")
                wbv, wnw = triton_ops.pick_decode_tiling(
                    b, HV, DV, triton_ops.WINDOW_TUNING, tiles_factor=4,
                    itemsize=itemsize,
                )
                self.assertLessEqual(
                    2 * DK * wbv / (32 * wnw), 255, f"window itemsize={itemsize} B={b}"
                )

    def test_non_model_shape_falls_back_to_the_analytic_rule(self):
        """The tables were only swept at HV=48, V=128 (and C=10240); anything
        else must not silently reuse them."""
        bv, nw = triton_ops.pick_decode_tiling(1, 6, 64, itemsize=4)
        self.assertLessEqual(bv, 64)
        self.assertGreaterEqual(nw, 1)
        bc, _ = triton_ops.pick_conv_tiling(1, 4096)
        self.assertNotEqual(bc, 2048)  # not the conv table's B=1 answer

    def test_explicit_override_beats_the_table(self):
        old = dict(triton_ops.DECODE_TUNING)
        try:
            triton_ops.DECODE_TUNING.update(BV=128, num_warps=4)
            self.assertEqual(triton_ops.pick_decode_tiling(1, HV, DV), (128, 4))
            self.assertEqual(triton_ops.pick_decode_tiling(512, HV, DV), (128, 4))
        finally:
            triton_ops.DECODE_TUNING.update(old)


class TestVariantSelection(unittest.TestCase):
    """Kernel-variant selection — pure host logic, no GPU needed.

    The three flags (``packed`` / ``hoist`` / ``sched``) are compile-time
    constexprs, so a mis-parse silently compiles the *wrong kernel* and every
    numerical test still passes.  Pin the parsing and the table plumbing here.
    """

    def test_parse_and_name_round_trip(self):
        for name in triton_ops.all_variants():
            self.assertEqual(
                triton_ops.variant_name(triton_ops.parse_variant(name)), name
            )

    def test_all_variants_is_the_full_product(self):
        self.assertEqual(len(triton_ops.all_variants()), 2 ** len(triton_ops.VARIANT_FLAGS))
        self.assertEqual(triton_ops.all_variants()[0], "base")
        self.assertEqual(len(set(triton_ops.all_variants())), len(triton_ops.all_variants()))

    def test_parse_is_order_and_separator_insensitive(self):
        want = {"packed": True, "hoist": True, "sched": False}
        for s in ("packed_hoist", "hoist+packed", "hoist,packed", "PACKED_HOIST"):
            self.assertEqual(triton_ops.parse_variant(s), want, s)

    def test_base_aliases_are_all_flags_off(self):
        for s in ("", None, "base", "v4", "none"):
            self.assertEqual(set(triton_ops.parse_variant(s).values()), {False}, repr(s))
        self.assertEqual(set(triton_ops.parse_variant("all").values()), {True})

    def test_unknown_flag_is_rejected(self):
        """A typo must not silently degrade to `base` — that would make a
        whole sweep row a duplicate of the baseline without saying so."""
        with self.assertRaises(ValueError):
            triton_ops.parse_variant("packd")

    def test_fp32_default_variant_is_frozen_base(self):
        """The fp32 numbers are frozen (80-82% of HBM); the fp32 path must
        stay the original kernel so the variant comparison has a fixed reference."""
        for b in (1, 8, 64, 256, 512):
            self.assertEqual(
                triton_ops.pick_decode_variant(b, HV, DV, itemsize=4), "base"
            )

    def test_env_style_override_beats_the_table(self):
        old = dict(triton_ops.DECODE_TUNING)
        try:
            triton_ops.DECODE_TUNING["variant"] = "packed_hoist"
            self.assertEqual(
                triton_ops.pick_decode_variant(64, HV, DV, itemsize=2),
                "packed_hoist",
            )
        finally:
            triton_ops.DECODE_TUNING.update(old)

    def test_tiling_table_rows_may_carry_a_variant(self):
        """The baked table must be extensible to ``(BV, warps, variant)``
        without breaking either reader."""
        old = triton_ops.DECODE_TABLE[2]
        try:
            triton_ops.DECODE_TABLE[2] = ((16, (64, 2, "packed_hoist")), (0, (16, 1)))
            self.assertEqual(
                triton_ops.pick_decode_tiling(64, HV, DV, itemsize=2), (64, 2)
            )
            self.assertEqual(
                triton_ops.pick_decode_variant(64, HV, DV, itemsize=2), "packed_hoist"
            )
            # a 2-tuple row still falls through to the variant table
            self.assertEqual(
                triton_ops.pick_decode_variant(1, HV, DV, itemsize=2), "base"
            )
        finally:
            triton_ops.DECODE_TABLE[2] = old

    def test_non_model_shape_gets_base(self):
        self.assertEqual(triton_ops.pick_decode_variant(64, 6, 64, itemsize=2), "base")

    # The adopted variant table.  Exactly one entry cleared the
    # adoption bar (>= 3%, same answer from fp16 and bf16); everything else
    # measured inside this kernel's own run-to-run spread and stayed `base`.
    # Numbers, not a model — do not "tidy" this into a rule.
    DECODE_VARIANT = {
        4: {1: "base", 8: "base", 64: "base", 256: "base", 512: "base"},
        2: {1: "base", 8: "base", 64: "base", 128: "base", 256: "base",
            511: "base", 512: "packed_sched", 1024: "packed_sched"},
    }

    def test_decode_variant_table_matches_the_adopted_measurement(self):
        for itemsize, table in self.DECODE_VARIANT.items():
            for b, want in table.items():
                self.assertEqual(
                    triton_ops.pick_decode_variant(b, HV, DV, itemsize=itemsize),
                    want, f"itemsize={itemsize} B={b}",
                )

    def test_window_variant_table_is_base_everywhere(self):
        """`base` won every measured window case; the packed variants were
        12-13% *slower* at B=32 (BV is already down at 8-16 to fit two state
        tiles, and BV/2 is too narrow to pay for the unpack)."""
        for itemsize in (4, 2):
            for b in (1, 8, 32, 64, 256, 512):
                self.assertEqual(
                    triton_ops.pick_decode_variant(
                        b, HV, DV, triton_ops.WINDOW_TUNING, itemsize=itemsize
                    ),
                    "base", f"itemsize={itemsize} B={b}",
                )


class TestPackedStateView(unittest.TestCase):
    """``packed_state_view`` reinterprets a 16-bit pool as int32 words.

    Everything about the packed kernel rests on "one int32 == the two state
    elements at ``[.., 2j]`` and ``[.., 2j+1]``, low half first" — so pin the
    endianness and the silent-degrade rules rather than trusting them.
    """

    def _pool(self, dtype, n=4):
        return torch.zeros(n, HV, DK, DV, dtype=dtype)

    def test_fp16_and_bf16_pools_pack(self):
        for dtype, code in ((torch.float16, 1), (torch.bfloat16, 2)):
            pool = self._pool(dtype)
            view, stride, dt = triton_ops.packed_state_view(pool, 64)
            self.assertEqual(dt, code)
            self.assertEqual(view.dtype, torch.int32)
            self.assertEqual(tuple(view.shape), (4, HV, DK, DV // 2))
            self.assertEqual(stride, pool.stride(0) // 2)

    def test_low_half_is_the_even_value_column(self):
        """Little-endian: bits 15:0 of word j are value column 2j.  If this
        flips, the kernel silently transposes pairs along V."""
        pool = torch.zeros(1, HV, DK, DV, dtype=torch.float16)
        pool[0, 0, 0, 0] = 1.0  # fp16 1.0 == 0x3C00
        pool[0, 0, 0, 1] = 2.0  # fp16 2.0 == 0x4000
        view, _, _ = triton_ops.packed_state_view(pool, 64)
        word = view[0, 0, 0, 0].item() & 0xFFFFFFFF
        self.assertEqual(word & 0xFFFF, 0x3C00)
        self.assertEqual((word >> 16) & 0xFFFF, 0x4000)

    def test_fp32_pool_degrades_silently(self):
        pool = self._pool(torch.float32)
        view, stride, dt = triton_ops.packed_state_view(pool, 64)
        self.assertEqual(dt, 0)
        self.assertIs(view, pool)
        self.assertEqual(stride, pool.stride(0))

    def test_odd_bv_degrades_silently(self):
        """BV must be even for the [NK, BV/2] word tile to exist."""
        pool = self._pool(torch.float16)
        self.assertEqual(triton_ops.packed_state_view(pool, 33)[2], 0)

    def test_layer_view_of_a_whole_model_pool_packs(self):
        """The real caller passes ``state.layer_state(pool, l)`` — a strided,
        non-contiguous-in-dim-0 view.  ``Tensor.view(int32)`` must still take
        it (even strides, even storage offset)."""
        whole = st.alloc_state_pool(2, n_layers=4, dtype=torch.float16)
        for layer in range(4):
            lay = st.layer_state(whole, layer)
            view, stride, dt = triton_ops.packed_state_view(lay, 64)
            self.assertEqual(dt, 1, f"layer {layer}")
            self.assertEqual(stride, lay.stride(0) // 2)


class TestPrenorm(unittest.TestCase):
    """Hoisting ``l2norm(q)``, ``l2norm(k)`` and ``q.k`` out of the kernel.

    Each program repeats all three, and ``GVA * (NV/BV)`` programs share a
    k-head — 24x at the fp32 B=1 decode tiling, 48x at the window kernel's
    B>=64 tiling.  On this kernel a cross-thread reduction is latency-bound, so
    a 128-element one is not much cheaper than an 8192-element one; an earlier kernel traded
    one for a serial dependency between the two big ones and came out exactly
    even, which is the evidence for doing this.
    """

    def test_prenormalize_matches_in_kernel_normalisation(self):
        inp = make_inputs(3, 2, seed=111)
        qn, kn, qk = torch_ops.prenormalize_qk(inp["q"], inp["k"])
        self.assertEqual(tuple(qk.shape), (3, 2, HK))
        ref_q = torch_ops.l2norm(inp["q"].float()) * (DK ** -0.5)
        ref_k = torch_ops.l2norm(inp["k"].float())
        self.assertEqual(maxdiff(qn, ref_q), 0.0)
        self.assertEqual(maxdiff(kn, ref_k), 0.0)
        self.assertLess(maxdiff(qk, (ref_q * ref_k).sum(-1)), 1e-6)

    def test_decode_step_prenorm_matches(self):
        b = 3
        inp = make_inputs(b, 1, seed=112)
        s0 = make_state(b, seed=113)
        slots = torch.arange(b, dtype=torch.int32)
        qn, kn, qk = torch_ops.prenormalize_qk(inp["q"], inp["k"])

        p1 = s0.clone()
        o1 = api.gdn_decode_step(**inp, state_pool=p1, slot_ids=slots, backend="torch")
        p2 = s0.clone()
        o2 = api.gdn_decode_step(
            qn, kn, inp["v"], inp["g"], inp["beta"], p2, slots,
            backend="torch", qk=qk,
        )
        self.assertLess(maxdiff(o1, o2), ATOL_FP32)
        self.assertLess(maxdiff(p1, p2), ATOL_FP32)

    def test_verify_and_commit_prenorm_matches(self):
        b, n = 3, 4
        inp = make_inputs(b, n, seed=114)
        s0 = make_state(b, seed=115)
        slots = torch.arange(b, dtype=torch.int32)
        m = torch.tensor([0, 2, 4], dtype=torch.int32)
        qn, kn, qk = torch_ops.prenormalize_qk(inp["q"], inp["k"])
        for method in ("two_phase", "fused"):
            p1, p2 = s0.clone(), s0.clone()
            o1 = api.gdn_verify_and_commit(
                **inp, state_pool=p1, slot_ids=slots, m=m,
                method=method, backend="torch",
            )
            o2 = api.gdn_verify_and_commit(
                qn, kn, inp["v"], inp["g"], inp["beta"], p2, slots, m,
                method=method, backend="torch", qk=qk,
            )
            self.assertLess(maxdiff(o1, o2), ATOL_FP32, method)
            self.assertLess(maxdiff(p1, p2), ATOL_FP32, method)

    def test_bad_qk_shape_raises(self):
        if not triton_ops.is_available():
            self.skipTest("needs triton for the shape check")
        with self.assertRaises(ValueError):
            triton_ops._prenorm_args(torch.zeros(3, 9), False, 3, 1, HK)


class TestOutputIdentity(unittest.TestCase):
    """``o = S''^T q`` == ``S'^T q + (q.k) d``.

    The Triton kernels compute the second form so that the two K-axis
    reductions read the same pre-update state ``S'`` and are therefore
    independent — the textbook form serialises them through the rank-1 update.
    The identity is exact: ``(k (x) d)^T q = (q.k) d``.  If this test ever
    fails, the kernels are computing a different function.
    """

    def test_identity_holds_elementwise(self):
        torch.manual_seed(0)
        b, hv, kd, vd = 2, 5, 128, 64
        s1 = torch.randn(b, hv, kd, vd, dtype=torch.float64)  # S' (post-decay)
        q = torch.randn(b, hv, kd, dtype=torch.float64)
        k = torch.randn(b, hv, kd, dtype=torch.float64)
        d = torch.randn(b, hv, vd, dtype=torch.float64)

        s2 = s1 + k.unsqueeze(-1) * d.unsqueeze(-2)   # S'' = S' + k (x) d
        textbook = (s2 * q.unsqueeze(-1)).sum(-2)     # S''^T q
        fused = (s1 * q.unsqueeze(-1)).sum(-2) + (q * k).sum(-1, keepdim=True) * d
        self.assertLess(maxdiff(textbook, fused), 1e-10)

    def test_full_step_both_ways_agree_in_fp32(self):
        """End to end at the real shape, in fp32, at the 2e-4 contract."""
        b = 2
        inp = make_inputs(b, 1, seed=101)
        s0 = make_state(b, seed=102)
        q = torch_ops.l2norm(inp["q"].float())[:, 0].repeat_interleave(REP, dim=1)
        k = torch_ops.l2norm(inp["k"].float())[:, 0].repeat_interleave(REP, dim=1)
        q = q * (DK ** -0.5)
        v, g, beta = inp["v"][:, 0], inp["g"][:, 0], inp["beta"][:, 0]

        s1 = s0 * g.exp()[..., None, None]
        kv = (s1 * k.unsqueeze(-1)).sum(-2)
        d = (v - kv) * beta.unsqueeze(-1)
        textbook = ((s1 + k.unsqueeze(-1) * d.unsqueeze(-2)) * q.unsqueeze(-1)).sum(-2)
        fused = (s1 * q.unsqueeze(-1)).sum(-2) + (q * k).sum(-1, keepdim=True) * d
        self.assertLess(maxdiff(textbook, fused), ATOL_FP32)
        # ... and both are what the torch backend produces
        ref, _ = torch_ops.recurrent_gdn(**inp, initial_state=s0.clone())
        self.assertLess(maxdiff(fused, ref[:, 0]), ATOL_FP32)


# =========================================================================== #
# CPU: prefill / varlen
# =========================================================================== #
class TestPrefill(unittest.TestCase):
    def test_initial_state_threads_through_split_calls(self):
        """Splitting a request across two chunks is exact."""
        inp = make_inputs(1, 100, seed=21)
        s0 = make_state(1, seed=22)
        o_full, s_full = api.gdn_prefill_chunked(
            **inp, initial_state=s0.clone(), backend="torch"
        )
        cut = 64
        first = {kk: vv[:, :cut] for kk, vv in inp.items()}
        second = {kk: vv[:, cut:] for kk, vv in inp.items()}
        o1, s1 = api.gdn_prefill_chunked(
            **first, initial_state=s0.clone(), backend="torch"
        )
        o2, s2 = api.gdn_prefill_chunked(**second, initial_state=s1, backend="torch")
        self.assertLess(maxdiff(torch.cat([o1, o2], dim=1), o_full), ATOL_FP32)
        self.assertLess(maxdiff(s2, s_full), ATOL_FP32)

    def test_varlen_matches_per_sequence(self):
        lens = [37, 64, 3, 90]
        cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32)
        total = int(cu[-1])
        inp = make_inputs(1, total, seed=23)
        s0 = make_state(len(lens), seed=24)

        o_pack, s_pack = api.gdn_prefill_chunked(
            **inp, cu_seqlens=cu, initial_state=s0.clone(), backend="torch"
        )
        for i, _ in enumerate(lens):
            lo, hi = int(cu[i]), int(cu[i + 1])
            one = {kk: vv[:, lo:hi] for kk, vv in inp.items()}
            o_i, s_i = api.gdn_prefill_chunked(
                **one, initial_state=s0[i : i + 1].clone(), backend="torch"
            )
            self.assertLess(maxdiff(o_pack[:, lo:hi], o_i), ATOL_FP32)
            self.assertLess(maxdiff(s_pack[i : i + 1], s_i), ATOL_FP32)

    def test_varlen_scatters_into_pool(self):
        lens = [20, 40]
        cu = torch.tensor([0, 20, 60], dtype=torch.int32)
        inp = make_inputs(1, 60, seed=25)
        pool = torch.zeros(5, HV, DK, DV)
        slots = torch.tensor([4, 2], dtype=torch.int32)
        _, final = api.gdn_prefill_chunked(
            **inp, cu_seqlens=cu, state_pool=pool, slot_ids=slots, backend="torch"
        )
        self.assertEqual(final.shape[0], len(lens))
        self.assertLess(maxdiff(pool[slots.long()], final), ATOL_FP32)
        self.assertEqual(pool[0].abs().max().item(), 0.0)

    def test_varlen_rejects_batch_gt_1(self):
        inp = make_inputs(2, 8, seed=26)
        with self.assertRaises(ValueError):
            api.gdn_prefill_chunked(
                **inp, cu_seqlens=torch.tensor([0, 4, 8], dtype=torch.int32),
                backend="torch",
            )

    def test_varlen_conv_matches_per_sequence(self):
        c, w = 128, shapes.CONV_KERNEL
        cu = torch.tensor([0, 11, 11, 30], dtype=torch.int32)  # incl. an empty seq
        x = torch.randn(1, c, 30)
        weight = torch.randn(c, w) * 0.1
        pool = torch.randn(4, c, w - 1)
        slots = torch.tensor([3, 1, 0], dtype=torch.int32)

        pool_a = pool.clone()
        got = api.causal_conv_prefill(
            x, weight, cu_seqlens=cu, conv_state_pool=pool_a, slot_ids=slots
        )
        for i, s in enumerate(slots.tolist()):
            lo, hi = int(cu[i]), int(cu[i + 1])
            if hi == lo:
                self.assertEqual(maxdiff(pool_a[s], pool[s]), 0.0)
                continue
            ref_state = pool[s : s + 1].clone()
            ref = oracle_conv1d(x[:, :, lo:hi], ref_state, weight)
            self.assertLess(maxdiff(got[:, :, lo:hi], ref), ATOL_FP32)
            self.assertLess(maxdiff(pool_a[s : s + 1], ref_state), ATOL_FP32)


# =========================================================================== #
# CPU: speculative verify-and-commit
# =========================================================================== #
def _sequential_reference(inp, s0, m, backend="torch"):
    """Ground truth: run ``m`` plain decode steps from ``s0``."""
    pool = s0.clone()
    b = s0.shape[0]
    slots = torch.arange(b, dtype=torch.int32, device=s0.device)
    for t in range(m):
        one = {kk: vv[:, t : t + 1] for kk, vv in inp.items()}
        api.gdn_decode_step(**one, state_pool=pool, slot_ids=slots, backend=backend)
    return pool


class TestVerifyAndCommit(unittest.TestCase):
    N = 4  # k = 3 draft tokens + 1 bonus, the default MTP configuration

    def _check(self, backend, method, atol=ATOL_FP32, device="cpu"):
        b = 3
        inp = make_inputs(b, self.N, device=device, seed=31)
        s0 = make_state(b, device=device, seed=32)
        slots = torch.tensor([2, 0, 1], dtype=torch.int32, device=device)
        pool0 = torch.zeros(4, HV, DK, DV, device=device)
        pool0[slots.long()] = s0

        # outputs for the whole window are independent of m
        o_ref, _ = torch_ops.recurrent_gdn(**inp, initial_state=s0.clone())

        for m_val in range(self.N + 1):
            pool = pool0.clone()
            m = torch.full((b,), m_val, dtype=torch.int32, device=device)
            out = api.gdn_verify_and_commit(
                **inp, state_pool=pool, slot_ids=slots, m=m,
                method=method, backend=backend,
            )
            self.assertLess(
                maxdiff(out, o_ref), atol,
                f"{backend}/{method} outputs differ at m={m_val}",
            )
            ref_state = _sequential_reference(inp, s0, m_val)
            self.assertLess(
                maxdiff(pool[slots.long()], ref_state), atol,
                f"{backend}/{method} committed state differs at m={m_val}",
            )
            if m_val == 0:
                self.assertLess(maxdiff(pool[slots.long()], s0), atol)

    def test_torch_two_phase(self):
        self._check("torch", "two_phase")

    def test_torch_fused(self):
        self._check("torch", "fused")

    def test_per_row_accept_lengths(self):
        """Different ``m`` per row in one batch — the realistic case."""
        b = 4
        inp = make_inputs(b, self.N, seed=33)
        s0 = make_state(b, seed=34)
        slots = torch.arange(b, dtype=torch.int32)
        m = torch.tensor([0, 1, 3, 4], dtype=torch.int32)
        for method in ("two_phase", "fused"):
            pool = s0.clone()
            api.gdn_verify_and_commit(
                **inp, state_pool=pool, slot_ids=slots, m=m,
                method=method, backend="torch",
            )
            for i, mi in enumerate(m.tolist()):
                one = {kk: vv[i : i + 1] for kk, vv in inp.items()}
                ref = _sequential_reference(one, s0[i : i + 1], mi)
                self.assertLess(
                    maxdiff(pool[i : i + 1], ref), ATOL_FP32,
                    f"{method}: row {i} (m={mi})",
                )

    def test_verify_alone_does_not_commit(self):
        b = 2
        inp = make_inputs(b, self.N, seed=35)
        s0 = make_state(b, seed=36)
        pool = s0.clone()
        slots = torch.arange(b, dtype=torch.int32)
        out = api.gdn_verify(
            **inp, state_pool=pool, slot_ids=slots, backend="torch"
        )
        self.assertEqual(maxdiff(pool, s0), 0.0)
        o_ref, _ = torch_ops.recurrent_gdn(**inp, initial_state=s0.clone())
        self.assertLess(maxdiff(out, o_ref), ATOL_FP32)

    def test_verify_then_commit_pair(self):
        b = 2
        inp = make_inputs(b, self.N, seed=37)
        s0 = make_state(b, seed=38)
        slots = torch.arange(b, dtype=torch.int32)
        for m_val in range(self.N + 1):
            pool = s0.clone()
            api.gdn_verify(**inp, state_pool=pool, slot_ids=slots, backend="torch")
            api.gdn_commit(
                **inp, state_pool=pool, slot_ids=slots,
                m=torch.full((b,), m_val, dtype=torch.int32), backend="torch",
            )
            ref = _sequential_reference(inp, s0, m_val)
            self.assertLess(maxdiff(pool, ref), ATOL_FP32, f"m={m_val}")

    def test_conv_verify_and_commit(self):
        c, w, n = 64, shapes.CONV_KERNEL, self.N
        b = 2
        x = torch.randn(b, c, n)
        weight = torch.randn(c, w) * 0.1
        pool0 = torch.randn(3, c, w - 1)
        slots = torch.tensor([2, 0], dtype=torch.int32)

        ref_full = oracle_conv1d(x, pool0[slots.long()].clone(), weight)
        for m_val in range(n + 1):
            pool = pool0.clone()
            got = api.causal_conv_verify_and_commit(
                x, pool, slots, weight,
                torch.full((b,), m_val, dtype=torch.int32),
            )
            self.assertLess(maxdiff(got, ref_full), ATOL_FP32)
            # committed ring == running the first m tokens one at a time
            ref_pool = pool0.clone()
            for t in range(m_val):
                api.causal_conv_update(
                    x[:, :, t], ref_pool, slots, weight, backend="torch"
                )
            self.assertLess(
                maxdiff(pool[slots.long()], ref_pool[slots.long()]),
                ATOL_FP32, f"m={m_val}",
            )


# =========================================================================== #
# CPU: dispatch
# =========================================================================== #
class TestBackendDispatch(unittest.TestCase):
    def test_available_backends_reports_torch(self):
        av = api.available_backends()
        self.assertTrue(av["torch"])
        self.assertIn("fla", av)
        self.assertIn("triton", av)

    def test_unknown_backend_raises(self):
        with self.assertRaises(ValueError):
            api.resolve_backend("cutlass", ("torch",))

    def test_auto_falls_back_to_torch_without_gpu(self):
        if HAS_CUDA:
            self.skipTest("GPU present")
        self.assertEqual(api.resolve_backend("auto", ("triton", "fla", "torch")), "torch")

    def test_explicit_missing_backend_raises(self):
        if triton_ops.is_available():
            self.skipTest("triton available")
        with self.assertRaises(RuntimeError):
            api.resolve_backend("triton", ("triton", "torch"))

    def test_set_default_backend_round_trip(self):
        old = api.get_default_backend()
        try:
            api.set_default_backend("torch")
            self.assertEqual(api.resolve_backend("auto", ("triton", "torch")), "torch")
        finally:
            api.set_default_backend(old)


# =========================================================================== #
# GPU tiers
# =========================================================================== #
@unittest.skipUnless(HAS_CUDA, "needs CUDA")
class TestGpuBackendsVsOracle(unittest.TestCase):
    DEV = "cuda"

    def _oracle(self, inp, s0):
        return oracle_recurrent(
            expand(inp["q"]), expand(inp["k"]), inp["v"], inp["g"], inp["beta"],
            initial_state=s0.clone(), output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )

    def _decode_case(self, backend, pool_dtype, atol):
        b = 8
        inp = make_inputs(b, 1, device=self.DEV, seed=41)
        s0 = make_state(b, device=self.DEV, seed=42)
        o_ref, s_ref = self._oracle(inp, s0)

        pool = torch.zeros(16, HV, DK, DV, device=self.DEV, dtype=pool_dtype)
        slots = torch.tensor(
            [15, 0, 7, 3, 11, 1, 9, 5], dtype=torch.int32, device=self.DEV
        )
        pool[slots.long()] = s0.to(pool_dtype)
        o = api.gdn_decode_step(
            **inp, state_pool=pool, slot_ids=slots, backend=backend
        )
        self.assertLess(maxdiff(o, o_ref), atol, f"{backend}/{pool_dtype} out")
        self.assertLess(
            maxdiff(pool[slots.long()], s_ref), atol, f"{backend}/{pool_dtype} state"
        )
        untouched = [i for i in range(16) if i not in slots.tolist()]
        self.assertEqual(pool[untouched].abs().max().item(), 0.0)

    @unittest.skipUnless(HAS_CUDA and fla_ops.is_available(), "needs fla")
    def test_fla_decode_fp32_state(self):
        self._decode_case("fla", torch.float32, ATOL_FP32)

    @unittest.skipUnless(HAS_CUDA and fla_ops.is_available(), "needs fla")
    def test_fla_decode_fp16_state(self):
        self._decode_case("fla", torch.float16, ATOL_FP16_STATE)

    @unittest.skipUnless(HAS_CUDA, "needs triton")
    def test_triton_decode_fp32_state(self):
        if not triton_ops.is_available():
            self.skipTest(triton_ops.unavailable_reason() or "no triton")
        self._decode_case("triton", torch.float32, ATOL_FP32)

    @unittest.skipUnless(HAS_CUDA, "needs triton")
    def test_triton_decode_fp16_state(self):
        if not triton_ops.is_available():
            self.skipTest(triton_ops.unavailable_reason() or "no triton")
        self._decode_case("triton", torch.float16, ATOL_FP16_STATE)

    @unittest.skipUnless(HAS_CUDA, "needs triton")
    def test_triton_decode_bf16_state(self):
        if not triton_ops.is_available():
            self.skipTest(triton_ops.unavailable_reason() or "no triton")
        self._decode_case("triton", torch.bfloat16, ATOL_BF16_STATE)

    @unittest.skipUnless(HAS_CUDA, "needs triton")
    def test_triton_decode_bv64_tuning(self):
        """The dv tiling knob must not change the answer."""
        if not triton_ops.is_available():
            self.skipTest(triton_ops.unavailable_reason() or "no triton")
        old = dict(triton_ops.DECODE_TUNING)
        try:
            triton_ops.DECODE_TUNING.update(BV=64, num_warps=4)
            self._decode_case("triton", torch.float32, ATOL_FP32)
        finally:
            triton_ops.DECODE_TUNING.update(old)

    @unittest.skipUnless(HAS_CUDA, "needs triton")
    def test_triton_negative_slot_is_skipped(self):
        if not triton_ops.is_available():
            self.skipTest(triton_ops.unavailable_reason() or "no triton")
        b = 3
        inp = make_inputs(b, 1, device=self.DEV, seed=43)
        pool = torch.randn(4, HV, DK, DV, device=self.DEV) * 0.02
        before = pool.clone()
        slots = torch.tensor([1, -1, 3], dtype=torch.int32, device=self.DEV)
        api.gdn_decode_step(
            **inp, state_pool=pool, slot_ids=slots, backend="triton"
        )
        self.assertEqual(maxdiff(pool[0], before[0]), 0.0)
        self.assertEqual(maxdiff(pool[2], before[2]), 0.0)
        self.assertGreater(maxdiff(pool[1], before[1]), 0.0)

    @unittest.skipUnless(HAS_CUDA, "needs triton")
    def test_triton_multi_token_matches_oracle(self):
        if not triton_ops.is_available():
            self.skipTest(triton_ops.unavailable_reason() or "no triton")
        b, t = 4, 5
        inp = make_inputs(b, t, device=self.DEV, seed=44)
        s0 = make_state(b, device=self.DEV, seed=45)
        o_ref, s_ref = self._oracle(inp, s0)
        pool = s0.clone()
        slots = torch.arange(b, dtype=torch.int32, device=self.DEV)
        o = api.gdn_decode_multi(
            **inp, state_pool=pool, slot_ids=slots, backend="triton"
        )
        self.assertLess(maxdiff(o, o_ref), ATOL_FP32)
        self.assertLess(maxdiff(pool, s_ref), ATOL_FP32)

    @unittest.skipUnless(HAS_CUDA, "needs triton")
    def test_triton_gate_in_kernel_matches_precomputed(self):
        if not triton_ops.is_available():
            self.skipTest(triton_ops.unavailable_reason() or "no triton")
        b, dev = 6, self.DEV
        inp = make_inputs(b, 1, device=dev, seed=76)
        gen = torch.Generator(device="cpu").manual_seed(77)
        a_raw = torch.randn(b, 1, HV, generator=gen).to(dev)
        b_raw = torch.randn(b, 1, HV, generator=gen).to(dev)
        A_log = torch.log(torch.empty(HV).uniform_(0.01, 16, generator=gen)).to(dev)
        dt_bias = torch.ones(HV, device=dev)
        g, beta = torch_ops.apply_gate(a_raw, b_raw, A_log, dt_bias)
        s0 = make_state(b, device=dev, seed=78)
        slots = torch.arange(b, dtype=torch.int32, device=dev)

        p1 = s0.clone()
        o1 = api.gdn_decode_step(
            inp["q"], inp["k"], inp["v"], g, beta, p1, slots, backend="triton"
        )
        p2 = s0.clone()
        o2 = api.gdn_decode_step(
            inp["q"], inp["k"], inp["v"], a_raw, b_raw, p2, slots,
            backend="triton", A_log=A_log, dt_bias=dt_bias,
        )
        self.assertLess(maxdiff(o1, o2), ATOL_FP32)
        self.assertLess(maxdiff(p1, p2), ATOL_FP32)

    @unittest.skipUnless(HAS_CUDA, "needs triton")
    def test_triton_prenorm_matches_in_kernel(self):
        """PRENORM drops 3 of the kernel's 5 cross-thread reductions; it must
        not change the answer."""
        if not triton_ops.is_available():
            self.skipTest(triton_ops.unavailable_reason() or "no triton")
        b, dev = 8, self.DEV
        inp = make_inputs(b, 1, device=dev, seed=121)
        s0 = make_state(b, device=dev, seed=122)
        slots = torch.arange(b, dtype=torch.int32, device=dev)
        qn, kn, qk = torch_ops.prenormalize_qk(inp["q"], inp["k"])

        p1 = s0.clone()
        o1 = api.gdn_decode_step(**inp, state_pool=p1, slot_ids=slots, backend="triton")
        p2 = s0.clone()
        o2 = api.gdn_decode_step(
            qn, kn, inp["v"], inp["g"], inp["beta"], p2, slots,
            backend="triton", qk=qk,
        )
        self.assertLess(maxdiff(o1, o2), ATOL_FP32)
        self.assertLess(maxdiff(p1, p2), ATOL_FP32)
        # and against the oracle
        o_ref, s_ref = self._oracle(inp, s0)
        self.assertLess(maxdiff(o2, o_ref), ATOL_FP32)
        self.assertLess(maxdiff(p2, s_ref), ATOL_FP32)

    @unittest.skipUnless(HAS_CUDA, "needs triton")
    def test_triton_prenorm_window_matches(self):
        if not triton_ops.is_available():
            self.skipTest(triton_ops.unavailable_reason() or "no triton")
        b, n, dev = 4, 4, self.DEV
        inp = make_inputs(b, n, device=dev, seed=123)
        s0 = make_state(b, device=dev, seed=124)
        slots = torch.arange(b, dtype=torch.int32, device=dev)
        m = torch.tensor([0, 1, 3, 4], dtype=torch.int32, device=dev)
        qn, kn, qk = torch_ops.prenormalize_qk(inp["q"], inp["k"])
        for method in ("fused", "two_phase"):
            p1, p2 = s0.clone(), s0.clone()
            o1 = api.gdn_verify_and_commit(
                **inp, state_pool=p1, slot_ids=slots, m=m,
                method=method, backend="triton",
            )
            o2 = api.gdn_verify_and_commit(
                qn, kn, inp["v"], inp["g"], inp["beta"], p2, slots, m,
                method=method, backend="triton", qk=qk,
            )
            self.assertLess(maxdiff(o1, o2), ATOL_FP32, method)
            self.assertLess(maxdiff(p1, p2), ATOL_FP32, method)

    @unittest.skipUnless(HAS_CUDA, "needs triton")
    def test_triton_conv_width_major_matches_channel_major(self):
        if not triton_ops.is_available():
            self.skipTest(triton_ops.unavailable_reason() or "no triton")
        c, w, b, dev = shapes.CONV_DIM, shapes.CONV_KERNEL, 5, self.DEV
        x = torch.randn(b, c, device=dev, dtype=torch.bfloat16)
        weight = (torch.randn(c, w, device=dev) * 0.1).to(torch.bfloat16)
        cw = torch.randn(8, c, w - 1, device=dev, dtype=torch.bfloat16)
        wc = cw.transpose(1, 2).contiguous()
        slots = torch.tensor([7, 0, 4, 2, 6], dtype=torch.int32, device=dev)

        a, bb = cw.clone(), wc.clone()
        y1 = api.causal_conv_update(x, a, slots, weight, backend="triton")
        y2 = api.causal_conv_update(x, bb, slots, weight, backend="triton")
        self.assertEqual(maxdiff(y1, y2), 0.0)
        self.assertEqual(maxdiff(a, bb.transpose(1, 2)), 0.0)
        # width-major weight (stride(0)==1) must not change the answer either
        y3 = api.causal_conv_update(
            x, wc.clone(), slots, st.prepare_conv_weight(weight), backend="triton"
        )
        self.assertEqual(maxdiff(y3, y1), 0.0)

    @unittest.skipUnless(HAS_CUDA, "needs triton")
    def test_triton_conv_update_matches_torch(self):
        if not triton_ops.is_available():
            self.skipTest(triton_ops.unavailable_reason() or "no triton")
        c, w, b = shapes.CONV_DIM, shapes.CONV_KERNEL, 5
        x = torch.randn(b, c, device=self.DEV, dtype=torch.bfloat16)
        weight = (torch.randn(c, w, device=self.DEV) * 0.1).to(torch.bfloat16)
        pool = torch.randn(8, c, w - 1, device=self.DEV, dtype=torch.bfloat16)
        slots = torch.tensor([7, 0, 4, 2, 6], dtype=torch.int32, device=self.DEV)

        pool_a, pool_b = pool.clone(), pool.clone()
        o_t = api.causal_conv_update(x, pool_a, slots, weight, backend="torch")
        o_k = api.causal_conv_update(x, pool_b, slots, weight, backend="triton")
        self.assertLess(maxdiff(o_k, o_t), 1e-2)  # bf16 activations
        self.assertEqual(maxdiff(pool_b, pool_a), 0.0)


@unittest.skipUnless(HAS_CUDA, "needs CUDA")
class TestGpuStateDtypeDrift(unittest.TestCase):
    """How far does an fp16 state pool drift from fp32 over a long generation?

    This is the number the risk register's fp16 gate is really about: a single
    step rounds at ~2^-11, but the state is *recurrent*, so the question is
    whether that accumulates or is damped.  It should be damped — the decay
    ``exp(g) < 1`` contracts old error every step and the delta update is a
    correction toward ``v`` — but "should" is not a measurement.

    Steps default to 2048 (~a full generation); override with
    ``QWENFAST_GDN_DRIFT_STEPS`` to make CI cheaper.
    """

    STEPS = int(os.environ.get("QWENFAST_GDN_DRIFT_STEPS", "2048"))
    N_INPUTS = 64

    def _run(self, backend):
        if backend == "triton" and not triton_ops.is_available():
            self.skipTest(triton_ops.unavailable_reason() or "no triton")
        b = 4
        dev = "cuda"
        pool32 = make_state(b, device=dev, seed=81)
        pool16 = pool32.to(torch.float16)
        pool32 = pool32.clone()
        slots = torch.arange(b, dtype=torch.int32, device=dev)
        bank = [make_inputs(b, 1, device=dev, seed=200 + i) for i in range(self.N_INPUTS)]

        worst_o = 0.0
        sum_o = 0.0
        for t in range(self.STEPS):
            inp = bank[t % self.N_INPUTS]
            o32 = api.gdn_decode_step(
                **inp, state_pool=pool32, slot_ids=slots, backend=backend
            )
            o16 = api.gdn_decode_step(
                **inp, state_pool=pool16, slot_ids=slots, backend=backend
            )
            d = maxdiff(o32, o16)
            worst_o = max(worst_o, d)
            sum_o += d

        ds = (pool32 - pool16.float()).abs()
        scale = pool32.abs().max().item()
        print(
            f"\n  [fp16 drift, {backend}, {self.STEPS} steps]"
            f" state: max={ds.max().item():.3e} mean={ds.mean().item():.3e}"
            f" (|S|max={scale:.3f}, rel={ds.max().item()/max(scale,1e-9):.3e})"
            f" | out: max={worst_o:.3e} mean={sum_o/self.STEPS:.3e}"
        )
        # the recurrence must not *amplify* fp16 rounding: relative state error
        # has to stay within a small multiple of fp16 ULP (2^-11 = 4.9e-4).
        self.assertLess(
            ds.max().item(), 0.05 * max(scale, 1e-6),
            "fp16 state drift exceeded 5% of the state magnitude — the "
            "recurrence is amplifying rounding, not damping it",
        )

    def test_drift_triton(self):
        self._run("triton")

    @unittest.skipUnless(HAS_CUDA and fla_ops.is_available(), "needs fla")
    def test_drift_fla(self):
        self._run("fla")


@unittest.skipUnless(HAS_CUDA, "needs CUDA")
class TestGpuCudaGraph(unittest.TestCase):
    """The whole decode step is captured once and replayed.  The Triton
    path must therefore contain no host sync and no per-step allocation."""

    def test_decode_step_captures_and_replays(self):
        if not triton_ops.is_available():
            self.skipTest(triton_ops.unavailable_reason() or "no triton")
        b, dev = 8, "cuda"
        inp = make_inputs(b, 1, device=dev, seed=91)
        pool = make_state(b, device=dev, seed=92)
        slots = torch.arange(b, dtype=torch.int32, device=dev)
        out = torch.empty(b, 1, HV, DV, device=dev)

        def step():
            api.gdn_decode_step(
                **inp, state_pool=pool, slot_ids=slots, backend="triton", out=out
            )

        # eager ground truth for 3 steps from a fresh copy
        ref_pool = pool.clone()
        outs = []
        for _ in range(3):
            api.gdn_decode_step(
                **inp, state_pool=ref_pool, slot_ids=slots, backend="triton", out=out
            )
            outs.append(out.clone())
        pool_after_3 = ref_pool.clone()

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            step()
        torch.cuda.current_stream().wait_stream(side)

        g = torch.cuda.CUDAGraph()
        pool.copy_(make_state(b, device=dev, seed=92))
        with torch.cuda.graph(g):
            step()
        # capture itself ran no real work; reset and replay 3 times
        pool.copy_(make_state(b, device=dev, seed=92))
        for i in range(3):
            g.replay()
            torch.cuda.synchronize()
            self.assertLess(maxdiff(out, outs[i]), ATOL_FP32, f"replay {i}")
        self.assertLess(maxdiff(pool, pool_after_3), ATOL_FP32)

    def test_verify_and_commit_fused_captures(self):
        if not triton_ops.is_available():
            self.skipTest(triton_ops.unavailable_reason() or "no triton")
        b, n, dev = 4, 4, "cuda"
        inp = make_inputs(b, n, device=dev, seed=93)
        pool = make_state(b, device=dev, seed=94)
        slots = torch.arange(b, dtype=torch.int32, device=dev)
        m = torch.full((b,), 2, dtype=torch.int32, device=dev)
        out = torch.empty(b, n, HV, DV, device=dev)

        def step():
            triton_ops.window(
                **inp, state_pool=pool, slot_ids=slots, m=m, commit=True, out=out
            )

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            step()
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            step()

        # m is read on the device, so changing it between replays must change
        # the committed state without any recapture
        s0 = make_state(b, device=dev, seed=94)
        for m_val in (0, 1, 3):
            pool.copy_(s0)
            m.fill_(m_val)
            g.replay()
            torch.cuda.synchronize()
            ref = _sequential_reference(inp, s0, m_val, "triton")
            self.assertLess(maxdiff(pool, ref), ATOL_FP32, f"graph replay m={m_val}")


@unittest.skipUnless(HAS_CUDA, "needs CUDA")
class TestGpuPrefill(unittest.TestCase):
    DEV = "cuda"

    @unittest.skipUnless(HAS_CUDA and fla_ops.is_available(), "needs fla")
    def test_fla_chunk_matches_oracle(self):
        inp = make_inputs(2, 130, device=self.DEV, seed=51, dtype=torch.float32)
        s0 = make_state(2, device=self.DEV, seed=52)
        o_ref, s_ref = oracle_chunk(
            expand(inp["q"]), expand(inp["k"]), inp["v"], inp["g"], inp["beta"],
            chunk_size=64, initial_state=s0.clone(), output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        o, s = api.gdn_prefill_chunked(
            **inp, initial_state=s0.clone(), backend="fla"
        )
        d_o, d_s = maxdiff(o, o_ref), maxdiff(s, s_ref)
        print(f"\n  [fla chunk vs oracle] out={d_o:.2e} (|o|={o_ref.abs().max():.3f})"
              f" state={d_s:.2e} (|s|={s_ref.abs().max():.3f})")
        self.assertLess(d_o, scaled_tol(o_ref, rtol=RTOL_CHUNK_TF32))
        self.assertLess(d_s, scaled_tol(s_ref, rtol=RTOL_CHUNK_TF32))

    @unittest.skipUnless(HAS_CUDA and fla_ops.is_available(), "needs fla")
    def test_fla_varlen_matches_per_sequence(self):
        lens = [37, 64, 3, 90]
        cu = torch.tensor(
            [0] + list(torch.tensor(lens).cumsum(0)),
            dtype=torch.int32, device=self.DEV,
        )
        total = int(cu[-1])
        inp = make_inputs(1, total, device=self.DEV, seed=53)
        s0 = make_state(len(lens), device=self.DEV, seed=54)
        o_pack, s_pack = api.gdn_prefill_chunked(
            **inp, cu_seqlens=cu, initial_state=s0.clone(), backend="fla"
        )
        for i in range(len(lens)):
            lo, hi = int(cu[i]), int(cu[i + 1])
            one = {kk: vv[:, lo:hi] for kk, vv in inp.items()}
            o_i, s_i = api.gdn_prefill_chunked(
                **one, initial_state=s0[i : i + 1].clone(), backend="fla"
            )
            self.assertLess(
                maxdiff(o_pack[:, lo:hi], o_i), scaled_tol(o_i, rtol=RTOL_CHUNK_TF32)
            )
            self.assertLess(
                maxdiff(s_pack[i : i + 1], s_i), scaled_tol(s_i, rtol=RTOL_CHUNK_TF32)
            )

    @unittest.skipUnless(HAS_CUDA and fla_ops.is_available(), "needs fla")
    def test_fla_varlen_scatters_into_pool(self):
        cu = torch.tensor([0, 20, 60], dtype=torch.int32, device=self.DEV)
        inp = make_inputs(1, 60, device=self.DEV, seed=55)
        pool = torch.zeros(5, HV, DK, DV, device=self.DEV)
        slots = torch.tensor([4, 2], dtype=torch.int32, device=self.DEV)
        _, final = api.gdn_prefill_chunked(
            **inp, cu_seqlens=cu, state_pool=pool, slot_ids=slots, backend="fla"
        )
        self.assertLess(maxdiff(pool[slots.long()], final), ATOL_FP32)  # exact copy
        self.assertEqual(pool[0].abs().max().item(), 0.0)


@unittest.skipUnless(HAS_CUDA, "needs CUDA")
class TestGpuVerifyAndCommit(TestVerifyAndCommit):
    @unittest.skipUnless(HAS_CUDA and fla_ops.is_available(), "needs fla")
    def test_fla_two_phase(self):
        self._check("fla", "two_phase", ATOL_FP32, device="cuda")

    def test_triton_fused(self):
        if not triton_ops.is_available():
            self.skipTest(triton_ops.unavailable_reason() or "no triton")
        self._check("triton", "fused", ATOL_FP32, device="cuda")

    def test_triton_two_phase(self):
        if not triton_ops.is_available():
            self.skipTest(triton_ops.unavailable_reason() or "no triton")
        self._check("triton", "two_phase", ATOL_FP32, device="cuda")

    def test_triton_fused_fp16_state(self):
        if not triton_ops.is_available():
            self.skipTest(triton_ops.unavailable_reason() or "no triton")
        b, n = 3, self.N
        inp = make_inputs(b, n, device="cuda", seed=61)
        s0 = make_state(b, device="cuda", seed=62)
        slots = torch.arange(b, dtype=torch.int32, device="cuda")
        for m_val in range(n + 1):
            pool16 = s0.to(torch.float16)
            m = torch.full((b,), m_val, dtype=torch.int32, device="cuda")
            api.gdn_verify_and_commit(
                **inp, state_pool=pool16, slot_ids=slots, m=m,
                method="fused", backend="triton",
            )
            ref = _sequential_reference(inp, s0.to(torch.float16), m_val, "torch")
            self.assertLess(
                maxdiff(pool16, ref), ATOL_FP16_STATE, f"fp16 fused m={m_val}"
            )


# =========================================================================== #
# GPU: packed / hoisted / scheduled variants
# =========================================================================== #
@unittest.skipUnless(HAS_CUDA, "needs CUDA")
class TestGpuPackedDecode(unittest.TestCase):
    """The packed 32-bit state path.

    A 16-bit pool is addressed through an ``int32`` view so each instruction
    moves two elements; the pair is split into even/odd fp32 register tiles and
    re-packed on the way out.  Two ways for that to be wrong that no timing run
    would catch:

    * the halves swapped — every value column ``2j`` gets column ``2j+1``'s
      state.  Caught by any oracle comparison, since the columns hold
      unrelated values.
    * the pack rounding changed (truncate instead of round-to-nearest), which
      would double the fp16 drift while still passing a 2e-3 tolerance for one
      step.  Caught by the head-to-head against the unpacked kernel, which is
      held to a *fp16-ulp* bound rather than the loose oracle bound.

    B=512 is in the sweep because the packed path changes the register tile
    shape ([NK, BV] -> 2 x [NK, BV/2]), and register pressure is what decides
    whether the winning tiling spills.
    """

    DEV = "cuda"
    #: override on a busy GPU: QWENFAST_GDN_PACK_BATCHES=1,64
    BATCHES = tuple(
        int(x) for x in os.environ.get(
            "QWENFAST_GDN_PACK_BATCHES", "1,64,512"
        ).split(",") if x
    )
    PACKED_VARIANTS = ("packed", "packed_hoist", "packed_hoist_sched")

    def setUp(self):
        if not triton_ops.is_available():
            self.skipTest(triton_ops.unavailable_reason() or "no triton")

    def tearDown(self):
        torch.cuda.empty_cache()

    def _oracle(self, inp, s0):
        return oracle_recurrent(
            expand(inp["q"]), expand(inp["k"]), inp["v"], inp["g"], inp["beta"],
            initial_state=s0.clone(), output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )

    def _run(self, b, pool_dtype, variant, seed=301):
        """-> (out, pool) after one decode step on the pinned variant."""
        inp = make_inputs(b, 1, device=self.DEV, seed=seed)
        s0 = make_state_on(b, self.DEV, seed=seed + 1)
        pool = s0.to(pool_dtype).clone()
        slots = torch.arange(b, dtype=torch.int32, device=self.DEV)
        with pinned_variant(triton_ops.DECODE_TUNING, variant):
            o = api.gdn_decode_step(
                **inp, state_pool=pool, slot_ids=slots, backend="triton"
            )
            eff = dict(triton_ops.LAST_VARIANT["decode"])
        return inp, s0, o, pool, eff

    def _check_vs_oracle(self, pool_dtype, atol, expect_pack):
        for b in self.BATCHES:
            for variant in self.PACKED_VARIANTS:
                with self.subTest(B=b, dtype=str(pool_dtype), variant=variant):
                    inp, s0, o, pool, eff = self._run(b, pool_dtype, variant)
                    self.assertEqual(
                        eff["PACK_DT"], expect_pack,
                        "the packed path did not engage — the wrapper degraded "
                        "silently and this test is measuring nothing",
                    )
                    o_ref, s_ref = self._oracle(inp, s0)
                    self.assertLess(maxdiff(o, o_ref), atol, "out")
                    self.assertLess(maxdiff(pool.float(), s_ref), atol, "state")
                    del inp, s0, o, pool, o_ref, s_ref
                    torch.cuda.empty_cache()

    def test_packed_fp16_matches_oracle(self):
        self._check_vs_oracle(torch.float16, ATOL_FP16_STATE, 1)

    def test_packed_bf16_matches_oracle(self):
        self._check_vs_oracle(torch.bfloat16, ATOL_BF16_STATE, 2)

    def test_packed_matches_unpacked_within_one_ulp(self):
        """Head-to-head against ``base``: same fp32 math, different memory
        path, so they may differ only by reduction re-association (~1e-6 in the
        output) and at most one storage ulp in the state.  A pack/unpack bug
        lands far outside that; the loose oracle tolerance would hide it."""
        for b in self.BATCHES:
            for pool_dtype, ulp in (
                (torch.float16, 2 ** -10), (torch.bfloat16, 2 ** -7)
            ):
                for variant in self.PACKED_VARIANTS:
                    with self.subTest(B=b, dtype=str(pool_dtype), variant=variant):
                        _, _, o0, p0, _ = self._run(b, pool_dtype, "base")
                        _, _, o1, p1, eff = self._run(b, pool_dtype, variant)
                        self.assertGreater(eff["PACK_DT"], 0)
                        self.assertLess(maxdiff(o0, o1), 1e-4, "out vs base")
                        scale = p0.float().abs().max().item()
                        self.assertLessEqual(
                            maxdiff(p0.float(), p1.float()), ulp * max(scale, 1e-6),
                            "state differs by more than one storage ulp — the "
                            "pack is not round-to-nearest, or the halves are "
                            "swapped",
                        )
                        del o0, p0, o1, p1
                        torch.cuda.empty_cache()

    def test_every_variant_agrees_with_base(self):
        """All 8 combinations, both 16-bit dtypes and fp32, at one batch."""
        b = 64
        for pool_dtype, atol in (
            (torch.float32, ATOL_FP32),
            (torch.float16, ATOL_FP16_STATE),
            (torch.bfloat16, ATOL_BF16_STATE),
        ):
            _, _, o0, p0, _ = self._run(b, pool_dtype, "base")
            for variant in triton_ops.all_variants():
                with self.subTest(dtype=str(pool_dtype), variant=variant):
                    _, _, o1, p1, eff = self._run(b, pool_dtype, variant)
                    self.assertLess(maxdiff(o0, o1), atol, "out")
                    self.assertLess(maxdiff(p0.float(), p1.float()), atol, "state")
                    # packed must engage iff it was asked for *and* applies
                    want = pool_dtype is not torch.float32 and "packed" in variant
                    self.assertEqual(bool(eff["PACK_DT"]), want)
                    del o1, p1
            del o0, p0
            torch.cuda.empty_cache()

    def test_baked_variants_are_bit_identical_to_base(self):
        """The gate on adopting a variant.

        A variant is only allowed into `DECODE_VARIANT_TABLE` if it is
        **exactly** equal to `base` — not "within tolerance".  `packed`
        re-expresses the memory access and `sched` only re-orders the issue;
        neither touches the arithmetic, so the state and the output must match
        to the last bit, and this asserts equality rather than a bound.  If a
        future variant needs a tolerance here, it is a numerics change and it
        needs the eval gate, not a kernel review.

        Run over several steps so a difference has somewhere to accumulate: the
        recurrence would amplify a one-ulp disagreement into a visible one.
        """
        baked = {
            v for rows in triton_ops.DECODE_VARIANT_TABLE.values()
            for _, v in rows
        } | {
            v for rows in triton_ops.WINDOW_VARIANT_TABLE.values()
            for _, v in rows
        }
        baked.discard("base")
        if not baked:
            self.skipTest("nothing but `base` is baked — nothing to gate")
        # B=16 deliberately: it resolves to the *same* (BV, num_warps) = (64, 2)
        # tiling as the B>=512 rows where the variant is actually baked, and the
        # reduction tree — the only place a re-association could creep in —
        # depends on that tiling, not on the batch.
        b, dev, steps = 16, self.DEV, 32
        bank = [make_inputs(b, 1, device=dev, seed=400 + i) for i in range(8)]
        slots = torch.arange(b, dtype=torch.int32, device=dev)
        for pool_dtype in (torch.float16, torch.bfloat16):
            for variant in sorted(baked):
                with self.subTest(dtype=str(pool_dtype), variant=variant):
                    s0 = make_state_on(b, dev, seed=401, dtype=pool_dtype)
                    p_base, p_var = s0.clone(), s0.clone()
                    for t in range(steps):
                        inp = bank[t % len(bank)]
                        with pinned_variant(triton_ops.DECODE_TUNING, "base"):
                            o0 = api.gdn_decode_step(
                                **inp, state_pool=p_base, slot_ids=slots,
                                backend="triton",
                            )
                        with pinned_variant(triton_ops.DECODE_TUNING, variant):
                            o1 = api.gdn_decode_step(
                                **inp, state_pool=p_var, slot_ids=slots,
                                backend="triton",
                            )
                        self.assertEqual(
                            maxdiff(o0, o1), 0.0,
                            f"output differs at step {t} — {variant} is not "
                            "bit-identical to base and must not be baked",
                        )
                    self.assertEqual(
                        maxdiff(p_base.float(), p_var.float()), 0.0,
                        f"state differs after {steps} steps — {variant} is not "
                        "bit-identical to base and must not be baked",
                    )

    def test_packed_on_fp32_pool_degrades_to_base(self):
        """`packed` is meaningless for a 4-byte pool; the wrapper must run the
        unpacked kernel rather than raise inside the decode path."""
        _, _, o, pool, eff = self._run(8, torch.float32, "packed_hoist")
        self.assertEqual(eff["PACK_DT"], 0)
        self.assertTrue(eff["HOIST"])

    def test_packed_negative_slot_is_skipped(self):
        """`hoist` replaces the per-element state mask with a uniform branch
        around the store, so the negative-slot safety net has to be re-proved
        on that path (and on the packed one, which stores int32 words)."""
        b = 3
        for variant in ("packed", "packed_hoist", "hoist", "packed_hoist_sched"):
            with self.subTest(variant=variant):
                inp = make_inputs(b, 1, device=self.DEV, seed=311)
                pool = make_state_on(4, self.DEV, seed=312, dtype=torch.float16)
                before = pool.clone()
                slots = torch.tensor([1, -1, 3], dtype=torch.int32, device=self.DEV)
                with pinned_variant(triton_ops.DECODE_TUNING, variant):
                    api.gdn_decode_step(
                        **inp, state_pool=pool, slot_ids=slots, backend="triton"
                    )
                self.assertEqual(maxdiff(pool[0], before[0]), 0.0, "slot 0 touched")
                self.assertEqual(maxdiff(pool[2], before[2]), 0.0, "slot 2 touched")
                self.assertGreater(maxdiff(pool[1], before[1]), 0.0, "slot 1 not written")

    def test_packed_with_a_masked_value_tail(self):
        """``NV % BV != 0``: the wrapper must keep the element mask (NOMASK
        off) and the packed tail must still address ``2j`` / ``2j+1``
        correctly.  Unreachable at the model shape (128 % any power of two is
        0), so it is constructed here."""
        b, hv, hk, dk, dv = 2, 3, 1, 128, 96
        inp = make_inputs(b, 1, device=self.DEV, seed=321, hv=hv, hk=hk, dk=dk, dv=dv)
        s0 = make_state_on(b, self.DEV, seed=322, hv=hv, dk=dk, dv=dv)
        slots = torch.arange(b, dtype=torch.int32, device=self.DEV)
        ref_pool = s0.clone()
        o_ref = api.gdn_decode_step(
            **inp, state_pool=ref_pool, slot_ids=slots, backend="torch"
        )
        pool = s0.to(torch.float16).clone()
        with pinned_variant(triton_ops.DECODE_TUNING, "packed_hoist"):
            old = dict(triton_ops.DECODE_TUNING)
            try:
                triton_ops.DECODE_TUNING.update(BV=64, num_warps=2)
                o = api.gdn_decode_step(
                    **inp, state_pool=pool, slot_ids=slots, backend="triton"
                )
                eff = dict(triton_ops.LAST_VARIANT["decode"])
            finally:
                triton_ops.DECODE_TUNING.update(old)
        self.assertEqual(eff["PACK_DT"], 1)
        self.assertFalse(eff["NOMASK"], "NOMASK must stay off when NV % BV != 0")
        self.assertLess(maxdiff(o, o_ref), ATOL_FP16_STATE)
        self.assertLess(maxdiff(pool.float(), ref_pool), ATOL_FP16_STATE)

    def test_packed_graph_captures_and_replays(self):
        """The packed wrapper takes an extra ``Tensor.view(int32)`` on
        the host.  That must happen at capture time and leave the replay a pure
        kernel launch."""
        b = 8
        inp = make_inputs(b, 1, device=self.DEV, seed=331)
        pool = make_state_on(b, self.DEV, seed=332, dtype=torch.float16)
        slots = torch.arange(b, dtype=torch.int32, device=self.DEV)
        out = torch.empty(b, 1, HV, DV, device=self.DEV)

        def step():
            api.gdn_decode_step(
                **inp, state_pool=pool, slot_ids=slots, backend="triton", out=out
            )

        with pinned_variant(triton_ops.DECODE_TUNING, "packed_hoist_sched"):
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    step()
            torch.cuda.current_stream().wait_stream(side)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                step()
            before = pool.clone()
            g.replay()
            torch.cuda.synchronize()
        self.assertGreater(maxdiff(pool, before), 0.0, "replay did not run")
        del g


@unittest.skipUnless(HAS_CUDA, "needs CUDA")
class TestGpuPackedVerifyAndCommit(unittest.TestCase):
    """The fused verify-and-commit kernel shares the state path, so it takes
    the same flags — including ``COMMIT``'s second register tile, which under
    ``packed`` becomes two half tiles."""

    N = 4  # window = k+1

    def setUp(self):
        if not triton_ops.is_available():
            self.skipTest(triton_ops.unavailable_reason() or "no triton")

    def test_fused_packed_matches_sequential_for_every_accept_length(self):
        b, n, dev = 3, self.N, "cuda"
        inp = make_inputs(b, n, device=dev, seed=341)
        s0 = make_state_on(b, dev, seed=342)
        slots = torch.arange(b, dtype=torch.int32, device=dev)
        for pool_dtype, atol in (
            (torch.float16, ATOL_FP16_STATE), (torch.bfloat16, ATOL_BF16_STATE)
        ):
            for variant in ("packed", "packed_hoist", "packed_hoist_sched"):
                for m_val in range(n + 1):
                    with self.subTest(dtype=str(pool_dtype), variant=variant, m=m_val):
                        pool = s0.to(pool_dtype).clone()
                        m = torch.full((b,), m_val, dtype=torch.int32, device=dev)
                        with pinned_variant(triton_ops.WINDOW_TUNING, variant):
                            api.gdn_verify_and_commit(
                                **inp, state_pool=pool, slot_ids=slots, m=m,
                                method="fused", backend="triton",
                            )
                            eff = dict(triton_ops.LAST_VARIANT["window"])
                        self.assertGreater(eff["PACK_DT"], 0)
                        ref = _sequential_reference(inp, s0.to(pool_dtype), m_val, "torch")
                        self.assertLess(maxdiff(pool.float(), ref.float()), atol)

    def test_two_phase_packed_matches_fused(self):
        """The two-phase replay pass writes the state through the same packed store
        with ``MASK_PAST_M``; it must land on the same S_m as the fused kernel."""
        b, n, dev = 3, self.N, "cuda"
        inp = make_inputs(b, n, device=dev, seed=351)
        s0 = make_state_on(b, dev, seed=352)
        slots = torch.arange(b, dtype=torch.int32, device=dev)
        m = torch.tensor([0, 2, 4], dtype=torch.int32, device=dev)
        with pinned_variant(triton_ops.WINDOW_TUNING, "packed_hoist"):
            p1 = s0.to(torch.float16).clone()
            o1 = api.gdn_verify_and_commit(
                **inp, state_pool=p1, slot_ids=slots, m=m,
                method="fused", backend="triton",
            )
            p2 = s0.to(torch.float16).clone()
            o2 = api.gdn_verify_and_commit(
                **inp, state_pool=p2, slot_ids=slots, m=m,
                method="two_phase", backend="triton",
            )
        self.assertLess(maxdiff(o1, o2), ATOL_FP16_STATE)
        self.assertLess(maxdiff(p1.float(), p2.float()), ATOL_FP16_STATE)

    def test_verify_only_writes_nothing_under_every_variant(self):
        """``commit=False`` is phase A: 1 read, **0** writes.  ``hoist`` moves
        the store under a uniform branch, so re-prove it."""
        b, n, dev = 2, self.N, "cuda"
        inp = make_inputs(b, n, device=dev, seed=361)
        slots = torch.arange(b, dtype=torch.int32, device=dev)
        for variant in triton_ops.all_variants():
            with self.subTest(variant=variant):
                pool = make_state_on(b, dev, seed=362, dtype=torch.float16)
                before = pool.clone()
                with pinned_variant(triton_ops.WINDOW_TUNING, variant):
                    triton_ops.window(
                        inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"],
                        pool, slots, None, commit=False,
                    )
                self.assertEqual(maxdiff(pool, before), 0.0)


@unittest.skipUnless(HAS_CUDA, "needs CUDA")
class TestGpuPackedDrift(unittest.TestCase):
    """Does the packed path drift like the unpacked one over a generation?

    The pack narrows with round-to-nearest, exactly as ``.to(fp16)`` does, so
    it should — but "should" is not a measurement, and a truncating pack would
    bias every step in the same direction, which is the one rounding error the
    recurrence's decay does *not* damp.
    """

    STEPS = int(os.environ.get("QWENFAST_GDN_DRIFT_STEPS", "2048"))
    HEAD_TO_HEAD_STEPS = int(os.environ.get("QWENFAST_GDN_PACK_DRIFT_STEPS", "256"))
    N_INPUTS = 64
    VARIANT = "packed_hoist_sched"

    def setUp(self):
        if not triton_ops.is_available():
            self.skipTest(triton_ops.unavailable_reason() or "no triton")

    def _bank(self, b, dev):
        return [make_inputs(b, 1, device=dev, seed=200 + i) for i in range(self.N_INPUTS)]

    def test_packed_fp16_drift_over_2048_steps(self):
        b, dev = 4, "cuda"
        pool32 = make_state_on(b, dev, seed=81)
        pool16 = pool32.to(torch.float16)
        slots = torch.arange(b, dtype=torch.int32, device=dev)
        bank = self._bank(b, dev)
        worst_o = 0.0
        for t in range(self.STEPS):
            inp = bank[t % self.N_INPUTS]
            o32 = api.gdn_decode_step(
                **inp, state_pool=pool32, slot_ids=slots, backend="triton"
            )
            with pinned_variant(triton_ops.DECODE_TUNING, self.VARIANT):
                o16 = api.gdn_decode_step(
                    **inp, state_pool=pool16, slot_ids=slots, backend="triton"
                )
            worst_o = max(worst_o, maxdiff(o32, o16))
        ds = (pool32 - pool16.float()).abs()
        scale = pool32.abs().max().item()
        print(
            f"\n  [fp16 drift, packed({self.VARIANT}), {self.STEPS} steps]"
            f" state: max={ds.max().item():.3e} rel={ds.max().item()/max(scale,1e-9):.3e}"
            f" | out max={worst_o:.3e}"
        )
        self.assertLess(
            ds.max().item(), 0.05 * max(scale, 1e-6),
            "packed fp16 state drift exceeded 5% of the state magnitude — the "
            "recurrence is amplifying rounding, not damping it",
        )

    def test_packed_and_unpacked_drift_together(self):
        """Both paths, same inputs, from the same fp16 start: they must stay
        within a few storage ulps of each other for hundreds of steps.  A
        biased (truncating) pack shows up as a divergence that grows with t."""
        b, dev = 4, "cuda"
        s0 = make_state_on(b, dev, seed=91, dtype=torch.float16)
        pool_base = s0.clone()
        pool_pack = s0.clone()
        slots = torch.arange(b, dtype=torch.int32, device=dev)
        bank = self._bank(b, dev)
        for t in range(self.HEAD_TO_HEAD_STEPS):
            inp = bank[t % self.N_INPUTS]
            with pinned_variant(triton_ops.DECODE_TUNING, "base"):
                api.gdn_decode_step(
                    **inp, state_pool=pool_base, slot_ids=slots, backend="triton"
                )
            with pinned_variant(triton_ops.DECODE_TUNING, self.VARIANT):
                api.gdn_decode_step(
                    **inp, state_pool=pool_pack, slot_ids=slots, backend="triton"
                )
        d = maxdiff(pool_base.float(), pool_pack.float())
        scale = pool_base.float().abs().max().item()
        print(
            f"\n  [packed vs base, {self.HEAD_TO_HEAD_STEPS} steps]"
            f" state max={d:.3e} rel={d/max(scale,1e-9):.3e}"
        )
        self.assertLess(d, ATOL_FP16_STATE * max(scale, 1.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
