"""Tests for the attention / paged-KV subpackage.

CPU tests (always run, no GPU needed): paged pool round-trip (bf16 + fp8),
page allocator, truncate/rollback, RoPE vs the reference, QK-RMSNorm vs the
reference, and end-to-end attention (fused norm+RoPE -> paged KV -> SDPA ->
sigmoid gate -> o_proj) vs ``engine/qwenfast/model.py``'s dense ``Attention``
module -- the dense reference -- on random data, for both a varlen-prefill batch
and a decode step continuing a prefill.

GPU tests (skipped automatically when CUDA or flashinfer aren't available):
FlashInfer decode/prefill vs the torch fallback, in bf16 and fp8 KV. These
are the ones meant to run on the remote GPU host (see ``README.md``).

Run::

    python engine/qwenfast/attn/tests/test_attn.py
    # or
    python -m unittest discover -s engine/qwenfast/attn/tests -v
"""

from __future__ import annotations

import os
import sys
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))  # -> engine/

from qwenfast.model import Attention, RotaryEmbedding  # noqa: E402
from qwenfast.weights import QwenFastConfig  # noqa: E402

from qwenfast.attn.kv_pool import FP8_DTYPE, KVPoolConfig, PageAllocator, PagedKVPool  # noqa: E402
from qwenfast.attn.rope import RotaryTable, apply_rotary_pos_emb, compute_inv_freq, rotate_half  # noqa: E402
from qwenfast.attn.flashinfer_attn import (  # noqa: E402
    HAS_FLASHINFER,
    apply_output_gate,
    fused_qk_norm_rope,
    rms_norm_head,
    split_q_gate,
    torch_fallback_decode,
    torch_fallback_prefill,
)

HAS_CUDA = torch.cuda.is_available()


# =========================================================================== #
# reference copies (verbatim from engine/reference/modeling_qwen3_5.py, same
# as engine/qwenfast/tests/test_ops.py's TestRope/TestNorms references)
# =========================================================================== #
def ref_rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def ref_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (ref_rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (ref_rotate_half(k_rot) * sin)
    return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)


class RefRMSNorm(nn.Module):
    """modeling_qwen3_5.Qwen3_5RMSNorm: normalize fp32, scale (1 + weight)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        out = out * (1.0 + self.weight.float())
        return out.type_as(x)


# =========================================================================== #
# helpers
# =========================================================================== #
def attn_config(hidden_size=96, num_heads=24, num_kv_heads=4, head_dim=256) -> QwenFastConfig:
    """Full production attention shapes (24q/4kv/256d, theta=1e7, rotary 64),
    a small `hidden_size` so the q_proj/k_proj/v_proj GEMMs stay cheap on CPU."""
    return QwenFastConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        attn_output_gate=True,
        rms_norm_eps=1e-6,
        rope_theta=1e7,
        partial_rotary_factor=0.25,
    )


def build_attention_module(cfg: QwenFastConfig, seed=0) -> Attention:
    torch.manual_seed(seed)
    m = Attention(cfg, layer_idx=0).to(torch.float32)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if "norm" in name:
                p.normal_(0.0, 0.05)
            else:
                p.normal_(0.0, 0.02)
    m.eval()
    return m


def project_qkv(m: Attention, hidden: torch.Tensor):
    """Run just the projections (q_proj/k_proj/v_proj + split_q_gate), packed
    ``[T, ...]`` form -- the boundary between the GEMM layer and the
    attention stack (everything this test module is about)."""
    cfg = m.config
    qg = m.q_proj(hidden)
    q, gate = split_q_gate(qg, cfg.num_attention_heads, cfg.head_dim)
    k = m.k_proj(hidden).view(-1, cfg.num_key_value_heads, cfg.head_dim)
    v = m.v_proj(hidden).view(-1, cfg.num_key_value_heads, cfg.head_dim)
    return q, k, v, gate


# =========================================================================== #
# page allocator
# =========================================================================== #
class TestPageAllocator(unittest.TestCase):
    def test_alloc_free_roundtrip(self):
        alloc = PageAllocator(8)
        self.assertEqual(alloc.num_free, 8)
        pages = alloc.alloc(3)
        self.assertEqual(len(pages), 3)
        self.assertEqual(alloc.num_free, 5)
        self.assertEqual(len(set(pages)), 3)  # distinct pages
        alloc.free(pages)
        self.assertEqual(alloc.num_free, 8)

    def test_exhaustion_raises(self):
        alloc = PageAllocator(2)
        alloc.alloc(2)
        with self.assertRaises(RuntimeError):
            alloc.alloc(1)

    def test_alloc_zero_is_a_noop(self):
        alloc = PageAllocator(4)
        self.assertEqual(alloc.alloc(0), [])
        self.assertEqual(alloc.num_free, 4)


# =========================================================================== #
# paged KV pool round-trip
# =========================================================================== #
class TestPagedKVPool(unittest.TestCase):
    def _pool(self, dtype="bf16", page_size=4, n_pages=16, num_kv_heads=4, head_dim=256):
        cfg = KVPoolConfig(
            n_layers=2, num_kv_heads=num_kv_heads, head_dim=head_dim, page_size=page_size,
            n_pages=n_pages, max_seqs=4, max_pages_per_seq=8, dtype=dtype, device="cpu",
        )
        return PagedKVPool(cfg)

    def test_bf16_roundtrip(self):
        torch.manual_seed(0)
        pool = self._pool(dtype="bf16")
        slot = pool.alloc_slot()
        n_tok = 10  # spans 3 pages at page_size=4
        pool.ensure_capacity(slot, n_tok)
        k = torch.randn(n_tok, 4, 256, dtype=torch.bfloat16)
        v = torch.randn(n_tok, 4, 256, dtype=torch.bfloat16)
        slot_ids = torch.full((n_tok,), slot, dtype=torch.int64)
        positions = torch.arange(n_tok, dtype=torch.int64)
        pool.append_kv(0, slot_ids, positions, k, v)
        self.assertEqual(int(pool.seq_len[slot]), n_tok)

        k_out, v_out = pool.gather_dense(0, slot)
        torch.testing.assert_close(k_out, k, rtol=0, atol=0)
        torch.testing.assert_close(v_out, v, rtol=0, atol=0)

        # a different layer's slice must stay untouched (all zeros)
        k1, v1 = pool.gather_dense(1, slot)
        self.assertTrue(bool((k1 == 0).all()))
        self.assertTrue(bool((v1 == 0).all()))

    def test_fp8_roundtrip_within_quant_tolerance(self):
        if FP8_DTYPE is None:
            self.skipTest("torch build has no float8_e4m3fn")
        torch.manual_seed(1)
        pool = self._pool(dtype="fp8", page_size=8, n_pages=8)
        slot = pool.alloc_slot()
        n_tok = 8
        pool.ensure_capacity(slot, n_tok)
        # calibrate the (single) page's scale from the exact data we're about
        # to write, then the roundtrip error is bounded by e4m3's ~2-3 bit
        # mantissa, not by scale mismatch.
        k = torch.randn(n_tok, 4, 256, dtype=torch.bfloat16) * 2.0
        v = torch.randn(n_tok, 4, 256, dtype=torch.bfloat16) * 2.0
        page0 = int(pool.page_table[slot, 0])
        pool.calibrate_page_scale(0, page0, 0, k)
        pool.calibrate_page_scale(0, page0, 1, v)

        slot_ids = torch.full((n_tok,), slot, dtype=torch.int64)
        positions = torch.arange(n_tok, dtype=torch.int64)
        pool.append_kv(0, slot_ids, positions, k, v)

        k_out, v_out = pool.gather_dense(0, slot)
        # e4m3 relative error is coarse; a generous but meaningful bound.
        torch.testing.assert_close(k_out.float(), k.float(), rtol=0.1, atol=0.05)
        torch.testing.assert_close(v_out.float(), v.float(), rtol=0.1, atol=0.05)

    def test_append_without_capacity_raises(self):
        pool = self._pool()
        slot = pool.alloc_slot()
        k = torch.randn(1, 4, 256, dtype=torch.bfloat16)
        v = torch.randn(1, 4, 256, dtype=torch.bfloat16)
        with self.assertRaises(RuntimeError):
            pool.append_kv(0, torch.tensor([slot]), torch.tensor([0]), k, v)

    def test_free_slot_returns_pages(self):
        pool = self._pool(page_size=4, n_pages=8)
        slot = pool.alloc_slot()
        pool.ensure_capacity(slot, 10)  # 3 pages
        self.assertEqual(pool.num_free_pages, 5)
        pool.free_slot(slot)
        self.assertEqual(pool.num_free_pages, 8)

    def test_slot_capacity_grows_incrementally(self):
        pool = self._pool(page_size=4, n_pages=8)
        slot = pool.alloc_slot()
        pool.ensure_capacity(slot, 3)
        self.assertEqual(pool.pages_allocated(slot), 1)
        pool.ensure_capacity(slot, 5)
        self.assertEqual(pool.pages_allocated(slot), 2)
        # shrinking the request must not free anything
        pool.ensure_capacity(slot, 1)
        self.assertEqual(pool.pages_allocated(slot), 2)


class TestTruncateRollback(unittest.TestCase):
    def _filled_pool_slot(self, n_tok=9, page_size=4, seed=0):
        torch.manual_seed(seed)
        cfg = KVPoolConfig(
            n_layers=1, num_kv_heads=2, head_dim=16, page_size=page_size,
            n_pages=8, max_seqs=2, max_pages_per_seq=8, dtype="bf16", device="cpu",
        )
        pool = PagedKVPool(cfg)
        slot = pool.alloc_slot()
        pool.ensure_capacity(slot, n_tok)
        k = torch.randn(n_tok, 2, 16, dtype=torch.bfloat16)
        v = torch.randn(n_tok, 2, 16, dtype=torch.bfloat16)
        slot_ids = torch.full((n_tok,), slot, dtype=torch.int64)
        positions = torch.arange(n_tok, dtype=torch.int64)
        pool.append_kv(0, slot_ids, positions, k, v)
        return pool, slot, k, v

    def test_truncate_is_an_o1_pointer_move(self):
        pool, slot, k, v = self._filled_pool_slot(n_tok=9)
        pages_before = pool.pages_allocated(slot)
        pool.truncate(slot, 4)
        self.assertEqual(int(pool.seq_len[slot]), 4)
        # no pages freed by truncate itself (pure pointer move)
        self.assertEqual(pool.pages_allocated(slot), pages_before)
        k_out, v_out = pool.gather_dense(0, slot)
        torch.testing.assert_close(k_out, k[:4], rtol=0, atol=0)
        torch.testing.assert_close(v_out, v[:4], rtol=0, atol=0)

    def test_rewrite_after_truncate_overwrites_stale_tail(self):
        """The spec-decode use case: roll back to `m`, then append the
        accepted continuation, which must fully replace the rejected tail."""
        pool, slot, k, v = self._filled_pool_slot(n_tok=9)
        pool.truncate(slot, 4)
        new_tail = torch.randn(3, 2, 16, dtype=torch.bfloat16)
        slot_ids = torch.full((3,), slot, dtype=torch.int64)
        positions = torch.arange(4, 7, dtype=torch.int64)
        pool.append_kv(0, slot_ids, positions, new_tail, new_tail)
        self.assertEqual(int(pool.seq_len[slot]), 7)
        k_out, _ = pool.gather_dense(0, slot)
        torch.testing.assert_close(k_out[:4], k[:4], rtol=0, atol=0)
        torch.testing.assert_close(k_out[4:7], new_tail, rtol=0, atol=0)

    def test_reclaim_trailing_pages_frees_only_unused(self):
        pool, slot, _, _ = self._filled_pool_slot(n_tok=9, page_size=4)  # 3 pages
        pool.truncate(slot, 4)  # needs only 1 page now
        freed = pool.reclaim_trailing_pages(slot)
        self.assertEqual(freed, 2)
        self.assertEqual(pool.pages_allocated(slot), 1)

    def test_truncate_beyond_allocated_pages_raises(self):
        pool, slot, _, _ = self._filled_pool_slot(n_tok=9, page_size=4)
        with self.assertRaises(ValueError):
            pool.truncate(slot, 10_000)


# =========================================================================== #
# RoPE vs reference
# =========================================================================== #
class TestRope(unittest.TestCase):
    def test_rotate_half_matches_reference(self):
        x = torch.randn(2, 3, 8)
        torch.testing.assert_close(rotate_half(x), ref_rotate_half(x))

    def test_apply_rotary_pos_emb_matches_reference(self):
        torch.manual_seed(0)
        b, h, s, d, rot = 2, 4, 5, 16, 8
        q = torch.randn(b, h, s, d)
        k = torch.randn(b, h, s, d)
        inv_freq = compute_inv_freq(rot, 1e7)
        pos = torch.arange(s).float()
        freqs = pos[:, None] * inv_freq[None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().unsqueeze(0).expand(b, s, rot)
        sin = emb.sin().unsqueeze(0).expand(b, s, rot)
        q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
        q_ref, k_ref = ref_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
        torch.testing.assert_close(q_out, q_ref, rtol=0, atol=0)
        torch.testing.assert_close(k_out, k_ref, rtol=0, atol=0)
        # partial RoPE: dims >= rot must pass through untouched
        torch.testing.assert_close(q_out[..., rot:], q[..., rot:], rtol=0, atol=0)

    def test_rotary_table_matches_model_rotary_embedding(self):
        cfg = attn_config()
        model_rope = RotaryEmbedding(cfg)
        table = RotaryTable(
            max_positions=4096, rotary_dim=cfg.rotary_dim, theta=cfg.rope_theta, dtype=torch.bfloat16
        )
        positions = torch.tensor([[0, 1, 5, 100, 4095]])
        cos_ref, sin_ref = model_rope(positions, dtype=torch.bfloat16)
        cos_tab, sin_tab = table.lookup(positions)
        torch.testing.assert_close(cos_tab, cos_ref, rtol=0, atol=0)
        torch.testing.assert_close(sin_tab, sin_ref, rtol=0, atol=0)

    def test_rotary_table_out_of_range_position_is_a_bug_signal(self):
        table = RotaryTable(max_positions=8, rotary_dim=4, theta=1e7)
        with self.assertRaises(IndexError):
            table.lookup(torch.tensor([100]))


# =========================================================================== #
# QK-RMSNorm vs reference
# =========================================================================== #
class TestNorm(unittest.TestCase):
    def test_rms_norm_head_matches_reference(self):
        torch.manual_seed(0)
        dim = 256
        w = torch.randn(dim) * 0.3
        x = torch.randn(2, 5, 24, dim)  # [B, S, H, D] -- reduces over D only
        ref = RefRMSNorm(dim)
        with torch.no_grad():
            ref.weight.copy_(w)
        got = rms_norm_head(x, w)
        torch.testing.assert_close(got, ref(x), rtol=0, atol=0)

    def test_rms_norm_head_is_one_plus_weight(self):
        dim = 32
        x = torch.randn(4, dim)
        w = torch.zeros(dim)
        expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        torch.testing.assert_close(rms_norm_head(x, w), expected, rtol=1e-6, atol=1e-6)


# =========================================================================== #
# end-to-end attention vs the dense reference (model.py::Attention)
# =========================================================================== #
class TestAttentionParityWithM0(unittest.TestCase):
    def test_varlen_prefill_matches_dense_attention(self):
        torch.manual_seed(42)
        cfg = attn_config(hidden_size=64)
        m = build_attention_module(cfg, seed=42)
        rotary = RotaryEmbedding(cfg)

        lengths = [5, 1, 7]  # a batch of varlen sequences, incl. a length-1 one
        hiddens = [torch.randn(1, L, cfg.hidden_size) for L in lengths]

        # --- reference: run model.py's Attention densely, one seq at a time
        # (no padding needed -> exactly the varlen-causal answer per seq).
        ref_outs = []
        for h, L in zip(hiddens, lengths):
            pos = torch.arange(L).unsqueeze(0)
            cos, sin = rotary(pos, dtype=torch.float32)
            out = m(h, position_embeddings=(cos, sin), cache=None, is_causal=True)
            ref_outs.append(out.squeeze(0))
        ref = torch.cat(ref_outs, dim=0)  # [T, hidden]

        # --- ours: pack all sequences, run projections, fused norm+RoPE,
        # write into a paged pool, torch-fallback varlen SDPA, gate, o_proj.
        hidden_packed = torch.cat([h.squeeze(0) for h in hiddens], dim=0)  # [T, hidden]
        q, k, v, gate = project_qkv(m, hidden_packed)

        cu_seqlens = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()), dtype=torch.int64)
        positions = torch.cat([torch.arange(L) for L in lengths]).long()
        cos_tab = RotaryTable(max_positions=64, rotary_dim=cfg.rotary_dim, theta=cfg.rope_theta, dtype=torch.float32)
        cos, sin = cos_tab.lookup(positions)
        q, k = fused_qk_norm_rope(q, k, m.q_norm.weight, m.k_norm.weight, cos, sin, eps=cfg.rms_norm_eps)

        pool_cfg = KVPoolConfig(
            n_layers=1, num_kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim,
            page_size=4, n_pages=64, max_seqs=len(lengths), max_pages_per_seq=8, dtype="bf16",
        )
        pool = PagedKVPool(pool_cfg)
        slots = [pool.alloc_slot() for _ in lengths]
        slot_ids = torch.cat([torch.full((L,), s, dtype=torch.int64) for s, L in zip(slots, lengths)])
        for s, L in zip(slots, lengths):
            pool.ensure_capacity(s, L)
        pool.append_kv(0, slot_ids, positions, k.to(torch.bfloat16), v.to(torch.bfloat16))

        # read K/V back *through the pool* (not the pre-append tensors) so
        # this test exercises the full paged round-trip, not just the write.
        k_read = torch.cat([pool.gather_dense(0, s, L)[0] for s, L in zip(slots, lengths)], dim=0)
        v_read = torch.cat([pool.gather_dense(0, s, L)[1] for s, L in zip(slots, lengths)], dim=0)

        num_kv_groups = cfg.num_attention_heads // cfg.num_key_value_heads
        attn_out = torch_fallback_prefill(
            q, k_read, v_read, cu_seqlens, m.scaling,
            num_kv_groups=num_kv_groups, causal=True,
        )
        attn_out = attn_out.reshape(attn_out.shape[0], -1)
        gated = apply_output_gate(attn_out, gate)
        ours = m.o_proj(gated)

        torch.testing.assert_close(ours, ref, rtol=1.5e-2, atol=1.5e-2)  # bf16 KV storage rounding

    def test_decode_step_matches_dense_attention_continuation(self):
        torch.manual_seed(7)
        cfg = attn_config(hidden_size=48)
        m = build_attention_module(cfg, seed=7)
        rotary = RotaryEmbedding(cfg)

        B, prompt_len = 3, 6
        prompt = torch.randn(B, prompt_len, cfg.hidden_size)
        new_tok = torch.randn(B, 1, cfg.hidden_size)

        # --- reference: model.py's own HybridCache prefill + one decode step.
        from qwenfast.model import HybridCache  # local import: only needed here

        full_cfg = cfg
        full_cfg.layer_types = ["full_attention"]
        full_cfg.num_hidden_layers = 1
        cache = HybridCache(full_cfg, batch_size=B, max_seq_len=prompt_len + 1, device=torch.device("cpu"), dtype=torch.float32)
        pos_prompt = torch.arange(prompt_len).unsqueeze(0).expand(B, prompt_len)
        cos_p, sin_p = rotary(pos_prompt, dtype=torch.float32)
        m(prompt, position_embeddings=(cos_p, sin_p), cache=cache, cache_start=0, is_causal=True)
        pos_new = torch.full((B, 1), prompt_len, dtype=torch.long)
        cos_n, sin_n = rotary(pos_new, dtype=torch.float32)
        ref = m(new_tok, position_embeddings=(cos_n, sin_n), cache=cache, cache_start=prompt_len, is_causal=False)
        ref = ref.squeeze(1)  # [B, hidden]

        # --- ours: paged pool prefill-append, then one decode step.
        pool_cfg = KVPoolConfig(
            n_layers=1, num_kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim,
            page_size=4, n_pages=64, max_seqs=B, max_pages_per_seq=8, dtype="bf16",
        )
        pool = PagedKVPool(pool_cfg)
        slots = [pool.alloc_slot() for _ in range(B)]

        # prefill: pack [B, prompt_len, hidden] -> [B*prompt_len, hidden]
        hp = prompt.reshape(B * prompt_len, cfg.hidden_size)
        # project per-sequence (batched projection is fine, it's a plain GEMM)
        qg = m.q_proj(hp)
        q, gate = split_q_gate(qg, cfg.num_attention_heads, cfg.head_dim)
        k = m.k_proj(hp).view(-1, cfg.num_key_value_heads, cfg.head_dim)
        v = m.v_proj(hp).view(-1, cfg.num_key_value_heads, cfg.head_dim)
        positions = torch.arange(prompt_len).unsqueeze(0).expand(B, prompt_len).reshape(-1).long()
        table = RotaryTable(max_positions=64, rotary_dim=cfg.rotary_dim, theta=cfg.rope_theta, dtype=torch.float32)
        cos, sin = table.lookup(positions)
        q, k = fused_qk_norm_rope(q, k, m.q_norm.weight, m.k_norm.weight, cos, sin, eps=cfg.rms_norm_eps)
        slot_ids = torch.tensor([slots[b] for b in range(B) for _ in range(prompt_len)], dtype=torch.int64)
        for s in slots:
            pool.ensure_capacity(s, prompt_len + 1)
        pool.append_kv(0, slot_ids, positions, k.to(torch.bfloat16), v.to(torch.bfloat16))

        # decode step: one new token per sequence
        qg1 = m.q_proj(new_tok.squeeze(1))
        q1, gate1 = split_q_gate(qg1, cfg.num_attention_heads, cfg.head_dim)
        k1 = m.k_proj(new_tok.squeeze(1)).view(-1, cfg.num_key_value_heads, cfg.head_dim)
        v1 = m.v_proj(new_tok.squeeze(1)).view(-1, cfg.num_key_value_heads, cfg.head_dim)
        pos1 = torch.full((B,), prompt_len, dtype=torch.long)
        cos1, sin1 = table.lookup(pos1)
        q1, k1 = fused_qk_norm_rope(q1, k1, m.q_norm.weight, m.k_norm.weight, cos1, sin1, eps=cfg.rms_norm_eps)
        slot_ids1 = torch.tensor(slots, dtype=torch.int64)
        pool.append_kv(0, slot_ids1, pos1, k1.to(torch.bfloat16), v1.to(torch.bfloat16))

        attn_out = torch_fallback_decode(pool, 0, slot_ids1, q1.to(torch.bfloat16), m.scaling)
        attn_out = attn_out.to(torch.float32).reshape(B, -1)
        gated = apply_output_gate(attn_out, gate1)
        ours = m.o_proj(gated)

        torch.testing.assert_close(ours, ref, rtol=1.5e-2, atol=1.5e-2)  # bf16 KV storage rounding


# =========================================================================== #
# GPU tests -- skipped automatically on CPU-only hosts
# =========================================================================== #
@unittest.skipUnless(HAS_CUDA and HAS_FLASHINFER, "requires CUDA + flashinfer")
class TestFlashInferGPU(unittest.TestCase):
    def test_decode_matches_torch_fallback_bf16(self):
        """Fix #1 (flashinfer_attn module docstring): FlashInferDecodeAttention
        now defaults to use_tensor_cores=True, which is required at our GQA
        group_size (24/4=6) -- the non-tensor-core kernel raises "Unsupported
        group_size: 6" on H200/flashinfer 0.6.16.post3. This test exercises
        that default directly (no override)."""
        from qwenfast.attn.flashinfer_attn import FlashInferDecodeAttention

        device = "cuda:0"
        torch.manual_seed(0)
        num_qo, num_kv, head_dim, page_size = 24, 4, 256, 16
        B, ctx = 4, 37
        cfg = KVPoolConfig(
            n_layers=1, num_kv_heads=num_kv, head_dim=head_dim, page_size=page_size,
            n_pages=256, max_seqs=B, max_pages_per_seq=16, dtype="bf16", device=device,
        )
        pool = PagedKVPool(cfg)
        slots = [pool.alloc_slot() for _ in range(B)]
        for s in slots:
            pool.ensure_capacity(s, ctx)
            k = (torch.randn(ctx, num_kv, head_dim, device=device) * 0.1).to(torch.bfloat16)
            v = (torch.randn(ctx, num_kv, head_dim, device=device) * 0.1).to(torch.bfloat16)
            pool.append_kv(0, torch.full((ctx,), s, dtype=torch.int64, device=device),
                            torch.arange(ctx, dtype=torch.int64, device=device), k, v)

        q = (torch.randn(B, num_qo, head_dim, device=device) * 0.1).to(torch.bfloat16)
        scaling = head_dim ** -0.5

        ref = torch_fallback_decode(pool, 0, torch.tensor(slots, device=device), q, scaling)

        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        wrapper = FlashInferDecodeAttention(
            workspace, num_qo, num_kv, head_dim, page_size,
            max_batch_size=B, max_pages=256, use_cuda_graph=False, device=device,
        )
        kv_indptr, kv_indices, kv_last_page_len, _ = pool.build_flashinfer_indices(slots)
        wrapper.plan(kv_indptr, kv_indices, kv_last_page_len)
        got = wrapper.run(q, pool.kv[0])

        torch.testing.assert_close(got.float(), ref.float(), rtol=0.08, atol=0.08)

    def test_prefill_matches_torch_fallback_bf16(self):
        from qwenfast.attn.flashinfer_attn import FlashInferPrefillAttention

        device = "cuda:0"
        torch.manual_seed(1)
        num_qo, num_kv, head_dim, page_size = 24, 4, 256, 16
        lengths = [11, 33, 5]
        cfg = KVPoolConfig(
            n_layers=1, num_kv_heads=num_kv, head_dim=head_dim, page_size=page_size,
            n_pages=256, max_seqs=len(lengths), max_pages_per_seq=16, dtype="bf16", device=device,
        )
        pool = PagedKVPool(cfg)
        slots = [pool.alloc_slot() for _ in lengths]
        q_list, k_list, v_list = [], [], []
        for s, L in zip(slots, lengths):
            pool.ensure_capacity(s, L)
            k = (torch.randn(L, num_kv, head_dim, device=device) * 0.1).to(torch.bfloat16)
            v = (torch.randn(L, num_kv, head_dim, device=device) * 0.1).to(torch.bfloat16)
            q = (torch.randn(L, num_qo, head_dim, device=device) * 0.1).to(torch.bfloat16)
            pos = torch.arange(L, dtype=torch.int64, device=device)
            pool.append_kv(0, torch.full((L,), s, dtype=torch.int64, device=device), pos, k, v)
            q_list.append(q)
            k_list.append(k)
            v_list.append(v)

        q_packed = torch.cat(q_list, dim=0)
        k_packed = torch.cat(k_list, dim=0)
        v_packed = torch.cat(v_list, dim=0)
        cu_seqlens = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()),
                                   dtype=torch.int64, device=device)
        scaling = head_dim ** -0.5
        num_kv_groups = num_qo // num_kv
        ref = torch_fallback_prefill(q_packed, k_packed, v_packed, cu_seqlens, scaling,
                                      num_kv_groups=num_kv_groups, causal=True)

        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        wrapper = FlashInferPrefillAttention(workspace, num_qo, num_kv, head_dim, page_size)
        kv_indptr, kv_indices, kv_last_page_len, _ = pool.build_flashinfer_indices(slots)
        wrapper.plan(cu_seqlens.to(torch.int32), kv_indptr, kv_indices, kv_last_page_len, causal=True)
        got = wrapper.run(q_packed, pool.kv[0])

        torch.testing.assert_close(got.float(), ref.float(), rtol=0.08, atol=0.08)

    def test_decode_matches_torch_fallback_fp8_kv(self):
        """FlashInferDecodeAttention.run() must forward k_scale/v_scale to
        FlashInfer's kernel, which silently defaults unset scales to 1.0 --
        wrong whenever the pool's calibrated scale isn't 1.0. FlashInfer's
        k_scale/v_scale are a single Python float per call (not
        per-page/per-head, per the flashinfer 0.6.16 source), so this test
        calibrates with the pool's `calibrate_uniform_scale` (one scale per
        layer/k-or-v, shared by every page and head) rather than the
        per-page `calibrate_page_scale`, and passes the same float into both the
        torch fallback's dequantization (via the pool) and
        `wrapper.run(k_scale=..., v_scale=...)` -- both are dequantizing the
        exact same fp8-quantized-from-fp32 data with the exact same scale,
        so this is effectively "FlashInfer vs. a pure fp32 reference
        computed from the dequantized cache" (torch_fallback_decode's SDPA
        already runs in float32 on `gather_dense`'s dequantized output)."""
        from qwenfast.attn.flashinfer_attn import FlashInferDecodeAttention

        device = "cuda:0"
        torch.manual_seed(2)
        num_qo, num_kv, head_dim, page_size = 24, 4, 256, 16
        B, ctx = 4, 20
        cfg = KVPoolConfig(
            n_layers=1, num_kv_heads=num_kv, head_dim=head_dim, page_size=page_size,
            n_pages=64, max_seqs=B, max_pages_per_seq=16, dtype="fp8", device=device,
        )
        pool = PagedKVPool(cfg)
        slots = [pool.alloc_slot() for _ in range(B)]

        # Generate every sequence's K/V up front so ONE uniform, layer-wide
        # scale can be calibrated from all of it before any append -- see
        # `calibrate_uniform_scale`'s docstring for why a per-page/per-head
        # scale (as `calibrate_page_scale` produces) isn't
        # representable through FlashInfer's kernel in a single call.
        ks = [(torch.randn(ctx, num_kv, head_dim, device=device) * 0.1).to(torch.bfloat16) for _ in range(B)]
        vs = [(torch.randn(ctx, num_kv, head_dim, device=device) * 0.1).to(torch.bfloat16) for _ in range(B)]
        k_scale = pool.calibrate_uniform_scale(0, 0, torch.cat(ks, dim=0))
        v_scale = pool.calibrate_uniform_scale(0, 1, torch.cat(vs, dim=0))

        for s, k, v in zip(slots, ks, vs):
            pool.ensure_capacity(s, ctx)
            pool.append_kv(0, torch.full((ctx,), s, dtype=torch.int64, device=device),
                            torch.arange(ctx, dtype=torch.int64, device=device), k, v)

        q = (torch.randn(B, num_qo, head_dim, device=device) * 0.1).to(torch.bfloat16)
        scaling = head_dim ** -0.5
        ref = torch_fallback_decode(pool, 0, torch.tensor(slots, device=device), q, scaling)

        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        wrapper = FlashInferDecodeAttention(
            workspace, num_qo, num_kv, head_dim, page_size,
            max_batch_size=B, max_pages=64, kv_dtype=pool.storage_dtype,
            use_cuda_graph=False, device=device,
        )
        kv_indptr, kv_indices, kv_last_page_len, _ = pool.build_flashinfer_indices(slots)
        wrapper.plan(kv_indptr, kv_indices, kv_last_page_len)
        got = wrapper.run(q, pool.kv[0], k_scale=k_scale, v_scale=v_scale)

        torch.testing.assert_close(got.float(), ref.float(), rtol=0.15, atol=0.15)

    def test_decode_via_prefill_kernel_matches_torch_fallback_bf16(self):
        """Fix #2 (module docstring): route a decode step through
        BatchPrefillWithPagedKVCacheWrapper with qo_indptr = decode_qo_indptr(B)
        -- the prefill kernel has no GQA group-size restriction, so this is
        an independent workaround for the same group_size=6 failure fix #1
        addresses a different way."""
        from qwenfast.attn.flashinfer_attn import FlashInferPrefillAttention, decode_qo_indptr

        device = "cuda:0"
        torch.manual_seed(3)
        num_qo, num_kv, head_dim, page_size = 24, 4, 256, 16
        B, ctx = 4, 29
        cfg = KVPoolConfig(
            n_layers=1, num_kv_heads=num_kv, head_dim=head_dim, page_size=page_size,
            n_pages=256, max_seqs=B, max_pages_per_seq=16, dtype="bf16", device=device,
        )
        pool = PagedKVPool(cfg)
        slots = [pool.alloc_slot() for _ in range(B)]
        for s in slots:
            pool.ensure_capacity(s, ctx)
            k = (torch.randn(ctx, num_kv, head_dim, device=device) * 0.1).to(torch.bfloat16)
            v = (torch.randn(ctx, num_kv, head_dim, device=device) * 0.1).to(torch.bfloat16)
            pool.append_kv(0, torch.full((ctx,), s, dtype=torch.int64, device=device),
                            torch.arange(ctx, dtype=torch.int64, device=device), k, v)

        q = (torch.randn(B, num_qo, head_dim, device=device) * 0.1).to(torch.bfloat16)
        scaling = head_dim ** -0.5
        ref = torch_fallback_decode(pool, 0, torch.tensor(slots, device=device), q, scaling)

        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        wrapper = FlashInferPrefillAttention(workspace, num_qo, num_kv, head_dim, page_size)
        qo_indptr = decode_qo_indptr(B, device=device)
        kv_indptr, kv_indices, kv_last_page_len, _ = pool.build_flashinfer_indices(slots)
        wrapper.plan(qo_indptr, kv_indptr, kv_indices, kv_last_page_len, causal=True)
        got = wrapper.run(q, pool.kv[0])

        torch.testing.assert_close(got.float(), ref.float(), rtol=0.08, atol=0.08)

    def test_decode_via_prefill_kernel_supports_cuda_graph_buffers(self):
        """The prefill-as-decode workaround must also support the
        persistent-buffer / plan-outside-run-inside CUDA graph split
        when it's pressed into service as a graphed decode backend -- not
        exercised inside an actual torch.cuda.graph() capture here (that
        integration is covered by the runtime tests), just that
        construction+plan+run work with use_cuda_graph=True and fixed-size
        buffers.

        Guards against "q.shape[0] (8) does not match qo_indptr[-1] (3)": `q`
        is always bucket-sized (fixed shape for CUDA-graph replay -- every
        row, including padding, is a real memory slot fed to the kernel), so
        `qo_indptr` must describe exactly one live query token per *bucket*
        row, not per live-batch row. `pad_indptr_to_bucket(
        decode_qo_indptr(B), max_batch_size)` would repeat the live batch's
        last cumulative value into the padded tail (zero-length rows),
        leaving `qo_indptr[-1] == B` while `q.shape[0] == max_batch_size` --
        FlashInfer's `run()` requires them equal. So this uses
        `decode_qo_indptr(max_batch_size)` directly (arange over the whole
        bucket) and pads the *slot list* (not a raw indptr array) with a
        real scratch slot -- one committed page, matching
        `runtime/fused_model.py`'s `AttentionRunner.plan_decode` /
        `FusedModel.scratch_slot` convention -- so every bucket row,
        including padding, has both a real query token AND a real
        (non-empty) KV range to attend to (a zero-length KV range under
        `causal=True` is itself invalid: softmax over zero keys)."""
        from qwenfast.attn.flashinfer_attn import FlashInferPrefillAttention, decode_qo_indptr

        device = "cuda:0"
        torch.manual_seed(4)
        num_qo, num_kv, head_dim, page_size = 24, 4, 256, 16
        B, ctx = 3, 18
        max_batch_size, max_pages = 8, 64  # B=3 < max_batch_size=8: exercises real padding
        cfg = KVPoolConfig(
            n_layers=1, num_kv_heads=num_kv, head_dim=head_dim, page_size=page_size,
            n_pages=max_pages, max_seqs=max_batch_size + 1, max_pages_per_seq=16, dtype="bf16", device=device,
        )
        pool = PagedKVPool(cfg)
        slots = [pool.alloc_slot() for _ in range(B)]
        for s in slots:
            pool.ensure_capacity(s, ctx)
            k = (torch.randn(ctx, num_kv, head_dim, device=device) * 0.1).to(torch.bfloat16)
            v = (torch.randn(ctx, num_kv, head_dim, device=device) * 0.1).to(torch.bfloat16)
            pool.append_kv(0, torch.full((ctx,), s, dtype=torch.int64, device=device),
                            torch.arange(ctx, dtype=torch.int64, device=device), k, v)

        # A real, committed scratch slot (one page, one valid token) for the
        # bucket's padding rows -- see the docstring above.
        scratch_slot = pool.alloc_slot()
        pool.ensure_capacity(scratch_slot, 1)
        pool.append_kv(
            0, torch.full((1,), scratch_slot, dtype=torch.int64, device=device),
            torch.zeros(1, dtype=torch.int64, device=device),
            torch.zeros(1, num_kv, head_dim, dtype=torch.bfloat16, device=device),
            torch.zeros(1, num_kv, head_dim, dtype=torch.bfloat16, device=device),
        )

        q_live = (torch.randn(B, num_qo, head_dim, device=device) * 0.1).to(torch.bfloat16)
        scaling = head_dim ** -0.5
        ref = torch_fallback_decode(pool, 0, torch.tensor(slots, device=device), q_live, scaling)

        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        wrapper = FlashInferPrefillAttention(
            workspace, num_qo, num_kv, head_dim, page_size,
            use_cuda_graph=True, max_batch_size=max_batch_size, max_pages=max_pages, device=device,
        )
        padded_slots = slots + [scratch_slot] * (max_batch_size - B)
        qo_indptr = decode_qo_indptr(max_batch_size, device=device)
        kv_indptr, kv_indices, kv_last_page_len, _ = pool.build_flashinfer_indices(padded_slots)
        q = torch.cat([q_live, torch.zeros(max_batch_size - B, num_qo, head_dim,
                                            dtype=torch.bfloat16, device=device)])
        wrapper.plan(qo_indptr, kv_indptr, kv_indices, kv_last_page_len, causal=True)
        got = wrapper.run(q, pool.kv[0])

        torch.testing.assert_close(got[:B].float(), ref.float(), rtol=0.08, atol=0.08)


@unittest.skipUnless(HAS_CUDA, "requires CUDA")
class TestFA3GPU(unittest.TestCase):
    """Fix #3 (module docstring): FlashAttention-3's paged-KV path
    (flash_attn_varlen_func with block_table/seqused_k) as a third
    independent workaround for the group_size=6 decode failure. Skipped
    (not errored) if no FA3 build is importable, same convention as the
    FlashInfer GPU tests above."""

    def setUp(self):
        from qwenfast.attn.flashinfer_attn import HAS_FA3

        if not HAS_FA3:
            self.skipTest("requires a FlashAttention-3 build (vllm_flash_attn or flash_attn_interface)")

    def test_decode_matches_torch_fallback_bf16(self):
        from qwenfast.attn.flashinfer_attn import fa3_decode_with_kvcache

        device = "cuda:0"
        torch.manual_seed(5)
        num_qo, num_kv, head_dim, page_size = 24, 4, 256, 16
        B, ctx = 4, 23
        cfg = KVPoolConfig(
            n_layers=1, num_kv_heads=num_kv, head_dim=head_dim, page_size=page_size,
            n_pages=256, max_seqs=B, max_pages_per_seq=16, dtype="bf16", device=device,
        )
        pool = PagedKVPool(cfg)
        slots = [pool.alloc_slot() for _ in range(B)]
        for s in slots:
            pool.ensure_capacity(s, ctx)
            k = (torch.randn(ctx, num_kv, head_dim, device=device) * 0.1).to(torch.bfloat16)
            v = (torch.randn(ctx, num_kv, head_dim, device=device) * 0.1).to(torch.bfloat16)
            pool.append_kv(0, torch.full((ctx,), s, dtype=torch.int64, device=device),
                            torch.arange(ctx, dtype=torch.int64, device=device), k, v)

        q = (torch.randn(B, num_qo, head_dim, device=device) * 0.1).to(torch.bfloat16)
        scaling = head_dim ** -0.5
        ref = torch_fallback_decode(pool, 0, torch.tensor(slots, device=device), q, scaling)

        got = fa3_decode_with_kvcache(pool, 0, torch.tensor(slots, device=device), q, scaling)

        torch.testing.assert_close(got.float(), ref.float(), rtol=0.08, atol=0.08)


if __name__ == "__main__":
    unittest.main(verbosity=2)
