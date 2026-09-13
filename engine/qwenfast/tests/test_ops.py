"""CPU unit tests for the qwenfast reference model.

Everything here runs on CPU in float32 and takes a few seconds, so it can gate
every commit without a GPU.  The reference implementations at the top are
copied verbatim out of ``engine/reference/modeling_qwen3_5.py`` (HF
``transformers``) — the tests assert that our versions agree with them.

Run::

    python -m unittest discover -s engine/qwenfast/tests -v
    # or
    python engine/qwenfast/tests/test_ops.py
"""

from __future__ import annotations

import os
import sys
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))  # -> engine/

from qwenfast.model import (  # noqa: E402
    Generator,
    HybridCache,
    QwenFastForCausalLM,
    RMSNorm,
    RMSNormGated,
    RotaryEmbedding,
    apply_rotary_pos_emb,
    causal_conv1d,
    interleave_mrope,
    l2norm,
    rotate_half,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)
from qwenfast.weights import QwenFastConfig, dequant_block128  # noqa: E402


# =========================================================================== #
# reference implementations (verbatim from modeling_qwen3_5.py)
# =========================================================================== #
class RefRMSNorm(nn.Module):
    """modeling_qwen3_5.Qwen3_5RMSNorm (L723)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class RefRMSNormGated(nn.Module):
    """modeling_qwen3_5.Qwen3_5RMSNormGated (L168)."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


def ref_rotate_half(x):
    """modeling_qwen3_5.rotate_half (L549)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def ref_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """modeling_qwen3_5.apply_rotary_pos_emb (L557) — partial RoPE."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (ref_rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (ref_rotate_half(k_rot) * sin)
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


def ref_compute_default_rope_parameters(head_dim, base, partial_rotary_factor):
    """modeling_qwen3_5.Qwen3_5TextRotaryEmbedding.compute_default_rope_parameters (L105)."""
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    return inv_freq, 1.0


def ref_apply_interleaved_mrope(freqs, mrope_section):
    """modeling_qwen3_5.Qwen3_5TextRotaryEmbedding.apply_interleaved_mrope (L149)."""
    freqs_t = freqs[0]  # overwritten in place, as in the reference
    for dim, offset in enumerate((1, 2), start=1):
        length = mrope_section[dim] * 3
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[dim, ..., idx]
    return freqs_t


def ref_mrope_cos_sin(inv_freq, position_ids_3, mrope_section, attention_scaling=1.0):
    """modeling_qwen3_5.Qwen3_5TextRotaryEmbedding.forward (L129), fp32 path.

    ``position_ids_3``: ``[3, B, S]``.
    """
    inv_freq_expanded = inv_freq[None, None, :, None].float().expand(3, position_ids_3.shape[1], -1, 1)
    position_ids_expanded = position_ids_3[:, :, None, :].float()
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)
    freqs = ref_apply_interleaved_mrope(freqs, mrope_section)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos() * attention_scaling, emb.sin() * attention_scaling


def ref_causal_conv1d_fn(hidden_states, weight, bias=None, activation="silu"):
    """modeling_qwen3_5.causal_conv1d_fn (L220)."""
    _, hidden_size, seq_len = hidden_states.shape
    padding = weight.shape[-1] - 1
    out = F.conv1d(
        hidden_states.to(weight.dtype),
        weight=weight.unsqueeze(1),
        bias=bias,
        padding=padding,
        groups=hidden_size,
    )[:, :, :seq_len]
    if activation is not None:
        out = F.silu(out)
    return out.to(hidden_states.dtype)


def ref_causal_conv1d_update(hidden_states, conv_state, weight, bias=None, activation="silu"):
    """modeling_qwen3_5.causal_conv1d_update (L200)."""
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = out[:, :, -seq_len:]
    if activation is not None:
        out = F.silu(out)
    return out.to(hidden_states.dtype)


# =========================================================================== #
# helpers
# =========================================================================== #
def make_gdn_inputs(b, s, h, dk, dv, seed=0, device="cpu"):
    g_cpu = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(b, s, h, dk, generator=g_cpu, dtype=torch.float32).to(device)
    k = torch.randn(b, s, h, dk, generator=g_cpu, dtype=torch.float32).to(device)
    v = torch.randn(b, s, h, dv, generator=g_cpu, dtype=torch.float32).to(device)
    # g must be <= 0 (it is -exp(A_log) * softplus(...) in the model)
    a = torch.randn(b, s, h, generator=g_cpu, dtype=torch.float32).to(device)
    g = -F.softplus(a) * 0.5
    beta = torch.rand(b, s, h, generator=g_cpu, dtype=torch.float32).to(device)
    return q, k, v, g, beta


def tiny_config() -> QwenFastConfig:
    return QwenFastConfig(
        hidden_size=64,
        intermediate_size=32,
        num_hidden_layers=4,
        vocab_size=101,
        rms_norm_eps=1e-6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        attn_output_gate=True,
        full_attention_interval=4,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        rope_theta=10000.0,
        partial_rotary_factor=0.25,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        mtp_num_hidden_layers=1,
        eos_token_id=100,
        bos_token_id=99,
    )


def build_tiny_model(seed=1234, with_mtp=False):
    torch.manual_seed(seed)
    cfg = tiny_config()
    m = QwenFastForCausalLM(cfg, with_mtp=with_mtp).to(torch.float32)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if name.endswith("A_log"):
                p.copy_(torch.log(torch.empty_like(p).uniform_(0.01, 4.0)))
            elif name.endswith("dt_bias"):
                p.uniform_(-1.0, 1.0)
            elif "norm" in name and p.dim() == 1:
                p.normal_(0.0, 0.05)
            else:
                p.normal_(0.0, 0.02)
    m.eval()
    return cfg, m


# =========================================================================== #
# tests
# =========================================================================== #
class TestNorms(unittest.TestCase):
    def test_rmsnorm_matches_reference(self):
        torch.manual_seed(0)
        dim = 96
        ours, ref = RMSNorm(dim), RefRMSNorm(dim)
        w = torch.randn(dim) * 0.3
        with torch.no_grad():
            ours.weight.copy_(w)
            ref.weight.copy_(w)
        x = torch.randn(3, 7, dim)
        torch.testing.assert_close(ours(x), ref(x), rtol=0, atol=0)

    def test_rmsnorm_is_one_plus_weight(self):
        """Zero weight must be an identity scale — catches a (w) vs (1+w) slip."""
        dim = 16
        n = RMSNorm(dim)  # weight initialised to zeros
        x = torch.randn(5, dim)
        expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + n.eps)
        torch.testing.assert_close(n(x), expected, rtol=1e-6, atol=1e-6)

    def test_rmsnorm_gated_matches_reference(self):
        """The GDN output norm uses plain `w *`, NOT `(1 + w) *`."""
        torch.manual_seed(1)
        dim = 128
        ours, ref = RMSNormGated(dim), RefRMSNormGated(dim)
        w = torch.randn(dim) * 0.5 + 1.0
        with torch.no_grad():
            ours.weight.copy_(w)
            ref.weight.copy_(w)
        x = torch.randn(11, dim)
        gate = torch.randn(11, dim)
        torch.testing.assert_close(ours(x, gate), ref(x, gate), rtol=0, atol=0)

    def test_l2norm(self):
        x = torch.randn(2, 3, 8)
        y = l2norm(x, dim=-1, eps=1e-6)
        torch.testing.assert_close(
            y.norm(dim=-1), torch.ones(2, 3), rtol=1e-4, atol=1e-4
        )


class TestRope(unittest.TestCase):
    HEAD_DIM = 256
    THETA = 1e7
    PRF = 0.25
    MROPE_SECTION = [11, 11, 10]

    def test_rotate_half_matches_reference(self):
        x = torch.randn(2, 3, 4, 64)
        torch.testing.assert_close(rotate_half(x), ref_rotate_half(x), rtol=0, atol=0)

    def test_partial_rope_matches_reference(self):
        torch.manual_seed(2)
        b, h, s = 2, 4, 6
        rot = int(self.HEAD_DIM * self.PRF)
        q = torch.randn(b, h, s, self.HEAD_DIM)
        k = torch.randn(b, 2, s, self.HEAD_DIM)
        cos = torch.randn(b, s, rot)
        sin = torch.randn(b, s, rot)
        qo, ko = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
        qr, kr = ref_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
        torch.testing.assert_close(qo, qr, rtol=0, atol=0)
        torch.testing.assert_close(ko, kr, rtol=0, atol=0)
        # the non-rotated tail must be untouched
        torch.testing.assert_close(qo[..., rot:], q[..., rot:], rtol=0, atol=0)

    def test_inv_freq_matches_reference(self):
        cfg = QwenFastConfig(
            head_dim=self.HEAD_DIM, rope_theta=self.THETA, partial_rotary_factor=self.PRF
        )
        self.assertEqual(cfg.rotary_dim, 64)
        rope = RotaryEmbedding(cfg)
        ref_inv, _ = ref_compute_default_rope_parameters(self.HEAD_DIM, self.THETA, self.PRF)
        torch.testing.assert_close(rope.inv_freq(torch.device("cpu")), ref_inv, rtol=0, atol=0)
        self.assertEqual(ref_inv.numel(), 32)

    def test_text_only_mrope_degenerates_to_standard_rope(self):
        """For text-only inputs the 3 mRoPE rows are identical, so the
        interleave is a no-op and mRoPE == standard partial RoPE."""
        cfg = QwenFastConfig(
            head_dim=self.HEAD_DIM, rope_theta=self.THETA, partial_rotary_factor=self.PRF
        )
        rope = RotaryEmbedding(cfg)
        b, s = 2, 13
        pos = torch.arange(s).unsqueeze(0).expand(b, s).contiguous()
        cos, sin = rope(pos, dtype=torch.float32)

        pos3 = pos[None].expand(3, b, s).contiguous().float()
        ref_cos, ref_sin = ref_mrope_cos_sin(
            rope.inv_freq(torch.device("cpu")), pos3, self.MROPE_SECTION
        )
        torch.testing.assert_close(cos, ref_cos, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(sin, ref_sin, rtol=1e-6, atol=1e-6)
        self.assertEqual(cos.shape, (b, s, 64))

    def test_interleave_mrope_matches_reference_when_rows_differ(self):
        torch.manual_seed(3)
        freqs = torch.randn(3, 2, 5, 32)
        got = interleave_mrope(freqs.clone(), self.MROPE_SECTION)
        want = ref_apply_interleaved_mrope(freqs.clone(), self.MROPE_SECTION)
        torch.testing.assert_close(got, want, rtol=0, atol=0)


class TestGatedDeltaRule(unittest.TestCase):
    def _check(self, b, s, h, dk, dv, chunk_size, seed, atol=2e-4):
        q, k, v, g, beta = make_gdn_inputs(b, s, h, dk, dv, seed=seed)
        o_c, s_c = torch_chunk_gated_delta_rule(
            q, k, v, g, beta,
            chunk_size=chunk_size, initial_state=None,
            output_final_state=True, use_qk_l2norm_in_kernel=True,
        )
        o_r, s_r = torch_recurrent_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True,
        )
        torch.testing.assert_close(o_c, o_r, rtol=1e-3, atol=atol)
        torch.testing.assert_close(s_c, s_r, rtol=1e-3, atol=atol)

    def test_chunk_equals_recurrent_exact_multiple(self):
        self._check(b=2, s=32, h=3, dk=16, dv=16, chunk_size=16, seed=0)

    def test_chunk_equals_recurrent_ragged_length(self):
        self._check(b=2, s=37, h=3, dk=16, dv=16, chunk_size=16, seed=1)

    def test_chunk_equals_recurrent_single_token(self):
        self._check(b=1, s=1, h=2, dk=8, dv=8, chunk_size=16, seed=2)

    def test_chunk_equals_recurrent_default_chunk64(self):
        self._check(b=1, s=100, h=2, dk=32, dv=32, chunk_size=64, seed=3)

    def test_chunk_equals_recurrent_nonsquare_state(self):
        """Qwen3.8 uses dk == dv == 128, but the math must not assume it."""
        self._check(b=2, s=20, h=2, dk=16, dv=8, chunk_size=8, seed=4)

    def test_initial_state_is_honoured(self):
        """Splitting a sequence in two and threading the state through must
        equal a single pass — this is the chunked-prefill invariant."""
        b, s, h, dk, dv = 2, 48, 3, 16, 16
        q, k, v, g, beta = make_gdn_inputs(b, s, h, dk, dv, seed=5)
        full_o, full_s = torch_chunk_gated_delta_rule(
            q, k, v, g, beta, chunk_size=16, initial_state=None,
            output_final_state=True, use_qk_l2norm_in_kernel=True,
        )
        cut = 32
        o1, s1 = torch_chunk_gated_delta_rule(
            q[:, :cut], k[:, :cut], v[:, :cut], g[:, :cut], beta[:, :cut],
            chunk_size=16, initial_state=None, output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        o2, s2 = torch_chunk_gated_delta_rule(
            q[:, cut:], k[:, cut:], v[:, cut:], g[:, cut:], beta[:, cut:],
            chunk_size=16, initial_state=s1, output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        torch.testing.assert_close(torch.cat([o1, o2], dim=1), full_o, rtol=1e-3, atol=2e-4)
        torch.testing.assert_close(s2, full_s, rtol=1e-3, atol=2e-4)

    def test_recurrent_step_by_step_equals_chunk(self):
        """The decode path: one token at a time, threading the state."""
        b, s, h, dk, dv = 1, 24, 2, 16, 16
        q, k, v, g, beta = make_gdn_inputs(b, s, h, dk, dv, seed=6)
        ref_o, ref_s = torch_chunk_gated_delta_rule(
            q, k, v, g, beta, chunk_size=8, initial_state=None,
            output_final_state=True, use_qk_l2norm_in_kernel=True,
        )
        state = None
        outs = []
        for t in range(s):
            o, state = torch_recurrent_gated_delta_rule(
                q[:, t : t + 1], k[:, t : t + 1], v[:, t : t + 1],
                g[:, t : t + 1], beta[:, t : t + 1],
                initial_state=state, output_final_state=True, use_qk_l2norm_in_kernel=True,
            )
            outs.append(o)
        torch.testing.assert_close(torch.cat(outs, dim=1), ref_o, rtol=1e-3, atol=2e-4)
        torch.testing.assert_close(state, ref_s, rtol=1e-3, atol=2e-4)

    def test_zero_inputs_leave_state_untouched(self):
        """Why left-padding is safe: k=v=0 makes the delta update a no-op."""
        b, h, dk, dv = 2, 3, 16, 16
        state0 = torch.randn(b, h, dk, dv)
        s = 5
        q = torch.zeros(b, s, h, dk)
        k = torch.zeros(b, s, h, dk)
        v = torch.zeros(b, s, h, dv)
        beta = torch.full((b, s, h), 0.5)
        g = torch.zeros(b, s, h)  # no decay -> state must be bit-stable
        _, st = torch_recurrent_gated_delta_rule(
            q, k, v, g, beta, initial_state=state0.clone(),
            output_final_state=True, use_qk_l2norm_in_kernel=False,
        )
        torch.testing.assert_close(st, state0, rtol=1e-5, atol=1e-6)


class TestCausalConv(unittest.TestCase):
    def test_prefill_matches_reference_conv_fn(self):
        torch.manual_seed(7)
        b, c, s, kk = 2, 12, 9, 4
        x = torch.randn(b, c, s)
        w = torch.randn(c, kk)
        state = torch.zeros(b, c, kk - 1)
        got = causal_conv1d(x, state, w)
        want = ref_causal_conv1d_fn(x, w)
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
        # the state must hold the last (K-1) inputs
        torch.testing.assert_close(state, x[:, :, -(kk - 1) :], rtol=0, atol=0)

    def test_decode_matches_reference_update(self):
        torch.manual_seed(8)
        b, c, kk = 2, 12, 4
        w = torch.randn(c, kk)
        x = torch.randn(b, c, 1)
        s1 = torch.randn(b, c, kk - 1)
        s2 = s1.clone()
        got = causal_conv1d(x, s1, w)
        want = ref_causal_conv1d_update(x, s2, w)
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(s1, s2, rtol=0, atol=0)

    def test_prefill_then_decode_equals_full_prefill(self):
        torch.manual_seed(9)
        b, c, s, kk = 1, 8, 11, 4
        w = torch.randn(c, kk)
        x = torch.randn(b, c, s)
        full = causal_conv1d(x, torch.zeros(b, c, kk - 1), w)
        state = torch.zeros(b, c, kk - 1)
        head = causal_conv1d(x[:, :, : s - 3], state, w)
        tail = [causal_conv1d(x[:, :, t : t + 1], state, w) for t in range(s - 3, s)]
        got = torch.cat([head] + tail, dim=-1)
        torch.testing.assert_close(got, full, rtol=1e-5, atol=1e-5)


class TestFp8Dequant(unittest.TestCase):
    def test_block128_dequant(self):
        torch.manual_seed(10)
        n, k = 256, 384
        w = torch.randn(n, k)
        try:
            q = w.to(torch.float8_e4m3fn)
        except (RuntimeError, TypeError) as exc:  # very old torch
            self.skipTest(f"no float8_e4m3fn on this build: {exc}")
        scale = torch.rand(n // 128, k // 128) + 0.5
        got = dequant_block128(q, scale, out_dtype=torch.float32)
        want = q.to(torch.float32) * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
        torch.testing.assert_close(got, want, rtol=0, atol=0)
        self.assertEqual(got.shape, (n, k))

    def test_block128_dequant_padded_shape(self):
        torch.manual_seed(11)
        n, k = 130, 200
        w = torch.randn(n, k)
        scale = torch.rand(2, 2) + 0.5
        got = dequant_block128(w.to(torch.float32), scale, out_dtype=torch.float32)
        self.assertEqual(got.shape, (n, k))
        self.assertAlmostEqual(float(got[0, 0]), float(w[0, 0] * scale[0, 0]), places=5)
        self.assertAlmostEqual(float(got[129, 199]), float(w[129, 199] * scale[1, 1]), places=5)


class TestConfig(unittest.TestCase):
    def test_derived_shapes_match_qwen38_27b(self):
        path = os.path.abspath(
            os.path.join(_HERE, "..", "..", "reference", "config-Qwen3.8-27B.json")
        )
        if not os.path.exists(path):
            self.skipTest("reference config not available")
        cfg = QwenFastConfig.from_json(path)
        self.assertEqual(cfg.hidden_size, 5120)
        self.assertEqual(cfg.num_hidden_layers, 64)
        self.assertEqual(len(cfg.linear_layer_indices), 48)
        self.assertEqual(len(cfg.attention_layer_indices), 16)
        self.assertEqual(cfg.key_dim, 2048)
        self.assertEqual(cfg.value_dim, 6144)
        self.assertEqual(cfg.conv_dim, 10240)
        self.assertEqual(cfg.rotary_dim, 64)
        self.assertEqual(cfg.num_v_per_k, 3)
        self.assertEqual(cfg.mtp_layer_idx, 64)
        # 3.0 MiB per GDN layer per sequence in fp32
        per_layer = cfg.linear_num_value_heads * cfg.linear_key_head_dim * cfg.linear_value_head_dim * 4
        self.assertEqual(per_layer, 3 * 1024 * 1024)
        self.assertEqual(per_layer * len(cfg.linear_layer_indices), 144 * 1024 * 1024)
        # 32 KiB per token of fp8 KV across the 16 attention layers
        kv_per_token_fp8 = len(cfg.attention_layer_indices) * cfg.num_key_value_heads * cfg.head_dim * 2
        self.assertEqual(kv_per_token_fp8, 32 * 1024)


class TestTinyModelEndToEnd(unittest.TestCase):
    """Whole-stack checks on a randomly initialised tiny model (CPU, fp32)."""

    def test_parameter_names_map_to_checkpoint_names(self):
        from qwenfast.weights import our_name_to_ckpt_name

        cfg, m = build_tiny_model(with_mtp=True)
        names = {n for n, _ in m.named_parameters()}
        self.assertIn("model.layers.0.linear_attn.in_proj_qkv.weight", names)
        self.assertIn("model.layers.3.self_attn.q_proj.weight", names)
        self.assertIn("mtp.fc.weight", names)
        self.assertIn("mtp.layers.0.self_attn.q_proj.weight", names)
        self.assertEqual(
            our_name_to_ckpt_name("model.layers.0.linear_attn.in_proj_qkv.weight"),
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        )
        self.assertEqual(our_name_to_ckpt_name("lm_head.weight"), "lm_head.weight")
        self.assertEqual(our_name_to_ckpt_name("mtp.fc.weight"), "mtp.fc.weight")

    def test_q_proj_carries_the_output_gate(self):
        cfg, m = build_tiny_model()
        attn = m.model.layers[3].self_attn
        self.assertEqual(
            attn.q_proj.weight.shape,
            (cfg.num_attention_heads * cfg.head_dim * 2, cfg.hidden_size),
        )
        self.assertEqual(
            attn.o_proj.weight.shape,
            (cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim),
        )

    def test_prefill_plus_decode_equals_one_shot_forward(self):
        """Incremental decode must reproduce a single full-sequence forward."""
        cfg, m = build_tiny_model()
        torch.manual_seed(21)
        prompt = torch.randint(0, cfg.vocab_size, (1, 12))
        extra = torch.randint(0, cfg.vocab_size, (1, 4))
        full = torch.cat([prompt, extra], dim=1)

        with torch.inference_mode():
            ref_logits = m(full, num_logits=full.shape[1], use_fla=False)

            cache = m.make_cache(1, 32, device=torch.device("cpu"), dtype=torch.float32)
            gen = Generator(m, use_fla=False)
            _, last, pos = gen.prefill([prompt[0].tolist()], cache)
            torch.testing.assert_close(
                last, ref_logits[:, prompt.shape[1] - 1, :], rtol=1e-4, atol=1e-4
            )
            for i in range(extra.shape[1]):
                _, lg = gen.decode_step(extra[:, i], pos, cache)
                pos = pos + 1
                torch.testing.assert_close(
                    lg, ref_logits[:, prompt.shape[1] + i, :], rtol=1e-4, atol=1e-4
                )

    def test_left_padded_batch_matches_single_sequence(self):
        """Left padding must not leak into the GDN state or the KV cache."""
        cfg, m = build_tiny_model()
        torch.manual_seed(22)
        p_long = torch.randint(0, cfg.vocab_size, (14,)).tolist()
        p_short = torch.randint(0, cfg.vocab_size, (6,)).tolist()

        with torch.inference_mode():
            gen = Generator(m, pad_id=0, use_fla=False)
            cache = m.make_cache(2, 32, device=torch.device("cpu"), dtype=torch.float32)
            _, batched, _ = gen.prefill([p_long, p_short], cache)

            singles = []
            for p in (p_long, p_short):
                c = m.make_cache(1, 32, device=torch.device("cpu"), dtype=torch.float32)
                _, lg, _ = Generator(m, pad_id=0, use_fla=False).prefill([p], c)
                singles.append(lg)
            torch.testing.assert_close(batched[0:1], singles[0], rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(batched[1:2], singles[1], rtol=1e-4, atol=1e-4)

    def test_batched_greedy_generation_matches_single(self):
        cfg, m = build_tiny_model()
        torch.manual_seed(23)
        prompts = [
            torch.randint(0, cfg.vocab_size, (n,)).tolist() for n in (5, 11, 8)
        ]
        gen = Generator(m, pad_id=0, use_fla=False)
        batched = gen.generate(prompts, max_new_tokens=6, eos_token_id=-1)
        for i, p in enumerate(prompts):
            single = Generator(m, pad_id=0, use_fla=False).generate(
                [p], max_new_tokens=6, eos_token_id=-1
            )[0]
            self.assertEqual(batched[i], single, f"sequence {i} diverged")

    def test_cache_byte_accounting(self):
        cfg, m = build_tiny_model()
        cache = m.make_cache(3, 16, device=torch.device("cpu"), dtype=torch.float32)
        b = cache.nbytes()
        want_ssm = 3 * len(cfg.linear_layer_indices) * cfg.linear_num_value_heads * 8 * 8 * 4
        self.assertEqual(b["ssm"], want_ssm)
        want_kv = 3 * len(cfg.attention_layer_indices) * cfg.num_key_value_heads * 16 * cfg.head_dim * 4 * 2
        self.assertEqual(b["kv"], want_kv)

    def test_ssm_snapshot_restore_roundtrip(self):
        """The reference model's stand-in for MTP rollback."""
        cfg, m = build_tiny_model()
        torch.manual_seed(24)
        gen = Generator(m, pad_id=0, use_fla=False)
        cache = m.make_cache(1, 32, device=torch.device("cpu"), dtype=torch.float32)
        prompt = torch.randint(0, cfg.vocab_size, (7,)).tolist()
        tok, _, pos = gen.prefill([prompt], cache)

        snap = cache.clone_ssm()
        saved_len = cache.seq_len
        tok_a, logits_a = gen.decode_step(tok, pos, cache)

        cache.restore_ssm(snap)
        cache.seq_len = saved_len
        cache.valid[:, saved_len:] = False
        tok_b, logits_b = gen.decode_step(tok, pos, cache)
        torch.testing.assert_close(logits_a, logits_b, rtol=1e-5, atol=1e-5)
        self.assertEqual(int(tok_a), int(tok_b))

    def test_mtp_head_runs_and_shapes(self):
        cfg, m = build_tiny_model(with_mtp=True)
        torch.manual_seed(25)
        self.assertEqual(m.mtp.fc.weight.shape, (cfg.hidden_size, 2 * cfg.hidden_size))
        b, s = 2, 5
        ids = torch.randint(0, cfg.vocab_size, (b, s))
        pos = torch.arange(s).unsqueeze(0).expand(b, s)
        with torch.inference_mode():
            _, hidden = m(ids, position_ids=pos, num_logits=1, return_hidden=True, use_fla=False)
            self.assertEqual(hidden.shape, (b, s, cfg.hidden_size))
            emb = m.model.embed_tokens(ids)
            cos, sin = m.model.rotary_emb(pos, dtype=hidden.dtype)
            h = m.mtp(emb, hidden, (cos, sin), cache=None, is_causal=True)
            self.assertEqual(h.shape, (b, s, cfg.hidden_size))
            draft = m.lm_head(h)
            self.assertEqual(draft.shape, (b, s, cfg.vocab_size))
            self.assertTrue(torch.isfinite(draft).all())


def config_to_json_dict(cfg: QwenFastConfig, quant: bool = False) -> dict:
    text = {
        "hidden_size": cfg.hidden_size,
        "intermediate_size": cfg.intermediate_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "vocab_size": cfg.vocab_size,
        "rms_norm_eps": cfg.rms_norm_eps,
        "hidden_act": cfg.hidden_act,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": cfg.head_dim,
        "attention_bias": cfg.attention_bias,
        "attn_output_gate": cfg.attn_output_gate,
        "full_attention_interval": cfg.full_attention_interval,
        "linear_num_value_heads": cfg.linear_num_value_heads,
        "linear_num_key_heads": cfg.linear_num_key_heads,
        "linear_key_head_dim": cfg.linear_key_head_dim,
        "linear_value_head_dim": cfg.linear_value_head_dim,
        "linear_conv_kernel_dim": cfg.linear_conv_kernel_dim,
        "layer_types": cfg.layer_types,
        "mtp_num_hidden_layers": cfg.mtp_num_hidden_layers,
        "mtp_use_dedicated_embeddings": False,
        "eos_token_id": cfg.eos_token_id,
        "bos_token_id": cfg.bos_token_id,
        "tie_word_embeddings": False,
        "max_position_embeddings": cfg.max_position_embeddings,
        "rope_parameters": {
            "rope_theta": cfg.rope_theta,
            "partial_rotary_factor": cfg.partial_rotary_factor,
            "mrope_section": cfg.mrope_section,
            "mrope_interleaved": True,
            "rope_type": "default",
        },
    }
    raw = {"architectures": ["Qwen3_5ForConditionalGeneration"], "text_config": text}
    if quant:
        raw["quantization_config"] = {"quant_method": "fp8", "fmt": "e4m3",
                                      "activation_scheme": "dynamic"}
    return raw


class TestCheckpointLoading(unittest.TestCase):
    """End-to-end load through weights.py, on a synthetic tiny checkpoint."""

    def _write_ckpt(self, tmpdir: str, model, cfg, fp8_for=()):
        import json

        from safetensors.torch import save_file

        from qwenfast.weights import our_name_to_ckpt_name

        tensors = {}
        for name, p in model.state_dict().items():
            ck = our_name_to_ckpt_name(name)
            t = p.detach().clone().to(torch.float32)
            if any(name.endswith(s) for s in fp8_for) and t.dim() == 2:
                n, k = t.shape
                nb, kb = (n + 127) // 128, (k + 127) // 128
                scale = torch.full((nb, kb), 0.25)
                q = (t / scale[0, 0]).to(torch.float8_e4m3fn)
                tensors[ck] = q
                tensors[ck + "_scale_inv"] = scale
            else:
                tensors[ck] = t.to(torch.bfloat16)
        # a tensor the loader must ignore
        tensors["model.visual.blocks.0.attn.qkv.weight"] = torch.zeros(4, 4, dtype=torch.bfloat16)
        save_file(tensors, os.path.join(tmpdir, "model.safetensors"))
        with open(os.path.join(tmpdir, "config.json"), "w") as f:
            json.dump(config_to_json_dict(cfg, quant=bool(fp8_for)), f)

    def _roundtrip(self, fp8_for=()):
        import tempfile

        try:
            import safetensors.torch  # noqa: F401
        except ImportError:
            self.skipTest("safetensors not installed")
        cfg, m = build_tiny_model(seed=77, with_mtp=True)
        with tempfile.TemporaryDirectory() as tmp:
            self._write_ckpt(tmp, m, cfg, fp8_for=fp8_for)
            loaded = QwenFastForCausalLM.from_pretrained(
                tmp, device="cpu", dtype=torch.float32, with_mtp=True
            )
        got = dict(loaded.state_dict())
        want = dict(m.state_dict())
        self.assertEqual(set(got), set(want))
        for name in want:
            self.assertFalse(got[name].is_meta, f"{name} left on meta")
            self.assertEqual(got[name].shape, want[name].shape, name)
        return cfg, m, loaded

    def test_bf16_roundtrip_is_exact_and_runs(self):
        cfg, m, loaded = self._roundtrip()
        for name, w in m.state_dict().items():
            torch.testing.assert_close(
                loaded.state_dict()[name], w.to(torch.bfloat16).to(torch.float32),
                rtol=0, atol=0, msg=name,
            )
        ids = torch.randint(0, cfg.vocab_size, (1, 9))
        with torch.inference_mode():
            lg = loaded(ids, num_logits=1, use_fla=False)
        self.assertEqual(lg.shape, (1, 1, cfg.vocab_size))
        self.assertTrue(torch.isfinite(lg).all())

    def test_fp8_block_dequant_roundtrip(self):
        """FP8 weights must come back within e4m3 quantisation error."""
        cfg, m, loaded = self._roundtrip(
            fp8_for=("in_proj_qkv.weight", "gate_proj.weight", "q_proj.weight")
        )
        ref = m.state_dict()["model.layers.0.linear_attn.in_proj_qkv.weight"]
        got = loaded.state_dict()["model.layers.0.linear_attn.in_proj_qkv.weight"]
        # e4m3 has 3 mantissa bits -> ~6% relative error
        torch.testing.assert_close(got, ref, rtol=0.07, atol=1e-3)
        self.assertGreater(float((got - ref).abs().max()), 0.0, "fp8 path did not quantise")

    def test_missing_tensor_is_reported(self):
        import json
        import tempfile

        from safetensors.torch import save_file

        cfg, m = build_tiny_model(seed=78)
        with tempfile.TemporaryDirectory() as tmp:
            self._write_ckpt(tmp, m, cfg)
            # drop one tensor and rewrite
            from safetensors.torch import load_file

            t = load_file(os.path.join(tmp, "model.safetensors"))
            del t["model.language_model.layers.0.linear_attn.out_proj.weight"]
            save_file(t, os.path.join(tmp, "model.safetensors"))
            with open(os.path.join(tmp, "config.json"), "w") as f:
                json.dump(config_to_json_dict(cfg), f)
            with self.assertRaises(KeyError):
                QwenFastForCausalLM.from_pretrained(tmp, device="cpu", dtype=torch.float32)


class TestVerifyHarness(unittest.TestCase):
    """Guards the teacher-forced slicing in verify_vs_hf.py.

    An off-by-one in either slice would produce a plausible-looking but wrong
    verdict on the GPU, so it is pinned here on CPU.
    """

    def _load_harness(self):
        import importlib.util

        path = os.path.abspath(os.path.join(_HERE, "..", "verify_vs_hf.py"))
        spec = importlib.util.spec_from_file_location("qf_verify", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_to_id_list_normalises_every_shape(self):
        v = self._load_harness()

        class BatchEncodingLike(dict):
            pass

        self.assertEqual(v._to_id_list([1, 2, 3]), [1, 2, 3])
        self.assertEqual(v._to_id_list([[1, 2, 3]]), [1, 2, 3])
        self.assertEqual(v._to_id_list(torch.tensor([[1, 2, 3]])), [1, 2, 3])
        self.assertEqual(v._to_id_list(BatchEncodingLike(input_ids=[[1, 2, 3]])), [1, 2, 3])
        self.assertEqual(
            v._to_id_list(BatchEncodingLike(input_ids=torch.tensor([[4, 5]]))), [4, 5]
        )

    def test_truncate_at_eos(self):
        v = self._load_harness()
        self.assertEqual(v.truncate_at_eos([1, 2, 9, 3], [9]), ([1, 2], 2))
        self.assertEqual(v.truncate_at_eos([1, 2, 3], [9]), ([1, 2, 3], None))
        self.assertEqual(v.truncate_at_eos([9, 1], [9]), ([], 0))

    def test_teacher_forced_logits_reproduce_the_greedy_continuation(self):
        """`tf[j]` must be the logits that predicted `continuation[j]`.

        The continuation was produced greedily through the *cached* path, so if
        the *uncached* teacher-forced slice is aligned, its argmax must
        reproduce the continuation exactly. This checks the slice and the
        cached/uncached equivalence in one shot.
        """
        v = self._load_harness()
        cfg, m = build_tiny_model(seed=31)
        torch.manual_seed(32)
        seqs = [torch.randint(0, cfg.vocab_size, (n,)).tolist() for n in (9, 5)]
        n_new = 6
        o = v.ours_pass(m, seqs, torch.device("cpu"), n_new, use_fla=False)

        for i, cont in enumerate(o["continuations"]):
            self.assertEqual(len(cont), n_new)
            tf = o["tf"][i]
            self.assertEqual(tf.shape, (n_new, cfg.vocab_size))
            self.assertEqual(
                tf.argmax(-1).tolist(), cont,
                f"teacher-forced slice is misaligned for sequence {i}",
            )

    def test_hf_side_slice_lines_up_with_ours(self):
        """Feed the same model through the HF-shaped path; both must agree."""
        v = self._load_harness()
        cfg, m = build_tiny_model(seed=33)
        torch.manual_seed(34)
        seqs = [torch.randint(0, cfg.vocab_size, (n,)).tolist() for n in (7, 4)]
        n_new = 5
        o = v.ours_pass(m, seqs, torch.device("cpu"), n_new, use_fla=False)

        class HFLike:
            """Minimal stand-in exposing .logits over all positions and .generate."""

            class Out:
                def __init__(self, logits):
                    self.logits = logits

            def __call__(self, input_ids=None, attention_mask=None):
                with torch.inference_mode():
                    lg = m(input_ids, num_logits=input_ids.shape[1], use_fla=False)
                return HFLike.Out(lg)

            def generate(self, input_ids=None, attention_mask=None, max_new_tokens=1, **kw):
                g = Generator(m, pad_id=0, use_fla=False)
                out = g.generate([input_ids[0].tolist()], max_new_tokens=max_new_tokens,
                                 eos_token_id=-1)[0]
                return torch.cat([input_ids, torch.tensor([out], dtype=torch.long)], dim=1)

        h = v.hf_pass(HFLike(), seqs, o["continuations"], torch.device("cpu"), n_new)
        for i in range(len(seqs)):
            self.assertEqual(h["tf"][i].shape, o["tf"][i].shape)
            torch.testing.assert_close(h["tf"][i], o["tf"][i], rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(
                h["prefill_last"][i], o["prefill_last"][i], rtol=1e-4, atol=1e-4
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
