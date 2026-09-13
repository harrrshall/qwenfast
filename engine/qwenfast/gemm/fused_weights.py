"""Fused per-layer weight builder.

Builds the engine's fused weight layout from a
:class:`~qwenfast.weights.SafetensorsStore`, **without dequantizing any
FP8 tensor**. FP8 weights stay ``float8_e4m3fn`` end to end; only the
``weight_scale_inv`` blocks are concatenated.

Fusions (all legal because the concatenated N is a multiple of 128, so the
block-128 ``weight_scale_inv`` grid concatenates cleanly along axis 0):

    GDN   in_proj_qkv(10240) + in_proj_z(6144)  -> in_proj_qkvz [16384, 5120] fp8, scale [128, 40]
    GDN   in_proj_b(48)      + in_proj_a(48)    -> in_proj_ba   [   96, 5120] bf16 (not 128-aligned; kept unquantized)
    Attn  q_proj(12288) + k_proj(1024) + v_proj(1024) -> qkv_proj [14336, 5120] fp8, scale [112, 40]
    MLP   gate_proj(17408) + up_proj(17408)     -> gate_up_proj [34816, 5120] fp8, scale [272, 40]

``out_proj`` (GDN), ``o_proj`` (attention) and ``down_proj`` (MLP) are already
single GEMMs and are carried through unfused, as FP8 tensors.

The MTP head (``mtp.*``) is one full-attention decoder layer plus an MLP, so
it goes through exactly the same attention/MLP fusion helpers.

``embed_tokens`` and ``lm_head`` stay bf16 (a gather is not a GEMM, and
the untied lm_head is bf16 in the checkpoint). An **optional** fp8 per-block
quantized ``lm_head`` is offered behind ``quantize_lm_head=True``.

``save_fused``/``load_fused`` round-trip the whole result through a single
safetensors file plus a small JSON-in-metadata blob, so the GPU host can load
fused weights from disk in seconds instead of re-fusing on every boot.

This module must import cleanly with **no torch/CUDA available** (py_compile
on a Mac laptop) -- torch is only touched inside function bodies typed with
``"torch.Tensor"`` string annotations, never at import time.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime import cost
    import torch

BLOCK = 128
FP8_MAX = 448.0  # e4m3 max representable magnitude (matches kernels/microbench/gemm_bench.py)

FUSED_FORMAT_VERSION = "qwenfast-fused-v1"
FUSED_WEIGHTS_FILENAME = "fused_weights.safetensors"


# --------------------------------------------------------------------------- #
# containers
# --------------------------------------------------------------------------- #
@dataclass
class FP8Tensor:
    """A block-128 FP8 weight: ``weight`` is ``[N, K]`` ``float8_e4m3fn``,
    ``scale_inv`` is ``[N/128, K/128]`` -- the value at ``[n, k]`` dequantizes
    as ``weight[n, k].float() * scale_inv[n // 128, k // 128]`` (same
    convention as :func:`qwenfast.weights.dequant_block128`).
    """

    weight: "torch.Tensor"
    scale_inv: "torch.Tensor"
    # lazily-computed backend-specific caches (repacked weight/scale
    # layouts) used by gemm/dispatch.py's GPU backends. Never persisted;
    # excluded from dataclass equality/repr so FP8Tensor stays a plain value
    # type everywhere else (tests, save/load round-trips). Populated on first
    # use of the corresponding backend, so repacking (which can be
    # non-trivial -- e.g. Marlin's GPTQ-style repack + scale permutation) is
    # paid once per weight, not once per decode step.
    _pertensor_cache: Optional[Tuple["torch.Tensor", "torch.Tensor"]] = field(
        default=None, repr=False, compare=False
    )
    _deepgemm_cache: Optional[Tuple["torch.Tensor", "torch.Tensor", bool]] = field(
        default=None, repr=False, compare=False
    )
    _marlin_cache: Optional[Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]] = field(
        default=None, repr=False, compare=False
    )
    #: Machete's prepacked int8 weight + bf16 group scales.
    #: Unlike the three above this one is **not** a pure re-layout of the same
    #: numbers -- Machete has no compiled fp8 b_type in vLLM 0.28.0, so this
    #: cache holds a *re-quantized* int8 copy of the dequantized fp8 weight
    #: (see ``gemm/dispatch.py::_machete_repacked``). Same byte cost as the
    #: fp8 weight it stands in for (1 B/elt) plus bf16 [K/128, N] group scales.
    _machete_cache: Optional[Tuple["torch.Tensor", "torch.Tensor"]] = field(
        default=None, repr=False, compare=False
    )

    @property
    def shape(self) -> Tuple[int, int]:
        return tuple(self.weight.shape)  # type: ignore[return-value]

    def dequant(self, out_dtype: "torch.dtype" = None) -> "torch.Tensor":
        """Dequantize to ``out_dtype`` (default bf16). Reference/test path only
        -- the whole point of keeping ``FP8Tensor`` around is to *avoid* this
        on the hot path."""
        import torch

        from ..weights import dequant_block128

        return dequant_block128(
            self.weight, self.scale_inv, out_dtype=out_dtype or torch.bfloat16
        )


@dataclass
class GDNFusedWeights:
    layer_idx: int
    in_proj_qkvz: FP8Tensor  # [16384, 5120]
    in_proj_ba: "torch.Tensor"  # [96, 5120] bf16 (b(48) ; a(48))
    out_proj: FP8Tensor  # [5120, 6144]
    conv1d_weight: "torch.Tensor"  # [10240, 1, 4] bf16
    A_log: "torch.Tensor"  # [48]
    dt_bias: "torch.Tensor"  # [48]
    norm_weight: "torch.Tensor"  # [128]


@dataclass
class AttnFusedWeights:
    layer_idx: int
    qkv_proj: FP8Tensor  # [14336, 5120]  (q(12288) ; k(1024) ; v(1024))
    o_proj: FP8Tensor  # [5120, 6144]
    q_norm: "torch.Tensor"  # [256]
    k_norm: "torch.Tensor"  # [256]


@dataclass
class MLPFusedWeights:
    layer_idx: int
    gate_up_proj: FP8Tensor  # [34816, 5120]  (gate(17408) ; up(17408))
    down_proj: FP8Tensor  # [5120, 17408]


@dataclass
class MTPFusedWeights:
    fc_weight: "torch.Tensor"  # [5120, 10240] bf16
    pre_fc_norm_embedding: "torch.Tensor"
    pre_fc_norm_hidden: "torch.Tensor"
    norm: "torch.Tensor"
    input_layernorm: "torch.Tensor"
    post_attention_layernorm: "torch.Tensor"
    attn: AttnFusedWeights
    mlp: MLPFusedWeights


@dataclass
class FusedModelWeights:
    """The whole fused per-layer weight set for one checkpoint snapshot."""

    config: "object"  # QwenFastConfig; typed loosely to avoid an eager import
    embed_tokens: "torch.Tensor"  # bf16 [vocab, hidden]
    lm_head: "torch.Tensor"  # bf16 [vocab, hidden]
    final_norm: "torch.Tensor"  # [hidden]
    layernorms: Dict[int, Tuple["torch.Tensor", "torch.Tensor"]]  # layer -> (input, post_attn)
    gdn: Dict[int, GDNFusedWeights]  # linear_attention layers only
    attn: Dict[int, AttnFusedWeights]  # full_attention layers only
    mlp: Dict[int, MLPFusedWeights]  # every layer (0..num_hidden_layers-1)
    mtp: Optional[MTPFusedWeights] = None
    lm_head_fp8: Optional[FP8Tensor] = None  # only set when quantize_lm_head=True

    # -- introspection -------------------------------------------------- #
    def nbytes(self) -> Dict[str, int]:
        """Byte inventory per component (gdn, attn, mlp, embed/lm_head, mtp, total)."""

        def sz(t) -> int:
            return int(t.numel()) * t.element_size() if t is not None else 0

        def fp8_sz(t) -> int:
            # `FusedModelWeights` accepts a plain tensor wherever it accepts an
            # `FP8Tensor` (that is the CPU/no-quant path
            # `runtime.fused_model.fused_weights_from_module` builds), so this
            # inventory has to handle both or it AttributeErrors on the tiny
            # test model -- which is exactly the model the memory-plan test
            # measures against.
            if isinstance(t, FP8Tensor):
                return sz(t.weight) + sz(t.scale_inv)
            return sz(t)

        gdn_bytes = sum(
            fp8_sz(g.in_proj_qkvz) + sz(g.in_proj_ba) + fp8_sz(g.out_proj)
            + sz(g.conv1d_weight) + sz(g.A_log) + sz(g.dt_bias) + sz(g.norm_weight)
            for g in self.gdn.values()
        )
        attn_bytes = sum(
            fp8_sz(a.qkv_proj) + fp8_sz(a.o_proj) + sz(a.q_norm) + sz(a.k_norm)
            for a in self.attn.values()
        )
        mlp_bytes = sum(fp8_sz(m.gate_up_proj) + fp8_sz(m.down_proj) for m in self.mlp.values())
        embed_lm_head = sz(self.embed_tokens) + sz(self.lm_head)
        mtp_bytes = 0
        if self.mtp is not None:
            mtp_bytes = (
                sz(self.mtp.fc_weight)
                + sz(self.mtp.pre_fc_norm_embedding)
                + sz(self.mtp.pre_fc_norm_hidden)
                + sz(self.mtp.norm)
                + sz(self.mtp.input_layernorm)
                + sz(self.mtp.post_attention_layernorm)
                + fp8_sz(self.mtp.attn.qkv_proj)
                + fp8_sz(self.mtp.attn.o_proj)
                + sz(self.mtp.attn.q_norm)
                + sz(self.mtp.attn.k_norm)
                + fp8_sz(self.mtp.mlp.gate_up_proj)
                + fp8_sz(self.mtp.mlp.down_proj)
            )
        return {
            "gdn": gdn_bytes,
            "attn": attn_bytes,
            "mlp": mlp_bytes,
            "embed_lm_head": embed_lm_head,
            "mtp": mtp_bytes,
            "total": gdn_bytes + attn_bytes + mlp_bytes + embed_lm_head + mtp_bytes,
        }


# --------------------------------------------------------------------------- #
# fusion primitives
# --------------------------------------------------------------------------- #
def _assert_block_aligned(n: int, k: int, label: str) -> None:
    if n % BLOCK != 0 or k % BLOCK != 0:
        raise ValueError(
            f"{label}: shape [{n}, {k}] is not block-{BLOCK}-aligned; fusion "
            "requires every part's N and K to be a multiple of 128 so the "
            "weight_scale_inv grids concatenate cleanly."
        )


def fuse_fp8_rows(store, ckpt_names: Sequence[str], label: str = "") -> FP8Tensor:
    """Concatenate ``len(ckpt_names)`` block-128 FP8 tensors along N (axis 0).

    Each part is read straight off the safetensors store (never dequantized).
    Every part must share K, and every part's N must be a multiple of 128 --
    both hold for every fusion in the module docstring's table. The fused
    ``weight_scale_inv`` is simply the row-concatenation of the parts'
    scales, because block boundaries are preserved by a block-aligned N-cat.
    """
    import torch

    weights: List["torch.Tensor"] = []
    scales: List["torch.Tensor"] = []
    k_ref: Optional[int] = None
    for name in ckpt_names:
        w = store.raw(name)
        s_key = name + "_scale_inv"
        if not store.has(s_key):
            raise KeyError(f"{name} has no {s_key} -- not an FP8 block-quantized tensor")
        s = store.raw(s_key)
        n, k = w.shape
        _assert_block_aligned(n, k, f"{label or 'fuse_fp8_rows'}:{name}")
        if k_ref is None:
            k_ref = k
        elif k != k_ref:
            raise ValueError(f"fuse_fp8_rows: K mismatch {k} != {k_ref} for {name}")
        if tuple(s.shape) != (n // BLOCK, k // BLOCK):
            raise ValueError(
                f"{name}: weight_scale_inv shape {tuple(s.shape)} != expected "
                f"{(n // BLOCK, k // BLOCK)}"
            )
        weights.append(w)
        scales.append(s)
    weight = torch.cat(weights, dim=0).contiguous()
    scale = torch.cat(scales, dim=0).contiguous()
    return FP8Tensor(weight=weight, scale_inv=scale)


def fuse_bf16_rows(store, ckpt_names: Sequence[str]) -> "torch.Tensor":
    """Plain (unquantized) row-concat, for parts too narrow to be block-128
    aligned (``in_proj_b``/``in_proj_a``, 48 rows each)."""
    import torch

    return torch.cat([store.raw(name) for name in ckpt_names], dim=0).contiguous()


def quantize_bf16_to_fp8_block128(w: "torch.Tensor", block: int = BLOCK) -> FP8Tensor:
    """Quantize a bf16 ``[N, K]`` weight to block-128 FP8, matching the
    checkpoint's own convention (``fmt=e4m3``, ``weight_block_size=[128,128]``,
    per ``engine/reference/config-Qwen3.8-27B-FP8.json``). Used for the
    optional fp8 ``lm_head``."""
    import torch

    n, k = w.shape
    if n % block != 0 or k % block != 0:
        raise ValueError(f"quantize_bf16_to_fp8_block128: [{n},{k}] not block-{block}-aligned")
    wf = w.detach().float().view(n // block, block, k // block, block)
    amax = wf.abs().amax(dim=(1, 3)).clamp(min=1e-8)
    scale = amax / FP8_MAX
    wq = (
        (wf / scale[:, None, :, None])
        .clamp(-FP8_MAX, FP8_MAX)
        .to(torch.float8_e4m3fn)
        .reshape(n, k)
        .contiguous()
    )
    return FP8Tensor(weight=wq, scale_inv=scale.to(torch.bfloat16).contiguous())


# --------------------------------------------------------------------------- #
# per-layer builders
# --------------------------------------------------------------------------- #
def _build_gdn_layer(store, layer_prefix: str, layer_idx: int) -> GDNFusedWeights:
    la = layer_prefix + "linear_attn."
    qkvz = fuse_fp8_rows(store, [la + "in_proj_qkv.weight", la + "in_proj_z.weight"], label=la)
    ba = fuse_bf16_rows(store, [la + "in_proj_b.weight", la + "in_proj_a.weight"])
    out_proj = FP8Tensor(
        weight=store.raw(la + "out_proj.weight"), scale_inv=store.raw(la + "out_proj.weight_scale_inv")
    )
    return GDNFusedWeights(
        layer_idx=layer_idx,
        in_proj_qkvz=qkvz,
        in_proj_ba=ba,
        out_proj=out_proj,
        conv1d_weight=store.raw(la + "conv1d.weight"),
        A_log=store.raw(la + "A_log"),
        dt_bias=store.raw(la + "dt_bias"),
        norm_weight=store.raw(la + "norm.weight"),
    )


def _build_attn_layer(store, layer_prefix: str, layer_idx: int) -> AttnFusedWeights:
    sa = layer_prefix + "self_attn."
    qkv = fuse_fp8_rows(
        store, [sa + "q_proj.weight", sa + "k_proj.weight", sa + "v_proj.weight"], label=sa
    )
    o_proj = FP8Tensor(
        weight=store.raw(sa + "o_proj.weight"), scale_inv=store.raw(sa + "o_proj.weight_scale_inv")
    )
    return AttnFusedWeights(
        layer_idx=layer_idx,
        qkv_proj=qkv,
        o_proj=o_proj,
        q_norm=store.raw(sa + "q_norm.weight"),
        k_norm=store.raw(sa + "k_norm.weight"),
    )


def _build_mlp(store, layer_prefix: str, layer_idx: int) -> MLPFusedWeights:
    mp = layer_prefix + "mlp."
    gate_up = fuse_fp8_rows(store, [mp + "gate_proj.weight", mp + "up_proj.weight"], label=mp)
    down = FP8Tensor(
        weight=store.raw(mp + "down_proj.weight"), scale_inv=store.raw(mp + "down_proj.weight_scale_inv")
    )
    return MLPFusedWeights(layer_idx=layer_idx, gate_up_proj=gate_up, down_proj=down)


def _build_mtp(store, config) -> Optional[MTPFusedWeights]:
    if not store.has("mtp.fc.weight"):
        return None
    layer_prefix = "mtp.layers.0."
    attn = _build_attn_layer(store, layer_prefix, config.mtp_layer_idx)
    mlp = _build_mlp(store, layer_prefix, config.mtp_layer_idx)
    return MTPFusedWeights(
        fc_weight=store.raw("mtp.fc.weight"),
        pre_fc_norm_embedding=store.raw("mtp.pre_fc_norm_embedding.weight"),
        pre_fc_norm_hidden=store.raw("mtp.pre_fc_norm_hidden.weight"),
        norm=store.raw("mtp.norm.weight"),
        input_layernorm=store.raw(layer_prefix + "input_layernorm.weight"),
        post_attention_layernorm=store.raw(layer_prefix + "post_attention_layernorm.weight"),
        attn=attn,
        mlp=mlp,
    )


# --------------------------------------------------------------------------- #
# top-level builder
# --------------------------------------------------------------------------- #
def build_fused_weights(
    model_dir: str,
    device: str = "cpu",
    include_mtp: bool = True,
    quantize_lm_head: bool = False,
    verbose: bool = False,
) -> FusedModelWeights:
    """Build the fused weight set straight from a safetensors snapshot.

    ``device`` is the device tensors are *read onto* (``SafetensorsStore``'s
    device) -- pass ``"cuda:0"`` on the GPU host to skip an extra H2D copy.
    Every FP8 tensor stays FP8 the whole way through; nothing is dequantized
    (aside from the optional ``quantize_lm_head`` path, which quantizes the
    checkpoint's bf16 ``lm_head`` *into* FP8, not the other way around).
    """
    from ..weights import QwenFastConfig, SafetensorsStore, resolve_snapshot

    model_dir = resolve_snapshot(model_dir)
    config = QwenFastConfig.from_pretrained(model_dir)
    store = SafetensorsStore(model_dir, device=device)

    gdn: Dict[int, GDNFusedWeights] = {}
    attn: Dict[int, AttnFusedWeights] = {}
    mlp: Dict[int, MLPFusedWeights] = {}
    layernorms: Dict[int, Tuple["torch.Tensor", "torch.Tensor"]] = {}

    for i in range(config.num_hidden_layers):
        prefix = f"model.language_model.layers.{i}."
        layernorms[i] = (
            store.raw(prefix + "input_layernorm.weight"),
            store.raw(prefix + "post_attention_layernorm.weight"),
        )
        if config.layer_types[i] == "linear_attention":
            gdn[i] = _build_gdn_layer(store, prefix, i)
        else:
            attn[i] = _build_attn_layer(store, prefix, i)
        mlp[i] = _build_mlp(store, prefix, i)
        if verbose:
            print(f"[fused_weights] layer {i} ({config.layer_types[i]}) fused")

    embed_tokens = store.raw("model.language_model.embed_tokens.weight")
    lm_head = store.raw("lm_head.weight")
    final_norm = store.raw("model.language_model.norm.weight")

    lm_head_fp8 = quantize_bf16_to_fp8_block128(lm_head) if quantize_lm_head else None

    mtp = _build_mtp(store, config) if include_mtp else None
    if verbose:
        print(f"[fused_weights] mtp {'present' if mtp is not None else 'absent/skipped'}")

    store.close()
    return FusedModelWeights(
        config=config,
        embed_tokens=embed_tokens,
        lm_head=lm_head,
        final_norm=final_norm,
        layernorms=layernorms,
        gdn=gdn,
        attn=attn,
        mlp=mlp,
        mtp=mtp,
        lm_head_fp8=lm_head_fp8,
    )


# --------------------------------------------------------------------------- #
# safetensors cache: save_fused / load_fused
# --------------------------------------------------------------------------- #
def _flatten(fw: FusedModelWeights) -> Dict[str, "torch.Tensor"]:
    out: Dict[str, "torch.Tensor"] = {
        "embed_tokens.weight": fw.embed_tokens,
        "lm_head.weight": fw.lm_head,
        "model.norm.weight": fw.final_norm,
    }
    if fw.lm_head_fp8 is not None:
        out["lm_head.weight_fp8"] = fw.lm_head_fp8.weight
        out["lm_head.weight_fp8_scale_inv"] = fw.lm_head_fp8.scale_inv

    for i, (ln1, ln2) in fw.layernorms.items():
        out[f"layers.{i}.input_layernorm.weight"] = ln1
        out[f"layers.{i}.post_attention_layernorm.weight"] = ln2

    for i, g in fw.gdn.items():
        p = f"layers.{i}.linear_attn."
        out[p + "in_proj_qkvz.weight"] = g.in_proj_qkvz.weight
        out[p + "in_proj_qkvz.weight_scale_inv"] = g.in_proj_qkvz.scale_inv
        out[p + "in_proj_ba.weight"] = g.in_proj_ba
        out[p + "out_proj.weight"] = g.out_proj.weight
        out[p + "out_proj.weight_scale_inv"] = g.out_proj.scale_inv
        out[p + "conv1d.weight"] = g.conv1d_weight
        out[p + "A_log"] = g.A_log
        out[p + "dt_bias"] = g.dt_bias
        out[p + "norm.weight"] = g.norm_weight

    for i, a in fw.attn.items():
        p = f"layers.{i}.self_attn."
        out[p + "qkv_proj.weight"] = a.qkv_proj.weight
        out[p + "qkv_proj.weight_scale_inv"] = a.qkv_proj.scale_inv
        out[p + "o_proj.weight"] = a.o_proj.weight
        out[p + "o_proj.weight_scale_inv"] = a.o_proj.scale_inv
        out[p + "q_norm.weight"] = a.q_norm
        out[p + "k_norm.weight"] = a.k_norm

    for i, m in fw.mlp.items():
        p = f"layers.{i}.mlp."
        out[p + "gate_up_proj.weight"] = m.gate_up_proj.weight
        out[p + "gate_up_proj.weight_scale_inv"] = m.gate_up_proj.scale_inv
        out[p + "down_proj.weight"] = m.down_proj.weight
        out[p + "down_proj.weight_scale_inv"] = m.down_proj.scale_inv

    if fw.mtp is not None:
        out["mtp.fc.weight"] = fw.mtp.fc_weight
        out["mtp.pre_fc_norm_embedding.weight"] = fw.mtp.pre_fc_norm_embedding
        out["mtp.pre_fc_norm_hidden.weight"] = fw.mtp.pre_fc_norm_hidden
        out["mtp.norm.weight"] = fw.mtp.norm
        out["mtp.input_layernorm.weight"] = fw.mtp.input_layernorm
        out["mtp.post_attention_layernorm.weight"] = fw.mtp.post_attention_layernorm
        pa, pm = "mtp.self_attn.", "mtp.mlp."
        out[pa + "qkv_proj.weight"] = fw.mtp.attn.qkv_proj.weight
        out[pa + "qkv_proj.weight_scale_inv"] = fw.mtp.attn.qkv_proj.scale_inv
        out[pa + "o_proj.weight"] = fw.mtp.attn.o_proj.weight
        out[pa + "o_proj.weight_scale_inv"] = fw.mtp.attn.o_proj.scale_inv
        out[pa + "q_norm.weight"] = fw.mtp.attn.q_norm
        out[pa + "k_norm.weight"] = fw.mtp.attn.k_norm
        out[pm + "gate_up_proj.weight"] = fw.mtp.mlp.gate_up_proj.weight
        out[pm + "gate_up_proj.weight_scale_inv"] = fw.mtp.mlp.gate_up_proj.scale_inv
        out[pm + "down_proj.weight"] = fw.mtp.mlp.down_proj.weight
        out[pm + "down_proj.weight_scale_inv"] = fw.mtp.mlp.down_proj.scale_inv

    return {k: v.contiguous() for k, v in out.items()}


def save_fused(fw: FusedModelWeights, out_dir: str) -> str:
    """Persist ``fw`` as a single safetensors file plus JSON metadata (stored
    in the safetensors header) so :func:`load_fused` can reconstruct it
    without re-reading the original checkpoint. Returns the file path."""
    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)
    tensors = _flatten(fw)
    meta = {
        "format": FUSED_FORMAT_VERSION,
        "config": json.dumps(dataclasses.asdict(fw.config)),
        "has_mtp": json.dumps(fw.mtp is not None),
        "has_lm_head_fp8": json.dumps(fw.lm_head_fp8 is not None),
    }
    path = os.path.join(out_dir, FUSED_WEIGHTS_FILENAME)
    save_file(tensors, path, metadata=meta)
    return path


def load_fused(in_dir: str, device: str = "cpu") -> FusedModelWeights:
    """Load a fused weight set previously written by :func:`save_fused`."""
    from safetensors import safe_open

    from ..weights import QwenFastConfig

    path = in_dir if in_dir.endswith(".safetensors") else os.path.join(in_dir, FUSED_WEIGHTS_FILENAME)
    with safe_open(path, framework="pt", device=device) as f:
        meta = f.metadata() or {}
        tensors = {k: f.get_tensor(k) for k in f.keys()}

    if meta.get("format") != FUSED_FORMAT_VERSION:
        raise ValueError(
            f"{path}: unrecognized fused-weights format {meta.get('format')!r} "
            f"(expected {FUSED_FORMAT_VERSION!r})"
        )
    config = QwenFastConfig(**json.loads(meta["config"]))
    has_mtp = json.loads(meta.get("has_mtp", "false"))
    has_lm_head_fp8 = json.loads(meta.get("has_lm_head_fp8", "false"))

    lm_head_fp8 = None
    if has_lm_head_fp8:
        lm_head_fp8 = FP8Tensor(tensors["lm_head.weight_fp8"], tensors["lm_head.weight_fp8_scale_inv"])

    layernorms: Dict[int, Tuple["torch.Tensor", "torch.Tensor"]] = {}
    gdn: Dict[int, GDNFusedWeights] = {}
    attn: Dict[int, AttnFusedWeights] = {}
    mlp: Dict[int, MLPFusedWeights] = {}
    for i in range(config.num_hidden_layers):
        p = f"layers.{i}."
        layernorms[i] = (
            tensors[p + "input_layernorm.weight"],
            tensors[p + "post_attention_layernorm.weight"],
        )
        if config.layer_types[i] == "linear_attention":
            la = p + "linear_attn."
            gdn[i] = GDNFusedWeights(
                layer_idx=i,
                in_proj_qkvz=FP8Tensor(
                    tensors[la + "in_proj_qkvz.weight"], tensors[la + "in_proj_qkvz.weight_scale_inv"]
                ),
                in_proj_ba=tensors[la + "in_proj_ba.weight"],
                out_proj=FP8Tensor(
                    tensors[la + "out_proj.weight"], tensors[la + "out_proj.weight_scale_inv"]
                ),
                conv1d_weight=tensors[la + "conv1d.weight"],
                A_log=tensors[la + "A_log"],
                dt_bias=tensors[la + "dt_bias"],
                norm_weight=tensors[la + "norm.weight"],
            )
        else:
            sa = p + "self_attn."
            attn[i] = AttnFusedWeights(
                layer_idx=i,
                qkv_proj=FP8Tensor(
                    tensors[sa + "qkv_proj.weight"], tensors[sa + "qkv_proj.weight_scale_inv"]
                ),
                o_proj=FP8Tensor(tensors[sa + "o_proj.weight"], tensors[sa + "o_proj.weight_scale_inv"]),
                q_norm=tensors[sa + "q_norm.weight"],
                k_norm=tensors[sa + "k_norm.weight"],
            )
        mp = p + "mlp."
        mlp[i] = MLPFusedWeights(
            layer_idx=i,
            gate_up_proj=FP8Tensor(
                tensors[mp + "gate_up_proj.weight"], tensors[mp + "gate_up_proj.weight_scale_inv"]
            ),
            down_proj=FP8Tensor(tensors[mp + "down_proj.weight"], tensors[mp + "down_proj.weight_scale_inv"]),
        )

    mtp = None
    if has_mtp:
        pa, pm = "mtp.self_attn.", "mtp.mlp."
        mtp = MTPFusedWeights(
            fc_weight=tensors["mtp.fc.weight"],
            pre_fc_norm_embedding=tensors["mtp.pre_fc_norm_embedding.weight"],
            pre_fc_norm_hidden=tensors["mtp.pre_fc_norm_hidden.weight"],
            norm=tensors["mtp.norm.weight"],
            input_layernorm=tensors["mtp.input_layernorm.weight"],
            post_attention_layernorm=tensors["mtp.post_attention_layernorm.weight"],
            attn=AttnFusedWeights(
                layer_idx=config.mtp_layer_idx,
                qkv_proj=FP8Tensor(
                    tensors[pa + "qkv_proj.weight"], tensors[pa + "qkv_proj.weight_scale_inv"]
                ),
                o_proj=FP8Tensor(tensors[pa + "o_proj.weight"], tensors[pa + "o_proj.weight_scale_inv"]),
                q_norm=tensors[pa + "q_norm.weight"],
                k_norm=tensors[pa + "k_norm.weight"],
            ),
            mlp=MLPFusedWeights(
                layer_idx=config.mtp_layer_idx,
                gate_up_proj=FP8Tensor(
                    tensors[pm + "gate_up_proj.weight"], tensors[pm + "gate_up_proj.weight_scale_inv"]
                ),
                down_proj=FP8Tensor(
                    tensors[pm + "down_proj.weight"], tensors[pm + "down_proj.weight_scale_inv"]
                ),
            ),
        )

    return FusedModelWeights(
        config=config,
        embed_tokens=tensors["embed_tokens.weight"],
        lm_head=tensors["lm_head.weight"],
        final_norm=tensors["model.norm.weight"],
        layernorms=layernorms,
        gdn=gdn,
        attn=attn,
        mlp=mlp,
        mtp=mtp,
        lm_head_fp8=lm_head_fp8,
    )


__all__ = [
    "BLOCK",
    "FP8_MAX",
    "FUSED_FORMAT_VERSION",
    "FUSED_WEIGHTS_FILENAME",
    "FP8Tensor",
    "GDNFusedWeights",
    "AttnFusedWeights",
    "MLPFusedWeights",
    "MTPFusedWeights",
    "FusedModelWeights",
    "fuse_fp8_rows",
    "fuse_bf16_rows",
    "quantize_bf16_to_fp8_block128",
    "build_fused_weights",
    "save_fused",
    "load_fused",
]
