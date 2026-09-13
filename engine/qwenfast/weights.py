"""qwenfast reference model: checkpoint loading for Qwen3.8-27B (BF16 and block-128 FP8).

Design notes
------------
* Our module tree is named so that our ``state_dict`` keys differ from the
  checkpoint keys by exactly one rewrite: ``model.`` -> ``model.language_model.``
  (see :func:`our_name_to_ckpt_name`).  Keeping the names aligned removes a
  whole class of silent mis-mapping bugs.
* FP8 checkpoints store ``<w>.weight`` as ``float8_e4m3fn`` of shape ``[N, K]``
  plus ``<w>.weight_scale_inv`` of shape ``[ceil(N/128), ceil(K/128)]`` in
  fp32.  The reference model dequantizes to bf16 **at load time** (`w * scale`),
  so it is identical to the BF16 one.  The fused runtime keeps the FP8 tensors
  and does the scaling inside the GEMM.
* All Qwen3.8-27B linear shapes are multiples of 128 in both dims, so the fast
  ``view``-based dequant path is always taken; a padded fallback exists anyway.
* ``visual.*`` is skipped entirely (the engine serves text only).
"""

from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import torch

try:  # safetensors is required to load real checkpoints, but not to import.
    from safetensors import safe_open
except Exception:  # pragma: no cover - exercised only without safetensors
    safe_open = None  # type: ignore[assignment]


BLOCK = 128


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
@dataclass
class QwenFastConfig:
    """Flattened ``text_config`` of Qwen3.8-27B."""

    hidden_size: int = 5120
    intermediate_size: int = 17408
    num_hidden_layers: int = 64
    vocab_size: int = 248320
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"

    # full attention
    num_attention_heads: int = 24
    num_key_value_heads: int = 4
    head_dim: int = 256
    attention_bias: bool = False
    attn_output_gate: bool = True
    full_attention_interval: int = 4

    # gated deltanet
    linear_num_value_heads: int = 48
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    mamba_ssm_dtype: str = "float32"

    # rope
    rope_theta: float = 1e7
    partial_rotary_factor: float = 0.25
    mrope_section: List[int] = field(default_factory=lambda: [11, 11, 10])
    mrope_interleaved: bool = True
    max_position_embeddings: int = 262144

    # misc
    layer_types: List[str] = field(default_factory=list)
    mtp_num_hidden_layers: int = 1
    mtp_use_dedicated_embeddings: bool = False
    bos_token_id: int = 248044
    eos_token_id: int = 248044
    tie_word_embeddings: bool = False
    quant_method: Optional[str] = None  # "fp8" for the FP8 checkpoint

    # ---- derived ---------------------------------------------------------- #
    @property
    def key_dim(self) -> int:
        return self.linear_key_head_dim * self.linear_num_key_heads  # 2048

    @property
    def value_dim(self) -> int:
        return self.linear_value_head_dim * self.linear_num_value_heads  # 6144

    @property
    def conv_dim(self) -> int:
        return self.key_dim * 2 + self.value_dim  # 10240

    @property
    def num_v_per_k(self) -> int:
        return self.linear_num_value_heads // self.linear_num_key_heads  # 3

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)  # 64

    @property
    def attention_layer_indices(self) -> List[int]:
        return [i for i, t in enumerate(self.layer_types) if t == "full_attention"]

    @property
    def linear_layer_indices(self) -> List[int]:
        return [i for i, t in enumerate(self.layer_types) if t == "linear_attention"]

    @property
    def mtp_layer_idx(self) -> int:
        """Layer index used by the MTP block's KV-cache slot."""
        return self.num_hidden_layers

    # ---- constructors ------------------------------------------------------ #
    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "QwenFastConfig":
        text = raw.get("text_config", raw)
        rope = text.get("rope_parameters", {}) or {}
        quant = raw.get("quantization_config") or text.get("quantization_config") or {}
        cfg = cls(
            hidden_size=text["hidden_size"],
            intermediate_size=text["intermediate_size"],
            num_hidden_layers=text["num_hidden_layers"],
            vocab_size=text["vocab_size"],
            rms_norm_eps=text.get("rms_norm_eps", 1e-6),
            hidden_act=text.get("hidden_act", "silu"),
            num_attention_heads=text["num_attention_heads"],
            num_key_value_heads=text["num_key_value_heads"],
            head_dim=text.get("head_dim", text["hidden_size"] // text["num_attention_heads"]),
            attention_bias=text.get("attention_bias", False),
            attn_output_gate=text.get("attn_output_gate", True),
            full_attention_interval=text.get("full_attention_interval", 4),
            linear_num_value_heads=text["linear_num_value_heads"],
            linear_num_key_heads=text["linear_num_key_heads"],
            linear_key_head_dim=text["linear_key_head_dim"],
            linear_value_head_dim=text["linear_value_head_dim"],
            linear_conv_kernel_dim=text.get("linear_conv_kernel_dim", 4),
            mamba_ssm_dtype=text.get("mamba_ssm_dtype", "float32"),
            rope_theta=rope.get("rope_theta", text.get("rope_theta", 1e7)),
            partial_rotary_factor=rope.get(
                "partial_rotary_factor", text.get("partial_rotary_factor", 1.0)
            ),
            mrope_section=list(rope.get("mrope_section", [11, 11, 10])),
            mrope_interleaved=bool(rope.get("mrope_interleaved", True)),
            max_position_embeddings=text.get("max_position_embeddings", 262144),
            layer_types=list(text.get("layer_types", [])),
            mtp_num_hidden_layers=text.get("mtp_num_hidden_layers", 0),
            mtp_use_dedicated_embeddings=text.get("mtp_use_dedicated_embeddings", False),
            bos_token_id=text.get("bos_token_id", 248044),
            eos_token_id=text.get("eos_token_id", 248044),
            tie_word_embeddings=text.get("tie_word_embeddings", False),
            quant_method=quant.get("quant_method"),
        )
        if not cfg.layer_types:
            iv = cfg.full_attention_interval
            cfg.layer_types = [
                "full_attention" if (i + 1) % iv == 0 else "linear_attention"
                for i in range(cfg.num_hidden_layers)
            ]
        return cfg

    @classmethod
    def from_json(cls, path: str) -> "QwenFastConfig":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_pretrained(cls, model_dir: str) -> "QwenFastConfig":
        return cls.from_json(os.path.join(model_dir, "config.json"))


# --------------------------------------------------------------------------- #
# fp8 dequant
# --------------------------------------------------------------------------- #
def dequant_block128(
    w: torch.Tensor, scale_inv: torch.Tensor, out_dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """Dequantize a block-128 FP8 weight.

    ``w`` is ``[N, K]`` (float8_e4m3fn), ``scale_inv`` is
    ``[ceil(N/128), ceil(K/128)]`` fp32; the value at ``[n, k]`` is scaled by
    ``scale_inv[n // 128, k // 128]``.
    """
    if w.ndim != 2:
        raise ValueError(f"expected a 2-D weight, got shape {tuple(w.shape)}")
    n, k = w.shape
    s = scale_inv.to(torch.float32)
    wf = w.to(torch.float32)
    if n % BLOCK == 0 and k % BLOCK == 0:
        out = wf.view(n // BLOCK, BLOCK, k // BLOCK, BLOCK)
        out = out * s[:, None, :, None]
        return out.reshape(n, k).to(out_dtype)
    # generic (padded) fallback
    s_full = s.repeat_interleave(BLOCK, dim=0).repeat_interleave(BLOCK, dim=1)[:n, :k]
    return (wf * s_full).to(out_dtype)


# --------------------------------------------------------------------------- #
# name mapping
# --------------------------------------------------------------------------- #
_LM_PREFIX = "model.language_model."


def our_name_to_ckpt_name(name: str) -> str:
    """qwenfast parameter name -> checkpoint tensor name."""
    if name.startswith("model."):
        return _LM_PREFIX + name[len("model.") :]
    return name  # lm_head.*, mtp.*


def ckpt_name_to_our_name(name: str) -> str:
    if name.startswith(_LM_PREFIX):
        return "model." + name[len(_LM_PREFIX) :]
    return name


# --------------------------------------------------------------------------- #
# weight store
# --------------------------------------------------------------------------- #
class SafetensorsStore:
    """Random access over a sharded safetensors snapshot, with FP8 dequant."""

    def __init__(self, model_dir: str, device: str = "cpu"):
        if safe_open is None:
            raise RuntimeError("safetensors is not installed")
        self.model_dir = model_dir
        self.device = device
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                self.weight_map: Dict[str, str] = json.load(f)["weight_map"]
        else:
            shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
            if not shards:
                raise FileNotFoundError(f"no safetensors under {model_dir}")
            self.weight_map = {}
            for shard in shards:
                with safe_open(shard, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        self.weight_map[key] = os.path.basename(shard)
        self._handles: Dict[str, Any] = {}

    # -- low level ---------------------------------------------------------- #
    def _handle(self, shard: str):
        h = self._handles.get(shard)
        if h is None:
            h = safe_open(os.path.join(self.model_dir, shard), framework="pt", device=self.device)
            self._handles[shard] = h
        return h

    def has(self, ckpt_name: str) -> bool:
        return ckpt_name in self.weight_map

    def raw(self, ckpt_name: str) -> torch.Tensor:
        shard = self.weight_map[ckpt_name]
        return self._handle(shard).get_tensor(ckpt_name)

    def keys(self) -> Iterable[str]:
        return self.weight_map.keys()

    # -- high level --------------------------------------------------------- #
    def get(self, ckpt_name: str, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        """Fetch a tensor, dequantizing block-128 FP8 when a scale is present."""
        if not self.has(ckpt_name):
            raise KeyError(ckpt_name)
        t = self.raw(ckpt_name)
        scale_key = ckpt_name + "_scale_inv"
        if self.has(scale_key):
            return dequant_block128(t, self.raw(scale_key), out_dtype=dtype)
        if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            # per-tensor / dynamic-activation FP8 without a block scale
            return t.to(dtype)
        if t.is_floating_point():
            return t.to(dtype)
        return t

    def close(self) -> None:
        self._handles.clear()


def resolve_snapshot(path: str) -> str:
    """Accept a snapshot dir, a ``models--*`` dir, or a glob and return a dir."""
    if os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json")):
        return path
    cands = sorted(glob.glob(os.path.join(path, "snapshots", "*")))
    cands = [c for c in cands if os.path.exists(os.path.join(c, "config.json"))]
    if cands:
        return cands[-1]
    cands = sorted(c for c in glob.glob(path) if os.path.exists(os.path.join(c, "config.json")))
    if cands:
        return cands[-1]
    raise FileNotFoundError(f"no snapshot with config.json under {path!r}")


# --------------------------------------------------------------------------- #
# model loading
# --------------------------------------------------------------------------- #
_SKIP_RE = re.compile(r"(^|\.)visual\.")


def _split(name: str):
    parent, _, leaf = name.rpartition(".")
    return parent, leaf


def _resolve_attr(model: torch.nn.Module, name: str) -> torch.Tensor:
    parent, leaf = _split(name)
    mod = model.get_submodule(parent) if parent else model
    return getattr(mod, leaf)


def _assign_attr(model: torch.nn.Module, name: str, tensor: torch.Tensor) -> None:
    parent, leaf = _split(name)
    mod = model.get_submodule(parent) if parent else model
    cur = getattr(mod, leaf)
    if isinstance(cur, torch.nn.Parameter):
        setattr(mod, leaf, torch.nn.Parameter(tensor, requires_grad=False))
    else:
        mod._buffers[leaf] = tensor


def load_into_model(
    model: torch.nn.Module,
    store: SafetensorsStore,
    dtype: torch.dtype = torch.bfloat16,
    strict: bool = True,
    verbose: bool = False,
) -> None:
    """Populate ``model`` (possibly on ``meta``) from ``store``, in place."""
    wanted = set(model.state_dict().keys())
    missing: List[str] = []
    nbytes = 0
    n_loaded = 0
    # Assign one tensor at a time so we never hold two copies of the model.
    for name in sorted(wanted):
        if _SKIP_RE.search(name):
            continue
        ckpt = our_name_to_ckpt_name(name)
        if not store.has(ckpt):
            missing.append(f"{name} (looked for {ckpt})")
            continue
        target = _resolve_attr(model, name)
        t = store.get(ckpt, dtype=dtype)
        if tuple(t.shape) != tuple(target.shape):
            # conv1d is [C, 1, K] in the checkpoint; keep our shape authoritative
            if t.numel() == target.numel():
                t = t.reshape(target.shape)
            else:
                raise ValueError(
                    f"shape mismatch for {name}: ckpt {tuple(t.shape)} vs model {tuple(target.shape)}"
                )
        _assign_attr(model, name, t)
        nbytes += t.numel() * t.element_size()
        n_loaded += 1
    if missing and strict:
        raise KeyError("missing checkpoint tensors:\n  " + "\n  ".join(missing[:20]))
    if verbose and missing:
        print(f"[qwenfast] {len(missing)} tensors missing (non-strict): {missing[:5]} ...")
    still_meta = [n for n, p in model.named_parameters() if p.is_meta]
    if still_meta and strict:
        raise RuntimeError(f"{len(still_meta)} parameters left on meta, e.g. {still_meta[:5]}")
    if verbose:
        print(f"[qwenfast] loaded {n_loaded} tensors, {nbytes / 2**30:.2f} GiB")


__all__ = [
    "QwenFastConfig",
    "SafetensorsStore",
    "dequant_block128",
    "load_into_model",
    "our_name_to_ckpt_name",
    "ckpt_name_to_our_name",
    "resolve_snapshot",
]
