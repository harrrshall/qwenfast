"""``flash-linear-attention`` (>= 0.5.2) backend.

What v0.5.2 gives us, verified against the tagged source:

* **GVA is native.**  ``q``/``k`` are passed with ``H = 16`` heads and ``v`` /
  ``g`` / ``beta`` with ``HV = 48``; the kernels map ``i_h = i_hv // (HV // H)``,
  i.e. exactly ``repeat_interleave(3)``.  We must **not** pre-expand — doing so
  triples the q/k read for no benefit.
* **State layout** defaults to ``[N, HV, K, V]`` (``state_v_first=False``),
  which is the engine's pool layout.  (vLLM's *fork* is hardcoded V-first; do
  not copy code between them.)
* ``use_qk_l2norm_in_kernel=True`` uses ``x / sqrt(sum(x*x) + 1e-6)`` and
  ``scale`` defaults to ``K ** -0.5`` — both identical to our contract.
* ``cu_seqlens`` requires ``q.shape[0] == 1`` and
  ``initial_state.shape[0] == len(cu_seqlens) - 1``.

What it does **not** give us, and why the Triton backend exists:

* There is **no** ``ssm_state_indices`` / ``inplace_final_state`` upstream
  (those are vLLM-fork additions).  ``final_state`` is a freshly allocated fp32
  tensor every call.  So a pool-based decode costs
  ``gather (read+write) + kernel (read+write) + scatter (read+write)``
  = **3x** the ideal traffic.  At B=256 that is ~2.3 GB/layer/step instead of
  0.77 GB, the single reason this package has a custom decode kernel.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from . import shapes
from .state import gather_states, scatter_states
from .torch_ops import normalize_qkv, collapse_gva, maybe_gate

# fla wants int32 offsets in its Triton kernels; it accepts int64 too but the
# chunk path derives index tensors with `.to(cu_seqlens)`, so int32 is cheaper.
CU_SEQLENS_DTYPE = torch.int32

_CHUNK = None
_RECURRENT = None
_IMPORT_ERROR: Optional[str] = None


def _load():
    global _CHUNK, _RECURRENT, _IMPORT_ERROR
    if _CHUNK is not None or _IMPORT_ERROR is not None:
        return
    try:
        from fla.ops.gated_delta_rule import (  # type: ignore
            chunk_gated_delta_rule,
            fused_recurrent_gated_delta_rule,
        )

        _CHUNK = chunk_gated_delta_rule
        _RECURRENT = fused_recurrent_gated_delta_rule
    except Exception as exc:  # pragma: no cover - depends on the host environment
        _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


def is_available() -> bool:
    _load()
    return _CHUNK is not None


def unavailable_reason() -> Optional[str]:
    _load()
    return _IMPORT_ERROR


def version() -> Optional[str]:
    try:
        import fla  # type: ignore

        return getattr(fla, "__version__", "unknown")
    except Exception:  # pragma: no cover
        return None


#: The chunk sizes fla's gated-delta-rule kernels accept when the installed
#: build takes ``chunk_size`` at all. From ``fla/ops/gated_delta_rule/chunk.py``
#: (v0.5.2): ``if chunk_size not in (16, 32, 64): raise ValueError``.
FLA_CHUNK_SIZES: Tuple[int, ...] = (16, 32, 64)

_SUPPORTS_CHUNK_SIZE: Optional[bool] = None
_SUPPORTS_CU_SEQLENS_CPU: Optional[bool] = None


def supports_cu_seqlens_cpu() -> bool:
    """Does the installed fla take a ``cu_seqlens_cpu=`` keyword?

    A named parameter, so a plain signature check is exact here (see
    :func:`_cu_seqlens_cpu_kwarg` for why it matters)."""
    global _SUPPORTS_CU_SEQLENS_CPU
    if _SUPPORTS_CU_SEQLENS_CPU is not None:
        return _SUPPORTS_CU_SEQLENS_CPU
    _load()
    ok = False
    if _CHUNK is not None:
        try:
            import inspect

            ok = "cu_seqlens_cpu" in inspect.signature(_CHUNK).parameters
        except Exception:  # pragma: no cover -- depends on the host environment
            ok = False
    _SUPPORTS_CU_SEQLENS_CPU = ok
    return ok


def supports_chunk_size() -> bool:
    """Does the installed fla honour a ``chunk_size=`` keyword?

    Older fla releases baked the chunk size into the kernel's autotune config
    and accepted only 64. ``fla-core`` 0.5.2 does not: its wrapper opens with
    ``chunk_size = kwargs.pop('chunk_size', 64)`` and accepts 16/32/64 (``BT =
    chunk_size`` all the way down through ``chunk_fwd.py`` /
    ``chunk_delta_h.py`` / ``chunk_o.py``, with the autotune cache keyed on
    ``BT``). A prefill chunk is the one shape where BT is a real trade -- 64
    is tuned for training-shaped sequences -- so this is a knob the profiler
    can sweep rather than a constant to be asserted.

    Detected from the wrapper's own source rather than from a version string,
    because an install can import ``fla-core`` 0.5.2 while its
    ``flash-linear-attention`` dist-info says 0.5.0 (an editable stub), so
    a version comparison would read the wrong one. Falls back to ``False`` (the
    old, always-64 behaviour) whenever the source is unreadable, so an
    unknown build is never silently given a chunk size it will drop into
    ``**kwargs`` and ignore.
    """
    global _SUPPORTS_CHUNK_SIZE
    if _SUPPORTS_CHUNK_SIZE is not None:
        return _SUPPORTS_CHUNK_SIZE
    _load()
    ok = False
    if _CHUNK is not None:
        import inspect
        import sys

        # 1. the module's own source, which is where `kwargs.pop('chunk_size',
        #    64)` lives. `inspect.getsource(_CHUNK)` does NOT work: fla decorates
        #    the wrapper with `@torch.compiler.disable`, so `getsourcefile`
        #    resolves to `torch/_dynamo/eval_frame.py` and the text that comes
        #    back is dynamo's, not fla's.
        try:
            mod = sys.modules.get(getattr(_CHUNK, "__module__", "") or "")
            src = inspect.getsource(mod) if mod is not None else ""
            ok = "chunk_size" in src
        except Exception:  # pragma: no cover -- depends on the host environment
            src = ""
        # 2. signature fallback: `chunk_size` is consumed from `**kwargs`, so it
        #    never appears as a named parameter -- but the build that introduced
        #    it also introduced `use_beta_sigmoid_in_kernel`, which does.
        if not ok:
            try:
                params = inspect.signature(_CHUNK).parameters
                ok = any(p.kind is p.VAR_KEYWORD for p in params.values()) and (
                    "use_beta_sigmoid_in_kernel" in params
                )
            except Exception:  # pragma: no cover
                ok = False
    _SUPPORTS_CHUNK_SIZE = ok
    return ok


# --------------------------------------------------------------------------- #
def _prep(q, k, v, g, beta):
    """Normalise dims and collapse q/k back to the 16-head GVA form."""
    q, k, v, g, beta = normalize_qkv(q, k, v, g, beta)
    hv = v.shape[2]
    h = q.shape[2]
    if h == hv and hv % shapes.NUM_K_HEADS == 0 and hv != shapes.NUM_K_HEADS:
        # caller pre-expanded; undo it so fla reads q/k once per k-head
        q = collapse_gva(q, shapes.NUM_K_HEADS).contiguous()
        k = collapse_gva(k, shapes.NUM_K_HEADS).contiguous()
    return q, k, v, g, beta


def recurrent_gdn(
    q,
    k,
    v,
    g,
    beta,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
    use_qk_l2norm: bool = True,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    _load()
    if _RECURRENT is None:
        raise RuntimeError(f"fla unavailable: {_IMPORT_ERROR}")
    q, k, v, g, beta = _prep(q, k, v, g, beta)
    return _RECURRENT(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        scale=(q.shape[-1] ** -0.5) if scale is None else scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm,
        cu_seqlens=cu_seqlens,
    )


def chunk_gdn(
    q,
    k,
    v,
    g,
    beta,
    cu_seqlens: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
    use_qk_l2norm: bool = True,
    scale: Optional[float] = None,
    chunk_size: int = shapes.DEFAULT_CHUNK_SIZE,
    cu_seqlens_cpu: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    _load()
    if _CHUNK is None:
        raise RuntimeError(f"fla unavailable: {_IMPORT_ERROR}")
    extra = {}
    if chunk_size != shapes.DEFAULT_CHUNK_SIZE:
        if not supports_chunk_size():
            raise ValueError(
                f"the installed fla ({version()}) bakes its chunk size into the "
                f"kernel; only chunk_size={shapes.DEFAULT_CHUNK_SIZE} is available "
                f"(see `supports_chunk_size`)"
            )
        if chunk_size not in FLA_CHUNK_SIZES:
            raise ValueError(
                f"fla accepts chunk_size in {sorted(FLA_CHUNK_SIZES)}, got {chunk_size}"
            )
        extra["chunk_size"] = int(chunk_size)
    q, k, v, g, beta = _prep(q, k, v, g, beta)
    if cu_seqlens is not None:
        cu_seqlens = cu_seqlens.to(device=q.device, dtype=CU_SEQLENS_DTYPE)
    return _CHUNK(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        scale=(q.shape[-1] ** -0.5) if scale is None else scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm,
        cu_seqlens=cu_seqlens,
        **_cu_seqlens_cpu_kwarg(cu_seqlens_cpu),
        **extra,
    )


def _cu_seqlens_cpu_kwarg(cu_seqlens_cpu) -> dict:
    """``{'cu_seqlens_cpu': t}`` when the installed fla takes it, else ``{}``.

    **The sync this removes.** fla's varlen path builds its chunk
    index/offset tensors with ``torch.repeat_interleave`` over a *host* copy of
    ``cu_seqlens`` (``fla/ops/utils/index.py::_segmented_arange``), i.e. one
    implicit D2H per distinct ``cu_seqlens`` **object**. fla memoises that on
    Python ``is`` identity in a 4-deep deque, so today this engine pays it once
    per chunk rather than once per GDN layer -- but only by luck: the identity
    survives solely because ``PrefillBatch.cu_seqlens`` is *already* int32 on
    the right device, so ``chunk_gdn``'s ``.to(device=..., dtype=int32)``
    returns ``self``. Anything that made that a real conversion (an int64
    ``cu_seqlens``, a slice, a ``clone()``) would silently turn 1 pipeline
    drain per chunk into **48**, with no error and no test to catch it -- the
    same failure mode, on the same path, as the conv prefill's per-layer
    ``cu_seqlens.to("cpu")`` that ``conv_prefill_varlen`` avoids.

    Passing the host copy explicitly removes the sync unconditionally and stops
    the engine depending on an upstream memo's eviction policy. vLLM reaches
    the same place from the other side: it computes ``chunk_indices``/
    ``chunk_offsets`` once per *step* on the host and threads them into every
    layer (``vllm/v1/attention/backends/gdn_attn.py::_build_chunk_metadata``).

    Guarded by a signature check because ``cu_seqlens_cpu`` is a named
    parameter (unlike ``chunk_size``, which is popped from ``**kwargs``) and an
    older fla would raise ``TypeError`` on it.
    """
    if cu_seqlens_cpu is None or not supports_cu_seqlens_cpu():
        return {}
    return {"cu_seqlens_cpu": cu_seqlens_cpu}


# --------------------------------------------------------------------------- #
# pool-aware entry points (gather -> kernel -> scatter)
# --------------------------------------------------------------------------- #
def decode_step(
    q, k, v, g, beta, state_pool, slot_ids, *, scale=None, use_qk_l2norm=True,
    out=None, A_log=None, dt_bias=None,
):
    # fla exposes use_gate_in_kernel/use_beta_sigmoid_in_kernel, but their exact
    # parameterisation is not pinned by the tests here, so the gate is applied
    # eagerly on this backend.  Only the Triton backend fuses it.
    g, beta = maybe_gate(g, beta, A_log, dt_bias)
    s0 = gather_states(state_pool, slot_ids, torch.float32)
    o, s1 = recurrent_gdn(
        q, k, v, g, beta, s0, True, use_qk_l2norm=use_qk_l2norm, scale=scale
    )
    scatter_states(state_pool, slot_ids, s1)
    if out is not None:
        out.copy_(o.reshape(out.shape).to(out.dtype))
        return out
    return o


def verify_and_commit(
    q,
    k,
    v,
    g,
    beta,
    state_pool,
    slot_ids,
    m,
    *,
    scale=None,
    use_qk_l2norm=True,
    method: str = "two_phase",
    A_log=None,
    dt_bias=None,
):
    """Two-phase verify, stock kernels, **no host sync**.

    Phase A runs the window with ``output_final_state=False`` for the logits.
    Phase B runs it again with ``g`` and ``beta`` zeroed for ``t >= m``: with
    ``beta = 0`` the delta update vanishes and with ``g = 0`` the decay is 1, so
    the recurrence is the identity past ``m`` and the final state is exactly
    ``S_m``, the masking form of the rollback identity
    ``S_m = S exp(G_m) + sum_{i<=m} exp(G_m - G_i) k_i (x) d_i``.
    ``m`` therefore never leaves the device and the whole thing stays capturable.
    """
    if method != "two_phase":
        raise ValueError(
            f"fla backend implements method='two_phase' only (got {method!r}); "
            "use backend='triton' for the fused kernel"
        )
    g, beta = maybe_gate(g, beta, A_log, dt_bias)
    q, k, v, g, beta = normalize_qkv(q, k, v, g, beta)
    n = v.shape[1]
    s0 = gather_states(state_pool, slot_ids, torch.float32)
    o, _ = recurrent_gdn(
        q, k, v, g, beta, s0, False, use_qk_l2norm=use_qk_l2norm, scale=scale
    )
    t_idx = torch.arange(n, device=s0.device)
    keep = (t_idx[None, :] < m.to(s0.device)[:, None].to(t_idx.dtype)).to(g.dtype)
    _, s1 = recurrent_gdn(
        q,
        k,
        v,
        g * keep[:, :, None],
        beta * keep[:, :, None],
        s0,
        True,
        use_qk_l2norm=use_qk_l2norm,
        scale=scale,
    )
    scatter_states(state_pool, slot_ids, s1)
    return o


__all__ = [
    "CU_SEQLENS_DTYPE",
    "is_available",
    "unavailable_reason",
    "version",
    "recurrent_gdn",
    "chunk_gdn",
    "decode_step",
    "verify_and_commit",
]
