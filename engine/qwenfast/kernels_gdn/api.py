"""Public API of the GDN kernels: the frozen interface the runtime calls.

Every entry point takes a ``backend=`` argument and falls back automatically:

===================  ==========================================================
op                   preference order
===================  ==========================================================
``gdn_decode_step``  ``triton`` -> ``fla`` -> ``torch``
``gdn_verify_*``     ``triton`` -> ``fla`` -> ``torch``
``gdn_prefill_*``    ``fla`` -> ``torch``   (no hand-written chunk kernel: fla's
                     is already compute-bound and cuLA is the planned drop-in)
``causal_conv_*``    ``triton`` -> ``torch``
===================  ==========================================================

Tensor conventions
------------------
=================  =========================================================
``q``, ``k``       ``[B, T, 16, 128]`` (canonical GVA form) or ``[B, T, 48, 128]``
                   (pre-expanded, accepted and collapsed), or ``[B, H, 128]``
                   for ``T == 1``.  **Not** L2-normalised — the kernels do it.
``v``              ``[B, T, 48, 128]``
``g``              ``[B, T, 48]`` fp32, already ``-exp(A_log)*softplus(a+dt)``
``beta``           ``[B, T, 48]`` fp32, already sigmoided
``state_pool``     ``[n_slots, 48, 128, 128]`` — one *layer's* view of the
                   ``[n_slots, 48, 48, 128, 128]`` pool; see
                   :func:`.state.layer_state`
``slot_ids``       ``[B]`` int32 on device.  Padded graph rows must point at a
                   scratch slot.  The ``triton`` backend additionally
                   treats a **negative** id as "skip" (state left untouched,
                   output row undefined) as a safety net; ``torch``/``fla`` go
                   through ``index_select`` and reject it.
``cu_seqlens``     ``[N+1]`` int32, requires ``B == 1`` (packed varlen)
``m``              ``[B]`` int32 **on device** — accepted draft length
=================  =========================================================

All state math is fp32 regardless of the pool dtype; ``--ssm-state-dtype``
only changes what is stored.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch

from . import fla_ops, fla_static, shapes, torch_ops, triton_ops
from .state import gather_states, scatter_states

BACKENDS = ("torch", "fla", "triton")
_DEFAULT_BACKEND = "auto"


# --------------------------------------------------------------------------- #
# backend selection
# --------------------------------------------------------------------------- #
def available_backends() -> dict:
    """``{name: True | reason-string}`` — what can actually run right now."""
    return {
        "torch": True,
        "fla": True if fla_ops.is_available() else (fla_ops.unavailable_reason() or "n/a"),
        # Not a backend, a *capability* of the fla backend: can its
        # chunk path be driven with caller-supplied index tensors (and hence
        # CUDA-graph captured)?  See `kernels_gdn.fla_static`.
        "fla_static": True
        if fla_static.is_available()
        else (fla_static.unavailable_reason() or "n/a"),
        "triton": True
        if triton_ops.is_available()
        else (triton_ops.unavailable_reason() or "n/a"),
    }


def set_default_backend(backend: str) -> None:
    """Override what ``backend='auto'`` resolves to (``'auto'`` restores)."""
    global _DEFAULT_BACKEND
    if backend != "auto" and backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; pick from {BACKENDS} or 'auto'")
    _DEFAULT_BACKEND = backend


def get_default_backend() -> str:
    return _DEFAULT_BACKEND


def resolve_backend(backend: str, prefer: Sequence[str]) -> str:
    """Resolve ``'auto'`` against ``prefer``; validate an explicit choice."""
    if backend == "auto":
        backend = _DEFAULT_BACKEND
    if backend != "auto":
        if backend not in BACKENDS:
            raise ValueError(f"unknown backend {backend!r}; pick from {BACKENDS}")
        if backend == "triton" and not triton_ops.is_available():
            raise RuntimeError(
                f"backend='triton' requested but unavailable: "
                f"{triton_ops.unavailable_reason()}"
            )
        if backend == "fla" and not fla_ops.is_available():
            raise RuntimeError(
                f"backend='fla' requested but unavailable: "
                f"{fla_ops.unavailable_reason()}"
            )
        if backend not in prefer:
            # e.g. backend='triton' for prefill -> fall through to the next
            # available implementation rather than failing the call
            for cand in prefer:
                if _backend_ok(cand):
                    return cand
            return "torch"
        return backend
    for cand in prefer:
        if _backend_ok(cand):
            return cand
    return "torch"


def _backend_ok(name: str) -> bool:
    if name == "torch":
        return True
    if name == "fla":
        return fla_ops.is_available()
    if name == "triton":
        return triton_ops.is_available()
    return False


def _prenorm(be: str, qk, use_qk_l2norm: bool, scale):
    """Split a pre-normalised call into per-backend arguments.

    Contract when ``qk`` is given: ``q`` is already L2-normalised **and**
    scaled, ``k`` is already L2-normalised, and ``qk = (q.k)`` per k-head (see
    :func:`.torch_ops.prenormalize_qk`).  The Triton kernels take it as the
    PRENORM path; ``torch``/``fla`` just skip their own norm and scale.
    """
    if qk is None:
        return {}, use_qk_l2norm, scale
    if be == "triton":
        return {"qk": qk}, use_qk_l2norm, scale
    return {}, False, 1.0


_DECODE_PREF = ("triton", "fla", "torch")
_PREFILL_PREF = ("fla", "torch")
_CONV_PREF = ("triton", "torch")


# --------------------------------------------------------------------------- #
# 1. decode
# --------------------------------------------------------------------------- #
def gdn_decode_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state_pool: torch.Tensor,
    slot_ids: torch.Tensor,
    *,
    backend: str = "auto",
    scale: Optional[float] = None,
    use_qk_l2norm: bool = True,
    out: Optional[torch.Tensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    qk: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """One GDN decode step for ``B`` sequences; ``state_pool`` updated in place.

    Returns ``out`` of shape ``[B, 1, 48, 128]`` in ``v``'s dtype (or writes
    into ``out`` if given, which is what the CUDA-graph path does).

    Ideal traffic is 1 state read + 1 state write = ``2 * 3 MiB * B`` per layer.
    The ``triton`` backend hits that; ``fla`` costs 3x because upstream 0.5.2
    cannot index a pool (see :mod:`.fla_ops`).
    """
    be = resolve_backend(backend, _DECODE_PREF)
    extra, use_qk_l2norm, scale = _prenorm(be, qk, use_qk_l2norm, scale)
    fn = {
        "triton": triton_ops.decode_step,
        "fla": fla_ops.decode_step,
        "torch": torch_ops.decode_step,
    }[be]
    return fn(
        q,
        k,
        v,
        g,
        beta,
        state_pool,
        slot_ids,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm,
        out=out,
        A_log=A_log,
        dt_bias=dt_bias,
        **extra,
    )


def gdn_decode_multi(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state_pool: torch.Tensor,
    slot_ids: torch.Tensor,
    *,
    backend: str = "auto",
    scale: Optional[float] = None,
    use_qk_l2norm: bool = True,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    qk: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``T`` sequential steps in one pass (1 read + 1 write total).

    Used for short windows: the MTP verify pass, and prefill continuations too
    short for the chunk kernel to pay off.
    """
    be = resolve_backend(backend, _DECODE_PREF)
    extra, use_qk_l2norm, scale = _prenorm(be, qk, use_qk_l2norm, scale)
    if be == "triton":
        return triton_ops.window(
            q, k, v, g, beta, state_pool, slot_ids, None,
            commit=True, scale=scale, use_qk_l2norm=use_qk_l2norm,
            A_log=A_log, dt_bias=dt_bias, **extra,
        )
    mod = fla_ops if be == "fla" else torch_ops
    g, beta = torch_ops.maybe_gate(g, beta, A_log, dt_bias)
    s0 = gather_states(state_pool, slot_ids, torch.float32)
    o, s1 = mod.recurrent_gdn(
        q, k, v, g, beta, s0, True, use_qk_l2norm=use_qk_l2norm, scale=scale
    )
    scatter_states(state_pool, slot_ids, s1)
    return o


# --------------------------------------------------------------------------- #
# 2. prefill
# --------------------------------------------------------------------------- #
def gdn_prefill_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    *,
    output_final_state: bool = True,
    state_pool: Optional[torch.Tensor] = None,
    slot_ids: Optional[torch.Tensor] = None,
    backend: str = "auto",
    scale: Optional[float] = None,
    use_qk_l2norm: bool = True,
    chunk_size: int = shapes.DEFAULT_CHUNK_SIZE,
    cu_seqlens_cpu: Optional[torch.Tensor] = None,
    chunk_indices: Optional[torch.Tensor] = None,
    chunk_offsets: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Chunked (prefill) gated delta rule.

    ``cu_seqlens=[N+1]`` selects the packed-varlen form and then ``B`` must be
    1 and ``initial_state`` must have ``N`` rows — matching fla's contract and
    the engine's rule that requests are packed varlen, not padded.

    When ``state_pool``/``slot_ids`` are given the initial state is gathered
    from the pool (unless ``initial_state`` is passed explicitly) and the final
    state is scattered back, so a whole prefill chunk threads through the pool
    with no caller-side bookkeeping.  Splitting a request across two calls is
    exact — the state threads through.
    """
    be = resolve_backend(backend, _PREFILL_PREF)
    mod = fla_ops if be == "fla" else torch_ops

    if initial_state is None and state_pool is not None and slot_ids is not None:
        initial_state = gather_states(state_pool, slot_ids, torch.float32)

    need_final = output_final_state or (state_pool is not None and slot_ids is not None)

    # Caller-supplied index tensors -> the *capturable*
    # fla path: fla does zero index preparation, zero host syncs and zero H2D,
    # so the whole call can sit inside a CUDA graph. Same kernels, same state
    # layout (`state_v_first=False`), same numerics -- the only difference is
    # who built `chunk_indices`/`chunk_offsets`. Silently ignored by the
    # `torch` backend, which has no index tensors at all.
    if chunk_indices is not None and be == "fla" and cu_seqlens is not None:
        if chunk_offsets is None:
            raise ValueError(
                "gdn_prefill_chunked: chunk_indices without chunk_offsets -- fla's "
                "`chunk_gated_delta_rule_fwd_h` needs both, and building one of "
                "them here would reintroduce the host sync the pair exists to remove"
            )
        out, final = fla_static.chunk_gdn_static(
            q, k, v, g, beta,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            initial_state=initial_state,
            output_final_state=need_final,
            use_qk_l2norm=use_qk_l2norm,
            scale=scale,
            chunk_size=chunk_size,
        )
        if state_pool is not None and slot_ids is not None and final is not None:
            scatter_states(state_pool, slot_ids, final)
        return out, (final if output_final_state else None)

    out, final = mod.chunk_gdn(
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens=cu_seqlens,
        initial_state=initial_state,
        output_final_state=need_final,
        use_qk_l2norm=use_qk_l2norm,
        scale=scale,
        chunk_size=chunk_size,
        # The host copy of `cu_seqlens`, so fla's varlen index prep does
        # not read the device tensor back to build it -- see
        # `fla_ops._cu_seqlens_cpu_kwarg` for the sync this removes and why the
        # engine was only accidentally paying it once per chunk instead of once
        # per GDN layer. `None` (every caller that does not have it to hand) is
        # exactly the previous behaviour.
        cu_seqlens_cpu=cu_seqlens_cpu,
    )
    if state_pool is not None and slot_ids is not None and final is not None:
        scatter_states(state_pool, slot_ids, final)
    return out, (final if output_final_state else None)


# --------------------------------------------------------------------------- #
# 3. causal conv
# --------------------------------------------------------------------------- #
def causal_conv_update(
    x: torch.Tensor,
    conv_state_pool: torch.Tensor,
    slot_ids: torch.Tensor,
    w: torch.Tensor,
    *,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    backend: str = "auto",
) -> torch.Tensor:
    """Decode-step depthwise causal conv, ring state updated in place.

    ``x``: ``[B, C]`` or ``[B, C, 1]`` · ``w``: ``[C, W]`` ·
    ``conv_state_pool``: ``[n_slots, C, W-1]``.  Returns ``x``'s shape.
    """
    be = resolve_backend(backend, _CONV_PREF)
    fn = triton_ops.conv_update if be == "triton" else torch_ops.conv_update
    return fn(x, conv_state_pool, slot_ids, w, bias, activation)


def causal_conv_prefill(
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    cu_seqlens: Optional[torch.Tensor] = None,
    conv_state_pool: Optional[torch.Tensor] = None,
    slot_ids: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    backend: str = "auto",
    tile_tokens: Optional[int] = None,
) -> torch.Tensor:
    """Prefill depthwise causal conv.  ``x``: ``[B, C, T]``, returns ``[B, C, T]``.

    ``cu_seqlens`` requires ``B == 1``.  Always torch/cuDNN today: ``F.conv1d``
    with ``groups=C`` is already a good depthwise kernel at prefill widths and
    is not on the critical path.
    There is **no Triton conv-prefill kernel** in ``triton_ops`` -- it only
    implements the decode-step ``conv_update`` -- so ``tile_tokens`` is how
    this path's fp32 working set is bounded instead
    (``RuntimeConfig.conv_prefill_tile_tokens``).
    """
    _ = resolve_backend(backend, ("torch",))
    return torch_ops.conv_prefill(
        x,
        w,
        cu_seqlens=cu_seqlens,
        conv_state_pool=conv_state_pool,
        slot_ids=slot_ids,
        initial_state=initial_state,
        bias=bias,
        activation=activation,
        tile_tokens=tile_tokens,
    )


def causal_conv_prefill_varlen(
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    seq_lens: Sequence[int],
    cu_seqlens: torch.Tensor,
    conv_state_pool: torch.Tensor,
    slot_ids: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    backend: str = "auto",
    max_seqlen: Optional[int] = None,
) -> torch.Tensor:
    """Token-major packed-varlen prefill conv.  ``x``: ``[T, C]`` -> ``[T, C]``.

    Supersedes :func:`causal_conv_prefill` on the serving
    path. Three things it does that the ``[B, C, T]`` entry point cannot:

    1. it takes the layout ``FusedGDN.prefill`` already holds, deleting two
       ``[C, T]`` transposes per GDN layer (336 MB each way at C=10240,
       T=8192; 32 GB of pure copy per chunk over 48 layers);
    2. ``seq_lens`` comes from the host (``PrefillBatch.q_lens``), so no
       ``cu_seqlens.to("cpu")`` D2H sync per layer -- 48 pipeline drains per
       chunk on the old path; and
    3. the ``triton`` backend does the whole thing in **two launches** with an
       fp32 accumulator over a bf16 input, instead of ~10 eager fp32 ops per
       tile per sequence.

    ``cu_seqlens`` ``[N+1]`` int32 on device is still required (the Triton
    kernel indexes with it); it must agree with ``seq_lens``. The ``torch``
    backend ignores it and uses ``seq_lens`` alone.

    ``max_seqlen`` overrides ``max(seq_lens)``, which is
    the kernel's **grid** in the token axis and therefore the one thing about
    this launch that a CUDA graph freezes. Passing the chunk's token cap
    instead of the step's longest segment makes the grid a function of the
    padded shape alone; the kernel already masks every tile with
    ``if t0 >= n: return``, so the extra tiles are exits. It must be an
    **upper** bound on ``max(seq_lens)`` -- a smaller value would silently
    drop the tail of a segment -- and is checked as one.
    """
    # Argument validation first, before any backend question: `max_seqlen` is
    # the kernel's *grid*, so a value below the longest segment silently drops
    # that segment's tail rather than failing -- and a caller that got it wrong
    # deserves the same error whether or not this machine has Triton.
    real_max = max((int(n) for n in seq_lens), default=0)
    if max_seqlen is not None and int(max_seqlen) < real_max:
        raise ValueError(
            f"causal_conv_prefill_varlen: max_seqlen={int(max_seqlen)} is below the "
            f"longest segment ({real_max}); the grid would not cover it"
        )
    be = resolve_backend(backend, _CONV_PREF)
    # `resolve_backend` answers "is triton importable", not "can triton run on
    # these tensors". On a GPU host triton imports fine, so a CPU tensor --
    # every unit test, and `from_m0_module`'s CPU parity path -- would be
    # handed to a CUDA kernel and fail at launch. The decode-side
    # `causal_conv_update` never hit this because nothing calls it on CPU; this
    # entry point is on the path the CPU test suite exercises, so the device
    # check belongs here rather than in the caller.
    if be == "triton" and x.device.type != "cuda":
        be = "torch"
    if be == "triton":
        grid_max = real_max if max_seqlen is None else int(max_seqlen)
        return triton_ops.conv_prefill_varlen(
            x,
            w,
            cu_seqlens,
            conv_state_pool,
            slot_ids,
            max_seqlen=grid_max,
            bias=bias,
            activation=activation,
        )
    return torch_ops.conv_prefill_varlen(
        x,
        w,
        seq_lens,
        conv_state_pool,
        slot_ids,
        bias=bias,
        activation=activation,
    )


def causal_conv_verify_and_commit(
    x: torch.Tensor,
    conv_state_pool: torch.Tensor,
    slot_ids: torch.Tensor,
    w: torch.Tensor,
    m: torch.Tensor,
    *,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    backend: str = "auto",
) -> torch.Tensor:
    """Conv over an ``n``-token draft window, committing only the first ``m``.

    Conv rollback is a pointer move.  ``x``: ``[B, C, n]`` ->
    ``[B, C, n]``; the ring becomes ``concat(state, x)[..., m : m+W-1]``.
    ``m`` stays on the device (gathered, not sliced).
    """
    _ = resolve_backend(backend, ("torch",))
    return torch_ops.conv_verify_and_commit(
        x, conv_state_pool, slot_ids, w, m, bias, activation
    )


# --------------------------------------------------------------------------- #
# 4. speculative decoding
# --------------------------------------------------------------------------- #
def gdn_verify(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state_pool: torch.Tensor,
    slot_ids: torch.Tensor,
    *,
    backend: str = "auto",
    scale: Optional[float] = None,
    use_qk_l2norm: bool = True,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    qk: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Verify **phase A**: outputs for all ``k+1`` draft tokens, *no* state write.

    1 state read, 0 writes.  Pair with :func:`gdn_commit` once ``m`` is known,
    or use :func:`gdn_verify_and_commit` which does both.
    """
    be = resolve_backend(backend, _DECODE_PREF)
    extra, use_qk_l2norm, scale = _prenorm(be, qk, use_qk_l2norm, scale)
    if be == "triton":
        return triton_ops.window(
            q, k, v, g, beta, state_pool, slot_ids, None,
            commit=False, scale=scale, use_qk_l2norm=use_qk_l2norm,
            A_log=A_log, dt_bias=dt_bias, **extra,
        )
    mod = fla_ops if be == "fla" else torch_ops
    g, beta = torch_ops.maybe_gate(g, beta, A_log, dt_bias)
    s0 = gather_states(state_pool, slot_ids, torch.float32)
    o, _ = mod.recurrent_gdn(
        q, k, v, g, beta, s0, False, use_qk_l2norm=use_qk_l2norm, scale=scale
    )
    return o


def gdn_commit(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state_pool: torch.Tensor,
    slot_ids: torch.Tensor,
    m: torch.Tensor,
    *,
    backend: str = "auto",
    scale: Optional[float] = None,
    use_qk_l2norm: bool = True,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    qk: Optional[torch.Tensor] = None,
) -> None:
    """Verify **phase B**: advance the state to ``S_m`` from the cached inputs.

    Implemented by zeroing ``g`` and ``beta`` for ``t >= m`` — with ``beta = 0``
    the delta update vanishes and with ``g = 0`` the decay is 1, so the
    recurrence is the identity past ``m``.  ``m`` never leaves the device, so
    the commit is CUDA-graph-safe, unlike a ``cu_seqlens``-packed
    replay, which would need the accept lengths on the host.

    On the Triton backend the masking happens **inside** the kernel
    (``MASK_PAST_M``), so this is one launch and zero eager ops.
    """
    be = resolve_backend(backend, _DECODE_PREF)
    extra, use_qk_l2norm, scale = _prenorm(be, qk, use_qk_l2norm, scale)
    if be == "triton":
        triton_ops.commit(
            q, k, v, g, beta, state_pool, slot_ids, m,
            scale=scale, use_qk_l2norm=use_qk_l2norm,
            A_log=A_log, dt_bias=dt_bias, **extra,
        )
        return
    g, beta = torch_ops.maybe_gate(g, beta, A_log, dt_bias)
    n = v.shape[1] if v.dim() == 4 else 1
    t_idx = torch.arange(n, device=state_pool.device)
    keep = (t_idx[None, :] < m.to(state_pool.device)[:, None].to(t_idx.dtype)).to(
        torch.float32
    )
    if g.dim() == 2:
        g = g.unsqueeze(1)
        beta = beta.unsqueeze(1)
    gdn_decode_multi(
        q,
        k,
        v,
        g.float() * keep[:, :, None],
        beta.float() * keep[:, :, None],
        state_pool,
        slot_ids,
        backend=be,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm,
    )


def gdn_verify_and_commit(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state_pool: torch.Tensor,
    slot_ids: torch.Tensor,
    m: torch.Tensor,
    *,
    method: str = "auto",
    backend: str = "auto",
    scale: Optional[float] = None,
    use_qk_l2norm: bool = True,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    qk: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Speculative verify + rollback-free commit.

    Computes the GDN outputs for **all** ``n = k+1`` draft tokens starting from
    the committed state, then writes back only ``S_m`` — the state after the
    accepted prefix.  ``m`` is a ``[B]`` **device** int32 tensor, so nothing
    syncs to the host and the whole speculative step stays graph-capturable.

    ``method='two_phase'``
        Two stock-kernel passes: verify (1 read, 0 writes) then masked replay
        (1 read, 1 write).  **1.5x** plain-decode SSM traffic.  Works on every
        backend.
    ``method='fused'``
        One pass; the kernel keeps ``S_m`` in a second register tile and stores
        it once.  **1.0x** plain-decode traffic — 1 read, 1 write, identical to
        a non-speculative step.  Triton (and the torch reference) only.
    ``method='auto'``
        ``'fused'`` where a Triton kernel is available, else ``'two_phase'``.

    Note the ordering constraint this puts on the caller: ``m`` must already be
    known when the fused kernel runs, so the engine's speculative step either
    (a) runs :func:`gdn_verify` for all 48 layers, samples ``m``, then commits
    — the two-phase shape — or (b) defers the commit by one step and replays
    the previous window's accepted prefix at the head of the next one, which is
    the ReplaySSM-style flow the fused kernel is built for.  This function
    implements the kernel-level contract for both.
    """
    be = resolve_backend(backend, _DECODE_PREF)
    extra, use_qk_l2norm, scale = _prenorm(be, qk, use_qk_l2norm, scale)
    if method == "auto":
        method = "fused" if be in ("triton", "torch") else "two_phase"
    fn = {
        "triton": triton_ops.verify_and_commit,
        "fla": fla_ops.verify_and_commit,
        "torch": torch_ops.verify_and_commit,
    }[be]
    return fn(
        q,
        k,
        v,
        g,
        beta,
        state_pool,
        slot_ids,
        m,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm,
        method=method,
        A_log=A_log,
        dt_bias=dt_bias,
        **extra,
    )


__all__ = [
    "BACKENDS",
    "available_backends",
    "set_default_backend",
    "get_default_backend",
    "resolve_backend",
    "gdn_decode_step",
    "gdn_decode_multi",
    "gdn_prefill_chunked",
    "causal_conv_update",
    "causal_conv_prefill",
    "causal_conv_prefill_varlen",
    "causal_conv_verify_and_commit",
    "gdn_verify",
    "gdn_commit",
    "gdn_verify_and_commit",
]
