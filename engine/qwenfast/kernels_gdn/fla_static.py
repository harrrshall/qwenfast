"""The fla chunk kernel with **static** index tensors.

Why this module exists
----------------------
Without it, the graphed mixed prefill+decode step is captured in **49 graph
segments with 48 eager holes**, one hole per GDN layer, because one call in the
step could not be captured: ``fla.ops.gated_delta_rule.chunk_gated_delta_rule``.
Its varlen index tensors come from ``fla.ops.utils.index.prepare_chunk_indices``,
whose body (fla-core 0.5.2) is::

    @tensor_cache
    def prepare_chunk_indices(cu_seqlens, chunk_size, cu_seqlens_cpu=None):
        src = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
        chunk_counts = (prepare_lens(src) + (chunk_size - 1)).div(chunk_size, ...)
        seg_id, intra_chunk_idx = _segmented_arange(chunk_counts)
        return torch.stack([seg_id, intra_chunk_idx], 1).to(cu_seqlens)

Two independent reasons that cannot be inside a capture: it ends in a pageable
H2D ``.to(cu_seqlens)``, and ``@tensor_cache`` memoises on argument
``is``-identity, so a persistent ``cu_seqlens`` buffer whose *contents* change
per step would be handed the **previous** step's segmentation. Not slow:
silently wrong.

The fix, and it is smaller than it looks
----------------------------------------
fla 0.5.2 already threads a precomputed ``chunk_indices`` all the way down.
``chunk_gated_delta_rule_fwd`` — the function the autograd wrapper calls, one
level below the public entry point — takes ``chunk_indices=`` and passes it to
every one of its four sub-ops (``chunk_local_cumsum``,
``chunk_gated_delta_rule_fwd_intra``, ``chunk_gated_delta_rule_fwd_h``,
``chunk_fwd_o``), each of which uses it as ``NT = len(chunk_indices)`` for its
grid and reads the rest **on the device**. It also takes ``state_v_first`` and
defaults it to ``False``, i.e. the ``[N, HV, K, V]`` state layout that is
already this engine's pool layout, so unlike vLLM's
vendored fork (hardcoded V-first, ``vllm/third_party/flash_linear_attention``),
nothing has to be transposed.

So the whole hole closes by **passing the index tensors in** rather than
letting fla build them:

* ``chunk_indices``  ``[NT, 2]`` int32/int64 on device — ``(segment id, chunk
  index within that segment)`` for every chunk in the packed batch;
* ``chunk_offsets``  ``[N + 1]`` — the exclusive prefix sum of each segment's
  chunk count, used by ``chunk_gated_delta_rule_fwd_h`` only as the base row of
  its ``h`` scratch for segment ``n``;
* ``cu_seqlens``     ``[N + 1]`` int32 on device — already static in the graphed mixed step.

Both are computed **on the host** from ``q_lens`` (which the scheduler already
has: no D2H, ever) and copied into persistent device buffers once per step, the
same design vLLM's ``gdn_attn.py::_build_chunk_metadata`` uses.

The one remaining variable: ``NT``
----------------------------------
``NT = sum_n ceil(len_n / BT)`` is **not** fixed by ``(chunk_tokens,
n_segments)`` — the segment *lengths* still vary step to step, and every
partial trailing chunk costs a row. It is bounded, though::

    NT <= (chunk_tokens - n_segments) // BT + n_segments        (:func:`max_chunk_rows`)

(each of the ``n_segments`` segments wastes at most ``BT - 1`` tokens, and the
bound is tight: 8 segments over a 1,024-token chunk at ``BT=64`` gives 23.)

So the device buffer is allocated at ``max_chunk_rows`` and every step fills the
first ``NT`` rows with the real segmentation and the remaining
``max_chunk_rows - NT`` rows with a **copy of the last real row**.

Why a copy and not a past-the-end index. Both work — every store in fla's four
sub-kernels is masked (``mask=m_t`` where ``m_t = o_t < T``, or a boundary-checked
``make_block_ptr``), so a row pointing past its segment's end writes nothing —
but the duplicate is safe for a *stronger* reason that does not depend on
reading fla's masks correctly: the four kernels are pure per-chunk functions
with **no atomics** (zero ``atomic``
occurrences in ``chunk_fwd.py`` / ``wy_fast.py`` / ``chunk_o.py`` /
``chunk_delta_h.py`` / ``cumsum.py``), so re-running a chunk writes the same
bytes to the same addresses. A duplicate row is idempotent whether or not the
store is masked; a past-the-end row is correct only if it is.

The padding is close to free on the shape that matters. At the serving geometry
(chunk 1,024, 8 plan rows, 1-2 real segments and 6-7 one-token pad segments)
``NT`` is already 22-23 against a bound of 23: the pad *segments* the graphed
mixed step introduces are themselves the fragmentation, and the duplicate rows are what is
left of it.

``chunk_offsets`` and fla's memo
--------------------------------
``chunk_gated_delta_rule_fwd_h`` does **not** take ``chunk_offsets``; it calls
``prepare_chunk_offsets(cu_seqlens, BT)`` itself. That function is pure device
arithmetic (``diff`` / ``cdiv`` / ``pad`` / ``cumsum``) and would therefore
capture and replay correctly — except that it is ``@tensor_cache``d on identity
too, so with a persistent ``cu_seqlens`` buffer it would return whatever tensor
the warmup pass produced, frozen into the graph as a constant. :func:`static_index_scope`
replaces the symbol in the fla modules that hold it, for the duration of the
call, with one that returns our buffer — and replaces ``prepare_chunk_indices``
with one that **raises**, so that a future fla release which stops honouring
``chunk_indices=`` fails loudly instead of silently reintroducing the H2D.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import List, Optional, Sequence, Tuple

import torch

from . import shapes
from .torch_ops import collapse_gva, normalize_qkv

#: The dtype fla wants for ``cu_seqlens`` / ``chunk_indices`` / ``chunk_offsets``.
#: fla's kernels ``tl.load(...).to(tl.int32)`` all three, and
#: ``chunk_gated_delta_rule_fwd_h`` indexes ``h`` with ``chunk_offsets``; int32
#: is what ``kernels_gdn.fla_ops.CU_SEQLENS_DTYPE`` already uses for
#: ``cu_seqlens`` and keeps the three consistent.
INDEX_DTYPE = torch.int32


# --------------------------------------------------------------------------- #
# 1. the host-side index arithmetic (pure, CPU-testable, no torch needed)
# --------------------------------------------------------------------------- #
def max_chunk_rows(chunk_tokens: int, n_segments: int, chunk_size: int) -> int:
    """Upper bound on ``sum_n ceil(len_n / chunk_size)``.

    Over all ways of splitting ``chunk_tokens`` tokens into exactly
    ``n_segments`` non-empty segments. Each segment rounds up by at most
    ``chunk_size - 1`` tokens, so::

        NT <= (chunk_tokens + n_segments * (chunk_size - 1)) // chunk_size
            == (chunk_tokens - n_segments) // chunk_size + n_segments

    and the bound is attained (``n_segments - 1`` segments of one token and one
    of the rest). Both forms agree for every ``chunk_tokens >= n_segments``;
    the second is used because it is obviously an integer.
    """
    if chunk_tokens < n_segments or n_segments < 1 or chunk_size < 1:
        raise ValueError(
            f"max_chunk_rows: chunk_tokens={chunk_tokens} must be >= n_segments="
            f"{n_segments} >= 1 and chunk_size={chunk_size} >= 1"
        )
    return (chunk_tokens - n_segments) // chunk_size + n_segments


def build_chunk_meta(
    q_lens: Sequence[int], chunk_size: int, n_rows: Optional[int] = None
) -> Tuple[List[List[int]], List[int], int]:
    """``(chunk_indices, chunk_offsets, n_real_rows)`` for one packed chunk.

    ``chunk_indices[i] == [segment_id, chunk_index_within_segment]``, exactly
    what ``fla.ops.utils.index.prepare_chunk_indices`` builds -- verified
    against its source line for line, and pinned by
    ``tests/test_fla_static.py::TestAgainstFla``.

    ``chunk_offsets[n]`` is the number of chunks in segments ``[0, n)``;
    ``chunk_offsets[-1] == n_real_rows``.

    With ``n_rows`` given the index list is padded to exactly that length by
    repeating the **last real row** (see this module's docstring for why a
    duplicate rather than a past-the-end index). ``n_rows`` must be at least
    the real row count, which :func:`max_chunk_rows` guarantees for any
    segmentation of a ``chunk_tokens``-token chunk into ``n_segments`` parts.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    indices: List[List[int]] = []
    offsets: List[int] = [0]
    for n, length in enumerate(q_lens):
        length = int(length)
        if length < 0:
            raise ValueError(f"segment {n} has negative length {length}")
        n_chunks = -(-length // chunk_size)  # ceil
        indices.extend([n, i] for i in range(n_chunks))
        offsets.append(offsets[-1] + n_chunks)
    n_real = len(indices)
    if n_rows is None:
        return indices, offsets, n_real
    if n_rows < n_real:
        raise ValueError(
            f"build_chunk_meta: {n_real} chunk rows do not fit a static buffer of "
            f"{n_rows} (segment lengths {list(q_lens)}, chunk_size {chunk_size}); "
            "the caller's `max_chunk_rows` bound is wrong for this shape"
        )
    if n_real == 0:
        # Nothing to duplicate. Only reachable for an all-empty chunk, which no
        # caller produces (every plan row of the graphed mixed step is non-empty), but
        # a zero-filled buffer is a defined answer rather than an IndexError.
        indices = [[0, 0] for _ in range(n_rows)]
        return indices, offsets, 0
    last = indices[-1]
    indices = indices + [list(last) for _ in range(n_rows - n_real)]
    return indices, offsets, n_real


# --------------------------------------------------------------------------- #
# 2. availability
# --------------------------------------------------------------------------- #
_FWD = None
_L2NORM = None
_IMPORT_ERROR: Optional[str] = None
_PATCH_TARGETS: Tuple[object, ...] = ()


def _load() -> None:
    global _FWD, _L2NORM, _IMPORT_ERROR, _PATCH_TARGETS
    if _FWD is not None or _IMPORT_ERROR is not None:
        return
    try:
        import inspect

        from fla.modules.l2norm import l2norm_fwd  # type: ignore
        from fla.ops.gated_delta_rule.chunk import (  # type: ignore
            chunk_gated_delta_rule_fwd,
        )

        params = inspect.signature(chunk_gated_delta_rule_fwd).parameters
        missing = [p for p in ("chunk_indices", "state_v_first", "chunk_size")
                   if p not in params]
        if missing:
            raise RuntimeError(
                "fla's chunk_gated_delta_rule_fwd does not take "
                + ", ".join(missing)
                + " -- this build cannot be driven with static index tensors"
            )
        # Every module that holds a *reference* to the two index builders. They
        # are imported by value (`from fla.ops.utils import prepare_chunk_offsets`
        # in `fla/ops/common/chunk_delta_h.py`, and so on), so patching
        # `fla.ops.utils.index` alone would not be seen by the callers.
        # Imported explicitly rather than read out of `sys.modules`: which of
        # these fla has already pulled in depends on import order, and a module
        # that is missed here is a *silent* stale-index bug, which is the exact
        # failure this module exists to make impossible.
        import importlib

        targets = []
        for name in (
            "fla.ops.utils.index",
            "fla.ops.utils",
            "fla.ops.utils.cumsum",
            "fla.ops.common.chunk_delta_h",
            "fla.ops.common.chunk_o",
            "fla.ops.gated_delta_rule.chunk",
            "fla.ops.gated_delta_rule.chunk_fwd",
            "fla.ops.gated_delta_rule.wy_fast",
        ):
            try:
                mod = importlib.import_module(name)
            except Exception:
                continue
            if hasattr(mod, "prepare_chunk_offsets") or hasattr(
                mod, "prepare_chunk_indices"
            ):
                targets.append(mod)
        if not targets:
            raise RuntimeError(
                "no fla module exposes prepare_chunk_indices/prepare_chunk_offsets; "
                "the static-index patch would be a no-op and fla would build a "
                "stale segmentation inside the capture"
            )
        _PATCH_TARGETS = tuple(targets)
        _FWD = chunk_gated_delta_rule_fwd
        _L2NORM = l2norm_fwd
    except Exception as exc:  # pragma: no cover - depends on the host environment
        _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


def is_available() -> bool:
    _load()
    return _FWD is not None


def unavailable_reason() -> Optional[str]:
    _load()
    return _IMPORT_ERROR


# --------------------------------------------------------------------------- #
# 3. the fla index-builder scope
# --------------------------------------------------------------------------- #
class StaticIndexViolation(RuntimeError):
    """fla asked to *build* an index tensor while static ones were supplied.

    Raised instead of silently letting the (uncapturable, host-syncing,
    identity-memoised) builder run. If this fires, an fla upgrade stopped
    honouring one of the ``chunk_indices=`` / ``chunk_offsets`` paths this
    module depends on, and the graphed mixed step must go back to its
    per-layer eager holes rather than quietly produce a stale segmentation.
    """


_LOCAL = threading.local()


def _active_offsets() -> Optional[torch.Tensor]:
    return getattr(_LOCAL, "chunk_offsets", None)


def _patched_prepare_chunk_offsets(cu_seqlens, chunk_size):
    off = _active_offsets()
    if off is not None:
        return off
    raise StaticIndexViolation(  # pragma: no cover - the scope always sets it
        "prepare_chunk_offsets called with no static buffer bound"
    )


def _patched_prepare_chunk_indices(cu_seqlens, chunk_size, cu_seqlens_cpu=None):
    raise StaticIndexViolation(
        "fla tried to build `chunk_indices` itself inside a static-index call. "
        "That path does host index prep and a pageable H2D copy and is memoised "
        "on tensor identity -- it cannot be captured and would return a stale "
        "segmentation if it were. See kernels_gdn/fla_static.py."
    )


@contextmanager
def static_index_scope(chunk_offsets: torch.Tensor):
    """Bind ``chunk_offsets`` and forbid fla from building indices itself.

    Re-entrant per thread and restores every patched symbol on the way out,
    including on an exception -- a half-patched fla would make every later
    *eager* prefill wrong, which is a much worse failure than this call not
    running.
    """
    _load()
    prev = getattr(_LOCAL, "chunk_offsets", None)
    _LOCAL.chunk_offsets = chunk_offsets
    saved: List[Tuple[object, str, object]] = []
    try:
        for mod in _PATCH_TARGETS:
            for attr, repl in (
                ("prepare_chunk_offsets", _patched_prepare_chunk_offsets),
                ("prepare_chunk_indices", _patched_prepare_chunk_indices),
            ):
                if hasattr(mod, attr):
                    saved.append((mod, attr, getattr(mod, attr)))
                    setattr(mod, attr, repl)
        yield
    finally:
        for mod, attr, orig in saved:
            setattr(mod, attr, orig)
        _LOCAL.chunk_offsets = prev


# --------------------------------------------------------------------------- #
# 4. the call
# --------------------------------------------------------------------------- #
def _prep(q, k, v, g, beta):
    """Normalise dims and collapse q/k back to the 16-head GVA form.

    Byte-identical to :func:`kernels_gdn.fla_ops._prep`; duplicated rather than
    imported so this module has no import-time dependency on ``fla_ops``
    (which loads the public fla entry points and would make an fla without
    them fail differently here than there).
    """
    q, k, v, g, beta = normalize_qkv(q, k, v, g, beta)
    hv = v.shape[2]
    h = q.shape[2]
    if h == hv and hv % shapes.NUM_K_HEADS == 0 and hv != shapes.NUM_K_HEADS:
        q = collapse_gva(q, shapes.NUM_K_HEADS).contiguous()
        k = collapse_gva(k, shapes.NUM_K_HEADS).contiguous()
    return q, k, v, g, beta


def chunk_gdn_static(
    q,
    k,
    v,
    g,
    beta,
    *,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    chunk_offsets: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
    use_qk_l2norm: bool = True,
    scale: Optional[float] = None,
    chunk_size: int = shapes.DEFAULT_CHUNK_SIZE,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """``fla_ops.chunk_gdn`` with the index tensors supplied, not built.

    Same inputs, same outputs, same kernels -- the only difference is that
    ``chunk_indices``/``chunk_offsets`` come from the caller, so the call does
    **no host index preparation, no host sync and no H2D copy** and is
    therefore CUDA-graph capturable. ``state_v_first`` is left at fla's default
    ``False``, i.e. the ``[N, HV, K, V]`` pool layout, so the states go in and
    out untransposed.

    Bypasses ``ChunkGatedDeltaRuleFunction.apply``: this is an inference-only
    engine, the autograd wrapper's only extra work on the forward path is the
    L2 norm (done here) and ``save_for_backward``, and going through
    ``Function.apply`` inside a capture records nothing useful.
    """
    _load()
    if _FWD is None:
        raise RuntimeError(f"fla static-index path unavailable: {_IMPORT_ERROR}")
    q, k, v, g, beta = _prep(q, k, v, g, beta)
    # The two invariants fla's public wrapper checks and this path bypasses.
    # Both are pure shape reads -- no device sync -- and both are silent
    # corruption if violated: a `B > 1` varlen call reads the wrong rows, and
    # a short `initial_state` reads past the end of the gathered states.
    if q.shape[0] != 1:
        raise ValueError(
            f"chunk_gdn_static: packed varlen needs B == 1, got {q.shape[0]}"
        )
    n_seq = int(cu_seqlens.shape[0]) - 1
    if initial_state is not None and int(initial_state.shape[0]) != n_seq:
        raise ValueError(
            f"chunk_gdn_static: initial_state has {int(initial_state.shape[0])} rows "
            f"but cu_seqlens describes {n_seq} sequences"
        )
    if int(chunk_offsets.shape[0]) != n_seq + 1:
        raise ValueError(
            f"chunk_gdn_static: chunk_offsets has {int(chunk_offsets.shape[0])} entries, "
            f"expected {n_seq + 1} (one per sequence, plus the total)"
        )
    if chunk_indices.dim() != 2 or int(chunk_indices.shape[1]) != 2:
        raise ValueError(
            f"chunk_gdn_static: chunk_indices must be [NT, 2], got "
            f"{tuple(chunk_indices.shape)}"
        )
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    g = g.contiguous()
    beta = beta.contiguous()
    if use_qk_l2norm:
        # `l2norm_fwd` returns `(out, rstd)`; the rstd is only for the backward.
        q = _L2NORM(q)[0]
        k = _L2NORM(k)[0]
    if scale is None:
        scale = q.shape[-1] ** -0.5
    with static_index_scope(chunk_offsets):
        _g, o, _A, final_state, _h0, _gin = _FWD(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            state_v_first=False,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=int(chunk_size),
        )
    return o.to(q.dtype), final_state


__all__ = [
    "INDEX_DTYPE",
    "StaticIndexViolation",
    "max_chunk_rows",
    "build_chunk_meta",
    "is_available",
    "unavailable_reason",
    "static_index_scope",
    "chunk_gdn_static",
]
