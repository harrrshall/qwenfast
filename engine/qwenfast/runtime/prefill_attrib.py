#!/usr/bin/env python
"""Per-component attribution for **one prefill chunk**, CUDA-event timed.

A served step can be closed *arithmetically* (e.g. a 195 ms step as 45.8 ms
of decode plus 151 ms of amortised prefill, with the chunk itself at
~1,130 ms and 7,240 tok/s).  That does not say **where inside the chunk**
those 1,130 ms are.  ``profile_serving``'s ablation table answers that by deletion
(stub a component out, re-time the chunk), which is honest but non-additive:
removing work also removes the launch-latency tail it was hiding, so four
ablation deltas do not sum to a chunk and cannot be read as a budget.

This module answers it by **addition** instead.  Every layer-type call site in
the prefill path is bracketed by a pair of ``torch.cuda.Event``s recorded on
the compute stream; the whole chunk runs; then **one** ``synchronize()`` at
the end reads every pair.  That gives a per-component ms table that does sum
to the chunk (up to the un-bracketed glue, which is reported as the residual)
without a single mid-chunk sync.  That matters more here than anywhere else
in the engine, because prefill is the one path that is eager and therefore
already launch-latency-bound (a per-layer D2H sync inflates the measured conv
cost substantially).

Three properties, each of which is a requirement and not a nicety:

* **Sync-free.**  ``Event.record()`` is an enqueue, not a wait.  Nothing in
  ``resolve()``'s path runs before the chunk is done, so the CPU keeps running
  ahead exactly as it does un-instrumented.  A ``perf_counter`` bracket would
  measure launch time and nothing else.
* **Graph-free.**  Prefill is never captured as a CUDA graph, so a monkeypatch
  is legal here in a way it is not on the decode path.  ``instrument()``
  refuses to install itself while a capture is in progress.
* **Restorable.**  Every patch is recorded with its original attribute and put
  back by the ``restore()`` callable, so a profiler can interleave an
  instrumented chunk with un-instrumented A/B timings in one process.

The instrumentation is a monkeypatch rather than an edit to
``fused_model.py``'s hot path on purpose: the serving path must not pay a
branch per GEMM for a profiler that runs for a handful of chunks per run.

Overhead: two event records per bracketed call, ~1-2 us each on the launch
side.  A 64-layer chunk brackets ~700 calls, i.e. ~1.5-3 ms of *launch* time
on a ~1,100 ms chunk (0.2%), and ``profile_chunk_attribution`` reports the
instrumented and un-instrumented chunk times side by side so that overhead is
visible rather than assumed.

CPU-testable: with no CUDA device the tape falls back to ``perf_counter``
brackets, so the patch/restore bookkeeping, the label scheme and the table
arithmetic are all exercised by ``tests/test_serving_path.py`` on a laptop.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

# --------------------------------------------------------------------------- #
# 1. the tape
# --------------------------------------------------------------------------- #
#: Component a label belongs to, longest prefix first. The table groups on
#: this so a 40-row per-shape GEMM breakdown still rolls up to one line.
_GROUP_PREFIXES: Tuple[Tuple[str, str], ...] = (
    ("gemm.", "gemm"),
    ("gdn.conv", "gdn_conv"),
    ("gdn.chunk", "gdn_chunk"),
    ("gdn.state_io", "gdn_state_io"),
    ("gdn.gate", "gdn_gate"),
    ("gdn.norm", "norm"),
    ("attn.prefill", "attn_prefill"),
    ("attn.kv_append", "kv_append"),
    ("attn.qk_norm_rope", "attn_rope"),
    ("attn.gate", "attn_gate"),
    ("norm.", "norm"),
    ("mlp.act", "mlp_act"),
)


def group_of(label: str) -> str:
    """The component a region label rolls up to (``"other"`` if unknown)."""
    for prefix, group in _GROUP_PREFIXES:
        if label.startswith(prefix):
            return group
    return "other"


class EventTape:
    """A list of ``(label, start, stop)`` timing brackets, resolved once.

    ``enter(label)`` records a start event and returns a handle; ``exit(handle,
    flops=)`` records the stop.  Nothing is read back until :meth:`resolve`,
    which synchronizes **once** and then calls ``elapsed_time`` on every pair.

    Events are pooled and reused across :meth:`reset` calls, so a five-repeat
    profile allocates its ~1,400 events once rather than once per repeat
    (``cudaEventCreate`` is not free, and a fresh event per call would put
    allocator work back on the hot path this is trying to measure).
    """

    __slots__ = ("device", "cuda", "active", "_spans", "_pool", "_n_used", "_flops", "_reps")

    def __init__(self, device: Optional[torch.device] = None):
        self.device = torch.device(device) if device is not None else None
        self.cuda = bool(
            self.device is not None
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        )
        #: Set True only while a chunk is being profiled; every patched call
        #: site checks it first, so the patches are a single attribute read
        #: when the profiler is not looking.
        self.active = False
        self._spans: List[Tuple[str, Any, Any]] = []
        self._pool: List[Any] = []
        self._n_used = 0
        self._flops: "OrderedDict[str, float]" = OrderedDict()
        self._reps = 0

    # -- lifecycle -------------------------------------------------------- #
    def reset(self) -> None:
        """Drop the recorded spans, keep the event pool."""
        self._spans = []
        self._n_used = 0
        self._flops = OrderedDict()
        self._reps = 0

    def new_repeat(self) -> None:
        """Mark a repeat boundary -- :meth:`resolve` divides by this count."""
        self._reps += 1

    # -- recording -------------------------------------------------------- #
    def _event(self):
        if self._n_used < len(self._pool):
            ev = self._pool[self._n_used]
        else:
            ev = torch.cuda.Event(enable_timing=True)
            self._pool.append(ev)
        self._n_used += 1
        return ev

    def enter(self, label: str):
        """Open a bracket. Returns the handle to hand to :meth:`exit`."""
        if self.cuda:
            ev = self._event()
            ev.record()
            return (label, ev)
        return (label, time.perf_counter())

    def exit(self, handle, *, flops: float = 0.0) -> None:
        label, start = handle
        if self.cuda:
            stop = self._event()
            stop.record()
            self._spans.append((label, start, stop))
        else:
            self._spans.append((label, start, time.perf_counter()))
        if flops:
            self._flops[label] = self._flops.get(label, 0.0) + float(flops)

    # -- readback --------------------------------------------------------- #
    def resolve(self) -> "OrderedDict[str, Dict[str, float]]":
        """One sync, then ``{label: {ms, calls, flops, tflops}}``.

        ``ms`` is **per repeat** when :meth:`new_repeat` was called, so the
        table is directly comparable with a single chunk's wall time.
        """
        if self.cuda:
            torch.cuda.synchronize(self.device)
        reps = max(self._reps, 1)
        out: "OrderedDict[str, Dict[str, float]]" = OrderedDict()
        for label, start, stop in self._spans:
            ms = start.elapsed_time(stop) if self.cuda else (stop - start) * 1e3
            row = out.get(label)
            if row is None:
                row = out[label] = {"ms": 0.0, "calls": 0.0, "flops": 0.0}
            row["ms"] += ms
            row["calls"] += 1
        for label, row in out.items():
            row["ms"] = row["ms"] / reps
            row["calls"] = row["calls"] / reps
            fl = self._flops.get(label, 0.0) / reps
            row["flops"] = fl
            row["tflops"] = (fl / (row["ms"] / 1e3) / 1e12) if row["ms"] > 0 and fl else 0.0
        return out


# --------------------------------------------------------------------------- #
# 2. the table
# --------------------------------------------------------------------------- #
def attribution_table(
    labels: "OrderedDict[str, Dict[str, float]]", *, chunk_ms: float, tokens: int
) -> Dict[str, Any]:
    """Roll a resolved tape up into the additive attribution table.

    ``chunk_ms`` is the **un-instrumented** chunk time, so ``residual_ms``
    (chunk minus the sum of the brackets) is the honest "norms, glue,
    elementwise and launch gaps nothing bracketed" line rather than a
    balancing figure invented to make the column add up.
    """
    groups: "OrderedDict[str, Dict[str, float]]" = OrderedDict()
    for label, row in labels.items():
        g = group_of(label)
        acc = groups.get(g)
        if acc is None:
            acc = groups[g] = {"ms": 0.0, "calls": 0.0, "flops": 0.0}
        acc["ms"] += row["ms"]
        acc["calls"] += row["calls"]
        acc["flops"] += row.get("flops", 0.0)
    attributed = sum(g["ms"] for g in groups.values())
    rows = []
    for name, g in sorted(groups.items(), key=lambda kv: -kv[1]["ms"]):
        rows.append(
            {
                "component": name,
                "ms": round(g["ms"], 3),
                "pct_of_chunk": round(100.0 * g["ms"] / chunk_ms, 2) if chunk_ms else 0.0,
                "calls": int(round(g["calls"])),
                "tflops": round(g["flops"] / (g["ms"] / 1e3) / 1e12, 1)
                if g["ms"] > 0 and g["flops"]
                else 0.0,
            }
        )
    return {
        "chunk_ms": round(chunk_ms, 3),
        "tokens": int(tokens),
        "tok_s": round(tokens / (chunk_ms / 1e3), 1) if chunk_ms else 0.0,
        "attributed_ms": round(attributed, 3),
        "residual_ms": round(chunk_ms - attributed, 3),
        "residual_pct": round(100.0 * (chunk_ms - attributed) / chunk_ms, 2) if chunk_ms else 0.0,
        "components": rows,
        "labels": {
            k: {
                "ms": round(v["ms"], 4),
                "calls": int(round(v["calls"])),
                "tflops": round(v.get("tflops", 0.0), 1),
            }
            for k, v in sorted(labels.items(), key=lambda kv: -kv[1]["ms"])
        },
    }


def format_table(table: Dict[str, Any], *, top_labels: int = 24) -> str:
    """The block a run log should be readable from without downloading."""
    lines = [
        f"  chunk {table['tokens']} tok in {table['chunk_ms']:.1f} ms "
        f"= {table['tok_s']:,.0f} tok/s   "
        f"(attributed {table['attributed_ms']:.1f} ms, residual "
        f"{table['residual_ms']:.1f} ms = {table['residual_pct']:.1f}%)",
        f"  {'component':<16} {'ms':>9} {'% chunk':>8} {'calls':>7} {'TFLOP/s':>9}",
    ]
    for r in table["components"]:
        lines.append(
            f"  {r['component']:<16} {r['ms']:>9.2f} {r['pct_of_chunk']:>7.1f}% "
            f"{r['calls']:>7} {r['tflops']:>9.1f}"
        )
    lines.append(f"  {'-- top labels':<16}")
    for label, v in list(table["labels"].items())[:top_labels]:
        lines.append(
            f"    {label:<46} {v['ms']:>8.2f} ms  {v['calls']:>4} calls  "
            f"{v['tflops']:>7.1f} TFLOP/s"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 3. the patches
# --------------------------------------------------------------------------- #
def _wrap_fn(tape: EventTape, fn: Callable, label: str) -> Callable:
    def wrapper(*a, **kw):
        if not tape.active:
            return fn(*a, **kw)
        h = tape.enter(label)
        try:
            return fn(*a, **kw)
        finally:
            tape.exit(h)

    wrapper.__qwenfast_attrib__ = True  # type: ignore[attr-defined]
    return wrapper


def _gemm_label(site: str, m: int, backend: str) -> str:
    return f"gemm.{site}.M{m}.{backend}"


def _resolved_backend_name(rl, m: int, gemm_dispatch, fm) -> str:
    """What ``ResolvedLinear.__call__`` will actually route this call to.

    Mirrors its own branch order (forced pin -> prefill scope -> per-M-bucket
    memo) rather than reading ``rl.backend``, which is only the *last* resolved
    name and is therefore wrong for every layer whose bucket differs from the
    one warmup happened to resolve last.
    """
    if rl.forced is not None:
        return str(rl.forced)
    scoped = fm._PREFILL_GEMM_BACKEND[0]  # noqa: SLF001 -- profiler introspection
    if scoped is not None:
        return str(scoped if rl.is_fp8 else "bf16_native")
    try:
        bucket = gemm_dispatch.m_bucket(gemm_dispatch.sequence_m(m))
    except Exception:  # pragma: no cover -- dispatch is import-safe everywhere
        return "unknown"
    return str(rl._by_bucket.get(bucket, "unresolved"))  # noqa: SLF001


def linear_sites(model) -> Dict[int, str]:
    """``{id(ResolvedLinear): site name}`` for every linear in the model.

    Keyed on ``id`` because ``ResolvedLinear`` defines ``__slots__`` and
    therefore cannot carry a profiler attribute, and because the label must be
    the *site* (``mlp.gate_up``) rather than anything derivable from the
    weight -- two layers with identical shapes are not the same line of the
    attribution table.
    """
    sites: Dict[int, str] = {}

    def add(obj, name):
        if obj is not None:
            sites[id(obj)] = name

    for layer in getattr(model, "layers", []):
        mixer = getattr(layer, "mixer", None)
        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            add(getattr(mlp, "gate_up", None), "mlp.gate_up")
            add(getattr(mlp, "down", None), "mlp.down")
        if mixer is not None:
            add(getattr(mixer, "in_proj_qkvz", None), "gdn.in_proj_qkvz")
            add(getattr(mixer, "in_proj_ba", None), "gdn.in_proj_ba")
            add(getattr(mixer, "out_proj", None), "gdn.out_proj")
            add(getattr(mixer, "qkv", None), "attn.qkv")
            add(getattr(mixer, "o_proj", None), "attn.o_proj")
    add(getattr(model, "lm_head", None), "lm_head")
    return sites


def instrument(model, tape: EventTape) -> Callable[[], None]:
    """Bracket every prefill component of ``model``; returns ``restore()``.

    Patches, in one place so the list is auditable:

    ===========================  =================================================
    label prefix                 what it brackets
    ===========================  =================================================
    ``gemm.<site>.M<m>.<be>``    every ``ResolvedLinear`` call, by site, M and
                                 resolved backend -- the by-shape/by-backend
                                 GEMM table against a *real* chunk
    ``gdn.conv.<layout>``        the depthwise causal conv (both layouts)
    ``gdn.chunk``                fla's ``chunk_gated_delta_rule``
    ``gdn.state_io.{gather,scatter}``  the per-layer fp16<->fp32 state copies
    ``gdn.gate``                 the GDN gate epilogue (fused or eager)
    ``gdn.norm``                 the gated RMSNorm
    ``attn.prefill``             FlashInfer ``BatchPrefillWithPagedKVCache``
    ``attn.kv_append``           the paged KV write
    ``attn.qk_norm_rope``        fused QK-norm + RoPE
    ``norm.{rms,add_rms}``       the two per-layer RMSNorms
    ``mlp.act``                  SwiGLU
    ===========================  =================================================

    Anything not in that list -- the embedding gather, the residual adds, the
    ``index_select`` of last tokens, the ``.float()`` on logits, and every gap
    between launches -- lands in the table's ``residual``.
    """
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "prefill_attrib.instrument() must not run inside a CUDA graph capture: "
            "the patches are Python-level and a captured graph would replay the "
            "unpatched kernels while the tape recorded nothing."
        )

    from ..attn import flashinfer_attn as fi
    from ..attn import fused_qk_rope
    from ..gemm import dispatch as gemm_dispatch
    from ..kernels_gdn import api as gdn_api
    from ..kernels_gdn import fla_ops as gdn_fla_ops
    from ..kernels_gdn import torch_ops as gdn_torch_ops
    from . import fused_model as fm
    from .fused_ops import gdn_gate as fused_gdn_gate
    from .fused_ops import swiglu as fused_swiglu

    undo: List[Callable[[], None]] = []

    def patch_module(mod, attr: str, label: str) -> None:
        fn = getattr(mod, attr, None)
        if fn is None or getattr(fn, "__qwenfast_attrib__", False):
            return
        setattr(mod, attr, _wrap_fn(tape, fn, label))
        undo.append(lambda: setattr(mod, attr, fn))

    # -- GDN ---------------------------------------------------------------- #
    # The *kernel*, not `gdn_api.gdn_prefill_chunked` -- the API function also
    # contains the state gather/scatter, and bracketing both would double-count
    # them and break the one property this table has that the ablation table
    # does not (that its rows add up).
    patch_module(gdn_fla_ops, "chunk_gdn", "gdn.chunk.fla")
    patch_module(gdn_torch_ops, "chunk_gdn", "gdn.chunk.torch")
    patch_module(gdn_api, "causal_conv_prefill_varlen", "gdn.conv.token_major")
    patch_module(gdn_api, "causal_conv_prefill", "gdn.conv.channel_major")
    patch_module(gdn_api, "gather_states", "gdn.state_io.gather")
    patch_module(gdn_api, "scatter_states", "gdn.state_io.scatter")
    patch_module(fused_gdn_gate, "triton_gdn_gate", "gdn.gate")
    patch_module(fm, "triton_rms_norm_gated", "gdn.norm")
    patch_module(fm, "rms_norm_gated", "gdn.norm")

    # -- norms / activations ------------------------------------------------ #
    patch_module(fm, "triton_rms_norm", "norm.rms")
    patch_module(fm, "rms_norm_w1p", "norm.rms")
    patch_module(fm, "triton_add_rms_norm", "norm.add_rms")
    patch_module(fm, "add_rms_norm_w1p", "norm.add_rms")
    patch_module(fused_swiglu, "triton_swiglu", "mlp.act")
    patch_module(fm, "swiglu", "mlp.act")

    # -- attention ---------------------------------------------------------- #
    # `FusedAttention._qkv` calls
    # `fused_qk_rope.qk_norm_rope` (prefill/mixed: the new Triton kernel;
    # decode/window: an unchanged pass-through to `fi.fused_qk_norm_rope`),
    # so *that* is the call site to bracket -- patching `fi.fused_qk_norm_rope`
    # alone would miss every prefill call, since `fused_qk_rope`'s eager
    # fallback holds its own direct reference to it captured at import time,
    # not a dynamic `fi.fused_qk_norm_rope` attribute lookup.
    patch_module(fused_qk_rope, "qk_norm_rope", "attn.qk_norm_rope")
    patch_module(fi, "apply_output_gate", "attn.gate")

    # `AttnRunner.prefill` and `PagedKVPool.append_kv` are bound methods on
    # singletons; patch the instance so an un-instrumented second model in the
    # same process (there never is one, but the profiler's A/Bs rebuild
    # wrappers) is untouched.
    for obj, attr, label in (
        (getattr(model, "attn", None), "prefill", "attn.prefill"),
        (getattr(model, "kv_pool", None), "append_kv", "attn.kv_append"),
    ):
        if obj is None:
            continue
        fn = getattr(obj, attr, None)
        if fn is None or getattr(fn, "__qwenfast_attrib__", False):
            continue
        try:
            setattr(obj, attr, _wrap_fn(tape, fn, label))
        except AttributeError:  # pragma: no cover -- __slots__ class
            continue
        undo.append(lambda o=obj, a=attr: delattr(o, a))

    # -- GEMMs -------------------------------------------------------------- #
    sites = linear_sites(model)
    orig_call = fm.ResolvedLinear.__call__
    if not getattr(orig_call, "__qwenfast_attrib__", False):

        def call(self, x):
            if not tape.active:
                return orig_call(self, x)
            site = sites.get(id(self), "other")
            m = 1
            for d in x.shape[:-1]:
                m *= int(d)
            k = int(x.shape[-1])
            h = tape.enter(
                _gemm_label(site, m, _resolved_backend_name(self, m, gemm_dispatch, fm))
            )
            try:
                return orig_call(self, x)
            finally:
                tape.exit(h, flops=2.0 * m * k * int(self.n))

        call.__qwenfast_attrib__ = True  # type: ignore[attr-defined]
        fm.ResolvedLinear.__call__ = call  # type: ignore[method-assign]
        undo.append(lambda: setattr(fm.ResolvedLinear, "__call__", orig_call))

    def restore() -> None:
        for fn in reversed(undo):
            fn()
        undo.clear()

    return restore


__all__ = [
    "EventTape",
    "attribution_table",
    "format_table",
    "group_of",
    "instrument",
    "linear_sites",
]
