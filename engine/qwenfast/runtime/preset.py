"""qwenfast: the one canonical benchmark configuration.

Why this module exists. The same nominal config ("default qwenfast decode,
bf16 KV, ctx 2048") measured by different runs produced B=32 step times
anywhere from 18.1 to 41.4 ms/step: a 2.3x spread on the same headline
number. Two of the causes are identified, and neither is noise:

1. **`fused_ops_backend`.** The fused GDN-gate/SwiGLU Triton kernels live
   behind `RuntimeConfig.fused_ops_backend`, which defaults to `"torch"`.
   Some runs used `"triton"` and some `"torch"`, the flag was not recorded
   in most result JSONs, and `serve.py` did not expose it at all, so the
   server ran the slower path while benchmarks quoted the faster one.
2. **The CUDA-graph bucket table.** Three bf16-KV runs used 13, 6 and 2
   captured buckets respectively, and their B=32 times were 30.0 / 20.4 / 18.1
   ms: monotone in the number of buckets, and a 66% effect at the extremes.
   `run_sweep` derives `buckets` from `--buckets` (or the 15-entry
   `RuntimeConfig.graph_buckets` default) independently of `--batch`, so two
   runs benching the same batch sizes can capture different graph sets.
   Whatever the mechanism (one shared `graph_pool_handle` whose working set
   grows with the number of captures is the leading candidate), a ms/step
   number is not interpretable without the bucket table beside it.

So: one preset, one place, every bench prints it.

    `CANONICAL_FAST`  — the RuntimeConfig knobs. Must equal `serve.M1_DEFAULTS`
                        on every key the two share; `tests/test_preset.py
                        ::TestPresetMatchesServe` is that assertion, so the
                        thing being benchmarked is the thing being served.
    `CANONICAL_BENCH` — the measurement knobs (bucket rule, steps, warmup,
                        repeats, ctx). Not part of RuntimeConfig, but they
                        change the number just as much.

Usage from a bench CLI::

    p.add_argument("--preset", choices=sorted(PRESETS), default=None)
    ...
    args = apply_preset(args, p)            # fills in unset knobs
    print(format_resolved_config(rt, ...))  # always, preset or not

`--preset fastest` never *overrides* an explicitly-passed flag; it only fills
in what the caller left at the CLI default, so an A/B like
`--preset fastest --gemm-backend deepgemm` means exactly what it reads as.

This module must import with no torch available (it is imported by
`tests/test_preset.py`, which runs on CPU-only machines): `RuntimeConfig` is
imported lazily inside the two functions that build one.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# 1. the preset
# --------------------------------------------------------------------------- #
#: The measured-fastest `RuntimeConfig` knobs.
#:
#: Every entry is a *measurement*:
#:   ssm_state_dtype fp16   a wash at B<=8, 5-17% faster from B=32, and it
#:                          halves the 144 MiB/slot state pool.
#:   norm_backend triton    B=1 15.375 -> 12.638 ms (-17.8%),
#:                          B=32 22.629 -> 18.407 ms (-18.7%).
#:   fused_ops_backend triton  B=1 12.638 -> 12.352 ms, B=128
#:                          28.677 -> 27.708 ms, launches/step 1,945 -> 1,689.
#:   kv_cache_dtype bf16    fp8 KV is +2.6% at B=1/8 and -23% at B=256; it is
#:                          a *capacity* lever, not a speed one. Not the
#:                          fastest value at every batch size, and
#:                          deliberately so: this preset is "the config the
#:                          engine ships", not "the argmax per data point".
#:   gemm_backend None      pinning bucket 1's winner costs 5x at M=256.
#:                          Dispatch per M-bucket.
#:   gemm_weight_cache single  "multi" duplicates all 23.0 GiB of fp8 linears
#:                          and runs the server out of memory.
#:   gemm_priority v8       v7 (the graph-timed gemm_v7 sweep) plus the four
#:                          prefill-scale buckets it clamped to the M=512
#:                          answer.
#:   gemm_accuracy fast     "strict" is measured for accuracy but not yet
#:                          priced for speed.
#:   attn_backend auto      -> flashinfer, which graph capture requires.
CANONICAL_FAST: Dict[str, Any] = {
    "dtype": "bf16",
    "ssm_state_dtype": "fp16",
    "kv_cache_dtype": "bf16",
    "page_size": 16,
    "gdn_backend": "auto",
    "gemm_backend": None,
    "gemm_weight_cache": "single",
    "gemm_priority": "v8",
    "gemm_accuracy": "fast",
    "attn_backend": "auto",
    "norm_backend": "triton",
    "fused_ops_backend": "triton",
    "use_cuda_graphs": True,
    "sampler_candidates": 2048,
    "attn_workspace_mb": 512,
    "conv_prefill_tile_tokens": 2048,
    "conv_prefill_layout": "token_major",
    "prefill_gemm_backend": None,
    "prefill_chunk_tokens": 0,
    #: One forward per step over [prefill chunk || decode rows].
    #: **False**, and deliberately: this preset is "the config the engine
    #: ships", and the mixed step is projected to win only at concurrency >= 64
    #: (at conc 1-8 it replaces a graphed decode step with an eager one and can
    #: only lose). It is in the preset because it changes the M every GEMM in a
    #: step routes on, which is precisely the class of knob this module exists
    #: for, and so both sides of an A/B on it are nameable here.
    "mixed_forward": False,
    #: Whether the mixed step is CUDA-graphed. In the preset for the same
    #: reason `mixed_forward` is: it changes the step's shape (every mixed step
    #: is padded to a fixed chunk/bucket), which changes the M every GEMM in
    #: that step routes on.
    "mixed_graphs": False,
    #: The prefill plan-row count a graphed mixed step is padded to.
    #: Inert while `mixed_graphs` is off; part of the step's shape when it is.
    "mixed_graph_segments": 8,
    #: Inert while `mixed_graphs` is off. False == the one-graph capture;
    #: True == 49 graph segments + 48 eager holes.
    "mixed_graph_holes": False,
    "mixed_graph_min_bucket": 32,
    #: Two streams, two graphs, no fused forward. In the preset
    #: for the same reason `mixed_forward` is: it changes what a step *is*.
    "overlap_streams": False,
    #: Inert while `overlap_streams` is off.
    "overlap_min_fill": 0.75,
    #: Asynchronous step scheduling. Off in the canonical
    #: preset: the preset is the *shape* of the fast configuration, and this is
    #: a serving-loop mode that the offline bench (which drives one step at a
    #: time by hand) cannot exercise at all.
    "async_scheduling": False,
    #: The prefill MLP tile. 2048 is the historical value and is kept as the
    #: canonical one until raising it is priced, but it is *in* the preset
    #: because it changes which GEMM M-bucket 64 of the model's 305 linears
    #: route on, which is exactly the kind of knob that is easy to bench one
    #: way and serve the other (as happened with `fused_ops_backend` and
    #: `gemm_priority`).
    "mlp_tile_tokens": 2048,
    #: Which M-bucket's winner claims the single repack-cache
    #: slot under `gemm_weight_cache="single"`. Tracked here (rather than left
    #: to serve.py alone) because it decides whether a priority-table change at
    #: M>=32 is reachable at all -- see `dispatch.V7_BACKEND_PRIORITY_BY_M_BUCKET`.
    "gemm_cache_owner": "decode",
    #: fla's `chunk_gated_delta_rule` `chunk_size` (`BT`) at the real
    #: 8,192-token prefill shape. `RuntimeConfig`'s own field default (64) is
    #: fla's library default, tuned for training-length sequences; an A/B on
    #: the real model (`64=565ms 32=550ms 16=559ms`) measured 32 as 2.7%
    #: faster than 64, and `test_serving_path.py::TestGdnChunkSizeParity` pins
    #: that the torch-backend (CPU-testable) math is bit-for-bit invariant
    #: across 16/32/64 on the tiny model. Flipping the preset without that
    #: parity check would be the "benched one way, served another" failure
    #: this module exists to prevent, just for correctness instead of speed.
    "gdn_chunk_size": 32,
}

#: The measurement knobs. Not RuntimeConfig fields, but they move ms/step by as
#: much as any of them (see this module's docstring).
#:
#: ``buckets = "batches"`` is the rule, not a list: **capture exactly the
#: buckets you bench.** It is the only rule that is reproducible across runs
#: that bench different batch sets, and it is what `profile_step.py` already
#: does (`buckets = tuple(sorted(set(args.batch)))`). It is *not* what a
#: server does (a server captures all 15 of `RuntimeConfig.graph_buckets`),
#: so the preset also names `serve_buckets` for measuring the gap between
#: them.
CANONICAL_BENCH: Dict[str, Any] = {
    "buckets": "batches",
    "ctx_len": 2048,
    "steps": 30,
    "warmup": 5,
    "repeats": 3,
    "batches": (1, 8, 32, 64, 128, 256),
}

#: The serve-realistic bucket table, for the A/B that prices the difference.
SERVE_BUCKETS: Tuple[int, ...] = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)

PRESETS: Dict[str, Dict[str, Any]] = {"fastest": CANONICAL_FAST}

#: Keys of `CANONICAL_FAST` that `serve.M1_DEFAULTS` must agree with. Anything
#: in `CANONICAL_FAST` and not here is a knob serve.py does not expose, which
#: `tests/test_preset.py` treats as a failure, not an exemption: that kind of
#: gap is how a knob ends up benched fast and served slow.
SERVE_SHARED_KEYS: Tuple[str, ...] = tuple(CANONICAL_FAST)


# --------------------------------------------------------------------------- #
# 2. building / reporting a config
# --------------------------------------------------------------------------- #
def canonical_fast_config(**overrides: Any):
    """A :class:`~.fused_model.RuntimeConfig` at the canonical preset.

    ``overrides`` are applied last, so
    ``canonical_fast_config(gemm_backend="deepgemm")`` is the A/B arm."""
    from .fused_model import RuntimeConfig

    kwargs = dict(CANONICAL_FAST)
    kwargs.update(overrides)
    return RuntimeConfig(**kwargs)


def buckets_for_batches(batches: Sequence[int], rule: str = "batches") -> Tuple[int, ...]:
    """The canonical bucket table for a set of benched batch sizes.

    ``"batches"`` (the canonical rule): capture exactly what you bench.
    ``"serve"``: the 15-entry table a real server captures, capped at the
    largest batch (which is what `run_sweep` would otherwise do implicitly)."""
    top = max(batches) if batches else 1
    if rule == "serve":
        keep = tuple(b for b in SERVE_BUCKETS if b <= top)
        return keep if keep and keep[-1] == top else keep + (top,)
    if rule != "batches":
        raise ValueError(f"bucket rule must be 'batches' or 'serve', got {rule!r}")
    return tuple(sorted(set(int(b) for b in batches))) or (1,)


#: The RuntimeConfig fields that can change a decode number. Anything absent
#: from this list is either a pool-geometry field (recorded separately by the
#: bench) or genuinely inert; adding a knob to `RuntimeConfig` without adding it
#: here is how the next `fused_ops_backend` happens.
REPORTED_FIELDS: Tuple[str, ...] = (
    "dtype", "ssm_state_dtype", "kv_cache_dtype", "page_size",
    "gdn_backend", "gemm_backend", "gemm_weight_cache", "gemm_priority",
    "gemm_accuracy", "attn_backend", "norm_backend", "fused_ops_backend",
    "use_cuda_graphs", "sampler_candidates", "attn_workspace_mb",
    "conv_prefill_layout", "conv_prefill_tile_tokens", "mlp_tile_tokens",
    "prefill_gemm_backend", "prefill_chunk_tokens", "gemm_cache_owner",
    "gdn_chunk_size", "mixed_forward", "mixed_graphs", "mixed_graph_segments",
    "mixed_graph_holes", "mixed_graph_min_bucket", "overlap_streams",
    "overlap_min_fill", "async_scheduling",
    "max_num_seqs", "n_kv_pages", "max_pages_per_seq", "graph_buckets",
)


def resolved_config(rt, **extra: Any) -> Dict[str, Any]:
    """``{knob: value}`` for everything that can move a number, plus ``extra``
    (the measurement knobs the bench owns: steps, warmup, repeats, ctx_len).

    Includes ``preset_match``: the keys where ``rt`` differs from
    `CANONICAL_FAST`, so a results JSON says *how* it deviates rather than
    leaving a reader to diff two documents by eye."""
    out: Dict[str, Any] = {}
    for name in REPORTED_FIELDS:
        v = getattr(rt, name, None)
        out[name] = list(v) if isinstance(v, tuple) else v
    out.update(extra)
    out["preset"] = "fastest"
    out["preset_deviations"] = {
        k: getattr(rt, k, None)
        for k, v in CANONICAL_FAST.items()
        if getattr(rt, k, v) != v
    }
    return out


def format_resolved_config(rt, **extra: Any) -> str:
    """One-line-per-knob block every bench prints before its first number."""
    cfg = resolved_config(rt, **extra)
    dev = cfg.pop("preset_deviations")
    lines = ["[config] resolved runtime configuration (preset=fastest):"]
    for k, v in cfg.items():
        lines.append(f"[config]   {k} = {v}")
    if dev:
        lines.append(f"[config]   *** DEVIATIONS from CANONICAL_FAST: {dev}")
    else:
        lines.append("[config]   (exactly CANONICAL_FAST -- comparable to every "
                     "other --preset fastest run)")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 3. argparse glue
# --------------------------------------------------------------------------- #
def add_preset_arg(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument(
        "--preset", choices=sorted(PRESETS), default=None,
        help="fill every knob left at its CLI default from the named canonical "
             "preset (qwenfast.runtime.preset.CANONICAL_FAST -- the measured-"
             "fastest, serve-identical config). Explicitly-passed flags always "
             "win. Use this for any number meant to be compared with another "
             "run's.",
    )
    return p


def apply_preset(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    *,
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    """Fill in every ``args`` attribute the caller left at the parser default
    with the preset's value. Returns ``args`` (mutated in place).

    "Left at the default" is decided by re-parsing with an all-``None``
    default map rather than by comparing values, so passing a flag whose value
    happens to equal the parser default still counts as explicit."""
    name = getattr(args, "preset", None)
    if not name:
        return args
    preset = PRESETS[name]
    explicit = _explicitly_passed(parser, argv)
    for key, value in preset.items():
        if not hasattr(args, key):
            continue
        if key in explicit:
            continue
        # Some CLIs spell a scalar knob as `nargs="+"` so one run can sweep it
        # (`bench_runtime --ssm-state-dtype fp32 fp16`). Setting a bare string
        # there would make the caller iterate its characters, so match the
        # existing container-ness rather than the preset's.
        current = getattr(args, key)
        if isinstance(current, list) and not isinstance(value, list):
            value = [value]
        setattr(args, key, value)
    # `use_cuda_graphs` is spelled `--no-graphs` / `--graphs {on,off,both}` by
    # the different CLIs; leave those alone, they default to graphs-on already.
    return args


def _explicitly_passed(parser: argparse.ArgumentParser, argv: Optional[Sequence[str]]) -> set:
    """Dest names the user actually typed on the command line."""
    import sys

    tokens = list(sys.argv[1:] if argv is None else argv)
    typed = {t.split("=", 1)[0] for t in tokens if t.startswith("--")}
    out = set()
    for action in parser._actions:  # noqa: SLF001 -- argparse has no public API for this
        if any(opt in typed for opt in action.option_strings):
            out.add(action.dest)
    return out


__all__ = [
    "CANONICAL_FAST",
    "CANONICAL_BENCH",
    "SERVE_BUCKETS",
    "SERVE_SHARED_KEYS",
    "PRESETS",
    "REPORTED_FIELDS",
    "canonical_fast_config",
    "buckets_for_batches",
    "resolved_config",
    "format_resolved_config",
    "add_preset_arg",
    "apply_preset",
]
