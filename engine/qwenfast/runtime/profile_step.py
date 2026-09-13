#!/usr/bin/env python
"""Where does a decode step's time actually go?

``bench_runtime.py`` answers *how fast*; this answers *why*.  End-to-end step
times can be several times the sum of the component microbenchmarks that are
supposed to compose into them (best GEMM backend 8.2 ms whole-model at M=1,
GDN decode 3.8 us x 48 layers, FlashInfer decode 48 us x 16 layers), and
neither number on its own explains such a gap.

Four independent views of one decode step, all on the real checkpoint:

1. **Resolved GEMM backend per linear layer.**  Which backend each
   ``ResolvedLinear`` actually pinned, per M-bucket, printed as a histogram
   and dumped in full to the JSON.  A single wrong entry here is worth
   ~23 ms/step at M=1 (``vllm_block_fp8_triton`` 31.5 ms vs marlin 8.2 ms),
   so it is view #1, not a footnote.

2. **Host vs device split.**  ``GraphedDecoder.step`` is
   ``plan_decode()`` (host, outside the graph) + ``graph.replay()``.  Timed
   separately with ``perf_counter`` around a synchronised loop, because a
   host-side sync storm inside ``plan_decode`` is invisible to any
   device-side profiler -- it shows up only as the GPU sitting idle.

3. **Kernel table from graph replay** (``torch.profiler`` with CUDA
   activities): top-N kernels by total device time, plus the same rows
   folded into op/backend groups (GEMM-by-backend, GDN, attention,
   norm/elementwise, sampler, embedding, copies).  Graph replay is the
   only regime worth profiling: eager timings rank backends *differently*
   than replay does.

4. **Ablation deltas.**  The same step re-captured with one component
   stubbed out (sampler / lm_head / attention mixers / GDN mixers / MLPs),
   each timed by CUDA events around ``replay()``.  Deltas are **not
   additive** -- removing work also removes the launch-latency tail it was
   hiding -- so they are reported as "cost attributable to", not "cost of".

Runs on a GPU host only (needs the real checkpoint); imports cleanly on a
CPU-only machine.  Example::

    python -m qwenfast.runtime.profile_step \\
        --model $FP8 --batch 1 32 --ctx-len 2048 \\
        --ssm-state-dtype fp16 --out /home/qwenfast-results/profile_step.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Sequence

import torch

from . import graphs as graphs_mod
from .bench_runtime import (
    _fake_context,
    calibrate_kv_fp8,
    check_resolved_backends,
    collect_gemm_backends,
    decode_ceiling_tok_s,
    derive_pool_sizes,
    summarize_backends,
)
from .engine import EngineComponents, build_engine
from .fused_model import RuntimeConfig
from .preset import (
    add_preset_arg,
    apply_preset,
    format_resolved_config,
    resolved_config,
)

# --------------------------------------------------------------------------- #
# kernel-name -> group.  Substring match, first hit wins, so order matters.
# The names are the CUDA kernel symbols torch.profiler reports; the tags come
# from the backends registered in ``gemm/dispatch.py`` and the kernels in
# ``kernels_gdn/`` and ``attn/``.
# --------------------------------------------------------------------------- #
_KERNEL_GROUPS: "OrderedDict[str, List[str]]" = OrderedDict(
    [
        ("gemm/marlin", ["marlin", "Marlin", "MARLIN"]),
        ("gemm/flashinfer_blockscale", ["fp8_blockscale", "blockscale_gemm", "trtllm_gemm"]),
        ("gemm/cutlass", ["cutlass", "Cutlass", "CUTLASS", "sm90_", "sm100_"]),
        ("gemm/cublas_scaled_mm", ["gemm_", "nvjet", "cublas", "Kernel_", "ampere_", "hopper_"]),
        ("gemm/triton_block_fp8", ["_w8a8_block_fp8_matmul", "triton_block", "w8a8"]),
        ("gemm/quant", ["per_token_group_quant", "scaled_fp8_quant", "quant_fp8"]),
        ("gdn", ["gdn", "delta_rule", "recurrent", "conv_update", "causal_conv"]),
        ("attention", ["BatchDecode", "BatchPrefill", "paged", "flashinfer", "attention", "Attn"]),
        ("sampler", ["topk", "TopK", "radix", "sort", "Sort", "cumsum", "softmax", "gumbel",
                     "argmax", "philox", "rand"]),
        ("embedding", ["embedding", "index_select", "IndexSelect", "gather"]),
        ("copy", ["copy", "Copy", "memcpy", "direct_copy", "vectorized_elementwise_kernel<4"]),
    ]
)


def classify_kernel(name: str) -> str:
    for group, needles in _KERNEL_GROUPS.items():
        for nd in needles:
            if nd in name:
                return group
    return "elementwise/other"


# --------------------------------------------------------------------------- #
# "before the fix" emulation
# --------------------------------------------------------------------------- #
def install_legacy_plan() -> Callable[[], None]:
    """Restore the original ``build_flashinfer_indices``: one D2H sync per
    KV page per sequence.

    This exists so a before/after comparison of the sync-free index build is
    two *measurements* rather than one measurement and one remembered number.  It is the
    original method verbatim -- a Python loop over slots doing
    ``int(self.seq_len[slot])`` and ``int(self.page_table[slot, i])`` --
    reinstalled on the class.
    """
    from ..attn.kv_pool import PagedKVPool

    saved = PagedKVPool.build_flashinfer_indices

    def legacy(self, slot_ids, seq_lens=None, *, staged=False):
        # `staged` accepted and ignored: this emulation is
        # deliberately the *slow* path, and its whole point is the syncs.
        page_size = self.cfg.page_size
        indptr = [0]
        indices: List[int] = []
        last_page_len: List[int] = []
        lens: List[int] = []
        for slot in slot_ids:
            n = int(self.seq_len[slot])           # D2H sync #1, per sequence
            n_pages = self.pages_needed(n)
            pages = [int(self.page_table[slot, i]) for i in range(n_pages)]  # D2H per PAGE
            indices.extend(pages)
            indptr.append(indptr[-1] + n_pages)
            last = n - (n_pages - 1) * page_size if n_pages > 0 else 0
            last_page_len.append(last if last > 0 else (page_size if n_pages > 0 else 0))
            lens.append(n)
        device = self.kv.device
        return (
            torch.tensor(indptr, dtype=torch.int32, device=device),
            torch.tensor(indices or [0], dtype=torch.int32, device=device),
            torch.tensor(last_page_len or [0], dtype=torch.int32, device=device),
            torch.tensor(lens or [0], dtype=torch.int32, device=device),
        )

    PagedKVPool.build_flashinfer_indices = legacy  # type: ignore[assignment]

    def restore() -> None:
        PagedKVPool.build_flashinfer_indices = saved  # type: ignore[assignment]

    return restore


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def build(
    model_dir: str,
    *,
    ssm_state_dtype: str,
    use_cuda_graphs: bool,
    buckets: Sequence[int],
    ctx_len: int,
    page_size: int,
    kv_cache_dtype: str,
    gemm_backend: Optional[str],
    norm_backend: str,
    device: str,
    verbose: bool,
    fused_ops_backend: str = "torch",
    gemm_accuracy: str = "fast",
    gemm_weight_cache: str = "multi",
    gemm_priority: str = "v8",
) -> EngineComponents:
    max_num_seqs = max(buckets)
    geom = derive_pool_sizes(max_num_seqs, ctx_len, page_size)
    rt = RuntimeConfig(
        device=device,
        ssm_state_dtype=ssm_state_dtype,
        use_cuda_graphs=use_cuda_graphs,
        max_num_seqs=max_num_seqs,
        graph_buckets=tuple(buckets),
        page_size=page_size,
        kv_cache_dtype=kv_cache_dtype,
        gemm_backend=gemm_backend,
        gemm_accuracy=gemm_accuracy,
        gemm_weight_cache=gemm_weight_cache,
        gemm_priority=gemm_priority,
        norm_backend=norm_backend,
        fused_ops_backend=fused_ops_backend,
        n_kv_pages=geom["n_kv_pages"],
        max_pages_per_seq=geom["max_pages_per_seq"],
    )
    comps = build_engine(model_dir, rt=rt, verbose=verbose)
    # fp8 KV calibration before warmup/capture, the same
    # requirement as bench_runtime._build.
    calibrate_kv_fp8(comps.model, verbose=verbose)
    comps.decoder.warmup()
    if use_cuda_graphs:
        comps.decoder.capture()
    return comps


def prepare_batch(comps: EngineComponents, batch: int, ctx_len: int) -> List[int]:
    """Fake ``batch`` sequences with ``ctx_len`` of committed context and fill
    the persistent device buffers, exactly as ``bench_runtime.bench_decode`` does."""
    model, buf, decoder = comps.model, comps.buf, comps.decoder
    slots = list(range(batch))
    _fake_context(model, slots, ctx_len)
    bucket = decoder.bucket_for(batch)
    pad = bucket - batch
    full_slots = slots + [model.scratch_slot] * pad

    buf.host["input_ids"][:bucket] = torch.zeros(bucket, dtype=torch.int32)
    buf.host["positions"][:bucket] = torch.tensor([ctx_len] * batch + [0] * pad, dtype=torch.int32)
    buf.host["slot_ids"][:bucket] = torch.tensor(full_slots, dtype=torch.int32)
    buf.host["temperature"][:bucket] = torch.ones(bucket)
    buf.host["top_p"][:bucket] = torch.ones(bucket)
    buf.host["top_k"][:bucket] = torch.zeros(bucket)
    buf.upload(["input_ids", "positions", "slot_ids", "temperature", "top_p", "top_k"])
    return full_slots


# --------------------------------------------------------------------------- #
# view 2: host vs device
# --------------------------------------------------------------------------- #
def host_device_split(
    comps: EngineComponents, batch: int, full_slots: Sequence[int], *, steps: int, warmup: int
) -> Dict[str, float]:
    """ms/step for: the whole ``decoder.step``, its ``plan_decode`` half, and
    its ``replay``/eager half, plus the device-only replay time from CUDA
    events.

    ``plan_decode`` is host-side and (with the legacy index build) does one D2H sync
    per KV page per sequence; that time is pure GPU idle and shows up in no
    kernel trace, which is precisely why it is measured here with a wall
    clock and not inferred from the profiler.
    """
    decoder, model = comps.decoder, comps.model
    bucket = decoder.bucket_for(batch)

    for _ in range(warmup):
        decoder.step(batch, full_slots)
    torch.cuda.synchronize()

    # (a) whole step
    t0 = time.perf_counter()
    for _ in range(steps):
        decoder.step(batch, full_slots)
    torch.cuda.synchronize()
    total_ms = (time.perf_counter() - t0) / steps * 1e3

    # (b) plan only
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        model.attn.plan_decode(list(full_slots), bucket)
    torch.cuda.synchronize()
    plan_ms = (time.perf_counter() - t0) / steps * 1e3

    # (c) replay only, host wall time
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        if bucket in decoder._graphs:  # noqa: SLF001
            decoder._graphs[bucket].replay()  # noqa: SLF001
        else:
            decoder._run_step(bucket)  # noqa: SLF001
    torch.cuda.synchronize()
    replay_wall_ms = (time.perf_counter() - t0) / steps * 1e3

    # (d) replay only, device time from CUDA events
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(steps):
        if bucket in decoder._graphs:  # noqa: SLF001
            decoder._graphs[bucket].replay()  # noqa: SLF001
        else:
            decoder._run_step(bucket)  # noqa: SLF001
    end.record()
    torch.cuda.synchronize()
    replay_dev_ms = start.elapsed_time(end) / steps

    # Same measurement again, but *last*: (a) runs straight out of warmup, so
    # if the two disagree the difference is clock/warmup state, not the step.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        decoder.step(batch, full_slots)
    torch.cuda.synchronize()
    total_ms_repeat = (time.perf_counter() - t0) / steps * 1e3

    return {
        "step_ms": total_ms,
        "step_ms_repeat": total_ms_repeat,
        "plan_decode_ms": plan_ms,
        "replay_wall_ms": replay_wall_ms,
        "replay_device_ms": replay_dev_ms,
        "unaccounted_ms": total_ms - plan_ms - replay_wall_ms,
    }


# --------------------------------------------------------------------------- #
# view 3: kernel table
# --------------------------------------------------------------------------- #
def kernel_table(
    comps: EngineComponents, batch: int, full_slots: Sequence[int], *, steps: int, top_n: int
) -> Dict:
    """``torch.profiler`` over ``steps`` graph replays -> per-kernel and
    per-group total device time (us/step)."""
    decoder = comps.decoder
    bucket = decoder.bucket_for(batch)
    replay: Callable[[], None]
    if bucket in decoder._graphs:  # noqa: SLF001
        g = decoder._graphs[bucket]  # noqa: SLF001
        replay = g.replay
    else:
        replay = lambda: decoder._run_step(bucket)  # noqa: SLF001, E731

    for _ in range(3):
        replay()
    torch.cuda.synchronize()

    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(steps):
            replay()
        torch.cuda.synchronize()

    rows: List[Dict] = []
    for ev in prof.key_averages():
        dev_us = float(getattr(ev, "self_device_time_total", 0.0) or 0.0)
        if dev_us <= 0.0:
            continue
        rows.append(
            {
                "name": ev.key,
                "calls_per_step": ev.count / steps,
                "us_per_step": dev_us / steps,
                "group": classify_kernel(ev.key),
            }
        )
    rows.sort(key=lambda r: -r["us_per_step"])
    total_us = sum(r["us_per_step"] for r in rows)

    groups: Dict[str, Dict[str, float]] = {}
    for r in rows:
        gsum = groups.setdefault(r["group"], {"us_per_step": 0.0, "calls_per_step": 0.0, "kernels": 0})
        gsum["us_per_step"] += r["us_per_step"]
        gsum["calls_per_step"] += r["calls_per_step"]
        gsum["kernels"] += 1
    for gsum in groups.values():
        gsum["pct"] = 100.0 * gsum["us_per_step"] / total_us if total_us else 0.0

    return {
        "total_device_us_per_step": total_us,
        "n_distinct_kernels": len(rows),
        "total_launches_per_step": sum(r["calls_per_step"] for r in rows),
        "top_kernels": rows[:top_n],
        "groups": dict(sorted(groups.items(), key=lambda kv: -kv[1]["us_per_step"])),
    }


# --------------------------------------------------------------------------- #
# view 4: ablations
# --------------------------------------------------------------------------- #
class _Ablation:
    """Stub one component out, re-capture the step, time it, put it back."""

    def __init__(self, name: str, apply: Callable[[EngineComponents], Callable[[], None]]):
        self.name = name
        self.apply = apply


def _stub_mixers(comps: EngineComponents, kind: str) -> Callable[[], None]:
    """Replace every GDN (or attention) mixer's ``decode`` with a zero of the
    right shape.  Keeps the residual stream's *shape* and dtype so the rest
    of the step is unchanged; the values are meaningless, which is fine --
    this measures time, and nothing downstream branches on a value."""
    saved = []
    for layer in comps.model.layers:
        mixer = layer.mixer
        is_gdn = hasattr(mixer, "in_proj_qkvz")
        if (kind == "gdn") != is_gdn:
            continue
        saved.append((mixer, mixer.decode))
        mixer.decode = lambda h, ctx: torch.zeros_like(h)  # type: ignore[assignment]

    def restore() -> None:
        for m, fn in saved:
            m.decode = fn  # type: ignore[assignment]

    return restore


def _stub_mlp(comps: EngineComponents) -> Callable[[], None]:
    saved = []
    for layer in comps.model.layers:
        saved.append((layer.mlp, layer.mlp.__call__))
        layer.mlp = _ZeroCallable(layer.mlp)  # type: ignore[assignment]

    def restore() -> None:
        for i, (mlp, _) in enumerate(saved):
            comps.model.layers[i].mlp = mlp

    return restore


class _ZeroCallable:
    def __init__(self, wrapped):
        self._wrapped = wrapped

    def __call__(self, x, tiled: bool = False):
        return torch.zeros_like(x)

    def nbytes(self) -> int:
        return self._wrapped.nbytes()


def _stub_lm_head(comps: EngineComponents) -> Callable[[], None]:
    model = comps.model
    saved = model.lm_head
    vocab = model.config.vocab_size
    zeros = torch.zeros(comps.buf.max_batch, vocab, dtype=model.dtype, device=model.device)

    class _Stub:
        def __call__(self, h):
            return zeros[: h.shape[0]]

        def nbytes(self):
            return saved.nbytes()

        def resolved_backends(self):
            return saved.resolved_backends()

    model.lm_head = _Stub()  # type: ignore[assignment]

    def restore() -> None:
        model.lm_head = saved

    return restore


def _stub_sampler(comps: EngineComponents) -> Callable[[], None]:
    saved = graphs_mod.sample_tokens

    def _noop(logits, temperature, top_p, top_k, *, candidates=2048, generator=None):
        return logits[:, :1].squeeze(-1).to(torch.int32)

    graphs_mod.sample_tokens = _noop  # type: ignore[assignment]

    def restore() -> None:
        graphs_mod.sample_tokens = saved  # type: ignore[assignment]

    return restore


ABLATIONS: List[_Ablation] = [
    _Ablation("no_sampler", _stub_sampler),
    _Ablation("no_lm_head", _stub_lm_head),
    _Ablation("no_gdn_mixers", lambda c: _stub_mixers(c, "gdn")),
    _Ablation("no_attn_mixers", lambda c: _stub_mixers(c, "attn")),
    _Ablation("no_mlp", _stub_mlp),
]


def _time_step_body(comps: EngineComponents, bucket: int, *, steps: int, warmup: int) -> float:
    """Device ms for one *uncaptured-but-recaptured* step body at ``bucket``.

    Re-captures a throwaway graph so an ablation is measured under the same
    replay regime as the baseline (eager and replay rank things
    differently, so an eager ablation would not be comparable).
    """
    decoder = comps.decoder
    if not decoder.graphs_enabled:
        for _ in range(warmup):
            decoder._run_step(bucket)  # noqa: SLF001
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(steps):
            decoder._run_step(bucket)  # noqa: SLF001
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / steps

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        decoder._run_step(bucket)  # noqa: SLF001
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    if decoder._generator is not None and decoder._generator.device.type == "cuda":  # noqa: SLF001
        g.register_generator_state(decoder._generator)  # noqa: SLF001
    with torch.cuda.graph(g, pool=decoder._pool):  # noqa: SLF001
        decoder._run_step(bucket)  # noqa: SLF001
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(steps):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / steps
    del g
    return ms


def ablation_table(
    comps: EngineComponents, batch: int, full_slots: Sequence[int], *, steps: int, warmup: int
) -> Dict:
    bucket = comps.decoder.bucket_for(batch)
    comps.model.attn.plan_decode(list(full_slots), bucket)
    base = _time_step_body(comps, bucket, steps=steps, warmup=warmup)
    out: Dict = {"baseline_ms": base, "ablations": []}
    for ab in ABLATIONS:
        restore = ab.apply(comps)
        try:
            comps.model.attn.plan_decode(list(full_slots), bucket)
            ms = _time_step_body(comps, bucket, steps=steps, warmup=warmup)
        finally:
            restore()
        out["ablations"].append(
            {"name": ab.name, "ms": ms, "delta_ms": base - ms, "delta_pct": 100.0 * (base - ms) / base}
        )
    comps.model.attn.plan_decode(list(full_slots), bucket)
    return out


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def _print_report(res: Dict) -> None:
    b = res["batch"]
    print(f"\n{'=' * 78}\n== B={b}  ctx={res['ctx_len']}  ssm={res['ssm_state_dtype']}  "
          f"graphs={res['graphs']}\n{'=' * 78}")

    hd = res["host_device"]
    print(f"\n-- host/device split (ms per decode step) --")
    print(f"   decoder.step total : {hd['step_ms']:8.3f}   (repeat, measured last: "
          f"{hd.get('step_ms_repeat', float('nan')):.3f})")
    print(f"     plan_decode (host, outside graph) : {hd['plan_decode_ms']:8.3f}"
          f"  ({100 * hd['plan_decode_ms'] / hd['step_ms']:5.1f}%)")
    print(f"     replay      (host wall)           : {hd['replay_wall_ms']:8.3f}")
    print(f"     replay      (device, CUDA events) : {hd['replay_device_ms']:8.3f}")
    print(f"     unaccounted                       : {hd['unaccounted_ms']:8.3f}")

    kt = res.get("kernels")
    if kt:
        print(f"\n-- kernel groups (device us per step; {kt['total_launches_per_step']:.0f} launches, "
              f"{kt['n_distinct_kernels']} distinct kernels, {kt['total_device_us_per_step']:.1f} us total) --")
        print(f"   {'group':<30} {'us/step':>10} {'%':>7} {'launches':>10} {'kernels':>8}")
        for name, g in kt["groups"].items():
            print(f"   {name:<30} {g['us_per_step']:10.1f} {g['pct']:6.1f}% "
                  f"{g['calls_per_step']:10.1f} {g['kernels']:8d}")
        print(f"\n-- top {len(kt['top_kernels'])} kernels by total device time --")
        print(f"   {'us/step':>9} {'calls':>8}  {'group':<28} name")
        for r in kt["top_kernels"]:
            print(f"   {r['us_per_step']:9.1f} {r['calls_per_step']:8.1f}  {r['group']:<28} {r['name'][:70]}")

    ab = res.get("ablations")
    if ab:
        print(f"\n-- ablations (graph-replay device ms; deltas are NOT additive) --")
        print(f"   baseline: {ab['baseline_ms']:.3f} ms")
        for row in ab["ablations"]:
            print(f"   {row['name']:<18} {row['ms']:8.3f} ms   attributable: "
                  f"{row['delta_ms']:7.3f} ms ({row['delta_pct']:5.1f}%)")


def _print_backends(summary: Dict[str, Dict[str, int]]) -> None:
    print("\n-- resolved GEMM backend per ResolvedLinear (m_bucket -> backend: n_layers) --")
    for mb in sorted(summary, key=lambda x: int(x)):
        parts = ", ".join(f"{k}: {v}" for k, v in sorted(summary[mb].items(), key=lambda kv: -kv[1]))
        print(f"   M<={mb:<5} {parts}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--batch", type=int, nargs="+", default=[1, 32])
    p.add_argument("--ctx-len", type=int, default=2048)
    p.add_argument("--ssm-state-dtype", default="fp16", choices=["fp32", "fp16"])
    p.add_argument("--graphs", choices=["on", "off"], default="on")
    p.add_argument("--page-size", type=int, default=16)
    p.add_argument("--kv-cache-dtype", choices=["bf16", "fp8"], default="bf16")
    p.add_argument("--gemm-backend", default=None)
    p.add_argument("--norm-backend", choices=["torch", "triton"], default="torch")
    p.add_argument("--fused-ops-backend", choices=["torch", "triton"], default="torch",
                    help="SwiGLU / GDN-gate-epilogue implementation.")
    p.add_argument("--gemm-accuracy", choices=["fast", "strict"], default="fast")
    p.add_argument("--gemm-weight-cache", choices=["multi", "single", "none"], default="multi")
    p.add_argument("--gemm-priority", choices=["v9", "v8", "v7", "v4"], default="v8")
    add_preset_arg(p)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--top-n", type=int, default=30)
    p.add_argument("--no-profiler", action="store_true", help="skip the torch.profiler kernel table")
    p.add_argument("--no-ablations", action="store_true", help="skip the ablation sweep")
    p.add_argument("--legacy-plan", action="store_true",
                    help="reinstall the pre-fix per-page-D2H build_flashinfer_indices, so the "
                    "'before' row of a before/after comparison is measured rather than remembered. "
                    "Pair with --gemm-backend vllm_block_fp8_triton to reproduce the full "
                    "pre-fix step.")
    p.add_argument("--out", default=None)
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    # One canonical config, shared with bench_runtime,
    # bench_spec and serve. Explicit flags still win.
    args = apply_preset(args, parser, argv=argv)
    if not torch.cuda.is_available():
        print("profile_step needs a CUDA device", file=sys.stderr)
        return 2

    use_graphs = args.graphs == "on"
    buckets = tuple(sorted(set(args.batch)))
    restore_plan = install_legacy_plan() if args.legacy_plan else (lambda: None)
    comps = build(
        args.model,
        ssm_state_dtype=args.ssm_state_dtype,
        use_cuda_graphs=use_graphs,
        buckets=buckets,
        ctx_len=args.ctx_len,
        page_size=args.page_size,
        kv_cache_dtype=args.kv_cache_dtype,
        gemm_backend=args.gemm_backend,
        norm_backend=args.norm_backend,
        fused_ops_backend=args.fused_ops_backend,
        gemm_accuracy=args.gemm_accuracy,
        gemm_weight_cache=args.gemm_weight_cache,
        gemm_priority=args.gemm_priority,
        device=args.device,
        verbose=not args.quiet,
    )

    print(format_resolved_config(
        comps.model.rt, ctx_len=args.ctx_len, steps=args.steps,
        warmup=args.warmup, batches=list(args.batch),
    ), file=sys.stderr)
    backends = collect_gemm_backends(comps.model)
    summary = summarize_backends(backends)
    _print_backends(summary)
    backend_warnings = check_resolved_backends(comps.model, backends, verbose=not args.quiet)

    out: Dict = {
        "resolved_config": resolved_config(
            comps.model.rt, ctx_len=args.ctx_len, steps=args.steps,
            warmup=args.warmup, batches=list(args.batch),
        ),
        "backend_warnings": backend_warnings,
        "ctx_len": args.ctx_len,
        "ssm_state_dtype": args.ssm_state_dtype,
        "graphs": use_graphs,
        "kv_cache_dtype": args.kv_cache_dtype,
        "gemm_backend": args.gemm_backend,
        "gemm_accuracy": args.gemm_accuracy,
        "gemm_weight_cache": args.gemm_weight_cache,
        "gemm_priority": args.gemm_priority,
        "norm_backend": args.norm_backend,
        "fused_ops_backend": args.fused_ops_backend,
        "legacy_plan": bool(args.legacy_plan),
        "buckets": list(buckets),
        "gemm_backend_summary": summary,
        "gemm_backend_per_layer": backends,
        "points": [],
    }

    for b in args.batch:
        full_slots = prepare_batch(comps, b, args.ctx_len)
        res: Dict = {
            "batch": b,
            "ctx_len": args.ctx_len,
            "ssm_state_dtype": args.ssm_state_dtype,
            "graphs": use_graphs,
            "bucket": comps.decoder.bucket_for(b),
        }
        res["host_device"] = host_device_split(
            comps, b, full_slots, steps=args.steps, warmup=args.warmup
        )
        res["tok_s"] = b / (res["host_device"]["step_ms"] / 1e3)
        res["ceiling_tok_s"] = decode_ceiling_tok_s(b, args.ctx_len, args.ssm_state_dtype)
        res["efficiency"] = res["tok_s"] / res["ceiling_tok_s"]
        if not args.no_profiler:
            res["kernels"] = kernel_table(comps, b, full_slots, steps=max(5, args.steps // 3),
                                          top_n=args.top_n)
        if not args.no_ablations:
            res["ablations"] = ablation_table(
                comps, b, full_slots, steps=args.steps, warmup=args.warmup
            )
        _print_report(res)
        out["points"].append(res)
        if args.out:
            with open(args.out + ".tmp", "w") as f:
                json.dump(out, f, indent=2)
            import os

            os.replace(args.out + ".tmp", args.out)

    restore_plan()
    if args.out:
        print(f"\n[profile_step] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
