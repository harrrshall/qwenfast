#!/usr/bin/env python
"""Where does a *served* step's time actually go?

``profile_step.py`` answers that for one decode step in isolation.  This
answers it for the engine as the sweep actually drives it, because those two
numbers can be 4.3x apart:

    conc 256, 2.1k-token prompts, 500 output
      decode-only, graph-timed, B=256 ...............  45.8 ms/step
      served, (e2e - TTFT)/500 ...................... ~195   ms/step

Nothing in ``profile_step`` can see that gap: it never runs a prefill chunk,
never runs the scheduler, and never shares an interpreter with uvicorn.  So
this module drives the **real** :class:`~.scheduler.Scheduler` against the
**real** model with ``bench_serve.py``'s arrival pattern (a closed loop of
``concurrency`` in-flight requests, each replaced the instant it finishes --
``benchmarks/bench_serve.py::run_level``'s ``asyncio.Semaphore``), and
attributes wall time to six places:

1. **The step ledger.**  Every ``Scheduler.step()`` call is timed on the host
   (``perf_counter``) and on the device (CUDA events) and labelled
   prefill-chunk or decode-step, with its token count, live-sequence count and
   graph bucket.  This alone answers "how much of the 195 ms is prefill?"
   without a single hypothesis, and it is the first view for that reason.

2. **Prefill-chunk components.**  ``torch.profiler`` over one chunk, folded
   into the same groups ``profile_step`` uses (GEMM-by-backend, GDN, conv,
   attention, norm/elementwise, copy), plus ablation deltas for the four
   candidate hot spots (conv prefill, GDN chunk kernel, attention prefill,
   MLP) measured by re-running the chunk with each stubbed out.

3. **Host overhead.**  Named counters around the six host-side things a step
   does outside a kernel: ``make_prefill_batch``, ``plan_prefill``, the decode
   step's device-buffer fill, the ``out -> host`` token harvest, the scheduler
   bookkeeping, and event dispatch.  A sync storm here is invisible to any
   device profiler: it shows up only as the GPU sitting idle.

4. **GIL contention.**  The same drive re-run with a background thread doing
   detokenizer-shaped Python work at the served token rate, standing in for
   uvicorn's event loop.  This is the one serving-specific overhead with no
   offline analogue, so it has to be measured here.

5. **Prefill A/Bs**: prefill GEMM backend at the real chunk M
   (``m_bucket`` clamps at 512, so M=8192 routes on a table measured at M=512
   and never validated above it), and the conv-prefill layout
   (the Triton token-major kernel against the older transpose + fp32
   ``F.conv1d`` fallback).

6. **The physics ceiling.**  Prefill is compute-bound, so its ceiling is
   FLOPs, not bytes: this counts the model's actual linear-layer FLOPs per
   token from the loaded config rather than quoting "2 x 27B", and divides by
   the assumed dense FP8 peak.

Runs on a GPU host only (needs the real checkpoint); imports cleanly on a
CPU-only machine.  Example::

    python -m qwenfast.runtime.profile_serving \\
        --model $FP8 --concurrency 32 256 --duration 60 \\
        --input-len 2139 --output-len 500 \\
        --ssm-state-dtype fp16 --norm-backend triton \\
        --out /home/qwenfast-results/profile_serving.json

Cost note: a 60 s drive at two concurrencies plus the component views is
~6-8 min of GPU time.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from . import prefill_attrib
from .bench_runtime import (
    _release_all_slots,
    calibrate_kv_fp8,
    collect_gemm_backends,
    derive_pool_sizes,
    summarize_backends,
)
from .engine import EngineComponents, build_engine
from .fused_model import (
    RuntimeConfig,
    default_claim_gemm_cache_m,
    make_mixed_batch,
    make_prefill_batch,
    prefill_gemm_scope,
)
from .preset import add_preset_arg, apply_preset, format_resolved_config
from .profile_step import classify_kernel
from .scheduler import GenParams, Request, Scheduler

# --------------------------------------------------------------------------- #
# 0. physics: the prefill FLOPs ceiling
# --------------------------------------------------------------------------- #
#: Dense (non-sparse) FP8 tensor-core peak assumed for the H200, TFLOP/s.
#: The commonly quoted figure is "~990 TFLOPS dense fp8", so
#: that is what this defaults to and prints, with the caveat attached: NVIDIA's
#: H200 SXM sheet lists 3,958 TFLOPS FP8 *with sparsity*, and which half of
#: that is "dense" is exactly the kind of thing that should be measured rather
#: than cited -- `--peak-tflops` overrides it, and `gemm_backend_sweep` below
#: reports the *achieved* rate so the ratio can be read off a measurement.
DEFAULT_PEAK_TFLOPS = 990.0


def linear_flops_per_token(cfg) -> Dict[str, float]:
    """FLOPs per prefill token, by group, from the loaded config.

    2 x params for every linear the token passes through.  Counted from the
    config rather than from "2 x 27B" because the two differ by the parts a
    prefill token does *not* pay: the embedding table is a gather, and
    ``lm_head`` runs on one row per *sequence* (``prefill_forward`` takes
    ``index_select(last_indices)`` precisely so the ``[8192, 248320]`` fp32
    logits are never materialised), so 2.5 of the checkpoint's 27B are not in
    the per-token number at all.
    """
    h = cfg.hidden_size
    n_layers = cfg.num_hidden_layers
    n_attn = len(cfg.attention_layer_indices) or (n_layers // cfg.full_attention_interval)
    n_gdn = n_layers - n_attn

    mlp = (2 * cfg.intermediate_size * h) + (cfg.intermediate_size * h)
    gdn = (
        (cfg.conv_dim + cfg.value_dim) * h  # in_proj_qkvz
        + (2 * cfg.linear_num_value_heads) * h  # in_proj_ba
        + cfg.value_dim * h  # out_proj
    )
    q_out = cfg.num_attention_heads * cfg.head_dim * (2 if cfg.attn_output_gate else 1)
    kv_out = 2 * cfg.num_key_value_heads * cfg.head_dim
    attn = (q_out + kv_out) * h + (cfg.num_attention_heads * cfg.head_dim) * h

    per_group = {
        "mlp": 2.0 * mlp * n_layers,
        "gdn_proj": 2.0 * gdn * n_gdn,
        "attn_proj": 2.0 * attn * n_attn,
    }
    per_group["total_linear"] = sum(per_group.values())
    per_group["lm_head_per_sequence"] = 2.0 * cfg.vocab_size * h
    return per_group


def prefill_ceiling(cfg, *, peak_tflops: float = DEFAULT_PEAK_TFLOPS) -> Dict[str, float]:
    """Compute-bound prefill ceiling in tok/s, plus the mixer terms it ignores.

    The quadratic/state terms are reported separately rather than folded in:
    at this model's shape they are ~2% of the linear FLOPs, which is worth
    knowing *and* worth not pretending to have modelled precisely.
    """
    f = linear_flops_per_token(cfg)
    n_attn = len(cfg.attention_layer_indices) or (
        cfg.num_hidden_layers // cfg.full_attention_interval
    )
    n_gdn = cfg.num_hidden_layers - n_attn
    # GDN chunked delta rule: per token per layer the state [HV, K, V] is
    # touched by ~3 matvec-equivalents (k^T S, outer-product update, S^T q).
    gdn_state = 2.0 * 3.0 * cfg.linear_num_value_heads * cfg.linear_key_head_dim * (
        cfg.linear_value_head_dim
    ) * n_gdn
    return {
        "flops_per_token_linear": f["total_linear"],
        "flops_per_token_gdn_state": gdn_state,
        "flops_per_token_total": f["total_linear"] + gdn_state,
        "peak_tflops": peak_tflops,
        "ceiling_tok_s": peak_tflops * 1e12 / (f["total_linear"] + gdn_state),
        "by_group_gflops": {k: v / 1e9 for k, v in f.items()},
    }


def attn_flops_per_token(cfg, ctx_len: int) -> float:
    """Causal self-attention FLOPs per prefill token at a given prompt length."""
    n_attn = len(cfg.attention_layer_indices) or (
        cfg.num_hidden_layers // cfg.full_attention_interval
    )
    return 2.0 * 2.0 * cfg.num_attention_heads * cfg.head_dim * (ctx_len / 2.0) * n_attn


# --------------------------------------------------------------------------- #
# 1. the step ledger
# --------------------------------------------------------------------------- #
#: Step kinds that carry a prefill chunk *and* decode rows in one
#: ``Scheduler.step()``: the fused ``mixed`` forward and the
#: two-stream ``overlap`` pair. Named once so the ledger's per-kind
#: bookkeeping cannot learn about one and not the other.
_BOTH_HALVES = ("mixed", "overlap")
#: Every kind that emits output tokens.
_EMITTING = ("decode", "spec", "mixed", "overlap")


@dataclass
class StepRecord:
    """One ``Scheduler.step()`` call."""

    kind: str  # "prefill" | "decode" | "spec" | "idle"
    wall_ms: float
    device_ms: float
    tokens: int  # prefill: chunk tokens; decode/mixed: tokens emitted
    n_seqs: int  # prefill: sequences in the chunk; decode: live rows
    bucket: int  # decode only; 0 for prefill
    running: int
    waiting: int
    #: A mixed step does both jobs, so `tokens` cannot mean
    #: both: it holds the *output* tokens (which is what `attribute` divides
    #: by) and this holds the prefill chunk it carried. 0 on every other kind.
    prefill_tokens: int = 0
    decode_rows: int = 0
    #: True when this mixed step was replayed from the
    #: captured graph rather than launched eagerly. A ledger where this is not
    #: 100% of the mixed steps is measuring a mixture of the two arms -- the
    #: padding does not fit every step (more decode rows than the widest
    #: bucket, say), and then the step silently runs eager.
    graphed: bool = False


@dataclass
class HostCounters:
    """Named host-side costs, summed over a drive (ms) and counted.

    Not derived from ``wall - device``: on the eager prefill path the host runs
    *ahead* of the device, so that difference is noise, while these are the
    actual Python the engine loop executes.
    """

    totals: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def add(self, name: str, ms: float) -> None:
        self.totals[name] += ms
        self.counts[name] += 1

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        return {
            name: {
                "total_ms": round(self.totals[name], 3),
                "calls": self.counts[name],
                "ms_per_call": round(self.totals[name] / max(self.counts[name], 1), 4),
            }
            for name in sorted(self.totals, key=lambda k: -self.totals[k])
        }


class _Timed:
    """Wrap a bound callable so every call adds to a :class:`HostCounters`."""

    __slots__ = ("fn", "counters", "name")

    def __init__(self, fn, counters: HostCounters, name: str):
        self.fn, self.counters, self.name = fn, counters, name

    def __call__(self, *a, **kw):
        t0 = time.perf_counter()
        try:
            return self.fn(*a, **kw)
        finally:
            self.counters.add(self.name, (time.perf_counter() - t0) * 1e3)


class _TimedHarvest(_Timed):
    """``_Timed`` for ``Scheduler._harvest``, split into its two halves.

    A single ``decode harvest (D2H)`` counter is easy to misread as a
    multi-millisecond stall on a 1 KiB pinned copy. It is not: the
    harvest holds the step's **only** ``synchronize()``, so it drains the whole
    graph replay that ``decoder.step`` queued a fraction of a millisecond
    earlier. ``Scheduler._harvest`` now times that drain itself; this wrapper
    reports it as its own counter so the two are never added together.
    """

    __slots__ = ("sched",)

    def __init__(self, fn, counters: HostCounters, name: str, sched):
        super().__init__(fn, counters, name)
        self.sched = sched

    def __call__(self, *a, **kw):
        w0 = getattr(self.sched, "harvest_gpu_wait_s", 0.0)
        try:
            return super().__call__(*a, **kw)
        finally:
            dw = getattr(self.sched, "harvest_gpu_wait_s", 0.0) - w0
            if dw > 0:
                self.counters.add("  of which GPU wait (not the copy)", dw * 1e3)


def install_host_counters(sched: Scheduler, model) -> Tuple[HostCounters, Callable[[], None]]:
    """Instrument the host-side seams of one step; returns ``(counters, restore)``.

    Deliberately *instance*-level (and one class-level patch for
    ``make_prefill_batch``, which is a module function the scheduler calls
    directly), so nothing here leaks into a build that is not being profiled.
    """
    from . import scheduler as sched_mod

    counters = HostCounters()
    saved: List[Tuple[object, str, object]] = []

    def patch(obj, attr, name):
        orig = getattr(obj, attr)
        saved.append((obj, attr, orig))
        setattr(obj, attr, _Timed(orig, counters, name))

    patch(sched_mod, "make_prefill_batch", "make_prefill_batch")
    patch(model.attn, "plan_prefill", "plan_prefill")
    patch(model.attn, "plan_decode", "plan_decode")
    patch(model, "prefill_forward", "prefill_forward(host launch)")
    # `prefill_forward(host launch)` can show a 1,024-token *prefill* chunk
    # costing ~126 ms of host time against ~69 ms of device work. The mixed
    # step needs the equivalent line so its own launch cost is a direct
    # reading rather than inferred from a slope, and the difference between
    # the two is exactly what the graph is supposed to remove.
    patch(model, "mixed_forward", "mixed_forward(host launch)")
    if getattr(sched, "mixed_runner", None) is not None:
        patch(sched.mixed_runner, "step", "mixed_runner.step(plan+replay+holes)")
        # `_run_mixed_step` and `_run_overlap_step` both call the split
        # pair directly, so `step` alone would read 0 on the graphed mixed
        # path. Nested inside it wherever `step` *is* used: a breakdown, not
        # a third disjoint cost.
        patch(sched.mixed_runner, "prepare_step", "  mixed_runner.prepare_step(load+plan)")
        patch(sched.mixed_runner, "replay", "  mixed_runner.replay(launch+holes)")
    _h_orig = sched._harvest
    saved.append((sched, "_harvest", _h_orig))
    sched._harvest = _TimedHarvest(
        _h_orig, counters, "decode harvest (D2H+GPU wait)", sched
    )
    patch(sched.decoder, "step", "decoder.step(plan+replay)")
    # `_run_decode_step` calls `prepare_step` and `replay`
    # directly (so the step profile can time the FlashInfer plan apart
    # from the launch), and only the eager overlap branches still go through
    # `step`. Both are counted, and the two lines below are *nested* inside
    # `decoder.step(plan+replay)` on the paths that do use it; read them as
    # a breakdown of it, not as three disjoint costs.
    patch(sched.decoder, "prepare_step", "  decoder.prepare_step(plan_decode)")
    patch(sched.decoder, "replay", "  decoder.replay(launch)")

    def restore() -> None:
        for obj, attr, orig in saved:
            setattr(obj, attr, orig)

    return counters, restore


class ClosedLoopDriver:
    """``bench_serve.py``'s arrival pattern, without HTTP or asyncio.

    ``bench_serve`` gates every request behind ``asyncio.Semaphore(conc)``, so
    the server sees exactly ``conc`` requests in flight at all times and a
    finished one is replaced immediately.  Reproducing that *shape* is the
    whole point -- the engine's behaviour at conc 256 is dominated by the
    resulting steady state (a continuous supply of fresh 2.1k-token prompts to
    prefill against a full running set), and a profiler that prefilled once and
    then decoded would measure a regime the sweep never enters.

    What is deliberately *not* reproduced: HTTP, tokenization and the asyncio
    handoff.  Those are measured separately (``gil_probe``) rather than mixed
    in, so the ledger below is the engine's own time and nothing else.
    """

    def __init__(
        self,
        sched: Scheduler,
        *,
        concurrency: int,
        input_len: int,
        output_len: int,
        vocab_size: int,
        seed: int = 0,
    ):
        self.sched = sched
        self.concurrency = concurrency
        self.input_len = input_len
        self.output_len = output_len
        self.vocab = vocab_size
        self._g = torch.Generator(device="cpu").manual_seed(seed)
        self._n = 0
        self.records: List[StepRecord] = []
        self.completed = 0
        self.prompt_tokens = 0
        self.output_tokens = 0

    def _new_request(self) -> Request:
        self._n += 1
        ids = torch.randint(
            0, self.vocab, (self.input_len,), generator=self._g, dtype=torch.int64
        ).tolist()
        return Request(
            request_id=f"r{self._n}",
            prompt_token_ids=ids,
            params=GenParams(
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                max_tokens=self.output_len,
                # `bench_serve.build_payload` sends `ignore_eos: True`, so the
                # sweep's requests always run the full 500 tokens. Anything
                # else changes the running-set dynamics this exists to measure.
                ignore_eos=True,
            ),
        )

    def _top_up(self) -> None:
        live = len(self.sched.running) + len(self.sched.waiting)
        for _ in range(max(self.concurrency - live, 0)):
            req = self._new_request()
            self.prompt_tokens += len(req.prompt_token_ids)
            self.sched.add_request(req)

    def _classify(self, events, before_running: int) -> Tuple[str, int, int, int]:
        """``(kind, tokens, n_seqs, bucket)`` for the step just run."""
        sched = self.sched
        toks = getattr(sched, "last_chunk_tokens", 0)
        if getattr(sched, "last_step_mixed", False):
            # Mixed step: one forward, both jobs. Labelled `mixed` and *not* folded
            # into `prefill`+`decode`, because the whole claim being measured
            # is that its ms/step is far less than the two apart.
            # An *overlapped* step does both jobs too, on two streams
            # and two graphs, and it is labelled apart so a ledger cannot
            # silently compare the two arms as one kind.
            kind = "overlap" if getattr(sched, "last_step_overlapped", False) else "mixed"
            return kind, sum(len(e.new_token_ids) for e in events), \
                getattr(sched, "last_chunk_seqs", 0), 0
        if toks:
            # `Scheduler.step` clears this at its top and sets it only when a
            # chunk actually ran, so it is a per-step signal -- unlike
            # `_prefill_progressed`, which also fires for a swap-restore that
            # touches the model not at all.
            return "prefill", toks, getattr(sched, "last_chunk_seqs", 0), 0
        n = sum(len(e.new_token_ids) for e in events)
        if n == 0:
            return "idle", 0, before_running, 0
        rows = sum(1 for e in events if e.new_token_ids)
        bucket = sched.decoder.bucket_for(max(before_running, 1))
        kind = "spec" if n > rows else "decode"
        return kind, n, rows, bucket

    def run(self, duration_s: float, *, warmup_s: float = 5.0,
            max_steps: Optional[int] = None) -> None:
        """Drive for ``duration_s`` after ``warmup_s``, or ``max_steps`` steps.

        ``max_steps`` exists for the CPU tests: a wall-clock bound makes a test
        that asserts "a prefill and a decode step both happened" depend on how
        loaded the machine is, and one that runs green on a laptop and red on
        the GPU host is worse than no test.
        """
        dev = self.sched.device
        cuda = torch.device(dev).type == "cuda"
        ev0 = torch.cuda.Event(enable_timing=True) if cuda else None
        ev1 = torch.cuda.Event(enable_timing=True) if cuda else None

        t_start = time.perf_counter()
        deadline = t_start + warmup_s + duration_s
        measure_from = t_start + warmup_s
        steps_run = 0
        while time.perf_counter() < deadline or (
            max_steps is not None and steps_run < max_steps
        ):
            if max_steps is not None and steps_run >= max_steps:
                break
            steps_run += 1
            self._top_up()
            before = len(self.sched.running)
            if cuda:
                ev0.record()
            t0 = time.perf_counter()
            events = self.sched.step()
            wall = (time.perf_counter() - t0) * 1e3
            if cuda:
                ev1.record()
                ev1.synchronize()
                device_ms = ev0.elapsed_time(ev1)
            else:
                device_ms = wall
            kind, toks, seqs, bucket = self._classify(events, before)
            if time.perf_counter() >= measure_from:
                st = self.sched.stats()
                self.records.append(
                    StepRecord(
                        kind=kind,
                        wall_ms=wall,
                        device_ms=device_ms,
                        tokens=toks,
                        n_seqs=seqs,
                        bucket=bucket,
                        running=st.num_running,
                        waiting=st.num_waiting,
                        prefill_tokens=(
                            getattr(self.sched, "last_chunk_tokens", 0)
                            if kind in _BOTH_HALVES else 0
                        ),
                        decode_rows=(
                            getattr(self.sched, "last_mixed_decode_rows", 0)
                            if kind in _BOTH_HALVES else 0
                        ),
                        graphed=bool(
                            kind in _BOTH_HALVES
                            and getattr(self.sched, "last_step_mixed_graphed", False)
                        ),
                    )
                )
                if kind in _EMITTING:
                    self.output_tokens += toks
            self.completed += sum(1 for e in events if e.finished)


def _pct(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    i = min(int(round((p / 100.0) * (len(v) - 1))), len(v) - 1)
    return v[i]


def attribute(records: Sequence[StepRecord], *, duration_s: float) -> Dict:
    """The headline table: where the served step's milliseconds went.

    ``ms_per_output_token`` is the number to compare against a decode-only
    step: it is each kind's total wall time divided by the *output* tokens the
    drive produced, i.e. how much of a client's observed TPOT that kind is
    responsible for.  A prefill chunk emits at most one token per sequence, so
    almost all of its time lands on this line -- which is the point.
    """
    by_kind: Dict[str, List[StepRecord]] = defaultdict(list)
    for r in records:
        by_kind[r.kind].append(r)
    out_tokens = sum(r.tokens for r in records if r.kind in _EMITTING) or 1
    total_wall = sum(r.wall_ms for r in records) or 1.0

    kinds = {}
    for kind, rs in sorted(by_kind.items()):
        wall = sum(r.wall_ms for r in rs)
        toks = sum(r.tokens for r in rs)
        kinds[kind] = {
            "steps": len(rs),
            "wall_ms_total": round(wall, 1),
            "wall_pct": round(100.0 * wall / total_wall, 1),
            "ms_per_step_mean": round(wall / len(rs), 3),
            "ms_per_step_p50": round(_pct([r.wall_ms for r in rs], 50), 3),
            "ms_per_step_p99": round(_pct([r.wall_ms for r in rs], 99), 3),
            "device_ms_per_step_mean": round(
                sum(r.device_ms for r in rs) / len(rs), 3
            ),
            "tokens": toks,
            "tok_s": round(toks / (wall / 1e3), 1) if wall else 0.0,
            "seqs_per_step_mean": round(sum(r.n_seqs for r in rs) / len(rs), 1),
            "ms_per_output_token": round(wall / out_tokens, 3),
        }
        if kind in _BOTH_HALVES:  # mixed/overlap: what the step actually carried
            pre = sum(r.prefill_tokens for r in rs)
            kinds[kind]["prefill_tokens"] = pre
            kinds[kind]["prefill_tokens_per_step_mean"] = round(pre / len(rs), 1)
            kinds[kind]["decode_rows_per_step_mean"] = round(
                sum(r.decode_rows for r in rs) / len(rs), 1
            )
            kinds[kind]["prefill_tok_s"] = round(pre / (wall / 1e3), 1) if wall else 0.0
            n_graphed = sum(1 for r in rs if r.graphed)
            kinds[kind]["graphed_steps"] = n_graphed
            kinds[kind]["graphed_pct"] = round(100.0 * n_graphed / len(rs), 1)
    return {
        "steps": len(records),
        "measured_s": round(total_wall / 1e3, 2),
        "output_tokens": out_tokens,
        "output_tok_s": round(out_tokens / (total_wall / 1e3), 1),
        # `output_tok_s` above divides by the **sum of step wall times**, so
        # it is arithmetically incapable of seeing a gap *between* steps, and
        # a GIL probe judged by it alone cannot produce a meaningful number.
        # These three divide by the real clock instead:
        # `step_busy_pct` is the loop's own view of the utilisation
        # `bench_serve --nvidia-smi` samples from outside the process, and
        # `wall_output_tok_s` is what a client would actually observe.
        "wall_s": round(duration_s, 2),
        "wall_output_tok_s": round(out_tokens / duration_s, 1) if duration_s else 0.0,
        "step_busy_pct": (
            round(100.0 * (total_wall / 1e3) / duration_s, 1) if duration_s else 0.0
        ),
        "served_ms_per_output_token": round(total_wall / out_tokens, 3),
        "running_p50": _pct([float(r.running) for r in records], 50),
        "by_kind": kinds,
    }


# --------------------------------------------------------------------------- #
# 2. prefill-chunk components
# --------------------------------------------------------------------------- #
def chunk_shape(budget: int, prompt_len: int) -> List[int]:
    """The per-sequence token counts ``Scheduler._run_prefill_step`` would pack.

    Not ``[budget]`` and not ``[prompt_len] * k``: the real loop takes whole
    prompts until one does not fit and then gives the remainder of the budget
    to the head of the next one, so at budget 8192 / prompt 2139 a chunk is
    ``[2139, 2139, 2139, 1775]`` -- four varlen segments, not three and not
    one. The segment count is what the conv and GDN kernels' launch grids key
    on, so a profile that got it wrong would measure a different kernel.
    """
    lens: List[int] = []
    left = budget
    while left > 0:
        take = min(prompt_len, left)
        lens.append(take)
        left -= take
    return lens


def synthetic_chunk(model, seq_lens: Sequence[int], *, start_pos: int = 0):
    """One ``PrefillBatch`` of the shape :func:`chunk_shape` describes."""
    slots = list(range(len(seq_lens)))
    g = torch.Generator(device="cpu").manual_seed(1234)
    ids = []
    for s, n in zip(slots, seq_lens):
        model.reset_slot(s)
        # `+ 1` so a later `reset_chunk_slots` (which sizes from `kv_lens`)
        # asks for the same capacity this did, and the page count is stable
        # across repeats instead of drifting by one page every few chunks.
        model.kv_pool.ensure_capacity(s, start_pos + int(n) + 1)
        ids.append(
            torch.randint(
                0, model.config.vocab_size, (int(n),), generator=g, dtype=torch.int64
            ).tolist()
        )
    return make_prefill_batch(ids, [start_pos] * len(slots), slots, model.device)


def reset_chunk_slots(model, batch) -> None:
    """Put ``batch``'s slots back to "this chunk has not run yet".

    Required between repeats and it is not a detail: ``prefill_forward``
    *appends* KV and advances each slot's device ``seq_len``, and
    ``reset_slot`` returns the slot's pages to the allocator. Re-running the
    same batch without this both (a) grows ``seq_len`` past the ``kv_lens``
    the plan was built from, so every repeat after the first measures a
    longer attention than the one being reported, and (b) walks the page pool
    to exhaustion in a handful of repeats.
    """
    for slot, kv_len in zip(batch.seq_slots, batch.kv_lens):
        model.reset_slot(int(slot))
        model.kv_pool.ensure_capacity(int(slot), int(kv_len) + 1)


def time_chunk(model, batch, *, steps: int = 5, warmup: int = 2,
               gemm_backend: Optional[str] = None) -> float:
    """Mean ms for one ``prefill_forward`` over ``batch``.

    Synchronised per repeat rather than once around the whole loop, because
    each repeat has to be preceded by :func:`reset_chunk_slots`. At ~1 s per
    chunk the extra sync is <0.01% and it buys every repeat being the *same*
    measurement rather than a progressively longer one.
    """
    cuda = model.device.type == "cuda"
    total = 0.0
    for i in range(warmup + steps):
        reset_chunk_slots(model, batch)
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with prefill_gemm_scope(gemm_backend):
            model.prefill_forward(batch, all_logits=False)
        if cuda:
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1e3
        if i >= warmup:
            total += dt
    return total / max(steps, 1)


def chunk_kernel_table(model, batch, *, top_n: int = 30) -> Dict:
    """``torch.profiler`` over one prefill chunk, folded into op groups."""
    from torch.profiler import ProfilerActivity, profile

    reset_chunk_slots(model, batch)
    model.prefill_forward(batch, all_logits=False)  # warm
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    acts = [ProfilerActivity.CPU]
    if model.device.type == "cuda":
        acts.append(ProfilerActivity.CUDA)
    reset_chunk_slots(model, batch)
    with profile(activities=acts, record_shapes=False) as prof:
        model.prefill_forward(batch, all_logits=False)
        if model.device.type == "cuda":
            torch.cuda.synchronize()

    rows = []
    groups: "OrderedDict[str, Dict[str, float]]" = OrderedDict()
    for evt in prof.key_averages():
        us = float(getattr(evt, "self_device_time_total", 0.0) or 0.0)
        if us <= 0:
            continue
        rows.append({"name": evt.key, "us": round(us, 1), "calls": int(evt.count)})
        g = classify_kernel(evt.key)
        slot = groups.setdefault(g, {"us": 0.0, "launches": 0})
        slot["us"] += us
        slot["launches"] += int(evt.count)
    rows.sort(key=lambda r: -r["us"])
    total = sum(v["us"] for v in groups.values()) or 1.0
    return {
        "top_kernels": rows[:top_n],
        "groups": {
            k: {
                "us": round(v["us"], 1),
                "pct": round(100.0 * v["us"] / total, 1),
                "launches": v["launches"],
            }
            for k, v in sorted(groups.items(), key=lambda kv: -kv[1]["us"])
        },
        "device_us_total": round(total, 1),
    }


class _Identity:
    """Stub that returns its (first) input unchanged, shape-preserving."""

    def __init__(self, wrapped):
        self.wrapped = wrapped

    def __call__(self, x, *a, **kw):
        return x


def _stub_conv_prefill(model) -> Callable[[], None]:
    """Replace every GDN layer's prefill conv with a pass-through."""
    import qwenfast.kernels_gdn as gdn_api

    saved = (gdn_api.causal_conv_prefill, gdn_api.causal_conv_prefill_varlen)

    def noop_bct(x, w, **kw):
        return x

    def noop_tc(x, w, **kw):
        return x

    gdn_api.causal_conv_prefill = noop_bct
    gdn_api.causal_conv_prefill_varlen = noop_tc

    def restore() -> None:
        gdn_api.causal_conv_prefill, gdn_api.causal_conv_prefill_varlen = saved

    return restore


def _stub_gdn_core(model) -> Callable[[], None]:
    """Replace the chunked delta rule with a zero of the right shape."""
    import qwenfast.kernels_gdn as gdn_api

    saved = gdn_api.gdn_prefill_chunked

    def zeros(q, k, v, g, beta, **kw):
        return torch.zeros_like(v), None

    gdn_api.gdn_prefill_chunked = zeros

    def restore() -> None:
        gdn_api.gdn_prefill_chunked = saved

    return restore


def _stub_attn_prefill(model) -> Callable[[], None]:
    saved = model.attn.prefill

    def zeros(kv_layer, q, *a, **kw):
        return torch.zeros_like(q)

    model.attn.prefill = zeros

    def restore() -> None:
        model.attn.prefill = saved

    return restore


def _stub_mlp(model) -> Callable[[], None]:
    saved = [(layer, layer.mlp) for layer in model.layers]
    for layer, mlp in saved:
        layer.mlp = _Identity(mlp)

    def restore() -> None:
        for layer, mlp in saved:
            layer.mlp = mlp

    return restore


# --------------------------------------------------------------------------- #
# 2a. headroom, and never losing a section to a late OOM
# --------------------------------------------------------------------------- #
def free_gib(device=None) -> Optional[float]:
    """Device-free GiB from ``cudaMemGetInfo``, or ``None`` off CUDA."""
    if not torch.cuda.is_available():
        return None
    try:
        free, _total = torch.cuda.mem_get_info(device)
        return free / (1024.0 ** 3)
    except Exception:  # pragma: no cover
        return None


def run_probe(name: str, fn: Callable[[], Dict], *, need_gib: float = 0.0,
              device=None, verbose: bool = True) -> Dict:
    """Run one component view; never let it take the whole run down.

    A late probe that runs out of memory must not discard the sections that
    already measured. Three hazards this guards against, together with the
    callers:

    1. A probe that forces a cache-owning GEMM backend through
       `prefill_gemm_scope` bypasses the weight-cache policy (see
       `_prefill_backends_for` below) and can leave ~20 GiB of extra repack
       cache resident for every probe that runs after it.
    2. If the chunk dict is built as one literal, the *first* probe to raise
       discards every table already computed and the run produces no JSON.
    3. A probe that does not check whether the memory it is about to ask for
       exists fails late instead of skipping early.

    So: check the headroom first, catch everything, empty the cache
    afterwards (an OOM leaves the allocator fragmented and the *next* probe
    pays for it), and return the failure as data. A skipped probe is a row
    that says why; a crashed probe loses the whole run.
    """
    have = free_gib(device)
    if need_gib and have is not None and have < need_gib:
        if verbose:
            print(f"    [probe] SKIP {name}: needs {need_gib:.1f} GiB free, "
                  f"has {have:.1f}", flush=True)
        return {"skipped": f"needs {need_gib:.1f} GiB free, has {have:.1f} GiB"}
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 -- a probe that cannot run is data
        if verbose:
            print(f"    [probe] FAIL {name}: {type(exc).__name__}: {exc}"[:300], flush=True)
        return {"error": f"{type(exc).__name__}: {exc}"[:400],
                "free_gib_at_failure": have}
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


#: Ablation deltas are **not additive** -- removing work also removes the
#: launch-latency tail it was hiding -- so they are reported as "cost
#: attributable to", exactly as ``profile_step`` does.
_ABLATIONS: "OrderedDict[str, Callable]" = OrderedDict(
    [
        ("conv_prefill", _stub_conv_prefill),
        ("gdn_chunk_kernel", _stub_gdn_core),
        ("attention_prefill", _stub_attn_prefill),
        ("mlp", _stub_mlp),
    ]
)


def chunk_ablations(model, batch, base_ms: float, *, steps: int = 5) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for name, apply in _ABLATIONS.items():
        restore = apply(model)
        try:
            ms = time_chunk(model, batch, steps=steps)
        finally:
            restore()
        out[name] = {
            "chunk_ms_without": round(ms, 2),
            "delta_ms": round(base_ms - ms, 2),
            "pct_of_chunk": round(100.0 * (base_ms - ms) / base_ms, 1) if base_ms else 0.0,
        }
    return out


# --------------------------------------------------------------------------- #
# 3. A/Bs
# --------------------------------------------------------------------------- #
#: The cache-free FP8 backends, i.e. the ones a server running
#: ``--gemm-weight-cache single`` may actually use for prefill without
#: allocating a second 23 GiB repacked weight copy.
#: ``scaled_mm_pertensor`` is included because it is the *measured* winner at
#: M=512 and the sweep should price what it would buy, but it can
#: only be run under ``--gemm-weight-cache multi``.
CANDIDATE_PREFILL_BACKENDS = (
    None,  # whatever m_bucket dispatch picks -- the "before"
    "flashinfer_fp8_blockscale",
    "vllm_block_fp8_cutlass",
    "scaled_mm_pertensor",
    "deepgemm",
)


def _prefill_backends_for(model, candidates=CANDIDATE_PREFILL_BACKENDS) -> List[Tuple]:
    """``[(name, allowed, why_not)]`` under the *running* weight-cache policy.

    ``CANDIDATE_PREFILL_BACKENDS`` lists ``scaled_mm_pertensor`` and
    ``deepgemm``, and the sweep runs them through
    :func:`~.fused_model.prefill_gemm_scope`, whose own docstring warns:

        *Forcing a backend that owns a repack cache when a different one is
        already cached will allocate that second copy on the first prefill
        chunk.*

    Under the serving default ``--gemm-weight-cache single``, marlin already
    holds the one slot, so running such a backend would materialise a
    **second ~23 GiB repacked copy of every fp8 linear**, and, because the
    sweep catches per-backend exceptions, a *partially* built cache would stay
    resident for everything that runs after it (on a 140 GiB card with a
    ~118 GiB steady plan, that is an OOM).

    So the policy decides: a cache-owning backend whose slot is not already its own
    is reported as **skipped, with the reason**, which is a more useful row
    than a number bought with 23 GiB the profile then does not have.
    """
    from ..gemm import dispatch as gemm_dispatch

    weight = None
    for layer in getattr(model, "layers", []):
        mlp = getattr(layer, "mlp", None)
        cand = getattr(getattr(mlp, "gate_up", None), "weight", None)
        if cand is not None:
            weight = cand
            break
    out: List[Tuple] = []
    for name in candidates:
        if name is None or weight is None:
            out.append((name, True, ""))
            continue
        try:
            ok = gemm_dispatch._cache_policy_allows(name, weight)  # noqa: SLF001
        except Exception:  # pragma: no cover
            ok = True
        why = ""
        if not ok:
            why = (
                f"would allocate a second ~{gemm_dispatch.repack_cache_bytes(weight) / 2**30:.0f} "
                f"GiB repack cache under --gemm-weight-cache "
                f"{gemm_dispatch.get_weight_cache_policy()!r} ("
                f"re-run with --gemm-weight-cache multi and 23 GiB spare)"
            )
        out.append((name, ok, why))
    return out


def gemm_backend_sweep(model, batch, cfg, *, steps: int = 5,
                       peak_tflops: float = DEFAULT_PEAK_TFLOPS,
                       force_cache_owning: bool = False) -> Dict:
    """Time one whole prefill chunk under each candidate backend.

    Whole-chunk rather than per-GEMM on purpose: the question that needs
    answering is "what does a chunk cost", and a per-GEMM microbench at M=8192
    would still have to be re-composed with the tiling ``FusedMLP`` applies
    (``mlp_tile_tokens``, so the MLP runs at M=2048 while ``in_proj_qkvz`` runs
    at the full chunk M) to answer it.

    Every fp8-activation backend here is the same 2.7e-2 relL2 class and that
    error is flat in M, so at prefill shapes this is purely a
    speed choice.  ``vllm_marlin_fp8_w8a16`` is deliberately absent: it is the
    only accurate one, and it is 88.83 ms/step at M=512 against
    ``scaled_mm``'s 28.08 and degrades monotonically -- the wrong answer here
    by a factor that no accuracy argument reaches.
    """
    n_tokens = int(batch.token_ids.numel())
    flops = prefill_ceiling(cfg, peak_tflops=peak_tflops)["flops_per_token_linear"] * n_tokens
    out: Dict[str, Dict] = {}
    for name, allowed, why in _prefill_backends_for(model):
        key = name or "dispatch(m_bucket)"
        if not allowed and not force_cache_owning:
            out[key] = {"skipped": why}
            continue
        try:
            ms = time_chunk(model, batch, steps=steps, gemm_backend=name)
        except Exception as exc:  # noqa: BLE001 -- a missing backend is data
            out[key] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
            continue
        out[key] = {
            "chunk_ms": round(ms, 2),
            "chunk_tok_s": round(n_tokens / (ms / 1e3), 1),
            "linear_tflops_achieved": round(flops / (ms / 1e3) / 1e12, 1),
            "pct_of_peak": round(100.0 * flops / (ms / 1e3) / (peak_tflops * 1e12), 1),
        }
    return out


# --------------------------------------------------------------------------- #
# 2b. the additive attribution table
# --------------------------------------------------------------------------- #
def chunk_attribution(model, batch, *, steps: int = 3, warmup: int = 1,
                      chunk_ms: Optional[float] = None) -> Dict:
    """Per-component ms for one prefill chunk, CUDA-event timed.

    The primary view of this profiler.  Unlike :func:`chunk_ablations`, the
    rows **add up**: every component is bracketed by a pair of events on the
    compute stream and read back after a single sync, so the table is a budget
    and the difference between it and the un-instrumented chunk time is
    reported as ``residual`` rather than absorbed.

    ``chunk_ms`` should be the *un-instrumented* time (``time_chunk``); when it
    is not given, this measures its own instrumented time and says so, which
    makes the residual an over-estimate by whatever the event records cost.
    """
    tape = prefill_attrib.EventTape(model.device)
    restore = prefill_attrib.instrument(model, tape)
    try:
        # Warm first with the tape *inactive*: the first instrumented chunk
        # would otherwise attribute every Triton autotune and FlashInfer JIT to
        # whichever component happened to trigger it.
        for _ in range(max(warmup, 1)):
            reset_chunk_slots(model, batch)
            model.prefill_forward(batch, all_logits=False)
        if model.device.type == "cuda":
            torch.cuda.synchronize()
        tape.reset()
        tape.active = True
        t0 = time.perf_counter()
        for _ in range(steps):
            reset_chunk_slots(model, batch)
            tape.new_repeat()
            model.prefill_forward(batch, all_logits=False)
        labels = tape.resolve()  # the single sync
        instrumented_ms = (time.perf_counter() - t0) * 1e3 / max(steps, 1)
    finally:
        tape.active = False
        restore()
    base = float(chunk_ms) if chunk_ms else instrumented_ms
    table = prefill_attrib.attribution_table(
        labels, chunk_ms=base, tokens=int(batch.token_ids.numel())
    )
    table["instrumented_chunk_ms"] = round(instrumented_ms, 2)
    table["instrumentation_overhead_ms"] = (
        round(instrumented_ms - base, 2) if chunk_ms else None
    )
    table["chunk_ms_is_instrumented"] = chunk_ms is None
    return table


def mlp_tile_ab(model, batch, tiles: Sequence[int], *, steps: int = 5) -> Dict:
    """``FusedMLP.tile`` sweep at a real chunk.

    ``mlp_tile_tokens`` defaults to 2,048, so at an 8,192-token chunk **the
    two largest GEMMs in the model never see the chunk's M at all**: they run
    as four M=2,048 tiles with a ``torch.cat`` of the four ``[2048, 5120]``
    outputs after each one, x 64 layers.  That is (a) a bucket-2048 route
    where the v8 priority table has a bucket-8192 row, and (b) ~10 GB of pure
    concat traffic per chunk.  The tile exists to bound the ``[T, 2I]``
    intermediate -- but ``serve.plan_memory``'s prefill term already budgets
    that at the **untiled** ``max_num_batched_tokens`` (``t * 2 * i *
    act_elem``), so raising the tile to the chunk size spends memory the plan
    has already reserved.
    """
    mlps = [l.mlp for l in model.layers if getattr(l, "mlp", None) is not None]
    if not mlps:
        return {"error": "no MLP layers"}
    saved = [m.tile for m in mlps]
    out: Dict[str, Dict] = {}
    n_tokens = int(batch.token_ids.numel())
    try:
        for tile in tiles:
            for m in mlps:
                m.tile = int(tile)
            # Per-tile, not per-probe: a large tile OOMing must not cost the
            # rows that already measured, nor the whole run's JSON.
            try:
                ms = time_chunk(model, batch, steps=steps)
            except Exception as exc:  # noqa: BLE001 -- a tile that does not fit is data
                out[str(int(tile))] = {
                    "error": f"{type(exc).__name__}: {exc}"[:300],
                    "free_gib": round(free_gib(model.device) or -1.0, 2),
                }
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            out[str(int(tile))] = {
                "chunk_ms": round(ms, 2),
                "chunk_tok_s": round(n_tokens / (ms / 1e3), 1),
                "n_tiles": math.ceil(n_tokens / max(int(tile), 1)),
            }
    finally:
        for m, t in zip(mlps, saved):
            m.tile = t
    base = out.get(str(int(saved[0])))
    if base and "chunk_ms" in base:
        for k, v in out.items():
            v["delta_ms_vs_default"] = round(base["chunk_ms"] - v["chunk_ms"], 2)
    return out


def gdn_chunk_size_ab(model, batch, sizes: Sequence[int], *, steps: int = 5) -> Dict:
    """fla's ``chunk_size`` (BT) at a real prefill chunk.

    fla's chunk size is not baked into the kernel's autotune config, at least
    in ``fla-core`` 0.5.2: its wrapper opens with
    ``chunk_size = kwargs.pop('chunk_size', 64)`` and accepts 16/32/64, with
    ``BT = chunk_size`` threaded all the way down and the autotune cache keyed
    on it.  Skipped with a reason (rather than silently reporting 64 three
    times) when the installed build does not take the keyword.
    """
    from ..kernels_gdn import fla_ops as gdn_fla_ops

    layers = [l.mixer for l in model.layers if hasattr(l.mixer, "conv_prefill_layout")]
    if not layers:
        return {"error": "no GDN layers"}
    if not gdn_fla_ops.supports_chunk_size():
        return {
            "skipped": "installed fla does not accept chunk_size=",
            "fla_version": gdn_fla_ops.version(),
        }
    saved = [l.rt.gdn_chunk_size for l in layers]
    out: Dict[str, Dict] = {"fla_version": gdn_fla_ops.version()}
    n_tokens = int(batch.token_ids.numel())
    try:
        for size in sizes:
            for l in layers:
                l.rt.gdn_chunk_size = int(size)
            try:
                ms = time_chunk(model, batch, steps=steps)
            except Exception as exc:  # noqa: BLE001 -- an unsupported BT is data
                out[str(int(size))] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
                continue
            out[str(int(size))] = {
                "chunk_ms": round(ms, 2),
                "chunk_tok_s": round(n_tokens / (ms / 1e3), 1),
            }
    finally:
        for l, s in zip(layers, saved):
            l.rt.gdn_chunk_size = s
    return out


def chunk_budget_sweep(model, args, budgets: Sequence[int], *, steps: int = 3) -> Dict:
    """Chunk cost vs chunk size, 8,192 / 4,096 / 2,048.

    From ``total = prefill_tokens / prefill_rate + steps x step_time`` one can
    argue that the chunk cap is a TPOT knob and cannot move aggregate
    throughput.  That argument holds only if ``prefill_rate`` is **independent
    of chunk size**, which is precisely what a GEMM-bound prefill makes
    doubtful: a 2,048-token chunk routes the GDN/attention projections on
    M-bucket 2048 instead of 8192 and gives every kernel a quarter of the work
    to amortise its launch over.  This measures tok/s per chunk size directly,
    so that model is falsifiable rather than assumed.
    """
    out: Dict[str, Dict] = {}
    for budget in budgets:
        seq_lens = chunk_shape(int(budget), args.input_len)
        batch = synthetic_chunk(model, seq_lens)
        n_tokens = sum(seq_lens)
        try:
            ms = time_chunk(model, batch, steps=steps)
        except Exception as exc:  # noqa: BLE001
            out[str(int(budget))] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
            continue
        out[str(int(budget))] = {
            "n_seqs": len(seq_lens),
            "tokens": n_tokens,
            "chunk_ms": round(ms, 2),
            "tok_s": round(n_tokens / (ms / 1e3), 1),
            "ms_per_8192_tokens": round(ms * 8192.0 / n_tokens, 1),
        }
        _release_all_slots(model)
    return out


def attn_prefill_probe(model, batch) -> Dict:
    """Paged (FlashInfer) vs ragged (FA3) prefill attention, isolated.

    A standalone microbenchmark prices attention prefill at ~5 ms of a
    1,130 ms chunk; this checks that number against
    the *chunk's own* shape, and prices the alternative vLLM uses (FA3 ragged
    varlen over packed k/v, no page table) at the same shape.

    Deliberately a microbenchmark and not a model change: the ragged path is
    only *correct* when every segment in the chunk starts at position 0
    (``kv_len == q_len``), which a chunked-prefill scheduler cannot guarantee
    -- the fourth segment of an 8,192-token chunk at prompt 2,139 is a
    1,775-token prefix whose next chunk has 1,775 tokens of context.  So the
    question this answers is "is it worth building the fast path plus a
    fallback", and at <1% of the chunk the answer is expected to be no.
    """
    from ..attn import flashinfer_attn as fi

    if model.device.type != "cuda":
        return {"skipped": "cuda only"}
    cfg = model.config
    seq_lens = [int(n) for n in batch.q_lens]
    t = int(sum(seq_lens))
    dev, dt = model.device, model.dtype
    hq, hkv, d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    q = torch.randn(t, hq, d, device=dev, dtype=dt)
    k = torch.randn(t, hkv, d, device=dev, dtype=dt)
    v = torch.randn(t, hkv, d, device=dev, dtype=dt)
    cu = torch.zeros(len(seq_lens) + 1, device=dev, dtype=torch.int32)
    cu[1:] = torch.tensor(seq_lens, device=dev, dtype=torch.int32).cumsum(0)
    # The paged wrapper needs the chunk's own plan; `prefill_forward` builds it
    # per chunk and this probe runs outside one.
    reset_chunk_slots(model, batch)
    model.attn.plan_prefill(
        [int(s) for s in batch.seq_slots], list(batch.q_lens), list(batch.kv_lens)
    )
    out: Dict[str, Any] = {
        "has_fa3": bool(fi.HAS_FA3),
        "seq_lens": seq_lens,
        "n_attn_layers": 16,
    }

    def timed(fn, reps: int = 20) -> float:
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1e3 / reps

    if fi.HAS_FA3:
        try:
            ms = timed(lambda: fi.fa3_varlen_prefill(q, k, v, cu, max(seq_lens)))
            out["fa3_ragged"] = {"us_per_layer": round(ms * 1e3, 1)}
        except Exception as exc:  # noqa: BLE001
            out["fa3_ragged"] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
    try:
        ms = timed(lambda: model.attn.prefill(0, q.contiguous()))
        out["flashinfer_paged"] = {"us_per_layer": round(ms * 1e3, 1)}
    except Exception as exc:  # noqa: BLE001
        out["flashinfer_paged"] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
    for key in ("fa3_ragged", "flashinfer_paged"):
        row = out.get(key, {})
        if "us_per_layer" in row:
            row["ms_per_chunk_16_layers"] = round(row["us_per_layer"] * 16 / 1e3, 2)
    return out


def decode_overlap_probe(comps, batch, *, batch_size: int = 256, reps: int = 5) -> Dict:
    """Can a decode graph replay hide inside a prefill chunk?

    Decode is 23% of the wall clock at conc 256 and is bandwidth-bound; a
    prefill chunk is compute-bound.  If the two really do use disjoint
    resources, running one graph replay on a second stream *during* a chunk
    should cost close to nothing, and the ceiling on the idea is the whole
    23%.  This measures that ceiling before anything is built, because
    building it means giving up the scheduler's "a step is never mixed"
    invariant and every guarantee that rests on it.

    **This is a probe, not a feature.** It replays the decode graph over
    whatever rows the graph's own static buffers hold, concurrently with a
    chunk over disjoint slots, and reports only wall time -- it does not check
    the decode output, because on this path the answer would be meaningless
    anyway (the point is the timing, and the correctness argument for a real
    implementation is a different, harder question).
    """
    if not torch.cuda.is_available() or comps.model.device.type != "cuda":
        return {"skipped": "cuda only"}
    model, decoder = comps.model, comps.decoder
    graphs = getattr(decoder, "_graphs", {})  # noqa: SLF001 -- no public replay()
    # Decode rows must not collide with the chunk's slots: `reset_chunk_slots`
    # frees the chunk slots' pages between repeats, and a decode plan built over
    # those rows would then read page indices the allocator has handed back.
    first_free = len(batch.seq_slots)
    room = int(model.n_slots) - first_free
    candidates = [b for b in graphs if b <= min(batch_size, room)]
    if not candidates:
        return {"skipped": f"no captured decode graph fits {room} free slots"}
    bucket = max(candidates)
    slots = list(range(first_free, first_free + bucket))
    try:
        for s in slots:
            model.reset_slot(s)
            model.kv_pool.ensure_capacity(s, model.kv_pool.cfg.page_size)
        model.attn.plan_decode(slots, bucket, seq_lens=[model.kv_pool.cfg.page_size] * bucket)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"plan_decode: {type(exc).__name__}: {exc}"[:300]}

    def replay():
        graphs[bucket].replay()

    def chunk():
        reset_chunk_slots(model, batch)
        model.prefill_forward(batch, all_logits=False)

    def timeit(fn, n: int) -> float:
        fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1e3 / n

    try:
        decode_ms = timeit(replay, 10)
        chunk_ms = timeit(chunk, reps)
        side = torch.cuda.Stream()
        main = torch.cuda.current_stream()

        def both():
            reset_chunk_slots(model, batch)
            side.wait_stream(main)
            with torch.cuda.stream(side):
                graphs[bucket].replay()
            model.prefill_forward(batch, all_logits=False)
            main.wait_stream(side)

        both_ms = timeit(both, reps)
    except Exception as exc:  # noqa: BLE001 -- a probe that cannot run is data
        return {"error": f"{type(exc).__name__}: {exc}"[:300]}
    serial = chunk_ms + decode_ms
    return {
        "bucket": int(bucket),
        "decode_replay_ms": round(decode_ms, 2),
        "chunk_ms": round(chunk_ms, 2),
        "serial_ms": round(serial, 2),
        "overlapped_ms": round(both_ms, 2),
        "hidden_ms": round(serial - both_ms, 2),
        "hidden_pct_of_decode": round(100.0 * (serial - both_ms) / decode_ms, 1)
        if decode_ms
        else 0.0,
    }


def mixed_vs_separate(comps, batch, *, batch_size: int = 256, steps: int = 3,
                      warmup: int = 1) -> Dict:
    """The mixed-forward measurement: one mixed forward vs a chunk + a decode step.

    ``batch`` is the same synthetic chunk every other view here
    uses; this adds ``batch_size`` decode rows on slots **disjoint** from the
    chunk's (``reset_chunk_slots`` hands the chunk slots' pages back between
    repeats, so a decode row planned over one of them would read a freed page)
    and times three things:

    * ``chunk_ms``   -- ``prefill_forward`` alone, the "before" prefill step;
    * ``decode_ms``  -- the graphed decode step at the same batch, the "before"
      decode step (a graph replay when one is captured, else the eager body --
      whichever the server would actually run);
    * ``mixed_ms``   -- ``mixed_forward`` over both at once.

    ``hidden_ms = chunk_ms + decode_ms - mixed_ms`` is the decode time the
    mixed step absorbs into the chunk's GEMMs, and ``hidden_pct_of_decode`` is
    the fraction of a decode step it hides. The prediction it exists to test:
    at conc 256 a B=256 decode step is 42.7 ms of almost pure weight reads,
    every one of which the chunk is already paying, so this should come back
    near 100% minus the eager-launch overhead the graph was hiding.

    Correctness is *not* checked here -- that is the CPU parity suite's job
    (``tests/test_serving_path.py::TestMixedStepMatchesSeparateSteps``). This
    is wall time only.
    """
    if comps.model is None or comps.model.device.type != "cuda":
        return {"skipped": "cuda only"}
    model, decoder = comps.model, comps.decoder
    first_free = len(batch.seq_slots)
    room = int(model.n_slots) - first_free
    b = min(int(batch_size), room)
    if b < 1:
        return {"skipped": f"no free slots for decode rows ({room} free)"}
    slots = list(range(first_free, first_free + b))
    page = model.kv_pool.cfg.page_size
    ctx_len = page  # one page of committed context per decode row
    buf = decoder.buf
    try:
        for sl in slots:
            model.reset_slot(sl)
            model.kv_pool.ensure_capacity(sl, ctx_len + 1)
        bucket = decoder.bucket_for(b)
        pad = bucket - b
        scratch = model.scratch_slot
        pad_slots = slots + [scratch] * pad
        buf.host["input_ids"][:bucket] = torch.tensor([1] * b + [0] * pad, dtype=torch.int32)
        buf.host["positions"][:bucket] = torch.tensor(
            [ctx_len] * b + [0] * pad, dtype=torch.int32
        )
        buf.host["slot_ids"][:bucket] = torch.tensor(pad_slots, dtype=torch.int32)
        buf.host["temperature"][:bucket] = torch.zeros(bucket, dtype=torch.float32)
        buf.host["top_p"][:bucket] = torch.ones(bucket, dtype=torch.float32)
        buf.host["top_k"][:bucket] = torch.zeros(bucket, dtype=torch.float32)
        buf.upload(["input_ids", "positions", "slot_ids", "temperature", "top_p", "top_k"])
        seq_lens = [ctx_len + 1] * b + [1] * pad
        mixed = make_mixed_batch(
            batch, slots, [1] * b, [ctx_len] * b, model.device
        )
    except Exception as exc:  # noqa: BLE001 -- a probe that cannot run is data
        return {"error": f"setup: {type(exc).__name__}: {exc}"[:300]}

    def reset_decode_slots() -> None:
        for sl in slots:
            model.kv_pool.seq_len[sl] = ctx_len

    def chunk() -> None:
        reset_chunk_slots(model, batch)
        model.prefill_forward(batch, all_logits=False)

    def decode() -> None:
        reset_decode_slots()
        decoder.step(b, pad_slots, seq_lens=seq_lens)

    def mixed_step() -> None:
        reset_chunk_slots(model, batch)
        reset_decode_slots()
        model.mixed_forward(mixed)

    def timeit(fn, n: int) -> float:
        for _ in range(warmup + 1):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1e3 / n

    try:
        chunk_ms = timeit(chunk, steps)
        decode_ms = timeit(decode, max(steps * 2, 4))
        mixed_ms = timeit(mixed_step, steps)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"[:300]}
    serial = chunk_ms + decode_ms
    n_tokens = sum(batch.q_lens)
    return {
        "decode_rows": b,
        "bucket": int(bucket),
        "chunk_tokens": int(n_tokens),
        "chunk_ms": round(chunk_ms, 2),
        "decode_ms": round(decode_ms, 2),
        "separate_ms": round(serial, 2),
        "mixed_ms": round(mixed_ms, 2),
        "hidden_ms": round(serial - mixed_ms, 2),
        "hidden_pct_of_decode": round(100.0 * (serial - mixed_ms) / decode_ms, 1)
        if decode_ms else 0.0,
        "speedup": round(serial / mixed_ms, 3) if mixed_ms else 0.0,
        # The projected throughput: output tok/s if
        # every step looked like this one and every decode row emitted a token.
        "projected_out_tok_s_separate": round(b / (serial / 1e3), 1) if serial else 0.0,
        "projected_out_tok_s_mixed": round(b / (mixed_ms / 1e3), 1) if mixed_ms else 0.0,
    }


def mixed_graph_attrib(comps, *, chunk_tokens: int, batch_size: int = 256,
                       n_segments: int = 8, prompt_len: int = 2139,
                       steps: int = 3, warmup: int = 1, top_n: int = 20,
                       check: bool = True) -> Dict:
    """Where a mixed step's time goes, eager vs graphed.

    One padded mixed step at ``(chunk_tokens, bucket(batch_size))``, run four
    ways and reported as one table:

    * **eager** -- ``mixed_forward`` on the padded batch, the ungraphed
      mixed path. Wall ms, the *host issue* time (how long the call
      takes to return before any sync -- i.e. how long the host spends
      launching), device ms from CUDA events, and the ``torch.profiler``
      kernel table folded into op groups, whose ``launches`` column is the
      number this probe exists to produce.
    * **graphed** -- the same batch through a freshly captured
      :class:`~.mixed_graphs.MixedGraphRunner`: the same four numbers, plus how
      many graph segments and eager holes the capture produced.

    ``host_issue_ms`` is the whole point. A step whose device work is 90 ms and
    whose host work is 50 ms does not take 90 ms: the small ops are host-bound
    one at a time, so the GPU idles between the GEMMs and the step takes the
    sum. Removing the launches is worth ``eager_step_ms - graphed_step_ms``,
    and ``eager_host_issue_ms`` says how much of that was ever available.

    Correctness is not checked here -- ``tests/test_mixed_graphs.py`` owns
    that, on CPU, where a padded step and an unpadded one can be compared
    token for token.
    """
    if comps.model is None or comps.model.device.type != "cuda":
        return {"skipped": "cuda only"}
    from .mixed_graphs import MixedGraphRunner, MixedPadSpec, pad_mixed_step, reset_scratch_state

    model, decoder, rt = comps.model, comps.decoder, comps.rt
    spec = MixedPadSpec(
        chunk_tokens=int(chunk_tokens), n_segments=int(n_segments),
        buckets=tuple(rt.buckets_for()), scratch_slot=model.scratch_slot,
        max_pad_len=int(getattr(rt, "max_model_len", 0) or 0),
    )
    if spec.budget < 1:
        return {"skipped": f"chunk_tokens={chunk_tokens} < n_segments={n_segments}"}
    lens = chunk_shape(spec.budget, prompt_len)[: spec.max_segments]
    n_real_seg = len(lens)
    b = min(int(batch_size), int(model.n_slots) - n_real_seg)
    bucket = spec.bucket_for(b)
    if b < 1 or bucket is None:
        return {"skipped": f"no free slots / bucket for {batch_size} decode rows"}

    page = model.kv_pool.cfg.page_size
    ctx_len = page
    dec_slots = list(range(n_real_seg, n_real_seg + b))
    try:
        gen = torch.Generator(device="cpu").manual_seed(1234)
        real_ids = []
        for sl, n in zip(range(n_real_seg), lens):
            model.reset_slot(sl)
            model.kv_pool.ensure_capacity(sl, int(n) + 1)
            real_ids.append(torch.randint(
                0, model.config.vocab_size, (int(n),), generator=gen, dtype=torch.int64
            ).tolist())
        for sl in dec_slots:
            model.reset_slot(sl)
            model.kv_pool.ensure_capacity(sl, ctx_len + 1)
        model.kv_pool.ensure_capacity(model.scratch_slot, spec.chunk_tokens)
        padded = pad_mixed_step(
            real_ids, [0] * n_real_seg, list(range(n_real_seg)),
            dec_slots, [1] * b, [ctx_len] * b, spec,
        )
        pbatch = make_prefill_batch(
            padded.token_ids, padded.start_positions, padded.slots, model.device
        )
        mixed = make_mixed_batch(
            pbatch, padded.decode_slots, padded.decode_token_ids,
            padded.decode_positions, model.device,
        )
    except Exception as exc:  # noqa: BLE001 -- a probe that cannot run is data
        return {"error": f"setup: {type(exc).__name__}: {exc}"[:300]}

    def reset() -> None:
        reset_chunk_slots(model, pbatch)
        for sl in dec_slots:
            model.kv_pool.seq_len[sl] = ctx_len
        reset_scratch_state(model)

    _dec_idx = torch.tensor(dec_slots, dtype=torch.long, device=model.device)

    def reset_full() -> None:
        """``reset()`` plus the decode rows' recurrent state.

        ``reset()`` deliberately does *not* do this -- it runs inside
        the wall-clock loop, and zeroing 256 slots of SSM pool is ~19 GB of
        writes (~10 ms), which would land in the number being measured. But it
        means the decode rows' state **advances across timed iterations**, so
        two arms measured one after the other do not start from the same
        state and comparing their outputs measures the drift, not the arms.
        Only :func:`fingerprint` uses this one, and it is outside every clock.
        """
        reset()
        model.state_pool.index_fill_(0, _dec_idx, 0)
        model.conv_pool.index_fill_(0, _dec_idx, 0)

    def measure(run, n: int) -> Dict[str, float]:
        for _ in range(warmup + 1):
            reset()
            run()
        torch.cuda.synchronize()
        # (a) wall: the number the ledger sees.
        t0 = time.perf_counter()
        for _ in range(n):
            reset()
            run()
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) * 1e3 / n
        # (b) host issue: how long the call takes to *return*, no sync. This
        # is the launch cost, and it is invisible to any device profiler.
        reset()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            run()
        issue = (time.perf_counter() - t0) * 1e3 / n
        torch.cuda.synchronize()
        # (c) device: CUDA events around the same calls.
        ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
        reset()
        torch.cuda.synchronize()
        ev0.record()
        for _ in range(n):
            run()
        ev1.record()
        torch.cuda.synchronize()
        # (d) host issue, ISOLATED: sync before each call, so the clock is the
        # host's own work and not the queue it is standing behind.
        #
        # (b) above loops `n` times with no sync, which is the right
        # measurement for an eager step -- the host really is blocked behind
        # its own launch queue there, and that blocking *is* the cost. It is
        # the wrong one for a graphed step: FlashInfer's `plan()` does a
        # blocking H2D of its indptr buffers, so from the second iteration on
        # the plan waits for the *previous* replay to finish and (b) reads back
        # ~(n-1)/n of the device time no matter how cheap the launch is. The
        # difference between (b) and (d) is the difference between "the graph
        # did not help" and "the graph helped and (b) is now measuring the GPU".
        iso = 0.0
        for _ in range(n):
            reset()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            run()
            iso += time.perf_counter() - t0
        torch.cuda.synchronize()
        return {
            "step_ms": round(wall, 2),
            "host_issue_ms": round(issue, 2),
            "host_issue_isolated_ms": round(iso * 1e3 / n, 2),
            "device_ms": round(ev0.elapsed_time(ev1) / n, 2),
        }

    def plan_only_ms(n: int) -> float:
        """Host time in the FlashInfer plan alone.

        Once the step is one graph replay, ``host_issue_ms`` is no
        longer "the launches" -- the plan runs on the host *outside* the graph
        on every step, by construction (the same plan()-outside/run()-inside
        split ``GraphedDecoder`` has always had), and it is the only host work
        left that scales with the row count. Splitting it out is the
        difference between "the graph did not help" and "the graph helped and
        the plan is the next floor".
        """
        reset()
        model.attn.plan_mixed_graph(
            mixed.plan_slots, mixed.plan_q_lens, mixed.plan_kv_lens
        )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            model.attn.plan_mixed_graph(
                mixed.plan_slots, mixed.plan_q_lens, mixed.plan_kv_lens
            )
        ms = (time.perf_counter() - t0) * 1e3 / n
        torch.cuda.synchronize()
        return round(ms, 2)

    def eager_plan_only_ms(n: int) -> float:
        reset()
        model.attn.plan_prefill(mixed.plan_slots, mixed.plan_q_lens, mixed.plan_kv_lens)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            model.attn.plan_prefill(
                mixed.plan_slots, mixed.plan_q_lens, mixed.plan_kv_lens
            )
        ms = (time.perf_counter() - t0) * 1e3 / n
        torch.cuda.synchronize()
        return round(ms, 2)

    def kernels(run) -> Dict:
        from torch.profiler import ProfilerActivity, profile

        reset()
        run()
        torch.cuda.synchronize()
        reset()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=False) as prof:
            run()
            torch.cuda.synchronize()
        groups: "OrderedDict[str, Dict[str, float]]" = OrderedDict()
        rows = []
        for evt in prof.key_averages():
            us = float(getattr(evt, "self_device_time_total", 0.0) or 0.0)
            if us <= 0:
                continue
            rows.append({"name": evt.key, "us": round(us, 1), "calls": int(evt.count)})
            g = classify_kernel(evt.key)
            slot = groups.setdefault(g, {"us": 0.0, "launches": 0})
            slot["us"] += us
            slot["launches"] += int(evt.count)
        rows.sort(key=lambda r: -r["us"])
        total_us = sum(v["us"] for v in groups.values()) or 1.0
        return {
            "launches": int(sum(v["launches"] for v in groups.values())),
            "device_us_total": round(total_us, 1),
            "groups": {
                k: {"us": round(v["us"], 1), "launches": v["launches"],
                    "pct": round(100.0 * v["us"] / total_us, 1)}
                for k, v in sorted(groups.items(), key=lambda kv: -kv[1]["launches"])
            },
            "top_kernels": rows[:top_n],
        }

    def fingerprint(run) -> Dict[str, object]:
        """One arm's observable output: the logits and every pool it wrote.

        The three arms run *different code* (eager fla with its own
        index prep; a 49-segment capture whose fla call is an eager hole; one
        graph whose fla call is inside it), and the whole claim is that they
        compute the same step. Comparing wall time without comparing this
        would be measuring three things and reporting one.

        The scratch slot is deliberately **excluded**: every arm's padding
        threads through it and nothing reads the result.

        Only ``check_slots`` are read back, not the whole pool: at conc 256 the
        SSM pool is 263 live slots x 48 layers x 48 heads x 128 x 128, i.e.
        ~20 GiB in fp16 and 40 in fp32, and cloning it is an 18.6 GiB
        allocation into ~16 GiB of headroom. The slots kept are every *prefill*
        segment (the half the one-graph capture changes, and the only one whose
        state is written by the fla chunk kernel) plus a sample of decode rows
        (written by the Triton GDN decode kernel, which is inside the graph in both graphed
        arms). A padding row leaking into a real slot, or a segment boundary
        off by one, moves those by O(1).
        """
        reset_full()
        o = run()
        logits = (o[0] if isinstance(o, tuple) else o).detach().float().clone()
        k = torch.tensor(check_slots, device=model.device)
        return {
            "logits": logits,
            "state": model.state_pool.index_select(0, k).detach().float().clone(),
            "conv": model.conv_pool.index_select(0, k).detach().float().clone(),
        }

    # Every prefill segment, plus up to 8 decode rows. `n_real_seg` is 1-2 at
    # the serving geometry, so this is ~10 slots (~1.5 GiB read, 190 MiB kept
    # in fp32) instead of 263.
    check_slots = list(range(n_real_seg)) + dec_slots[:8]

    def compare(a: Dict[str, object], c: Dict[str, object]) -> Dict[str, float]:
        outd: Dict[str, float] = {}
        for key in ("logits", "state", "conv"):
            x, y = a[key], c[key]  # type: ignore[index]
            d = (x - y).abs()
            scale = x.abs().max().clamp_min(1e-6)
            outd[f"max_abs_{key}"] = float(d.max())
            outd[f"max_rel_{key}"] = float(d.max() / scale)
        am = a["logits"].argmax(-1)  # type: ignore[union-attr]
        cm = c["logits"].argmax(-1)  # type: ignore[union-attr]
        outd["argmax_mismatches"] = float((am != cm).sum())
        outd["argmax_rows"] = float(am.numel())
        return outd

    out: Dict = {
        "chunk_tokens": int(spec.chunk_tokens),
        "real_prefill_tokens": int(sum(lens)),
        "n_segments": int(n_segments),
        "n_real_segments": n_real_seg,
        "decode_rows": b,
        "bucket": int(bucket),
    }
    try:
        out["eager"] = measure(lambda: model.mixed_forward(mixed), steps)
        out["eager"].update(kernels(lambda: model.mixed_forward(mixed)))
        out["eager"]["plan_ms"] = eager_plan_only_ms(max(steps, 3))
        out["eager"]["host_issue_ex_plan_ms"] = round(
            out["eager"]["host_issue_isolated_ms"] - out["eager"]["plan_ms"], 2
        )
        ref = fingerprint(lambda: model.mixed_forward(mixed)) if check else None
    except Exception as exc:  # noqa: BLE001
        return {**out, "error": f"eager: {type(exc).__name__}: {exc}"[:300]}

    # Three arms, not two. `graphed_holes` is the
    # 49-segment/48-hole capture and `graphed` is the one-graph capture, so the
    # ledger can attribute the remaining host time to the holes specifically
    # rather than to "graphing" in general.
    runner = None
    for key, holes in (("graphed_holes", True), ("graphed", False)):
        try:
            runner = MixedGraphRunner(
                model, rt, chunk_tokens=spec.chunk_tokens, buckets=(bucket,),
                n_segments=n_segments, holes=holes,
            )
            if holes != runner.holes:
                out[f"{key}_skipped"] = runner.static_index_reason or "unavailable"
                continue
            runner.warmup()
            runner.capture()
            out[key] = measure(lambda: runner.step(mixed), steps)
            out[key].update(kernels(lambda: runner.step(mixed)))
            out[key].update(runner.stats())
            out[key]["plan_ms"] = plan_only_ms(max(steps, 3))
            out[key]["host_issue_ex_plan_ms"] = round(
                out[key]["host_issue_isolated_ms"] - out[key]["plan_ms"], 2
            )
            if ref is not None:
                r = runner  # bind for the closure below
                out[f"parity_{key}"] = compare(ref, fingerprint(lambda: r.step(mixed)))
        except Exception as exc:  # noqa: BLE001
            out[f"{key}_error"] = f"{type(exc).__name__}: {exc}"[:300]
        finally:
            del runner
            runner = None
            torch.cuda.empty_cache()
    # Leave the model's attention runner pointing at the ordinary prefill
    # wrapper: every later probe plans through `plan_prefill`, and a stale
    # `_active_prefill` would run those against a fixed-size graph wrapper.
    model.attn._active_prefill = model.attn._prefill_wrapper  # noqa: SLF001

    e = out["eager"]
    out["projected_out_tok_s_eager"] = (
        round(b / (e["step_ms"] / 1e3), 1) if e["step_ms"] else 0.0
    )
    for key in ("graphed_holes", "graphed"):
        g = out.get(key)
        if not g:
            continue
        suffix = "" if key == "graphed" else "_holes"
        out[f"saved_ms{suffix}"] = round(e["step_ms"] - g["step_ms"], 2)
        out[f"launches_removed{suffix}"] = int(e["launches"] - g["launches"])
        out[f"speedup{suffix}"] = (
            round(e["step_ms"] / g["step_ms"], 3) if g["step_ms"] else 0.0
        )
        out[f"projected_out_tok_s_graphed{suffix}"] = (
            round(b / (g["step_ms"] / 1e3), 1) if g["step_ms"] else 0.0
        )
    if "graphed" in out and "graphed_holes" in out:
        # What closing the 48 holes was worth on its own.
        out["holes_cost_ms"] = round(
            out["graphed_holes"]["step_ms"] - out["graphed"]["step_ms"], 2
        )
        out["holes_cost_host_issue_ms"] = round(
            out["graphed_holes"]["host_issue_ms"] - out["graphed"]["host_issue_ms"], 2
        )
    torch.cuda.empty_cache()
    return out


def conv_layout_ab(model, batch, *, steps: int = 5) -> Dict:
    """``token_major`` (Triton kernel) vs ``channel_major`` (older fallback).

    Flips the layout on every GDN layer rather than rebuilding the model, so
    the two rows differ in exactly one thing.
    """
    layers = [l for l in model.layers if hasattr(l.mixer, "conv_prefill_layout")]
    if not layers:
        return {"error": "no GDN layers expose conv_prefill_layout"}
    saved = [l.mixer.conv_prefill_layout for l in layers]
    out: Dict[str, Dict] = {}
    try:
        for layout in ("channel_major", "token_major"):
            for l in layers:
                l.mixer.conv_prefill_layout = layout
            try:
                ms = time_chunk(model, batch, steps=steps)
            except Exception as exc:  # noqa: BLE001
                out[layout] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
                continue
            out[layout] = {"chunk_ms": round(ms, 2)}
    finally:
        for l, v in zip(layers, saved):
            l.mixer.conv_prefill_layout = v
    if "channel_major" in out and "token_major" in out and "chunk_ms" in out["token_major"]:
        a, b = out["channel_major"].get("chunk_ms"), out["token_major"]["chunk_ms"]
        if a:
            out["speedup"] = round(a / b, 2)
            out["saved_ms_per_chunk"] = round(a - b, 2)
    return out


def conv_layout_parity(model, batch, *, atol: float = 2e-2) -> Dict:
    """Do the two conv layouts agree on the chunk's logits?

    The Triton kernel is a from-scratch reimplementation of a numerically
    exact tiling, so "same answer" is a claim that has to be checked on the
    real checkpoint and not only on the tiny CPU model -- and it has to be
    checked *here*, because a wrong conv state silently poisons every
    subsequent decode step rather than raising.
    """
    layers = [l for l in model.layers if hasattr(l.mixer, "conv_prefill_layout")]
    saved = [l.mixer.conv_prefill_layout for l in layers]
    ref = None
    got = None
    try:
        for layout, sink in (("channel_major", "ref"), ("token_major", "got")):
            for l in layers:
                l.mixer.conv_prefill_layout = layout
            reset_chunk_slots(model, batch)
            logits = model.prefill_forward(batch, all_logits=False).float()
            if sink == "ref":
                ref = logits.clone()
            else:
                got = logits.clone()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"[:200]}
    finally:
        for l, v in zip(layers, saved):
            l.mixer.conv_prefill_layout = v
    if ref is None or got is None:
        return {"error": "one of the two layouts did not run"}
    diff = (ref - got).abs()
    rel = float(diff.norm() / ref.norm().clamp(min=1e-9))
    return {
        "max_abs": round(float(diff.max()), 6),
        "rel_l2": float(f"{rel:.3e}"),
        "argmax_agree": float((ref.argmax(-1) == got.argmax(-1)).float().mean()),
        "ok": rel < atol,
    }


# --------------------------------------------------------------------------- #
# 4. GIL contention with the HTTP thread
# --------------------------------------------------------------------------- #
class DetokLoad:
    """A background thread doing detokenizer-shaped Python at a target rate.

    The real thing is ``server/tokenization.py`` running inside uvicorn's event
    loop on the same interpreter as the engine thread; this stands in for it
    with the same *character*: pure-Python string building and dict work, no
    C-extension call long enough to drop the GIL.  That is what makes it a GIL
    probe rather than a CPU probe -- a numpy loop would release the GIL and
    measure nothing.
    """

    def __init__(self, tokens_per_s: float, *, batch: int = 64):
        self.tokens_per_s = max(tokens_per_s, 1.0)
        self.batch = batch
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.iterations = 0

    def _work(self) -> None:
        vocab = [f"tok{i}" for i in range(4096)]
        period = self.batch / self.tokens_per_s
        while not self._stop.is_set():
            t0 = time.perf_counter()
            buf: List[str] = []
            for i in range(self.batch):
                buf.append(vocab[(i * 7919) % len(vocab)])
            text = "".join(buf)
            _ = {"choices": [{"delta": {"content": text}, "index": 0}]}
            self.iterations += 1
            slack = period - (time.perf_counter() - t0)
            if slack > 0:
                time.sleep(slack)

    def __enter__(self) -> "DetokLoad":
        self._thread = threading.Thread(target=self._work, name="detok-probe", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


# --------------------------------------------------------------------------- #
# 5. build
# --------------------------------------------------------------------------- #
def serving_pool_sizes(max_num_seqs: int, ctx_len: int, page_size: int,
                       slack_pages: int = 16) -> Dict[str, int]:
    """The geometry ``runtime.serve`` derives -- literally its function.

    A re-implementation here could drift (e.g. a different slack) and size a
    pool the server would not have.  Delegating to
    ``bench_runtime.derive_pool_sizes`` (which is what ``serve.
    runtime_config_from_args`` calls) is the only way "the profile runs the
    served config" can be true rather than intended.
    """
    return derive_pool_sizes(max_num_seqs, ctx_len, page_size, slack_pages=slack_pages)


def ctx_len_for(args) -> int:
    return int(args.input_len + args.output_len + 64)


def _pages_per_seq_floor(derived: int, args, ctx_len: int = 0) -> int:
    """``max_pages_per_seq``, raised if a graphed mixed step needs more.

    See ``mixed_graphs.scratch_page_floor``. Only ever raises, and
    only when ``--mixed-graphs`` is on, so every non-graphed configuration is
    unaffected.
    """
    if not (getattr(args, "mixed_forward", False) and getattr(args, "mixed_graphs", False)):
        return derived
    from .mixed_graphs import scratch_page_floor

    chunk = getattr(args, "prefill_chunk_tokens", 0) or getattr(
        args, "max_num_batched_tokens", 0
    )
    return max(
        int(derived),
        scratch_page_floor(chunk, args.page_size, int(ctx_len or 0)) + 1,
    )


def runtime_config_for(args, concurrency: int) -> RuntimeConfig:
    """The ``RuntimeConfig`` this profile will build -- **no CUDA required**.

    Split out of :func:`build` so the memory plan can be computed, printed and
    *tested* without a GPU (``tests/test_serving_path.py::TestProfilerMemory
    Plan``).  Every knob comes from ``args``, which ``--preset fastest`` has
    already filled from ``preset.CANONICAL_FAST`` -- so this is the served
    config by construction rather than by a list of flags kept in sync by
    hand (a default-``RuntimeConfig`` engine is not the served config and can
    OOM on ``alloc_state_pool``).
    """
    ctx_len = ctx_len_for(args)
    # The pools are sized for `--pool-max-num-seqs` (default:
    # `concurrency`), which is a *separate* number from the chunk composition.
    # `chunk_shape(8192, 2139)` is `[2139, 2139, 2139, 1775]` -- **four**
    # sequences -- so every component view below is byte-identical at
    # `--pool-max-num-seqs 64` and at 256, while the KV pool goes 44.9 -> 11.4
    # GiB and the SSM pool 18.1 -> 4.6 GiB. That is ~47 GiB of headroom that
    # the component probes can use at no cost to what they measure.
    # Only the closed-loop ledger genuinely needs `max_num_seqs == concurrency`,
    # which `build_arg_parser` enforces (see `--pool-max-num-seqs`).
    n_slots = int(getattr(args, "pool_max_num_seqs", 0) or 0) or concurrency
    buckets = tuple(b for b in RuntimeConfig.graph_buckets if b <= n_slots) or (1,)
    if buckets[-1] < n_slots:
        buckets = buckets + (n_slots,)
    geom = serving_pool_sizes(n_slots, ctx_len, args.page_size)
    return RuntimeConfig(
        device=args.device,
        dtype=args.dtype,
        ssm_state_dtype=args.ssm_state_dtype,
        kv_cache_dtype=args.kv_cache_dtype,
        page_size=args.page_size,
        max_num_seqs=n_slots,
        graph_buckets=buckets,
        use_cuda_graphs=not args.no_graphs,
        max_model_len=ctx_len,
        n_kv_pages=geom["n_kv_pages"],
        max_pages_per_seq=_pages_per_seq_floor(geom["max_pages_per_seq"], args, ctx_len),
        max_num_batched_tokens=args.max_num_batched_tokens,
        prefill_decode_ratio=args.prefill_decode_ratio,
        prefill_chunk_tokens=args.prefill_chunk_tokens,
        mixed_forward=getattr(args, "mixed_forward", False),
        mixed_graphs=getattr(args, "mixed_graphs", False),
        mixed_graph_segments=getattr(args, "mixed_graph_segments", 8),
        mixed_graph_holes=getattr(args, "mixed_graph_holes", False),
        mixed_graph_min_bucket=getattr(args, "mixed_graph_min_bucket", 32),
        mixed_graph_buckets=tuple(getattr(args, "mixed_graph_buckets", None) or ()),
        overlap_streams=getattr(args, "overlap_streams", False),
        overlap_decode_priority=getattr(args, "overlap_decode_priority", 0),
        overlap_min_fill=getattr(args, "overlap_min_fill", 0.75),
        mlp_tile_tokens=args.mlp_tile_tokens,
        conv_prefill_layout=args.conv_prefill_layout,
        conv_prefill_tile_tokens=args.conv_prefill_tile_tokens,
        prefill_gemm_backend=args.prefill_gemm_backend,
        gemm_backend=args.gemm_backend,
        gemm_weight_cache=args.gemm_weight_cache,
        gemm_priority=args.gemm_priority,
        gemm_accuracy=args.gemm_accuracy,
        gemm_cache_owner=args.gemm_cache_owner,
        gdn_chunk_size=args.gdn_chunk_size,
        norm_backend=args.norm_backend,
        fused_ops_backend=args.fused_ops_backend,
        attn_backend=args.attn_backend,
        gdn_backend=args.gdn_backend,
        sampler_candidates=args.sampler_candidates,
        attn_workspace_mb=args.attn_workspace_mb,
    )


def plan_for(args, concurrency: int) -> Tuple[RuntimeConfig, Dict[str, float]]:
    """``(rt, plan)`` for one level, using **``serve.plan_memory`` itself**.

    Without a memory plan the profiler would build whatever ``RuntimeConfig``
    its flags imply and find out on the device.  At concurrency 256 that is a 46 GiB KV pool and an 18 GiB state pool on top
    of 28 GiB of weights and a 23 GiB repack cache, and the failure mode is a
    ``torch.OutOfMemoryError`` *inside the constructor*, 40 GiB into the load.  This is the same arithmetic
    ``serve.build_engine_from_args`` prints before it loads anything.

    No CUDA is touched, so ``tests/test_serving_path.py`` can assert the
    conc-256 plan fits.
    """
    from . import serve as serve_mod  # local: `serve` pulls in the HTTP stack

    rt = runtime_config_for(args, concurrency)
    plan = serve_mod.plan_memory(
        max_num_seqs=rt.max_num_seqs,
        max_model_len=rt.max_model_len,
        page_size=rt.page_size,
        n_kv_pages=rt.n_kv_pages,
        max_pages_per_seq=rt.max_pages_per_seq,
        kv_cache_dtype=rt.kv_cache_dtype,
        ssm_state_dtype=rt.ssm_state_dtype,
        dtype=rt.dtype,
        enable_mtp=rt.enable_mtp,
        arch=serve_mod.arch_from_checkpoint(args.model),
        max_num_batched_tokens=rt.max_num_batched_tokens,
        conv_prefill_tile_tokens=rt.conv_prefill_tile_tokens,
        n_graph_buckets=len(rt.buckets_for()),
        max_batch=rt.buckets_for()[-1] if rt.use_cuda_graphs else rt.max_num_seqs,
        attn_workspace_mb=rt.attn_workspace_mb,
        gemm_weight_cache=rt.gemm_weight_cache,
        use_cuda_graphs=rt.use_cuda_graphs,
        mixed_forward=rt.mixed_forward,
        mixed_graphs=rt.mixed_graphs,
        mixed_graph_chunk=(rt.prefill_chunk_tokens or rt.max_num_batched_tokens),
        mixed_graph_holes=rt.mixed_graph_holes,
        mixed_graph_segments=rt.mixed_graph_segments,
        mixed_graph_buckets=len(rt.mixed_buckets_for()),
        gdn_chunk_size=rt.gdn_chunk_size,
    )
    return rt, plan


def build(args, concurrency: int) -> EngineComponents:
    """Build the engine for one level, refusing to start if it will not fit.

    Three safeguards, in order:

    1. **Plan, print, and gate on the budget** (:func:`plan_for`), so an
       over-budget config is a one-line refusal before the first byte of
       weights is read rather than an OOM traceback inside
       ``alloc_state_pool``.
    2. **Check what is actually free right now**, not what was free when the
       process started: the *previous* concurrency level's engine may still
       be resident (see :func:`teardown`).
    3. **Claim the GEMM repack cache for whoever ``--gemm-cache-owner`` says**,
       before warmup resolves bucket 1 -- the same ordering
       ``serve.build_engine_from_args`` uses, because otherwise the profile
       measures a backend the server would not have used.
    """
    import gc

    from . import serve as serve_mod

    # The *previous* level's tensors may still be sitting in the caching
    # allocator's reserved pool, which `cudaMemGetInfo` reports as used -- so
    # collect and release before asking how much is free, or the budget gate
    # refuses a config that fits.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rt, plan = plan_for(args, concurrency)
    if args.verbose or not args.quiet_plan:
        print(f"[profile] concurrency {concurrency}", flush=True)
        print(serve_mod.format_memory_plan(plan, rt, rt.max_model_len), flush=True)
    free_gib = serve_mod._free_hbm_gib(rt.device)  # noqa: SLF001 -- same package
    if free_gib is not None:
        budget = free_gib * args.gpu_memory_utilization
        print(
            f"[profile] free HBM {free_gib:.1f} GiB, budget {budget:.1f} GiB "
            f"(--gpu-memory-utilization {args.gpu_memory_utilization})",
            flush=True,
        )
        if plan["total_gib"] > budget and not args.skip_memory_check:
            raise SystemExit(
                f"error: profile_serving at concurrency {concurrency} needs "
                f"{plan['total_gib']:.1f} GiB ({plan['steady_gib']:.1f} steady + "
                f"{plan['prefill_gib']:.1f} prefill) but only {budget:.1f} GiB is free x "
                f"{args.gpu_memory_utilization}. Lower --concurrency, --input-len/"
                f"--output-len (ctx {rt.max_model_len}), or --max-num-batched-tokens; or "
                f"pass --skip-memory-check to reproduce the OOM deliberately."
            )
    if args.verbose or not args.quiet_plan:
        print(format_resolved_config(rt, concurrency=concurrency), flush=True)

    comps = build_engine(args.model, rt=rt, verbose=args.verbose)
    if rt.gemm_cache_owner == "prefill":
        comps.model.claim_gemm_cache(
            m=default_claim_gemm_cache_m(rt), verbose=args.verbose
        )
    calibrate_kv_fp8(comps.model, verbose=args.verbose)
    comps.decoder.warmup()
    if rt.use_cuda_graphs:
        comps.decoder.capture()
    if comps.mixed is not None:
        # Same order and the same shared mempool as `QwenFastEngine
        # .start`, so the ledger's mixed steps are the ones the server runs.
        comps.mixed.warmup()
        if rt.use_cuda_graphs:
            # Under `--overlap` the two graphs are in flight at the same
            # time, so they must not share a mempool -- the decode graph's
            # intermediates would be the prefill graph's, at the same
            # addresses, being written concurrently.
            comps.mixed.capture(
                pool_handle=None if rt.overlap_streams else comps.decoder._pool  # noqa: SLF001
            )
    return comps


def teardown(comps: Optional[EngineComponents]) -> None:
    """Give one level's device memory back before the next level builds.

    A plain ``del comps, sched, model`` plus ``empty_cache()`` is not enough:
    the driver, the host-counter ``restore`` closure, the result dict and the
    GIL probe's second scheduler all still reference the model, and the pools
    are also reachable through reference cycles.  Without an explicit
    teardown a conc-32 engine can still hold ~60 GiB when the conc-256 build
    asks for its 46 GiB KV pool, and the OOM lands on ``alloc_state_pool``
    (the allocation right after the KV pool) rather than on the weights.

    Explicitly drops the pools first, then collects, then empties the cache,
    and prints what came back so the next level's plan can be believed.
    """
    import gc

    before = None
    if torch.cuda.is_available():
        before = torch.cuda.memory_reserved() / (1024.0 ** 3)
    if comps is not None:
        model = getattr(comps, "model", None)
        for holder, attr in (
            (model, "state_pool"),
            (model, "conv_pool"),
            (model, "kv_pool"),
            (getattr(model, "attn", None), "workspace"),
            (comps, "decoder"),
            (comps, "model"),
        ):
            if holder is not None and hasattr(holder, attr):
                try:
                    setattr(holder, attr, None)
                except Exception:  # pragma: no cover -- read-only property
                    pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        after = torch.cuda.memory_reserved() / (1024.0 ** 3)
        print(
            f"[profile] teardown: reserved {before:.1f} -> {after:.1f} GiB",
            flush=True,
        )


# --------------------------------------------------------------------------- #
# 6. one concurrency level
# --------------------------------------------------------------------------- #
def run_level(args, concurrency: int,
              emit: Optional[Callable[[Dict], None]] = None) -> Dict:
    """One concurrency level. ``emit`` is called with the (partial) level dict
    after every section, so :func:`main` can write the JSON incrementally.

    If the level dict were built as one literal and only dumped at the end,
    one late probe raising would lose every table already computed. A run's
    worth of GPU time is not something to hold in a local variable.
    """
    if emit is None:
        emit = lambda _r: None  # noqa: E731 -- a null sink, deliberately trivial
    comps = build(args, concurrency)
    model, rt = comps.model, comps.rt
    sched = Scheduler(model, comps.decoder, rt, mixed_runner=comps.mixed)

    counters, restore = install_host_counters(sched, model)
    driver = ClosedLoopDriver(
        sched,
        concurrency=concurrency,
        input_len=args.input_len,
        output_len=args.output_len,
        vocab_size=model.config.vocab_size,
        seed=args.seed,
    )
    try:
        # `--no-drive`: the component views (the attribution table, the
        # ablations, the A/Bs) need a built engine and one synthetic chunk, not
        # a 60-second closed-loop drive per level. Skipping the drive turns a
        # ~10-minute ablation arm into a ~3-minute one.
        if not args.no_drive:
            driver.run(args.duration, warmup_s=args.warmup)
    finally:
        restore()

    result: Dict = {
        "concurrency": concurrency,
        "config": {
            "input_len": args.input_len,
            "output_len": args.output_len,
            "max_num_batched_tokens": rt.max_num_batched_tokens,
            "prefill_chunk_tokens": rt.prefill_chunk_tokens or rt.max_num_batched_tokens,
            "prefill_decode_ratio": rt.prefill_decode_ratio,
            "mixed_forward": rt.mixed_forward,
            "mixed_graphs": rt.mixed_graphs,
            "mixed_graph_holes": rt.mixed_graph_holes,
            "conv_prefill_layout": rt.conv_prefill_layout,
            "prefill_gemm_backend": rt.prefill_gemm_backend,
            "gemm_weight_cache": rt.gemm_weight_cache,
            "gemm_priority": rt.gemm_priority,
            "norm_backend": rt.norm_backend,
            "fused_ops_backend": rt.fused_ops_backend,
            "ssm_state_dtype": rt.ssm_state_dtype,
            "graphs": rt.use_cuda_graphs,
        },
        "attribution": (
            attribute(driver.records, duration_s=args.duration)
            if driver.records
            else {"skipped": "--no-drive", "by_kind": {}, "output_tok_s": 0.0,
                  "served_ms_per_output_token": 0.0, "steps": 0}
        ),
        "host_counters_ms": counters.snapshot(),
        "requests_completed": driver.completed,
        "gemm_backends": summarize_backends(collect_gemm_backends(model)),
        "_physics": prefill_ceiling(model.config, peak_tflops=args.peak_tflops),
    }

    emit(result)  # the ledger is worth keeping even if a probe dies later

    # -- GIL probe: the same drive, with a detokenizer-shaped thread running -- #
    if not args.no_gil_probe and driver.records:
        base = result["attribution"]["output_tok_s"]
        # A fresh `Scheduler` has a fresh `SlotManager` that believes every
        # slot is free, but the model still holds the first drive's KV
        # reservations -- hand them back or the probe dies on page exhaustion.
        _release_all_slots(model)
        sched2 = Scheduler(model, comps.decoder, rt, mixed_runner=comps.mixed)
        d2 = ClosedLoopDriver(
            sched2,
            concurrency=concurrency,
            input_len=args.input_len,
            output_len=args.output_len,
            vocab_size=model.config.vocab_size,
            seed=args.seed + 1,
        )
        # Both arms run the **same** duration and the same warmup. With a
        # short warmup and a shorter contended arm this is not an A/B: at
        # concurrency 256, 256 prompts of 2,139 tokens are ~36 s of prefill
        # at ~15k prefill tok/s, so a short arm never reaches steady decode
        # and reports a large "slowdown" that is really the ramp. Same
        # duration, same warmup, or the number means nothing.
        with DetokLoad(tokens_per_s=max(base, 1.0)) as load:
            d2.run(args.gil_duration, warmup_s=args.warmup)
        contended = attribute(d2.records, duration_s=args.gil_duration)
        quiet = result["attribution"]
        result["gil_probe"] = {
            "detok_iterations": load.iterations,
            "duration_s": args.gil_duration,
            "warmup_s": args.warmup,
            "comparable": abs(args.gil_duration - args.duration) < 1e-6,
            "output_tok_s_quiet": base,
            "output_tok_s_contended": contended["output_tok_s"],
            "slowdown_pct": round(100.0 * (1.0 - contended["output_tok_s"] / base), 1)
            if base
            else 0.0,
            # The pair that can actually see the GIL: `output_tok_s` divides by
            # the sum of step walls (blind to between-step gaps by
            # construction), `wall_*` divides by the real clock.
            "wall_output_tok_s_quiet": quiet.get("wall_output_tok_s", 0.0),
            "wall_output_tok_s_contended": contended["wall_output_tok_s"],
            "wall_slowdown_pct": (
                round(
                    100.0
                    * (1.0 - contended["wall_output_tok_s"] / quiet["wall_output_tok_s"]),
                    1,
                )
                if quiet.get("wall_output_tok_s")
                else 0.0
            ),
            "step_busy_pct_quiet": quiet.get("step_busy_pct", 0.0),
            "step_busy_pct_contended": contended["step_busy_pct"],
            "attribution_contended": contended,
        }
        emit(result)

    # -- component views on one representative chunk ------------------------- #
    # Every section goes through `run_probe` and every section flushes, so a
    # late OOM costs *that section* and not the run (a chunk dict built as one
    # literal would lose every finished table when a later entry raises).
    # `need_gib` is the other half: a
    # probe that cannot fit says so instead of discovering it 20 GiB in.
    if not args.no_components:
        _release_all_slots(model)
        budget = rt.prefill_chunk_tokens or rt.max_num_batched_tokens
        seq_lens = chunk_shape(budget, args.input_len)[:concurrency]
        batch = synthetic_chunk(model, seq_lens)
        base_ms = time_chunk(model, batch, steps=args.chunk_steps)
        n_tokens = sum(seq_lens)
        n_seqs, per_seq = len(seq_lens), seq_lens[0]
        ceil_ = prefill_ceiling(model.config, peak_tflops=args.peak_tflops)
        head = free_gib(model.device)
        chunk: Dict[str, Any] = {
            "n_seqs": n_seqs,
            "seq_lens": seq_lens,
            "tokens": n_tokens,
            "ms": round(base_ms, 2),
            "tok_s": round(n_tokens / (base_ms / 1e3), 1),
            "pct_of_ceiling": round(
                100.0 * (n_tokens / (base_ms / 1e3)) / ceil_["ceiling_tok_s"], 1
            ),
            "headroom_gib_after_capture": round(head, 2) if head is not None else None,
        }
        result["chunk"] = chunk
        emit(result)
        if head is not None and head < args.min_headroom_gib:
            print(
                f"  [profile] WARNING: only {head:.1f} GiB free after capture "
                f"(< --min-headroom-gib {args.min_headroom_gib}). The component "
                f"probes will mostly be skipped -- re-run with a smaller "
                f"--pool-max-num-seqs (the chunk composition does not depend on it).",
                flush=True,
            )

        # (name, need_gib, thunk). Ordered by value: the attribution table is
        # the primary output, so it runs first and flushes first.
        sections: List[Tuple[str, float, Callable[[], Dict]]] = [
            ("attribution", 2.0, lambda: chunk_attribution(
                model, batch, steps=max(args.chunk_steps // 2, 2), chunk_ms=base_ms)),
            ("ablations", 2.0, lambda: chunk_ablations(
                model, batch, base_ms, steps=args.chunk_steps)),
            ("kernels", 3.0, (lambda: {}) if args.no_profiler
             else (lambda: chunk_kernel_table(model, batch, top_n=args.top_n))),
            ("conv_layout_ab", 4.0, lambda: conv_layout_ab(
                model, batch, steps=args.chunk_steps)),
            ("conv_layout_parity", 3.0, lambda: conv_layout_parity(model, batch)),
            ("gemm_backend_sweep", 3.0, lambda: gemm_backend_sweep(
                model, batch, model.config, steps=args.chunk_steps,
                peak_tflops=args.peak_tflops,
                force_cache_owning=args.gemm_sweep_force_cache_owning)),
        ]
        if not args.no_ws_g2_probes:
            sections += [
                # 8 GiB: `--mlp-tile-tokens 8192` un-tiles the `[T, 2I]`
                # intermediate (570 MB) plus SwiGLU and the down-proj input,
                # with allocator churn on top. This is the most
                # memory-hungry probe, so its headroom is checked explicitly.
                ("mlp_tile_ab", 8.0, lambda: mlp_tile_ab(
                    model, batch, args.mlp_tiles, steps=args.chunk_steps)),
                # 6 GiB: fla materialises the per-chunk state `h` for all
                # `T/BT` chunks, so BT=16 asks for ~4x what BT=64 does.
                ("gdn_chunk_size_ab", 6.0, lambda: gdn_chunk_size_ab(
                    model, batch, args.gdn_chunk_sizes, steps=args.chunk_steps)),
                ("attn_prefill_probe", 3.0, lambda: attn_prefill_probe(model, batch)),
                # Deliberately *before* `decode_overlap_probe`: both want free
                # slots for decode rows and this is the more important one.
                ("mixed_vs_separate", 4.0, lambda: mixed_vs_separate(
                    comps, batch, batch_size=concurrency,
                    steps=max(args.chunk_steps // 2, 2))),
                # 8 GiB: it captures a whole mixed step
                # into a private graph pool on top of everything resident.
                ("mixed_graph_attrib", 8.0, lambda: mixed_graph_attrib(
                    comps,
                    chunk_tokens=(args.prefill_chunk_tokens
                                  or args.max_num_batched_tokens),
                    batch_size=concurrency,
                    n_segments=getattr(args, "mixed_graph_segments", 8),
                    prompt_len=args.input_len,
                    steps=max(args.chunk_steps // 2, 2))),
                ("decode_overlap_probe", 4.0, lambda: decode_overlap_probe(
                    comps, batch, batch_size=concurrency,
                    reps=max(args.chunk_steps // 2, 2))),
                ("chunk_budget_sweep", 4.0, lambda: chunk_budget_sweep(
                    model, args, args.chunk_budgets,
                    steps=max(args.chunk_steps // 2, 2))),
            ]
        for name, need, fn in sections:
            if name == "chunk_budget_sweep":
                _release_all_slots(model)
            chunk[name] = run_probe(
                name, fn, need_gib=need, device=model.device, verbose=True
            )
            emit(result)  # flush after every section

    # `del comps, sched, model` alone drops only some of the names holding the
    # engine: `driver`, `d2`/`sched2` and the closure in `restore` keep it
    # alive, so the next level would build on top of a resident 60 GiB engine
    # and OOM in `alloc_state_pool`. Clear the references explicitly.
    driver.records = []
    driver.sched = None  # type: ignore[assignment]
    sched.model = None  # type: ignore[assignment]
    batch = None  # the chunk's device tensors, if the component views ran
    del sched, driver, model, batch
    teardown(comps)
    del comps
    return result


# --------------------------------------------------------------------------- #
# 7. report
# --------------------------------------------------------------------------- #
def print_report(res: Dict) -> None:
    ph = res["physics"]
    print("\n=== prefill ceiling ===")
    print(
        f"  {ph['flops_per_token_total'] / 1e9:8.1f} GFLOP/token  "
        f"@ {ph['peak_tflops']:.0f} TFLOP/s dense fp8  ->  "
        f"{ph['ceiling_tok_s']:,.0f} prefill tok/s"
    )
    for lvl in res["levels"]:
        a = lvl["attribution"]
        print(f"\n=== concurrency {lvl['concurrency']} ===")
        print(
            f"  served {a['served_ms_per_output_token']:.1f} ms/output token   "
            f"({a['output_tok_s']:,.0f} out tok/s, {a['steps']} steps, "
            f"{lvl['requests_completed']} requests done)"
        )
        # The same drive on the real clock. `output_tok_s` above is
        # tokens / sum-of-step-walls and cannot see a gap between steps;
        # `step_busy_pct` is exactly that gap, measured from inside the process.
        print(
            f"  wall   {a.get('wall_output_tok_s', 0):,.0f} out tok/s over "
            f"{a.get('wall_s', 0):.0f} s, step-busy {a.get('step_busy_pct', 0):.1f}%"
        )
        print(f"  {'kind':<9} {'steps':>7} {'wall%':>7} {'ms/step':>9} {'tok/s':>10} {'ms/out tok':>11}")
        for kind, k in a["by_kind"].items():
            print(
                f"  {kind:<9} {k['steps']:>7} {k['wall_pct']:>6.1f}% "
                f"{k['ms_per_step_mean']:>9.2f} {k['tok_s']:>10,.0f} "
                f"{k['ms_per_output_token']:>11.2f}"
            )
        print("  host counters (ms total / calls / ms per call):")
        for name, c in list(lvl["host_counters_ms"].items())[:8]:
            print(f"    {name:<32} {c['total_ms']:>10.1f} {c['calls']:>8} {c['ms_per_call']:>9.3f}")
        gp = lvl.get("gil_probe")
        if gp:
            print(
                f"  GIL probe ({gp.get('duration_s', 0):.0f}s vs {a.get('wall_s', 0):.0f}s"
                f"{'' if gp.get('comparable', True) else ', NOT COMPARABLE'}):"
            )
            print(
                f"    step-wall basis {gp['output_tok_s_quiet']:,.0f} -> "
                f"{gp['output_tok_s_contended']:,.0f} out tok/s "
                f"({gp['slowdown_pct']:+.1f}%)"
            )
            print(
                f"    real clock      {gp.get('wall_output_tok_s_quiet', 0):,.0f} -> "
                f"{gp.get('wall_output_tok_s_contended', 0):,.0f} out tok/s "
                f"({gp.get('wall_slowdown_pct', 0):+.1f}%)   "
                f"step-busy {gp.get('step_busy_pct_quiet', 0):.1f}% -> "
                f"{gp.get('step_busy_pct_contended', 0):.1f}%"
            )
        ch = lvl.get("chunk")
        if ch:
            print(
                f"  chunk {ch['tokens']} tok ({ch['n_seqs']} seqs): {ch['ms']:.1f} ms = "
                f"{ch['tok_s']:,.0f} tok/s = {ch['pct_of_ceiling']:.0f}% of ceiling"
            )
            att = ch.get("attribution")
            if att:
                print("  --- per-component attribution (CUDA events, additive) ---")
                print(prefill_attrib.format_table(att))
            for name, view, fmt in (
                ("mlp tile", ch.get("mlp_tile_ab"), "chunk_ms"),
                ("gdn BT", ch.get("gdn_chunk_size_ab"), "chunk_ms"),
                ("chunk budget", ch.get("chunk_budget_sweep"), "chunk_ms"),
            ):
                if not isinstance(view, dict):
                    continue
                rows = [(k, v) for k, v in view.items() if isinstance(v, dict) and fmt in v]
                if not rows:
                    continue
                print(f"  --- {name} ---")
                for k, v in rows:
                    print(
                        f"    {k:<8} {v[fmt]:>9.1f} ms  "
                        f"{v.get('tok_s', v.get('chunk_tok_s', 0)):>9,.0f} tok/s"
                    )
            mx = ch.get("mixed_vs_separate") or {}
            if "mixed_ms" in mx:
                print(
                    f"  --- mixed forward (chunk {mx['chunk_tokens']} tok + "
                    f"{mx['decode_rows']} decode rows): separate "
                    f"{mx['chunk_ms']:.1f}+{mx['decode_ms']:.1f}="
                    f"{mx['separate_ms']:.1f} ms -> mixed {mx['mixed_ms']:.1f} ms "
                    f"({mx['speedup']:.2f}x, hid {mx['hidden_ms']:.1f} ms = "
                    f"{mx['hidden_pct_of_decode']:.0f}% of the decode step)"
                )
                print(
                    f"      projected out tok/s at this shape: "
                    f"{mx['projected_out_tok_s_separate']:,.0f} -> "
                    f"{mx['projected_out_tok_s_mixed']:,.0f}"
                )
            elif mx:
                print(f"  --- mixed forward: {mx.get('skipped') or mx.get('error')}")
            mg = ch.get("mixed_graph_attrib") or {}
            if "eager" in mg:
                e = mg["eager"]
                print(
                    f"  --- mixed step, eager (chunk {mg['chunk_tokens']} padded / "
                    f"{mg['real_prefill_tokens']} real tok + {mg['decode_rows']} rows -> "
                    f"bucket {mg['bucket']}): {e['step_ms']:.1f} ms wall, "
                    f"{e['host_issue_ms']:.1f} ms host launch, {e['device_ms']:.1f} ms device, "
                    f"{e['launches']:,} launches"
                )
                for g, v in list(e.get("groups", {}).items())[:6]:
                    print(f"        {g:<26} {v['launches']:>6,} launches  {v['us']:>9,.0f} us")
                for key, label, sfx in (
                    ("graphed_holes", "graphed, with eager holes", "_holes"),
                    ("graphed", "graphed, one graph", ""),
                ):
                    gr = mg.get(key)
                    if not gr:
                        err = mg.get(f"{key}_error") or mg.get(f"{key}_skipped")
                        if err:
                            print(f"      {label}: FAILED -- {err}")
                        continue
                    print(
                        f"      {label} ({gr.get('graph_segments', 0):.0f} segments, "
                        f"{gr.get('eager_holes', 0):.0f} eager holes, hole buffer "
                        f"{gr.get('hole_buffer_mib', 0):.0f} MiB, "
                        f"{gr.get('real_chunk_rows', 0):.0f}/"
                        f"{gr.get('chunk_index_rows', 0):.0f} chunk rows): "
                        f"{gr['step_ms']:.1f} ms wall, "
                        f"{gr['host_issue_ms']:.1f} ms host launch, {gr['device_ms']:.1f} ms "
                        f"device, {gr['launches']:,} launches"
                    )
                    print(
                        f"      -> {mg[f'speedup{sfx}']:.2f}x, "
                        f"{mg[f'saved_ms{sfx}']:.1f} ms saved, "
                        f"{mg[f'launches_removed{sfx}']:,} launches removed; projected out "
                        f"tok/s at this shape {mg['projected_out_tok_s_eager']:,.0f} -> "
                        f"{mg[f'projected_out_tok_s_graphed{sfx}']:,.0f}"
                    )
                    par = mg.get(f"parity_{key}")
                    if par:
                        print(
                            f"      -> vs eager: logits {par['max_abs_logits']:.2e} abs / "
                            f"{par['max_rel_logits']:.2e} rel, SSM state "
                            f"{par['max_abs_state']:.2e}, conv {par['max_abs_conv']:.2e}, "
                            f"argmax mismatches {par['argmax_mismatches']:.0f}/"
                            f"{par['argmax_rows']:.0f}"
                        )
                if mg.get("holes_cost_ms") is not None:
                    print(
                        f"      -> closing the 48 holes: "
                        f"{mg['holes_cost_ms']:.1f} ms wall, "
                        f"{mg['holes_cost_host_issue_ms']:.1f} ms host launch"
                    )
            elif mg:
                print(f"  --- mixed graphs: {mg.get('skipped') or mg.get('error')}")
            ov = ch.get("decode_overlap_probe") or {}
            if "hidden_ms" in ov:
                print(
                    f"  --- decode/prefill overlap (B={ov['bucket']}): serial "
                    f"{ov['serial_ms']:.1f} ms -> overlapped {ov['overlapped_ms']:.1f} ms, "
                    f"hid {ov['hidden_ms']:.1f} ms ({ov['hidden_pct_of_decode']:.0f}% of the "
                    f"decode step)"
                )
            ap = ch.get("attn_prefill_probe") or {}
            if "flashinfer_paged" in ap:
                fa = ap.get("fa3_ragged", {})
                print(
                    f"  --- attention prefill: paged "
                    f"{ap['flashinfer_paged'].get('us_per_layer', float('nan')):.0f} us/layer"
                    + (
                        f" vs FA3 ragged {fa['us_per_layer']:.0f} us/layer"
                        if "us_per_layer" in fa
                        else " (no FA3)"
                    )
                )
            for name, ab in ch["ablations"].items():
                print(f"    -{name:<20} {ab['delta_ms']:>8.1f} ms  ({ab['pct_of_chunk']:>4.1f}%)")
            for g, v in list(ch.get("kernels", {}).get("groups", {}).items())[:8]:
                print(f"    {g:<28} {v['us'] / 1e3:>8.1f} ms {v['pct']:>5.1f}%  {v['launches']} launches")
            ab = ch.get("conv_layout_ab", {})
            if "speedup" in ab:
                print(
                    f"    conv layout: channel_major {ab['channel_major']['chunk_ms']:.1f} ms -> "
                    f"token_major {ab['token_major']['chunk_ms']:.1f} ms "
                    f"({ab['speedup']:.2f}x, {ab['saved_ms_per_chunk']:.1f} ms/chunk)"
                )
            par = ch.get("conv_layout_parity", {})
            if "rel_l2" in par:
                print(f"    conv layout parity: relL2 {par['rel_l2']} argmax {par['argmax_agree']:.3f}")
            for name, v in ch.get("gemm_backend_sweep", {}).items():
                if "chunk_ms" in v:
                    print(
                        f"    gemm {name:<26} {v['chunk_ms']:>8.1f} ms  "
                        f"{v['chunk_tok_s']:>8,.0f} tok/s  {v['pct_of_peak']:>5.1f}% peak"
                    )


# --------------------------------------------------------------------------- #
# 8. cli
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True)
    p.add_argument("--concurrency", type=int, nargs="+", default=[32, 256])
    p.add_argument("--duration", type=float, default=60.0, help="measured seconds per level")
    p.add_argument("--warmup", type=float, default=8.0, help="unmeasured seconds per level")
    p.add_argument("--gil-duration", type=float, default=15.0)
    p.add_argument("--input-len", type=int, default=2139,
                   help="prompt length in tokens; the serving sweep's re-tokenised prompts are 2090-2195")
    p.add_argument("--output-len", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--page-size", type=int, default=16)
    p.add_argument("--ssm-state-dtype", default="fp16", choices=["fp32", "fp16"])
    p.add_argument("--kv-cache-dtype", default="bf16", choices=["bf16", "fp8"])
    p.add_argument("--norm-backend", default="triton", choices=["torch", "triton"])
    # The CLI default is `"torch"` while `preset.CANONICAL_FAST` and
    # `serve.M1_DEFAULTS` both say `"triton"`; without the preset the profiler
    # would measure a slower model than the server runs, which is the class of
    # drift `preset.py` exists to stop. `--preset fastest` fills this and
    # every other knob below from the one canonical table.
    p.add_argument("--fused-ops-backend", default="torch", choices=["torch", "triton"])
    p.add_argument("--attn-backend", default="auto")
    p.add_argument("--gdn-backend", default="auto")
    p.add_argument("--gemm-backend", default=None)
    p.add_argument("--gemm-accuracy", default="fast", choices=["fast", "strict"])
    p.add_argument("--gemm-cache-owner", default="decode", choices=["decode", "prefill"],
                   help="who claims the single repack-cache slot. "
                        "'prefill' resolves every linear at the real chunk M first.")
    p.add_argument("--sampler-candidates", type=int, default=2048)
    p.add_argument("--attn-workspace-mb", type=int, default=512)
    p.add_argument("--gdn-chunk-size", type=int, default=64,
                   help="fla/torch GDN chunk BT for prefill (fla accepts 16/32/64)")
    p.add_argument("--gemm-weight-cache", default="single", choices=["multi", "single", "none"])
    p.add_argument("--gemm-priority", default="v8", choices=["v9", "v8", "v7", "v4"],
                    help="GEMM cold-start priority table. 'v8' "
                         "(default) routes a real prefill chunk (M up to max-num-batched-tokens) "
                         "on its own measured bucket; 'v7' clamps every M>512 to the M=512 "
                         "decode answer (the rollback / before-after arm).")
    p.add_argument("--no-graphs", action="store_true")

    p.add_argument("--max-num-batched-tokens", type=int, default=8192)
    p.add_argument("--prefill-decode-ratio", type=int, default=4)
    p.add_argument("--prefill-chunk-tokens", type=int, default=0)
    p.add_argument("--mixed-forward", action="store_true", default=False,
                   help="one forward per step over [prefill chunk || "
                        "every running decode row]. Changes what the ledger measures "
                        "(steps come back labelled `mixed`); `mixed_vs_separate` prices "
                        "the same thing on one synthetic chunk with or without it.")
    p.add_argument("--mixed-graphs", action="store_true", default=False,
                   help="CUDA-graph the mixed step (padded to "
                        "(--prefill-chunk-tokens, decode bucket), replayed as ~49 "
                        "captured segments with one eager hole per GDN layer). Needs "
                        "--mixed-forward; `mixed_graph_attrib` prices it on one "
                        "synthetic step either way.")
    p.add_argument("--mixed-graph-segments", type=int, default=8,
                   help="prefill plan rows per graphed mixed step")
    p.add_argument("--mixed-graph-min-bucket", type=int, default=32,
                   help="smallest decode-row bucket a graphed mixed step is captured "
                        "for; 1 keeps the whole ladder")
    p.add_argument("--mixed-graph-buckets", type=int, nargs="+", default=None,
                   help="capture the mixed-step graph for exactly these "
                        "decode-row buckets, overriding the derived ladder and "
                        "--mixed-graph-min-bucket. `--overlap` implies `1`.")
    p.add_argument("--overlap", dest="overlap_streams", action="store_true",
                   default=False,
                   help="run the prefill chunk and the decode "
                        "step of one scheduler step on two CUDA streams instead "
                        "of fusing them into one row-concatenated forward. "
                        "Needs --mixed-forward --mixed-graphs.")
    p.add_argument("--overlap-min-fill", type=float, default=0.75,
                   help="below this fraction of --prefill-chunk-tokens an "
                        "overlapped step runs its prefill half eagerly rather "
                        "than padding to the graph shape")
    p.add_argument("--overlap-decode-priority", type=int, default=0,
                   help="CUDA stream priority for the decode half of an "
                        "overlapped step (0 = default, -1 = high).")
    p.add_argument("--mixed-graph-holes", action="store_true", default=False,
                   help="rollback: 49 graph segments + 48 eager holes instead "
                        "of ONE graph. The ledger arm; "
                        "`mixed_graph_attrib` measures both regardless of this flag.")
    p.add_argument("--mlp-tile-tokens", type=int, default=2048)
    p.add_argument("--conv-prefill-layout", default="token_major",
                   choices=["token_major", "channel_major"])
    p.add_argument("--conv-prefill-tile-tokens", type=int, default=2048)
    p.add_argument("--prefill-gemm-backend", default=None)

    # -- the memory plan ------------------------------------------------------ #
    p.add_argument("--gpu-memory-utilization", type=float, default=0.94,
                   help="fraction of *free* HBM the plan may use, as serve.py")
    p.add_argument("--pool-max-num-seqs", type=int, default=0,
                   help="size the KV/SSM pools for this many sequences instead of "
                        "--concurrency (0 = follow --concurrency). The chunk "
                        "composition does not depend on it -- an 8192-token chunk at "
                        "input-len 2139 is 4 sequences either way -- so an "
                        "attribution-only run should use 64 and keep ~47 GiB of "
                        "headroom. Requires --no-drive when below --concurrency, "
                        "because the closed-loop ledger really does need the slots.")
    p.add_argument("--min-headroom-gib", type=float, default=10.0,
                   help="warn when less than this is free after graph capture; "
                        "individual probes skip themselves against their own needs")
    p.add_argument("--gemm-sweep-force-cache-owning", action="store_true",
                   help="let gemm_backend_sweep run backends that own a weight-repack "
                        "cache even under --gemm-weight-cache single. This allocates a "
                        "SECOND ~23 GiB copy of every fp8 linear and can OOM the "
                        "device; only pass it with --gemm-weight-cache multi "
                        "and the headroom to match.")
    p.add_argument("--skip-memory-check", action="store_true",
                   help="print the plan but build anyway (may OOM)")
    p.add_argument("--quiet-plan", action="store_true")

    # -- prefill probes ------------------------------------------------------ #
    p.add_argument("--mlp-tiles", type=int, nargs="+", default=[2048, 4096, 8192],
                   help="FusedMLP.tile values to A/B at the real chunk")
    p.add_argument("--gdn-chunk-sizes", type=int, nargs="+", default=[64, 32, 16],
                   help="fla chunk BT values to A/B at the real chunk")
    p.add_argument("--chunk-budgets", type=int, nargs="+", default=[8192, 4096, 2048],
                   help="prefill chunk token budgets to time")
    p.add_argument("--no-ws-g2-probes", action="store_true",
                   help="skip the mlp-tile / gdn-BT / attention / overlap / chunk-size probes")

    p.add_argument("--peak-tflops", type=float, default=DEFAULT_PEAK_TFLOPS)
    p.add_argument("--chunk-steps", type=int, default=5)
    p.add_argument("--top-n", type=int, default=30)
    p.add_argument("--no-drive", action="store_true",
                   help="skip the closed-loop ledger + GIL probe; component views only "
                        "(what an ablation arm actually needs)")
    p.add_argument("--no-components", action="store_true")
    p.add_argument("--no-profiler", action="store_true")
    p.add_argument("--no-gil-probe", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--verbose", action="store_true")
    # The profile must build the *served* config, not a hand-kept list of
    # flags that drifts from it (e.g. `fused_ops_backend="torch"` against a
    # server that runs `"triton"`, or missing `--gemm-accuracy`,
    # `--gemm-cache-owner`, `--sampler-candidates` or `--attn-workspace-mb`). `--preset fastest` fills every knob left at its CLI default from
    # `preset.CANONICAL_FAST`, which `tests/test_preset.py` pins equal to
    # `serve.M1_DEFAULTS`.
    add_preset_arg(p)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args = apply_preset(args, parser, argv=argv)
    pool = int(args.pool_max_num_seqs or 0)
    if pool and not args.no_drive and pool < max(args.concurrency):
        parser.error(
            f"--pool-max-num-seqs {pool} is below --concurrency "
            f"{max(args.concurrency)}, so the closed-loop ledger would have "
            f"fewer slots than in-flight requests. Pass --no-drive (component "
            f"views only, which do not depend on the pool size) or raise it."
        )
    levels: List[Dict] = []

    def write(partial: Optional[Dict] = None) -> None:
        """Dump everything known so far to ``--out``.

        Called after every section of every level (see `run_level`'s `emit`),
        so the file on disk is always the most complete thing measured. Writes
        through a temp file and renames, because a run killed mid-`json.dump`
        would otherwise leave a truncated file that looks like data."""
        if not args.out:
            return
        cur = list(levels) + ([partial] if partial is not None else [])
        doc = {
            "schema": "qwenfast.profile_serving/1",
            "partial": partial is not None,
            "args": vars(args),
            "physics": (cur[0].get("_physics") if cur else {}) or {},
            "levels": cur,
        }
        tmp = args.out + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(doc, fh, indent=2, default=str)
        os.replace(tmp, args.out)

    for conc in args.concurrency:
        levels.append(run_level(args, conc, emit=write))
        write()
    # The ceiling only needs the config, and every level loaded the same one --
    # so take it off a level rather than paying a second 28 GiB load for it.
    physics = levels[0].pop("_physics") if levels else {}
    for lvl in levels[1:]:
        lvl.pop("_physics", None)

    res = {
        "schema": "qwenfast.profile_serving/1",
        "args": vars(args),
        "physics": physics,
        "levels": levels,
    }
    print_report(res)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=2, default=str)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
