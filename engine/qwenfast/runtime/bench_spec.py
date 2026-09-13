#!/usr/bin/env python
"""Speculative-decoding benchmark and real-model correctness gate.

Three things, in this order, so a failure of the cheap one never costs the
expensive one's GPU time:

1. **Correctness on the real model.**  ``--correct-prompts`` real prompts x
   ``--correct-tokens`` greedy tokens, decoded twice -- once with the plain
   decode step, once with ``SpecDecoder(k)`` -- and the two token streams are
   compared by the greedy-equivalence gate (see :func:`run_correctness`).  The
   tiny-model suite (``tests/test_spec_decode.py``) is the bit-identity
   assertion at commit granularity.
2. **Graph-timed decode sweep**, spec on/off, over ``--batch`` x ``--k`` at a real
   ``--ctx`` context, reporting ms/step, mean accepted tokens/step and effective
   tok/s.
3. **JSON + Markdown artefacts** of both.

Why real prompts.  ``bench_runtime.py`` fakes the context (reserve pages, set
``seq_len``) because plain decode's cost is a function of KV *traffic*, not KV
*content*.  That is exactly wrong here: the whole point of the measurement is the
**acceptance rate**, which is a function of what the model is actually saying.  So
this script really prefills ``--ctx`` tokens of ``evals/data/gsm8k_200.jsonl``
text per sequence before it times anything.

The "no spec" baseline is the fastest measured plain-decode configuration
-------------------------------------------------------------------------
Library defaults are not the fastest configuration: with
``RuntimeConfig.norm_backend`` at its ``"torch"`` default, B=1 measures
15.375 ms against 12.638 at ``triton`` (B=32: 22.629 vs 18.407), so a baseline
that inherits it is not comparable to the plain-decode benchmarks.  This script
therefore defaults to the fastest measured runtime config and prints it:

========================= ============ =====================================
knob                      default here rationale
========================= ============ =====================================
``--norm-backend``        ``triton``   -17.8% at B=1 / -18.7% at B=32
``--fused-ops-backend``   ``triton``   -256 launches/step, B=32 18.09 ms
                                       (3-repeat) vs 18.407 without
``--kv-cache-dtype``      ``bf16``     the default; fp8 is a memory lever
                                       only
``--ssm-state-dtype``     ``fp32``     **not** fp16 -- see below
graph buckets             == ``--batch`` a 512-bucket table allocates
                                       state-pool slots nothing touches
``--graphs``              on           4.3x at B=1
========================= ============ =====================================

The comparator is the plain-decode fp32-state column (12.651 / 14.880 /
19.488 / 32.863 ms at B = 1 / 8 / 32 / 128), and this bench should land at or
slightly under it; coming in *above* it means a knob regressed.

``--ssm-state-dtype`` is the one deliberate departure from the fastest config.
fp16 state is faster (a wash at B<=8, 5-17% from B=32 up), but the gate cannot
pass with it: the window kernel holds the running SSM state in fp32 registers
across all ``n`` positions and writes only ``S_m``, while plain decode rounds
the state to the pool dtype after *every* token.  fp32 makes that round trip
exact and the two paths bit-equivalent; fp16 does not, by construction.  Pass
``--ssm-state-dtype fp16`` for a throughput-oriented sweep and read the gate as
advisory there (``SpecDecoder`` warns).

Memory: this bench is planned by ``serve.plan_memory``, like the server
-------------------------------------------------------------------------
Without a plan, an over-budget geometry OOMs on an otherwise empty H200 inside
the gate's first prefill.  ``serve.plan_memory`` at ``B=256``, ``ctx 2048``,
``n_kv_pages=34065`` shows why, and the answer is arithmetic, not
fragmentation:

======================================= ======= =========================
term                                    GiB     why
======================================= ======= =========================
weights (fp8 + MTP)                      27.89  the checkpoint
**gemm repack caches x2**                46.74  ``gemm_weight_cache``
                                                ``"multi"``
KV pool, bf16, 34065 pages               35.35  256 seqs x ctx 2112
**SSM state, fp32, 257 slots**           36.14  144 MiB/slot
conv state + workspaces + buffers         2.50
--------------------------------------- ------- -------------------------
plan steady                             149.61  on a **139.80 GiB** card
======================================= ======= =========================

The *second* repack cache (the per-tensor fp8 one) is materialised lazily,
weight by weight, the first time a GEMM resolves at prefill's M-bucket 512,
i.e. inside the gate's first prefill.  So such an OOM reports a small
allocation failing, but the real cause is 23 GiB of duplicate weights arriving
under it.

This script therefore does what ``serve.py`` does:

* ``--gemm-weight-cache single`` by default (the server's pin), not
  ``RuntimeConfig``'s ``"multi"`` -- **-23.4 GiB**;
* the KV pool is sized from the bench's own span (``--ctx`` + the tokens it
  will actually generate), not from ``--max-model-len``;
* ``--batch`` defaults to ``1 8 32 128``. B=256 doubles *both* pools
  (+17.7 GiB KV, +18.1 GiB fp32 state) and does not fit beside the repack
  cache; run it as a separate invocation with ``--ssm-state-dtype fp16``;
* the plan is printed, compared against free HBM **before** the weights are
  loaded, and refused (exit 2) if it does not fit
  ``--gpu-memory-utilization`` of it; and
* ``torch.cuda.memory_allocated()`` is printed against the plan after load,
  after capture, after the gate and after each spec capture, so a plan that
  stops tracking reality is visible in the log rather than in an OOM.

The correctness gate is a **hard** gate: with ``--gate`` (the default) a failing
greedy-equivalence check exits before the sweep runs, so a broken build cannot
spend GPU time producing numbers nobody may quote.

    python -m qwenfast.runtime.bench_spec \\
        --model /path/to/Qwen3.8-27B-FP8 \\
        --out-json spec.json \\
        --out-md   spec.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence

import torch

from .bench_runtime import derive_pool_sizes
from .engine import build_engine
from .fused_model import RuntimeConfig
from .preset import add_preset_arg, apply_preset, format_resolved_config, resolved_config
from .scheduler import GenParams, Request, Scheduler
from .serve import (
    DEFAULT_MAX_MODEL_LEN,
    M1_DEFAULTS,
    arch_from_checkpoint,
    format_memory_plan,
    measure_allocation,
    plan_memory,
)
from .spec_decode import SpecConfig, SpecDecoder

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_DEFAULT_PROMPTS = os.path.join(_REPO_ROOT, "evals", "data", "gsm8k_200.jsonl")


# --------------------------------------------------------------------------- #
# prompts
# --------------------------------------------------------------------------- #
def load_texts(path: str, n: int) -> List[str]:
    out: List[str] = []
    with open(path) as f:
        for line in f:
            if len(out) >= n:
                break
            row = json.loads(line)
            out.append((row.get("question", "") + "\n" + row.get("full_solution", "")).strip())
    if not out:
        raise RuntimeError(f"no prompts in {path}")
    return out


def build_prompts(tokenizer, texts: Sequence[str], count: int, target_len: Optional[int]) -> List[List[int]]:
    """``count`` token-id prompts.

    ``target_len=None`` -> the natural prompt (used by the correctness pass).
    Otherwise every prompt is grown by concatenating further corpus items until
    it reaches ``target_len`` and then truncated -- real text, exact length, so
    the timed sweep's KV traffic is the advertised ``--ctx``.
    """
    enc = [tokenizer.encode(t, add_special_tokens=False) for t in texts]
    prompts: List[List[int]] = []
    for i in range(count):
        ids = list(enc[i % len(enc)])
        if target_len is not None:
            j = i
            while len(ids) < target_len:
                j += 1
                ids.extend(enc[j % len(enc)])
            ids = ids[:target_len]
        prompts.append(ids)
    return prompts


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def sweep_max_tokens(steps: int, warmup: int, k: int) -> int:
    """``max_tokens`` one timed cell must grant each sequence.

    ``+2`` steps and ``+8`` tokens of slack so no sequence retires inside the
    timed window; ``run_point`` asserts that none did. Also the bench's own
    generation budget, which is what the KV pool is sized for -- see
    :func:`main`.
    """
    return (steps + warmup + 2) * (k + 1) + 8


def _report_memory(plan: Dict, label: str, device: str, verbose: bool) -> Optional[Dict]:
    """Print what the device really holds against what the plan said.

    The server's discipline, applied to the offline bench: the plan is an
    estimate and graph capture is where the real number lands, so the two are
    printed side by side at every point where the footprint changes. A plan
    that does not track measurement is a bug in :func:`plan_memory`, and the
    only way to find that out is to print both.
    """
    measured = measure_allocation(device)
    if measured is None:
        return None
    if verbose:
        steady = plan["steady_gib"]
        err = (measured["in_use_gib"] - steady) / steady * 100.0 if steady else 0.0
        print(f"[bench_spec] {label}: in-use {measured['in_use_gib']:.2f} GiB "
              f"(torch.cuda.memory_allocated() {measured['allocated_gib']:.2f}, reserved "
              f"{measured['reserved_gib']:.2f}, free {measured['free_gib']:.2f}) "
              f"vs plan steady {steady:.2f} GiB -> {err:+.1f}%", file=sys.stderr)
    return measured


def _drain_prefill(sched: Scheduler, limit: int = 100000) -> None:
    """Run scheduler steps until every request is out of ``waiting``.

    **Prefill-only stepping**, and that is the whole point of the function.
    ``Scheduler.step`` interleaves ``prefill_decode_ratio`` (4) decode steps
    between prefill chunks, which is right for a *server* and wrong for a
    bench: filling B=128 sequences of ctx 2048 takes
    ``ceil(128*2048 / max_num_batched_tokens)`` chunks, so the first-admitted
    request would decode 4x that many times before the last one is even
    prefilled -- past its ``max_tokens``, at which point the scheduler retires
    it and the "B=128" cell times whatever is left.

    With interleaving, a B=128 "no spec" cell reports 23.83 ms/step at 881
    tok/s, i.e. **21 live sequences**, not 128 -- which is also why it comes
    out *faster* than B=32's 30.74 ms. Setting the ratio to 0 makes ``_should_prefill``
    true on every step while anything is waiting, so no token is generated
    until the batch is fully prefilled. :func:`run_point` then asserts the
    live count, so a regression of this kind fails loudly instead of quietly
    reporting a smaller batch's number under a larger batch's label.
    """
    saved = sched.prefill_decode_ratio
    sched.prefill_decode_ratio = 0
    try:
        guard = 0
        while sched.waiting and guard < limit:
            sched.step()
            guard += 1
    finally:
        sched.prefill_decode_ratio = saved
    if sched.waiting:
        raise RuntimeError("prefill did not drain -- pool too small for this batch/ctx")


def _release(sched: Scheduler, reqs: Sequence[Request]) -> None:
    for r in list(reqs):
        if r.slot is not None:
            sched._finish(r, "abort")
    sched.waiting.clear()
    sched.running.clear()


def run_point(
    comps,
    prompts: Sequence[Sequence[int]],
    batch: int,
    spec: Optional[SpecDecoder],
    *,
    steps: int,
    warmup: int,
) -> Dict:
    """One (batch, spec-config) cell: prefill for real, then time ``steps``
    decode steps and report what came out of them."""
    model, rt = comps.model, comps.rt
    sched = Scheduler(model, comps.decoder, rt, spec=spec)
    n = spec.n if spec is not None else 1
    max_tokens = sweep_max_tokens(steps, warmup, n - 1)

    reqs = [
        Request(
            request_id=f"b{batch}_{i}",
            prompt_token_ids=list(prompts[i % len(prompts)]),
            params=GenParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True),
        )
        for i in range(batch)
    ]
    for r in reqs:
        sched.add_request(r)

    t_prefill = time.perf_counter()
    _drain_prefill(sched)
    _sync(model.device)
    prefill_s = time.perf_counter() - t_prefill
    if len(sched.running) != batch:
        raise RuntimeError(
            f"B={batch}: only {len(sched.running)} sequences are live after prefill "
            f"-- this cell would measure a different batch than its label"
        )

    for _ in range(warmup):
        sched.step()
    _sync(model.device)
    live = len(sched.running)
    if live != batch:
        raise RuntimeError(
            f"B={batch}: {live} sequences live after warmup, expected {batch} -- "
            f"max_tokens too small for {steps}+{warmup} steps at n={n}"
        )

    before_tokens = sum(len(r.output_token_ids) for r in reqs)
    before_steps = spec.steps if spec is not None else 0
    before_emitted = spec.emitted_tokens if spec is not None else 0

    t0 = time.perf_counter()
    for _ in range(steps):
        sched.step()
    _sync(model.device)
    dt = time.perf_counter() - t0

    emitted = sum(len(r.output_token_ids) for r in reqs) - before_tokens
    live_after = len(sched.running)
    accept_len = (
        (spec.emitted_tokens - before_emitted) / max(spec.steps - before_steps, 1)
        if spec is not None
        else 1.0
    )
    mem = measure_allocation(str(model.device)) or {}
    _release(sched, reqs)

    if live_after != batch:
        raise RuntimeError(
            f"B={batch}: {live_after} sequences live after the timed window, expected "
            f"{batch} -- sequences retired mid-measurement"
        )
    return {
        "batch": batch,
        "spec": spec is not None,
        "k": (spec.k if spec is not None else 0),
        "ms_per_step": dt / steps * 1000.0,
        "tokens_per_step_per_seq": emitted / steps / batch,
        "accept_length": accept_len,
        "tok_s": emitted / dt,
        "prefill_s": prefill_s,
        # the label must be the batch that was actually timed.
        "live_seqs": live_after,
        "gib_in_use": round(mem.get("in_use_gib", 0.0), 2),
        "gib_allocated": round(mem.get("allocated_gib", 0.0), 2),
    }


def probe_gemm_backends(comps, batch: int, ks: Sequence[int], verbose: bool) -> Dict:
    """Which GEMM backend each path resolves.

    ``ResolvedLinear`` memoises per ``m_bucket(M)``.  If the verify pass keyed
    on ``M = bucket*(k+1)`` while the decode step it must reproduce keys on
    ``M = bucket``, then at ``bucket=32, k>=1`` that would cross
    ``gemm.dispatch``'s measured M=64 threshold and swaps 256 of the model's 305
    linears from ``vllm_marlin_fp8_w8a16`` to ``flashinfer_fp8_blockscale``,
    which is a 2.6e-2 relative difference, not a tiling difference.
    ``gemm_dispatch.rows_per_sequence`` prevents that.

    This records the *resolved* names for both paths in the artefact so a
    regression is visible in the JSON rather than only in a diverging token.
    """
    from ..gemm import dispatch as gemm_dispatch

    bucket = comps.decoder.bucket_for(batch)
    out: Dict = {"batch": batch, "bucket": bucket, "decode_m": bucket, "verify": {}}
    out["decode_backend"] = gemm_dispatch.priority_for_m(bucket)[0]
    for k in ks:
        n = k + 1
        with gemm_dispatch.rows_per_sequence(n):
            name = gemm_dispatch.priority_for_m(bucket * n)[0]
        out["verify"][str(k)] = {
            "rows": bucket * n,
            "backend": name,
            "matches_decode": name == out["decode_backend"],
        }
    if verbose:
        print(
            f"[bench_spec] GEMM routing at bucket={bucket}: decode M={bucket} -> "
            f"{out['decode_backend']}", file=sys.stderr,
        )
        for k, row in sorted(out["verify"].items()):
            flag = "OK" if row["matches_decode"] else "MISMATCH"
            print(
                f"[bench_spec]   verify k={k}: {row['rows']} rows -> {row['backend']} "
                f"[{flag}]", file=sys.stderr,
            )
    return out


def _greedy_run(
    comps,
    prompts: Sequence[Sequence[int]],
    spec: Optional[SpecDecoder],
    max_tokens: int,
    size: int,
) -> tuple:
    """Greedy-decode ``prompts`` in groups of ``size``.

    Returns ``(streams, emits)``.  ``streams[i]`` is prompt ``i``'s token list;
    ``emits[i]`` is the number of tokens each *step* contributed to it, i.e. the
    committed window prefix ``m`` for a speculative run and a list of 1s for the
    plain path.  ``emits`` is what turns "prompt 8 diverged at token 10" into
    "prompt 8 diverged at window position 0 of step 4, whose previous step
    rejected 2 drafts" -- the difference between a rollback bug and rounding.
    """
    model, rt = comps.model, comps.rt
    streams: List[List[int]] = []
    emits: List[List[int]] = []
    for start in range(0, len(prompts), size):
        chunk = prompts[start : start + size]
        sched = Scheduler(model, comps.decoder, rt, spec=spec)
        reqs = [
            Request(
                request_id=f"c{start + i}",
                prompt_token_ids=list(p),
                params=GenParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True),
            )
            for i, p in enumerate(chunk)
        ]
        per_req: Dict[str, List[int]] = {r.request_id: [] for r in reqs}
        for r in reqs:
            sched.add_request(r)
        guard = 0
        while sched.has_work() and guard < 100000:
            for ev in sched.step():
                if ev.new_token_ids:
                    per_req[ev.request.request_id].append(len(ev.new_token_ids))
            guard += 1
        streams.extend(list(r.output_token_ids) for r in reqs)
        emits.extend(per_req[r.request_id] for r in reqs)
        _release(sched, reqs)
    return streams, emits


def _first_divergence(a: Sequence[int], b: Sequence[int]) -> Optional[int]:
    for j, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return j
    return None if len(a) == len(b) else min(len(a), len(b))


def _locate(emits: Sequence[int], j: int, n: int) -> Dict:
    """Where does emitted-token index ``j`` sit in the step/window structure?

    A **rollback** bug puts every divergence at ``offset_in_window == 0`` of a
    step whose predecessor rejected something; **rounding** scatters them
    uniformly across offsets.  Both are recorded so the artefact answers the
    question without another window.
    """
    cum = 0
    for step, m in enumerate(emits):
        if cum + m > j:
            return {
                "step": step,
                "offset_in_window": j - cum,
                "m_at_step": int(m),
                "step_rejected": bool(m < n),
                "prev_step_rejected": bool(step > 0 and emits[step - 1] < n),
            }
        cum += m
    return {"step": -1, "offset_in_window": -1, "m_at_step": -1,
            "step_rejected": False, "prev_step_rejected": False}


def probe_logit_parity(
    comps,
    contexts_by_offset: Dict[int, List[List[int]]],
    *,
    buckets,
    verbose: bool,
) -> Dict:
    """**The decisive measurement**: plain decode's logits vs the
    verify window's logits, for the *same* context and the *same* state.

    ``SpecConfig(k=0)`` is the control that makes this possible -- a
    one-position window, no drafts, ``m`` always 1, one token per step, i.e.
    exactly the plain decode step's *contract* run through the verify path's
    *kernels* (``FusedGDN.window`` + the torch conv, ``SpecAttentionRunner
    .window`` on ``BatchPrefillWithPagedKVCacheWrapper``, packed-window GEMM
    shapes).  Anything this probe reports is therefore a property of the
    verify path alone, with speculation removed as a variable.

    For each offset it reports, per row: the reference top-1 id, the window
    top-1 id, ``max|dlogit|`` over the vocabulary, and the reference's own
    **top-2 margin**.  A divergence whose margin is below ``max|dlogit|`` is a
    coin-flip the arithmetic cannot be expected to call the same way twice; a
    divergence at a comfortable margin is a bug.  That distinction is the whole
    content of the gate's path-parity check.
    """
    model, rt = comps.model, comps.rt
    spec = SpecDecoder(
        model, comps.buf, rt,
        SpecConfig(k=0, buckets=buckets, keep_window_logits=True),
    )
    spec.warmup()
    if rt.use_cuda_graphs:
        spec.capture(pool_handle=comps.decoder._pool)

    def one_pass(ctxs: List[List[int]], sp: Optional[SpecDecoder]):
        sched = Scheduler(model, comps.decoder, rt, spec=sp)
        reqs = [
            Request(
                request_id=f"p{i}",
                prompt_token_ids=list(c),
                params=GenParams(temperature=0.0, max_tokens=2, ignore_eos=True),
            )
            for i, c in enumerate(ctxs)
        ]
        for r in reqs:
            sched.add_request(r)
        _drain_prefill(sched)
        # `_run_decode_step`/`_run_spec_decode_step` both build their row order
        # as `list(self.running.values())`, so this is the row->request map.
        order = list(sched.running.values())
        sched.step()
        b = len(order)
        if sp is None:
            lg = comps.buf.logits[:b].detach().clone()
        else:
            lg = spec.window_logits[:b].detach().clone()  # n == 1
        idx = [int(r.request_id[1:]) for r in order]
        _release(sched, reqs)
        return lg, idx

    rows: List[Dict] = []
    summary: List[Dict] = []
    for off in sorted(contexts_by_offset):
        ctxs = contexts_by_offset[off]
        l_dec, idx_dec = one_pass(ctxs, None)
        l_win, idx_win = one_pass(ctxs, spec)
        # re-order both to prompt order
        inv_d = torch.as_tensor(idx_dec, dtype=torch.long).argsort()
        inv_w = torch.as_tensor(idx_win, dtype=torch.long).argsort()
        a = l_dec[inv_d.to(l_dec.device)].float()
        b = l_win[inv_w.to(l_win.device)].float()
        d = (b - a).abs()
        top2 = a.topk(2, dim=-1)
        margin = (top2.values[:, 0] - top2.values[:, 1])
        t1a, t1b = a.argmax(-1), b.argmax(-1)
        agree = (t1a == t1b)
        for i in range(a.shape[0]):
            rows.append({
                "offset": off,
                "prompt": i,
                "top1_decode": int(t1a[i]),
                "top1_window": int(t1b[i]),
                "agree": bool(agree[i]),
                "max_abs_dlogit": float(d[i].max()),
                "top2_margin": float(margin[i]),
                "rel_l2": float((b[i] - a[i]).norm() / a[i].norm().clamp_min(1e-30)),
            })
        summary.append({
            "offset": off,
            "rows": int(a.shape[0]),
            "top1_agreement": float(agree.float().mean()),
            "max_abs_dlogit": float(d.max()),
            "mean_abs_dlogit": float(d.mean()),
            "rel_l2": float((b - a).norm() / a.norm().clamp_min(1e-30)),
            "median_top2_margin": float(margin.median()),
            "min_top2_margin_at_disagreement": (
                float(margin[~agree].min()) if (~agree).any() else None
            ),
            "max_top2_margin_at_disagreement": (
                float(margin[~agree].max()) if (~agree).any() else None
            ),
        })
        if verbose:
            s = summary[-1]
            print(f"[bench_spec] logit parity @+{off:4d}: top1 agree "
                  f"{s['top1_agreement']*100:5.1f}%  max|dlogit|={s['max_abs_dlogit']:.3e}  "
                  f"relL2={s['rel_l2']:.3e}  median top-2 margin={s['median_top2_margin']:.3f}"
                  f"  worst disagreeing margin={s['max_top2_margin_at_disagreement']}",
                  file=sys.stderr)
    del spec
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    n_rows = len(rows) or 1
    n_agree = sum(1 for r in rows if r["agree"])
    return {
        "per_offset": summary,
        "rows": rows,
        "n_rows": len(rows),
        "top1_agreement": n_agree / n_rows,
        "max_abs_dlogit": max((r["max_abs_dlogit"] for r in rows), default=0.0),
        "max_disagreeing_margin": max(
            (r["top2_margin"] for r in rows if not r["agree"]), default=0.0
        ),
    }


#: how many top-2-margin units of slack a disagreement is allowed before it
#: stops being attributable to the verify path's own arithmetic noise.  The
#: probe measures ``max|dlogit|`` directly; a coin-flip needs the two candidates
#: to be within that of each other, and 2x is the slack for the fact that the
#: probe samples a few dozen rows and the stream visits thousands.
MARGIN_SLACK = 2.0
#: extra prompts a speculative run is allowed to diverge on, beyond the k=0
#: control's own count, before speculation itself is implicated.
SPEC_DIVERGENCE_SLACK = 2
#: the fraction of probed rows whose top-1 may disagree between the decode and
#: the verify forward before the *path* is considered broken rather than noisy.
#: A 1e-4 relative logit perturbation measures a 1e-3 argmax-flip rate over
#: this 248,320-token vocabulary; 1e-2 is an order of magnitude of headroom
#: above that floor, and a real bug (a wrong position, a stale state, per-row
#: GEMM routing to flashinfer_fp8_blockscale) is 10-100x over it.
PATH_DISAGREEMENT_MAX = 0.01


def run_correctness(
    comps,
    prompts: Sequence[Sequence[int]],
    ks: Sequence[int],
    max_tokens: int,
    verbose: bool,
    *,
    group: int = 0,
    buckets=(),
    diagnose: bool = True,
    max_probe_offsets: int = 24,
) -> Dict:
    """The speculative greedy-equivalence gate.

    ``group`` bounds how many prompts are in flight at once. ``0`` = all of
    them, which is the strongest geometry: 20 prompts
    round to graph bucket **32**, and bucket 32 is where the decode step
    resolves ``vllm_marlin_fp8_w8a16``.  A smaller group moves the gate to a
    smaller bucket and *weakens* it, so it is an explicit memory escape hatch,
    not a default.

    Why this is no longer a bit-identity check
    ------------------------------------------
    Even with the GEMM routing pinned (decode and verify resolve
    ``vllm_marlin_fp8_w8a16`` at every k), 8/20, 8/20, 9/20 prompts diverge
    on the real model, with **identical** first-divergence indices for k=1
    and k=2.  Measuring every place the verify forward calls a different
    kernel than the decode forward, at the real model's shapes:

    ======================================== ============ =================
    pair                                     relL2        n-dependent?
    ======================================== ============ =================
    conv ``conv_update`` vs ``verify_and_commit``  2.3e-6  no (identical at
                                                            n=2,3,4)
    GDN ``decode_step`` vs ``window``              1.9e-5  no (identical at
                                                            n=2,3,4)
    marlin M=32 vs M=64 / M=96                     0        bit-identical
    marlin M=32 vs M=128                           6.5e-5  yes -- and only
                                                            k=3 reaches it
    triton norm / swiglu / gdn-gate 32 vs 128 rows 0        bit-identical
    ======================================== ============ =================

    Three of those pairs are *unavoidable*: a forward that consumes ``n``
    tokens per sequence in one pass cannot be the same kernel as one that
    consumes a single token, and a 128-row GEMM cannot be tiled like a 32-row
    one.  There is no configuration of this engine in which the speculative
    verify pass is bit-identical to the plain decode step, so a gate that
    demands it is testing something the design cannot deliver -- and, worse,
    would keep failing after every real bug is gone.  vLLM's and SGLang's
    speculative decoders are not bit-identical to their own base decoders
    either; the standard they hold is distributional equivalence under greedy
    sampling.

    So the gate has four parts, and each one fails on a *bug*, not on
    arithmetic:

    1. **Routing** (hard) -- decode and verify must resolve the same
       GEMM backend at every k.  ``probe_gemm_backends`` records it.
    2. **Reference self-consistency** (hard) -- two identical plain-greedy
       runs must be token-identical, and a plain-greedy run at a *different
       graph bucket* is reported alongside.  If the reference is not itself
       reproducible there is nothing to compare against.
    3. **Path parity at the logit level** (hard) -- ``probe_logit_parity``
       holds the plain decode forward next to a ``k=0`` verify window over the
       same context and state.  Top-1 must agree on at least
       ``1 - PATH_DISAGREEMENT_MAX`` of probed rows, and **every** disagreement
       must sit at a top-2 margin below ``MARGIN_SLACK * max|dlogit|``: a
       flipped argmax at a comfortable margin is a bug, a flipped argmax inside
       the noise is not.
    4. **Speculation adds nothing** (hard) -- for every ``k``, the number
       of prompts whose stream diverges must not exceed the ``k=0`` control's
       count by more than ``SPEC_DIVERGENCE_SLACK``.  The k=0 control runs the
       identical verify path with the drafting removed, so any excess is
       attributable to drafting/acceptance/rollback, which *is* the
       speculative decoder's own code.
    """
    size = group if group and group > 0 else len(prompts)
    result: Dict = {
        "n_prompts": len(prompts),
        "max_tokens": max_tokens,
        "group": size,
        "decode_bucket": comps.decoder.bucket_for(min(size, len(prompts))),
        "gate_version": 2,
        "per_k": {},
        "controls": {},
    }

    ref, _ = _greedy_run(comps, prompts, None, max_tokens, size)

    # -- control 1: is the reference reproducible at all? -------------------- #
    ref2, _ = _greedy_run(comps, prompts, None, max_tokens, size)
    det = ref == ref2
    # -- control 2: is it *bucket*-invariant?  A plain decode step at bucket 8
    #    tiles every GEMM differently than one at bucket 32, so if the two
    #    disagree the "reference" is a function of the batch it was decoded in
    #    and bit-identity was never a well-posed gate.
    alt_size = 8 if size > 8 else max(1, size // 2) or 1
    ref_alt, _ = _greedy_run(comps, prompts, None, max_tokens, alt_size)
    alt_div = [
        {"prompt": i, "first_divergence": _first_divergence(a, b)}
        for i, (a, b) in enumerate(zip(ref, ref_alt))
        if a != b
    ]
    result["controls"] = {
        "reference_deterministic": det,
        "reference_bucket": comps.decoder.bucket_for(min(size, len(prompts))),
        "reference_alt_group": alt_size,
        "reference_alt_bucket": comps.decoder.bucket_for(alt_size),
        "reference_bucket_invariant": not alt_div,
        "reference_bucket_divergences": alt_div[:10],
        "n_reference_bucket_divergences": len(alt_div),
    }
    if verbose:
        print(f"[bench_spec] control: reference reproducible = {det}; "
              f"bucket-invariant (group {size} vs {alt_size}) = {not alt_div} "
              f"({len(alt_div)}/{len(ref)} prompts differ)", file=sys.stderr)

    # -- the runs: k=0 (the path control) first, then the real ks ------------ #
    all_ks = [0] + [k for k in ks if k != 0]
    for k in all_ks:
        spec = SpecDecoder(comps.model, comps.buf, comps.rt,
                           SpecConfig(k=k, buckets=tuple(buckets) or None))
        spec.warmup()
        if comps.rt.use_cuda_graphs:
            spec.capture(pool_handle=comps.decoder._pool)
        got, emits = _greedy_run(comps, prompts, spec, max_tokens, size)
        n = k + 1
        detail = []
        for i, (a, b) in enumerate(zip(ref, got)):
            if a == b:
                continue
            j = _first_divergence(a, b)
            row = {"prompt": i, "first_divergence": j,
                   "ref_token": a[j] if j is not None and j < len(a) else None,
                   "spec_token": b[j] if j is not None and j < len(b) else None}
            row.update(_locate(emits[i], j, n) if j is not None else {})
            detail.append(row)
        result["per_k"][str(k)] = {
            "identical": not detail and all(len(g) == max_tokens for g in got),
            "mismatched_prompts": len(detail),
            "detail": detail,
            "accept_length": spec.stats()["spec_accept_length"],
            "is_path_control": k == 0,
        }
        if verbose:
            label = "k=0 CONTROL (window path, no drafting)" if k == 0 else f"k={k}"
            print(f"[bench_spec] {label}: {len(detail)}/{len(ref)} prompts diverge "
                  f"from plain greedy, accept_len="
                  f"{spec.stats()['spec_accept_length']:.2f}", file=sys.stderr)
            for row in detail[:5]:
                print(f"[bench_spec]     prompt {row['prompt']:3d} @tok "
                      f"{row['first_divergence']:4d}  step {row.get('step')} "
                      f"offset_in_window {row.get('offset_in_window')} "
                      f"m={row.get('m_at_step')} "
                      f"prev_step_rejected={row.get('prev_step_rejected')}",
                      file=sys.stderr)
        del spec
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- the logit-level probe, at the offsets the streams actually broke at - #
    probe: Dict = {}
    if diagnose:
        offsets = sorted({
            int(row["first_divergence"])
            for kk in result["per_k"].values()
            for row in kk["detail"]
            if row.get("first_divergence") is not None
        } | {0, 8, 32, 96})
        offsets = [o for o in offsets if 0 <= o < max_tokens][:max_probe_offsets]
        contexts = {
            o: [list(p) + list(r[:o]) for p, r in zip(prompts, ref)] for o in offsets
        }
        probe = probe_logit_parity(
            comps, contexts, buckets=tuple(buckets) or None, verbose=verbose
        )
    result["logit_parity"] = probe

    # -- the verdict --------------------------------------------------------- #
    delta = probe.get("max_abs_dlogit", 0.0)
    margin_bound = MARGIN_SLACK * delta
    k0 = result["per_k"].get("0", {}).get("mismatched_prompts", 0)
    path_ok = True
    if probe:
        path_ok = (
            probe["top1_agreement"] >= 1.0 - PATH_DISAGREEMENT_MAX
            and probe["max_disagreeing_margin"] <= margin_bound
        )
    for k in all_ks:
        row = result["per_k"][str(k)]
        excess = row["mismatched_prompts"] - k0
        row["excess_over_path_control"] = excess
        row["ok"] = (
            row["identical"]
            or (k == 0 and path_ok)
            or (excess <= SPEC_DIVERGENCE_SLACK and path_ok)
        )
    result["path_parity_ok"] = path_ok
    result["margin_bound"] = margin_bound
    result["ok"] = bool(
        det
        and path_ok
        and all(result["per_k"][str(k)]["ok"] for k in all_ks)
    )
    if verbose:
        print(f"[bench_spec] gate: reference_deterministic={det} "
              f"path_parity_ok={path_ok} (top-1 agreement "
              f"{probe.get('top1_agreement', float('nan')):.4f}, max|dlogit|="
              f"{delta:.3e}, worst disagreeing margin="
              f"{probe.get('max_disagreeing_margin', 0.0):.3e} vs bound "
              f"{margin_bound:.3e}) -> {'PASS' if result['ok'] else 'FAIL'}",
              file=sys.stderr)
    return result


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def to_markdown(results: Dict) -> str:
    cfg = results["config"]
    lines = [
        "# spec: MTP speculative decoding",
        "",
        f"model: `{cfg['model']}` · ctx {cfg['ctx']} · "
        f"ssm_state {cfg['ssm_state_dtype']} · kv {cfg['kv_cache_dtype']} · "
        f"norm {cfg.get('norm_backend', '?')} · fused_ops {cfg.get('fused_ops_backend', '?')} · "
        f"graphs {cfg['use_cuda_graphs']} · "
        f"gemm_weight_cache {cfg.get('gemm_weight_cache', '?')} · "
        f"gemm_accuracy {cfg.get('gemm_accuracy', '?')} · {cfg['timestamp']}",
        "",]
    mem = results.get("memory") or {}
    if mem.get("plan_gib"):
        pl, ms = mem["plan_gib"], mem.get("measured_after_capture_gib") or {}
        lines += [
            "## 0a. Memory (serve.plan_memory)",
            "",
            "| term | GiB |",
            "|---|---|",
            f"| weights (fp8 + MTP) | {pl.get('weights_gib', 0):.2f} |",
            f"| gemm repack caches ({int(pl.get('n_repack_caches', 0))}x) "
            f"| {pl.get('repack_gib', 0):.2f} |",
            f"| KV pool ({cfg.get('n_kv_pages', '?')} pages) | {pl.get('kv_gib', 0):.2f} |",
            f"| SSM state ({cfg.get('max_num_seqs', '?')}+1 slots) | {pl.get('ssm_gib', 0):.2f} |",
            f"| conv state | {pl.get('conv_gib', 0):.2f} |",
            f"| spec window cache | {pl.get('spec_gib', 0):.2f} |",
            f"| workspaces + buffers + graph pool + ctx | "
            f"{pl.get('workspace_gib', 0) + pl.get('buffers_gib', 0) + pl.get('graph_gib', 0) + pl.get('context_gib', 0):.2f} |",
            f"| **plan steady** | **{pl.get('steady_gib', 0):.2f}** |",
            f"| prefill headroom (--max-num-batched-tokens "
            f"{cfg.get('max_num_batched_tokens', '?')}) | {pl.get('prefill_gib', 0):.2f} |",
            f"| **measured in-use after capture** | "
            f"**{ms.get('in_use_gib', float('nan')):.2f}** |",
            "",
        ]
    lines += [
        "## 0. GEMM routing (decode vs verify)",
        "",
    ]
    gb = results.get("gemm_backends") or {}
    if gb:
        lines += [
            f"bucket {gb['bucket']} · decode M={gb['decode_m']} → `{gb['decode_backend']}`",
            "",
            "| k | verify rows | verify backend | matches decode |",
            "|---|---|---|---|",
        ]
        for k, row in sorted(gb.get("verify", {}).items()):
            lines.append(
                f"| {k} | {row['rows']} | `{row['backend']}` | "
                f"{'**yes**' if row['matches_decode'] else '**NO — regression**'} |"
            )
        lines.append("")
    lines += [
        "## 1. Correctness gate",
        "",
        "Not bit-identity: a one-token decode kernel and an `n`-token verify",
        "kernel are not two orderings of the same arithmetic, so bit-identity is",
        "unreachable by construction. The",
        "verdict is *routing* + *reference self-consistency* + *logit-level path",
        "parity* + *speculation adds nothing over the k=0 control*.",
        "",
        "| k | verdict | prompts diverged from plain greedy | mean accepted tokens/step |",
        "|---|---|---|---|",
    ]
    corr = results.get("correctness", {})
    for k, row in sorted(corr.get("per_k", {}).items(), key=lambda kv: int(kv[0])):
        label = "0 (path control)" if k == "0" else k
        lines.append(
            f"| {label} | {'**PASS**' if row.get('ok') else '**FAIL**'} | "
            f"{row['mismatched_prompts']}/{corr['n_prompts']} | {row['accept_length']:.2f} |"
        )
    lines += [
        "",
        f"({corr.get('n_prompts', '?')} real prompts x {corr.get('max_tokens', '?')} greedy tokens.)",
        "",
        "`k = 0` is the **path control**: a one-position verify",
        "window, no drafting, one token per step -- the plain decode step's",
        "contract run through the verify path's kernels. Its divergence count is",
        "the floor every k >= 1 is measured against, because it is what the",
        "window/conv/attention kernel *pairs* cost on their own.",
        "",
        "### 1a. Controls",
        "",
        "| control | result |",
        "|---|---|",
    ]
    ctl = corr.get("controls", {})
    lines += [
        f"| plain greedy is reproducible (run twice) | "
        f"{'yes' if ctl.get('reference_deterministic') else '**NO**'} |",
        f"| plain greedy is bucket-invariant (bucket "
        f"{ctl.get('reference_bucket', '?')} vs {ctl.get('reference_alt_bucket', '?')}) | "
        f"{'yes' if ctl.get('reference_bucket_invariant') else 'no'} "
        f"({ctl.get('n_reference_bucket_divergences', '?')}/{corr.get('n_prompts', '?')} prompts) |",
    ]
    lp = corr.get("logit_parity") or {}
    if lp:
        lines += [
            f"| decode-vs-window top-1 agreement ({lp.get('n_rows', 0)} probed rows) | "
            f"{lp.get('top1_agreement', 0) * 100:.2f}% |",
            f"| max abs logit delta, decode vs window | "
            f"{lp.get('max_abs_dlogit', 0):.3e} |",
            f"| worst top-2 margin at a disagreement (bound "
            f"{corr.get('margin_bound', 0):.3e}) | "
            f"{lp.get('max_disagreeing_margin', 0):.3e} |",
        ]
        lines += [
            "",
            "### 1b. Logit parity by offset",
            "",
            "Plain decode forward vs a `k=0` verify window over the **same context**",
            "and the **same state**. `top-2 margin` is the reference's own gap between",
            "its best and second-best token: a flip whose margin is under `max|dlogit|`",
            "is a coin toss two different kernels cannot be expected to call alike.",
            "",
            "| offset | rows | top-1 agree | max abs dlogit | relL2 | "
            "median top-2 margin | worst disagreeing margin |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in lp.get("per_offset", []):
            worst = row.get("max_top2_margin_at_disagreement")
            lines.append(
                f"| +{row['offset']} | {row['rows']} | {row['top1_agreement'] * 100:.1f}% | "
                f"{row['max_abs_dlogit']:.3e} | {row['rel_l2']:.3e} | "
                f"{row['median_top2_margin']:.3f} | "
                f"{'-' if worst is None else f'{worst:.3e}'} |"
            )
    lines += [
        "",
        "## 2. Decode sweep (graph-timed, real prefilled context)",
        "",
        "| B | live | mode | ms/step | accepted tok/step | tok/s | vs no-spec | GiB in use |",
        "|---|---|---|---|---|---|---|---|",
    ]
    base = {r["batch"]: r for r in results["sweep"] if not r["spec"]}
    for r in results["sweep"]:
        mode = f"spec k={r['k']}" if r["spec"] else "no spec"
        b = base.get(r["batch"])
        rel = f"{r['tok_s'] / b['tok_s']:.2f}x" if b and b["tok_s"] else "-"
        live = r.get("live_seqs", r["batch"])
        lines.append(
            f"| {r['batch']} | {live} | {mode} | {r['ms_per_step']:.2f} | "
            f"{r['accept_length']:.2f} | {r['tok_s']:.0f} | {rel} | "
            f"{r.get('gib_in_use', 0):.1f} |"
        )
    lines += [
        "",
        "`accepted tok/step` is the mean of `m` (the committed window prefix, 1..k+1),",
        "i.e. `spec_accept_length`. Breakeven is 1.87.",
        "",
        "`live` is the number of sequences actually decoding during the timed window,",
        "and it must equal `B`. If the prefill drain interleaved decode steps, the",
        "first-admitted sequences could retire before timing began; `run_point` raises",
        "instead of reporting a smaller batch under this label.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", default=None, help="defaults to --model")
    p.add_argument("--prompts-file", default=_DEFAULT_PROMPTS)
    p.add_argument("--batch", type=int, nargs="+", default=[1, 8, 32, 128],
                   help="B=256 is deliberately NOT here: it doubles both pools "
                        "(KV 35.4 GiB, fp32 state 36.1 GiB) and the plan does not fit "
                        "an H200 beside the marlin repack cache. Run it as a separate, "
                        "memory-guarded invocation (for example with "
                        "--ssm-state-dtype fp16).")
    p.add_argument("--k", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--ctx", type=int, default=2048)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--correct-prompts", type=int, default=20)
    p.add_argument("--correct-tokens", type=int, default=128)
    p.add_argument("--correct-batch", type=int, default=0,
                   help="graph bucket the gate decodes at (0 = --correct-prompts). "
                        "The bucket matters: it is what sets every linear's M.")
    p.add_argument("--correct-group", type=int, default=0,
                   help="prompts in flight at once during the gate (0 = all). Bounds the "
                        "prefill transients; note that a group < --correct-prompts moves "
                        "the gate to a smaller graph bucket than the default 32.")
    p.add_argument("--skip-correctness", action="store_true")
    p.add_argument("--no-diagnose", action="store_true",
                   help="skip the logit-parity probe and the k=0 path control "
                        "The probe is what turns 'the streams "
                        "differ' into 'they differ by 3e-3 of a logit at a "
                        "top-2 margin of 1e-3', which is the only way to tell "
                        "arithmetic noise from a bug -- so it is on by default "
                        "and this flag exists for a memory-starved rerun.")
    p.add_argument("--gate", dest="gate", action="store_true", default=True,
                   help="stop before the timed sweep if the correctness gate fails (default)")
    p.add_argument("--no-gate", dest="gate", action="store_false",
                   help="run the sweep even if the gate fails (diagnostic only)")
    p.add_argument("--ssm-state-dtype", default="fp32", choices=["fp32", "fp16"],
                   help="fp32 (default) is the only setting the bit-identical gate can pass "
                        "-- see the module docstring")
    p.add_argument("--kv-cache-dtype", default="bf16", choices=["bf16", "fp8"])
    p.add_argument("--norm-backend", default="triton", choices=["torch", "triton"],
                   help="triton is -17.8%% at B=1. Default triton so the "
                        "'no spec' baseline matches the fastest measured config.")
    p.add_argument("--fused-ops-backend", default="triton", choices=["torch", "triton"],
                   help="Default triton: the fused SwiGLU + GDN gate "
                        "epilogue are -255.9 launches/step at B=1 and -255.7 at B=128, and "
                        "a 3-repeat run measures 18.09 ms at B=32 "
                        "against 18.407 without them for the same shape. Safe for the "
                        "bit-identical gate: both kernels are row-independent elementwise "
                        "fusions whose BLOCK is keyed on the *feature* dim (inter / h), "
                        "never on M, so -- unlike the GEMM dispatcher -- "
                        "decode at M=B and verify at M=B(k+1) run the same arithmetic.")
    # -- memory geometry: the same knobs, defaults and planner as serve.py --- #
    p.add_argument("--gemm-weight-cache", default=M1_DEFAULTS["gemm_weight_cache"],
                   choices=["multi", "single", "none"],
                   help="'multi' (RuntimeConfig's dataclass default) lets marlin AND the "
                        "per-tensor fp8 backend each memoise a full repacked copy of the "
                        "23.0 GiB of fp8 linears -- +23.4 GiB, built lazily during the "
                        "gate's first prefill, which is where an over-budget run OOMs. "
                        "'single' is what the server pins.")
    p.add_argument("--gemm-priority", default=M1_DEFAULTS["gemm_priority"], choices=["v9", "v8", "v7", "v4"],
                   help="GEMM cold-start priority table.")
    add_preset_arg(p)
    p.add_argument("--gemm-accuracy", default=M1_DEFAULTS["gemm_accuracy"],
                   choices=["fast", "strict"], help="See serve.py --gemm-accuracy.")
    p.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN,
                   help="rotary-table cap. The KV pool is sized from the "
                        "bench's own span (ctx + generated), not from this.")
    p.add_argument("--max-num-batched-tokens", type=int, default=2048,
                   help="chunked-prefill budget. Lower than serve.py's 8192 on purpose: "
                        "it bounds the fla prefill temporaries (where an over-budget run "
                        "OOMs) and the per-chunk prefill logits.")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                   help="pre-flight guard: refuse to build if the memory plan exceeds "
                        "this fraction of free HBM (serve.py's rule). Tighter "
                        "than serve.py's 0.94 deliberately: serve.py re-measures after "
                        "graph capture and can still refuse to accept traffic, while a "
                        "bench that guesses wrong has already spent the GPU time. "
                        "serve.py's own plan-vs-measurement tolerance is 5%%, so a plan "
                        "landing inside 6%% of the card is not known to fit.")
    p.add_argument("--skip-memory-check", action="store_true",
                   help="print the memory plan but do not enforce it")
    p.add_argument("--no-graphs", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-md", default=None)
    p.add_argument("--quiet", action="store_true")
    return p


def plan_for_args(args: argparse.Namespace):
    """``(rt, plan, seq_span, buckets)`` for ``args`` -- **no CUDA, no model**.

    Split out of :func:`main` so the byte budget is testable on a laptop
    (``tests/test_spec_decode.py::TestBenchSpecMemoryPlan``) without spending
    minutes loading 27.9 GiB of weights first; this function is that plan, and
    it is the same one :func:`main` enforces.
    """
    max_batch = max(args.batch)
    buckets = tuple(sorted(set(args.batch) | {32}))
    # The sequences this bench creates are exactly `ctx` prompt tokens plus the
    # tokens it generates -- never `--max-model-len`. Sizing the KV pool from
    # the real span instead of the rotary cap is 18.3 GiB rather than 23.4 at
    # B=128 (68 KiB/token/layer-set is the steepest term here).
    gen_budget = max(sweep_max_tokens(args.steps, args.warmup, max(args.k)), args.correct_tokens)
    seq_span = args.ctx + gen_budget
    geom = derive_pool_sizes(max_batch, seq_span, 16, slack_pages=64)
    rt = RuntimeConfig(
        device=args.device,
        enable_mtp=True,
        max_num_seqs=max(max_batch, 32),
        graph_buckets=buckets,
        use_cuda_graphs=not args.no_graphs,
        ssm_state_dtype=args.ssm_state_dtype,
        kv_cache_dtype=args.kv_cache_dtype,
        # the single knob that moves the "no spec" baseline at B=1 from
        # ~15.4 ms (torch) to 12.638 ms (triton).
        norm_backend=args.norm_backend,
        fused_ops_backend=args.fused_ops_backend,
        n_kv_pages=geom["n_kv_pages"],
        max_pages_per_seq=geom["max_pages_per_seq"],
        # -- the memory geometry, straight from serve.py's defaults ---------- #
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gemm_weight_cache=args.gemm_weight_cache,
        gemm_accuracy=args.gemm_accuracy,
        gemm_priority=getattr(args, "gemm_priority", M1_DEFAULTS["gemm_priority"]),
        conv_prefill_tile_tokens=M1_DEFAULTS["conv_prefill_tile_tokens"],
    )

    # -- the pre-flight memory plan. Same planner as the server -------------- #
    plan = plan_memory(
        max_num_seqs=rt.max_num_seqs,
        max_model_len=seq_span,
        page_size=rt.page_size,
        n_kv_pages=rt.n_kv_pages,
        max_pages_per_seq=rt.max_pages_per_seq,
        kv_cache_dtype=rt.kv_cache_dtype,
        ssm_state_dtype=rt.ssm_state_dtype,
        dtype=rt.dtype,
        enable_mtp=rt.enable_mtp,
        arch=arch_from_checkpoint(args.model),
        max_num_batched_tokens=rt.max_num_batched_tokens,
        conv_prefill_tile_tokens=rt.conv_prefill_tile_tokens,
        n_graph_buckets=len(buckets),
        max_batch=max_batch,
        attn_workspace_mb=rt.attn_workspace_mb,
        gemm_weight_cache=rt.gemm_weight_cache,
        use_cuda_graphs=rt.use_cuda_graphs,
        # The bench builds a SpecDecoder per k on top of the plain decoder; the widest
        # window (k_max + 1) is what sizes the resident GDN input cache.
        spec_window=(max(args.k) + 1) if args.k else 0,
    )
    return rt, plan, seq_span, buckets


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    # The same canonical config bench_runtime and profile_step use, so a
    # spec-decode ms/step is comparable with a plain decode ms/step. Without
    # it, bench_spec's no-spec B=256 measured 82 ms against bench_runtime's
    # 45.8 for what is nominally the same step.
    args = apply_preset(args, parser, argv=argv)
    verbose = not args.quiet
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model, trust_remote_code=True)
    texts = load_texts(args.prompts_file, 200)

    rt, plan, seq_span, buckets = plan_for_args(args)
    gen_budget = seq_span - args.ctx
    print(format_resolved_config(
        rt, ctx_len=args.ctx, steps=args.steps, warmup=args.warmup,
        batches=list(args.batch), k=list(args.k),
    ), file=sys.stderr)
    if verbose:
        print(f"[bench_spec] building engine: buckets={buckets} "
              f"n_kv_pages={rt.n_kv_pages} max_pages_per_seq={rt.max_pages_per_seq} "
              f"seq_span={seq_span} (ctx {args.ctx} + {gen_budget} generated)", file=sys.stderr)
        print(f"[bench_spec] runtime config (baseline comparability): "
              f"norm={rt.norm_backend} fused_ops={rt.fused_ops_backend} "
              f"ssm_state={rt.ssm_state_dtype} kv={rt.kv_cache_dtype} "
              f"graphs={rt.use_cuda_graphs} gemm_weight_cache={rt.gemm_weight_cache} "
              f"gemm_accuracy={rt.gemm_accuracy}", file=sys.stderr)
        print(format_memory_plan(plan, rt, seq_span), file=sys.stderr)

    free = measure_allocation(args.device)
    budget = None
    if free is not None:
        budget = free["free_gib"] * args.gpu_memory_utilization
        if verbose:
            print(f"[bench_spec] free HBM {free['free_gib']:.1f} GiB, budget "
                  f"{budget:.1f} GiB (--gpu-memory-utilization "
                  f"{args.gpu_memory_utilization})", file=sys.stderr)
        if plan["total_gib"] > budget and not args.skip_memory_check:
            print(
                f"[bench_spec] REFUSING TO BUILD: the memory plan needs "
                f"{plan['total_gib']:.1f} GiB ({plan['steady_gib']:.1f} steady + "
                f"{plan['prefill_gib']:.1f} prefill) but only {budget:.1f} GiB is "
                f"budgeted. Without this check the run would OOM, typically inside "
                f"the gate's first prefill. Halve --batch's largest cell "
                f"(both pools scale with it: KV {plan['kv_gib']:.0f} GiB, state "
                f"{plan['ssm_gib']:.0f} GiB), or --ssm-state-dtype fp16 "
                f"(halves {plan['ssm_gib']:.1f} GiB, gate becomes advisory), or "
                f"--gemm-weight-cache none (frees {plan['linear_fp8_gib']:.1f} GiB).",
                file=sys.stderr,
            )
            return 2

    comps = build_engine(args.model, rt=rt, verbose=verbose)
    if comps.model.mtp is None:
        raise RuntimeError(f"{args.model} ships no MTP weights -- spec decode needs mtp.*")
    _report_memory(plan, "after weight load", args.device, verbose)
    comps.decoder.warmup()
    if rt.use_cuda_graphs:
        comps.decoder.capture()
    measured = _report_memory(plan, "after decode capture", args.device, verbose)
    if verbose:
        print(f"[bench_spec] max_context_len={comps.model.max_context_len} "
              f"(needs {seq_span})", file=sys.stderr)
    if comps.model.max_context_len < seq_span:
        raise RuntimeError(
            f"max_context_len={comps.model.max_context_len} < the bench's span "
            f"{seq_span}: a sequence would index past the rotary table, which is a "
            f"device-side assert that kills the CUDA context. Raise "
            f"--max-model-len or lower --ctx."
        )

    results: Dict = {
        "config": {
            "model": args.model,
            "ctx": args.ctx,
            "batches": args.batch,
            "ks": args.k,
            "steps": args.steps,
            "ssm_state_dtype": args.ssm_state_dtype,
            "kv_cache_dtype": args.kv_cache_dtype,
            "norm_backend": rt.norm_backend,
            "fused_ops_backend": rt.fused_ops_backend,
            "graph_buckets": list(buckets),
            "use_cuda_graphs": rt.use_cuda_graphs,
            # -- memory geometry, recorded so a run's footprint is
            #    reconstructable from the artefact alone.
            "gemm_weight_cache": rt.gemm_weight_cache,
            "gemm_accuracy": rt.gemm_accuracy,
            "max_model_len": rt.max_model_len,
            "max_num_batched_tokens": rt.max_num_batched_tokens,
            "n_kv_pages": rt.n_kv_pages,
            "max_pages_per_seq": rt.max_pages_per_seq,
            "max_num_seqs": rt.max_num_seqs,
            "seq_span": seq_span,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        },
        # The canonical knob set + how this run deviates
        # from it, in the same shape bench_runtime and profile_step emit, so
        # three artefacts can be diffed mechanically instead of by prose.
        "resolved_config": resolved_config(
            rt, ctx_len=args.ctx, steps=args.steps, warmup=args.warmup,
            batches=list(args.batch), k=list(args.k),
        ),
        "memory": {
            "plan_gib": {k: round(v, 3) for k, v in plan.items()},
            "measured_after_capture_gib": (
                {k: round(v, 3) for k, v in measured.items()} if measured else {}
            ),
        },
        "gemm_backends": {},
        "correctness": {},
        "sweep": [],
    }

    def flush() -> None:
        if args.out_json:
            os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
            with open(args.out_json, "w") as f:
                json.dump(results, f, indent=2)
        if args.out_md:
            os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
            with open(args.out_md, "w") as f:
                f.write(to_markdown(results))

    # -- 1. correctness (cheap, and gates the rest) -------------------------- #
    if not args.skip_correctness:
        corr_prompts = build_prompts(tok, texts, args.correct_prompts, None)
        results["gemm_backends"] = probe_gemm_backends(comps, args.correct_batch or args.correct_prompts, args.k, verbose)
        results["correctness"] = run_correctness(
            comps, corr_prompts, args.k, args.correct_tokens, verbose,
            group=args.correct_group, buckets=buckets,
            diagnose=not args.no_diagnose,
        )
        _report_memory(plan, "after correctness gate", args.device, verbose)
        flush()
        if args.gate and not results["correctness"].get("ok", False):
            if verbose:
                print(
                    "[bench_spec] GATE FAILED -- refusing to run the timed sweep.\n"
                    "[bench_spec] The gate has four parts; section 1/1a of "
                    "the Markdown says which one failed:\n"
                    "[bench_spec]   routing      -- decode and verify must resolve the "
                    "same GEMM backend (section 0)\n"
                    "[bench_spec]   reference    -- two identical plain-greedy runs must "
                    "be token-identical\n"
                    "[bench_spec]   path parity  -- the k=0 window forward must agree with "
                    "the decode forward on top-1, and every disagreement must sit inside "
                    "the measured max|dlogit|\n"
                    "[bench_spec]   speculation  -- k>=1 must not diverge on more prompts "
                    "than the k=0 control does\n"
                    "[bench_spec] Only the last one implicates drafting/acceptance/rollback. "
                    "Re-run with --no-gate only to collect diagnostics.",
                    file=sys.stderr,
                )
            return 1

    # -- 2. timed sweep ------------------------------------------------------ #
    sweep_prompts = build_prompts(tok, texts, max(args.batch), args.ctx)
    for b in args.batch:
        r = run_point(comps, sweep_prompts, b, None, steps=args.steps, warmup=args.warmup)
        results["sweep"].append(r)
        if verbose:
            print(f"[bench_spec] B={b:4d} no spec      : {r['ms_per_step']:8.2f} ms/step "
                  f"{r['tok_s']:9.1f} tok/s", file=sys.stderr)
        flush()

    for k in args.k:
        spec = SpecDecoder(comps.model, comps.buf, rt, SpecConfig(k=k, buckets=buckets))
        spec.warmup(ctx_len=args.ctx)
        if rt.use_cuda_graphs:
            spec.capture(pool_handle=comps.decoder._pool, ctx_len=args.ctx)
        _report_memory(plan, f"after spec k={k} capture", args.device, verbose)
        for b in args.batch:
            r = run_point(comps, sweep_prompts, b, spec, steps=args.steps, warmup=args.warmup)
            results["sweep"].append(r)
            if verbose:
                print(f"[bench_spec] B={b:4d} spec k={k}      : {r['ms_per_step']:8.2f} ms/step "
                      f"{r['tok_s']:9.1f} tok/s  accept={r['accept_length']:.2f}", file=sys.stderr)
            flush()
        del spec
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    flush()
    if not args.out_json and not args.out_md:
        print(json.dumps(results, indent=2))
    ok = results.get("correctness", {}).get("ok", True)
    if verbose:
        print(f"[bench_spec] done. correctness={'PASS' if ok else 'FAIL'}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
