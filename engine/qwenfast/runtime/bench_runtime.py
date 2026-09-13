#!/usr/bin/env python
"""Offline runtime benchmark: decode tok/s and prefill tok/s vs the
analytic memory-bandwidth ceiling.

    decode:  B in {1, 8, 32, 64, 128, 256}, 2K prefilled context,
             with/without CUDA graphs, fp32 vs fp16 SSM state
    prefill: varlen chunks at 2K / 8K tokens

The headline number is **achieved / ceiling**, so that is what this script
reports for each point, not just raw tok/s.

Decode-step ceiling (fp8 KV throughout; the SSM state dtype only changes the
``ssm_ms_per_seq`` constant, halving it going fp32 -> fp16)::

    step_ms(B, ctx) = weight_ms + B * (ssm_ms_per_seq + kv_ms_per_tok * ctx)

The ceiling model is decode-only and has no prefill closed form; this script
uses the natural analogue (one weight read amortised over the whole chunk plus
the KV-write traffic for the chunk) and labels it explicitly as an
approximation.

Setting up "B sequences with 2K prefilled context" by actually running a 2K
prefill B times would dominate the benchmark's own wall time and measure
prefill, not decode. Since decode cost is a function of committed KV/SSM
*traffic*, not KV *content*, this script fakes the context cheaply: it
reserves the right number of KV pages and sets ``seq_len`` directly
(``_fake_context``), so every decode step reads exactly as many KV bytes as
a real 2K-context step would, without spending seconds per data point on a
real prefill.

Guarded like every other module here: importable on a CPU-only machine (only
``torch`` is required at import time), but the actual sweep needs a real
checkpoint and is meant to run on a GPU host, e.g.::

    /home/venv_vllm/bin/python -m qwenfast.runtime.bench_runtime \\
        --model /home/hf/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/* \\
        --out /home/qwenfast-results/bench_runtime.json
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

import torch

from .engine import EngineComponents, build_engine
from .fused_model import RuntimeConfig, make_prefill_batch
from .preset import (
    CANONICAL_BENCH,
    add_preset_arg,
    apply_preset,
    buckets_for_batches,
    format_resolved_config,
    resolved_config,
)

# --------------------------------------------------------------------------- #
# analytic ceilings (memory-bandwidth model)
# --------------------------------------------------------------------------- #
WEIGHT_MS = 6.19  # 29.71 GB FP8 weights / 4.8 TB/s
KV_MS_PER_TOK = 6.83e-6  # 32 KiB/token fp8 KV / 4.8 TB/s
SSM_MS_PER_SEQ = {"fp32": 0.0629, "fp16": 0.03145}  # halves with the state dtype


def decode_ceiling_tok_s(batch: int, ctx: int, ssm_state_dtype: str) -> float:
    ssm_ms = SSM_MS_PER_SEQ[ssm_state_dtype]
    step_ms = WEIGHT_MS + batch * (ssm_ms + KV_MS_PER_TOK * ctx)
    return batch / (step_ms / 1000.0)


def prefill_ceiling_tok_s(n_tokens: int) -> float:
    """Weight read amortised over the whole chunk + the chunk's own KV write.
    The decode ceiling model has no prefill form; this is the natural
    analogue, called out as an approximation in the module docstring."""
    t_s = WEIGHT_MS / 1000.0 + n_tokens * KV_MS_PER_TOK / 1000.0
    return n_tokens / t_s


# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #
@dataclass
class DecodeResult:
    batch: int
    ctx_len: int
    ssm_state_dtype: str
    graphs: bool
    ms_per_step: float
    tok_s: float
    ceiling_tok_s: float
    efficiency: float
    # An anomalous point (e.g. a batch that is non-monotone vs its neighbours)
    # cannot be told apart from noise or a first-measurement penalty (such as
    # page fragmentation) from one steady-state number alone. Both fields
    # below exist to make that diagnosis possible from the JSON without a
    # rerun. `bucket`: the CUDA-graph bucket this batch replays against
    # (`decoder.bucket_for(batch)`). A batch sharing a bucket with
    # a much larger one pads/replays differently than one that owns its own
    # bucket, which is a *structural* candidate explanation, not noise.
    bucket: int = -1
    # `repeat_ms_per_step`: one ms/step per `--repeats` measurement block
    # (each its own warmup-then-`steps`-steps timing loop, back to back in
    # the same process/graph capture) instead of collapsing straight to one
    # mean -- a first-repeat outlier vs N stable repeats distinguishes a
    # one-off allocator/fragmentation penalty from a reproducible structural
    # cost. `ms_per_step` above is the median of these when repeats > 1.
    repeat_ms_per_step: List[float] = field(default_factory=list)


@dataclass
class PrefillResult:
    n_tokens: int
    ms_per_chunk: float
    tok_s: float
    ceiling_tok_s: float
    efficiency: float


def calibrate_kv_fp8(
    model, *, n_prompts: int = 4, prompt_len: int = 256, seed: int = 0, verbose: bool = False
) -> Dict[int, "Tuple[float, float]"]:
    """Static fp8 KV-cache calibration for the offline bench/profile scripts.

    ``FusedQwenForCausalLM.calibrate_kv_scales`` wants "real
    prompts", but this script has no tokenizer, so a fixed pseudo-random
    (seeded, so runs are reproducible) token-id calibration set stands in
    for one here. ``verify_vs_hf.py`` uses real tokenized prompts instead
    (it has a tokenizer already loaded); this is the throughput-bench
    substitute, good enough to exercise the real quantization/dequant path
    and get a non-trivial (non-1.0) scale, which is all a *timing* run
    needs -- accuracy is what ``verify_vs_hf.py --kv-cache-dtype fp8`` and
    the tiny-model tests (``tests/test_runtime.py::TestFP8KVCache``) gate.
    No-op (returns ``{}``) for a bf16 pool. Must run before
    ``decoder.warmup()``/``.capture()`` (calibrate_kv_scales's docstring).
    """
    if model.kv_pool.cfg.dtype != "fp8":
        return {}
    g = torch.Generator(device="cpu").manual_seed(seed)
    vocab = model.config.vocab_size
    prompts = [
        torch.randint(0, vocab, (prompt_len,), generator=g).tolist() for _ in range(n_prompts)
    ]
    scales = model.calibrate_kv_scales(prompts)
    if verbose and scales:
        first_layer = next(iter(scales))
        print(f"[bench_runtime] fp8 KV calibrated: {len(scales)} layer(s), "
              f"e.g. layer {first_layer}: k_scale/v_scale={scales[first_layer]}", file=sys.stderr)
    return scales


def _release_all_slots(model) -> int:
    """Hand every non-scratch slot's KV pages back to the allocator.

    Without this the sweep would leak a whole batch's worth of pages at every
    step of the batch ladder: ``_fake_context`` reserves
    ``ceil(2049/16) = 129`` pages per slot, so B=1 -> 8 -> 32 -> 64 would
    accumulate 64 x 129 = 8,256 reservations against an 8,192-page pool and
    exhaust it at B=64.  ``reset_slot`` frees pages itself, but the sweep still has to release slots that the
    *next*, smaller batch will not touch, so this runs before every point.
    """
    freed = 0
    for slot in range(model.n_slots):
        freed += model.kv_pool.free_pages(slot)
    return freed


def _fake_context(model, slots: Sequence[int], ctx_len: int) -> None:
    """Reserve KV pages and set ``seq_len`` directly -- see module docstring.

    Reserves ``ctx_len + 1`` tokens of page capacity: the timed decode step
    itself writes the *next* token at position ``ctx_len``, one past the
    ``ctx_len`` tokens already "committed" (``seq_len``).
    """
    _release_all_slots(model)
    for slot in slots:
        model.reset_slot(slot)  # zeroes SSM/conv state AND frees stale pages
        model.kv_pool.ensure_capacity(slot, ctx_len + 1)
        if ctx_len > 0:
            model.kv_pool.seq_len[slot] = ctx_len


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def bench_decode(
    comps: EngineComponents, batch: int, ctx_len: int, *, steps: int = 30, warmup: int = 5,
    repeats: int = 1,
) -> DecodeResult:
    """``repeats > 1``: run ``repeats`` independent warmup+timing blocks back
    to back (same build, same captured graph) and report the per-repeat
    ms/step list alongside the usual median, so an anomalous point can be
    diagnosed (a first-measurement page-fragmentation penalty, or a
    reproducible per-bucket cost?). ``repeats=1`` (the default) is a single
    timing block."""
    model, buf, decoder, rt = comps.model, comps.buf, comps.decoder, comps.rt
    if batch > rt.max_num_seqs:
        raise ValueError(f"batch {batch} exceeds RuntimeConfig.max_num_seqs={rt.max_num_seqs}")

    slots = list(range(batch))
    _fake_context(model, slots, ctx_len)
    bucket = decoder.bucket_for(batch)
    pad = bucket - batch
    full_slots = slots + [model.scratch_slot] * pad

    buf.host["input_ids"][:bucket] = torch.zeros(bucket, dtype=torch.int32)
    buf.host["positions"][:bucket] = torch.tensor(
        [ctx_len] * batch + [0] * pad, dtype=torch.int32
    )
    buf.host["slot_ids"][:bucket] = torch.tensor(full_slots, dtype=torch.int32)
    buf.host["temperature"][:bucket] = torch.ones(bucket)
    buf.host["top_p"][:bucket] = torch.ones(bucket)
    buf.host["top_k"][:bucket] = torch.zeros(bucket)
    buf.upload(["input_ids", "positions", "slot_ids", "temperature", "top_p", "top_k"])

    repeat_ms: List[float] = []
    for r in range(max(1, repeats)):
        for _ in range(warmup):
            decoder.step(batch, full_slots)
        _sync(model.device)

        t0 = time.perf_counter()
        for _ in range(steps):
            decoder.step(batch, full_slots)
        _sync(model.device)
        dt = time.perf_counter() - t0
        repeat_ms.append(dt / steps * 1000.0)

    ms_per_step = statistics.median(repeat_ms)
    tok_s = batch * 1000.0 / ms_per_step
    ceiling = decode_ceiling_tok_s(batch, ctx_len, rt.ssm_state_dtype)
    return DecodeResult(
        batch=batch,
        ctx_len=ctx_len,
        ssm_state_dtype=rt.ssm_state_dtype,
        graphs=decoder.graphs_enabled,
        ms_per_step=ms_per_step,
        tok_s=tok_s,
        ceiling_tok_s=ceiling,
        efficiency=tok_s / ceiling,
        bucket=bucket,
        repeat_ms_per_step=repeat_ms,
    )


def bench_prefill(comps: EngineComponents, n_tokens: int, *, steps: int = 5, warmup: int = 1) -> PrefillResult:
    model = comps.model
    slot = 0
    ids = [i % model.config.vocab_size for i in range(n_tokens)]

    for _ in range(warmup):
        model.reset_slot(slot)
        model.kv_pool.ensure_capacity(slot, n_tokens)
        batch = make_prefill_batch([ids], [0], [slot], model.device)
        model.prefill_forward(batch, all_logits=False)
    _sync(model.device)

    t0 = time.perf_counter()
    for _ in range(steps):
        model.reset_slot(slot)
        model.kv_pool.ensure_capacity(slot, n_tokens)
        batch = make_prefill_batch([ids], [0], [slot], model.device)
        model.prefill_forward(batch, all_logits=False)
    _sync(model.device)
    dt = time.perf_counter() - t0

    ms_per_chunk = dt / steps * 1000.0
    tok_s = n_tokens * steps / dt
    ceiling = prefill_ceiling_tok_s(n_tokens)
    return PrefillResult(
        n_tokens=n_tokens, ms_per_chunk=ms_per_chunk, tok_s=tok_s, ceiling_tok_s=ceiling, efficiency=tok_s / ceiling
    )


def derive_pool_sizes(
    max_num_seqs: int, ctx_len: int, page_size: int, *, slack_pages: int = 16
) -> Dict[str, int]:
    """KV-pool geometry that actually fits the sweep being asked for.

    ``RuntimeConfig.n_kv_pages`` defaults to a flat 8192, which is
    ``8192 * 16 = 131,072`` tokens total -- less than *one* B=64 x ctx-2048
    data point needs (64 x 2049 = 131,136).  Deriving it from
    ``max_num_seqs x ctx`` is the difference between the sweep running and
    the sweep dying at its largest batch, and it is also what the real
    engine's admission control computes, so it is the honest
    default rather than a bench-only hack.

    ``+1`` page for ``FusedQwenForCausalLM``'s scratch slot (the row every
    CUDA-graph padding row points at), ``+slack_pages`` so a rounding
    difference never turns into an OOM at the last data point.
    """
    pages_per_seq = math.ceil((ctx_len + 1) / page_size)
    return {
        "pages_per_seq": pages_per_seq,
        "n_kv_pages": max_num_seqs * pages_per_seq + 1 + slack_pages,
        "max_pages_per_seq": pages_per_seq + 4,
    }


def collect_gemm_backends(model) -> Dict[str, Dict[str, str]]:
    """``{layer_name: {m_bucket: backend}}`` for every ``ResolvedLinear``.

    A dispatch regression can silently route every layer to a slow backend
    (e.g. ``vllm_block_fp8_triton`` everywhere); recording this table in the
    results JSON makes such a regression visible in the artefact rather than
    only as a 3x-worse number.
    """
    out: Dict[str, Dict[str, str]] = {}

    def add(name, lin):
        out[name] = {str(k): v for k, v in lin.resolved_backends().items()}

    for i, layer in enumerate(model.layers):
        mixer = layer.mixer
        if hasattr(mixer, "in_proj_qkvz"):
            add(f"layer{i}.gdn.in_proj_qkvz", mixer.in_proj_qkvz)
            add(f"layer{i}.gdn.in_proj_ba", mixer.in_proj_ba)
            add(f"layer{i}.gdn.out_proj", mixer.out_proj)
        else:
            add(f"layer{i}.attn.qkv", mixer.qkv)
            add(f"layer{i}.attn.o_proj", mixer.o_proj)
        add(f"layer{i}.mlp.gate_up", layer.mlp.gate_up)
        add(f"layer{i}.mlp.down", layer.mlp.down)
    add("lm_head", model.lm_head)
    return out


def summarize_backends(backends: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, int]]:
    """``{m_bucket: {backend: n_layers}}`` -- the one-line version."""
    summary: Dict[str, Dict[str, int]] = {}
    for per_bucket in backends.values():
        for mb, name in per_bucket.items():
            summary.setdefault(mb, {}).setdefault(name, 0)
            summary[mb][name] += 1
    return summary


#: A backend that resolving to it means something went wrong, mapped to what.
#: This is the "no linear ever silently lands on bf16_dequant" gate.
#: `bf16_dequant` on an fp8 weight is 208 ms/step whole-model, 25x the
#: engine's whole decode step, so a single layer on it is a defect. It must be
#: told apart from the 49 bf16-native layers for which it is simply correct
#: (see `gemm.dispatch._bf16_native`).
SUSPECT_BACKENDS: Dict[str, str] = {
    "bf16_dequant": (
        "materializes the full bf16 weight inside every call (208 ms/step "
        "whole-model). Reaching it means every fp8 backend raised "
        "for that weight -- check the printed rejection reasons"
    ),
    "vllm_block_fp8_triton": (
        "measured 2.1-4x slower than every peer at every M; "
        "it is the last fp8 entry in every priority list, so resolving to it "
        "means the seven above it all raised"
    ),
}


def check_resolved_backends(model, backends: Dict[str, Dict[str, str]],
                            *, verbose: bool = True) -> List[str]:
    """Report every ``ResolvedLinear`` that landed on a suspect backend, with
    the reason each better candidate was rejected. Returns the warning lines.

    Without this check the engine could run all ~305 of its linears on a
    backend nobody chose and show it only as a 3x-worse number.
    Non-fatal by design: a bench must still produce its measurement -- but it
    prints at WARNING volume and lands in the results JSON."""
    warnings: List[str] = []
    reasons_by_layer = _rejection_reasons(model)
    for layer, per_bucket in backends.items():
        for mb, name in per_bucket.items():
            if name not in SUSPECT_BACKENDS:
                continue
            why = reasons_by_layer.get(layer, {}).get(str(mb), {})
            warnings.append(
                f"[bench_runtime] WARNING {layer} @ M-bucket {mb} resolved to {name!r}: "
                f"{SUSPECT_BACKENDS[name]}; rejections={why or '(forced pin -- nothing resolved)'}"
            )
    if verbose:
        for line in warnings[:20]:
            print(line, file=sys.stderr)
        if len(warnings) > 20:
            print(f"[bench_runtime] WARNING ... and {len(warnings) - 20} more", file=sys.stderr)
        if not warnings:
            print("[bench_runtime] resolved-backend check: OK "
                  "(no linear on a suspect backend)", file=sys.stderr)
    return warnings


def _rejection_reasons(model) -> Dict[str, Dict[str, Dict[str, str]]]:
    """``{layer_name: {m_bucket: {backend: why}}}`` -- mirrors
    :func:`collect_gemm_backends`'s traversal."""
    out: Dict[str, Dict[str, Dict[str, str]]] = {}

    def add(name, lin):
        r = getattr(lin, "rejection_reasons", None)
        if r is None:
            return
        out[name] = {str(k): v for k, v in r().items()}

    for i, layer in enumerate(model.layers):
        mixer = layer.mixer
        if hasattr(mixer, "in_proj_qkvz"):
            add(f"layer{i}.gdn.in_proj_qkvz", mixer.in_proj_qkvz)
            add(f"layer{i}.gdn.in_proj_ba", mixer.in_proj_ba)
            add(f"layer{i}.gdn.out_proj", mixer.out_proj)
        else:
            add(f"layer{i}.attn.qkv", mixer.qkv)
            add(f"layer{i}.attn.o_proj", mixer.o_proj)
        add(f"layer{i}.mlp.gate_up", layer.mlp.gate_up)
        add(f"layer{i}.mlp.down", layer.mlp.down)
    add("lm_head", model.lm_head)
    return out


def _build(model_dir: str, *, ssm_state_dtype: str, use_cuda_graphs: bool, max_num_seqs: int,
           graph_buckets: Sequence[int], device: str, verbose: bool,
           ctx_len: int = 2048, page_size: int = 16, kv_cache_dtype: str = "bf16",
           gemm_backend: Optional[str] = None, n_kv_pages: Optional[int] = None,
           max_pages_per_seq: Optional[int] = None,
           norm_backend: str = "torch", fused_ops_backend: str = "torch",
           gemm_accuracy: str = "fast", gemm_weight_cache: str = "multi",
           gemm_priority: str = "v8") -> EngineComponents:
    geom = derive_pool_sizes(max_num_seqs, ctx_len, page_size)
    rt = RuntimeConfig(
        device=device,
        ssm_state_dtype=ssm_state_dtype,
        use_cuda_graphs=use_cuda_graphs,
        max_num_seqs=max_num_seqs,
        graph_buckets=tuple(graph_buckets),
        page_size=page_size,
        kv_cache_dtype=kv_cache_dtype,
        gemm_backend=gemm_backend,
        gemm_accuracy=gemm_accuracy,
        gemm_weight_cache=gemm_weight_cache,
        gemm_priority=gemm_priority,
        norm_backend=norm_backend,
        fused_ops_backend=fused_ops_backend,
        n_kv_pages=n_kv_pages if n_kv_pages is not None else geom["n_kv_pages"],
        max_pages_per_seq=(
            max_pages_per_seq if max_pages_per_seq is not None else geom["max_pages_per_seq"]
        ),
    )
    if verbose:
        print(f"[bench_runtime] kv pool: n_kv_pages={rt.n_kv_pages} "
              f"max_pages_per_seq={rt.max_pages_per_seq} page_size={rt.page_size} "
              f"kv_dtype={rt.kv_cache_dtype}", file=sys.stderr)
    comps = build_engine(model_dir, rt=rt, verbose=verbose)
    # fp8 KV calibration must run before warmup/capture: the
    # calibrated k_scale/v_scale floats get baked into the captured kernel
    # launch (calibrate_kv_scales's docstring).
    calibrate_kv_fp8(comps.model, verbose=verbose)
    comps.decoder.warmup()
    if use_cuda_graphs:
        comps.decoder.capture()
    return comps


class _IncrementalJSONWriter:
    """Writes ``results`` to ``path`` after every data point, so a run that
    crashes or is killed partway through (e.g. a bad ``--batch``/bucket
    combination) still leaves whatever it
    already measured on disk instead of losing the whole sweep."""

    def __init__(self, results: Dict, path: Optional[str]):
        self.results = results
        self.path = path

    def flush(self) -> None:
        if self.path is None:
            return
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.results, f, indent=2)
        os.replace(tmp, self.path)


def run_sweep(
    model_dir: str,
    *,
    batches: Sequence[int] = (1, 8, 32, 64, 128, 256),
    ctx_len: int = 2048,
    prefill_lens: Sequence[int] = (2048, 8192),
    ssm_state_dtypes: Sequence[str] = ("fp32", "fp16"),
    graph_modes: Sequence[bool] = (True, False),
    graph_buckets: Optional[Sequence[int]] = None,
    device: str = "cuda:0",
    steps: int = 30,
    warmup: int = 5,
    verbose: bool = False,
    out_path: Optional[str] = None,
    page_size: int = 16,
    kv_cache_dtype: str = "bf16",
    gemm_backend: Optional[str] = None,
    n_kv_pages: Optional[int] = None,
    max_pages_per_seq: Optional[int] = None,
    norm_backend: str = "torch",
    fused_ops_backend: str = "torch",
    gemm_accuracy: str = "fast",
    gemm_weight_cache: str = "multi",
    gemm_priority: str = "v8",
    tag: Optional[str] = None,
    repeats: int = 1,
) -> Dict:
    buckets = tuple(graph_buckets) if graph_buckets else RuntimeConfig().graph_buckets
    # Cap the bucket table at the largest batch actually being benched.
    # `max_num_seqs` == the SSM slot count, and the SSM pool is
    # 147 MiB/slot fp32 -- keeping the default 512-bucket table while
    # benching up to B=64 would allocate 73 GiB of state pool for slots no
    # data point ever touches, and the KV pool derived from it would not fit.
    top = max(batches)
    buckets = tuple(b for b in buckets if b <= top) or (top,)
    if buckets[-1] < top:
        buckets = buckets + (top,)
    max_num_seqs = max(top, buckets[-1], 1)
    geom = derive_pool_sizes(max_num_seqs, ctx_len, page_size)
    out: Dict = {
        "decode": [],
        "prefill": [],
        "batches": list(batches),
        "graph_buckets": list(buckets),
        "ctx_len": ctx_len,
        "max_num_seqs": max_num_seqs,
        "page_size": page_size,
        "kv_cache_dtype": kv_cache_dtype,
        "gemm_backend": gemm_backend,
        "gemm_accuracy": gemm_accuracy,
        "gemm_weight_cache": gemm_weight_cache,
        "gemm_priority": gemm_priority,
        "norm_backend": norm_backend,
        "fused_ops_backend": fused_ops_backend,
        "n_kv_pages": n_kv_pages if n_kv_pages is not None else geom["n_kv_pages"],
        "max_pages_per_seq": (
            max_pages_per_seq if max_pages_per_seq is not None else geom["max_pages_per_seq"]
        ),
        "steps": steps,
        "warmup": warmup,
        "repeats": repeats,
        "tag": tag,
        "gemm_backends": {},
        #: The full resolved knob set, so a number in this file can be
        #: compared with a number in another results file without guessing
        #: which configuration produced it. Filled in from the first build.
        "resolved_config": None,
        "backend_warnings": [],
    }
    writer = _IncrementalJSONWriter(out, out_path)

    for ssm_dtype in ssm_state_dtypes:
        for graphs in graph_modes:
            if verbose:
                print(f"[bench_runtime] building model: ssm_state_dtype={ssm_dtype} graphs={graphs} "
                      f"buckets={buckets}", file=sys.stderr)
            comps = _build(
                model_dir,
                ssm_state_dtype=ssm_dtype,
                use_cuda_graphs=graphs,
                max_num_seqs=max_num_seqs,
                graph_buckets=buckets,
                device=device,
                verbose=verbose,
                ctx_len=ctx_len,
                page_size=page_size,
                kv_cache_dtype=kv_cache_dtype,
                gemm_backend=gemm_backend,
                gemm_accuracy=gemm_accuracy,
                n_kv_pages=n_kv_pages,
                max_pages_per_seq=max_pages_per_seq,
                norm_backend=norm_backend,
                fused_ops_backend=fused_ops_backend,
                gemm_weight_cache=gemm_weight_cache,
                gemm_priority=gemm_priority,
            )
            key = f"{ssm_dtype}/graphs={graphs}"
            backends = collect_gemm_backends(comps.model)
            out["gemm_backends"][key] = summarize_backends(backends)
            # Print the resolved config and the resolved backend table
            # together, always: a ms/step figure quoted without both is not
            # comparable (configs differing in one knob can be 2.3x apart).
            if out["resolved_config"] is None:
                out["resolved_config"] = resolved_config(
                    comps.model.rt, ctx_len=ctx_len, steps=steps,
                    warmup=warmup, repeats=repeats, batches=list(batches),
                )
            print(format_resolved_config(
                comps.model.rt, ctx_len=ctx_len, steps=steps,
                warmup=warmup, repeats=repeats, batches=list(batches),
            ), file=sys.stderr)
            print(f"[bench_runtime] resolved GEMM backends {key}: "
                  f"{out['gemm_backends'][key]}", file=sys.stderr)
            out["backend_warnings"] += check_resolved_backends(
                comps.model, backends, verbose=verbose
            )
            writer.flush()
            for b in batches:
                r = bench_decode(comps, b, ctx_len, steps=steps, warmup=warmup, repeats=repeats)
                out["decode"].append(asdict(r))
                spread = ""
                if len(r.repeat_ms_per_step) > 1:
                    lo, hi = min(r.repeat_ms_per_step), max(r.repeat_ms_per_step)
                    spread = (f"  [repeats={r.repeat_ms_per_step[0]:.2f}/"
                              f"{'/'.join(f'{m:.2f}' for m in r.repeat_ms_per_step[1:])} ms, "
                              f"spread {100 * (hi - lo) / lo:.1f}%]")
                print(f"[decode] B={b:4d} ctx={ctx_len:5d} bucket={r.bucket:4d} "
                      f"graphs={str(graphs):5} ssm={ssm_dtype:4}: "
                      f"{r.tok_s:9.1f} tok/s  ({r.efficiency:.1%} of ceiling, "
                      f"{r.ms_per_step:.3f} ms/step){spread}")
                writer.flush()
            if ssm_dtype == ssm_state_dtypes[0] and graphs == graph_modes[0]:
                for n in prefill_lens:
                    r = bench_prefill(comps, n, steps=max(1, steps // 6), warmup=1)
                    out["prefill"].append(asdict(r))
                    print(f"[prefill] n={n:6d}: {r.tok_s:9.1f} tok/s  ({r.efficiency:.1%} of ceiling, "
                          f"{r.ms_per_chunk:.3f} ms/chunk)")
                    writer.flush()
            # Each (ssm_dtype, graphs) combination builds a *whole second
            # copy* of the 28 GiB weights + pools; without dropping the old
            # one first the second build OOMs. CPython frees `comps` only
            # when the last reference goes, and the caching allocator holds
            # the freed blocks until `empty_cache()`.
            del comps
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    writer.flush()
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--batch", type=int, nargs="+", default=None,
                    help="exact batch sizes to bench (default: the standard bucket table, or "
                    "derived from --max-batch if that's given instead)")
    p.add_argument("--max-batch", type=int, default=None,
                    help="shorthand for --batch: bench every standard/--buckets bucket <= this, "
                    "plus this value itself if it isn't already one. Takes precedence over --batch.")
    p.add_argument("--buckets", type=int, nargs="+", default=None,
                    help="override the CUDA-graph bucket table (RuntimeConfig.graph_buckets) -- "
                    "e.g. `--buckets 1 32 128` to capture/warm up only those 3 buckets instead of "
                    "the full ~15-bucket table, for a much faster dev-loop iteration. Default: the "
                    "standard table (1,2,4,8,16,24,32,48,64,96,128,192,256,384,512).")
    p.add_argument("--ctx-len", type=int, default=2048)
    p.add_argument("--prefill-lens", type=int, nargs="+", default=[2048, 8192])
    p.add_argument("--ssm-state-dtype", nargs="+", default=["fp32", "fp16"], choices=["fp32", "fp16"])
    p.add_argument("--graphs", choices=["both", "on", "off"], default="both")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--out", default=None,
                    help="write JSON results here, updated after every data point (not just at the "
                    "end) so a killed/crashed run still leaves partial results on disk")
    p.add_argument("--quiet", action="store_true", help="suppress the per-model-build diagnostic "
                    "lines; per-batch/per-prefill result lines always print")
    p.add_argument("--page-size", type=int, default=16, help="KV page size in tokens")
    p.add_argument("--kv-cache-dtype", choices=["bf16", "fp8"], default="bf16",
                    help="KV storage dtype. NOTE the analytic ceiling this script reports "
                    "efficiency against assumes fp8 KV (32 KiB/token); with bf16 the real KV "
                    "traffic is 2x that term, so the reported %-of-ceiling is conservative at "
                    "large B x ctx.")
    p.add_argument("--gemm-backend", default=None,
                    help="force ONE gemm.dispatch backend for every linear (A/B experiments); "
                    "default None = per-M-bucket resolution via "
                    "dispatch.DEFAULT_BACKEND_PRIORITY_BY_M_BUCKET")
    p.add_argument("--n-kv-pages", type=int, default=None,
                    help="override the KV page count; default is derived from "
                    "max_num_seqs x ceil((ctx+1)/page_size) (+scratch +slack), which is what "
                    "the sweep actually needs -- the old flat 8192 default could not hold even "
                    "one B=64/ctx-2048 point")
    p.add_argument("--max-pages-per-seq", type=int, default=None,
                    help="override the per-sequence page-table width (default: derived)")
    p.add_argument("--norm-backend", choices=["torch", "triton"], default="torch",
                    help="RMSNorm/RMSNormGated implementation. 'triton' collapses each norm from "
                    "~9-11 eager kernel launches to 1; parity is asserted by "
                    "tests/test_runtime.py::TestTritonNorms.")
    p.add_argument("--fused-ops-backend", choices=["torch", "triton"], default="torch",
                    help="SwiGLU / GDN-gate-epilogue implementation. "
                    "'triton' fuses each from 2-4 eager launches to 1; parity is asserted by "
                    "tests/test_runtime.py::TestFusedOps.")
    p.add_argument("--gemm-weight-cache", choices=["multi", "single", "none"], default="multi",
                    help="how many permanent repacked weight copies the GEMM dispatcher may "
                    "memoise per weight. The serving default and the canonical "
                    "preset's value is 'single'. NOTE: under 'single' a backend "
                    "that owns a repack cache is refused for every M-bucket after the first "
                    "one warmup resolved, so a priority-table change at M>=32 can be a no-op "
                    "-- pass 'multi' when the point of the run is to compare kernels.")
    p.add_argument("--gemm-priority", choices=["v9", "v8", "v7", "v4"], default="v8",
                    help="GEMM cold-start priority table: 'v8' (default) = v7 "
                    "unchanged at M<=512 plus real prefill-scale buckets (1024/2048/4096/8192); "
                    "'v7' = the graph-timed table (deepgemm rank 1 "
                    "at M in [32,512], clamps every M>512 to the M=512 answer); 'v4' = the "
                    "older table. 'v7'/'v4' are the rollback flags.")
    p.add_argument("--bucket-rule", choices=["batches", "serve"], default=None,
                    help="how to derive the CUDA-graph bucket table when --buckets is not "
                    "given. 'batches' (the canonical rule) captures exactly the batch sizes "
                    "being benched; 'serve' captures the 15-entry table a real server does. "
                    "These differ by up to 66%% at B=32, which is why the "
                    "rule is a flag and is recorded in the results JSON.")
    p.add_argument("--gemm-accuracy", choices=["fast", "strict"], default="fast",
                    help="GEMM accuracy bar. 'fast' (default, historical) "
                    "ranks backends on graph-timed speed alone, which routes every M>=64 to an "
                    "fp8-ACTIVATION kernel measuring relL2 2.6e-2 against an exact fp32 "
                    "reference. 'strict' admits only backends measured at or under "
                    "dispatch.STRICT_REL_L2_MAX (5e-3), i.e. W8A16 only. Pair the two runs to "
                    "price correctness.")
    p.add_argument("--tag", default=None, help="free-form label recorded in the results JSON")
    p.add_argument("--repeats", type=int, default=1,
                    help="independent warmup+timing blocks per decode data point (default 1, the "
                    "previous behaviour). >1 records each block's own ms/step in the JSON's "
                    "`repeat_ms_per_step` (and prints the spread) so a point that looks anomalous "
                    "can be told apart as noise/first-measurement penalty (reproducible drop across "
                    "repeats) vs a structural per-bucket cost (stable across repeats).")
    add_preset_arg(p)
    p.add_argument("--prefill", action="store_true",
                    help="also run the prefill points (off by default: the decode sweep is what "
                    "the headline numbers measure and prefill re-runs cost minutes of GPU time)")
    return p


def _resolve_batches(args: argparse.Namespace, buckets: Sequence[int]) -> List[int]:
    if args.max_batch is not None:
        chosen = sorted({b for b in buckets if b <= args.max_batch} | {args.max_batch})
        if args.batch is not None:
            print(f"[bench_runtime] --max-batch given; ignoring --batch {args.batch}", file=sys.stderr)
        return chosen
    if args.batch is not None:
        return list(args.batch)
    return [1, 8, 32, 64, 128, 256]


def _passed(flag: str, argv: Optional[Sequence[str]]) -> bool:
    toks = list(sys.argv[1:] if argv is None else argv)
    return any(t == flag or t.startswith(flag + "=") for t in toks)


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    # `--preset fastest` fills in every knob left at its
    # CLI default; explicit flags always win, so `--preset fastest
    # --gemm-backend deepgemm` is exactly "the canonical config, one knob
    # moved" -- which is the only kind of A/B whose delta means anything.
    args = apply_preset(args, parser, argv=argv)
    if args.preset and not _passed("--graphs", argv):
        # `use_cuda_graphs=True` in the preset; this CLI spells it `--graphs`,
        # whose default "both" would also build a second, non-graphed model
        # (4.3x slower, and not a number the preset describes).
        args.graphs = "on"
    graph_modes = {"both": (True, False), "on": (True,), "off": (False,)}[args.graphs]
    if args.buckets:
        buckets = tuple(args.buckets)
    elif args.bucket_rule or args.preset:
        # canonical: capture exactly what is benched.
        rule = args.bucket_rule or CANONICAL_BENCH["buckets"]
        prelim = _resolve_batches(args, RuntimeConfig().graph_buckets)
        buckets = buckets_for_batches(prelim, rule)
    else:
        buckets = RuntimeConfig().graph_buckets
    batches = _resolve_batches(args, buckets)

    results = run_sweep(
        args.model,
        batches=batches,
        ctx_len=args.ctx_len,
        prefill_lens=args.prefill_lens if args.prefill else (),
        ssm_state_dtypes=args.ssm_state_dtype,
        graph_modes=graph_modes,
        graph_buckets=buckets,
        device=args.device,
        steps=args.steps,
        warmup=args.warmup,
        verbose=not args.quiet,
        out_path=args.out,
        page_size=args.page_size,
        kv_cache_dtype=args.kv_cache_dtype,
        gemm_backend=args.gemm_backend,
        n_kv_pages=args.n_kv_pages,
        max_pages_per_seq=args.max_pages_per_seq,
        norm_backend=args.norm_backend,
        fused_ops_backend=args.fused_ops_backend,
        gemm_accuracy=args.gemm_accuracy,
        gemm_weight_cache=args.gemm_weight_cache,
        gemm_priority=args.gemm_priority,
        tag=args.tag,
        repeats=args.repeats,
    )
    results["bucket_rule"] = args.bucket_rule or (
        CANONICAL_BENCH["buckets"] if args.preset else "explicit" if args.buckets else "default"
    )
    results["preset"] = args.preset
    if args.out:
        # `run_sweep`'s incremental writer already flushed the last data point;
        # rewrite so the two keys added above (which only `main` knows) are in
        # the artefact too, not just in this process's return value.
        _IncrementalJSONWriter(results, args.out).flush()
        print(f"[bench_runtime] wrote {args.out}", file=sys.stderr)
    else:
        print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
