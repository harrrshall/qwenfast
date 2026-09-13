"""GEMM **accuracy** measurement.

``bench_gemm.py`` answers "how fast is each backend at each (shape, M)".
This module answers the question that one never asked: **how wrong is it.**

That gap is not academic. A priority table derived purely from
``bench_gemm``'s ``graph_us`` can pick, for every ``M >= 64``, a backend that
is an order of magnitude less accurate than the one it picks below 64. That
breaks speculative decoding's greedy-equivalence check, and means the engine
silently runs at two different arithmetic precisions depending on batch size.
Speed benchmarks cannot reveal this; only an accuracy measurement can.

## What is measured

**The reference.** ``y_ref = x.float() @ dequant_fp32(W).T``, i.e. an exact
fp32 dequantization of the block-128 FP8 checkpoint weight against an fp32
copy of the activation. This is deliberately *not* the original bf16 model:
the checkpoint's own fp8 quantization is a ~3.6e-2-class error that **every**
backend here shares and none of them can fix (its end-to-end effect is
about 98.1% top-1 agreement with the bf16 model). What this
module measures is each backend's error against what this engine's own
weights *mean* — the part a backend choice can actually change.

**relL2** = ``||y - y_ref||_2 / ||y_ref||_2``, plus ``maxabs`` for a tail view.

**The argmax-flip proxy.** relL2 is not a unit anybody has intuition for, so
``argmax_flip_rates`` converts it into "how often does this change the greedy
token": a relative perturbation of the *final hidden state* is pushed through
the model's **real** bf16 ``lm_head`` (248,320 x 5,120) and the argmax
disagreement rate is counted. Using the real lm_head matters — the logit
distribution's near-tie density is a property of that matrix, and it is what
decides how often a small perturbation flips the top-1 (a synthetic
Gaussian logit row would not capture it). It remains a **proxy**, and its two
honest caveats are stated in
``argmax_flip_rates``' docstring.

## Why random weights are enough, and why one real shape is run anyway

The error of every backend here is a property of *arithmetic*, not of the
particular numbers: an fp8-e4m3 activation quantizer has a mantissa-bound
relative error regardless of what it quantizes, and an int8 re-quantizer's
error is set by the amax/rms ratio of a 128-element group, which is ~2.9 for
anything roughly Gaussian. So a random block-scale weight should give the same
answer as the checkpoint's. *Should* is not *does* — real weight blocks have
outliers, and a block whose amax/rms is 6 instead of 2.9 doubles the int8
requant error. ``--real-weights`` therefore loads one real layer's FP8
tensors straight from the checkpoint safetensors and re-runs the same table on
them, so the claim "random is representative" is checked rather than assumed.

## Cost

Short and low-memory by construction — no model build, no checkpoint load
(except ``--real-weights``, which reads two tensors, and ``--lm-head-dir``,
which reads one). Designed to run as a short correctness test (< 2 min,
< 8 GB) on a shared GPU, without needing exclusive use of it. It records **no timing at all** — that is
``bench_gemm.py``'s job, and mixing the two would make this run a benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

# The (N, K) shapes the model actually runs, kept in sync with bench_gemm.py.
from .bench_gemm import SHAPES

#: The M sweep this module cares about: the decode buckets that the priority
#: table splits on (1/32/64 straddle the marlin->flashinfer crossover), plus
#: 2048/4096/8192, which are not decode buckets at all — they are *prefill
#: chunks*, where a W8A8 backend runs at ~2.6e-2 relL2 on every request.
#: 8192 == `RuntimeConfig.max_num_batched_tokens`'s default: every backend's
#: accuracy is flat in M, so this is confirming that holds at the real chunk ceiling,
#: not expecting a surprise -- cheap to check since this module records no
#: timing and is designed to run inside the short-test budget (module
#: docstring above), unlike `bench_gemm.py`'s graph-timed sweep.
DEFAULT_M: Tuple[int, ...] = (1, 32, 64, 128, 256, 512, 2048, 4096, 8192)

#: Shapes small enough to run the full M sweep against an fp32 reference
#: inside the memory budget. `lm_head` is excluded on purpose: it is **bf16**
#: in this checkpoint (fused_weights.py keeps it bf16; the fp8 form is the
#: optional `quantize_lm_head` path), so no fp8 backend runs on it in the real engine,
#: and its fp32 dequant alone is 5.1 GiB.
DEFAULT_SHAPES: Tuple[str, ...] = (
    "attn_qkv_proj",      # [14336, 5120]  x16
    "mlp_down_proj",      # [5120, 17408]  x64  (the largest K)
    "mlp_gate_up_proj",   # [34816, 5120]  x64  (the largest N)
    "gdn_in_proj_qkvz",   # [16384, 5120]  x48
)

#: Real checkpoint tensors used by ``--real-weights``. One per shape family,
#: chosen so the (N, K) matches an entry of ``SHAPES`` exactly where possible.
#: (``attn_qkv_proj`` and ``gdn_in_proj_qkvz`` are *fused* in this engine and
#: therefore have no single checkpoint tensor; ``mlp.down_proj`` and
#: ``mlp.gate_proj`` are stored unfused and match one-for-one.)
REAL_TENSORS: Tuple[Tuple[str, str], ...] = (
    ("mlp_down_proj", "model.language_model.layers.0.mlp.down_proj"),   # [5120, 17408]
    ("mlp_gate_proj", "model.language_model.layers.0.mlp.gate_proj"),   # [17408, 5120]
    ("attn_o_proj", "model.language_model.layers.3.linear_attn.out_proj"),
)

#: Perturbation sizes the flip proxy is evaluated at: the two measured
#: classes (2.7e-3 W8A16, 2.6e-2 W8A8), the strict threshold (5e-3), machete's
#: measured point (7e-3), and two anchors below.
DEFAULT_REL: Tuple[float, ...] = (1e-4, 1e-3, 2.7e-3, 5e-3, 7e-3, 1e-2, 2.6e-2)


# --------------------------------------------------------------------------- #
# weights
# --------------------------------------------------------------------------- #
def make_random_fp8(n: int, k: int, device: str, seed: int = 0):
    """A block-128 :class:`FP8Tensor` with the checkpoint's exact layout.

    Values are Gaussian with a per-block scale set so the codes fill e4m3's
    range the way a real block-quantized checkpoint's do (``amax -> ~448``),
    which is what makes the quantization-error arithmetic comparable."""
    import torch

    from .fused_weights import FP8Tensor

    g = torch.Generator(device=device).manual_seed(seed)
    w = torch.randn(n, k, device=device, dtype=torch.float32, generator=g) * 0.02
    wb = w.view(n // 128, 128, k // 128, 128)
    amax = wb.abs().amax(dim=(1, 3))                     # [N/128, K/128]
    scale = (amax / 448.0).clamp_min(torch.finfo(torch.float32).tiny)
    q = (wb / scale[:, None, :, None]).clamp(-448, 448).view(n, k)
    return FP8Tensor(weight=q.to(torch.float8_e4m3fn), scale_inv=scale)


def load_real_fp8(ckpt_dir: str, prefix: str, device: str):
    """Load one real FP8 linear (``<prefix>.weight`` + ``.weight_scale_inv``).

    Reads the two tensors directly out of the shard that holds them via the
    safetensors index — no model build, no full checkpoint load, so this is
    tens of MB and a second or two, not 30 GB."""
    import torch
    from safetensors.torch import safe_open

    from .fused_weights import FP8Tensor

    idx_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
    with open(idx_path) as f:
        weight_map = json.load(f)["weight_map"]
    wname, sname = prefix + ".weight", prefix + ".weight_scale_inv"
    if wname not in weight_map or sname not in weight_map:
        raise KeyError(f"{prefix!r} is not an FP8 linear in this checkpoint")
    out = {}
    for name in (wname, sname):
        with safe_open(os.path.join(ckpt_dir, weight_map[name]), framework="pt", device="cpu") as f:
            out[name] = f.get_tensor(name)
    return FP8Tensor(
        weight=out[wname].to(device), scale_inv=out[sname].to(device, torch.float32)
    )


# --------------------------------------------------------------------------- #
# the reference and the metric
# --------------------------------------------------------------------------- #
def fp32_reference(x, w, n_chunk: int = 8192):
    """``x.float() @ dequant_fp32(w).T``, materialized ``n_chunk`` output
    columns at a time.

    The chunking is not an optimization, it is what keeps this runnable as a
    *short, low-memory* test on a shared GPU: an unchunked fp32 dequant of
    ``mlp_gate_up_proj`` is 713 MiB and of ``lm_head`` 5.1 GiB, and this
    module is meant never to need exclusive use of the GPU."""
    import torch

    from .fused_weights import BLOCK, FP8Tensor

    x32 = x.float()
    if not isinstance(w, FP8Tensor):
        return x32 @ w.float().t()
    n, k = w.weight.shape
    step = max(BLOCK, (n_chunk // BLOCK) * BLOCK)
    outs = []
    for lo in range(0, n, step):
        hi = min(lo + step, n)
        wq = w.weight[lo:hi].float()
        s = w.scale_inv[lo // BLOCK: (hi + BLOCK - 1) // BLOCK]
        wdq = wq * s.repeat_interleave(BLOCK, 0)[: hi - lo].repeat_interleave(BLOCK, 1)[:, :k]
        outs.append(x32 @ wdq.t())
        del wq, wdq
    return torch.cat(outs, dim=1)


def rel_l2(y, ref) -> Tuple[float, float]:
    """``(relL2, maxabs)`` of ``y`` against ``ref``, both computed in fp32."""
    d = y.float() - ref
    return float(d.norm() / ref.norm()), float(d.abs().max())


# --------------------------------------------------------------------------- #
# the table
# --------------------------------------------------------------------------- #
def numerics_cell(x, w, backend: str) -> Dict[str, Any]:
    """Run **exactly** ``backend`` — never a fallback.

    Deliberately *not* ``dispatch.linear(..., backend=...)``: that function
    catches a failing backend and silently walks the rest of the priority
    order, so a table built on it would attribute some *other* backend's
    numbers to whichever one it was asked for (a probe built on ``linear``
    reports "backend X worked" for every X). A wrong *speed* number is
    obviously wrong; a wrong *accuracy* number looks plausible."""
    from . import dispatch as D

    fn = D._BACKENDS.get(backend)
    if fn is None:
        return {"backend": backend, "status": "unavailable", "error": "not registered"}
    try:
        y = fn(x, w)
    except Exception as exc:  # noqa: BLE001 -- an unavailable backend is a result
        return {"backend": backend, "status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
    return {"backend": backend, "status": "ok", "y": y}


def numerics_table(
    shapes: Sequence[Dict[str, Any]],
    m_list: Sequence[int],
    backends: Sequence[str],
    device: str = "cuda:0",
    seed: int = 0,
    weights: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """One row per ``(shape, M, backend)``: ``relL2``/``maxabs`` vs
    :func:`fp32_reference`.

    Loop order is **backend-outer** so each backend's one-time weight repack is
    paid once per shape rather than once per M, and every other backend's
    repack cache is freed before the next one runs — which is what keeps peak
    device memory at "one weight + one repack + one fp32 reference" instead of
    "one weight + every backend's repack" (~5x on the largest shape)."""
    import torch

    from . import dispatch as D

    cells: List[Dict[str, Any]] = []
    for shape in shapes:
        name, n, k = shape["name"], shape["n"], shape["k"]
        w = (weights or {}).get(name)
        if w is None:
            w = make_random_fp8(n, k, device, seed=seed)
            source = "random"
        else:
            source = "checkpoint"
            n, k = w.weight.shape
        # activations: bf16, unit-ish RMS scaled like a post-RMSNorm hidden state
        gen = torch.Generator(device=device).manual_seed(seed + 1)
        xs = {
            m: (torch.randn(m, k, device=device, dtype=torch.float32, generator=gen) * 0.5)
            .to(torch.bfloat16)
            for m in m_list
        }
        refs = {m: fp32_reference(xs[m], w) for m in m_list}
        for backend in backends:
            for m in m_list:
                cell = numerics_cell(xs[m], w, backend)
                cell.update({"shape": name, "n": n, "k": k, "m": m, "source": source})
                if cell.pop("status") == "ok":
                    r, mx = rel_l2(cell.pop("y"), refs[m])
                    cell.update({"status": "ok", "rel_l2": r, "maxabs": mx})
                    if verbose:
                        print(f"  {name:18s} [{n},{k}] M={m:5d} {backend:26s} "
                              f"relL2={r:.3e} maxabs={mx:.3e}", file=sys.stderr)
                else:
                    cell["status"] = "unavailable"
                    if verbose:
                        print(f"  {name:18s} [{n},{k}] M={m:5d} {backend:26s} "
                              f"{cell['error']}", file=sys.stderr)
                cells.append(cell)
            D.free_repack_caches(w)
            torch.cuda.empty_cache()
        del w, xs, refs
        torch.cuda.empty_cache()
    return cells


# --------------------------------------------------------------------------- #
# the argmax-flip proxy
# --------------------------------------------------------------------------- #
def argmax_flip_rates(
    lm_head,
    rels: Sequence[float] = DEFAULT_REL,
    rows: int = 8192,
    device: str = "cuda:0",
    seed: int = 7,
    chunk: int = 256,
) -> List[Dict[str, float]]:
    """Greedy-argmax flip rate as a function of relative hidden-state error.

    For each ``rel``: perturb a batch of final-hidden-state vectors by an
    independent Gaussian of relative rms ``rel``, push both the clean and the
    perturbed batch through the **real** ``lm_head``, and count how often the
    argmax over the 248,320-token vocabulary changes.

    **Two caveats, and they cut in opposite directions.**

    * *This understates it.* A GEMM's error enters at every one of the ~305
      linears in the model, not once at the end. Errors do partially wash out
      through the RMSNorms, so the accumulated final-hidden-state error is not
      64x a single layer's — but it is more than 1x, so the flip rate at a
      given per-GEMM relL2 is at least this.
    * *This overstates its importance.* A flipped argmax is a flipped
      **near-tie**. It makes the output non-reproducible, which is a hard
      requirement for speculative decoding and for any A/B
      whose prefill and decode must agree, but it is not by itself evidence
      of worse text. The checkpoint's own fp8 quantization already flips 1.9%
      of argmaxes against the bf16 model at unchanged eval
      scores. Read this table as a *determinism* metric.

    The hidden states are Gaussian (a post-RMSNorm hidden state has ~unit
    rms), which is a proxy; the ``lm_head`` is real, which is the part that
    actually sets how dense the near-ties are.

    The row count sets the resolution: a full ``[rows, 248320]`` fp32 logit
    matrix is 1 MiB per row, so the rows are processed in ``chunk``-sized
    blocks and only the argmaxes are kept. That is what lets ``rows`` be large
    enough for the answer to have more than one significant figure -- at 512
    rows the smallest resolvable rate is 0.2%, which is the same order as the
    numbers being measured."""
    import torch

    g = torch.Generator(device=device).manual_seed(seed)
    k = lm_head.shape[1]
    hits = {float(r): 0 for r in rels}
    done = 0
    while done < rows:
        m = min(chunk, rows - done)
        h = torch.randn(m, k, device=device, dtype=torch.float32, generator=g)
        base = _argmax_over_vocab(h, lm_head)
        for rel in rels:
            noise = torch.randn(m, k, device=device, dtype=torch.float32, generator=g) * rel
            pert = _argmax_over_vocab(h + noise, lm_head)
            hits[float(rel)] += int((base != pert).sum())
            del noise, pert
        del h, base
        done += m
    out = []
    for rel in rels:
        flips = hits[float(rel)] / float(rows)
        out.append({
            "rel": float(rel),
            "rows": rows,
            "flips": hits[float(rel)],
            "flip_rate": flips,
            "one_flip_per_tokens": (1.0 / flips) if flips > 0 else float("inf"),
        })
        print(f"  rel={rel:.1e}  flips={hits[float(rel)]}/{rows}  flip_rate={flips:.4f}  "
              f"~1 flip per {out[-1]['one_flip_per_tokens']:.0f} tokens", file=sys.stderr)
    return out


def _argmax_over_vocab(h, lm_head, vchunk: int = 32768):
    """``(h.float() @ lm_head.float().T).argmax(-1)`` without ever holding the
    fp32 ``lm_head`` (5.1 GiB) or the ``[rows, 248320]`` logit matrix.

    The whole point of this module is that it runs *alongside* whoever owns
    the GPU, so it walks the vocabulary in chunks and keeps a running
    ``(max, argmax)`` instead. The arithmetic is identical to the unchunked
    form -- a max is associative."""
    import torch

    best_v = best_i = None
    for lo in range(0, lm_head.shape[0], vchunk):
        wc = lm_head[lo: lo + vchunk].float()
        v, i = (h @ wc.t()).max(-1)
        i = i + lo
        if best_v is None:
            best_v, best_i = v, i
        else:
            take = v > best_v
            best_v = torch.where(take, v, best_v)
            best_i = torch.where(take, i, best_i)
        del wc, v, i
    return best_i


def load_lm_head(ckpt_dir: str, device: str):
    """The checkpoint's real ``lm_head.weight`` (bf16 even in the FP8 build)."""
    import json as _json

    from safetensors.torch import safe_open

    with open(os.path.join(ckpt_dir, "model.safetensors.index.json")) as f:
        weight_map = _json.load(f)["weight_map"]
    shard = weight_map["lm_head.weight"]
    with safe_open(os.path.join(ckpt_dir, shard), framework="pt", device="cpu") as f:
        return f.get_tensor("lm_head.weight").to(device)


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def to_markdown(results: Dict[str, Any]) -> str:
    from . import dispatch as D

    cells = results["cells"]
    backends = []
    for c in cells:
        if c["backend"] not in backends:
            backends.append(c["backend"])
    m_list = sorted({c["m"] for c in cells})
    lines = [
        "# GEMM numerics — relL2 vs an exact fp32 dequant reference",
        "",
        f"device: `{results['env'].get('gpu_name')}` · "
        f"torch {results['env'].get('torch_version')} · "
        f"vllm {results['env'].get('vllm_version')} · "
        f"{results['env'].get('timestamp_utc')}",
        "",
    ]
    shapes = []
    for c in cells:
        key = (c["shape"], c["n"], c["k"], c["source"])
        if key not in shapes:
            shapes.append(key)
    for name, n, k, source in shapes:
        lines += [f"## `{name}` [{n}, {k}] — {source} weights", "",
                  "| backend | act | " + " | ".join(f"M={m}" for m in m_list) + " |",
                  "|---|---|" + "---|" * len(m_list)]
        for b in backends:
            row = [f"`{b}`", D.BACKEND_ACT_DTYPE.get(b, "?")]
            for m in m_list:
                hit = [c for c in cells
                       if c["shape"] == name and c["source"] == source
                       and c["backend"] == b and c["m"] == m]
                if not hit:
                    row.append("—")
                elif hit[0]["status"] != "ok":
                    row.append("n/a")
                else:
                    row.append(f"{hit[0]['rel_l2']:.2e}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    if results.get("flips"):
        lines += ["## Greedy-argmax flip rate through the real `lm_head`", "",
                  "| relative hidden-state error | flip rate | ~1 flip per | flips / rows |",
                  "|---|---|---|---|"]
        for f in results["flips"]:
            lines.append(f"| {f['rel']:.1e} | {f['flip_rate']:.4f} | "
                         f"{f['one_flip_per_tokens']:.0f} tokens | "
                         f"{f.get('flips', '?')} / {f.get('rows', '?')} |")
        lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="gemm_numerics",
                   help="writes <out>.json and <out>.md")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--m", default=",".join(str(m) for m in DEFAULT_M))
    p.add_argument("--shapes", default=",".join(DEFAULT_SHAPES),
                   help="comma-separated bench_gemm.SHAPES names")
    p.add_argument("--backends", default=None,
                   help="comma-separated; default = every registered fp8 backend")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--real-weights", default=None, metavar="CKPT_DIR",
                   help="also run the table on real FP8 tensors loaded from this "
                        "checkpoint directory (checks that random weights are "
                        "representative rather than assuming it)")
    p.add_argument("--real-m", default="1,128",
                   help="M sweep for the --real-weights pass (kept short)")
    p.add_argument("--lm-head-dir", default=None, metavar="CKPT_DIR",
                   help="run the argmax-flip proxy through this checkpoint's real "
                        "bf16 lm_head (2.5 GiB read)")
    p.add_argument("--flip-rows", type=int, default=8192,
                   help="hidden-state rows for the argmax-flip proxy; sets its resolution "
                        "(1/rows). Processed 256 at a time, so this costs time, not memory.")
    args = p.parse_args(argv)

    import torch

    from . import dispatch as D
    from .bench_gemm import env_metadata

    torch.cuda.set_device(args.device)
    m_list = [int(x) for x in args.m.split(",") if x]
    wanted = set(args.shapes.split(","))
    shapes = [s for s in SHAPES if s["name"] in wanted and s["kind"] == "fp8"]
    # `bf16_native` is excluded, not forgotten: this sweep only ever runs fp8
    # shapes (`kind == "fp8"` above) and that backend raises for an FP8Tensor
    # by construction (see `dispatch._bf16_native`), so including it
    # would add one guaranteed-error row per cell and nothing else.
    _SKIP = {"bf16_dequant", "bf16_native"}
    backends = (args.backends.split(",") if args.backends
                else [b for b in D.available_backends() if b not in _SKIP] + ["bf16_dequant"])

    results: Dict[str, Any] = {
        "env": env_metadata(args.device),
        "args": vars(args),
        "reference": "x.float() @ dequant_fp32(W).T",
        "backend_act_dtype": dict(D.BACKEND_ACT_DTYPE),
        "strict_rel_l2_max": D.STRICT_REL_L2_MAX,
        "cells": [],
        "flips": [],
    }

    print("[numerics] random-weight table", file=sys.stderr)
    results["cells"] += numerics_table(shapes, m_list, backends, device=args.device, seed=args.seed)

    if args.real_weights:
        print("[numerics] real-checkpoint-weight table", file=sys.stderr)
        real_m = [int(x) for x in args.real_m.split(",") if x]
        for shape_name, prefix in REAL_TENSORS:
            try:
                w = load_real_fp8(args.real_weights, prefix, args.device)
            except Exception as exc:  # noqa: BLE001
                print(f"[numerics]   {prefix}: skipped ({type(exc).__name__}: {exc})",
                      file=sys.stderr)
                continue
            n, k = w.weight.shape
            fake_shape = [{"name": f"real:{prefix.split('.')[-1]}", "n": n, "k": k}]
            results["cells"] += numerics_table(
                fake_shape, real_m, backends, device=args.device, seed=args.seed,
                weights={fake_shape[0]["name"]: w},
            )
            del w
            torch.cuda.empty_cache()

    if args.lm_head_dir:
        print("[numerics] argmax-flip proxy through the real lm_head", file=sys.stderr)
        lm = load_lm_head(args.lm_head_dir, args.device)
        results["flips"] = argmax_flip_rates(lm, rows=args.flip_rows, device=args.device)
        del lm
        torch.cuda.empty_cache()

    out_json = args.out if args.out.endswith(".json") else args.out + ".json"
    os.makedirs(os.path.dirname(os.path.abspath(out_json)) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    out_md = out_json[: -len(".json")] + ".md"
    with open(out_md, "w") as f:
        f.write(to_markdown(results) + "\n")
    print(f"[numerics] wrote {out_json} and {out_md}", file=sys.stderr)
    print(to_markdown(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
