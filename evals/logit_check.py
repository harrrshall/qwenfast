#!/usr/bin/env python3
"""
Standalone (no server) numerical-drift check between two engines' saved
next-token logits for a fixed prompt set. See evals/README.md "logit_check
file format" for how to produce the .npy inputs.

File format (fixed, documented in README):
    logits_<engine>.npy : float32 array, shape (N, V)
        Row i = raw (pre-softmax) logits over the vocabulary for the next
        token, evaluated at the same fixed prompt set (same N prompts, same
        order) with the model run once in teacher-forced / prefill mode
        (no sampling). V must match between the two files (pad/truncate the
        smaller-vocab side is NOT done automatically -- it's treated as an
        error, since silently comparing misaligned vocabularies is unsafe).

Usage:
    python logit_check.py --a logits_reference.npy --b logits_candidate.npy \\
        --out logit_check_result.json [--topk 5]

Computes, per prompt row and aggregated:
    - top-1 agreement: fraction of rows where argmax(a) == argmax(b)
    - top-k agreement (optional): fraction where argmax(b) is in top-k of a
    - KL divergence: mean_i KL(softmax(a_i) || softmax(b_i)), plus max/median
    - max abs logit diff: max_i max_v |a_i,v - b_i,v|, plus mean of per-row max
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise KL(p || q). p, q: (N, V) probability distributions."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return np.sum(p * (np.log(p) - np.log(q)), axis=-1)


def load_logits(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"{path}: expected a 2-D (N, V) array, got shape {arr.shape}")
    return arr.astype(np.float64)


def compute(a: np.ndarray, b: np.ndarray, topk: int | None) -> dict:
    if a.shape != b.shape:
        raise ValueError(
            f"shape mismatch: a={a.shape} b={b.shape}. Both files must cover the same "
            f"prompt set with the same vocabulary size (see README file format)."
        )
    n, v = a.shape

    argmax_a = np.argmax(a, axis=-1)
    argmax_b = np.argmax(b, axis=-1)
    top1_matches = argmax_a == argmax_b
    top1_agreement = float(np.mean(top1_matches))

    topk_agreement = None
    if topk and topk > 1:
        topk_idx_a = np.argpartition(-a, kth=min(topk, v - 1), axis=-1)[:, :topk]
        in_topk = np.array([
            argmax_b[i] in topk_idx_a[i] for i in range(n)
        ])
        topk_agreement = float(np.mean(in_topk))

    pa = softmax(a)
    pb = softmax(b)
    kl_ab = kl_divergence(pa, pb)  # KL(a || b), a treated as reference
    kl_ba = kl_divergence(pb, pa)

    abs_diff = np.abs(a - b)
    row_max_abs_diff = np.max(abs_diff, axis=-1)

    mismatched_rows = [int(i) for i in np.nonzero(~top1_matches)[0][:50]]  # cap for readability

    return {
        "n_prompts": int(n),
        "vocab_size": int(v),
        "top1_agreement": top1_agreement,
        "topk": topk,
        "topk_agreement": topk_agreement,
        "kl_div_a_ref_to_b": {
            "mean": float(np.mean(kl_ab)),
            "median": float(np.median(kl_ab)),
            "max": float(np.max(kl_ab)),
        },
        "kl_div_b_ref_to_a": {
            "mean": float(np.mean(kl_ba)),
            "median": float(np.median(kl_ba)),
            "max": float(np.max(kl_ba)),
        },
        "max_abs_logit_diff": {
            "overall_max": float(np.max(row_max_abs_diff)),
            "mean_of_row_max": float(np.mean(row_max_abs_diff)),
            "median_of_row_max": float(np.median(row_max_abs_diff)),
        },
        "mismatched_row_indices_sample": mismatched_rows,
        "n_mismatched_rows": int(np.sum(~top1_matches)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help="reference engine logits .npy, shape (N, V)")
    p.add_argument("--b", required=True, help="candidate engine logits .npy, shape (N, V)")
    p.add_argument("--out", default=None, help="write JSON report here (default: stdout only)")
    p.add_argument("--topk", type=int, default=5)
    args = p.parse_args()

    a = load_logits(args.a)
    b = load_logits(args.b)
    report = compute(a, b, args.topk)

    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
