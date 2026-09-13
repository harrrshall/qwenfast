#!/usr/bin/env python
"""Numerical parity gate: the qwenfast reference model vs HF ``transformers`` (both bf16, same GPU).

The gate is **teacher-forced**, which makes it strict and EOS-independent:

1. Our model greedily generates ``--max-new-tokens`` tokens per prompt,
   *without* stopping at EOS, so every prompt yields a fixed-length continuation.
2. The identical ``prompt + continuation`` sequence is fed to **both** stacks in
   a single full forward.
3. Per continuation position we compare logits (max/mean ``|Δ|``) and top-1
   agreement.

Free-running generation is reported too, but is **informational only** and never
decides pass/fail.  Comparing free-running output is not a parity test: the two
stacks apply different stopping rules, so once one of them emits ``<|im_end|>``
and the other keeps going, every subsequent token is conditioned on a different
prefix and the comparison is meaningless.  Teacher forcing removes that entirely.

Also reported: the last-prompt-position logits taken through our **cached**
prefill path, which exercises the KV/SSM cache rather than the one-shot forward.

Exit code 0 = PASS, 1 = FAIL.

Memory
------
Two bf16 27B copies are ~108 GiB.  That fits on an H200 (141 GiB) but leaves
little headroom, so the default is ``--residency sequential``: run ours (which
also produces the continuations), stash the results on CPU, free it, then run
HF.  ``--residency both`` keeps both models resident.

Example::

    /home/venv_vllm/bin/python engine/qwenfast/verify_vs_hf.py \
        --model /home/hf/hub/models--Qwen--Qwen3.8-27B/snapshots/* --device cuda:0
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

from qwenfast.model import Generator, QwenFastForCausalLM  # noqa: E402
from qwenfast.weights import resolve_snapshot  # noqa: E402

PROMPTS: List[str] = [
    "What is 17 times 23? Answer with just the number.",
    "Name the capital of Australia.",
    "Write a single line of Python that reverses a string called s.",
    "In one sentence, why is the sky blue?",
    "List three prime numbers greater than 50, comma separated.",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _to_id_list(ids) -> List[int]:
    """Normalise whatever ``apply_chat_template`` returned into ``list[int]``.

    transformers 5 may return a ``BatchEncoding``/dict, a tensor, a nested list,
    or a flat list depending on the tokenizer and the kwargs.
    """
    if hasattr(ids, "keys"):  # BatchEncoding / dict
        ids = ids["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    while isinstance(ids, (list, tuple)) and len(ids) == 1 and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(t) for t in ids]


def build_inputs(tokenizer, enable_thinking: bool = False) -> List[List[int]]:
    """Apply the model's chat template to each prompt -> list of id lists."""
    seqs: List[List[int]] = []
    for p in PROMPTS:
        msgs = [{"role": "user", "content": p}]
        kwargs = dict(tokenize=True, add_generation_prompt=True)
        try:
            ids = tokenizer.apply_chat_template(msgs, enable_thinking=enable_thinking, **kwargs)
        except TypeError:
            # tokenizers whose template does not accept enable_thinking
            ids = tokenizer.apply_chat_template(msgs, **kwargs)
        seqs.append(_to_id_list(ids))
    return seqs


def free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def truncate_at_eos(toks: Sequence[int], eos_ids: Sequence[int]) -> Tuple[List[int], Optional[int]]:
    """Return the prefix before the first EOS and that EOS's index (or None)."""
    for i, t in enumerate(toks):
        if t in eos_ids:
            return list(toks[:i]), i
    return list(toks), None


def load_hf(model_dir: str, device: str, dtype: torch.dtype):
    """Load with whichever Auto* class this transformers version maps the arch to.

    ``device_map`` needs ``accelerate``; fall back to a plain ``.to(device)``.
    """
    import transformers

    last_err: Optional[Exception] = None
    for loader_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModel"):
        loader = getattr(transformers, loader_name, None)
        if loader is None:
            continue
        for use_device_map in (True, False):
            try:
                if use_device_map:
                    m = loader.from_pretrained(model_dir, dtype=dtype, device_map={"": device})
                else:
                    m = loader.from_pretrained(model_dir, dtype=dtype).to(device)
                how = "device_map" if use_device_map else ".to(device)"
                print(f"[verify] HF loader: {loader_name} via {how} -> {type(m).__name__}")
                return m.eval()
            except Exception as exc:  # pragma: no cover - depends on the environment
                last_err = exc
                if use_device_map:
                    print(f"[verify] device_map load failed ({type(exc).__name__}: {exc}); "
                          f"retrying with .to(device)")
    raise RuntimeError(f"could not load {model_dir} with transformers: {last_err}")


# --------------------------------------------------------------------------- #
# forward passes
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def ours_pass(
    model: QwenFastForCausalLM,
    seqs: List[List[int]],
    device: torch.device,
    n_new: int,
    use_fla: bool,
) -> Dict[str, list]:
    """Greedy continuations (no EOS stop) + teacher-forced logits + cached-prefill logits."""
    gen = Generator(model, use_fla=use_fla)
    conts: List[List[int]] = []
    prefill_last: List[torch.Tensor] = []
    tf: List[torch.Tensor] = []

    for ids in seqs:
        # --- cached path: prefill, then n_new greedy steps, never stopping ---
        cache = model.make_cache(1, len(ids) + n_new + 2, device=device)
        tok, last, pos = gen.prefill([ids], cache)
        prefill_last.append(last[0].float().cpu())
        toks = [int(tok)]
        for _ in range(n_new - 1):
            tok, _ = gen.decode_step(tok, pos, cache)
            pos = pos + 1
            toks.append(int(tok))
        conts.append(toks)
        del cache
        free_cuda()

        # --- teacher-forced: one no-cache forward over prompt + continuation ---
        full = torch.tensor([ids + toks], dtype=torch.long, device=device)
        lg = model(full, num_logits=n_new + 1, use_fla=use_fla)
        # positions P-1 .. P+n-1; the first n predict the n continuation tokens
        tf.append(lg[0, :-1, :].float().cpu())
        del full, lg
        free_cuda()

    return {"continuations": conts, "prefill_last": prefill_last, "tf": tf}


@torch.inference_mode()
def ours_pass_fused(
    model_dir: str,
    seqs: List[List[int]],
    device: torch.device,
    n_new: int,
    kv_cache_dtype: str,
) -> Dict[str, list]:
    """Same contract as :func:`ours_pass`, but through the fused
    runtime (``runtime.fused_model.FusedQwenForCausalLM``) instead of the
    reference ``Generator``/``HybridCache`` path. The fused runtime is the only
    stack in this tree with a fp8 KV cache (``RuntimeConfig(kv_cache_dtype="fp8")``).
    ``--kv-cache-dtype fp8`` routes ``main()`` here; ``bf16`` keeps
    using the reference path unchanged. This function adds fp8 KV coverage;
    it does not replace the reference gate for the bf16 case.

    Calibrates fp8 KV scales (``FusedQwenForCausalLM.calibrate_kv_scales``)
    on ``seqs`` themselves before generating -- "real prompts" per that
    method's docstring, and exactly the prompts this tool is about to score
    parity on, which is the calibration set this tool can most honestly
    claim is representative.

    Ungraphed (``use_cuda_graphs=False``): this is a correctness gate, not
    a benchmark, and running eager keeps the per-token loop below simple
    (no bucket padding, no scratch-slot bookkeeping) without touching the
    numbers this tool exists to check.
    """
    from qwenfast.runtime.fused_model import DeviceBuffers, FusedQwenForCausalLM, RuntimeConfig, make_prefill_batch

    rt = RuntimeConfig(
        device=str(device), dtype="bf16", kv_cache_dtype=kv_cache_dtype,
        attn_backend="auto", use_cuda_graphs=False,
        max_num_seqs=max(8, len(seqs)),
    )
    model = FusedQwenForCausalLM.from_pretrained(model_dir, rt, verbose=True)
    if kv_cache_dtype == "fp8":
        scales = model.calibrate_kv_scales(seqs)
        print(f"[verify] fp8 KV calibrated: {len(scales)} layer(s) "
              f"(headroom=0.1, static per-layer scale)")

    buf = DeviceBuffers(max_batch=1, vocab_size=model.config.vocab_size,
                         max_pages=model.kv_pool.cfg.n_pages, device=device)

    conts: List[List[int]] = []
    prefill_last: List[torch.Tensor] = []
    tf: List[torch.Tensor] = []
    slot = 0

    for ids in seqs:
        # --- cached path: prefill, then n_new greedy steps, never stopping ---
        model.reset_slot(slot)
        model.kv_pool.ensure_capacity(slot, len(ids) + n_new + 2)
        batch = make_prefill_batch([ids], [0], [slot], device)
        logits = model.prefill_forward(batch, all_logits=False)  # [1, vocab]
        prefill_last.append(logits[0].float().cpu())

        tok = int(logits[0].argmax())
        toks = [tok]
        pos = len(ids)
        for _ in range(n_new - 1):
            buf.host["input_ids"][:1] = torch.tensor([tok], dtype=torch.int32)
            buf.host["positions"][:1] = torch.tensor([pos], dtype=torch.int32)
            buf.host["slot_ids"][:1] = torch.tensor([slot], dtype=torch.int32)
            buf.upload(["input_ids", "positions", "slot_ids"])
            model.attn.plan_decode([slot], 1, seq_lens=[pos + 1])
            step_logits = model.decode_forward(buf, 1)
            tok = int(step_logits[0].argmax())
            pos += 1
            toks.append(tok)
        conts.append(toks)

        # --- teacher-forced: one fresh (reset-slot) forward over prompt +
        # continuation, exactly like the reference model's `model(full, ...)` one-shot call:
        # no reliance on the cache state the loop above just left behind.
        full_ids = ids + toks
        model.reset_slot(slot)
        model.kv_pool.ensure_capacity(slot, len(full_ids) + 1)
        full_batch = make_prefill_batch([full_ids], [0], [slot], device)
        full_logits = model.prefill_forward(full_batch, all_logits=True)  # [T, vocab]
        p = len(ids)
        # positions P-1 .. P+n-1 into the packed [T, vocab] logits, same
        # convention/slice as `ours_pass`'s `lg[0, :-1, :]`.
        tf.append(full_logits[p - 1 : p - 1 + n_new, :].float().cpu())
        model.reset_slot(slot)
        free_cuda()

    del model
    free_cuda()
    return {"continuations": conts, "prefill_last": prefill_last, "tf": tf}


@torch.inference_mode()
def hf_pass(
    model,
    seqs: List[List[int]],
    conts: List[List[int]],
    device: torch.device,
    n_new: int,
) -> Dict[str, list]:
    """Teacher-forced logits over the same sequences + informational free-run."""
    prefill_last: List[torch.Tensor] = []
    tf: List[torch.Tensor] = []
    free: List[List[int]] = []

    for ids, cont in zip(seqs, conts):
        p = len(ids)
        full = torch.tensor([ids + cont], dtype=torch.long, device=device)
        mask = torch.ones_like(full)
        out = model(input_ids=full, attention_mask=mask)
        lg = out.logits if hasattr(out, "logits") else out[0]
        prefill_last.append(lg[0, p - 1].float().cpu())
        tf.append(lg[0, p - 1 : p + n_new - 1, :].float().cpu())
        del full, mask, out, lg
        free_cuda()

        # informational only: HF's own greedy run, with its own stopping rules
        x = torch.tensor([ids], dtype=torch.long, device=device)
        g = model.generate(
            input_ids=x,
            attention_mask=torch.ones_like(x),
            max_new_tokens=n_new,
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )
        free.append([int(t) for t in g[0, x.shape[1] :].tolist()])
        del x, g
        free_cuda()

    return {"prefill_last": prefill_last, "tf": tf, "free": free}


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="snapshot dir (BF16 or FP8) or models--* dir")
    ap.add_argument("--hf-model", default=None, help="HF snapshot dir (default: --model)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--residency", choices=("sequential", "both"), default="sequential")
    ap.add_argument("--logit-tol", type=float, default=0.35, help="max abs teacher-forced logit diff")
    ap.add_argument("--agree-tol", type=float, default=0.97, help="min teacher-forced top-1 agreement")
    ap.add_argument("--no-fla", action="store_true", help="force the pure-torch GDN path")
    ap.add_argument("--enable-thinking", action="store_true")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--kv-cache-dtype", choices=("bf16", "fp8"), default="bf16",
                     help="'bf16' (default): unchanged reference Generator/HybridCache gate. 'fp8': "
                     "route 'ours' through the fused runtime instead "
                     "(runtime.fused_model.FusedQwenForCausalLM(RuntimeConfig(kv_cache_dtype="
                     "'fp8'))), calibrated (calibrate_kv_scales) on this run's own prompts -- "
                     "the only stack in this tree with a fp8 KV cache. "
                     "--no-fla/--enable-thinking are ignored in this mode (the fused runtime "
                     "has its own gdn/attn backend selection, RuntimeConfig defaults).")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16
    n_new = args.max_new_tokens
    ours_dir = resolve_snapshot(args.model)
    hf_dir = resolve_snapshot(args.hf_model or args.model)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(hf_dir)
    seqs = build_inputs(tok, enable_thinking=args.enable_thinking)
    eos_ids = [i for i in {getattr(tok, "eos_token_id", None)} if i is not None]
    print(f"[verify] prompts: {[len(s) for s in seqs]} tokens; eos={eos_ids}")
    print(f"[verify] ours = {ours_dir}")
    print(f"[verify] hf   = {hf_dir}")

    from qwenfast.model import HAS_FLA

    use_fla = HAS_FLA and not args.no_fla
    fp8_kv = args.kv_cache_dtype == "fp8"
    print(f"[verify] kv_cache_dtype: {args.kv_cache_dtype}"
          + ("" if not fp8_kv else " (routing 'ours' through the fused runtime)"))
    if not fp8_kv:
        print(f"[verify] fla kernels: {'ON' if use_fla else 'OFF (pure torch)'}")
    print(f"[verify] teacher-forced comparison over {n_new} continuation positions")

    def run_ours():
        if fp8_kv:
            t0 = time.time()
            o = ours_pass_fused(ours_dir, seqs, device, n_new, kv_cache_dtype="fp8")
            print(f"[verify] qwenfast (fused, fp8 KV) built + ran in {time.time() - t0:.1f}s")
            return o
        t0 = time.time()
        ours = QwenFastForCausalLM.from_pretrained(ours_dir, device=args.device, dtype=dtype, verbose=True)
        print(f"[verify] qwenfast loaded in {time.time() - t0:.1f}s")
        o = ours_pass(ours, seqs, device, n_new, use_fla)
        del ours
        free_cuda()
        return o

    # ours runs FIRST: it produces the continuations that both stacks are scored on
    if args.residency == "sequential" or fp8_kv:
        # the fused fp8 path already frees its own model inside ours_pass_fused
        # (residency="both" only matters for the bf16 reference path below)
        o = run_ours()

        t0 = time.time()
        hf = load_hf(hf_dir, args.device, dtype)
        print(f"[verify] HF loaded in {time.time() - t0:.1f}s")
        h = hf_pass(hf, seqs, o["continuations"], device, n_new)
        del hf
        free_cuda()
    else:
        ours = QwenFastForCausalLM.from_pretrained(ours_dir, device=args.device, dtype=dtype, verbose=True)
        hf = load_hf(hf_dir, args.device, dtype)
        o = ours_pass(ours, seqs, device, n_new, use_fla)
        h = hf_pass(hf, seqs, o["continuations"], device, n_new)
        del hf, ours
        free_cuda()

    # ------------------------------------------------------------------ #
    rows = []
    ok = True
    print()
    print("teacher-forced parity (the gate)")
    print(f"{'#':>2}  {'maxΔ':>8}  {'meanΔ':>9}  {'top1':>8}  {'first bad':>9}  {'prefillΔ':>9}")
    print("-" * 56)
    for i in range(len(seqs)):
        a, b = o["tf"][i], h["tf"][i]
        n = min(a.shape[0], b.shape[0])
        a, b = a[:n], b[:n]
        d = (a - b).abs()
        arg_a, arg_b = a.argmax(-1), b.argmax(-1)
        same = arg_a == arg_b
        agree = int(same.sum())
        first_bad = int((~same).nonzero()[0, 0]) if agree < n else -1
        pd = float((o["prefill_last"][i] - h["prefill_last"][i]).abs().max())

        cont_txt, eos_at = truncate_at_eos(o["continuations"][i], eos_ids)
        hf_free, hf_eos = truncate_at_eos(h["free"][i], eos_ids)
        rows.append(
            {
                "prompt": PROMPTS[i],
                "n_positions": n,
                "max_abs_logit_diff": float(d.max()),
                "mean_abs_logit_diff": float(d.mean()),
                "top1_agreement": agree / n,
                "matched_positions": agree,
                "first_mismatch": first_bad,
                "prefill_last_max_abs_diff": pd,
                # informational
                "ours_free_text": tok.decode(cont_txt),
                "hf_free_text": tok.decode(hf_free),
                "ours_eos_at": eos_at,
                "hf_eos_at": hf_eos,
                "free_text_match": tok.decode(cont_txt) == tok.decode(hf_free),
            }
        )
        print(f"{i:>2}  {float(d.max()):>8.4f}  {float(d.mean()):>9.5f}  {agree:>3}/{n:<4}"
              f"  {first_bad:>9}  {pd:>9.4f}")
        if float(d.max()) > args.logit_tol or agree / n < args.agree_tol:
            ok = False

    max_diff = max(r["max_abs_logit_diff"] for r in rows)
    mean_agree = sum(r["top1_agreement"] for r in rows) / len(rows)
    max_prefill = max(r["prefill_last_max_abs_diff"] for r in rows)
    print("-" * 56)
    print(f"max |Δlogit| (teacher-forced) : {max_diff:.4f}   (tol {args.logit_tol})")
    print(f"mean top-1 agreement          : {mean_agree:.4f}   (tol {args.agree_tol})")
    print(f"max |Δlogit| (cached prefill) : {max_prefill:.4f}   [cache-path check]")

    print("\nfree-running output (informational only, not part of the gate)")
    for i, r in enumerate(rows):
        flag = "same" if r["free_text_match"] else "differs"
        print(f"  [{i}] {flag}  (eos@ ours={r['ours_eos_at']} hf={r['hf_eos_at']})")
        if not r["free_text_match"]:
            print(f"       ours: {r['ours_free_text']!r}")
            print(f"       hf  : {r['hf_free_text']!r}")

    for r in rows:
        if r["top1_agreement"] < 1.0:
            print(f"\n  TEACHER-FORCED MISMATCH: {r['prompt']}"
                  f"\n    first bad position {r['first_mismatch']} of {r['n_positions']}")

    verdict = "PASS" if ok else "FAIL"
    print(f"\n=== {verdict} ===")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(
                {
                    "verdict": verdict,
                    "method": "teacher-forced",
                    "max_abs_logit_diff": max_diff,
                    "mean_top1_agreement": mean_agree,
                    "max_prefill_last_abs_diff": max_prefill,
                    "logit_tol": args.logit_tol,
                    "agree_tol": args.agree_tol,
                    "max_new_tokens": n_new,
                    "use_fla": use_fla,
                    "kv_cache_dtype": args.kv_cache_dtype,
                    "rows": rows,
                },
                f,
                indent=2,
            )
        print(f"[verify] wrote {args.json_out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
