#!/usr/bin/env python3
"""Prefill microbenchmark for the chunked Gated-DeltaNet kernel
(`fla.ops.gated_delta_rule.chunk_gated_delta_rule`), swept over sequence
length T at batch B=1 (the physics model says "prefill is compute-
bound ... chunked DeltaNet is cheap relative to a normal 27B"; this script
measures whether that holds in practice and at what TFLOP/s).

Shapes match the decode benchmark: 48 value heads / 16 key heads (GVA x3),
dk=dv=128, chunk_size=64 (fla's/HF's default). See `gdn_decode_bench.py` and
`common.py` for where these numbers come from.

Reports, per T:
  - wall time (ms) for one chunked-prefill forward
  - TFLOP/s, using an analytical FLOP count for the chunked delta-rule
    algorithm (see `common.gdn_chunk_algorithmic_flops` docstring for the
    exact derivation -- it is NOT a generic "2*FLOPs of a dense matmul"
    estimate, it's counted term-by-term from the reference chunked-delta-rule
    recurrence, so it is comparable across implementations that do the same
    asymptotic work).
  - HBM-bound floor: q/k/v/g/beta read + output write, at --gpu-bw-gbps

Falls back to the pure-torch `torch_chunk_gated_delta_rule` (copied from
engine/reference/modeling_qwen3_5.py, the HF fallback path) if `fla` isn't
importable, clearly labeled, so the sweep still produces numbers (much
slower, O(T) python-level chunk loop -- expect this to be very slow at
T=8192, that's expected and part of what it's measuring: the gap fla closes).

Example (remote):
    python gdn_prefill_bench.py --out /home/qwenfast-results/gdn_prefill.json
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Optional

import common


def build_prefill_inputs(B: int, T: int, dtype, device, seed: int = 0):
    import torch

    g = torch.Generator(device="cpu").manual_seed(seed)

    def randn(*shape, gen=g):
        return torch.randn(*shape, generator=gen).to(device=device, dtype=torch.float32)

    q_raw = randn(B, T, common.GDN_NUM_K_HEADS, common.GDN_HEAD_K_DIM)
    k_raw = randn(B, T, common.GDN_NUM_K_HEADS, common.GDN_HEAD_K_DIM)
    v = randn(B, T, common.GDN_NUM_V_HEADS, common.GDN_HEAD_V_DIM).to(dtype)

    A_log = torch.log(torch.empty(common.GDN_NUM_V_HEADS).uniform_(0.01, 16, generator=g)).to(device)
    dt_bias = torch.ones(common.GDN_NUM_V_HEADS, device=device)
    a_raw = randn(B, T, common.GDN_NUM_V_HEADS)
    b_raw = randn(B, T, common.GDN_NUM_V_HEADS)

    beta = b_raw.sigmoid()
    gate = -A_log.float().exp() * torch.nn.functional.softplus(a_raw.float() + dt_bias)

    rep = common.GDN_GVA_GROUP_SIZE
    q = q_raw.repeat_interleave(rep, dim=2).to(dtype)
    k = k_raw.repeat_interleave(rep, dim=2).to(dtype)

    return dict(q=q, k=k, v=v, g=gate, beta=beta)


def bench_fla_chunk(B: int, T: int, warmup: int, iters: int, device: str, chunk_size: int) -> dict[str, Any]:
    import torch
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    dtype = torch.bfloat16
    inputs = build_prefill_inputs(B, T, dtype, device)

    with torch.no_grad():
        def step():
            out, _ = chunk_gated_delta_rule(
                inputs["q"], inputs["k"], inputs["v"],
                g=inputs["g"], beta=inputs["beta"],
                initial_state=None, output_final_state=False,
                use_qk_l2norm_in_kernel=True,
            )
            return out

        timing = common.cuda_time_fn(step, warmup, iters)

    flops = common.gdn_chunk_algorithmic_flops(B, T, common.GDN_NUM_V_HEADS,
                                                 common.GDN_HEAD_K_DIM, common.GDN_HEAD_V_DIM, chunk_size)
    mean_s = timing["mean_us"] * 1e-6
    io_bytes = _io_bytes(B, T, dtype_bytes=2)
    return {
        "variant": "fla_chunk_gated_delta_rule",
        "batch": B, "seq_len": T, "chunk_size": chunk_size,
        **timing,
        "mean_ms": timing["mean_us"] / 1000.0,
        "algorithmic_flops": flops,
        "achieved_tflops": common.tflops(flops, mean_s),
        "io_bytes_qkvgbeta_plus_out": io_bytes,
        "hbm_bound_floor_ms": (io_bytes / 1e9) / (common.GPU_BW_GBPS_DEFAULT) * 1000.0,
    }


def _io_bytes(B: int, T: int, dtype_bytes: int) -> int:
    HV, dk, dv = common.GDN_NUM_V_HEADS, common.GDN_HEAD_K_DIM, common.GDN_HEAD_V_DIM
    q = B * T * HV * dk * dtype_bytes
    k = B * T * HV * dk * dtype_bytes
    v = B * T * HV * dv * dtype_bytes
    g = B * T * HV * 4  # gate kept fp32
    beta = B * T * HV * 4
    out = B * T * HV * dv * dtype_bytes
    return q + k + v + g + beta + out


def _l2norm(x, dim: int = -1, eps: float = 1e-6):
    import torch
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def _torch_chunk_gated_delta_rule_ref(query, key, value, g, beta, chunk_size=64,
                                        initial_state=None, output_final_state=False,
                                        use_qk_l2norm_in_kernel=False):
    """Direct, dependency-free port of
    `engine/reference/modeling_qwen3_5.py::torch_chunk_gated_delta_rule`
    (the HF no-fused-kernel fallback path), reproduced here verbatim rather
    than imported from the reference file. The reference file uses package-
    relative imports (`from ... import ...`) that assume it's being loaded as
    part of an installed `transformers.models.qwen3_5` package -- loading it
    standalone via a file path breaks those imports. Porting the ~80-line
    function body directly avoids that fragility entirely while staying
    byte-for-byte identical math."""
    import torch
    import torch.nn.functional as F

    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def bench_torch_chunk_fallback(B: int, T: int, warmup: int, iters: int, device: str, chunk_size: int) -> dict[str, Any]:
    """Uses `_torch_chunk_gated_delta_rule_ref`, a dependency-free direct port
    of the HF reference's no-fused-kernel fallback path (see that function's
    docstring for why it's a port rather than an import)."""
    import torch

    dtype = torch.bfloat16
    inputs = build_prefill_inputs(B, T, dtype, device)

    with torch.no_grad():
        def step():
            out, _ = _torch_chunk_gated_delta_rule_ref(
                inputs["q"], inputs["k"], inputs["v"],
                g=inputs["g"], beta=inputs["beta"], chunk_size=chunk_size,
                initial_state=None, output_final_state=False,
                use_qk_l2norm_in_kernel=True,
            )
            return out

        timing = common.cuda_time_fn(step, warmup, min(iters, 5))  # this path is O(chunk_size) python loop -- slow

    flops = common.gdn_chunk_algorithmic_flops(B, T, common.GDN_NUM_V_HEADS,
                                                 common.GDN_HEAD_K_DIM, common.GDN_HEAD_V_DIM, chunk_size)
    mean_s = timing["mean_us"] * 1e-6
    return {
        "variant": "torch_chunk_fallback",
        "batch": B, "seq_len": T, "chunk_size": chunk_size,
        **timing,
        "mean_ms": timing["mean_us"] / 1000.0,
        "algorithmic_flops": flops,
        "achieved_tflops": common.tflops(flops, mean_s),
        "note": "HF reference torch fallback -- has an O(chunk_size) python-level "
                "inner loop per chunk (the UT-transform), expect this to be much "
                "slower than the fla Triton kernel, especially at large T.",
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(parser, default_out="gdn_prefill_bench_results")
    parser.add_argument("--seq-lens", type=str, default=None,
                         help=f"Comma-separated T sweep (default: {','.join(str(t) for t in common.PREFILL_T_SWEEP)}).")
    parser.add_argument("--prefill-batch", type=int, default=common.PREFILL_BATCH)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--skip-torch-fallback", action="store_true",
                         help="Skip the slow pure-torch fallback path entirely.")
    args = parser.parse_args(argv)

    seq_lens = [int(x) for x in args.seq_lens.split(",")] if args.seq_lens else common.PREFILL_T_SWEEP
    B = args.prefill_batch

    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False -- this benchmark must run on a CUDA GPU.")
    except Exception as exc:  # noqa: BLE001
        payload = {"error": f"CUDA/torch unavailable: {exc}", "env": common.env_metadata(args.device)}
        json_path, md_path = common.write_results(args.out, payload, [f"# gdn_prefill_bench\n\nFAILED: {exc}\n"])
        print(f"[gdn_prefill_bench] FAILED (no CUDA): wrote {json_path}", file=sys.stderr)
        return 1

    device = args.device
    torch.cuda.set_device(device)

    results: dict[str, Any] = {"env": common.env_metadata(device), "args": vars(args), "per_t": []}
    md_rows_fla, md_rows_torch = [], []

    fla_ok, fla_err = common.try_import("fla.ops.gated_delta_rule", "chunk_gated_delta_rule")
    if fla_ok is None:
        print(f"[gdn_prefill_bench] WARNING: fla not importable ({fla_err}), "
              "relying entirely on the slow torch fallback.", file=sys.stderr)

    for T in seq_lens:
        print(f"[gdn_prefill_bench] B={B} T={T} ...", file=sys.stderr)
        row: dict[str, Any] = {"batch": B, "seq_len": T}

        row["fla_chunk"] = common.safe_run(
            lambda B=B, T=T: bench_fla_chunk(B, T, args.warmup, args.iters, device, args.chunk_size),
            label=f"fla_chunk_T{T}")
        if row["fla_chunk"]["status"] == "ok":
            r = row["fla_chunk"]
            md_rows_fla.append({"T": T, "mean_ms": r["mean_ms"], "TFLOP/s": r["achieved_tflops"],
                                 "HBM_floor_ms": r["hbm_bound_floor_ms"]})

        if not args.skip_torch_fallback:
            row["torch_fallback"] = common.safe_run(
                lambda B=B, T=T: bench_torch_chunk_fallback(B, T, args.warmup, args.iters, device, args.chunk_size),
                label=f"torch_chunk_T{T}")
            if row["torch_fallback"]["status"] == "ok":
                r = row["torch_fallback"]
                md_rows_torch.append({"T": T, "mean_ms": r["mean_ms"], "TFLOP/s": r["achieved_tflops"]})

        results["per_t"].append(row)

    md = [
        "# gdn_prefill_bench results",
        f"GPU: {results['env'].get('gpu_name', '?')} | fla "
        f"{results['env'].get('fla_version', 'MISSING: ' + str(results['env'].get('fla_import_error')))} | "
        f"B={B}, chunk_size={args.chunk_size}",
        "## fla.chunk_gated_delta_rule",
        common.dict_rows_to_markdown(md_rows_fla, ["T", "mean_ms", "TFLOP/s", "HBM_floor_ms"]),
        "## torch fallback (HF reference impl, no fused kernel)",
        common.dict_rows_to_markdown(md_rows_torch, ["T", "mean_ms", "TFLOP/s"]),
    ]

    json_path, md_path = common.write_results(args.out, results, md)
    print(f"[gdn_prefill_bench] wrote {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
