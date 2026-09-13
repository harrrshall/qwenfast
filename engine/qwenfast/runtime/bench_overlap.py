"""Overlap probe: does a graphed decode step hide under a graphed prefill chunk?

Motivation
----------
Decomposing a mixed step at chunk 1,024 / conc 256 shows that
**23.9 µs of every prefill token** — 24.5 ms of a 143 ms step — is not the
chunk at all: it is "attention over 256 sequences at ~2,139 context, the GDN
decode half, the per-layer norms and the sampler", i.e. one whole decode
step's worth of work, paid inside every mixed step. A decode-only step at
conc 256 is 42.7 ms of almost pure *memory* traffic (28 GiB of fp8 weights,
~36 GB of KV, the SSM state pool); a prefill chunk is ~58 % of fp8 dense
compute peak and reads those same 28 GiB of weights **once per chunk**, which
at a 573 ms chunk is 49 GB/s against an H200's ~4.8 TB/s.

So the two halves are bound by different resources and, run one after the
other, they *add*. This module measures what happens when they do not: the
graphed prefill chunk on one CUDA stream, the graphed decode step on another,
both in flight at once.

Why the existing probe does not answer this
-------------------------------------------
``profile_serving.decode_overlap_probe`` already overlaps a decode graph with
a chunk, and reports ``hidden_pct_of_decode`` of about **84 %** at chunk 1,024 /
conc 256. That number is not usable, for two reasons this module fixes:

1. **Its chunk is eager.** An eager mixed step's host issue measures
   about 142 ms against 109 ms of device work, so the GPU is *idle* for a
   third of an eager chunk, so a decode graph "hiding" in it is hiding in
   host-launch bubbles, not under compute. Here the chunk is a single
   ``MixedGraphRunner`` replay (host issue 6-10 ms).
2. **Its decode rows have one page of context.** ``ctx_len = page_size`` (16
   tokens), so the KV read — the larger half of what makes a decode step
   bandwidth-bound — is absent. Here ``--overlap-decode-ctx`` defaults to the
   real serving context.

What is measured
----------------
Five arms over the same batch, at a fixed shape (``chunk_tokens`` prefill
tokens in ``n_segments`` plan rows, ``B`` decode rows at ``decode_ctx``
context):

======================  =====================================================
``prefill_ms``          the prefill graph alone (one replay)
``decode_ms``           the decode graph alone (one replay)
``serial_ms``           both, back to back, on one stream
``overlap_ms``          both, forked onto two streams and joined
``serial_n_ms`` /       one prefill chunk against ``n`` decode replays, the
``overlap_n_ms``        shape a *scheduler* would run: at p/o 4.28 a chunk of
                        C tokens is owed C/4.28 output tokens, i.e.
                        ``ceil(C / (4.28 * B))`` decode steps
======================  =====================================================

``hidden_ms = serial_ms - overlap_ms`` is the decode time the prefill
absorbed; ``dilation`` is what the prefill half cost *because* the decode was
running (measured directly, with CUDA events recorded on each stream inside
the overlapped run — that is the contention number, and it is the one that
decides whether the idea survives).

**Correctness is not checked here.** The two graphs write disjoint slots by
construction (a request is either prefilling or decoding), and under
``--overlap`` they get separate mempools and separate FlashInfer workspaces —
but this module only reports wall time. Parity is
``tests/test_overlap.py`` and the ledger's own greedy-completion check.
"""

from __future__ import annotations

import json
import math
import time
from typing import Dict, List, Optional, Sequence

import torch

from .fused_model import make_mixed_batch, make_prefill_batch
from .mixed_graphs import MixedGraphRunner, pad_mixed_step, reset_scratch_state
from . import profile_serving as ps


# --------------------------------------------------------------------------- #
# 1. the fixture: one padded chunk + B decode rows on disjoint slots
# --------------------------------------------------------------------------- #
class OverlapFixture:
    """Everything the arms replay, built once and reset between repeats."""

    def __init__(self, comps, *, prompt_len: int, decode_rows: int, decode_ctx: int):
        self.comps = comps
        model = comps.model
        self.model = model
        self.runner: MixedGraphRunner = comps.mixed
        self.decoder = comps.decoder
        spec = self.runner.spec
        self.spec = spec
        self.chunk_tokens = int(self.runner.chunk_tokens)

        lens = ps.chunk_shape(spec.budget, prompt_len)[: spec.max_segments]
        self.n_seg = len(lens)
        self.seg_lens = lens

        n_slots = int(model.n_slots)
        b = min(int(decode_rows), n_slots - self.n_seg)
        if b < 1:
            raise ValueError(f"no free slots for decode rows ({n_slots} slots, "
                             f"{self.n_seg} taken by the chunk)")
        self.b = b
        self.bucket = self.decoder.bucket_for(b)
        self.decode_ctx = int(decode_ctx)
        self.dec_slots = list(range(self.n_seg, self.n_seg + b))

        gen = torch.Generator(device="cpu").manual_seed(1234)
        real_ids: List[List[int]] = []
        for sl, n in zip(range(self.n_seg), lens):
            model.reset_slot(sl)
            model.kv_pool.ensure_capacity(sl, int(n) + 1)
            real_ids.append(torch.randint(
                0, model.config.vocab_size, (int(n),), generator=gen, dtype=torch.int64
            ).tolist())
        for sl in self.dec_slots:
            model.reset_slot(sl)
            model.kv_pool.ensure_capacity(sl, self.decode_ctx + 1)

        # -- the prefill half: a padded mixed step with ONE padding decode row.
        # `MixedGraphRunner` has no zero-row shape (a mixed step always has at
        # least one decode row), and one scratch row on a 1,024-token chunk is
        # 0.1 % of the step's M -- so this is the graphed prefill chunk.
        scratch = model.scratch_slot
        padded = pad_mixed_step(
            real_ids, [0] * self.n_seg, list(range(self.n_seg)),
            [scratch], [0], [0], spec,
        )
        self.pbatch = make_prefill_batch(
            padded.token_ids, padded.start_positions, padded.slots, model.device
        )
        self.mixed_batch = make_mixed_batch(
            self.pbatch, padded.decode_slots, padded.decode_token_ids,
            padded.decode_positions, model.device,
        )

        # -- the *fused* arm (the fused mixed step), when its bucket was
        # captured too: the same chunk and the same decode rows, in one
        # row-concatenated forward. This is the arm `--overlap` has to beat,
        # not the serial one.
        self.mixed_bucket = spec.bucket_for(b)
        self.fused_batch = None
        if (self.mixed_bucket is not None
                and self.mixed_bucket != 1
                and self.mixed_bucket in self.runner.buckets):
            fused = pad_mixed_step(
                real_ids, [0] * self.n_seg, list(range(self.n_seg)),
                self.dec_slots, [1] * b, [self.decode_ctx] * b, spec,
            )
            fbatch = make_prefill_batch(
                fused.token_ids, fused.start_positions, fused.slots, model.device
            )
            self.fused_batch = make_mixed_batch(
                fbatch, fused.decode_slots, fused.decode_token_ids,
                fused.decode_positions, model.device,
            )
        else:
            self.mixed_bucket = None

        # -- the decode half: the ordinary graphed decode step's inputs.
        buf = self.decoder.buf
        pad = self.bucket - b
        self.padded_slots = self.dec_slots + [scratch] * pad
        buf.host["input_ids"][: self.bucket] = torch.tensor(
            [1] * b + [0] * pad, dtype=torch.int32)
        buf.host["positions"][: self.bucket] = torch.tensor(
            [self.decode_ctx] * b + [0] * pad, dtype=torch.int32)
        buf.host["slot_ids"][: self.bucket] = torch.tensor(
            self.padded_slots, dtype=torch.int32)
        buf.host["temperature"][: self.bucket] = torch.zeros(self.bucket, dtype=torch.float32)
        buf.host["top_p"][: self.bucket] = torch.ones(self.bucket, dtype=torch.float32)
        buf.host["top_k"][: self.bucket] = torch.zeros(self.bucket, dtype=torch.float32)
        buf.upload(["input_ids", "positions", "slot_ids", "temperature", "top_p", "top_k"])
        self.seq_lens = [self.decode_ctx + 1] * b + [1] * pad

        self._dec_idx = torch.tensor(self.dec_slots, dtype=torch.long, device=model.device)

    # -- plans (host-side, run once; the arms replay only) ------------------- #
    def plan(self) -> None:
        """Load + plan both halves. Host-side, outside every clock.

        Both plans stand for the whole measurement: neither graph's *shape*
        changes between repeats, and a replay never re-plans. That is exactly
        what the scheduler does too — it plans once per step and replays once.
        """
        self.runner._load(self.runner._shapes[self.runner.buckets[0]], self.mixed_batch)  # noqa: SLF001
        self.model.attn.plan_mixed_graph(
            self.mixed_batch.plan_slots, self.mixed_batch.plan_q_lens,
            self.mixed_batch.plan_kv_lens,
        )
        self.model.attn.plan_decode(self.padded_slots, self.bucket, seq_lens=self.seq_lens)

    def plan_fused(self) -> None:
        """Re-plan for the fused arm. Mutually exclusive with :meth:`plan` --
        both halves' plans live in the same FlashInfer wrappers, so the fused
        arm is timed in its own pass rather than interleaved with the others."""
        self.runner.prepare_step(self.fused_batch)

    @property
    def fused_graph(self):
        sh = self.runner._shapes[self.mixed_bucket]  # noqa: SLF001
        return sh.graphs[0]

    @property
    def prefill_graph(self):
        sh = self.runner._shapes[self.runner.buckets[0]]  # noqa: SLF001
        return sh.graphs[0]

    @property
    def decode_graph(self):
        return self.decoder._graphs[self.bucket]  # noqa: SLF001

    def reset(self) -> None:
        """Put both halves back to "this step has not run yet".

        The chunk's slots need their pages back (``prefill_forward`` appends
        KV and advances ``seq_len``; without this every repeat measures a
        longer attention than the last and the page pool drains) and the
        decode rows need their committed length pinned, or their KV grows a
        token per replay and the plan built above stops describing them.
        Run **outside** the clock -- ``reset_chunk_slots`` is a Python loop
        and the ``seq_len`` write is a kernel.
        """
        ps.reset_chunk_slots(self.model, self.pbatch)
        self.model.kv_pool.seq_len[self._dec_idx] = self.decode_ctx
        reset_scratch_state(self.model)


# --------------------------------------------------------------------------- #
# 2. the arms
# --------------------------------------------------------------------------- #
def _timeit(fix: OverlapFixture, fn, n: int, warmup: int) -> float:
    """Mean wall ms of ``fn``, with the fixture reset outside the clock."""
    for _ in range(warmup):
        fix.reset()
        torch.cuda.synchronize()
        fn()
        torch.cuda.synchronize()
    total = 0.0
    for _ in range(n):
        fix.reset()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        total += (time.perf_counter() - t0) * 1e3
    return total / max(n, 1)


def overlap_probe(comps, *, prompt_len: int = 2139, decode_rows: int = 256,
                  decode_ctx: int = 0, steps: int = 5, warmup: int = 2,
                  decode_repeats: int = 0, po_ratio: float = 4.278,
                  priority: int = 0) -> Dict:
    """The five arms of the module docstring, plus the contention split."""
    if comps.model is None or comps.model.device.type != "cuda":
        return {"skipped": "cuda only"}
    if comps.mixed is None or not comps.mixed.captured:
        return {"skipped": "needs --mixed-forward --mixed-graphs (a captured "
                           "MixedGraphRunner is the graphed prefill chunk)"}
    if not comps.decoder._graphs:  # noqa: SLF001
        return {"skipped": "needs captured decode graphs"}

    try:
        fix = OverlapFixture(comps, prompt_len=prompt_len, decode_rows=decode_rows,
                             decode_ctx=decode_ctx or prompt_len)
        fix.plan()
    except Exception as exc:  # noqa: BLE001 -- a probe that cannot run is data
        return {"error": f"setup: {type(exc).__name__}: {exc}"[:400]}

    pg, dg = fix.prefill_graph, fix.decode_graph
    main = torch.cuda.current_stream()
    side = torch.cuda.Stream(priority=int(priority))

    # How many decode steps a scheduler owes this chunk. At the sweep's
    # 2,139-in / 500-out geometry every output token is owed 4.278 prefill
    # tokens, so a C-token chunk buys C/4.278 output tokens = that many
    # divided by the decode batch.
    n_dec = int(decode_repeats) or max(
        1, int(math.ceil(fix.chunk_tokens / (po_ratio * fix.b)))
    )

    def prefill_only() -> None:
        pg.replay()

    def decode_only() -> None:
        dg.replay()

    def serial() -> None:
        pg.replay()
        dg.replay()

    def overlap() -> None:
        side.wait_stream(main)
        with torch.cuda.stream(side):
            dg.replay()
        pg.replay()
        main.wait_stream(side)

    def decode_n() -> None:
        for _ in range(n_dec):
            dg.replay()

    def serial_n() -> None:
        pg.replay()
        for _ in range(n_dec):
            dg.replay()

    def overlap_n() -> None:
        side.wait_stream(main)
        with torch.cuda.stream(side):
            for _ in range(n_dec):
                dg.replay()
        pg.replay()
        main.wait_stream(side)

    # -- the contention split: time each half *inside* the overlapped run --- #
    ev = {k: torch.cuda.Event(enable_timing=True) for k in
          ("p0", "p1", "d0", "d1", "start")}

    def overlap_timed() -> None:
        side.wait_stream(main)
        ev["start"].record(main)
        with torch.cuda.stream(side):
            ev["d0"].record(side)
            for _ in range(n_dec):
                dg.replay()
            ev["d1"].record(side)
        ev["p0"].record(main)
        pg.replay()
        ev["p1"].record(main)
        main.wait_stream(side)

    try:
        res: Dict[str, object] = {
            "chunk_tokens": fix.chunk_tokens,
            "n_segments": fix.n_seg,
            "decode_rows": fix.b,
            "decode_bucket": fix.bucket,
            "decode_ctx": fix.decode_ctx,
            "n_decode_replays": n_dec,
            "stream_priority": int(priority),
        }
        res["prefill_ms"] = round(_timeit(fix, prefill_only, steps, warmup), 3)
        res["decode_ms"] = round(_timeit(fix, decode_only, max(steps * 2, 6), warmup), 3)
        res["serial_ms"] = round(_timeit(fix, serial, steps, warmup), 3)
        res["overlap_ms"] = round(_timeit(fix, overlap, steps, warmup), 3)
        res["decode_n_ms"] = round(_timeit(fix, decode_n, steps, warmup), 3)
        res["serial_n_ms"] = round(_timeit(fix, serial_n, steps, warmup), 3)
        res["overlap_n_ms"] = round(_timeit(fix, overlap_n, steps, warmup), 3)
        if fix.fused_batch is not None:
            fix.plan_fused()
            fg = fix.fused_graph
            res["mixed_bucket"] = fix.mixed_bucket
            res["mixed_ms"] = round(_timeit(fix, lambda: fg.replay(), steps, warmup), 3)
            fix.plan()  # put the other arms' plans back
        _timeit(fix, overlap_timed, 2, 1)
        res["overlap_n_prefill_dev_ms"] = round(ev["p0"].elapsed_time(ev["p1"]), 3)
        res["overlap_n_decode_dev_ms"] = round(ev["d0"].elapsed_time(ev["d1"]), 3)
    except Exception as exc:  # noqa: BLE001
        res["error"] = f"{type(exc).__name__}: {exc}"[:400]
        return res

    p, d = float(res["prefill_ms"]), float(res["decode_ms"])
    s, o = float(res["serial_ms"]), float(res["overlap_ms"])
    sn, on = float(res["serial_n_ms"]), float(res["overlap_n_ms"])
    res["hidden_ms"] = round(s - o, 3)
    res["hidden_pct_of_decode"] = round(100.0 * (s - o) / d, 1) if d else 0.0
    res["hidden_n_ms"] = round(sn - on, 3)
    res["hidden_n_pct_of_decode"] = round(
        100.0 * (sn - on) / float(res["decode_n_ms"]), 1) if res["decode_n_ms"] else 0.0
    # The ceiling: if the decode half were free, one chunk + its decode steps
    # would cost `max(prefill, n * decode)`.
    res["ideal_n_ms"] = round(max(p, float(res["decode_n_ms"])), 3)
    res["overlap_vs_ideal"] = round(on / float(res["ideal_n_ms"]), 3) if res["ideal_n_ms"] else 0.0
    res["prefill_dilation_pct"] = round(
        100.0 * (float(res["overlap_n_prefill_dev_ms"]) - p) / p, 1) if p else 0.0
    # What the two arms project at the sweep's geometry: one chunk plus the
    # decode steps it owes emits `n_dec * B` output tokens.
    out_tok = n_dec * fix.b
    res["projected_out_tok_s_serial"] = round(out_tok / (sn / 1e3), 1) if sn else 0.0
    res["projected_out_tok_s_overlap"] = round(out_tok / (on / 1e3), 1) if on else 0.0
    res["projected_out_tok_s_ideal"] = round(
        out_tok / (float(res["ideal_n_ms"]) / 1e3), 1) if res["ideal_n_ms"] else 0.0
    if res.get("mixed_ms"):
        m = float(res["mixed_ms"])
        # A fused step carries one chunk and B decode rows; the chunk is owed
        # `n_dec` decode steps in total, of which the fused step *is* one.
        cycle = m + max(0, n_dec - 1) * d
        res["mixed_vs_serial_pct"] = round(100.0 * (s - m) / s, 1) if s else 0.0
        res["projected_out_tok_s_mixed"] = round(out_tok / (cycle / 1e3), 1) if cycle else 0.0
    # The steady-state model: at `po_ratio` prefill tokens
    # per output token, one output token costs one decode row plus its share
    # of a chunk. Independent of how the two are interleaved -- which is the
    # point.
    per_out_s = (d / 1e3) / fix.b + (po_ratio * (p / 1e3) / fix.chunk_tokens)
    res["steady_state_out_tok_s"] = round(1.0 / per_out_s, 1) if per_out_s else 0.0
    res["prefill_tok_s"] = round(fix.chunk_tokens / (p / 1e3), 1) if p else 0.0
    res["decode_tok_s"] = round(fix.b / (d / 1e3), 1) if d else 0.0
    return res


# --------------------------------------------------------------------------- #
# 3. CLI
# --------------------------------------------------------------------------- #
def build_arg_parser():
    p = ps.build_arg_parser()
    p.add_argument("--overlap-decode-ctx", type=int, nargs="+", default=[0],
                   help="committed context per decode row (0 = --input-len); "
                        "several values sweep it inside one build. A probe "
                        "that uses one page (16 tokens) measures a decode step "
                        "with no KV read at all, which makes the fused mixed "
                        "step look cheaper than it is.")
    p.add_argument("--overlap-decode-repeats", type=int, default=0,
                   help="decode replays per prefill chunk (0 = the number the "
                        "sweep's prompt/output ratio implies)")
    p.add_argument("--overlap-priorities", type=int, nargs="+", default=[0],
                   help="CUDA stream priorities to try for the decode half")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    from .preset import apply_preset

    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    args = apply_preset(args, parser, argv=list(argv) if argv is not None else None)
    # The probe *is* the overlap configuration: separate mempool for the
    # prefill graph, separate FlashInfer workspace for the decode wrappers.
    args.mixed_forward = True
    args.mixed_graphs = True
    args.overlap_streams = True

    out: List[Dict] = []
    for conc in args.concurrency:
        comps = None
        try:
            comps = ps.build(args, conc)
            for ctx in args.overlap_decode_ctx:
                for prio in args.overlap_priorities:
                    r = overlap_probe(
                        comps,
                        prompt_len=args.input_len,
                        decode_rows=conc,
                        decode_ctx=ctx,
                        steps=args.chunk_steps,
                        decode_repeats=args.overlap_decode_repeats,
                        po_ratio=max(1.0, args.input_len / max(args.output_len, 1)),
                        priority=prio,
                    )
                    r["concurrency"] = conc
                    out.append(r)
                    print(json.dumps(r, indent=2), flush=True)
        finally:
            ps.teardown(comps)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"schema": "qwenfast.bench_overlap/1",
                       "args": vars(args), "probes": out}, fh, indent=2, default=str)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
