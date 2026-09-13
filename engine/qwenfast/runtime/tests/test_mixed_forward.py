"""CPU tests for the mixed prefill+decode step.

The claim under test, in one sentence: **one mixed step computes exactly what
a prefill chunk followed by a decode step computes.** Everything here is an
instance of that, at two levels.

* :class:`TestMixedForwardMatchesSeparate` — the *model* level. Snapshot every
  pool, run ``mixed_forward``, snapshot again; restore, run ``prefill_forward``
  then ``decode_forward``, and compare logits, SSM state, conv rings and KV,
  row for row and slot for slot. This is the level where a mistake would be a
  silent numerical one (a decode row reading the wrong state slot, an
  attention plan one key short, a ``g``/``beta`` slice off by the segment
  boundary), so it is checked directly rather than through generated text.
  Covered shapes: one segment, several segments, a segment that *continues* a
  prompt from a previous chunk (``start_pos > 0``), and a segment whose prompt
  finishes inside the chunk.

* :class:`TestMixedSchedulerEquivalence` — the *scheduler* level. Two engines
  built from the same seed, one with ``mixed_forward`` on, driven over the same
  requests to completion: the emitted token ids must be identical. This is what
  says the policy (which requests are in which half of which step) cannot
  change an answer, even though it changes every step boundary.

Numerics, measured rather than assumed: the two paths run the *same kernels*
on the same rows, and the only difference is the ``M`` of the shared GEMMs
(``T_pre + B_dec`` at once, vs ``T_pre`` and ``B_dec`` apart). That is **not**
bit-exact — a one-off run with ``EXACT = 0.0`` reports a worst element of
**1.2e-7** on fp32 logits, i.e. a single ULP, from oneDNN picking a different
blocking for a different ``M``. It *is* exact where it has to be: the greedy
argmax over those logits is identical row for row (asserted below), which is
what actually reaches a client, and every scheduler-level test compares emitted
token ids with no tolerance at all.

:data:`EXACT` is therefore set two orders of magnitude above the observed
reassociation and three below anything a real defect could hide in — a swapped
state slot, an attention plan one key short or a ``g``/``beta`` slice off the
segment boundary all move logits by O(1), not by 1e-7. :func:`assert_same`
prints the worst element and its index on failure, so the two cases are never
confused.

Run::

    python -m unittest discover -s engine/qwenfast/runtime/tests -v
    pytest engine/qwenfast/runtime/tests/test_mixed_forward.py -v
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))  # -> engine/
sys.path.insert(0, _HERE)  # -> this dir, for the tiny-model builders

from qwenfast.runtime.fused_model import (  # noqa: E402
    DeviceBuffers,
    FusedQwenForCausalLM,
    make_mixed_batch,
    make_prefill_batch,
)
from qwenfast.runtime.graphs import GraphedDecoder  # noqa: E402
from qwenfast.runtime.scheduler import GenParams, Request, Scheduler  # noqa: E402
from qwenfast.runtime.serve import plan_memory  # noqa: E402
from qwenfast.runtime.spec_decode import SpecConfig, build_spec_decoder  # noqa: E402

from test_serving import (  # noqa: E402
    CPU,
    VOCAB,
    build_scheduler,
    build_tiny_model,
    build_tiny_mtp_model,
    gen_params,
    serving_rt,
)

#: fp32 on CPU, same kernels, same rows. Observed worst case 1.2e-7 (one ULP,
#: from the shared GEMMs' differing ``M``); anything above this is a bug, not
#: reassociation. Kept as a named constant so a future relaxation has to be a
#: deliberate edit with a reason next to it -- and so it can be set to 0.0 for
#: a one-off "how exact is it really" run, which is where the 1.2e-7 came from.
EXACT = 1e-6


def assert_same(case: unittest.TestCase, got: torch.Tensor, want: torch.Tensor, what: str) -> None:
    """``got == want`` to :data:`EXACT`, reporting the worst element if not."""
    case.assertEqual(tuple(got.shape), tuple(want.shape), f"{what}: shape")
    d = (got.float() - want.float()).abs()
    worst = float(d.max()) if d.numel() else 0.0
    case.assertLessEqual(
        worst, EXACT,
        f"{what}: max |mixed - separate| = {worst:.3e} > {EXACT:.0e} "
        f"(at flat index {int(d.argmax()) if d.numel() else -1})",
    )


# =========================================================================== #
# 0. a model + pool snapshot harness
# =========================================================================== #
def build_model(seed: int = 7, **rt_overrides):
    model, _cfg = build_tiny_model(seed)
    rt = serving_rt(**rt_overrides)
    return FusedQwenForCausalLM.from_m0_module(model, rt), rt


def snapshot(model) -> dict:
    """Every mutable pool a step writes, cloned.

    Deliberately *not* the page-table/allocator bookkeeping: both paths in
    these tests run after the same ``ensure_capacity`` calls, so they share a
    page assignment and only the contents can differ. Restoring the allocator's
    free list is impossible from outside it, which is exactly why capacity is
    reserved once, before the snapshot, and never again."""
    return {
        "state": model.state_pool.detach().clone(),
        "conv": model.conv_pool.detach().clone(),
        "kv": model.kv_pool.kv.detach().clone(),
        "seq_len": model.kv_pool.seq_len.detach().clone(),
    }


def restore(model, snap: dict) -> None:
    model.state_pool.copy_(snap["state"])
    model.conv_pool.copy_(snap["conv"])
    model.kv_pool.kv.copy_(snap["kv"])
    model.kv_pool.seq_len.copy_(snap["seq_len"])


def prime_slot(model, slot: int, ids, start: int = 0, *, extra: int = 0) -> None:
    """Give ``slot`` a real committed prefix: SSM state, conv ring and KV.

    A decode row whose state is all zeros would pass a mixed-vs-separate test
    for the wrong reason (both halves read zeros), so every decode row in these
    tests is a sequence that has actually been prefilled.

    ``extra`` is the page headroom the *next* step needs -- 1 for a row that is
    about to decode, which is exactly what ``Scheduler._admit_decode_rows``
    reserves (``extra_tokens=1``). Without it a prompt whose length lands on a
    page boundary has no page for position ``L`` and ``append_kv`` refuses,
    which is a real precondition, not a test artefact."""
    model.reset_slot(slot)
    model.kv_pool.ensure_capacity(slot, start + len(ids) + extra)
    batch = make_prefill_batch([list(ids)], [start], [slot], CPU)
    model.prefill_forward(batch, all_logits=False)


def toks(n: int, lo: int = 3, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(lo, VOCAB, (n,), generator=g, dtype=torch.int64).tolist()


# =========================================================================== #
# 1. the model level
# =========================================================================== #
class TestMixedForwardMatchesSeparate(unittest.TestCase):
    """``mixed_forward`` == ``prefill_forward`` then ``decode_forward``."""

    def _run_case(self, chunk, decode, *, seed: int = 7):
        """``chunk``: ``[(slot, ids, start_pos)]``. ``decode``: ``[(slot, prompt)]``.

        Returns nothing; asserts. Both paths start from the same snapshot and
        the same page assignment (capacity is reserved once, up front).
        """
        model, rt = build_model(seed)
        dec_slots = [s for s, _ in decode]
        dec_positions = []
        for slot, prompt in decode:
            # `+ 1`: this row is about to write position `len(prompt)`.
            prime_slot(model, slot, prompt, extra=1)
            dec_positions.append(len(prompt))
        dec_tokens = [7 + i for i in range(len(decode))]

        for slot, ids, start in chunk:
            if start == 0:
                model.reset_slot(slot)
                model.kv_pool.ensure_capacity(slot, len(ids))
            else:
                # a chunk that *continues* a prompt: the previous chunk's
                # tokens are already committed, this one starts at `start`
                prime_slot(model, slot, toks(start, seed=slot + 100))
                model.kv_pool.ensure_capacity(slot, start + len(ids))

        base = snapshot(model)
        prefill_ids = [ids for _, ids, _ in chunk]
        prefill_starts = [st for _, _, st in chunk]
        prefill_slots = [sl for sl, _, _ in chunk]

        # -- path A: one mixed forward ----------------------------------- #
        pb = make_prefill_batch(prefill_ids, prefill_starts, prefill_slots, CPU)
        mb = make_mixed_batch(pb, dec_slots, dec_tokens, dec_positions, CPU)
        mixed_logits = model.mixed_forward(mb)
        after_mixed = snapshot(model)
        n_seqs = len(chunk)
        self.assertEqual(mixed_logits.shape[0], n_seqs + len(decode))

        # -- path B: a prefill chunk, then a decode step ------------------ #
        restore(model, base)
        pb2 = make_prefill_batch(prefill_ids, prefill_starts, prefill_slots, CPU)
        pre_logits = model.prefill_forward(pb2, all_logits=False)

        buf = DeviceBuffers(
            max_batch=rt.max_num_seqs, vocab_size=model.config.vocab_size,
            max_pages=rt.n_kv_pages, device=CPU,
        )
        b = len(decode)
        buf.host["input_ids"][:b] = torch.tensor(dec_tokens, dtype=torch.int32)
        buf.host["positions"][:b] = torch.tensor(dec_positions, dtype=torch.int32)
        buf.host["slot_ids"][:b] = torch.tensor(dec_slots, dtype=torch.int32)
        buf.upload(["input_ids", "positions", "slot_ids"])
        model.attn.plan_decode(dec_slots, b, seq_lens=[p + 1 for p in dec_positions])
        dec_logits = model.decode_forward(buf, b, write_logits=False)
        after_sep = snapshot(model)

        # -- the comparison ------------------------------------------------ #
        assert_same(self, mixed_logits[:n_seqs], pre_logits, "prefill logits")
        assert_same(self, mixed_logits[n_seqs:], dec_logits, "decode logits")
        for name in ("state", "conv", "seq_len"):
            assert_same(self, after_mixed[name], after_sep[name], f"{name} pool")
        # The whole KV tensor, not a per-slot gather: both paths ran after the
        # same `ensure_capacity` calls, so they share a page assignment and any
        # difference -- including one written to a page nobody owns -- shows up
        # here.
        assert_same(self, after_mixed["kv"], after_sep["kv"], "kv pages")
        # Greedy decoding is what the scheduler runs by default, so pin the
        # argmax too: it is the value that actually reaches a client.
        self.assertTrue(
            torch.equal(mixed_logits.argmax(-1),
                        torch.cat([pre_logits, dec_logits]).argmax(-1)),
            "greedy token ids differ between the mixed and the separate step",
        )

    def test_one_segment_plus_decode_rows(self):
        self._run_case(chunk=[(4, toks(9, seed=1), 0)],
                       decode=[(0, toks(6, seed=2)), (1, toks(5, seed=3))])

    def test_several_segments_in_one_chunk(self):
        """The real serving shape: a chunk is several varlen segments."""
        self._run_case(
            chunk=[(4, toks(7, seed=1), 0), (5, toks(5, seed=4), 0), (6, toks(3, seed=5), 0)],
            decode=[(0, toks(6, seed=2)), (1, toks(4, seed=3)), (2, toks(9, seed=6))],
        )

    def test_segment_continuing_a_previous_chunk(self):
        """``start_pos > 0``: the segment's SSM/conv state and KV prefix are
        already committed, so the chunk kernels must thread through the pool
        rather than start from zero -- and the decode rows beside them must not
        notice."""
        self._run_case(chunk=[(4, toks(6, seed=1), 5), (5, toks(4, seed=4), 3)],
                       decode=[(0, toks(6, seed=2)), (1, toks(5, seed=3))])

    def test_one_decode_row(self):
        """The degenerate decode half. ``B_dec == 1`` exercises the
        ``unsqueeze``/``reshape`` path in ``FusedGDN.mixed`` with no batch
        dimension to hide a mistake."""
        self._run_case(chunk=[(4, toks(8, seed=1), 0)], decode=[(0, toks(7, seed=2))])

    def test_many_decode_rows_one_short_segment(self):
        """The conc-256 shape in miniature: the decode half dominates the row
        count, so the chunk is the small part of ``M``."""
        self._run_case(
            chunk=[(6, toks(3, seed=1), 0)],
            decode=[(i, toks(4 + i, seed=10 + i)) for i in range(5)],
        )

    def test_rejects_an_empty_half(self):
        model, _ = build_model()
        model.reset_slot(3)
        model.kv_pool.ensure_capacity(3, 4)
        pb = make_prefill_batch([toks(4, seed=1)], [0], [3], CPU)
        with self.assertRaises(ValueError):
            make_mixed_batch(pb, [], [], [], CPU)


# =========================================================================== #
# 2. the scheduler level
# =========================================================================== #
def build_mixed_scheduler(seed: int = 7, **rt_overrides):
    return build_scheduler(seed, mixed_forward=True, **rt_overrides)


def drive(sched, reqs, *, max_steps: int = 400):
    """Run to completion, returning ``{request_id: [token ids]}``."""
    for r in reqs:
        sched.add_request(r)
    kinds = []
    for _ in range(max_steps):
        if not sched.has_work():
            break
        sched.step()
        kinds.append("mixed" if sched.last_step_mixed
                     else ("prefill" if sched.last_chunk_tokens else "decode"))
    return {r.request_id: list(r.output_token_ids) for r in reqs}, kinds


def make_reqs(n: int, prompt_len: int, max_tokens: int = 6, seed: int = 0):
    return [
        Request(
            request_id=f"r{i}",
            prompt_token_ids=toks(prompt_len, seed=seed * 100 + i),
            params=gen_params(max_tokens=max_tokens),
        )
        for i in range(n)
    ]


class TestMixedSchedulerEquivalence(unittest.TestCase):
    """Same requests, same completions, with and without the mixed step."""

    def test_same_completions_as_the_separate_path(self):
        # `serving_rt` uses max_num_batched_tokens=12 against 20-token prompts,
        # so every request spans several chunks and the running set is busy
        # while they do -- i.e. the mixed step is reachable, repeatedly.
        base, _, _, _ = build_scheduler()
        mixed, _, _, _ = build_mixed_scheduler()
        want, base_kinds = drive(base, make_reqs(4, 20, seed=1))
        got, mixed_kinds = drive(mixed, make_reqs(4, 20, seed=1))
        self.assertNotIn("mixed", base_kinds)
        self.assertIn("mixed", mixed_kinds, "no mixed step ever ran -- test is vacuous")
        self.assertEqual(sorted(want), sorted(got))
        for rid in want:
            self.assertEqual(got[rid], want[rid], f"{rid}: completions differ")

    def test_a_mixed_step_emits_both_halves(self):
        """One step, a first token for the segment that finished its prompt
        *and* a token for every decode row."""
        sched, _, _, _ = build_mixed_scheduler(max_num_batched_tokens=64)
        running = make_reqs(3, 8, max_tokens=20, seed=2)
        for r in running:
            sched.add_request(r)
        # prefill + decode until all three are decoding
        for _ in range(10):
            sched.step()
            if len(sched.running) == 3:
                break
        self.assertEqual(len(sched.running), 3)
        fresh = Request(request_id="new", prompt_token_ids=toks(9, seed=99),
                        params=gen_params(max_tokens=5))
        sched.add_request(fresh)
        events = sched.step()
        self.assertTrue(sched.last_step_mixed)
        self.assertEqual(sched.last_chunk_seqs, 1)
        self.assertEqual(sched.last_chunk_tokens, 9)
        self.assertEqual(sched.last_mixed_decode_rows, 3)
        emitted = {e.request.request_id: e.new_token_ids for e in events}
        self.assertEqual(len(emitted["new"]), 1, "the finishing segment's first token")
        for r in running:
            self.assertEqual(len(emitted[r.request_id]), 1)
        self.assertIsNotNone(fresh.first_token_at)

    def test_a_chunked_prompt_emits_nothing_until_it_finishes(self):
        """A segment still mid-prompt after the mixed step contributes no
        token -- but the decode rows beside it still do."""
        sched, _, _, _ = build_mixed_scheduler(max_num_batched_tokens=6)
        running = make_reqs(2, 5, max_tokens=20, seed=3)
        for r in running:
            sched.add_request(r)
        for _ in range(10):
            sched.step()
            if len(sched.running) == 2:
                break
        long_req = Request(request_id="long", prompt_token_ids=toks(20, seed=42),
                           params=gen_params(max_tokens=5))
        sched.add_request(long_req)
        events = sched.step()
        self.assertTrue(sched.last_step_mixed)
        self.assertEqual(sched.last_chunk_tokens, 6)
        by_id = {e.request.request_id: e.new_token_ids for e in events}
        self.assertNotIn("long", by_id)
        self.assertEqual(len(by_id), 2)
        self.assertEqual(long_req.num_computed_tokens, 6)

    def test_no_mix_without_both_halves(self):
        sched, _, _, _ = build_mixed_scheduler()
        reqs = make_reqs(2, 8, max_tokens=3, seed=4)
        for r in reqs:
            sched.add_request(r)
        sched.step()  # nothing running yet -> a plain prefill chunk
        self.assertFalse(sched.last_step_mixed)
        while sched.waiting:
            sched.step()
        self.assertTrue(sched.running)
        sched.step()  # nothing waiting -> a plain (graphed) decode step
        self.assertFalse(sched.last_step_mixed)

    def test_mixed_off_by_default(self):
        sched, _, _, rt = build_scheduler()
        self.assertFalse(rt.mixed_forward)
        self.assertFalse(sched.mixed_forward)
        _, kinds = drive(sched, make_reqs(3, 20, seed=5))
        self.assertNotIn("mixed", kinds)

    def test_ratio_no_longer_gates_a_mixed_step(self):
        """``prefill_decode_ratio`` bounds how long decode waits behind
        prefill; a mixed step does not make it wait, so the counter must not
        keep prefill out of the step it belongs in."""
        sched, _, _, _ = build_mixed_scheduler(prefill_decode_ratio=1000)
        running = make_reqs(2, 8, max_tokens=20, seed=6)
        for r in running:
            sched.add_request(r)
        for _ in range(10):
            sched.step()
            if len(sched.running) == 2:
                break
        sched.add_request(Request(request_id="late", prompt_token_ids=toks(6, seed=7),
                                  params=gen_params(max_tokens=3)))
        sched.step()
        self.assertTrue(sched.last_step_mixed)


# =========================================================================== #
# 3. interaction with speculative decoding
# =========================================================================== #
def build_spec_scheduler(seed: int = 7, spec_k: int = 2, spec_max_batch=None,
                         mixed_forward: bool = True, **rt_overrides):
    model, _cfg = build_tiny_mtp_model(seed)
    rt = serving_rt(enable_mtp=True, mixed_forward=mixed_forward, **rt_overrides)
    fm = FusedQwenForCausalLM.from_m0_module(model, rt)
    buf = DeviceBuffers(max_batch=rt.max_num_seqs, vocab_size=fm.config.vocab_size,
                        max_pages=rt.n_kv_pages, device=CPU)
    decoder = GraphedDecoder(fm, buf, rt)
    spec = build_spec_decoder(fm, buf, rt, SpecConfig(k=spec_k))
    sched = Scheduler(fm, decoder, rt, spec=spec, spec_max_batch=spec_max_batch)
    return sched, fm, spec, rt


class TestMixedAndSpecPartitionTheBatchAxis(unittest.TestCase):
    """``_should_mix`` yields the step to spec whenever spec is eligible, so
    the two features never contend for the same step."""

    def _fill_running(self, sched, n, prompt_len=8, max_tokens=20, seed=8):
        reqs = make_reqs(n, prompt_len, max_tokens=max_tokens, seed=seed)
        for r in reqs:
            sched.add_request(r)
        for _ in range(20):
            sched.step()
            if len(sched.running) == n:
                break
        self.assertEqual(len(sched.running), n)
        return reqs

    def test_spec_wins_the_step_below_the_cap(self):
        sched, _, _, _ = build_spec_scheduler(spec_max_batch=8)
        self._fill_running(sched, 2)
        sched.add_request(Request(request_id="new", prompt_token_ids=toks(7, seed=11),
                                  params=gen_params(max_tokens=4)))
        sched.step()
        self.assertFalse(sched.last_step_mixed,
                         "spec is eligible at B=2 <= 8; the step must not be mixed")

    def test_mixed_takes_over_above_the_cap(self):
        sched, _, _, _ = build_spec_scheduler(spec_max_batch=1)
        self._fill_running(sched, 3)
        sched.add_request(Request(request_id="new", prompt_token_ids=toks(7, seed=12),
                                  params=gen_params(max_tokens=4)))
        sched.step()
        self.assertTrue(sched.last_step_mixed,
                        "B=3 > --spec-max-batch 1, so the mixed step must run")

    def test_mtp_carry_advances_across_a_mixed_step(self):
        """The MTP head's ``h_prev`` and its own KV layer must not develop a
        hole at a mixed step: ``on_prefill`` for the segments, ``on_plain_step``
        for the decode rows (``SpecDecoder.on_plain_step``)."""
        sched, fm, spec, _ = build_spec_scheduler(spec_max_batch=1)
        running = self._fill_running(sched, 3)
        before = spec.h_prev.detach().clone()
        fresh = Request(request_id="new", prompt_token_ids=toks(7, seed=13),
                        params=gen_params(max_tokens=4))
        sched.add_request(fresh)
        sched.step()
        self.assertTrue(sched.last_step_mixed)
        for r in running:
            self.assertFalse(
                torch.equal(before[r.slot], spec.h_prev[r.slot]),
                f"{r.request_id}: h_prev did not advance over the mixed step",
            )
        self.assertFalse(torch.equal(before[fresh.slot], spec.h_prev[fresh.slot]),
                         "the prefilled segment's h_prev was not seeded")

    def test_spec_and_mixed_agree_with_the_separate_path(self):
        """End to end with a ``SpecDecoder`` attached: same completions.

        ``--spec-max-batch 0`` keeps the speculative path *out* of both runs
        (``_use_spec`` vetoes every batch above the cap), which is the point:
        what is under test is that the MTP bookkeeping a mixed step still owes
        -- ``on_prefill`` for its segments, ``on_plain_step`` for its decode
        rows -- neither corrupts the main stack's answer nor is skipped. The
        spec path's own equivalence is ``test_spec_decode.py``'s job."""
        plain, _, _, _ = build_spec_scheduler(spec_max_batch=0, mixed_forward=False)
        mixed, _, _, _ = build_spec_scheduler(spec_max_batch=0, mixed_forward=True)
        want, plain_kinds = drive(plain, make_reqs(4, 20, max_tokens=6, seed=14))
        got, kinds = drive(mixed, make_reqs(4, 20, max_tokens=6, seed=14))
        self.assertNotIn("mixed", plain_kinds)
        self.assertIn("mixed", kinds, "no mixed step ever ran -- test is vacuous")
        for rid in want:
            self.assertEqual(got[rid], want[rid], f"{rid}: completions differ")


# =========================================================================== #
# 4. the memory plan
# =========================================================================== #
class TestMixedMemoryPlan(unittest.TestCase):
    """``serve.plan_memory`` must know that a mixed step's activations are
    ``max_num_batched_tokens + max_batch`` rows wide, not
    ``max_num_batched_tokens``."""

    BASE = dict(
        max_num_seqs=256, max_model_len=2752, page_size=16, n_kv_pages=8192,
        kv_cache_dtype="bf16", ssm_state_dtype="fp16", max_num_batched_tokens=8192,
        max_batch=256,
    )

    def test_mixed_widens_the_chunk_by_exactly_max_batch_rows(self):
        """The identity the term encodes: a mixed step's working set is the
        working set of a chunk ``max_batch`` tokens longer.

        Asserted as an identity rather than as a ratio because the prefill term
        is *affine* in the token count, not proportional -- the conv fp32 tile
        and the per-chunk logits rows do not scale with it -- so "+3.1%" is a
        statement about this geometry and this one is a statement about the
        arithmetic."""
        on = plan_memory(**self.BASE, mixed_forward=True)
        wider = plan_memory(
            **dict(self.BASE, max_num_batched_tokens=8192 + 256), mixed_forward=False
        )
        self.assertEqual(on["prefill_gib"], wider["prefill_gib"])
        off = plan_memory(**self.BASE, mixed_forward=False)
        self.assertGreater(on["prefill_gib"], off["prefill_gib"])
        self.assertEqual(on["steady_gib"], off["steady_gib"],
                         "the mixed step allocates nothing resident")

    def test_small_chunk_large_batch_is_where_it_matters(self):
        """The term is here because it scales with ``max_batch`` while the
        chunk cap does not.

        Measured: **+2.6%** at the serving geometry (chunk 8,192, conc 256) and
        **+10.6%** at chunk 2,048 / conc 512 -- a 4x bigger bite of the same
        headroom, from two flags a serving config is free to set. Neither
        number is ``max_batch / max_num_batched_tokens`` (3.1% and 25%),
        because the prefill term is affine, not proportional: the fp32 conv
        tile and the ``max_num_seqs x vocab`` logits row do not scale with the
        token count. That is exactly why this is asserted against the plan
        rather than reasoned about in a comment."""
        big = dict(self.BASE, max_num_batched_tokens=2048, max_num_seqs=512, max_batch=512)
        small = (
            plan_memory(**self.BASE, mixed_forward=True)["prefill_gib"]
            / plan_memory(**self.BASE, mixed_forward=False)["prefill_gib"]
        ) - 1.0
        wide = (
            plan_memory(**big, mixed_forward=True)["prefill_gib"]
            / plan_memory(**big, mixed_forward=False)["prefill_gib"]
        ) - 1.0
        self.assertAlmostEqual(small, 0.026, places=3)
        self.assertAlmostEqual(wide, 0.106, places=3)
        self.assertGreater(wide, 3.0 * small)

    def test_default_is_unchanged_to_the_byte(self):
        """A plan built without the flag must be byte-identical to the
        one built before mixed forward existed, so existing memory budgets still
        hold."""
        a = plan_memory(**self.BASE)
        b = plan_memory(**self.BASE, mixed_forward=False)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main(verbosity=2)
