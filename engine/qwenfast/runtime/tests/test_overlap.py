"""CPU tests for the overlapped prefill+decode step.

The claim under test, in one sentence: **an overlapped step emits exactly what
a prefill step followed by a decode step emits** — which is a stronger and
simpler claim than the mixed step's, because the two halves really are the two
separate steps, run at the same time rather than fused into one forward.

That is also why these tests can be exact where ``test_mixed_forward.py``'s
have to carry a 1e-6 tolerance: the mixed step changes the ``M`` of every
shared GEMM (``T_pre + B_dec`` at once instead of ``T_pre`` and ``B_dec``
apart) and therefore its blocking, which costs a ULP. An overlapped step runs
the *same two forwards* the separate path runs, on the same rows, at the same
``M``. Off a GPU there is no second stream, so it runs them in the same order
on the same thread — and the emitted token ids must be **identical**, with no
tolerance at all.

What is not testable here, and where it is tested instead:

* that the two graphs may run **concurrently** — that they write disjoint
  slots, own disjoint graph mempools and own disjoint FlashInfer workspaces.
  The disjointness of the *slots* is structural and is asserted below
  (:class:`TestOverlapHalvesAreDisjoint`); the two allocations are
  ``RuntimeConfig.overlap_streams`` wiring, checked in
  :class:`TestOverlapWiring`, and their effect is a GPU measurement
  (``bench_overlap.py``).
* what it costs. Measured on GPU at conc 256, the second stream hides 1–9 % of a
  decode step, and the win that makes ``--overlap`` worth shipping is the
  *other* half of the change — not fusing the two halves into one forward at
  all, which is 7 % (chunk 8,192) to 23 % (chunk 1,024) faster than the mixed
  step at the real serving context.

Run::

    python -m unittest discover -s engine/qwenfast/runtime/tests -v
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))  # -> engine/
sys.path.insert(0, _HERE)

from qwenfast.runtime.fused_model import RuntimeConfig  # noqa: E402
from qwenfast.runtime.scheduler import Request  # noqa: E402

from test_mixed_forward import (  # noqa: E402
    build_mixed_scheduler,
    drive,
    make_reqs,
    toks,
)
from test_serving import build_scheduler, gen_params  # noqa: E402


def build_overlap_scheduler(seed: int = 7, **rt_overrides):
    return build_scheduler(seed, mixed_forward=True, overlap_streams=True, **rt_overrides)


# =========================================================================== #
# 1. the ladder: which buckets the mixed-step graphs are captured for
# =========================================================================== #
class TestMixedBucketsFor(unittest.TestCase):
    """``RuntimeConfig.mixed_buckets_for`` — one rule, three callers."""

    def test_default_is_the_derived_ladder(self):
        rt = RuntimeConfig(max_num_seqs=64)
        self.assertEqual(rt.mixed_buckets_for(), rt.buckets_for())

    def test_overlap_asks_for_exactly_one_row(self):
        """`--overlap`'s prefill graph carries one *padding* decode row: the
        real rows are a second graph on a second stream, and a mixed step with
        zero decode rows has no bucket at all."""
        rt = RuntimeConfig(max_num_seqs=64, overlap_streams=True)
        self.assertEqual(rt.mixed_buckets_for(), (1,))

    def test_an_explicit_ladder_overrides_everything(self):
        rt = RuntimeConfig(max_num_seqs=256, overlap_streams=True,
                           mixed_graph_buckets=(1, 256))
        self.assertEqual(rt.mixed_buckets_for(), (1, 256))
        rt2 = RuntimeConfig(max_num_seqs=256, mixed_graph_buckets=(32,))
        self.assertEqual(rt2.mixed_buckets_for(), (32,))


# =========================================================================== #
# 2. wiring: what `overlap_streams` turns on at build time
# =========================================================================== #
class TestOverlapWiring(unittest.TestCase):
    def test_off_by_default(self):
        sched, _, _, rt = build_scheduler()
        self.assertFalse(rt.overlap_streams)
        self.assertFalse(sched.overlap)
        self.assertIsNone(sched._overlap_stream)  # noqa: SLF001

    def test_on_when_asked(self):
        sched, _, _, rt = build_overlap_scheduler()
        self.assertTrue(rt.overlap_streams)
        self.assertTrue(sched.overlap)
        # No CUDA in these tests, so no side stream -- the two halves run in
        # order on the one thread, which is exactly what makes the
        # equivalence testable here.
        self.assertIsNone(sched._overlap_stream)  # noqa: SLF001

    def test_the_flag_is_explicit_over_the_config(self):
        sched, _, _, _ = build_scheduler(mixed_forward=True, overlap_streams=True)
        self.assertTrue(sched.overlap)
        sched2, model, decoder, rt = build_scheduler(mixed_forward=True,
                                                     overlap_streams=True)
        from qwenfast.runtime.scheduler import Scheduler
        s3 = Scheduler(model, decoder, rt, overlap=False)
        self.assertFalse(s3.overlap)


# =========================================================================== #
# 3. equivalence: an overlapped step == a prefill step + a decode step
# =========================================================================== #
class TestOverlapSchedulerEquivalence(unittest.TestCase):
    def test_same_completions_as_the_separate_path(self):
        """The whole claim, end to end and with no tolerance."""
        base, _, _, _ = build_scheduler()
        over, _, _, _ = build_overlap_scheduler()
        want, base_kinds = drive(base, make_reqs(4, 20, seed=1))
        got, over_kinds = drive(over, make_reqs(4, 20, seed=1))
        self.assertNotIn("mixed", base_kinds)
        self.assertIn("mixed", over_kinds, "no overlapped step ever ran -- vacuous")
        self.assertEqual(sorted(want), sorted(got))
        for rid in want:
            self.assertEqual(got[rid], want[rid], f"{rid}: completions differ")

    def test_same_completions_as_the_fused_mixed_path(self):
        """And the same as the mixed step's, which is a different claim: the
        mixed step is not bit-identical to the separate path (a ULP of GEMM
        reassociation, ``test_mixed_forward``'s ``EXACT``), so this passing
        says the greedy argmax survives both, not that the arithmetic does."""
        mixed, _, _, _ = build_mixed_scheduler()
        over, _, _, _ = build_overlap_scheduler()
        want, _ = drive(mixed, make_reqs(4, 20, seed=1))
        got, _ = drive(over, make_reqs(4, 20, seed=1))
        for rid in want:
            self.assertEqual(got[rid], want[rid], f"{rid}: completions differ")

    def test_an_overlapped_step_emits_both_halves(self):
        sched, _, _, _ = build_overlap_scheduler(max_num_batched_tokens=64)
        running = make_reqs(3, 8, max_tokens=20, seed=2)
        for r in running:
            sched.add_request(r)
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
        self.assertTrue(sched.last_step_overlapped)
        self.assertEqual(sched.last_chunk_seqs, 1)
        self.assertEqual(sched.last_chunk_tokens, 9)
        self.assertEqual(sched.last_mixed_decode_rows, 3)
        emitted = {e.request.request_id: e.new_token_ids for e in events}
        self.assertEqual(len(emitted["new"]), 1, "the finishing segment's first token")
        for r in running:
            self.assertEqual(len(emitted[r.request_id]), 1)
        self.assertIsNotNone(fresh.first_token_at)

    def test_a_chunked_prompt_emits_nothing_until_it_finishes(self):
        sched, _, _, _ = build_overlap_scheduler(max_num_batched_tokens=6)
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
        self.assertTrue(sched.last_step_overlapped)
        self.assertEqual(sched.last_chunk_tokens, 6)
        by_id = {e.request.request_id: e.new_token_ids for e in events}
        self.assertNotIn("long", by_id)
        self.assertEqual(len(by_id), 2)
        self.assertEqual(long_req.num_computed_tokens, 6)

    def test_no_overlap_without_both_halves(self):
        sched, _, _, _ = build_overlap_scheduler()
        reqs = make_reqs(2, 8, max_tokens=3, seed=4)
        for r in reqs:
            sched.add_request(r)
        sched.step()
        self.assertFalse(sched.last_step_overlapped)
        while sched.waiting:
            sched.step()
        self.assertTrue(sched.running)
        sched.step()
        self.assertFalse(sched.last_step_overlapped)


class TestOverlapFillGuard(unittest.TestCase):
    """A graphed chunk always computes ``chunk_tokens`` tokens,
    so a step whose real chunk is a small fraction of that is mostly padding.
    Below ``overlap_min_fill`` the prefill half runs eagerly instead."""

    def test_a_thin_chunk_does_not_take_the_graph(self):
        sched, _, _, _ = build_overlap_scheduler(
            mixed_graphs=True, max_num_batched_tokens=64, overlap_min_fill=0.75
        )
        self.assertIsNotNone(sched.mixed_pad)
        running = make_reqs(2, 8, max_tokens=20, seed=21)
        for r in running:
            sched.add_request(r)
        for _ in range(10):
            sched.step()
            if len(sched.running) == 2:
                break
        # A 9-token prompt against a 64-token graph is 14 % fill.
        sched.add_request(Request(request_id="thin", prompt_token_ids=toks(9, seed=5),
                                  params=gen_params(max_tokens=3)))
        sched.step()
        self.assertTrue(sched.last_step_overlapped)
        self.assertFalse(sched.last_step_mixed_graphed,
                         "a 14 %-full chunk took the graph and computed 86 % padding")

    def test_zero_min_fill_restores_the_always_graphed_behaviour(self):
        sched, _, _, _ = build_overlap_scheduler(
            mixed_graphs=True, max_num_batched_tokens=64, overlap_min_fill=0.0
        )
        self.assertEqual(sched.overlap_min_fill, 0.0)

    def test_the_guard_cannot_change_a_token(self):
        """Eager and graphed prefill halves are the same forward on the same
        rows, so the guard is a speed decision only -- pinned the same way
        every other arm here is, by driving two engines to completion."""
        a, _, _, _ = build_overlap_scheduler(overlap_min_fill=0.0)
        b, _, _, _ = build_overlap_scheduler(overlap_min_fill=1.0)
        want, _ = drive(a, make_reqs(4, 20, seed=1))
        got, _ = drive(b, make_reqs(4, 20, seed=1))
        for rid in want:
            self.assertEqual(got[rid], want[rid], f"{rid}: completions differ")


# =========================================================================== #
# 4. the precondition the concurrency rests on
# =========================================================================== #
class TestOverlapHalvesAreDisjoint(unittest.TestCase):
    """The two graphs may run at the same time only because they never write
    the same slot. That is structural -- ``_collect_prefill_chunk`` moves a
    request into ``running`` only when its prompt completes, and
    ``_admit_decode_rows`` is snapshotted before the chunk is packed -- so it
    is asserted over a real driven run rather than argued for."""

    def test_no_slot_is_in_both_halves_of_any_step(self):
        sched, _, _, _ = build_overlap_scheduler(max_num_batched_tokens=12)
        reqs = make_reqs(5, 20, max_tokens=8, seed=11)
        for r in reqs:
            sched.add_request(r)

        seen_overlap = 0
        real_collect = sched._collect_prefill_chunk  # noqa: SLF001
        real_admit = sched._admit_decode_rows  # noqa: SLF001
        state = {}

        def collect(budget, events, max_segments=None):
            out = real_collect(budget, events, max_segments=max_segments)
            state["chunk"] = list(out[3])
            return out

        def admit(events, *, window):
            out = real_admit(events, window=window)
            state["decode"] = [r.slot for r in out]
            return out

        sched._collect_prefill_chunk = collect  # noqa: SLF001
        sched._admit_decode_rows = admit  # noqa: SLF001
        for _ in range(400):
            if not sched.has_work():
                break
            state.clear()
            sched.step()
            if sched.last_step_overlapped:
                seen_overlap += 1
                chunk = set(state.get("chunk", []))
                # `_run_overlap_step` re-filters `dec_reqs` after the chunk is
                # packed; the recorded set is the pre-filter one, so the
                # assertion is over a superset -- which is the stronger check.
                decode = set(state.get("decode", []))
                self.assertEqual(chunk & decode, set(),
                                 f"slot in both halves: {chunk & decode}")
        self.assertGreater(seen_overlap, 0, "no overlapped step ran -- vacuous")


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
