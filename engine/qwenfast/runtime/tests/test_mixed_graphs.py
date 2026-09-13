"""CPU tests for the graphed mixed step.

The claim under test, in one sentence: **padding a mixed step to a fixed shape
changes nothing except how long it takes.** A CUDA graph cannot be captured or
replayed on a CPU-only host, and does not need to be for that claim: the risk
in graphing the mixed step is not "does ``cudaGraphLaunch`` work", it is "does a step with 7 fake
segments and 136 fake decode rows in it compute the same thing for the real
ones, and leave every real slot's state untouched". All of that is padding
arithmetic and pool bookkeeping, and all of it runs on CPU.

Four levels:

* :class:`TestPadArithmetic` -- ``pad_mixed_step`` alone. Pure functions, no
  model: every graphed step has exactly ``n_segments`` prefill plan rows and
  exactly ``chunk_tokens`` prefill tokens, every row is non-empty (FlashInfer's
  graph-mode ``qo_indptr`` requires it), and the real content comes first and
  unmodified.
* :class:`TestPaddedStepMatchesUnpadded` -- the model level. Same step padded
  and unpadded, from the same pool snapshot: identical logits for the real
  rows, identical argmax, and every pool row except the scratch slot identical
  byte for byte.
* :class:`TestPaddingIsNotAccidentallyANoOp` -- the other direction. The
  scratch slot *does* change, i.e. the padding really ran; a test suite that
  only checked "nothing else moved" would pass with the padding silently
  dropped.
* :class:`TestSchedulerWithPadding` -- two engines, same seed, one padding and
  one not, driven to completion: identical completions, plus the budget and
  segment-cap arithmetic the padded path imposes on ``_collect_prefill_chunk``.

Run::

    python engine/qwenfast/runtime/tests/test_mixed_graphs.py
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))  # -> engine/
sys.path.insert(0, _HERE)

from qwenfast.runtime.fused_model import (  # noqa: E402
    RuntimeConfig,
    make_mixed_batch,
    make_prefill_batch,
)
from qwenfast.runtime.mixed_graphs import (  # noqa: E402
    PAD_TOKEN_ID,
    MixedGraphRunner,
    MixedPadSpec,
    pad_mixed_step,
    reset_scratch_state,
)
from qwenfast.kernels_gdn import fla_static  # noqa: E402

_FLA_STATIC_OK = fla_static.is_available()
from qwenfast.runtime.scheduler import GenParams, Request, Scheduler  # noqa: E402
from qwenfast.runtime.serve import plan_memory  # noqa: E402

from test_mixed_forward import (  # noqa: E402
    EXACT,
    assert_same,
    build_model,
    prime_slot,
    restore,
    snapshot,
    toks,
)
from test_serving import CPU, VOCAB, build_scheduler, gen_params  # noqa: E402


def spec_for(model, chunk_tokens=24, n_segments=4, buckets=(2, 4, 8)) -> MixedPadSpec:
    return MixedPadSpec(
        chunk_tokens=chunk_tokens, n_segments=n_segments,
        buckets=tuple(buckets), scratch_slot=model.scratch_slot,
    )


# =========================================================================== #
# 1. the padding arithmetic
# =========================================================================== #
class TestPadArithmetic(unittest.TestCase):
    SPEC = MixedPadSpec(chunk_tokens=1024, n_segments=8, buckets=(32, 64, 128, 256),
                        scratch_slot=99)

    def test_budget_leaves_one_token_for_every_padding_segment(self):
        """``budget = chunk - n_segments`` is not a round number, it is the
        precondition that makes every plan row non-empty."""
        s = self.SPEC
        self.assertEqual(s.budget, 1024 - 8)
        self.assertEqual(s.max_segments, 7)
        # Worst case: one real segment eating the whole budget -> 7 pad rows
        # to fill and exactly `chunk - budget` = 8 tokens to fill them with.
        pad = pad_mixed_step([[1] * s.budget], [0], [0], [5], [1], [3], s)
        self.assertEqual(len(pad.token_ids), s.n_segments)
        self.assertTrue(all(len(t) >= 1 for t in pad.token_ids))
        self.assertEqual(sum(len(t) for t in pad.token_ids), s.chunk_tokens)

    def test_shape_is_exact_for_every_real_shape(self):
        s = self.SPEC
        for n_seg in range(1, s.max_segments + 1):
            for rows in (1, 31, 32, 33, 200, 256):
                with self.subTest(segments=n_seg, rows=rows):
                    real = [[7] * 10 for _ in range(n_seg)]
                    pad = pad_mixed_step(
                        real, [0] * n_seg, list(range(n_seg)),
                        list(range(50, 50 + rows)), [1] * rows, [4] * rows, s,
                    )
                    self.assertEqual(len(pad.token_ids), s.n_segments)
                    self.assertEqual(sum(len(t) for t in pad.token_ids), s.chunk_tokens)
                    self.assertEqual(len(pad.decode_slots), pad.bucket)
                    self.assertEqual(pad.bucket, s.bucket_for(rows))
                    self.assertTrue(all(len(t) >= 1 for t in pad.token_ids))

    def test_real_content_is_first_and_unchanged(self):
        s = self.SPEC
        real = [[3, 4, 5], [6, 7]]
        pad = pad_mixed_step(real, [0, 11], [1, 2], [9, 8], [21, 22], [4, 5], s)
        self.assertEqual(pad.token_ids[:2], real)
        self.assertEqual(pad.start_positions[:2], [0, 11])
        self.assertEqual(pad.slots[:2], [1, 2])
        self.assertEqual(pad.decode_slots[:2], [9, 8])
        self.assertEqual(pad.decode_token_ids[:2], [21, 22])
        self.assertEqual(pad.decode_positions[:2], [4, 5])
        self.assertEqual(pad.n_real_segments, 2)
        self.assertEqual(pad.n_real_rows, 2)

    def test_every_padding_row_is_on_the_scratch_slot_at_position_zero(self):
        """The one safety property: padding writes state and KV, and all of it
        has to land on the slot no request owns."""
        s = self.SPEC
        pad = pad_mixed_step([[3, 4]], [0], [1], [9], [21], [4], s)
        self.assertTrue(all(sl == s.scratch_slot for sl in pad.slots[1:]))
        self.assertTrue(all(p == 0 for p in pad.start_positions[1:]))
        self.assertTrue(all(t == PAD_TOKEN_ID for seg in pad.token_ids[1:] for t in seg))
        self.assertTrue(all(sl == s.scratch_slot for sl in pad.decode_slots[1:]))
        self.assertTrue(all(p == 0 for p in pad.decode_positions[1:]))

    def test_a_step_that_does_not_fit_is_refused_not_silently_truncated(self):
        s = self.SPEC
        with self.assertRaises(ValueError):  # too many segments
            real = [[1]] * (s.max_segments + 1)
            pad_mixed_step(real, [0] * len(real), list(range(len(real))),
                           [1], [1], [1], s)
        with self.assertRaises(ValueError):  # over budget
            pad_mixed_step([[1] * (s.budget + 1)], [0], [0], [1], [1], [1], s)
        with self.assertRaises(ValueError):  # more rows than the widest bucket
            rows = s.buckets[-1] + 1
            pad_mixed_step([[1]], [0], [0], list(range(rows)), [1] * rows, [1] * rows, s)

    def test_bucket_ladder(self):
        s = self.SPEC
        self.assertEqual(s.bucket_for(1), 32)
        self.assertEqual(s.bucket_for(32), 32)
        self.assertEqual(s.bucket_for(33), 64)
        self.assertEqual(s.bucket_for(256), 256)
        self.assertIsNone(s.bucket_for(257))


# =========================================================================== #
# 2. the model level
# =========================================================================== #
def run_padded_case(case, chunk, decode, *, bucket, n_segments=4,
                    chunk_tokens=None, seed=7):
    """Run one step both ways and assert they agree. Returns
    ``(after_plain, after_padded, model)`` so a caller can also assert on the
    scratch slot (which is the *other* half of the claim -- see
    :class:`TestPaddingIsNotAccidentallyANoOp`)."""
    self = case  # the assertions below are the TestCase's, not this function's
    if True:  # noqa: SIM108 -- keeps the body's indentation stable
        model, _rt = build_model(seed)
        spec = spec_for(
            model,
            chunk_tokens=chunk_tokens or (sum(len(ids) for _, ids, _ in chunk) + n_segments),
            n_segments=n_segments,
            buckets=(bucket,),
        )
        dec_slots = [s for s, _ in decode]
        dec_positions = []
        for slot, prompt in decode:
            prime_slot(model, slot, prompt, extra=1)
            dec_positions.append(len(prompt))
        dec_tokens = [7 + i for i in range(len(decode))]
        for slot, ids, start in chunk:
            if start == 0:
                model.reset_slot(slot)
                model.kv_pool.ensure_capacity(slot, len(ids))
            else:
                prime_slot(model, slot, toks(start, seed=slot + 100))
                model.kv_pool.ensure_capacity(slot, start + len(ids))
        # The scratch slot needs room for a whole chunk of padding -- the one
        # thing `MixedGraphRunner.prepare` does at startup.
        model.kv_pool.ensure_capacity(model.scratch_slot, spec.chunk_tokens)

        ids = [i for _, i, _ in chunk]
        starts = [st for _, _, st in chunk]
        slots = [sl for sl, _, _ in chunk]

        base = snapshot(model)

        # -- A: unpadded (the eager mixed step) --------------------------- #
        pb = make_prefill_batch(ids, starts, slots, CPU)
        mb = make_mixed_batch(pb, dec_slots, dec_tokens, dec_positions, CPU)
        plain_logits = model.mixed_forward(mb)
        after_plain = snapshot(model)

        # -- B: padded (what the graph captures) -------------------------- #
        restore(model, base)
        reset_scratch_state(model)
        padded = pad_mixed_step(ids, starts, slots, dec_slots, dec_tokens,
                                dec_positions, spec)
        ppb = make_prefill_batch(padded.token_ids, padded.start_positions,
                                 padded.slots, CPU)
        pmb = make_mixed_batch(ppb, padded.decode_slots, padded.decode_token_ids,
                               padded.decode_positions, CPU)
        pad_logits = model.mixed_forward(pmb)
        after_pad = snapshot(model)

        # -- the real rows' logits ---------------------------------------- #
        n_real_seg, n_real_rows = len(chunk), len(decode)
        self.assertEqual(pad_logits.shape[0], spec.n_segments + padded.bucket)
        got_seg, want_seg = pad_logits[:n_real_seg], plain_logits[:n_real_seg]
        assert_same(self, got_seg, want_seg, "padded vs plain: segment logits")
        got_dec = pad_logits[spec.n_segments : spec.n_segments + n_real_rows]
        want_dec = plain_logits[n_real_seg : n_real_seg + n_real_rows]
        assert_same(self, got_dec, want_dec, "padded vs plain: decode logits")
        self.assertTrue(
            torch.equal(pad_logits[: n_real_seg].argmax(-1),
                        plain_logits[: n_real_seg].argmax(-1)),
            "padded and plain disagree on a segment's sampled token",
        )
        self.assertTrue(
            torch.equal(got_dec.argmax(-1), want_dec.argmax(-1)),
            "padded and plain disagree on a decode row's sampled token",
        )

        # -- every real slot's state ---------------------------------------- #
        # To `EXACT`, not bit-exact, and for the same measured reason the
        # mixed-forward suite's logits are (`test_mixed_forward.EXACT`): the padded step
        # runs the *same* kernels over the same rows, but the varlen GDN call
        # sees a different segment table, so its accumulation blocks differ and
        # a real slot's committed state moves by ~1e-10 on fp32 values of O(1).
        # A padding row leaking into a real slot -- a wrong `slot_id`, a
        # segment boundary off by one -- moves it by O(1), four orders above
        # this bar and eight above what is observed.
        scratch = model.scratch_slot
        real_rows = [r for r in range(model.state_pool.shape[0]) if r != scratch]
        for name in ("state", "conv"):
            assert_same(self, after_pad[name][real_rows], after_plain[name][real_rows],
                        f"padded vs plain: {name} pool of every real slot")
        # KV: the scratch slot's pages are the padding's, everything else must
        # match. Compare seq_len per slot rather than the whole page array, and
        # the pages of every real slot via the page table.
        for slot in real_rows:
            self.assertEqual(
                int(after_plain["seq_len"][slot]), int(after_pad["seq_len"][slot]),
                f"padding moved slot {slot}'s seq_len",
            )
        pt = model.kv_pool.page_table_host
        real_pages = sorted({
            int(p) for slot in real_rows for p in pt[slot].tolist() if int(p) >= 0
        })
        scratch_pages = {int(p) for p in pt[scratch].tolist() if int(p) >= 0}
        real_pages = [p for p in real_pages if p not in scratch_pages]
        if real_pages:
            idx = torch.tensor(real_pages, dtype=torch.long)
            assert_same(self, after_pad["kv"][:, idx], after_plain["kv"][:, idx],
                        "padded vs plain: every real sequence's KV pages")
        return after_plain, after_pad, model


class TestPaddedStepMatchesUnpadded(unittest.TestCase):
    """A padded mixed step computes what the unpadded one computes."""

    def test_one_segment_and_several_decode_rows(self):
        run_padded_case(
            self,
            chunk=[(0, toks(6, seed=1), 0)],
            decode=[(1, toks(5, seed=2)), (2, toks(7, seed=3))],
            bucket=4,
        )

    def test_several_segments(self):
        run_padded_case(
            self,
            chunk=[(0, toks(5, seed=1), 0), (3, toks(4, seed=4), 0)],
            decode=[(1, toks(5, seed=2)), (2, toks(6, seed=3))],
            bucket=4,
        )

    def test_a_segment_that_continues_a_previous_chunk(self):
        run_padded_case(
            self,
            chunk=[(0, toks(5, seed=1), 8)],
            decode=[(1, toks(5, seed=2))],
            bucket=2,
        )

    def test_one_decode_row_padded_up_to_a_wide_bucket(self):
        """The conc-256 shape in miniature: 1 live row, 7 padding rows."""
        run_padded_case(
            self,
            chunk=[(0, toks(6, seed=1), 0)],
            decode=[(1, toks(5, seed=2))],
            bucket=8,
        )

    def test_a_chunk_much_smaller_than_the_padded_shape(self):
        """The ramp shape: the scheduler could only pack 3 tokens but the graph
        is 24 wide, so 21 tokens and 3 segments of padding ride along."""
        run_padded_case(
            self,
            chunk=[(0, toks(3, seed=1), 0)],
            decode=[(1, toks(5, seed=2)), (2, toks(4, seed=6))],
            bucket=4,
            chunk_tokens=24,
        )


class TestPaddingIsNotAccidentallyANoOp(unittest.TestCase):
    def test_the_scratch_slot_really_is_written(self):
        """If padding were silently dropped, every assertion in the class above
        would still pass. This is the one that would not."""
        after_plain, after_pad, model = run_padded_case(
            self,
            chunk=[(0, toks(6, seed=1), 0)],
            decode=[(1, toks(5, seed=2))],
            bucket=4,
            chunk_tokens=24,
        )
        s = model.scratch_slot
        self.assertFalse(
            torch.equal(after_plain["state"][s], after_pad["state"][s]),
            "the padding segments did not write the scratch slot's SSM state -- "
            "either they were dropped or they are not running the GDN kernels",
        )
        self.assertGreater(
            int(after_pad["seq_len"][s]), int(after_plain["seq_len"][s]),
            "the padding segments did not append any KV",
        )


# =========================================================================== #
# 3. the runner, on a host with no CUDA
# =========================================================================== #
class TestRunnerWithoutCuda(unittest.TestCase):
    def test_it_builds_declines_to_capture_and_says_so(self):
        model, rt = build_model(7, mixed_forward=True, mixed_graphs=True,
                                mixed_graph_segments=4)
        runner = MixedGraphRunner(model, rt, chunk_tokens=24, buckets=(2, 4))
        self.assertFalse(runner.graphs_enabled)
        runner.capture()  # no-op, must not raise
        self.assertFalse(runner.captured)
        self.assertFalse(runner.ready(2))
        self.assertEqual(runner.spec.chunk_tokens, 24)
        self.assertEqual(runner.spec.budget, 20)

    def test_prepare_gives_the_scratch_slot_a_chunk_of_pages(self):
        model, rt = build_model(7, mixed_forward=True, mixed_graphs=True)
        runner = MixedGraphRunner(model, rt, chunk_tokens=24, buckets=(2, 4))
        runner.prepare()
        have = model.kv_pool.pages_allocated(model.scratch_slot)
        self.assertGreaterEqual(have, model.kv_pool.pages_needed(24))


# =========================================================================== #
# 4. the scheduler
# =========================================================================== #
class TestSchedulerWithPadding(unittest.TestCase):
    RT = dict(mixed_forward=True, max_num_batched_tokens=12, prefill_chunk_tokens=12)

    def _drive(self, sched, prompts, max_tokens=6):
        out = {}
        for i, p in enumerate(prompts):
            sched.add_request(Request(f"r{i}", list(p), gen_params(max_tokens=max_tokens)))
        for _ in range(400):
            if not sched.has_work():
                break
            for ev in sched.step():
                out.setdefault(ev.request.request_id, []).extend(ev.new_token_ids)
        return out

    def test_padded_and_unpadded_schedulers_emit_the_same_tokens(self):
        prompts = [toks(9, seed=s) for s in range(4)]
        plain = build_scheduler(seed=5, **self.RT)[0]
        got_plain = self._drive(plain, prompts)
        padded = build_scheduler(seed=5, mixed_graphs=True, mixed_graph_segments=4,
                                 **self.RT)[0]
        got_pad = self._drive(padded, prompts)
        self.assertEqual(
            got_plain, got_pad,
            "padding the mixed step changed the emitted tokens -- the padded step is "
            "not computing what the unpadded one computes",
        )
        self.assertTrue(got_plain, "the drive emitted nothing; the test proves nothing")

    def test_the_budget_and_segment_cap_are_the_padded_shape(self):
        sched = build_scheduler(seed=5, mixed_graphs=True, mixed_graph_segments=4,
                                **self.RT)[0]
        self.assertIsNotNone(sched.mixed_pad)
        self.assertEqual(sched.mixed_pad.chunk_tokens, 12)
        self.assertEqual(sched.mixed_pad.budget, 8)
        self.assertEqual(sched.mixed_pad.max_segments, 3)

    def test_collect_prefill_chunk_honours_max_segments(self):
        sched = build_scheduler(seed=5, **self.RT)[0]
        for i in range(6):
            sched.add_request(Request(f"r{i}", toks(2, seed=i), gen_params(max_tokens=2)))
        reqs, ids, starts, slots = sched._collect_prefill_chunk(12, [], max_segments=2)
        self.assertEqual(len(reqs), 2)
        self.assertLessEqual(sum(len(i) for i in ids), 12)

    def test_no_padding_without_the_flag(self):
        sched = build_scheduler(seed=5, **self.RT)[0]
        self.assertIsNone(sched.mixed_pad)
        self.assertFalse(sched.last_step_mixed_graphed)

    def test_a_chunk_too_small_for_the_segment_count_is_refused_at_build(self):
        with self.assertRaises(ValueError):
            build_scheduler(seed=5, mixed_forward=True, mixed_graphs=True,
                            mixed_graph_segments=32, max_num_batched_tokens=12,
                            prefill_chunk_tokens=12)


# =========================================================================== #
# 5. the memory plan
# =========================================================================== #
class TestMixedGraphMemoryPlan(unittest.TestCase):
    BASE = dict(
        max_num_seqs=256, max_model_len=2752, page_size=16, n_kv_pages=44353,
        kv_cache_dtype="bf16", ssm_state_dtype="fp16", max_num_batched_tokens=8192,
        mixed_forward=True,
    )

    def test_off_is_byte_identical(self):
        a = plan_memory(**self.BASE)
        b = plan_memory(**self.BASE, mixed_graphs=False, mixed_graph_chunk=1024)
        self.assertEqual(a["steady_gib"], b["steady_gib"])
        self.assertEqual(a.get("mixed_graph_gib", 0.0), 0.0)

    def test_on_adds_a_bounded_term_that_grows_with_the_chunk(self):
        off = plan_memory(**self.BASE)
        on1 = plan_memory(**self.BASE, mixed_graphs=True, mixed_graph_chunk=1024)
        on2 = plan_memory(**self.BASE, mixed_graphs=True, mixed_graph_chunk=2048)
        self.assertGreater(on1["mixed_graph_gib"], 0.0)
        self.assertGreater(on2["mixed_graph_gib"], on1["mixed_graph_gib"])
        self.assertAlmostEqual(
            on1["steady_gib"] - off["steady_gib"], on1["mixed_graph_gib"], places=5
        )
        # It is a *step*, not a pool: it must stay small next to the 119 GiB
        # steady state, or the whole idea is unaffordable.
        self.assertLess(on2["mixed_graph_gib"], 6.0)


# =========================================================================== #
# 6. the one-graph mode's host side
# =========================================================================== #
class TestStaticSegmentationBuffers(unittest.TestCase):
    """The runner's four persistent segmentation tensors.

    None of this needs a GPU: what closes the 48 holes is a *host* change --
    the index tensors are built here instead of by fla, and the conv's grid is
    pinned to the chunk cap instead of to this step's longest segment. Whether
    the resulting graph captures is a GPU question for the GPU tests;
    whether the numbers fed into it are the right numbers is this.
    """

    def _runner(self, holes: bool, chunk=24, n_segments=4, buckets=(2, 4)):
        model, rt = build_model(
            7, mixed_forward=True, mixed_graphs=True,
            mixed_graph_segments=n_segments, gdn_chunk_size=8,
        )
        return model, MixedGraphRunner(
            model, rt, chunk_tokens=chunk, buckets=buckets, holes=holes
        )

    def _padded_batch(self, model, runner, lens, slots, dec_slots):
        pad = pad_mixed_step(
            [toks(n, seed=i) for i, n in enumerate(lens)],
            [0] * len(lens), list(slots),
            list(dec_slots), [1] * len(dec_slots), [0] * len(dec_slots),
            runner.spec,
        )
        prefill = make_prefill_batch(
            pad.token_ids, pad.start_positions, pad.slots, model.device
        )
        return make_mixed_batch(
            prefill, pad.decode_slots, pad.decode_token_ids,
            pad.decode_positions, model.device,
        )

    def _bucket(self, runner, dec_slots):
        return runner.spec.bucket_for(len(dec_slots))

    def test_buffer_shapes_come_from_the_bound_not_from_a_step(self):
        model, runner = self._runner(holes=False)
        runner.prepare()
        self.assertEqual(runner.n_chunk_rows,
                         fla_static.max_chunk_rows(24, 4, 8))
        self.assertEqual(tuple(runner._chunk_indices.shape),
                         (runner.n_chunk_rows, 2))
        self.assertEqual(tuple(runner._chunk_offsets.shape), (5,))
        self.assertEqual(tuple(runner._cu_seqlens.shape), (5,))
        self.assertEqual(tuple(runner._seq_slot_ids.shape), (4,))

    def test_upload_matches_build_chunk_meta_and_the_batch(self):
        model, runner = self._runner(holes=False)
        runner.prepare()
        batch = self._padded_batch(model, runner, [9, 3], [0, 1], [2, 3])
        n_real = runner._upload_segmentation(batch)
        q_lens = [int(n) for n in batch.prefill.q_lens]
        idx, off, want_real = fla_static.build_chunk_meta(
            q_lens, runner.gdn_chunk_size, runner.n_chunk_rows
        )
        self.assertEqual(n_real, want_real)
        self.assertEqual(runner._chunk_indices.tolist(), idx)
        self.assertEqual(runner._chunk_offsets.tolist(), off)
        self.assertEqual(runner._cu_seqlens.tolist(),
                         batch.prefill.cu_seqlens.tolist())
        self.assertEqual(runner._seq_slot_ids.tolist(),
                         batch.prefill.seq_slot_ids.tolist())

    def test_every_admissible_segmentation_fits_the_static_buffer(self):
        """The one way this design fails silently: a step whose chunk rows
        overflow the buffer. The bound is proved in
        ``kernels_gdn/tests/test_fla_static.py``; this is the same claim
        against the shapes ``pad_mixed_step`` actually produces."""
        model, runner = self._runner(holes=False)
        runner.prepare()
        for real in ([20], [9, 3], [1, 1, 1], [17, 1, 1], [5, 5, 5]):
            if not runner.spec.fits(sum(real), len(real), 2):
                continue
            batch = self._padded_batch(
                model, runner, real, list(range(len(real))), [6, 7]
            )
            n_real = runner._upload_segmentation(batch)
            self.assertLessEqual(n_real, runner.n_chunk_rows, f"real={real}")

    @unittest.skipUnless(_FLA_STATIC_OK, "needs the fla static-index path (flash-linear-attention)")
    def test_no_holes_uses_persistent_objects_and_holes_uses_fresh_ones(self):
        """The segmented mode needs a *fresh* ``cu_seqlens`` object every step to
        defeat fla's identity memo; the one-graph mode needs a *persistent* one
        because the buffer
        is read inside the capture. Getting this backwards is silent: the
        graph would bake in the address of a tensor that is garbage-collected."""
        model, runner = self._runner(holes=False)
        runner.prepare()
        sh = runner._shape(self._bucket(runner, [2, 3]))
        b1 = self._padded_batch(model, runner, [9, 3], [0, 1], [2, 3])
        b2 = self._padded_batch(model, runner, [7, 5], [0, 1], [2, 3])
        runner._load(sh, b1)
        first = sh.ctx.cu_seqlens
        runner._load(sh, b2)
        self.assertIs(sh.ctx.cu_seqlens, first)
        self.assertIs(sh.ctx.cu_seqlens, runner._cu_seqlens)
        self.assertIs(sh.ctx.chunk_indices, runner._chunk_indices)
        self.assertEqual(sh.ctx.conv_max_seqlen, runner.chunk_tokens)
        # ...and the contents did follow the second step
        self.assertEqual(sh.ctx.cu_seqlens.tolist(), b2.prefill.cu_seqlens.tolist())

        model2, holed = self._runner(holes=True)
        holed.prepare()
        sh2 = holed._shape(self._bucket(holed, [2, 3]))
        c1 = self._padded_batch(model2, holed, [9, 3], [0, 1], [2, 3])
        c2 = self._padded_batch(model2, holed, [7, 5], [0, 1], [2, 3])
        holed._load(sh2, c1)
        first2 = sh2.ctx.cu_seqlens
        holed._load(sh2, c2)
        self.assertIsNot(sh2.ctx.cu_seqlens, first2)
        self.assertIsNone(sh2.ctx.chunk_indices)
        self.assertIsNone(sh2.ctx.conv_max_seqlen)

    def test_a_build_without_the_static_fla_path_degrades_to_holes(self):
        """Not a fallback for convenience: capturing one graph over an fla that
        builds its own indices would produce a *stale segmentation*, so the
        runner must refuse rather than try."""
        import qwenfast.kernels_gdn.fla_static as fs

        orig = fs.is_available
        fs.is_available = lambda: False
        try:
            model, rt = build_model(7, mixed_forward=True, mixed_graphs=True,
                                    mixed_graph_segments=4)
            r = MixedGraphRunner(model, rt, chunk_tokens=24, buckets=(2, 4),
                                 holes=False)
            self.assertTrue(r.holes)
            self.assertTrue(r.static_index_reason)
        finally:
            fs.is_available = orig

    def test_the_bucket_floor_drops_the_small_buckets_and_keeps_the_widest(self):
        model, rt = build_model(7, mixed_forward=True, mixed_graphs=True,
                                mixed_graph_segments=4, gdn_chunk_size=8,
                                mixed_graph_min_bucket=4)
        r = MixedGraphRunner(model, rt, chunk_tokens=24, buckets=(1, 2, 4, 8),
                             holes=False)
        self.assertEqual(r.buckets, (4, 8))
        # a step below the floor still graphs -- it pads up to it
        self.assertEqual(r.spec.bucket_for(1), 4)
        self.assertEqual(r.spec.bucket_for(8), 8)
        self.assertIsNone(r.spec.bucket_for(9))

    def test_a_floor_above_every_bucket_keeps_the_widest_rather_than_none(self):
        model, rt = build_model(7, mixed_forward=True, mixed_graphs=True,
                                mixed_graph_segments=4, gdn_chunk_size=8,
                                mixed_graph_min_bucket=999)
        r = MixedGraphRunner(model, rt, chunk_tokens=24, buckets=(1, 2, 4, 8),
                             holes=False)
        self.assertEqual(r.buckets, (8,))

    def test_a_chunk_bigger_than_the_context_splits_its_padding(self):
        """The failure this guards against is a *device-side assert* far from
        its cause: a pad segment longer than ``max_model_len``
        indexes past the rotary table, poisons the CUDA context, and the next
        GEMM reports ``CUBLAS_STATUS_EXECUTION_FAILED``. The runner splits the
        padding across the free plan rows instead, which is what lets a chunk
        several times the context length be graphed."""
        model, rt = build_model(7, mixed_forward=True, mixed_graphs=True,
                                mixed_graph_segments=4, max_model_len=32)
        r = MixedGraphRunner(model, rt, chunk_tokens=96, buckets=(2, 4))
        self.assertEqual(r.spec.max_pad_len, 32)
        self.assertEqual(r.spec.min_pad_rows, 3)   # ceil(96 / 32)
        self.assertEqual(r.spec.max_segments, 1)   # 4 rows - 3 for padding
        pad = pad_mixed_step([toks(5, seed=1)], [0], [0], [1], [1], [0], r.spec)
        lens = [len(t) for t in pad.token_ids]
        self.assertEqual(sum(lens), 96)
        self.assertEqual(len(lens), 4)
        for m in lens[1:]:
            self.assertLessEqual(m, 32, f"pad segment {m} > max_pad_len 32")
            self.assertGreaterEqual(m, 1)

    def test_a_chunk_that_cannot_be_padded_at_all_is_refused_at_build(self):
        model, rt = build_model(7, mixed_forward=True, mixed_graphs=True,
                                mixed_graph_segments=4, max_model_len=32)
        with self.assertRaises(ValueError) as cm:
            MixedGraphRunner(model, rt, chunk_tokens=512, buckets=(2, 4))
        self.assertIn("max_model_len", str(cm.exception))

    def test_uncapped_padding_is_ws_h2s_one_fat_segment(self):
        """`max_pad_len=0` must reproduce the single fat pad segment exactly, or
        every uncapped test in this file is testing a different function."""
        spec = MixedPadSpec(chunk_tokens=24, n_segments=4, buckets=(2, 4),
                            scratch_slot=99)
        self.assertEqual(spec.min_pad_rows, 1)
        self.assertEqual(spec.max_segments, 3)
        self.assertEqual(spec.pad_sizes(3, 20), [18, 1, 1])

    def test_the_synthetic_warmup_batch_uses_the_real_pad_spec(self):
        """The warmup batch must have the same shape a real step does.

        A spec rebuilt by hand here dropped ``max_pad_len`` and produced one
        8,185-token pad segment -- a ``kv_len`` of 8,185, i.e. 512 pages
        against a 173-wide page table, and an ``IndexError`` deep inside
        ``build_flashinfer_indices``."""
        model, rt = build_model(7, mixed_forward=True, mixed_graphs=True,
                                mixed_graph_segments=4, max_model_len=32)
        r = MixedGraphRunner(model, rt, chunk_tokens=96, buckets=(2, 4))
        batch = r._synthetic(4)
        self.assertEqual(sum(batch.prefill.q_lens), 96)
        self.assertEqual(len(batch.prefill.q_lens), 4)
        for n in batch.prefill.q_lens:
            self.assertLessEqual(int(n), r.spec.max_pad_len)

    def test_scratch_page_floor(self):
        from qwenfast.runtime.mixed_graphs import scratch_page_floor

        self.assertEqual(scratch_page_floor(1024, 16), 64)
        self.assertEqual(scratch_page_floor(4096, 16), 256)
        self.assertEqual(scratch_page_floor(1025, 16), 65)
        # capped: the slot holds one pad *segment*, not the whole chunk
        self.assertEqual(scratch_page_floor(8192, 16, 2752), 172)

    @unittest.skipUnless(_FLA_STATIC_OK, "needs the fla static-index path (flash-linear-attention)")
    def test_stats_reports_the_mode(self):
        model, runner = self._runner(holes=False)
        st = runner.stats()
        self.assertEqual(st["static_indices"], 1.0)
        self.assertEqual(st["holes_mode"], 0.0)
        self.assertEqual(st["chunk_index_rows"], float(runner.n_chunk_rows))


class TestConvGridBound(unittest.TestCase):
    """``max_seqlen`` is the conv's token-axis grid, so it is part of the
    captured shape -- and it must be an *upper* bound or the kernel silently
    drops the tail of a segment."""

    def test_a_bound_below_the_longest_segment_is_refused(self):
        from qwenfast.kernels_gdn import api as gdn_api

        with self.assertRaises(ValueError):
            gdn_api.causal_conv_prefill_varlen(
                torch.zeros(8, 4), torch.zeros(4, 4),
                seq_lens=[5, 3], cu_seqlens=torch.tensor([0, 5, 8], dtype=torch.int32),
                conv_state_pool=torch.zeros(2, 3, 4), slot_ids=torch.tensor([0, 1]),
                backend="torch", max_seqlen=4,
            )


class TestStaticIndexMemoryPlan(unittest.TestCase):
    BASE = dict(
        max_num_seqs=256, max_model_len=2752, page_size=16, n_kv_pages=44353,
        kv_cache_dtype="bf16", ssm_state_dtype="fp16", max_num_batched_tokens=8192,
        mixed_forward=True, mixed_graphs=True,
    )

    def test_one_graph_and_holes_are_priced_differently_and_both_stay_small(self):
        holes = plan_memory(**self.BASE, mixed_graph_chunk=1024,
                            mixed_graph_holes=True)
        full = plan_memory(**self.BASE, mixed_graph_chunk=1024,
                           mixed_graph_holes=False)
        self.assertNotEqual(holes["mixed_graph_gib"], full["mixed_graph_gib"])
        for plan in (holes, full):
            self.assertGreater(plan["mixed_graph_gib"], 0.0)
            self.assertLess(plan["mixed_graph_gib"], 6.0)

    def test_the_h_scratch_term_grows_with_the_chunk(self):
        a = plan_memory(**self.BASE, mixed_graph_chunk=1024, mixed_graph_holes=False)
        b = plan_memory(**self.BASE, mixed_graph_chunk=2048, mixed_graph_holes=False)
        self.assertGreater(b["mixed_graph_gib"], a["mixed_graph_gib"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
