"""CPU tests for O(1) scheduler admission.

O(1) admission is a **performance** change to the hottest host phase in the engine --
``Scheduler._admit_decode_rows``, 48.6 ms of a 63.2 ms decode step at
concurrency 256 -- so, exactly as for asynchronous scheduling, the bar it has to
clear is that it cannot change a token.

Three things changed, and each has its own failure mode:

1. ``PagedKVPool.pages_allocated`` is now an O(1) counter (``_n_alloc``)
   instead of a reduction over the host page-table row.  A counter is a cache
   of a derived quantity: the failure mode is a **desync**, where one writer
   of the page table forgets to update it and every later capacity decision
   for that slot is wrong.  ``verify_page_accounting`` is the assertion, and
   the tests below call it after every mutation *and* after every scheduler
   step.
2. The device half of ``ensure_capacity`` is **staged**: it updates the host
   mirror, records a dirty range, and defers the device write to one
   ``index_copy_`` per step.  The failure mode is a **missed flush**, where
   the device page table is stale when ``append_kv`` writes KV through it --
   which on CPU is directly observable, because the "device" table is a plain
   tensor and the tests can compare it to the mirror.
3. The FlashInfer decode plan is built into pinned staging and handed to
   FlashInfer on the host (``build_flashinfer_indices(staged=True)``).  That
   path is CUDA-only by construction, so what is tested here is that the
   CPU/degenerate path returns exactly what it always did.

Every end-to-end test is an A/B against ``legacy_admission()``, which
reinstalls the previous O(n) methods verbatim on the class, on the same tiny
random-weight model with the same seed, compared with **no tolerance**.

Run::

    python -m unittest discover -s engine/qwenfast/runtime/tests -v
    pytest engine/qwenfast/runtime/tests/test_admission.py -v
"""

from __future__ import annotations

import contextlib
import os
import sys
import unittest

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))  # -> engine/
sys.path.insert(0, _HERE)

from qwenfast.attn.kv_pool import KVPoolConfig, PagedKVPool  # noqa: E402
from qwenfast.runtime.scheduler import GenParams, Request  # noqa: E402

from test_serving import (  # noqa: E402
    TinyTokenizer,
    build_scheduler,
    build_spec_engine,
)

CPU = torch.device("cpu")


# =========================================================================== #
# 0. the A/B harness
# =========================================================================== #
@contextlib.contextmanager
def legacy_admission():
    """Reinstall the previous O(n) admission path on :class:`PagedKVPool`.

    Verbatim: ``pages_allocated`` as the host-mirror reduction it was, and
    ``ensure_capacity`` writing the device table inline with a pageable H2D.
    ``defer_page_table_writes`` becomes a no-op so nothing is batched.  This
    is what every parity test compares against -- it makes "identical to the
    old path" a *measurement* in this process rather than an appeal to the
    diff.
    """
    saved = (
        PagedKVPool.pages_allocated,
        PagedKVPool.ensure_capacity,
        PagedKVPool.defer_page_table_writes,
    )

    def pages_allocated(self, slot):
        return int((self.page_table_host[slot] >= 0).sum())

    def ensure_capacity(self, slot, num_tokens):
        self._check_slot(slot)
        need = self.pages_needed(num_tokens)
        have = int((self.page_table_host[slot] >= 0).sum())
        if need <= have:
            return
        if need > self.cfg.max_pages_per_seq:
            raise RuntimeError(
                f"slot {slot} needs {need} pages (> max_pages_per_seq={self.cfg.max_pages_per_seq})"
            )
        new_pages = self.allocator.alloc(need - have)
        t = torch.tensor(new_pages, dtype=self.page_table.dtype, device="cpu")
        self.page_table_host[slot, have:need] = t
        self.page_table[slot, have:need] = t.to(self.page_table.device)
        # keep the counter alive so a *mixed* run (legacy alloc, new free)
        # cannot desync -- the tests never mix, this is belt and braces.
        self._n_alloc[slot] = need

    @contextlib.contextmanager
    def defer(self):
        yield self

    PagedKVPool.pages_allocated = pages_allocated  # type: ignore[assignment]
    PagedKVPool.ensure_capacity = ensure_capacity  # type: ignore[assignment]
    PagedKVPool.defer_page_table_writes = defer  # type: ignore[assignment]
    try:
        yield
    finally:
        (
            PagedKVPool.pages_allocated,
            PagedKVPool.ensure_capacity,
            PagedKVPool.defer_page_table_writes,
        ) = saved


def params(**kw) -> GenParams:
    base = dict(
        temperature=0.0, top_p=1.0, top_k=0, max_tokens=8,
        ignore_eos=True, eos_token_id=None, stop_token_ids=(),
    )
    base.update(kw)
    return GenParams(**base)


def run_scheduler(prompts, gen, *, async_scheduling=False, max_steps=600,
                  check_accounting=True, **rt_overrides):
    """Drive a fresh tiny scheduler to completion; return what it emitted.

    Built from the **StepEvent stream**, like ``test_async_scheduling``'s twin
    of this helper: what a client sees is the events.  ``check_accounting``
    asserts the O(1) page counter still agrees with the host mirror *and* that
    the device page table is in sync with the mirror after every step -- the
    two invariants O(1) admission introduces.
    """
    sched, fm, _dec, _rt = build_scheduler(**rt_overrides)
    sched.async_scheduling = async_scheduling
    sched._async_ok = async_scheduling
    pool = fm.kv_pool
    for i, p in enumerate(prompts):
        sched.add_request(Request(f"r{i}", list(p), gen(i)))

    toks: dict = {}
    reason: dict = {}
    finished: set = set()
    steps = 0
    while sched.has_work() and steps < max_steps:
        for ev in sched.step():
            rid = ev.request.request_id
            toks.setdefault(rid, []).extend(ev.new_token_ids)
            if ev.finished:
                assert rid not in finished, f"{rid} finished twice"
                finished.add(rid)
                reason[rid] = ev.finish_reason
        if check_accounting:
            pool.verify_page_accounting()
            assert not pool._pt_dirty, "a step ended with an unflushed page-table write"
            assert torch.equal(pool.page_table, pool.page_table_host), (
                "device page table diverged from the host mirror"
            )
        steps += 1
    for ev in sched.drain():
        rid = ev.request.request_id
        toks.setdefault(rid, []).extend(ev.new_token_ids)
        if ev.finished:
            finished.add(rid)
            reason[rid] = ev.finish_reason
    return toks, reason, steps


def ab(prompts, gen, **kw):
    """``(new, legacy)`` from two independent runs of the same configuration."""
    new = run_scheduler(prompts, gen, **kw)
    with legacy_admission():
        old = run_scheduler(prompts, gen, check_accounting=False, **kw)
    return new, old


PROMPTS = [
    [3, 9, 14, 22, 5, 31, 8, 17, 2, 40],
    [11, 4, 27, 6, 19, 33, 7],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [45, 12, 30, 8],
    [1, 63, 7, 7, 21, 39, 18, 4, 4, 4, 51],
    [6, 6, 6, 25, 44],
    [17, 3, 28, 9, 62, 11, 2, 2, 35],
    [8, 41, 13, 29],
]


# =========================================================================== #
# 1. the pool: the O(1) counter and the staged write
# =========================================================================== #
def tiny_pool(**kw) -> PagedKVPool:
    cfg = dict(
        n_layers=1, num_kv_heads=1, head_dim=4, page_size=4,
        n_pages=64, max_seqs=8, max_pages_per_seq=16, dtype="bf16", device="cpu",
    )
    cfg.update(kw)
    return PagedKVPool(KVPoolConfig(**cfg))


class TestPageAccounting(unittest.TestCase):
    """``pages_allocated`` is O(1) now; it must still be *right*."""

    def test_counter_tracks_every_writer(self):
        pool = tiny_pool()
        s0, s1 = pool.alloc_slot(), pool.alloc_slot()
        pool.verify_page_accounting()

        pool.ensure_capacity(s0, 9)          # 3 pages
        self.assertEqual(pool.pages_allocated(s0), 3)
        pool.ensure_capacity(s0, 9)          # no-op
        self.assertEqual(pool.pages_allocated(s0), 3)
        pool.ensure_capacity(s0, 13)         # 4 pages
        self.assertEqual(pool.pages_allocated(s0), 4)
        pool.ensure_capacity(s1, 1)
        pool.verify_page_accounting()

        pool.seq_len[s0] = 5
        self.assertEqual(pool.reclaim_trailing_pages(s0), 2)
        self.assertEqual(pool.pages_allocated(s0), 2)
        pool.verify_page_accounting()

        self.assertEqual(pool.free_pages(s0), 2)
        self.assertEqual(pool.pages_allocated(s0), 0)
        pool.verify_page_accounting()

        pool.free_slot(s1)
        s2 = pool.alloc_slot()
        self.assertEqual(pool.pages_allocated(s2), 0)
        pool.verify_page_accounting()

    def test_counter_matches_the_reduction_it_replaced(self):
        """The old definition, evaluated after an arbitrary alloc/free mix."""
        pool = tiny_pool()
        slots = [pool.alloc_slot() for _ in range(4)]
        for i, s in enumerate(slots):
            pool.ensure_capacity(s, 1 + 5 * i)
        pool.free_pages(slots[1])
        pool.ensure_capacity(slots[1], 7)
        pool.ensure_capacity(slots[3], 30)
        for s in slots:
            self.assertEqual(
                pool.pages_allocated(s), int((pool.page_table_host[s] >= 0).sum())
            )

    def test_verify_catches_a_desync(self):
        pool = tiny_pool()
        s = pool.alloc_slot()
        pool.ensure_capacity(s, 5)
        pool._n_alloc[s] = 99
        with self.assertRaises(AssertionError):
            pool.verify_page_accounting()

    def test_map_page_keeps_the_counter(self):
        pool = tiny_pool()
        s = pool.alloc_slot()
        page = pool.allocator.alloc(1)[0]
        pool.map_page(s, 0, page)
        self.assertEqual(pool.pages_allocated(s), 1)
        self.assertEqual(int(pool.page_table[s, 0]), page)
        pool.verify_page_accounting()
        # re-mapping an existing cell is legal (`_init_scratch_slot` runs
        # twice: once in the model constructor, once in `SpecDecoder.warmup`)
        pool.map_page(s, 0, pool.allocator.alloc(1)[0])
        self.assertEqual(pool.pages_allocated(s), 1)
        pool.verify_page_accounting()
        with self.assertRaises(ValueError):
            pool.map_page(s, 5, 0)  # would leave a hole


class TestStagedPageTableWrite(unittest.TestCase):
    """One device write per pass -- and none at all in the common case."""

    def test_deferred_scope_batches_and_flushes(self):
        pool = tiny_pool()
        slots = [pool.alloc_slot() for _ in range(6)]
        with pool.defer_page_table_writes():
            for s in slots:
                pool.ensure_capacity(s, 5)          # 2 pages each: 6 dirty ranges
            self.assertEqual(len(pool._pt_dirty), 6)
            # host mirror leads inside the scope; every *host* read is exact
            for s in slots:
                self.assertEqual(pool.pages_allocated(s), 2)
        self.assertEqual(pool._pt_dirty, [])
        self.assertTrue(torch.equal(pool.page_table, pool.page_table_host))
        pool.verify_page_accounting()

    def test_no_boundary_crossed_is_zero_work(self):
        """The common decode step: every row still fits its last page."""
        pool = tiny_pool()
        slots = [pool.alloc_slot() for _ in range(6)]
        for s in slots:
            pool.ensure_capacity(s, 4)              # exactly 1 page each
        with pool.defer_page_table_writes():
            for s in slots:
                pool.ensure_capacity(s, 3)          # still 1 page: no-op
            self.assertEqual(pool._pt_dirty, [], "a no-op step staged a device write")
        self.assertEqual(pool.flush_page_table(), 0)

    def test_flush_is_idempotent(self):
        pool = tiny_pool()
        s = pool.alloc_slot()
        pool.ensure_capacity(s, 9)
        self.assertEqual(pool.flush_page_table(), 0)
        self.assertEqual(pool.flush_page_table(), 0)
        self.assertTrue(torch.equal(pool.page_table, pool.page_table_host))

    def test_nested_scopes_flush_once_at_the_outermost_exit(self):
        pool = tiny_pool()
        s = pool.alloc_slot()
        with pool.defer_page_table_writes():
            pool.ensure_capacity(s, 5)
            with pool.defer_page_table_writes():
                pool.ensure_capacity(s, 13)
                self.assertTrue(pool._pt_dirty)
            self.assertTrue(pool._pt_dirty, "an inner scope exit flushed early")
        self.assertEqual(pool._pt_dirty, [])
        self.assertTrue(torch.equal(pool.page_table, pool.page_table_host))

    def test_many_rows_cross_a_page_boundary_in_one_pass(self):
        """The step this whole change exists for, in miniature.

        Eight sequences all sitting exactly on a page boundary, all growing by
        one token in the same pass: eight new pages, one flush, and the page
        ids must be eight *distinct* pages in the right table cells.
        """
        pool = tiny_pool(page_size=4, n_pages=64, max_seqs=8)
        slots = [pool.alloc_slot() for _ in range(8)]
        for s in slots:
            pool.ensure_capacity(s, 8)              # 2 full pages
        before = pool.num_free_pages
        with pool.defer_page_table_writes():
            for s in slots:
                pool.ensure_capacity(s, 9)          # 3rd page for every row
            self.assertEqual(len(pool._pt_dirty), 8)
        self.assertEqual(pool.num_free_pages, before - 8)
        self.assertTrue(torch.equal(pool.page_table, pool.page_table_host))
        third = [int(pool.page_table[s, 2]) for s in slots]
        self.assertEqual(len(set(third)), 8, "two rows were handed the same page")
        self.assertTrue(all(p >= 0 for p in third))
        pool.verify_page_accounting()

    def test_append_kv_flushes_a_pending_write(self):
        """The safety net: ``append_kv`` reads the *device* table."""
        pool = tiny_pool()
        s = pool.alloc_slot()
        pool._pt_defer = 1                          # simulate a missed flush
        try:
            pool.ensure_capacity(s, 4)
            self.assertTrue(pool._pt_dirty)
        finally:
            pool._pt_defer = 0
        k = torch.zeros(1, 1, 4, dtype=torch.bfloat16)
        pool.append_kv(
            0,
            torch.tensor([s], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
            k, k,
        )
        self.assertEqual(pool._pt_dirty, [])
        self.assertTrue(torch.equal(pool.page_table, pool.page_table_host))

    def test_free_pages_only_scans_the_mapped_prefix(self):
        pool = tiny_pool(max_pages_per_seq=16)
        s = pool.alloc_slot()
        pool.ensure_capacity(s, 9)                  # 3 pages of 16 cells
        self.assertEqual(pool.free_pages(s), 3)
        self.assertEqual(pool.num_free_pages, pool.cfg.n_pages)
        self.assertEqual(pool.pages_allocated(s), 0)


class TestBuildIndicesUnchangedOnCpu(unittest.TestCase):
    """``staged=True`` is CUDA-only; on CPU it must be the identity."""

    def test_staged_matches_unstaged(self):
        pool = tiny_pool()
        slots = [pool.alloc_slot() for _ in range(3)]
        for i, s in enumerate(slots):
            pool.ensure_capacity(s, 3 + 4 * i)
        lens = [3, 7, 11]
        a = pool.build_flashinfer_indices(slots, seq_lens=lens)
        b = pool.build_flashinfer_indices(slots, seq_lens=lens, staged=True)
        for x, y in zip(a, b):
            self.assertTrue(torch.equal(x, y))


# =========================================================================== #
# 2. end-to-end parity with the path it replaces
# =========================================================================== #
class TestSchedulerParity(unittest.TestCase):
    """Token-for-token, finish-reason-for-finish-reason, no tolerance."""

    def _assert_same(self, new, old, what):
        n_toks, n_reason, _ = new
        o_toks, o_reason, _ = old
        self.assertEqual(sorted(n_toks), sorted(o_toks), f"{what}: request set differs")
        for rid in o_toks:
            self.assertEqual(n_toks[rid], o_toks[rid], f"{what}: {rid} token stream differs")
            self.assertEqual(n_reason.get(rid), o_reason.get(rid), f"{what}: {rid} reason")

    def test_plain_decode(self):
        new, old = ab(PROMPTS, lambda i: params(max_tokens=10))
        self._assert_same(new, old, "plain decode")

    def test_page_boundaries_crossed_by_many_rows_in_one_step(self):
        """``page_size=1``: *every* row allocates a page on *every* step.

        The worst case for a batched write -- 8 dirty ranges per step, every
        step, for the whole run -- and the case a stale device page table
        would corrupt immediately, because each new token's KV lands in a page
        the device table only learned about in this step's flush.
        """
        new, old = ab(
            PROMPTS, lambda i: params(max_tokens=12),
            page_size=1, n_kv_pages=1024, max_pages_per_seq=256,
        )
        self._assert_same(new, old, "page_size=1")

    def test_page_boundary_alignment_sweep(self):
        for page_size in (2, 3, 4, 8):
            with self.subTest(page_size=page_size):
                new, old = ab(
                    PROMPTS, lambda i: params(max_tokens=9),
                    page_size=page_size, n_kv_pages=512, max_pages_per_seq=128,
                )
                self._assert_same(new, old, f"page_size={page_size}")

    def test_async_scheduling(self):
        new, old = ab(PROMPTS, lambda i: params(max_tokens=10), async_scheduling=True)
        self._assert_same(new, old, "async")

    def test_async_scheduling_page_size_1(self):
        new, old = ab(
            PROMPTS, lambda i: params(max_tokens=12), async_scheduling=True,
            page_size=1, n_kv_pages=1024, max_pages_per_seq=256,
        )
        self._assert_same(new, old, "async, page_size=1")

    def test_eos_and_max_tokens(self):
        eos = TinyTokenizer().eos_token_id
        gen = lambda i: params(max_tokens=6 + i, ignore_eos=False, eos_token_id=eos)
        for async_scheduling in (False, True):
            with self.subTest(async_scheduling=async_scheduling):
                new, old = ab(PROMPTS, gen, async_scheduling=async_scheduling)
                self._assert_same(new, old, "eos")

    def test_mixed_and_overlapped_steps(self):
        for flags in (
            dict(mixed_forward=True),
            dict(mixed_forward=True, mixed_graphs=True, mixed_graph_segments=3,
                 prefill_chunk_tokens=12),
            dict(mixed_forward=True, mixed_graphs=True, mixed_graph_segments=3,
                 prefill_chunk_tokens=12, overlap_streams=True),
        ):
            for async_scheduling in (False, True):
                with self.subTest(flags=tuple(flags), async_scheduling=async_scheduling):
                    new, old = ab(
                        PROMPTS, lambda i: params(max_tokens=9),
                        async_scheduling=async_scheduling, **flags
                    )
                    self._assert_same(new, old, f"mixed {flags}")


class TestPreemptionParity(unittest.TestCase):
    """Preemption is order-dependent, so the loop had to stay a loop.

    A pool small enough that the running set cannot all grow forces
    ``_ensure_capacity_with_preemption`` to evict, and eviction *inside* the
    admission pass is exactly where a batched write could reorder something:
    the victim's ``free_pages`` resets a whole device row while other rows of
    the same pass have writes still staged.
    """

    def _run(self, **kw):
        return run_scheduler(
            PROMPTS, lambda i: params(max_tokens=14),
            page_size=2, n_kv_pages=48, max_pages_per_seq=64,
            max_num_seqs=8, **kw
        )

    def test_preemption_under_page_pressure(self):
        for async_scheduling in (False, True):
            with self.subTest(async_scheduling=async_scheduling):
                new = self._run(async_scheduling=async_scheduling)
                with legacy_admission():
                    old = run_scheduler(
                        PROMPTS, lambda i: params(max_tokens=14),
                        page_size=2, n_kv_pages=48, max_pages_per_seq=64,
                        max_num_seqs=8, async_scheduling=async_scheduling,
                        check_accounting=False,
                    )
                self.assertEqual(new[0], old[0])
                self.assertEqual(new[1], old[1])

    def test_page_exhaustion_is_survivable(self):
        """A pool too small for the working set must abort requests, not wedge.

        The same requests must be aborted, with the same reason, as on the
        legacy path -- an off-by-one in the O(1) counter would abort a
        different set.
        """
        tiny = dict(page_size=2, n_kv_pages=20, max_pages_per_seq=64, max_num_seqs=8)
        new = run_scheduler(PROMPTS, lambda i: params(max_tokens=20), **tiny)
        with legacy_admission():
            old = run_scheduler(
                PROMPTS, lambda i: params(max_tokens=20), check_accounting=False, **tiny
            )
        self.assertEqual(new[1], old[1], "different requests were aborted")
        self.assertEqual(new[0], old[0])


class TestSpecParity(unittest.TestCase):
    """A speculative step commits k+1 tokens and admits with ``window=k+1``.

    That is the one caller that asks ``_ensure_capacity_with_preemption`` for
    more than one new position, so it is the one that can allocate *two* pages
    for a row in a single pass -- and, at ``--spec-max-batch``, the one that
    alternates with asynchronous plain steps.
    """

    def _run_engine(self, *, spec_k, spec_max_batch, async_scheduling, **rt):
        engine, fm, _rt, _spec = build_spec_engine(
            spec_k=spec_k, spec_max_batch=spec_max_batch, **rt
        )
        sched = engine.scheduler
        sched.async_scheduling = async_scheduling
        sched._async_ok = async_scheduling
        out = {}
        for i, p in enumerate(PROMPTS[:4]):
            sched.add_request(Request(f"r{i}", list(p), params(max_tokens=10)))
        steps = 0
        while sched.has_work() and steps < 600:
            for ev in sched.step():
                out.setdefault(ev.request.request_id, []).extend(ev.new_token_ids)
            fm.kv_pool.verify_page_accounting()
            steps += 1
        for ev in sched.drain():
            out.setdefault(ev.request.request_id, []).extend(ev.new_token_ids)
        return out

    def test_spec_steps(self):
        for spec_k, spec_max_batch, async_scheduling in (
            (2, None, False),
            (2, None, True),
            (3, 2, True),      # alternates drained spec / asynchronous plain
        ):
            with self.subTest(k=spec_k, cap=spec_max_batch, a=async_scheduling):
                kw = dict(spec_k=spec_k, spec_max_batch=spec_max_batch,
                          async_scheduling=async_scheduling)
                new = self._run_engine(**kw)
                with legacy_admission():
                    old = self._run_engine(**kw)
                self.assertEqual(new, old)

    def test_spec_across_page_boundaries(self):
        """``page_size=4`` with ``window=k+1=4``: a new page per row, every step.

        Not ``page_size < n``, which is unrelatedly broken and has been since
        long before O(1) admission: ``SpecDecoder._fill_scratch_inputs`` is the
        only caller that reserves capacity on the *scratch* slot, and it runs
        only during CUDA-graph capture -- so an eager (``--no-graphs``)
        speculative step whose padding rows draft ``n`` tokens into the
        scratch slot needs ``page_size >= n`` for its single scratch page to
        hold them.  Confirmed pre-existing: page_size 1 and 2 fail identically
        with ``QWENFAST_WS_S3=0``, i.e. with the O(1) admission code paths
        turned off in the same binary.  Production runs page_size 16
        with ``--spec-k 3`` (n = 4) and captures graphs, so neither condition
        holds there.
        """
        kw = dict(spec_k=3, spec_max_batch=None, async_scheduling=False,
                  page_size=4, n_kv_pages=1024, max_pages_per_seq=256)
        new = self._run_engine(**kw)
        with legacy_admission():
            old = self._run_engine(**kw)
        self.assertEqual(new, old)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
