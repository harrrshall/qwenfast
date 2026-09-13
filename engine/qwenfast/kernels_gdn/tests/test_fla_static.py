"""The static fla chunk-index path.

What is under test, in one sentence: **the index tensors this engine builds on
the host are byte-for-byte the ones fla builds on the device**, and padding
them to a fixed row count changes nothing a kernel can observe.

Three levels, and only the third needs fla installed:

* :class:`TestIndexArithmetic` — :func:`build_chunk_meta` /
  :func:`max_chunk_rows` alone. Pure Python, no torch device, no fla. This is
  where "the graph's row count is an upper bound for *every* segmentation"
  gets proved by enumeration rather than asserted.
* :class:`TestPaddingRows` — the surplus rows are duplicates of a real row and
  nothing else, and the real prefix is untouched.
* :class:`TestAgainstFla` — the reference: ``fla.ops.utils.index
  .prepare_chunk_indices`` / ``prepare_chunk_offsets`` on the same input.
  Skipped where fla is not importable (this is the only class that needs it),
  which is every CPU-only test machine.
* :class:`TestStaticIndexScope` — the guard rail: inside the scope fla is
  *forbidden* to build an index tensor, and the patch is fully reverted
  afterwards even on an exception. A half-patched fla would make every later
  eager prefill wrong, which is worse than any capture failing.

Run::

    python engine/qwenfast/kernels_gdn/tests/test_fla_static.py
"""

from __future__ import annotations

import itertools
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))  # -> engine/

from qwenfast.kernels_gdn import fla_static  # noqa: E402

BT = 64


def compositions(total: int, parts: int):
    """Every way to write ``total`` as ``parts`` positive integers."""
    for cuts in itertools.combinations(range(1, total), parts - 1):
        prev, out = 0, []
        for c in cuts:
            out.append(c - prev)
            prev = c
        out.append(total - prev)
        yield out


# =========================================================================== #
# 1. the arithmetic
# =========================================================================== #
class TestIndexArithmetic(unittest.TestCase):
    def test_rows_are_segment_id_and_intra_chunk_index(self):
        idx, off, n = fla_static.build_chunk_meta([130, 64, 1], chunk_size=BT)
        # 130 -> ceil(130/64) = 3 chunks; 64 -> 1; 1 -> 1.
        self.assertEqual(
            idx, [[0, 0], [0, 1], [0, 2], [1, 0], [2, 0]]
        )
        self.assertEqual(off, [0, 3, 4, 5])
        self.assertEqual(n, 5)
        self.assertEqual(off[-1], n)

    def test_offsets_are_the_exclusive_prefix_sum_of_chunk_counts(self):
        lens = [1, 63, 64, 65, 127, 128, 129]
        idx, off, n = fla_static.build_chunk_meta(lens, chunk_size=BT)
        counts = [-(-l // BT) for l in lens]
        self.assertEqual(counts, [1, 1, 1, 2, 2, 2, 3])
        acc, want = 0, [0]
        for c in counts:
            acc += c
            want.append(acc)
        self.assertEqual(off, want)
        self.assertEqual(n, sum(counts))
        # and every row belongs to the segment its offset range says it does
        for seg, (lo, hi) in enumerate(zip(off[:-1], off[1:])):
            for r in range(lo, hi):
                self.assertEqual(idx[r][0], seg)
                self.assertEqual(idx[r][1], r - lo)

    def test_max_chunk_rows_is_an_upper_bound_over_every_segmentation(self):
        """The property the static buffer's size rests on.

        Enumerated exhaustively at a size where enumeration is possible; the
        formula is size-independent, so this is a proof of the formula and not
        of one shape.
        """
        for total, parts in ((16, 4), (20, 5), (24, 3), (12, 4)):
            for cs in (2, 3, 4, 8):
                bound = fla_static.max_chunk_rows(total, parts, cs)
                worst = max(
                    sum(-(-l // cs) for l in lens)
                    for lens in compositions(total, parts)
                )
                self.assertGreaterEqual(
                    bound, worst,
                    f"bound {bound} < worst case {worst} for total={total} "
                    f"parts={parts} chunk_size={cs}",
                )
                self.assertEqual(
                    bound, worst,
                    f"bound {bound} is loose (worst case is {worst}) for "
                    f"total={total} parts={parts} chunk_size={cs}",
                )

    def test_the_serving_shapes(self):
        """The two the window measures, spelled out so a change is visible."""
        self.assertEqual(fla_static.max_chunk_rows(1024, 8, 64), 23)
        self.assertEqual(fla_static.max_chunk_rows(2048, 8, 64), 39)
        # and the graphed mixed step's actual padding really does reach the bound: one real
        # segment of `budget - 6` tokens plus six one-token pads plus one fat
        # pad is exactly the shape the scheduler produces at conc 256.
        lens = [1016] + [1] * 6 + [1024 - 1016 - 6]
        self.assertEqual(sum(lens), 1024)
        self.assertEqual(len(lens), 8)
        _, _, n = fla_static.build_chunk_meta(lens, 64)
        self.assertEqual(n, 23)

    def test_a_segmentation_that_does_not_fit_is_refused(self):
        with self.assertRaises(ValueError):
            fla_static.build_chunk_meta([130, 64], chunk_size=BT, n_rows=2)

    def test_max_chunk_rows_validates(self):
        with self.assertRaises(ValueError):
            fla_static.max_chunk_rows(4, 8, 64)  # fewer tokens than segments


# =========================================================================== #
# 2. the padding rows
# =========================================================================== #
class TestPaddingRows(unittest.TestCase):
    def test_surplus_rows_duplicate_the_last_real_row(self):
        lens = [130, 1, 1]
        n_rows = 9
        idx, off, n = fla_static.build_chunk_meta(lens, BT, n_rows)
        self.assertEqual(len(idx), n_rows)
        self.assertEqual(n, 5)
        self.assertEqual(idx[:n], [[0, 0], [0, 1], [0, 2], [1, 0], [2, 0]])
        for row in idx[n:]:
            self.assertEqual(row, idx[n - 1])

    def test_padding_never_points_past_a_segment(self):
        """The safety property. A duplicate row recomputes a chunk that really
        exists, so it is idempotent whether or not fla's stores are masked --
        which is why this and not a past-the-end sentinel."""
        lens = [200, 5, 70]
        n_rows = fla_static.max_chunk_rows(sum(lens), len(lens), BT)
        idx, off, n = fla_static.build_chunk_meta(lens, BT, n_rows)
        counts = [-(-l // BT) for l in lens]
        for seg, chunk in idx:
            self.assertLess(chunk, counts[seg], f"row ({seg},{chunk}) is past the end")

    def test_real_prefix_is_identical_with_and_without_padding(self):
        lens = [130, 64, 1, 200]
        bare, off_b, n_b = fla_static.build_chunk_meta(lens, BT)
        wide, off_w, n_w = fla_static.build_chunk_meta(lens, BT, n_b + 11)
        self.assertEqual(off_b, off_w)
        self.assertEqual(n_b, n_w)
        self.assertEqual(bare, wide[:n_b])


# =========================================================================== #
# 3. against fla itself
# =========================================================================== #
class TestAgainstFla(unittest.TestCase):
    """``build_chunk_meta`` must equal what fla would have built.

    This is the one claim that cannot be argued from the source alone: if it
    is wrong the graphed step silently computes a different segmentation than
    the eager one, and every parity test above it passes.
    """

    def setUp(self):
        try:
            import torch  # noqa: F401
            from fla.ops.utils.index import (  # noqa: F401
                prepare_chunk_indices,
                prepare_chunk_offsets,
            )
        except Exception as exc:  # pragma: no cover - CPU-only boxes
            self.skipTest(f"fla not importable: {exc}")

    def test_matches_prepare_chunk_indices_and_offsets(self):
        import torch
        from fla.ops.utils.index import prepare_chunk_indices, prepare_chunk_offsets

        for lens in ([130, 64, 1], [1016, 1, 1, 1, 1, 1, 1, 2], [7], [64, 64],
                     [1, 1, 1, 1, 1, 1, 1, 1017]):
            for cs in (16, 32, 64):
                cu = [0]
                for n in lens:
                    cu.append(cu[-1] + n)
                # fresh object every call: fla memoises on `is`-identity.
                t = torch.tensor(cu, dtype=torch.int32)
                want_i = prepare_chunk_indices(t, cs).tolist()
                want_o = prepare_chunk_offsets(torch.tensor(cu, dtype=torch.int32), cs)
                got_i, got_o, n_real = fla_static.build_chunk_meta(lens, cs)
                self.assertEqual(got_i, want_i, f"lens={lens} chunk_size={cs}")
                self.assertEqual(got_o, want_o.tolist(), f"lens={lens} chunk_size={cs}")
                self.assertEqual(n_real, len(want_i))


# =========================================================================== #
# 4. the scope
# =========================================================================== #
class TestStaticIndexScope(unittest.TestCase):
    def setUp(self):
        if not fla_static.is_available():
            self.skipTest(
                f"fla static path unavailable: {fla_static.unavailable_reason()}"
            )

    def test_inside_the_scope_fla_may_not_build_indices(self):
        import torch
        from fla.ops.utils import index as fla_index

        off = torch.tensor([0, 1], dtype=torch.int32)
        with fla_static.static_index_scope(off):
            self.assertIs(
                fla_index.prepare_chunk_offsets(None, 64), off,
                "the scope must hand fla *our* offsets, not let it derive any",
            )
            with self.assertRaises(fla_static.StaticIndexViolation):
                fla_index.prepare_chunk_indices(None, 64)

    def test_the_patch_is_reverted_even_on_an_exception(self):
        import torch
        from fla.ops.utils import index as fla_index

        before_i = fla_index.prepare_chunk_indices
        before_o = fla_index.prepare_chunk_offsets
        with self.assertRaises(ZeroDivisionError):
            with fla_static.static_index_scope(torch.tensor([0], dtype=torch.int32)):
                raise ZeroDivisionError
        self.assertIs(fla_index.prepare_chunk_indices, before_i)
        self.assertIs(fla_index.prepare_chunk_offsets, before_o)

    def test_the_patch_reaches_the_module_that_actually_calls_it(self):
        """``chunk_delta_h`` imports the symbol *by value*; patching only
        ``fla.ops.utils.index`` would be a silent no-op there, and a silent
        no-op is a stale segmentation baked into the graph."""
        import torch
        from fla.ops.common import chunk_delta_h

        off = torch.tensor([0, 1], dtype=torch.int32)
        with fla_static.static_index_scope(off):
            self.assertIs(chunk_delta_h.prepare_chunk_offsets(None, 64), off)


if __name__ == "__main__":
    unittest.main(verbosity=2)
