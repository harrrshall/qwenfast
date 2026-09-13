"""CPU unit tests for the canonical benchmark preset.

No GPU, no checkpoint: every assertion here is about *configuration identity*,
which is exactly the class of bug this module exists to prevent. The bug it was
written for is `fused_ops_backend`: the Triton fused GDN-gate/SwiGLU kernels
were measured as the faster path and their numbers quoted as the engine's,
but `serve.py` did not expose the flag, so the server executed
`RuntimeConfig`'s conservative `"torch"` default while benchmarks described
the other one. Nothing failed; the numbers just stopped meaning what
they said.

Run::

    python -m unittest discover -s engine/qwenfast/runtime/tests -v
"""

from __future__ import annotations

import argparse
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))  # -> engine/

from qwenfast.runtime import preset  # noqa: E402
from qwenfast.runtime.fused_model import RuntimeConfig  # noqa: E402
from qwenfast.runtime.serve import M1_DEFAULTS, add_runtime_args  # noqa: E402


class TestPresetMatchesServe(unittest.TestCase):
    """The one assertion this whole module is for: **the config being
    benchmarked is the config being served.**"""

    def test_every_preset_knob_matches_serve(self):
        for key, value in preset.CANONICAL_FAST.items():
            with self.subTest(knob=key):
                self.assertIn(
                    key, M1_DEFAULTS,
                    f"{key!r} is in the canonical preset but serve.M1_DEFAULTS does not "
                    f"have it -- a knob the bench sets and the server cannot is exactly "
                    f"the fused_ops_backend gap. Add it to M1_DEFAULTS and "
                    f"to add_runtime_args.",
                )
                self.assertEqual(
                    M1_DEFAULTS[key], value,
                    f"serve.M1_DEFAULTS[{key!r}] = {M1_DEFAULTS[key]!r} but the canonical "
                    f"preset says {value!r}. One of them is wrong; whichever it is, they "
                    f"must not disagree -- a bench number and a serve number are otherwise "
                    f"not comparable.",
                )

    def test_serve_has_no_knob_the_preset_ignores(self):
        """The other direction. A `RuntimeConfig` knob the server sets and the
        preset does not is a knob whose value silently differs between the two,
        which is the same defect pointing the other way."""
        extra = set(M1_DEFAULTS) - set(preset.CANONICAL_FAST)
        self.assertEqual(
            extra, set(),
            f"serve.M1_DEFAULTS knobs missing from preset.CANONICAL_FAST: {sorted(extra)}",
        )

    def test_serve_cli_defaults_reproduce_the_preset(self):
        """Not just the dict — the actual argparse defaults `serve.py` ships,
        because a flag whose `default=` drifted from `M1_DEFAULTS` would pass
        the test above and still serve the wrong thing."""
        p = add_runtime_args(argparse.ArgumentParser())
        args = p.parse_args([])
        for key, value in preset.CANONICAL_FAST.items():
            if key == "use_cuda_graphs":
                self.assertEqual(not args.no_graphs, value)
                continue
            with self.subTest(knob=key):
                self.assertTrue(hasattr(args, key), f"serve.py has no --{key.replace('_','-')}")
                self.assertEqual(getattr(args, key), value)

    def test_every_preset_knob_is_a_real_runtime_config_field(self):
        rt = RuntimeConfig()
        for key in preset.CANONICAL_FAST:
            with self.subTest(knob=key):
                self.assertTrue(hasattr(rt, key), f"RuntimeConfig has no field {key!r}")

    def test_canonical_fast_config_builds_and_equals_the_preset(self):
        rt = preset.canonical_fast_config()
        for key, value in preset.CANONICAL_FAST.items():
            with self.subTest(knob=key):
                self.assertEqual(getattr(rt, key), value)

    def test_overrides_win(self):
        rt = preset.canonical_fast_config(gemm_backend="deepgemm", gemm_weight_cache="multi")
        self.assertEqual(rt.gemm_backend, "deepgemm")
        self.assertEqual(rt.gemm_weight_cache, "multi")
        self.assertEqual(rt.fused_ops_backend, "triton")  # untouched


class TestPresetIsTheMeasuredFastest(unittest.TestCase):
    """Pins the knobs whose *wrong* value produces irreconcilable numbers, so
    a future edit that flips them has to say why."""

    def test_fused_ops_backend_is_triton(self):
        """B=1 12.638 -> 12.352 ms, launches/step 1,945 -> 1,689."""
        self.assertEqual(preset.CANONICAL_FAST["fused_ops_backend"], "triton")

    def test_norm_backend_is_triton(self):
        """B=32 22.629 -> 18.407 ms (-18.7%)."""
        self.assertEqual(preset.CANONICAL_FAST["norm_backend"], "triton")

    def test_kv_is_bf16_and_that_is_deliberate(self):
        """fp8 KV is *faster* at B=256 and slower at B<=8. The
        preset is "the config the engine ships", not the per-point argmax."""
        self.assertEqual(preset.CANONICAL_FAST["kv_cache_dtype"], "bf16")

    def test_gemm_backend_is_not_pinned(self):
        """Pinning bucket 1's winner costs 5x at M=256."""
        self.assertIsNone(preset.CANONICAL_FAST["gemm_backend"])


class TestBucketRule(unittest.TestCase):
    """The second cause of benchmark drift: the same nominal config measured
    30.0 / 20.4 / 18.1 ms at B=32 with 13 / 6 / 2 captured graph buckets."""

    def test_batches_rule_captures_exactly_what_is_benched(self):
        self.assertEqual(preset.buckets_for_batches([1, 32, 128]), (1, 32, 128))
        self.assertEqual(preset.buckets_for_batches([128, 1, 32, 32]), (1, 32, 128))

    def test_serve_rule_uses_the_full_table_capped_at_the_top_batch(self):
        got = preset.buckets_for_batches([32], "serve")
        self.assertEqual(got, tuple(b for b in preset.SERVE_BUCKETS if b <= 32))
        self.assertEqual(got[-1], 32)

    def test_serve_rule_appends_a_non_bucket_top_batch(self):
        got = preset.buckets_for_batches([100], "serve")
        self.assertEqual(got[-1], 100)

    def test_the_two_rules_really_do_differ(self):
        """Guards the premise: if these ever coincided the whole flag would be
        pointless, and this test would say so."""
        self.assertNotEqual(
            preset.buckets_for_batches([1, 8, 32, 64, 128, 256]),
            preset.buckets_for_batches([1, 8, 32, 64, 128, 256], "serve"),
        )

    def test_unknown_rule_is_rejected(self):
        with self.assertRaises(ValueError):
            preset.buckets_for_batches([1], "whatever")

    def test_canonical_bench_rule_is_batches(self):
        self.assertEqual(preset.CANONICAL_BENCH["buckets"], "batches")


class TestResolvedConfigReport(unittest.TestCase):
    def test_reports_every_preset_knob(self):
        rt = preset.canonical_fast_config()
        cfg = preset.resolved_config(rt)
        for key in preset.CANONICAL_FAST:
            with self.subTest(knob=key):
                self.assertIn(key, cfg)

    def test_no_deviations_for_a_canonical_config(self):
        cfg = preset.resolved_config(preset.canonical_fast_config())
        self.assertEqual(cfg["preset_deviations"], {})

    def test_deviations_are_named(self):
        cfg = preset.resolved_config(preset.canonical_fast_config(fused_ops_backend="torch"))
        self.assertEqual(cfg["preset_deviations"], {"fused_ops_backend": "torch"})

    def test_extra_measurement_knobs_are_carried(self):
        cfg = preset.resolved_config(preset.canonical_fast_config(), ctx_len=2048, repeats=3)
        self.assertEqual(cfg["ctx_len"], 2048)
        self.assertEqual(cfg["repeats"], 3)

    def test_graph_buckets_is_reported(self):
        """The knob worth 66% at B=32, which must be recorded next to the
        number it moves."""
        self.assertIn("graph_buckets", preset.REPORTED_FIELDS)
        cfg = preset.resolved_config(preset.canonical_fast_config(graph_buckets=(1, 32)))
        self.assertEqual(cfg["graph_buckets"], [1, 32])

    def test_format_is_printable_and_mentions_deviations(self):
        text = preset.format_resolved_config(
            preset.canonical_fast_config(norm_backend="torch")
        )
        self.assertIn("DEVIATIONS", text)
        self.assertIn("norm_backend", text)

    def test_format_says_so_when_there_are_none(self):
        text = preset.format_resolved_config(preset.canonical_fast_config())
        self.assertIn("exactly CANONICAL_FAST", text)


class TestApplyPreset(unittest.TestCase):
    def _parser(self):
        p = argparse.ArgumentParser()
        preset.add_preset_arg(p)
        p.add_argument("--norm-backend", default="torch")
        p.add_argument("--fused-ops-backend", default="torch")
        p.add_argument("--gemm-backend", default=None)
        p.add_argument("--ssm-state-dtype", nargs="+", default=["fp32", "fp16"])
        return p

    def test_no_preset_changes_nothing(self):
        p = self._parser()
        args = preset.apply_preset(p.parse_args([]), p, argv=[])
        self.assertEqual(args.norm_backend, "torch")

    def test_preset_fills_unset_knobs(self):
        p = self._parser()
        argv = ["--preset", "fastest"]
        args = preset.apply_preset(p.parse_args(argv), p, argv=argv)
        self.assertEqual(args.norm_backend, "triton")
        self.assertEqual(args.fused_ops_backend, "triton")

    def test_explicit_flags_win(self):
        """`--preset fastest --fused-ops-backend torch` must mean the canonical
        config with exactly one knob moved -- that is the only kind of A/B
        whose delta is attributable."""
        p = self._parser()
        argv = ["--preset", "fastest", "--fused-ops-backend", "torch"]
        args = preset.apply_preset(p.parse_args(argv), p, argv=argv)
        self.assertEqual(args.fused_ops_backend, "torch")
        self.assertEqual(args.norm_backend, "triton")

    def test_explicit_flag_equal_to_the_default_still_counts_as_explicit(self):
        p = self._parser()
        argv = ["--preset", "fastest", "--norm-backend", "torch"]
        args = preset.apply_preset(p.parse_args(argv), p, argv=argv)
        self.assertEqual(args.norm_backend, "torch")

    def test_equals_form_is_recognised(self):
        p = self._parser()
        argv = ["--preset", "fastest", "--norm-backend=torch"]
        args = preset.apply_preset(p.parse_args(argv), p, argv=argv)
        self.assertEqual(args.norm_backend, "torch")

    def test_a_list_valued_knob_stays_a_list(self):
        """`bench_runtime` spells `--ssm-state-dtype` as `nargs="+"` so one run
        can sweep it. Writing the preset's bare string there would make the
        caller iterate its characters."""
        p = self._parser()
        argv = ["--preset", "fastest"]
        args = preset.apply_preset(p.parse_args(argv), p, argv=argv)
        self.assertEqual(args.ssm_state_dtype, ["fp16"])

    def test_knobs_the_cli_does_not_have_are_skipped(self):
        p = self._parser()
        argv = ["--preset", "fastest"]
        args = preset.apply_preset(p.parse_args(argv), p, argv=argv)
        self.assertFalse(hasattr(args, "page_size"))


class TestBenchesUseThePreset(unittest.TestCase):
    """Every bench that reports a decode number must accept `--preset` and
    print the resolved config; a preset only one bench honours is not a
    canonical config."""

    def test_bench_runtime_has_preset_and_config_knobs(self):
        from qwenfast.runtime import bench_runtime

        args = bench_runtime.build_arg_parser().parse_args(["--model", "x"])
        for knob in ("preset", "gemm_weight_cache", "gemm_priority", "bucket_rule"):
            with self.subTest(knob=knob):
                self.assertTrue(hasattr(args, knob))

    def test_profile_step_has_preset(self):
        from qwenfast.runtime import profile_step

        args = profile_step.build_arg_parser().parse_args(["--model", "x"])
        for knob in ("preset", "gemm_weight_cache", "gemm_priority", "gemm_accuracy"):
            with self.subTest(knob=knob):
                self.assertTrue(hasattr(args, knob))

    def test_bench_spec_has_preset(self):
        from qwenfast.runtime import bench_spec

        args = bench_spec.build_arg_parser().parse_args(["--model", "x"])
        self.assertTrue(hasattr(args, "preset"))
        self.assertTrue(hasattr(args, "gemm_priority"))

    def test_bench_runtime_preset_selects_the_canonical_bucket_rule(self):
        """The bucket table is derived, not passed, so this is the only place
        the canonical rule can be asserted without a GPU."""
        from qwenfast.runtime import bench_runtime, preset as P

        parser = bench_runtime.build_arg_parser()
        argv = ["--model", "x", "--preset", "fastest", "--batch", "1", "32", "128"]
        args = P.apply_preset(parser.parse_args(argv), parser, argv=argv)
        rule = args.bucket_rule or P.CANONICAL_BENCH["buckets"]
        self.assertEqual(P.buckets_for_batches(args.batch, rule), (1, 32, 128))


if __name__ == "__main__":
    unittest.main()
