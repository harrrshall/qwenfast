"""Speculative sampling (``spec_decode.spec_sample_accept``, ``--spec-sampling``).

The contract: the tokens a sampled row emits from one speculative window are
distributed exactly as plain sampling from the target would emit them, one at
a time; greedy rows are unaffected. CPU only.
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

from test_spec_decode import build_m0, make_scheduler, spec_rt  # noqa: E402

from qwenfast.runtime.graphs import sample_tokens  # noqa: E402
from qwenfast.runtime.scheduler import GenParams, Request  # noqa: E402
from qwenfast.runtime.spec_decode import SpecConfig, spec_sample_accept  # noqa: E402


def target_probs(logits, temp, top_p, top_k):
    """Reference target distribution, straight from the plain sampler's rule."""
    n = 200_000
    t = sample_tokens(
        logits.unsqueeze(0).expand(n, -1).contiguous(),
        torch.full((n,), temp), torch.full((n,), top_p), torch.full((n,), float(top_k)),
        candidates=logits.numel(),
    )
    return torch.bincount(t.long(), minlength=logits.numel()).float() / n


class TestSpecSampleAccept(unittest.TestCase):
    V = 7

    def _window(self, seed):
        g = torch.Generator().manual_seed(seed)
        return torch.randn(3, self.V, generator=g) * 1.5  # n = 3 positions, k = 2

    def _check(self, temp, top_p, top_k, drafts):
        torch.manual_seed(0)
        logits = self._window(1)
        B = 200_000
        L = logits.unsqueeze(0).expand(B, -1, -1).contiguous()
        d = torch.tensor(drafts, dtype=torch.int32).unsqueeze(0).expand(B, -1).contiguous()
        toks, acc = spec_sample_accept(
            L, d, torch.full((B,), temp), torch.full((B,), top_p), torch.full((B,), float(top_k)),
            candidates=self.V,
        )
        acc = acc.long()
        first = toks[:, 0].long()
        p0 = target_probs(logits[0], temp, top_p, top_k)
        f0 = torch.bincount(first, minlength=self.V).float() / B
        self.assertLess((f0 - p0).abs().max().item(), 0.006, (f0, p0))
        # emission stops after a rejection; it continues exactly when the draft was taken
        took_d1 = first == drafts[0]
        self.assertTrue(torch.equal(took_d1, acc >= 1))
        if took_d1.sum() > 5000:
            second = toks[took_d1, 1].long()
            p1 = target_probs(logits[1], temp, top_p, top_k)
            f1 = torch.bincount(second, minlength=self.V).float() / second.numel()
            self.assertLess((f1 - p1).abs().max().item(), 0.012, (f1, p1))
            both = took_d1 & (toks[:, 1] == drafts[1])
            if both.sum() > 5000:
                third = toks[both, 2].long()
                p2 = target_probs(logits[2], temp, top_p, top_k)
                f2 = torch.bincount(third, minlength=self.V).float() / third.numel()
                self.assertLess((f2 - p2).abs().max().item(), 0.015, (f2, p2))

    def test_distribution_matches_plain_sampling(self):
        logits = self._window(1)
        likely = [int(logits[0].argmax()), int(logits[1].argmax())]
        unlikely = [int(logits[0].argmin()), int(logits[1].argmin())]
        for temp, top_p, top_k in [(1.0, 1.0, 0), (0.7, 0.8, 0), (1.0, 0.95, 3), (0.4, 1.0, 5)]:
            for drafts in (likely, unlikely):
                with self.subTest(temp=temp, top_p=top_p, top_k=top_k, drafts=drafts):
                    self._check(temp, top_p, top_k, drafts)

    def test_rows_are_independent_and_mixed_params_work(self):
        logits = self._window(2).unsqueeze(0).expand(4, -1, -1).contiguous()
        d = logits[:, :2].argmax(-1).to(torch.int32)
        toks, acc = spec_sample_accept(
            logits, d, torch.tensor([1.0, 0.5, 1.0, 0.3]), torch.tensor([1.0, 0.9, 0.5, 1.0]),
            torch.tensor([0.0, 0.0, 2.0, 1.0]), candidates=self.V,
        )
        self.assertEqual(toks.shape, (4, 3))
        self.assertTrue(((acc >= 0) & (acc <= 2)).all())
        # top_k = 1 is deterministic: always the argmax, drafts are argmax -> all accepted
        self.assertEqual(int(acc[3]), 2)
        self.assertEqual(toks[3].tolist(), logits[3].argmax(-1).tolist())


def gen(sched, prompts, temps, max_tokens=24):
    reqs = []
    for i, (p, t) in enumerate(zip(prompts, temps)):
        r = Request(f"r{i}", list(p), GenParams(temperature=t, top_p=0.95, top_k=20, max_tokens=max_tokens,
                                                ignore_eos=True, eos_token_id=1))
        sched.add_request(r)
        reqs.append(r)
    steps = 0
    while sched.has_work() and steps < 1000:
        sched.step()
        steps += 1
    return reqs


class TestSchedulerSpecSampling(unittest.TestCase):
    def setUp(self):
        self.m0, _ = build_m0(seed=11, with_mtp=True)
        g = torch.Generator().manual_seed(3)
        self.prompts = [torch.randint(2, 32, (9 + i,), generator=g).tolist() for i in range(4)]

    def test_greedy_rows_unchanged_with_spec_sampling_on(self):
        _, plain, _ = make_scheduler(self.m0, spec_rt())
        ref = [r.output_token_ids for r in gen(plain, self.prompts, [0.0] * 4)]
        _, sched, spec = make_scheduler(self.m0, spec_rt(), SpecConfig(k=2, greedy_only=False))
        got = [r.output_token_ids for r in gen(sched, self.prompts, [0.0] * 4)]
        self.assertEqual(got, ref)

    def test_sampled_requests_run_speculatively(self):
        _, sched, spec = make_scheduler(self.m0, spec_rt(), SpecConfig(k=2, greedy_only=False))
        reqs = gen(sched, self.prompts, [1.0, 0.7, 0.0, 1.0])
        for r in reqs:
            self.assertTrue(r.is_finished)
            self.assertEqual(len(r.output_token_ids), 24)
            self.assertTrue(all(0 <= t < 32 for t in r.output_token_ids))
        st = spec.stats()
        self.assertGreater(spec.steps, 0, "sampled batches took the speculative path")
        self.assertGreaterEqual(st["spec_accept_length"], 1.0)

    def test_greedy_only_default_keeps_sampled_batches_off_spec(self):
        _, sched, spec = make_scheduler(self.m0, spec_rt(), SpecConfig(k=2))
        gen(sched, self.prompts[:2], [1.0, 1.0])
        self.assertEqual(spec.steps, 0)


if __name__ == "__main__":
    unittest.main()
