"""Turn to turn prefix cache (``runtime/prefix_cache.py``). CPU, tiny model.

The contract under test: a request that resumes from a cached prompt-end
snapshot produces exactly the tokens a cold scheduler produces for the same
prompt (greedy, fp32), and the cache never leaks or double-frees a kv page.
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)  # -> test_runtime / test_spec_decode fixtures
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))  # -> engine/

from test_runtime import base_rt, build_m0, build_pieces
from test_spec_decode import build_m0 as build_m0_cached, make_scheduler, spec_rt

from qwenfast.runtime.scheduler import GenParams, Request, Scheduler
from qwenfast.runtime.spec_decode import SpecConfig


def run(sched, rid, prompt, max_tokens=6):
    r = Request(rid, list(prompt), GenParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True, eos_token_id=1))
    sched.add_request(r)
    for _ in range(500):
        if r.is_finished:
            break
        sched.step()
    for _ in range(5):  # async pipelines finish one step late
        if not sched.has_work():
            break
        sched.step()
    assert r.is_finished, rid
    return r


def conversation(seed=0, turns=4, first=13, grow=(5, 9, 3)):
    g = torch.Generator().manual_seed(seed)
    prompt = torch.randint(2, 32, (first,), generator=g).tolist()
    return prompt, g


def rt_pc(**kw):
    kw.setdefault("prefix_cache_entries", 4)
    kw.setdefault("prefix_cache_min_tokens", 4)
    kw.setdefault("page_size", 4)
    kw.setdefault("n_kv_pages", 256)
    kw.setdefault("max_pages_per_seq", 64)
    kw.setdefault("max_model_len", 128)
    return base_rt(**kw)


class TestPrefixCacheParity(unittest.TestCase):
    def setUp(self):
        self.model, _ = build_m0(seed=7)

    def _sched(self, **kw):
        rt = rt_pc(**kw)
        comps = build_pieces(rt, self.model)
        return Scheduler(comps.model, comps.decoder, rt)

    def _agent_turns(self, sched, n_turns=4, seed=3):
        """An agent-shaped conversation: every prompt is the previous prompt,
        a slice of the previous answer, and some new 'tool output'."""
        prompt, g = conversation(seed)
        outs, prompts = [], []
        for t in range(n_turns):
            r = run(sched, f"t{t}", prompt)
            prompts.append(list(prompt))
            outs.append(list(r.output_token_ids))
            extra = torch.randint(2, 32, (5 + 3 * t,), generator=g).tolist()
            prompt = prompt + r.output_token_ids[:4] + extra
        return prompts, outs

    def test_cached_turns_match_cold_scheduler(self):
        for chunk in (64, 5):  # one chunk per prompt, and prompts split across chunks
            with self.subTest(chunk=chunk):
                warm = self._sched(max_num_batched_tokens=chunk)
                prompts, outs = self._agent_turns(warm)
                st = warm.stats()
                self.assertEqual(st.prefix_hits, 3, "turns 2..4 each resume from the previous prompt")
                self.assertEqual(st.prefix_hit_tokens, sum(len(p) for p in prompts[:-1]))
                for i, p in enumerate(prompts):
                    cold = self._sched(max_num_batched_tokens=chunk, prefix_cache_entries=0)
                    self.assertEqual(run(cold, "c", p).output_token_ids, outs[i], f"turn {i}")

    def test_prefill_really_skipped(self):
        sched = self._sched(max_num_batched_tokens=64)
        prompt, g = conversation(5, first=40)
        run(sched, "a", prompt)
        seen = []
        orig = sched.model.prefill_forward

        def spy(batch, *a, **k):
            seen.append((int(batch.token_ids.numel()), int(batch.positions[0])))
            return orig(batch, *a, **k)

        sched.model.prefill_forward = spy
        run(sched, "b", prompt + [7, 8, 9])
        self.assertEqual(seen, [(3, len(prompt))], "only the 3 new tokens are prefilled, from the cached length")

    def test_divergent_prompt_does_not_hit(self):
        sched = self._sched()
        prompt, _ = conversation(9)
        run(sched, "a", prompt)
        edited = list(prompt)
        edited[2] = (edited[2] + 1) % 30 + 2
        run(sched, "b", edited + [5, 6])
        self.assertEqual(sched.stats().prefix_hits, 0)

    def test_no_page_leaks_under_pressure(self):
        # 40 usable pages of 4 tokens; many 20-40 token conversations force
        # entries to be evicted by admission and by decode growth
        sched = self._sched(n_kv_pages=41, max_num_seqs=4, prefix_cache_entries=3)
        pool = sched.model.kv_pool
        free0 = pool.num_free_pages
        g = torch.Generator().manual_seed(1)
        for c in range(6):
            prompt = torch.randint(2, 32, (20,), generator=g).tolist()
            for t in range(3):
                r = run(sched, f"c{c}t{t}", prompt, max_tokens=8)
                prompt = prompt + r.output_token_ids[:3] + [11, 12]
        st = sched.stats()
        self.assertGreater(st.prefix_hits, 0)
        self.assertEqual(pool.num_free_pages + sched.prefix_cache.pages_held(), free0)
        sched.prefix_cache.clear()
        self.assertEqual(pool.num_free_pages, free0)
        self.assertEqual(sched.slots.num_free, sched.model.n_slots)
        pool.verify_page_accounting()

    def test_concurrent_conversations(self):
        sched = self._sched(max_num_batched_tokens=16)
        g = torch.Generator().manual_seed(2)
        convs = [torch.randint(2, 32, (15 + i,), generator=g).tolist() for i in range(3)]
        reqs = []
        for i, p in enumerate(convs):
            r = Request(f"a{i}", p, GenParams(temperature=0.0, max_tokens=5, ignore_eos=True, eos_token_id=1))
            sched.add_request(r)
            reqs.append(r)
        while sched.has_work():
            sched.step()
        second = [p + r.output_token_ids[:2] + [9, 9, 9] for p, r in zip(convs, reqs)]
        reqs2 = []
        for i, p in enumerate(second):
            r = Request(f"b{i}", p, GenParams(temperature=0.0, max_tokens=5, ignore_eos=True, eos_token_id=1))
            sched.add_request(r)
            reqs2.append(r)
        while sched.has_work():
            sched.step()
        self.assertEqual(sched.stats().prefix_hits, 3)
        for p, r in zip(second, reqs2):
            cold = self._sched(max_num_batched_tokens=16, prefix_cache_entries=0)
            self.assertEqual(run(cold, "c", p, max_tokens=5).output_token_ids, r.output_token_ids)


class TestPrefixCacheWithSpecDecoding(unittest.TestCase):
    """The mtp head's kv layer rides in the same pages and its h_prev carry
    is snapshotted, so speculative decoding after a hit must match."""

    def test_spec_parity(self):
        m0, _ = build_m0_cached(seed=11, with_mtp=True)
        _, warm, _ = make_scheduler(m0, spec_rt(prefix_cache_entries=4, prefix_cache_min_tokens=4), SpecConfig(k=2))
        prompt, g = conversation(4, first=17)
        prompts, outs = [], []
        for t in range(3):
            r = run(warm, f"t{t}", prompt, max_tokens=10)
            prompts.append(list(prompt))
            outs.append(list(r.output_token_ids))
            prompt = prompt + r.output_token_ids[:5] + torch.randint(2, 32, (6,), generator=g).tolist()
        self.assertEqual(warm.stats().prefix_hits, 2)
        for i, p in enumerate(prompts):
            _, cold, _ = make_scheduler(m0, spec_rt(), SpecConfig(k=2))
            self.assertEqual(run(cold, "c", p, max_tokens=10).output_token_ids, outs[i], f"turn {i}")


class TestPrefixCacheMixedAsync(unittest.TestCase):
    """Mixed prefill+decode steps and asynchronous scheduling are the serving
    configuration; resume while other sequences decode must still match."""

    def test_mixed_async_parity(self):
        model, _ = build_m0(seed=7)
        for flags in (dict(mixed_forward=True), dict(mixed_forward=True, async_scheduling=True)):
            with self.subTest(**flags):
                rt = rt_pc(max_num_batched_tokens=32, **flags)
                comps = build_pieces(rt, model)
                sched = Scheduler(comps.model, comps.decoder, rt)
                # a long-running neighbour keeps decode rows in every step
                bg = Request("bg", list(range(2, 30)), GenParams(temperature=0.0, max_tokens=60, ignore_eos=True, eos_token_id=1))
                sched.add_request(bg)
                prompt, g = conversation(6, first=21)
                prompts, outs = [], []
                for t in range(3):
                    r = run(sched, f"t{t}", prompt, max_tokens=6)
                    prompts.append(list(prompt))
                    outs.append(list(r.output_token_ids))
                    prompt = prompt + r.output_token_ids[:3] + torch.randint(2, 32, (4,), generator=g).tolist()
                while sched.has_work():
                    sched.step()
                self.assertEqual(sched.stats().prefix_hits, 2)
                for i, p in enumerate(prompts):
                    rt0 = rt_pc(max_num_batched_tokens=32, prefix_cache_entries=0, **flags)
                    c0 = build_pieces(rt0, model)
                    cold = Scheduler(c0.model, c0.decoder, rt0)
                    self.assertEqual(run(cold, "c", p, max_tokens=6).output_token_ids, outs[i], f"turn {i}")


if __name__ == "__main__":
    unittest.main()
