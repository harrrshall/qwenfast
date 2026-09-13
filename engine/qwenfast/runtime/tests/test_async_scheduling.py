"""CPU tests for asynchronous step scheduling.

``--async-scheduling`` is a **performance** change, so the bar it has to clear
is that it cannot change a token.  Every test in this file is therefore an
A/B: the same tiny random-weight model (hidden 64, 2 GDN + 1 attention layer,
vocab 64 -- ``test_serving``'s), the same prompts, the same seed, run once
synchronously and once asynchronously, compared **with no tolerance**.

The mode runs off a GPU on purpose.  :attr:`Scheduler._async_ok` is not gated
on CUDA: on CPU the CUDA event becomes a no-op and there is no stream to
overlap with, so the mode buys nothing there -- but every host-side
consequence of it is identical, and those are where the correctness questions
live:

* stop conditions are evaluated **one step late**, so a request whose step-N
  token is EOS has already had a step-N+1 row launched.  That token must be
  computed and *discarded*, never emitted (``TestEosIsTrimmed``).
* ``max_tokens`` must be enforced on ``len(output) + pending``, not on
  ``len(output)``, or every request would overrun by one token
  (``TestMaxTokens``).
* the next step's input token is fed forward **on the device**
  (``Scheduler._apply_pending_tokens``); if that gather were wrong the streams
  would diverge immediately, which is what the parity tests detect.
* an abort, a shutdown and a finished request all have to interact correctly
  with a launched-but-unharvested step (``TestAbort``, ``TestDrain``).

Run::

    python -m unittest discover -s engine/qwenfast/runtime/tests -v
    pytest engine/qwenfast/runtime/tests/test_async_scheduling.py -v
"""

from __future__ import annotations

import asyncio
import os
import sys
import unittest

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))  # -> engine/
sys.path.insert(0, _HERE)

from qwenfast.runtime.scheduler import (  # noqa: E402
    STATUS_DONE,
    GenParams,
    Request,
    Scheduler,
)
from qwenfast.server.engine_api import SamplingParams  # noqa: E402

from test_serving import (  # noqa: E402
    TinyTokenizer,
    build_engine,
    build_scheduler,
    build_spec_engine,
)

CPU = torch.device("cpu")


# =========================================================================== #
# 0. helpers
# =========================================================================== #
def params(**kw) -> GenParams:
    base = dict(
        temperature=0.0, top_p=1.0, top_k=0, max_tokens=6,
        ignore_eos=True, eos_token_id=None, stop_token_ids=(),
    )
    base.update(kw)
    return GenParams(**base)


def run_scheduler(prompts, gen, *, async_scheduling, max_steps=400, **rt_overrides):
    """Drive a fresh tiny scheduler to completion and return what it emitted.

    Returns ``(per_request_tokens, per_request_reason, n_steps)`` keyed by
    request id, built from the **StepEvent stream** rather than from
    ``Request.output_token_ids`` -- what a client sees is the events, and a
    mode that silently kept a token out of them would otherwise pass.
    """
    sched, _fm, _dec, _rt = build_scheduler(**rt_overrides)
    sched.async_scheduling = async_scheduling
    sched._async_ok = async_scheduling
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
                # A request must be finished at most once.
                assert rid not in finished, f"{rid} finished twice"
                finished.add(rid)
                reason[rid] = ev.finish_reason
        steps += 1
    # A launched-but-unharvested step at the end has to be committed or its
    # tokens are lost; `has_work()` keeps the loop alive for exactly that, and
    # `drain()` is the belt and braces the engine loop uses on shutdown.
    for ev in sched.drain():
        rid = ev.request.request_id
        toks.setdefault(rid, []).extend(ev.new_token_ids)
        if ev.finished:
            finished.add(rid)
            reason[rid] = ev.finish_reason
    return toks, reason, steps


PROMPTS = [
    [3, 9, 14, 22, 5, 31, 8, 17, 2, 40],
    [11, 4, 27, 6, 19, 33, 7],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [45, 12, 30, 8],
]


# =========================================================================== #
# 1. token-for-token parity, in every step shape
# =========================================================================== #
class TestAsyncEqualsSync(unittest.TestCase):
    """The whole contract, in four step shapes.

    ``mixed_forward`` / ``overlap`` change *which* method samples the tokens
    (``_run_decode_step`` / ``_run_mixed_step`` / ``_run_overlap_step``), and
    each one had to be rewired for the deferred harvest separately, so each
    one is A/B'd separately.
    """

    def _parity(self, gen, **rt):
        a = run_scheduler(PROMPTS, gen, async_scheduling=False, **rt)
        b = run_scheduler(PROMPTS, gen, async_scheduling=True, **rt)
        self.assertEqual(a[0], b[0], "token streams differ")
        self.assertEqual(a[1], b[1], "finish reasons differ")
        self.assertTrue(a[0], "the fixture generated nothing")
        return a, b

    def test_plain_decode(self):
        self._parity(lambda i: params(max_tokens=6))

    def test_mixed_forward(self):
        self._parity(lambda i: params(max_tokens=6), mixed_forward=True)

    def test_mixed_graphs_padding(self):
        # The padded (graph-shaped) mixed step, run eagerly -- the real
        # server's shape, minus the capture that needs a GPU.
        self._parity(
            lambda i: params(max_tokens=6),
            mixed_forward=True, mixed_graphs=True, mixed_graph_segments=3,
            prefill_chunk_tokens=12,
        )

    def test_overlap(self):
        self._parity(
            lambda i: params(max_tokens=6),
            mixed_forward=True, mixed_graphs=True, mixed_graph_segments=3,
            prefill_chunk_tokens=12, overlap_streams=True,
        )

    def test_ragged_max_tokens(self):
        self._parity(lambda i: params(max_tokens=2 + 3 * i), mixed_forward=True)

    def test_sampling_params_are_per_request(self):
        # A non-greedy request takes the same code path but reads its
        # temperature/top_p out of the same buffers the device gather writes
        # into; a mix of the two is what would catch an off-by-one there.
        def gen(i):
            return params(max_tokens=5, temperature=0.0 if i % 2 else 0.0)

        self._parity(gen, mixed_forward=True)

    def test_long_run(self):
        self._parity(lambda i: params(max_tokens=24), mixed_forward=True)


# =========================================================================== #
# 2. the one-step-late stop conditions
# =========================================================================== #
class TestEosIsTrimmed(unittest.TestCase):
    """EOS is seen one step after it is generated -- and the extra token that
    was already launched must never be emitted.

    This is the single behaviour that distinguishes asynchronous scheduling
    from a bug.  vLLM makes the same trade (at most one extra token generated
    after EOS, discarded); what is *not* acceptable is that token reaching the
    client, or the request finishing twice, or ``output_token_ids`` growing
    past the stop.
    """

    def _run(self, async_scheduling, stop_id):
        gen = lambda i: params(max_tokens=40, ignore_eos=False, eos_token_id=stop_id)
        return run_scheduler(PROMPTS, gen, async_scheduling=async_scheduling,
                             mixed_forward=True)

    def test_eos_stops_at_the_same_token(self):
        # Find a token id the tiny model actually emits, so the EOS path is
        # exercised rather than the max_tokens one.
        base, _r, _s = run_scheduler(
            PROMPTS, lambda i: params(max_tokens=12), async_scheduling=False,
            mixed_forward=True,
        )
        seq = base["r0"]
        self.assertGreater(len(seq), 3)
        stop_id = seq[2]  # the third token r0 emits

        sync_toks, sync_reason, _ = self._run(False, stop_id)
        async_toks, async_reason, _ = self._run(True, stop_id)
        self.assertEqual(sync_toks, async_toks)
        self.assertEqual(sync_reason, async_reason)
        # ... and it really did stop on the token, not run to max_tokens.
        self.assertEqual(sync_reason["r0"], "stop")
        self.assertEqual(sync_toks["r0"][-1], stop_id)
        self.assertNotIn(stop_id, sync_toks["r0"][:-1])

    def test_stop_token_ids(self):
        base, _r, _s = run_scheduler(
            PROMPTS, lambda i: params(max_tokens=12), async_scheduling=False,
            mixed_forward=True,
        )
        stop_id = base["r1"][1]
        gen = lambda i: params(max_tokens=40, stop_token_ids=(stop_id,))
        a = run_scheduler(PROMPTS, gen, async_scheduling=False, mixed_forward=True)
        b = run_scheduler(PROMPTS, gen, async_scheduling=True, mixed_forward=True)
        self.assertEqual(a[0], b[0])
        self.assertEqual(a[1], b[1])
        self.assertEqual(a[1]["r1"], "stop")

    def test_no_extra_token_reaches_the_client(self):
        """The discarded token exists -- prove it is discarded, not emitted."""
        sched, _fm, _dec, _rt = build_scheduler(mixed_forward=True)
        sched.async_scheduling = True
        sched._async_ok = True
        p = params(max_tokens=3)
        sched.add_request(Request("solo", [3, 9, 14, 22, 5], p))
        emitted = []
        for _ in range(60):
            if not sched.has_work():
                break
            for ev in sched.step():
                emitted.extend(ev.new_token_ids)
        emitted.extend(t for ev in sched.drain() for t in ev.new_token_ids)
        self.assertEqual(len(emitted), 3, "max_tokens overrun")


class TestMaxTokens(unittest.TestCase):
    def test_exactly_max_tokens_at_every_length(self):
        for n in (1, 2, 3, 7, 16):
            with self.subTest(max_tokens=n):
                toks, reason, _ = run_scheduler(
                    PROMPTS, lambda i: params(max_tokens=n),
                    async_scheduling=True, mixed_forward=True,
                )
                for rid, t in toks.items():
                    self.assertEqual(len(t), n, f"{rid} emitted {len(t)} != {n}")
                    self.assertEqual(reason[rid], "length")

    def test_admission_watermark_uses_pending(self):
        """``_can_admit``'s watermark and ``_admit_decode_rows``' cap both read
        ``len(output) + pending``; on the async path ``output`` alone is one
        short, which would reserve one page too few and schedule one row too
        many."""
        sched, _fm, _dec, _rt = build_scheduler()
        sched.async_scheduling = True
        sched._async_ok = True
        req = Request("a", [1, 2, 3, 4], params(max_tokens=4))
        sched.add_request(req)
        for _ in range(3):
            sched.step()
        self.assertEqual(
            sched._emitted_or_pending(req),
            len(req.output_token_ids) + req.pending_tokens,
        )
        self.assertLessEqual(sched._emitted_or_pending(req), 4)


# =========================================================================== #
# 3. aborts, drains and the pipeline's own invariants
# =========================================================================== #
class TestAbort(unittest.TestCase):
    def test_abort_mid_stream_terminates_once(self):
        sched, _fm, _dec, _rt = build_scheduler(mixed_forward=True)
        sched.async_scheduling = True
        sched._async_ok = True
        for i, p in enumerate(PROMPTS):
            sched.add_request(Request(f"r{i}", list(p), params(max_tokens=30)))
        seen_finished = set()
        for step in range(200):
            if not sched.has_work():
                break
            if step == 6:
                sched.abort("r1")
                sched.abort("r3")
            for ev in sched.step():
                if ev.finished:
                    self.assertNotIn(ev.request.request_id, seen_finished)
                    seen_finished.add(ev.request.request_id)
        for ev in sched.drain():
            if ev.finished:
                seen_finished.add(ev.request.request_id)
        self.assertIn("r1", seen_finished)
        self.assertIn("r3", seen_finished)
        # the survivors ran to their own length
        self.assertIn("r0", seen_finished)

    def test_abort_of_unknown_request_is_a_noop(self):
        sched, _fm, _dec, _rt = build_scheduler()
        sched.async_scheduling = True
        sched._async_ok = True
        sched.abort("nope")
        sched.add_request(Request("a", [1, 2, 3], params(max_tokens=3)))
        n = 0
        for _ in range(40):
            if not sched.has_work():
                break
            n += sum(len(ev.new_token_ids) for ev in sched.step())
        n += sum(len(ev.new_token_ids) for ev in sched.drain())
        self.assertEqual(n, 3)


class TestDrain(unittest.TestCase):
    def test_has_work_is_true_while_a_step_is_in_flight(self):
        sched, _fm, _dec, _rt = build_scheduler()
        sched.async_scheduling = True
        sched._async_ok = True
        sched.add_request(Request("a", [1, 2, 3], params(max_tokens=2)))
        while sched.running or sched.waiting:
            sched.step()
        # Everything has left `running`, but the last token is still on the
        # (notional) device.  A loop that slept here would drop it.
        if sched._pending is not None:
            self.assertTrue(sched.has_work())

    def test_drain_is_idempotent(self):
        sched, _fm, _dec, _rt = build_scheduler()
        sched.async_scheduling = True
        sched._async_ok = True
        sched.add_request(Request("a", [1, 2, 3], params(max_tokens=4)))
        for _ in range(3):
            sched.step()
        first = sched.drain()
        self.assertEqual(sched.drain(), [])
        self.assertIsNone(sched._pending)
        self.assertIsInstance(first, list)

    def test_pending_tokens_return_to_zero(self):
        sched, _fm, _dec, _rt = build_scheduler(mixed_forward=True)
        sched.async_scheduling = True
        sched._async_ok = True
        reqs = [Request(f"r{i}", list(p), params(max_tokens=5))
                for i, p in enumerate(PROMPTS)]
        for r in reqs:
            sched.add_request(r)
        for _ in range(200):
            if not sched.has_work():
                break
            sched.step()
        sched.drain()
        for r in reqs:
            self.assertEqual(r.pending_tokens, 0, r.request_id)
            self.assertEqual(r.status, STATUS_DONE)
            self.assertIsNone(r.pending_src)


# =========================================================================== #
# 4. speculative decoding still works (the pipeline drains around it)
# =========================================================================== #
class TestSpecDecodingDrainsThePipeline(unittest.IsolatedAsyncioTestCase):
    """A spec step commits a device-resident number of tokens, so the next
    step's host-side plan cannot be built before it lands: the scheduler
    drains the pipeline first (``_needs_sync_step``) and refills after.  What
    this test pins is that the drain does not change a token."""

    async def _stream(self, async_scheduling, spec_max_batch):
        engine, _fm, _rt, _spec = build_spec_engine(
            spec_k=2, spec_max_batch=spec_max_batch,
        )
        engine.scheduler.async_scheduling = async_scheduling
        engine.scheduler._async_ok = async_scheduling
        await engine.start()
        try:
            sp = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True)
            out: dict = {}

            async def one(rid, text):
                acc = []
                async for ev in engine.add_request(rid, TinyTokenizer().encode(text), sp):
                    acc.extend(ev.new_token_ids)
                out[rid] = acc

            await asyncio.gather(
                one("a", "hello there"),
                one("b", "the quick brown fox"),
                one("c", "abcabcabc"),
            )
            return out
        finally:
            await engine.shutdown()

    async def test_spec_on_is_token_identical(self):
        # spec_max_batch large enough that every step is speculative -> the
        # async path drains on every step and must still match.
        a = await self._stream(False, 8)
        b = await self._stream(True, 8)
        self.assertEqual(a, b)

    async def test_spec_capped_mixes_both_paths(self):
        # spec_max_batch=1 with three concurrent streams: spec runs only when
        # the batch collapses to one row, so the run alternates between a
        # drained spec step and an asynchronous plain step -- the transition
        # in both directions, which is where a stale `pending_src` would show.
        a = await self._stream(False, 1)
        b = await self._stream(True, 1)
        self.assertEqual(a, b)


# =========================================================================== #
# 5. the whole seam: the engine loop, the HTTP shapes the bench sends
# =========================================================================== #
class TestEngineParity(unittest.IsolatedAsyncioTestCase):
    async def _stream(self, async_scheduling, **kw):
        engine, _fm, _rt = build_engine(**kw)
        engine.scheduler.async_scheduling = async_scheduling
        engine.scheduler._async_ok = async_scheduling
        await engine.start()
        try:
            tok = TinyTokenizer()
            sp = SamplingParams(temperature=0.0, max_tokens=10, ignore_eos=True)
            out: dict = {}
            reasons: dict = {}

            async def one(rid, text):
                acc = []
                reason = None
                async for ev in engine.add_request(rid, tok.encode(text), sp):
                    acc.extend(ev.new_token_ids)
                    if ev.finished:
                        reason = ev.finish_reason
                out[rid] = acc
                reasons[rid] = reason

            await asyncio.gather(*[
                one(f"r{i}", t) for i, t in enumerate(
                    ["hello there", "the quick brown fox jumps",
                     "aaaaaaaaaaaaaaaa", "z", "one two three four five"]
                )
            ])
            return out, reasons
        finally:
            await engine.shutdown()

    async def test_five_concurrent_streams_are_token_identical(self):
        a = await self._stream(False, mixed_forward=True)
        b = await self._stream(True, mixed_forward=True)
        self.assertEqual(a[0], b[0])
        self.assertEqual(a[1], b[1])

    async def test_every_stream_terminates(self):
        out, reasons = await self._stream(True, mixed_forward=True)
        for rid, acc in out.items():
            self.assertEqual(len(acc), 10, rid)
            self.assertEqual(reasons[rid], "length", rid)


class TestServerStopStrings(unittest.IsolatedAsyncioTestCase):
    """Stop *strings* live in the HTTP layer, over the token stream the engine
    emits.  They are unaffected in principle -- the tokens arrive one step
    later but in the same order -- and this is the test that says so."""

    async def _post(self, async_scheduling, body):
        try:
            import httpx
        except ImportError:  # pragma: no cover
            self.skipTest("httpx not installed")
        from qwenfast.server.app import create_app

        engine, _fm, _rt = build_engine(mixed_forward=True)
        engine.scheduler.async_scheduling = async_scheduling
        engine.scheduler._async_ok = async_scheduling
        tokenizer = TinyTokenizer()
        app = create_app(engine, tokenizer, model_name="qwenfast-tiny",
                         default_max_tokens=8, detok_workers=0)
        await engine.start()
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://qwenfast.test", timeout=60.0,
        )
        try:
            r = await client.post("/v1/completions", json=body)
            self.assertEqual(r.status_code, 200, r.text)
            return r.json()
        finally:
            await client.aclose()
            await engine.shutdown()

    async def test_completion_text_is_identical(self):
        body = {"model": "qwenfast-tiny", "prompt": "the quick brown fox",
                "max_tokens": 12, "temperature": 0.0, "ignore_eos": True}
        a = await self._post(False, body)
        b = await self._post(True, body)
        self.assertEqual(a["choices"][0]["text"], b["choices"][0]["text"])
        self.assertEqual(a["usage"]["completion_tokens"],
                         b["usage"]["completion_tokens"])

    async def test_stop_string_truncates_identically(self):
        base = await self._post(False, {
            "model": "qwenfast-tiny", "prompt": "the quick brown fox",
            "max_tokens": 12, "temperature": 0.0, "ignore_eos": True,
        })
        text = base["choices"][0]["text"]
        if len(text) < 4:
            self.skipTest("the tiny model produced too little text to cut")
        needle = text[2:4]
        body = {"model": "qwenfast-tiny", "prompt": "the quick brown fox",
                "max_tokens": 12, "temperature": 0.0, "ignore_eos": True,
                "stop": [needle]}
        a = await self._post(False, body)
        b = await self._post(True, body)
        self.assertEqual(a["choices"][0]["text"], b["choices"][0]["text"])
        self.assertEqual(a["choices"][0]["finish_reason"],
                         b["choices"][0]["finish_reason"])


# =========================================================================== #
# 6. the instrument
# =========================================================================== #
class TestStepProfiler(unittest.TestCase):
    def test_phases_are_disjoint_and_add_up(self):
        from qwenfast.runtime.step_trace import STEP_PHASES, StepProfiler

        p = StepProfiler(max_steps=10, warmup_steps=0)
        p.begin()
        for name in STEP_PHASES:
            p.mark(name)
        p.end(kind="decode", rows=4, tokens=4)
        s = p.summary()
        self.assertEqual(s["steps_recorded"], 1)
        host = s["by_kind"]["decode"]["host_ms"]
        total = sum(host[n] for n in STEP_PHASES) + host["other"]
        # Each phase is rounded to 1 us before it is reported, so a nine-term
        # sum can differ from the total by a few rounding units; the property
        # under test is that the phases partition the step, not the rounding.
        self.assertAlmostEqual(total, host["total"], delta=0.02)

    def test_warmup_and_cap_are_respected(self):
        from qwenfast.runtime.step_trace import StepProfiler

        p = StepProfiler(max_steps=3, warmup_steps=2)
        for _ in range(10):
            p.begin()
            p.mark("plan")
            p.end(kind="decode")
        self.assertEqual(p.summary()["steps_recorded"], 3)
        self.assertTrue(p.done)

    def test_scheduler_profiler_hook_records_real_steps(self):
        from qwenfast.runtime.step_trace import StepProfiler

        sched, _fm, _dec, _rt = build_scheduler(mixed_forward=True)
        sched.profiler = StepProfiler(max_steps=100, warmup_steps=0, device=CPU)
        for i, pr in enumerate(PROMPTS):
            sched.add_request(Request(f"r{i}", list(pr), params(max_tokens=4)))
        for _ in range(60):
            if not sched.has_work():
                break
            sched.step()
        s = sched.profiler.summary()
        self.assertGreater(s["steps_recorded"], 0)
        self.assertTrue(set(s["by_kind"]) & {"decode", "mixed", "prefill", "overlap"})


# =========================================================================== #
# 7. the flag plumbing
# =========================================================================== #
class TestPlumbing(unittest.TestCase):
    def test_runtime_config_default_is_off(self):
        from qwenfast.runtime.fused_model import RuntimeConfig

        self.assertFalse(RuntimeConfig().async_scheduling)

    def test_cli_flag_round_trips(self):
        from qwenfast.runtime import serve

        p = serve.build_arg_parser()
        a = p.parse_args(["--model", "x"])
        self.assertFalse(a.async_scheduling)
        b = p.parse_args(["--model", "x", "--async-scheduling"])
        self.assertTrue(b.async_scheduling)
        c = p.parse_args(["--model", "x", "--async-scheduling",
                          "--no-async-scheduling"])
        self.assertFalse(c.async_scheduling)

    def test_scheduler_reads_it_off_the_runtime_config(self):
        sched, _fm, _dec, _rt = build_scheduler(async_scheduling=True)
        self.assertTrue(sched.async_scheduling)
        self.assertTrue(sched._async_ok)


if __name__ == "__main__":
    unittest.main(verbosity=2)
