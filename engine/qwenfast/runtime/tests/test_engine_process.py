"""CPU tests for the out-of-process engine core.

Everything here runs a **real second process**: ``EngineCoreClient`` spawns
``engine_core.run_core``, which builds the same tiny random-weight model
``test_serving.py`` uses (hidden 64, 2 GDN + 1 attention layer, vocab 64) and
drives the same :class:`~qwenfast.runtime.scheduler.Scheduler`.  No mocks, no
in-process shim: if the IPC, the batching, the abort path or the crash
handling were wrong, these fail.

The load-bearing test is :class:`TestIdenticalToInProcess`.  Moving the engine
loop into another interpreter is a **performance** change, so the bar it has to
clear is that it cannot change a single token: the same prompts through the
in-process ``QwenFastEngine`` and through ``EngineCoreClient`` must produce
byte-identical output ids, from the same seed, with no tolerance.

Run::

    python -m pytest engine/qwenfast/runtime/tests/test_engine_process.py -v
    python engine/qwenfast/runtime/tests/test_engine_process.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
import unittest

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))  # -> engine/
sys.path.insert(0, _HERE)

from qwenfast.runtime.engine import QwenFastEngine  # noqa: E402
from qwenfast.runtime.engine_core import (  # noqa: E402
    CODEC,
    EngineCoreClient,
    resolve_builder,
)
from qwenfast.runtime.step_trace import StepTrace  # noqa: E402
from qwenfast.server.engine_api import SamplingParams  # noqa: E402

try:
    import zmq  # noqa: F401

    HAS_ZMQ = True
except ImportError:  # pragma: no cover
    HAS_ZMQ = False

try:
    import httpx

    HAS_HTTPX = True
except ImportError:  # pragma: no cover
    HAS_HTTPX = False

#: The tiny-model builders live in ``test_serving``; importing them at module
#: scope means the spawned child re-imports this module and gets them too.
from test_serving import (  # noqa: E402
    TinyTokenizer,
    build_tiny_model,
    build_tiny_mtp_model,
    serving_rt,
)

from qwenfast.runtime.fused_model import DeviceBuffers, FusedQwenForCausalLM  # noqa: E402
from qwenfast.runtime.graphs import GraphedDecoder  # noqa: E402
from qwenfast.runtime.spec_decode import SpecConfig, build_spec_decoder  # noqa: E402

CPU = torch.device("cpu")

#: The builder string the client hands the core.  Resolved by ``importlib`` in
#: the child, so it must be the *package* path -- not ``__main__`` -- even when
#: this file is run as a script.
BUILDER = "qwenfast.runtime.tests.test_engine_process:build_core_engine"


# =========================================================================== #
# 0. the builder the core process runs
# =========================================================================== #
def build_core_engine(seed: int = 7, spec_k: int = 0, boom: str = "", **rt_overrides):
    """Build an unstarted :class:`QwenFastEngine` on CPU.

    ``boom`` exists for :class:`TestCoreCrash`: ``"build"`` raises before the
    core can report READY, ``"step"`` monkey-patches the scheduler so the Nth
    step raises -- the CPU stand-in for a CUDA OOM mid-step, which is the
    failure this whole liveness path exists for.
    """
    if boom == "build":
        raise RuntimeError("deliberate startup failure (test)")

    if spec_k:
        model, _cfg = build_tiny_mtp_model(seed)
        rt = serving_rt(enable_mtp=True, **rt_overrides)
    else:
        model, _cfg = build_tiny_model(seed)
        rt = serving_rt(**rt_overrides)
    fm = FusedQwenForCausalLM.from_m0_module(model, rt)
    buf = DeviceBuffers(
        max_batch=rt.max_num_seqs, vocab_size=fm.config.vocab_size,
        max_pages=rt.n_kv_pages, device=CPU,
    )
    decoder = GraphedDecoder(fm, buf, rt)
    spec = build_spec_decoder(fm, buf, rt, SpecConfig(k=spec_k)) if spec_k else None
    engine = QwenFastEngine(
        fm, decoder, rt,
        eos_token_id=TinyTokenizer().eos_token_id,
        capture_graphs=False, spec=spec,
        spec_max_batch=rt.max_num_seqs if spec_k else None,
    )
    if boom == "step":
        real_step = engine.scheduler.step
        state = {"n": 0}

        def exploding_step():
            state["n"] += 1
            if state["n"] >= 3:
                raise RuntimeError("deliberate step failure (test)")
            return real_step()

        engine.scheduler.step = exploding_step  # type: ignore[method-assign]
    return engine


def build_in_process_engine(**kw) -> QwenFastEngine:
    """The same engine, for the in-process arm of the parity test."""
    return build_core_engine(**kw)


# =========================================================================== #
# 1. helpers
# =========================================================================== #
def sp(max_tokens: int = 12, **kw) -> SamplingParams:
    base = dict(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=max_tokens, ignore_eos=True)
    base.update(kw)
    return SamplingParams(**base)


async def collect(engine, request_id: str, prompt_ids, params) -> dict:
    """Drain one request's stream into a plain summary."""
    ids, steps, finished, reason = [], 0, False, None
    stats = None
    async for out in engine.add_request(request_id, prompt_ids, params):
        ids.extend(out.new_token_ids)
        steps += 1
        stats = out.stats
        if out.finished:
            finished, reason = True, out.finish_reason
    return {
        "ids": ids, "steps": steps, "finished": finished, "finish_reason": reason,
        "prompt_tokens": stats.prompt_tokens if stats else None,
        "completion_tokens": stats.completion_tokens if stats else None,
    }


def prompts(n: int, tok: TinyTokenizer):
    """``n`` distinct, in-vocabulary prompts of a few tokens each."""
    return [tok.encode(f"prompt number {i} says hello.") for i in range(n)]


def run(coro, timeout: float = 240.0):
    async def _wrapped():
        return await asyncio.wait_for(coro, timeout)

    return asyncio.run(_wrapped())


class CoreCase(unittest.IsolatedAsyncioTestCase):
    """Base class that always shuts the core down, even on failure -- an
    orphaned spawn child holding a model is exactly what CI does not need."""

    async def make_client(self, **builder_kwargs) -> EngineCoreClient:
        client = EngineCoreClient(
            BUILDER, builder_kwargs, start_timeout_s=180.0,
            stats_interval_s=0.05, idle_poll_ms=1, verbose=False,
        )
        self.addAsyncCleanup(client.shutdown)
        await client.start()
        return client


# =========================================================================== #
# 2. the parity test -- the reason this file exists
# =========================================================================== #
@unittest.skipUnless(HAS_ZMQ, "pyzmq not installed")
class TestIdenticalToInProcess(CoreCase):
    async def test_one_stream_is_token_identical(self):
        """The same prompt, the same seed, the same weights -- so the same
        tokens, with no tolerance. Moving the loop across a process boundary
        is allowed to change *when* a token appears, never *which*."""
        tok = TinyTokenizer()
        prompt = tok.encode("the quick brown fox jumps over")

        inproc = build_in_process_engine(seed=11)
        await inproc.start()
        try:
            want = await collect(inproc, "r0", prompt, sp(max_tokens=16))
        finally:
            await inproc.shutdown()

        client = await self.make_client(seed=11)
        got = await collect(client, "r0", prompt, sp(max_tokens=16))

        self.assertEqual(got["ids"], want["ids"])
        self.assertEqual(len(got["ids"]), 16)
        self.assertTrue(got["finished"])
        self.assertEqual(got["finish_reason"], want["finish_reason"])
        self.assertEqual(got["prompt_tokens"], len(prompt))
        self.assertEqual(got["completion_tokens"], 16)

    async def test_many_concurrent_streams_are_token_identical(self):
        """Eight interleaved streams: the batching is where a per-step message
        could scramble which delta belongs to which request."""
        tok = TinyTokenizer()
        ps = prompts(8, tok)

        inproc = build_in_process_engine(seed=5)
        await inproc.start()
        try:
            want = await asyncio.gather(
                *[collect(inproc, f"r{i}", p, sp(max_tokens=10)) for i, p in enumerate(ps)]
            )
        finally:
            await inproc.shutdown()

        client = await self.make_client(seed=5)
        got = await asyncio.gather(
            *[collect(client, f"r{i}", p, sp(max_tokens=10)) for i, p in enumerate(ps)]
        )

        for i, (g, w) in enumerate(zip(got, want)):
            self.assertEqual(g["ids"], w["ids"], f"stream {i} diverged")
            self.assertEqual(len(g["ids"]), 10)
            self.assertTrue(g["finished"])

    async def test_speculative_decoding_is_token_identical(self):
        """`--spec-k` lives inside `Scheduler.step()`, so it must be invisible
        to the transport -- including the multi-token yields it produces,
        which are exactly what a per-token protocol would have broken."""
        tok = TinyTokenizer()
        ps = prompts(3, tok)

        inproc = build_in_process_engine(seed=9, spec_k=2)
        await inproc.start()
        try:
            want = await asyncio.gather(
                *[collect(inproc, f"s{i}", p, sp(max_tokens=12)) for i, p in enumerate(ps)]
            )
        finally:
            await inproc.shutdown()

        client = await self.make_client(seed=9, spec_k=2)
        got = await asyncio.gather(
            *[collect(client, f"s{i}", p, sp(max_tokens=12)) for i, p in enumerate(ps)]
        )
        for g, w in zip(got, want):
            self.assertEqual(g["ids"], w["ids"])
            self.assertEqual(len(g["ids"]), 12)
            # However the tokens were produced -- one per step, or a spec step's
            # 1..k+1 at once -- the two arms must agree step for step, which is
            # the property a per-token protocol would have lost.
            self.assertEqual(g["steps"], w["steps"])


# =========================================================================== #
# 3. control
# =========================================================================== #
@unittest.skipUnless(HAS_ZMQ, "pyzmq not installed")
class TestAbort(CoreCase):
    async def test_abort_terminates_the_stream(self):
        client = await self.make_client(seed=3)
        tok = TinyTokenizer()
        prompt = tok.encode("abort me please")

        # 60, not 200: the tiny model's `max_context_len` is 128 and the HTTP
        # layer's 400-gate (which `_add_request` also enforces as a backstop)
        # would reject anything longer -- correctly.
        seen = []
        agen = client.add_request("a0", prompt, sp(max_tokens=60))
        async for out in agen:
            seen.append(out)
            if len(seen) == 2:
                await client.abort("a0")
            if out.finished:
                break
        # The contract (`engine_api.AsyncEngine.abort`): the iterator must
        # terminate, and it must not hang. Anything under `max_tokens` proves
        # the abort reached the core and came back.
        total = sum(len(o.new_token_ids) for o in seen)
        self.assertTrue(seen[-1].finished)
        self.assertLess(total, 60)
        self.assertEqual(seen[-1].finish_reason, "abort")

    async def test_abort_of_an_unknown_request_is_a_noop(self):
        client = await self.make_client(seed=3)
        await client.abort("never-admitted")
        # ...and the core is still serving.
        tok = TinyTokenizer()
        got = await collect(client, "after", tok.encode("still alive"), sp(max_tokens=4))
        self.assertEqual(len(got["ids"]), 4)

    async def test_many_streams_with_half_aborted(self):
        """Aborts interleaved with live streams: the survivors must be
        unaffected, which is what a shared-slot bug would break."""
        client = await self.make_client(seed=17)
        tok = TinyTokenizer()
        ps = prompts(6, tok)

        async def one(i, p):
            ids = []
            async for out in client.add_request(f"m{i}", p, sp(max_tokens=40)):
                ids.extend(out.new_token_ids)
                if i % 2 == 0 and len(ids) >= 3:
                    await client.abort(f"m{i}")
                if out.finished:
                    break
            return ids

        got = await asyncio.gather(*[one(i, p) for i, p in enumerate(ps)])
        for i, ids in enumerate(got):
            if i % 2 == 0:
                self.assertLess(len(ids), 40, f"stream {i} should have been aborted")
            else:
                self.assertEqual(len(ids), 40, f"stream {i} should have run to length")


@unittest.skipUnless(HAS_ZMQ, "pyzmq not installed")
class TestContextAndStats(CoreCase):
    async def test_max_context_len_crosses_the_boundary(self):
        client = await self.make_client(seed=3)
        self.assertGreater(client.max_context_len, 0)
        # The 400-gate the HTTP layer asks for has to
        # work with the model in another process.
        self.assertIsNone(client.context_length_error(4, 4))
        self.assertIsNotNone(
            client.context_length_error(client.max_context_len, 10)
        )

    async def test_stats_are_pushed_and_non_blocking(self):
        client = await self.make_client(seed=3)
        tok = TinyTokenizer()
        await collect(client, "s", tok.encode("count my tokens"), sp(max_tokens=8))
        # The core pushes a snapshot every `stats_interval_s`; give it one.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not client._core_stats:
            await asyncio.sleep(0.05)
        st = client.get_stats()
        self.assertGreater(st.kv_slots_total, 0)
        self.assertEqual(st.generation_tokens_total, 8)
        self.assertGreater(st.prompt_tokens_total, 0)
        self.assertEqual(st.tpot.count + st.ttft.count, 8)

    async def test_health_is_none_while_serving(self):
        client = await self.make_client(seed=3)
        self.assertIsNone(client.health())


# =========================================================================== #
# 4. failure
# =========================================================================== #
@unittest.skipUnless(HAS_ZMQ, "pyzmq not installed")
class TestCoreCrash(unittest.IsolatedAsyncioTestCase):
    async def test_a_core_that_cannot_start_fails_start(self):
        """`--engine-process` must not leave a live HTTP port in front of a
        dead engine: a core that dies during build raises out of `start()`,
        which fails uvicorn's lifespan."""
        client = EngineCoreClient(BUILDER, {"boom": "build"}, start_timeout_s=90.0,
                                  verbose=False)
        try:
            with self.assertRaises(RuntimeError) as cm:
                await client.start()
            self.assertIn("deliberate startup failure", str(cm.exception))
        finally:
            await client.shutdown()

    async def test_a_core_that_dies_mid_stream_releases_every_waiter(self):
        """Engine liveness, across a process boundary: the engine dies, `/health` must go unhealthy, and every in-flight
        generator must raise instead of hanging until the client times out."""
        client = EngineCoreClient(BUILDER, {"seed": 3, "boom": "step"},
                                  start_timeout_s=180.0, verbose=False)
        self.addAsyncCleanup(client.shutdown)
        await client.start()
        self.assertIsNone(client.health())

        tok = TinyTokenizer()
        with self.assertRaises(RuntimeError):
            await asyncio.wait_for(
                collect(client, "doomed", tok.encode("this will die"), sp(max_tokens=50)),
                timeout=60.0,
            )
        # `health()` is what `server/app.py` turns into a 503.
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and client.health() is None:
            await asyncio.sleep(0.05)
        self.assertIsNotNone(client.health())
        # And a *new* request fails fast rather than being admitted to a dead
        # loop -- the same "503 on request 1" contract `QwenFastEngine` gives.
        with self.assertRaises(RuntimeError):
            await collect(client, "after", tok.encode("nope"), sp(max_tokens=4))


# =========================================================================== #
# 4b. the whole seam: HTTP -> app.py -> client -> core process -> model
# =========================================================================== #
@unittest.skipUnless(HAS_ZMQ and HAS_HTTPX, "pyzmq + httpx required")
class TestServerThroughTheCore(unittest.IsolatedAsyncioTestCase):
    """``server/app.py`` must not be able to tell the two engines apart.

    Drives the exact request shapes ``benchmarks/bench_serve.py`` and
    ``evals/run_eval.py`` send -- streaming, ``ignore_eos``, ``include_usage``
    -- through ``create_app`` over an engine that is a different process.
    """

    async def asyncSetUp(self):
        from qwenfast.server.app import create_app

        self.tokenizer = TinyTokenizer()
        self.engine = EngineCoreClient(
            BUILDER, {"seed": 21}, start_timeout_s=180.0,
            stats_interval_s=0.05, verbose=False,
        )
        self.app = create_app(
            self.engine, self.tokenizer, model_name="qwenfast-tiny-core",
            default_max_tokens=8, detok_workers=0,
        )
        await self.engine.start()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app),
            base_url="http://qwenfast.test", timeout=120.0,
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.engine.shutdown()

    async def test_health_and_metrics(self):
        r = await self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")
        m = await self.client.get("/metrics")
        self.assertEqual(m.status_code, 200)
        self.assertIn("qwenfast", m.text)

    async def test_streaming_completion(self):
        payload = {
            "model": "qwenfast-tiny-core", "prompt": "a prompt for the sweep",
            "max_tokens": 9, "stream": True, "ignore_eos": True,
            "temperature": 0.0, "stream_options": {"include_usage": True},
        }
        r = await self.client.post("/v1/completions", json=payload)
        self.assertEqual(r.status_code, 200, r.text)
        chunks, done = _sse_events(r.text)
        self.assertTrue(done)
        usage = [c["usage"] for c in chunks if c.get("usage")][-1]
        self.assertEqual(usage["completion_tokens"], 9)

    async def test_non_streaming_chat(self):
        r = await self.client.post(
            "/v1/chat/completions",
            json={
                "model": "qwenfast-tiny-core",
                "messages": [{"role": "user", "content": "hello there"}],
                "max_tokens": 6, "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertIsInstance(body["choices"][0]["message"]["content"], str)
        self.assertEqual(body["usage"]["completion_tokens"], 6)

    async def test_over_length_request_is_a_400_not_a_dead_engine(self):
        """The over-length gate, with the model in another process: the 400
        gate reads `max_context_len` off the client, which got it from the
        core's READY message.

        `max_tokens` is a *ceiling* rather than a reservation: a short prompt
        with a huge `max_tokens` is clamped to `context - prompt` and answered,
        which is OpenAI's own behaviour, so that case is a 200. An over-length
        **prompt** (a position past the end of the rotary table) is the one
        thing that still has to be a 400.
        """
        ctx = self.engine.max_context_len
        # clamped, not refused: `max_tokens` is a ceiling now
        ok = await self.client.post(
            "/v1/completions",
            json={"model": "qwenfast-tiny-core", "prompt": "x",
                  "max_tokens": ctx + 1000},
        )
        self.assertEqual(ok.status_code, 200, ok.text)

        # a prompt longer than the context on its own leaves no room to reply
        r = await self.client.post(
            "/v1/completions",
            json={"model": "qwenfast-tiny-core", "prompt": "x " * (ctx + 64),
                  "max_tokens": 8},
        )
        self.assertEqual(r.status_code, 400, r.text)
        self.assertIn("maximum context length", r.text)
        h = await self.client.get("/health")
        self.assertEqual(h.status_code, 200)


def _sse_events(text: str):
    """Parse an SSE body into (json chunks, saw_done)."""
    chunks, done = [], False
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):].strip()
        if payload == "[DONE]":
            done = True
            continue
        chunks.append(json.loads(payload))
    return chunks, done


# =========================================================================== #
# 5. plumbing
# =========================================================================== #
class TestPlumbing(unittest.TestCase):
    def test_builder_resolution(self):
        self.assertIs(resolve_builder(BUILDER), build_core_engine)
        with self.assertRaises(ValueError):
            resolve_builder("no_colon_here")
        with self.assertRaises(AttributeError):
            resolve_builder("qwenfast.runtime.engine_core:nope")

    def test_codec_round_trips_a_step_message(self):
        from qwenfast.runtime.engine_core import MSG_STEP, _decode, _encode

        rows = [("req-1", [5, 6], False, None, 12, 2, 0.031),
                ("req-2", [7], True, "stop", 9, 40, None)]
        got = _decode(_encode((MSG_STEP, rows)))
        self.assertEqual(got[0], MSG_STEP)
        self.assertEqual(len(got[1]), 2)
        rid, ids, finished, reason, n_p, n_c, ttft = got[1][0]
        self.assertEqual(rid, "req-1")
        self.assertEqual(list(ids), [5, 6])
        self.assertFalse(finished)
        self.assertIsNone(reason)
        self.assertEqual((n_p, n_c), (12, 2))
        self.assertAlmostEqual(ttft, 0.031, places=6)
        self.assertIsNone(got[1][1][6])
        self.assertIn(CODEC, ("msgpack", "pickle"))

    def test_serve_exposes_the_flag_and_the_builder(self):
        from qwenfast.runtime import serve

        p = serve.build_arg_parser()
        a = p.parse_args(["--model", "/tmp/x"])
        self.assertFalse(a.engine_process)
        b = p.parse_args(["--model", "/tmp/x", "--engine-process",
                          "--step-trace-out", "/tmp/t.json"])
        self.assertTrue(b.engine_process)
        self.assertEqual(b.step_trace_out, "/tmp/t.json")
        self.assertTrue(hasattr(serve, "build_engine_for_core"))
        self.assertIs(
            resolve_builder("qwenfast.runtime.serve:build_engine_for_core"),
            serve.build_engine_for_core,
        )


class TestStepTrace(unittest.TestCase):
    """The instrument itself: the five buckets must be disjoint and add up."""

    def test_buckets_are_disjoint_and_sum_to_the_wall(self):
        tr = StepTrace(meta={"mode": "test"})
        t = time.perf_counter()
        for _ in range(5):
            tr.record(t_start=t, t_drained=t + 0.001, t_stepped=t + 0.011,
                      t_end=t + 0.012)
            t += 0.020  # 8 ms of gap before the next iteration
        s = tr.summary()
        self.assertEqual(s["steps"], 5)
        self.assertAlmostEqual(s["ms"]["step"], 50.0, places=3)
        self.assertAlmostEqual(s["ms"]["drain"], 5.0, places=3)
        self.assertAlmostEqual(s["ms"]["emit"], 5.0, places=3)
        # Four gaps of 8 ms (the first iteration has no predecessor).
        self.assertAlmostEqual(s["ms"]["gap"], 32.0, places=3)
        self.assertEqual(sum(s["gap_hist"]["counts"]), 5)

    def test_idle_is_not_charged_as_stall(self):
        tr = StepTrace()
        t = time.perf_counter()
        tr.record_idle(t_start=t, t_drained=t + 0.0001, t_end=t + 0.005)
        s = tr.summary()
        self.assertEqual(s["steps"], 0)
        self.assertEqual(s["idle_iterations"], 1)
        self.assertAlmostEqual(s["ms"]["idle"], 4.9, places=1)

    def test_write_produces_readable_json(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sub", "trace.json")
            tr = StepTrace(path, meta={"mode": "test"})
            t = time.perf_counter()
            tr.record(t_start=t, t_drained=t + 0.001, t_stepped=t + 0.01, t_end=t + 0.011)
            self.assertEqual(tr.write(), path)
            with open(path) as fh:
                body = json.load(fh)
            self.assertEqual(body["meta"]["mode"], "test")
            self.assertEqual(body["steps"], 1)
            self.assertIn("series", body)
            self.assertIn("host_stall_pct", body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
