"""CPU end-to-end tests for serving: HTTP -> server -> scheduler -> model.

``test_runtime.py`` covers the scheduler and the ``AsyncEngine`` adapter in
isolation; ``server/tests/test_server.py`` covers the HTTP layer against
``MockEngine``. Nothing covered the *seam*: the real
:class:`~qwenfast.runtime.engine.QwenFastEngine` behind
``server/app.py::create_app``, driven by the exact request shapes
``benchmarks/bench_serve.py`` and ``evals/run_eval.py`` send. That seam is
what serving actually runs, so this file tests it -- on CPU, with a tiny random-weight
model (hidden 64, 2 GDN + 1 attention layer, vocab 64) and a character-level
tokenizer whose ids fit that vocab.

Two things here are *not* generic server tests and are the reason this file
exists at all:

* ``TestFlashInferPlanSeqLens`` pins two correctness invariants (the
  FlashInfer prefill/decode plans must be built over the **post-write**
  context length). Both bugs are invisible on the CPU/torch attention path,
  so these tests reach into
  ``AttentionRunner`` and assert on what would be handed to FlashInfer,
  rather than on output logits.
* ``TestBenchClientContract`` sends byte-for-byte what ``bench_serve.py``
  sends (``stream``, ``ignore_eos``, ``temperature: 0.0``,
  ``stream_options.include_usage``) and asserts on the fields it reads back,
  so an incompatibility fails here instead of on the GPU host 40 minutes
  into a sweep.

Run::

    python -m unittest discover -s engine/qwenfast/runtime/tests -v
    pytest engine/qwenfast/runtime/tests/test_serving.py -v
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import unittest

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))  # -> engine/
sys.path.insert(0, _HERE)  # -> this dir, for `test_runtime`'s tiny-model builders

from qwenfast.model import QwenFastForCausalLM  # noqa: E402
from qwenfast.runtime.engine import QwenFastEngine  # noqa: E402
from qwenfast.runtime.fused_model import (  # noqa: E402
    DeviceBuffers,
    FusedQwenForCausalLM,
    RuntimeConfig,
    make_prefill_batch,
)
from qwenfast.runtime.graphs import GraphedDecoder  # noqa: E402
from qwenfast.runtime.scheduler import GenParams, Request, Scheduler  # noqa: E402
from qwenfast.runtime.spec_decode import SpecConfig, build_spec_decoder  # noqa: E402
from qwenfast.server.app import create_app  # noqa: E402
from qwenfast.server.engine_api import SamplingParams  # noqa: E402

from test_runtime import tiny_config  # noqa: E402

try:
    import httpx

    HAS_HTTPX = True
except ImportError:  # pragma: no cover
    HAS_HTTPX = False

CPU = torch.device("cpu")
VOCAB = 64


# =========================================================================== #
# 0. tiny fixtures
# =========================================================================== #
class TinyTokenizer:
    """Character-level, lossless, ids inside the tiny model's 64-token vocab.

    ``server/fake_tokenizer.py`` cannot be reused here: its open vocabulary
    hands out ids from 1000 up, which would index straight off the end of a
    64-row embedding table. This one is deliberately dumb -- one id per
    character -- so ``decode(encode(text)) == text`` and the server's
    incremental detokenizer / stop-string matcher are exercised for real.
    """

    ALPHABET = "abcdefghijklmnopqrstuvwxyz .,?!0123456789\n:;'\"-()[]{}#*"

    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 1
        self._unk = 2
        self._c2i = {c: i + 3 for i, c in enumerate(self.ALPHABET)}
        self._i2c = {i: c for c, i in self._c2i.items()}
        self.vocab_size = VOCAB
        assert len(self._c2i) + 3 <= VOCAB, "alphabet must fit the tiny vocab"

    # -- HF surface the server actually calls ---------------------------------- #
    def encode(self, text, add_special_tokens: bool = False):
        return [self._c2i.get(c, self._unk) for c in str(text).lower()]

    def decode(self, ids, skip_special_tokens: bool = True, **_kw) -> str:
        out = []
        for i in ids:
            i = int(i)
            if i in (self.pad_token_id, self.eos_token_id):
                if not skip_special_tokens:
                    out.append("")
                continue
            out.append(self._i2c.get(i, ""))
        return "".join(out)

    def convert_tokens_to_ids(self, token):
        return self._c2i.get(token, self._unk)

    def apply_chat_template(self, messages, tokenize: bool = True, **_kw):
        text = "".join(str(m.get("content") or "") for m in messages)
        return self.encode(text) if tokenize else text


def build_tiny_model(seed: int = 7):
    """The ``test_runtime`` tiny model, widened to ``VOCAB`` so
    :class:`TinyTokenizer`'s ids are all in range."""
    torch.manual_seed(seed)
    cfg = tiny_config(head_dim=16)
    cfg.vocab_size = VOCAB
    model = QwenFastForCausalLM(cfg, with_mtp=False, mtp_hidden_first=False)
    for p in model.parameters():
        p.data.normal_(0, 0.02)
    return model.to(CPU).eval(), cfg


def serving_rt(**overrides) -> RuntimeConfig:
    """A CPU runtime config shaped like the production one, only tiny.

    ``device="cpu"`` is forced (not ``DEVICE``): these tests must never touch
    the GPU, which may be busy running a benchmark.
    ``max_num_batched_tokens=12`` with prompts of 20-40 characters
    guarantees chunked prefill spans several scheduler steps, which is the
    behaviour under test rather than an incidental detail.
    """
    kwargs = dict(
        device="cpu",
        dtype="fp32",
        ssm_state_dtype="fp32",
        kv_cache_dtype="bf16",
        page_size=4,
        max_num_seqs=8,
        n_kv_pages=512,
        max_pages_per_seq=64,
        max_num_batched_tokens=12,
        prefill_decode_ratio=4,
        mlp_tile_tokens=32,
        max_model_len=256,
        gdn_backend="torch",
        attn_backend="torch",
        norm_backend="torch",
        use_cuda_graphs=False,
    )
    kwargs.update(overrides)
    return RuntimeConfig(**kwargs)


def build_engine(seed: int = 7, **rt_overrides):
    model, _cfg = build_tiny_model(seed)
    rt = serving_rt(**rt_overrides)
    fm = FusedQwenForCausalLM.from_m0_module(model, rt)
    buf = DeviceBuffers(
        max_batch=rt.max_num_seqs, vocab_size=fm.config.vocab_size, max_pages=rt.n_kv_pages, device=CPU
    )
    decoder = GraphedDecoder(fm, buf, rt)
    engine = QwenFastEngine(fm, decoder, rt, eos_token_id=TinyTokenizer().eos_token_id, capture_graphs=False)
    return engine, fm, rt


def build_tiny_mtp_model(seed: int = 7):
    """``build_tiny_model``, plus the MTP head -- needed for spec decoding."""
    torch.manual_seed(seed)
    cfg = tiny_config(head_dim=16)
    cfg.vocab_size = VOCAB
    model = QwenFastForCausalLM(cfg, with_mtp=True, mtp_hidden_first=False)
    for p in model.parameters():
        p.data.normal_(0, 0.02)
    return model.to(CPU).eval(), cfg


def build_spec_engine(seed: int = 7, spec_k: int = 2, spec_max_batch=None, **rt_overrides):
    """The same seam ``build_engine`` tests, with
    ``--spec-k``/``--spec-max-batch`` wired the way ``serve.py`` wires them --
    a real :class:`~qwenfast.runtime.spec_decode.SpecDecoder` behind the exact
    :class:`~qwenfast.runtime.engine.QwenFastEngine` the HTTP layer drives."""
    model, _cfg = build_tiny_mtp_model(seed)
    rt = serving_rt(enable_mtp=True, **rt_overrides)
    fm = FusedQwenForCausalLM.from_m0_module(model, rt)
    buf = DeviceBuffers(
        max_batch=rt.max_num_seqs, vocab_size=fm.config.vocab_size, max_pages=rt.n_kv_pages, device=CPU
    )
    decoder = GraphedDecoder(fm, buf, rt)
    spec = build_spec_decoder(fm, buf, rt, SpecConfig(k=spec_k))
    engine = QwenFastEngine(
        fm, decoder, rt, eos_token_id=TinyTokenizer().eos_token_id, capture_graphs=False,
        spec=spec, spec_max_batch=spec_max_batch,
    )
    return engine, fm, rt, spec


def build_scheduler(seed: int = 7, **rt_overrides):
    model, _cfg = build_tiny_model(seed)
    rt = serving_rt(**rt_overrides)
    fm = FusedQwenForCausalLM.from_m0_module(model, rt)
    buf = DeviceBuffers(
        max_batch=rt.max_num_seqs, vocab_size=fm.config.vocab_size, max_pages=rt.n_kv_pages, device=CPU
    )
    decoder = GraphedDecoder(fm, buf, rt)
    return Scheduler(fm, decoder, rt), fm, decoder, rt


def gen_params(**kw) -> GenParams:
    base = dict(temperature=0.0, top_p=1.0, top_k=0, max_tokens=4, ignore_eos=True, eos_token_id=None)
    base.update(kw)
    return GenParams(**base)


# =========================================================================== #
# 1. FlashInfer plan correctness
# =========================================================================== #
class _RecordingPrefillWrapper:
    def __init__(self):
        self.calls = []

    def plan(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class TestFlashInferPlanSeqLens(unittest.TestCase):
    """The FlashInfer plan must cover the tokens the step is about to write.

    Both assertions below are about *ordering*, which is why neither can be
    written as a logits comparison on CPU:

    * prefill -- ``prefill_forward`` plans once, then each layer's
      ``FusedAttention.prefill`` calls ``append_kv``. So at plan time the
      pool's device ``seq_len`` still holds the previous chunk's length (0
      for a fresh sequence). Planning off that would attend to nothing.
    * decode -- ``GraphedDecoder.step`` plans, then replays the graph, and
      the append happens *inside* the replay. Planning off the device
      ``seq_len`` would be one token short every step.

    The torch fallback is correct in both cases by construction (prefill
    reads ``self._kv_lens``; ``torch_fallback_decode`` gathers post-append
    with ``length=None``), which is why a logits comparison on CPU cannot
    detect either hazard and these tests assert on the *plan inputs* instead.
    """

    def test_plan_prefill_uses_post_chunk_kv_lens(self):
        _sched, fm, _dec, _rt = build_scheduler()
        runner = fm.attn
        recorded = {}
        real_build = fm.kv_pool.build_flashinfer_indices

        def spy(slot_ids, seq_lens=None, **kw):
            # `**kw` for the keyword-only `staged=` argument.
            recorded["slots"] = list(slot_ids)
            recorded["seq_lens"] = None if seq_lens is None else list(seq_lens)
            return real_build(slot_ids, seq_lens=seq_lens, **kw)

        fm.kv_pool.build_flashinfer_indices = spy  # type: ignore[assignment]
        runner.pool.build_flashinfer_indices = spy  # type: ignore[assignment]
        prev_backend, prev_wrapper = runner.backend, runner._prefill_wrapper
        runner.backend = "flashinfer"
        runner._prefill_wrapper = _RecordingPrefillWrapper()
        try:
            # Two sequences, mid-prefill: slot 0 is on its second chunk
            # (8 committed + 5 new), slot 1 on its first (0 + 3).
            fm.reset_slot(0)
            fm.reset_slot(1)
            fm.kv_pool.ensure_capacity(0, 13)
            fm.kv_pool.ensure_capacity(1, 3)
            runner.plan_prefill([0, 1], q_lens=[5, 3], kv_lens=[13, 3])
        finally:
            runner.backend, runner._prefill_wrapper = prev_backend, prev_wrapper

        self.assertEqual(recorded["slots"], [0, 1])
        self.assertEqual(
            recorded["seq_lens"],
            [13, 3],
            "plan_prefill must forward kv_lens (post-chunk length); passing None reads the "
            "device seq_len, which is still the pre-chunk length at plan time",
        )

    def test_decode_step_plans_post_step_seq_lens(self):
        sched, fm, decoder, _rt = build_scheduler()
        captured = {}
        # The scheduler calls `prepare_step` and `replay`
        # rather than the `step` that composes them, so that the step profile
        # can time the FlashInfer plan apart from the launch. `prepare_step`
        # *is* where `seq_lens` is consumed (it is `plan_decode`'s argument),
        # so this spies one level down and asserts on exactly the same thing.
        real_prepare = decoder.prepare_step

        def spy(batch, slots, seq_lens=None):
            captured["batch"] = batch
            captured["slots"] = list(slots)
            captured["seq_lens"] = None if seq_lens is None else list(seq_lens)
            return real_prepare(batch, slots, seq_lens=seq_lens)

        decoder.prepare_step = spy  # type: ignore[assignment]

        sched.add_request(Request("a", [3, 4, 5, 6, 7], gen_params(max_tokens=3)))
        for _ in range(20):
            sched.step()
            if "seq_lens" in captured:
                break
        self.assertIn("seq_lens", captured, "no decode step ran")

        n_computed = captured["seq_lens"][0] - 1
        self.assertEqual(
            captured["seq_lens"][0],
            n_computed + 1,
            "decode must plan over num_computed_tokens + 1 (the token this step writes)",
        )
        # First decode step after a 5-token prompt: 5 committed, planning 6.
        self.assertEqual(captured["seq_lens"][0], 6)
        # Padding rows point at the scratch slot, which holds exactly 1 token.
        pad = captured["batch"] and captured["seq_lens"][captured["batch"] :]
        self.assertTrue(all(v == 1 for v in pad), f"padding rows must plan seq_len 1, got {pad}")

    def test_prefill_seq_len_matches_prompt_after_chunked_prefill(self):
        """The end state the plan fix depends on: after a prompt is fully
        prefilled in several chunks, the pool's own ``seq_len`` is the whole
        prompt (not the last chunk)."""
        sched, fm, _dec, rt = build_scheduler()
        prompt = list(range(3, 3 + 29))  # 29 tokens vs a 12-token chunk budget
        req = Request("p", prompt, gen_params(max_tokens=3))
        sched.add_request(req)
        seq_len_at_prompt_end = None
        for _ in range(60):
            sched.step()
            if seq_len_at_prompt_end is None and req.num_computed_tokens >= len(prompt):
                seq_len_at_prompt_end = int(fm.kv_pool.seq_len[req.slot])
            if req.is_finished:
                break
        self.assertTrue(req.is_finished)
        self.assertEqual(
            seq_len_at_prompt_end,
            len(prompt),
            "the pool's seq_len must be the whole prompt after the last prefill chunk, "
            "not just the last chunk (append_kv must fold duplicates with scatter_reduce_(amax))",
        )
        # The token sampled off the last prefill chunk is generated but not
        # yet committed to KV -- it becomes the *input* of the first decode
        # step -- so after 3 output tokens exactly 2 decode steps ran.
        self.assertEqual(len(req.output_token_ids), 3)
        self.assertEqual(req.num_computed_tokens, len(prompt) + 2)


# =========================================================================== #
# 2. scheduler-level serving behaviours
# =========================================================================== #
class TestChunkedPrefillAndBatching(unittest.TestCase):
    def test_chunked_prefill_spans_multiple_steps(self):
        sched, _fm, _dec, rt = build_scheduler()
        prompt = list(range(3, 3 + 30))
        req = Request("long", prompt, gen_params(max_tokens=2))
        sched.add_request(req)

        prefill_steps = 0
        while req.num_computed_tokens < len(prompt) and prefill_steps < 20:
            sched.step()
            prefill_steps += 1
        self.assertGreaterEqual(
            prefill_steps,
            3,
            f"a {len(prompt)}-token prompt at max_num_batched_tokens="
            f"{rt.max_num_batched_tokens} must take >=3 chunks",
        )
        self.assertEqual(req.num_computed_tokens, len(prompt))

    def test_concurrent_requests_of_different_lengths_all_complete(self):
        sched, _fm, _dec, _rt = build_scheduler()
        specs = [("s", list(range(3, 8)), 3), ("m", list(range(3, 23)), 5), ("l", list(range(3, 40)), 2)]
        reqs = []
        for rid, prompt, maxt in specs:
            r = Request(rid, prompt, gen_params(max_tokens=maxt))
            reqs.append(r)
            sched.add_request(r)

        for _ in range(300):
            sched.step()
            if all(r.is_finished for r in reqs):
                break

        for r, (_rid, prompt, maxt) in zip(reqs, specs):
            self.assertTrue(r.is_finished, f"{r.request_id} did not finish")
            self.assertEqual(len(r.output_token_ids), maxt, r.request_id)
            self.assertEqual(r.finish_reason, "length", r.request_id)
        self.assertEqual(sched.stats().num_running, 0)
        self.assertEqual(sched.stats().kv_pages_used, 1, "only the scratch slot's page should remain")


# =========================================================================== #
# 3. HTTP end-to-end against the real engine
# =========================================================================== #
def _sse_events(body: str):
    """Parse an SSE body the way ``bench_serve.py`` does."""
    chunks, done = [], False
    for line in body.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            done = True
            continue
        chunks.append(json.loads(payload))
    return chunks, done


@unittest.skipUnless(HAS_HTTPX, "httpx is required for the ASGI end-to-end tests")
class TestServerEndToEnd(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine, self.model, self.rt = build_engine()
        self.tokenizer = TinyTokenizer()
        self.app = create_app(
            self.engine, self.tokenizer, model_name="qwenfast-tiny", default_max_tokens=8, detok_workers=2
        )
        # httpx's ASGI transport does not run the lifespan, so drive the two
        # lifecycle hooks `create_app`'s lifespan would have called.
        await self.engine.start()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://qwenfast.test", timeout=60.0
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.engine.shutdown()

    # -- health / models (what the launch script's readiness probe uses) ------ #
    async def test_health_and_models(self):
        r = await self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        r = await self.client.get("/v1/models")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["data"][0]["id"], "qwenfast-tiny")

    async def test_metrics_scrape(self):
        r = await self.client.get("/metrics")
        self.assertEqual(r.status_code, 200)
        self.assertIn("qwenfast:num_requests_running", r.text)

    # -- /v1/completions ------------------------------------------------------- #
    async def test_completions_non_streaming_usage_and_length(self):
        r = await self.client.post(
            "/v1/completions",
            json={
                "model": "qwenfast-tiny",
                "prompt": "the quick brown fox jumps over the lazy dog",
                "max_tokens": 6,
                "temperature": 0.0,
                "ignore_eos": True,
            },
        )
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["usage"]["completion_tokens"], 6)
        self.assertEqual(body["usage"]["prompt_tokens"], len(self.tokenizer.encode("the quick brown fox jumps over the lazy dog")))
        self.assertEqual(body["choices"][0]["finish_reason"], "length")
        self.assertIsInstance(body["choices"][0]["text"], str)

    async def test_max_tokens_defaults_to_server_default(self):
        r = await self.client.post(
            "/v1/completions",
            json={"model": "qwenfast-tiny", "prompt": "hello", "temperature": 0.0, "ignore_eos": True},
        )
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["usage"]["completion_tokens"], 8)  # default_max_tokens=8

    async def test_greedy_is_reproducible(self):
        payload = {
            "model": "qwenfast-tiny",
            "prompt": "reproducible greedy decoding",
            "max_tokens": 6,
            "temperature": 0.0,
            "ignore_eos": True,
        }
        a = (await self.client.post("/v1/completions", json=payload)).json()
        b = (await self.client.post("/v1/completions", json=payload)).json()
        self.assertEqual(a["choices"][0]["text"], b["choices"][0]["text"])

    async def test_eos_stops_generation(self):
        """EOS handling: point the engine's EOS at the token this prompt
        greedily produces first, and the request must finish after exactly
        one token with ``finish_reason == "stop"`` (not ``"length"``)."""
        prompt = "eos handling check"
        base = await self.client.post(
            "/v1/completions",
            json={"model": "qwenfast-tiny", "prompt": prompt, "max_tokens": 4,
                  "temperature": 0.0, "ignore_eos": True},
        )
        self.assertEqual(base.status_code, 200, base.text)

        params = SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True)
        first_token = None
        async for step in self.engine.add_request("probe-eos", self.tokenizer.encode(prompt), params):
            if step.new_token_ids:
                first_token = step.new_token_ids[0]
                break
        await self.engine.abort("probe-eos")
        self.assertIsNotNone(first_token)

        self.engine.eos_token_id = int(first_token)
        r = await self.client.post(
            "/v1/completions",
            json={"model": "qwenfast-tiny", "prompt": prompt, "max_tokens": 4, "temperature": 0.0},
        )
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["usage"]["completion_tokens"], 1)
        self.assertEqual(r.json()["choices"][0]["finish_reason"], "stop")

    async def test_stop_string_truncates_output(self):
        """Stop strings are matched on detokenized text by the *server*
        (the engine only ever sees token ids), so the assertion is that the
        returned text stops before the stop string and is a strict prefix of
        the unstopped run."""
        payload = {
            "model": "qwenfast-tiny",
            "prompt": "stop string handling",
            "max_tokens": 12,
            "temperature": 0.0,
            "ignore_eos": True,
        }
        full = (await self.client.post("/v1/completions", json=payload)).json()["choices"][0]["text"]
        self.assertGreaterEqual(len(full), 4, f"need a few chars to cut on, got {full!r}")
        stop = full[3:5]

        stopped = await self.client.post("/v1/completions", json={**payload, "stop": [stop]})
        self.assertEqual(stopped.status_code, 200, stopped.text)
        text = stopped.json()["choices"][0]["text"]
        self.assertNotIn(stop, text)
        self.assertTrue(full.startswith(text), f"{text!r} is not a prefix of {full!r}")
        self.assertEqual(stopped.json()["choices"][0]["finish_reason"], "stop")

    # -- streaming ------------------------------------------------------------- #
    async def test_streaming_chunks_reassemble_to_the_non_streaming_text(self):
        payload = {
            "model": "qwenfast-tiny",
            "prompt": "streaming chunk correctness",
            "max_tokens": 7,
            "temperature": 0.0,
            "ignore_eos": True,
        }
        non_stream = (await self.client.post("/v1/completions", json=payload)).json()["choices"][0]["text"]

        r = await self.client.post(
            "/v1/completions",
            json={**payload, "stream": True, "stream_options": {"include_usage": True}},
        )
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.headers["content-type"].startswith("text/event-stream"))
        chunks, done = _sse_events(r.text)
        self.assertTrue(done, "stream must end with data: [DONE]")

        text = "".join(c["choices"][0].get("text") or "" for c in chunks if c.get("choices"))
        self.assertEqual(text, non_stream)

        finishes = [c["choices"][0]["finish_reason"] for c in chunks if c.get("choices")]
        self.assertIn("length", finishes)
        usages = [c["usage"] for c in chunks if c.get("usage")]
        self.assertTrue(usages, "stream_options.include_usage must produce a usage chunk")
        self.assertEqual(usages[-1]["completion_tokens"], 7)
        self.assertEqual(usages[-1]["total_tokens"],
                         usages[-1]["prompt_tokens"] + usages[-1]["completion_tokens"])

    async def test_chat_completions_streaming(self):
        r = await self.client.post(
            "/v1/chat/completions",
            json={
                "model": "qwenfast-tiny",
                "messages": [{"role": "user", "content": "chat streaming"}],
                "max_tokens": 5,
                "temperature": 0.0,
                "ignore_eos": True,
                "stream": True,
                "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        self.assertEqual(r.status_code, 200, r.text)
        chunks, done = _sse_events(r.text)
        self.assertTrue(done)
        roles = [c["choices"][0]["delta"].get("role") for c in chunks if c.get("choices")]
        self.assertIn("assistant", roles, "first chat chunk must carry the assistant role")
        usages = [c["usage"] for c in chunks if c.get("usage")]
        self.assertEqual(usages[-1]["completion_tokens"], 5)

    async def test_chat_completions_non_streaming_shape(self):
        """The exact response shape ``evals/run_eval.py`` reads."""
        r = await self.client.post(
            "/v1/chat/completions",
            json={
                "model": "qwenfast-tiny",
                "messages": [{"role": "user", "content": "what is two plus two?"}],
                "max_tokens": 6,
                "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertIsInstance(body["choices"][0]["message"]["content"], str)
        self.assertIn("completion_tokens", body["usage"])

    # -- concurrency ----------------------------------------------------------- #
    async def test_concurrent_requests_different_lengths(self):
        prompts = [
            ("short one", 3),
            ("a considerably longer prompt that will need several prefill chunks to finish", 6),
            ("middle sized prompt here", 4),
            ("x", 2),
            ("another long one, long enough that chunked prefill interleaves with decode", 5),
        ]

        async def one(i, prompt, maxt):
            r = await self.client.post(
                "/v1/completions",
                json={
                    "model": "qwenfast-tiny",
                    "prompt": prompt,
                    "max_tokens": maxt,
                    "temperature": 0.0,
                    "ignore_eos": True,
                },
                headers={"x-request-id": f"conc-{i}"},
            )
            return r, maxt, prompt

        results = await asyncio.gather(*[one(i, p, m) for i, (p, m) in enumerate(prompts)])
        for r, maxt, prompt in results:
            self.assertEqual(r.status_code, 200, r.text)
            body = r.json()
            self.assertEqual(body["usage"]["completion_tokens"], maxt, prompt)
            self.assertEqual(body["usage"]["prompt_tokens"], len(self.tokenizer.encode(prompt)))

        stats = self.engine.get_stats()
        self.assertEqual(stats.num_requests_running, 0)
        self.assertEqual(stats.num_requests_waiting, 0)

    async def test_concurrent_matches_serial_output(self):
        """Continuous batching must not change what a request generates:
        greedy output under a 5-way concurrent batch must equal the same
        request run alone."""
        prompts = ["alpha beta", "gamma delta epsilon", "zeta", "eta theta iota kappa", "lambda mu"]

        async def one(p):
            r = await self.client.post(
                "/v1/completions",
                json={"model": "qwenfast-tiny", "prompt": p, "max_tokens": 4,
                      "temperature": 0.0, "ignore_eos": True},
            )
            return r.json()["choices"][0]["text"]

        concurrent = await asyncio.gather(*[one(p) for p in prompts])
        serial = [await one(p) for p in prompts]
        self.assertEqual(concurrent, serial)

    # -- abort ------------------------------------------------------------------ #
    async def test_abort_releases_slot_and_pages(self):
        params = SamplingParams(temperature=0.0, max_tokens=64, ignore_eos=True)
        steps = []
        async for step in self.engine.add_request("abort-me", self.tokenizer.encode("abort test"), params):
            steps.append(step)
            if len(steps) == 2:
                await self.engine.abort("abort-me")
        self.assertTrue(steps[-1].finished)
        self.assertEqual(steps[-1].finish_reason, "abort")
        self.assertLess(len(steps), 64)

        for _ in range(200):
            if self.engine.get_stats().num_requests_running == 0:
                break
            await asyncio.sleep(0.01)
        stats = self.engine.get_stats()
        self.assertEqual(stats.num_requests_running, 0)
        self.assertEqual(stats.kv_slots_used, 1, "only the scratch slot's page should remain held")


# =========================================================================== #
# 4. the exact bench/eval client contract
# =========================================================================== #
@unittest.skipUnless(HAS_HTTPX, "httpx is required for the ASGI end-to-end tests")
class TestBenchClientContract(unittest.IsolatedAsyncioTestCase):
    """Byte-for-byte what ``benchmarks/bench_serve.py`` sends and reads."""

    async def asyncSetUp(self):
        self.engine, _model, _rt = build_engine(seed=11)
        self.tokenizer = TinyTokenizer()
        self.app = create_app(self.engine, self.tokenizer, model_name="qwen3.8-27b", default_max_tokens=512)
        await self.engine.start()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://qwenfast.test", timeout=60.0
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.engine.shutdown()

    async def test_bench_serve_payload(self):
        payload = {
            "model": "qwen3.8-27b",
            "max_tokens": 9,
            "stream": True,
            "ignore_eos": True,
            "temperature": 0.0,
            "stream_options": {"include_usage": True},
            "prompt": "a prompt of some length for the sweep",
        }
        r = await self.client.post(
            "/v1/completions", json=payload, headers={"Authorization": "Bearer EMPTY"}
        )
        self.assertEqual(r.status_code, 200, r.text)
        chunks, done = _sse_events(r.text)
        self.assertTrue(done)

        # bench_serve counts a "token" as any chunk with non-empty
        # choices[0].text, and takes prompt/completion counts from the last
        # usage it sees. ignore_eos must be honoured or the sweep's output
        # lengths (and therefore tok/s) are not comparable to vLLM's.
        text_chunks = [c for c in chunks if c.get("choices") and c["choices"][0].get("text")]
        self.assertTrue(text_chunks, "bench_serve fails the request if no non-empty text arrives")
        usage = [c["usage"] for c in chunks if c.get("usage")][-1]
        self.assertEqual(usage["completion_tokens"], 9)
        self.assertGreater(usage["prompt_tokens"], 0)

    async def test_run_eval_payload(self):
        r = await self.client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3.8-27b",
                "messages": [{"role": "user", "content": "solve this problem step by step."}],
                "max_tokens": 6,
                "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertIsInstance(body["choices"][0]["message"]["content"], str)
        self.assertIsInstance(body["usage"]["completion_tokens"], int)


# =========================================================================== #
# 4b. spec decoding through the real OpenAI streaming path
#     -- `--spec-k`/`--spec-max-batch` end to end,
#     over the exact seam `TestServerEndToEnd` covers for the plain path.
# =========================================================================== #
@unittest.skipUnless(HAS_HTTPX, "httpx is required for the ASGI end-to-end tests")
class TestSpecDecodingServerEndToEnd(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine, self.model, self.rt, self.spec = build_spec_engine(spec_k=2, spec_max_batch=2)
        self.tokenizer = TinyTokenizer()
        self.app = create_app(
            self.engine, self.tokenizer, model_name="qwenfast-tiny-spec", default_max_tokens=8, detok_workers=2
        )
        await self.engine.start()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://qwenfast.test", timeout=60.0
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.engine.shutdown()

    async def test_streaming_matches_non_streaming_and_usage_is_right(self):
        """The exact non-spec assertion (`test_streaming_chunks_reassemble_to
        _the_non_streaming_text`), against a server that actually has spec
        decoding turned on -- so a bug in the multi-token harvest or the
        streaming detokenizer's handling of it fails here, not silently."""
        payload = {
            "model": "qwenfast-tiny-spec",
            "prompt": "streaming with speculative decoding",
            "max_tokens": 9,
            "temperature": 0.0,
            "ignore_eos": True,
        }
        non_stream = (await self.client.post("/v1/completions", json=payload)).json()["choices"][0]["text"]

        r = await self.client.post(
            "/v1/completions",
            json={**payload, "stream": True, "stream_options": {"include_usage": True}},
        )
        self.assertEqual(r.status_code, 200, r.text)
        chunks, done = _sse_events(r.text)
        self.assertTrue(done, "stream must end with data: [DONE]")

        text = "".join(c["choices"][0].get("text") or "" for c in chunks if c.get("choices"))
        self.assertEqual(text, non_stream)

        usages = [c["usage"] for c in chunks if c.get("usage")]
        self.assertTrue(usages, "stream_options.include_usage must produce a usage chunk")
        self.assertEqual(usages[-1]["completion_tokens"], 9)
        self.assertEqual(usages[-1]["total_tokens"],
                         usages[-1]["prompt_tokens"] + usages[-1]["completion_tokens"])
        self.assertGreater(self.spec.steps, 0, "the request never actually spec-decoded")

    async def test_chat_completions_streaming_with_spec(self):
        r = await self.client.post(
            "/v1/chat/completions",
            json={
                "model": "qwenfast-tiny-spec",
                "messages": [{"role": "user", "content": "chat streaming with spec"}],
                "max_tokens": 6,
                "temperature": 0.0,
                "ignore_eos": True,
                "stream": True,
                "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        self.assertEqual(r.status_code, 200, r.text)
        chunks, done = _sse_events(r.text)
        self.assertTrue(done)
        usages = [c["usage"] for c in chunks if c.get("usage")]
        self.assertEqual(usages[-1]["completion_tokens"], 6)

    async def test_streaming_emits_at_most_k_plus_1_tokens_per_event(self):
        """The precise per-step contract: ``StepEvent
        .new_token_ids`` -- what the OpenAI streaming path bundles into one
        SSE delta -- must never carry more than ``k + 1`` tokens, whichever
        path (spec or, once ``--spec-max-batch`` vetoes it, plain -> exactly
        1) produced it. Checked at the engine's own streaming seam (the same
        async generator ``server/app.py`` iterates), because after
        detokenization a delta's *character* count no longer maps 1:1 to a
        token count.
        """
        params = SamplingParams(temperature=0.0, max_tokens=30, ignore_eos=True)
        n_events = 0
        async for step in self.engine.add_request(
            "probe-window", self.tokenizer.encode("token budget probe"), params
        ):
            n_events += 1
            self.assertLessEqual(
                len(step.new_token_ids), self.spec.n,
                f"one step emitted {len(step.new_token_ids)} tokens, more than k+1={self.spec.n}",
            )
        self.assertGreater(n_events, 0)

    async def test_toggling_across_spec_max_batch_matches_a_plain_engine_over_http(self):
        """Spec serving's core guarantee, exercised through real HTTP
        requests rather than the scheduler directly: several concurrent
        completions whose live batch crosses ``--spec-max-batch`` mid-stream
        (short ones finish and drop the batch) must read back byte-identical
        to the same prompts against a plain (no spec) engine.

        Both engines are built from **one** shared ``build_tiny_mtp_model``
        call (``from_m0_module`` is a deterministic transform of already-fixed
        weights, so calling it twice gives two numerically identical models
        with independent pools) -- building two *separately-seeded* models
        instead would not compare "plain vs spec", it would compare two
        different random models: ``with_mtp=True`` allocates extra parameter
        tensors during construction, which shifts every subsequent
        ``torch.manual_seed``-derived draw relative to a ``with_mtp=False``
        build even at the same seed.
        """
        prompts_and_lens = [("alpha request", 4), ("beta request", 4), ("gamma longer request", 20)]
        model, _cfg = build_tiny_mtp_model(seed=51)

        async def run(spec_k, spec_max_batch, model_name):
            rt = serving_rt(enable_mtp=True)
            fm = FusedQwenForCausalLM.from_m0_module(model, rt)
            buf = DeviceBuffers(
                max_batch=rt.max_num_seqs, vocab_size=fm.config.vocab_size,
                max_pages=rt.n_kv_pages, device=CPU,
            )
            decoder = GraphedDecoder(fm, buf, rt)
            spec = build_spec_decoder(fm, buf, rt, SpecConfig(k=spec_k)) if spec_k else None
            engine = QwenFastEngine(
                fm, decoder, rt, eos_token_id=self.tokenizer.eos_token_id, capture_graphs=False,
                spec=spec, spec_max_batch=spec_max_batch,
            )
            app = create_app(engine, self.tokenizer, model_name=model_name, default_max_tokens=8)
            await engine.start()
            client = httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://qwenfast.test", timeout=60.0
            )
            try:
                async def one(prompt, maxt):
                    r = await client.post(
                        "/v1/completions",
                        json={"model": model_name, "prompt": prompt, "max_tokens": maxt,
                              "temperature": 0.0, "ignore_eos": True},
                    )
                    self.assertEqual(r.status_code, 200, r.text)
                    return r.json()["choices"][0]["text"]

                texts = await asyncio.gather(*[one(p, m) for p, m in prompts_and_lens])
            finally:
                await client.aclose()
                await engine.shutdown()
            return texts, spec

        plain_texts, _no_spec = await run(spec_k=0, spec_max_batch=None, model_name="qwenfast-tiny-plain2")
        spec_texts, spec = await run(spec_k=2, spec_max_batch=2, model_name="qwenfast-tiny-spec3")

        self.assertEqual(plain_texts, spec_texts)
        self.assertGreater(spec.steps, 0, "the surviving long request never spec-decoded")


# =========================================================================== #
# 5. the serve.py memory plan
# =========================================================================== #
class TestMemoryPlan(unittest.TestCase):
    """``serve.plan_memory`` is what picks the production pool geometry, so
    its arithmetic is pinned against the model's documented geometry."""

    def setUp(self):
        from qwenfast.runtime import serve

        self.serve = serve

    def test_kv_bytes_per_token(self):
        # 17 KV layers (16 attention + 1 MTP) x 4 kv-heads x 256 head_dim x 2 (K,V)
        fp8 = self.serve.plan_memory(
            max_num_seqs=1, max_model_len=16, page_size=16, n_kv_pages=1,
            kv_cache_dtype="fp8", ssm_state_dtype="fp16",
        )
        bf16 = self.serve.plan_memory(
            max_num_seqs=1, max_model_len=16, page_size=16, n_kv_pages=1,
            kv_cache_dtype="bf16", ssm_state_dtype="fp16",
        )
        self.assertEqual(fp8["kv_kib_per_token"], 34.0)
        self.assertEqual(bf16["kv_kib_per_token"], 68.0)

    def test_ssm_bytes_per_slot(self):
        fp16 = self.serve.plan_memory(
            max_num_seqs=1, max_model_len=16, page_size=16, n_kv_pages=1,
            kv_cache_dtype="bf16", ssm_state_dtype="fp16",
        )
        fp32 = self.serve.plan_memory(
            max_num_seqs=1, max_model_len=16, page_size=16, n_kv_pages=1,
            kv_cache_dtype="bf16", ssm_state_dtype="fp32",
        )
        # 48 layers x 48 v-heads x 128 x 128 = 72 MiB fp16 / 144 MiB fp32,
        # plus 2.81 MiB of bf16 conv state.
        self.assertAlmostEqual(fp16["ssm_mib_per_slot"], 72.0 + 2.8125, places=3)
        self.assertAlmostEqual(fp32["ssm_mib_per_slot"], 144.0 + 2.8125, places=3)

    def test_default_serving_config_fits_an_h200(self):
        import argparse

        p = self.serve.add_runtime_args(argparse.ArgumentParser())
        args = p.parse_args([])
        rt = self.serve.runtime_config_from_args(args)
        plan = self.serve.plan_memory(
            max_num_seqs=rt.max_num_seqs, max_model_len=args.max_model_len, page_size=rt.page_size,
            n_kv_pages=rt.n_kv_pages, kv_cache_dtype=rt.kv_cache_dtype,
            ssm_state_dtype=rt.ssm_state_dtype, dtype=rt.dtype,
        )
        self.assertGreaterEqual(plan["kv_tokens_capacity"], plan["kv_tokens_needed"])
        self.assertLess(plan["total_gib"], 131.0, "must fit an H200's 141 GB = 131.3 GiB")

    def test_defaults_are_the_m1_measured_config(self):
        import argparse

        p = self.serve.add_runtime_args(argparse.ArgumentParser())
        args = p.parse_args([])
        rt = self.serve.runtime_config_from_args(args)
        self.assertEqual(rt.ssm_state_dtype, "fp16")
        self.assertEqual(rt.norm_backend, "triton")
        self.assertEqual(rt.attn_backend, "auto")  # -> flashinfer; graph capture requires it
        self.assertTrue(rt.use_cuda_graphs)
        self.assertIsNone(rt.gemm_backend, "pinning one GEMM backend costs 5x at M=256")
        # A single sequence must be able to hold the whole --max-model-len, or
        # `_ensure_capacity_with_preemption` aborts it at the last token.
        self.assertGreaterEqual(rt.max_pages_per_seq * rt.page_size, args.max_model_len)

    def test_spec_k_zero_is_the_pre_m4_plan_byte_for_byte(self):
        """``--spec-k 0`` (the default) must not add the spec term at
        all -- the plan for a spec-capable server with spec off must equal
        the plan with no spec machinery in the CLI surface whatsoever."""
        import argparse

        p = self.serve.build_arg_parser()
        args = p.parse_args(["--model", "/nonexistent"])
        self.assertEqual(args.spec_k, 0)
        rt = self.serve.runtime_config_from_args(args)
        self.assertFalse(rt.enable_mtp, "--spec-k 0 must not force --enable-mtp on")
        plan = self.serve.plan_memory_for(args, rt)
        self.assertEqual(plan["spec_gib"], 0.0)

    def test_spec_k_forces_enable_mtp_and_adds_the_spec_window_term(self):
        """The spec window cache term must be in ``serve.plan_memory`` when
        ``--spec-k`` is set, without the caller having to separately pass
        ``--enable-mtp`` (``SpecDecoder`` needs the MTP head loaded or it
        raises)."""
        p = self.serve.build_arg_parser()
        args = p.parse_args(["--model", "/nonexistent", "--spec-k", "3", "--spec-max-batch", "16"])
        rt = self.serve.runtime_config_from_args(args)
        self.assertTrue(rt.enable_mtp, "--spec-k > 0 must force --enable-mtp on")
        plan = self.serve.plan_memory_for(args, rt)
        self.assertGreater(plan["spec_gib"], 0.0)
        # n = k + 1 = 4, so the term must be > half of k=1's (n=2's) term at
        # the same spec_max_batch -- the same relationship
        # TestBenchSpecMemoryPlan.test_spec_window_cache_is_planned pins for
        # bench_spec's own plan.
        args_k1 = p.parse_args(["--model", "/nonexistent", "--spec-k", "1", "--spec-max-batch", "16"])
        rt_k1 = self.serve.runtime_config_from_args(args_k1)
        plan_k1 = self.serve.plan_memory_for(args_k1, rt_k1)
        self.assertGreater(plan_k1["spec_gib"], 0.0)
        self.assertGreater(plan["spec_gib"], plan_k1["spec_gib"] * 1.5)

    def test_spec_max_batch_caps_the_server_spec_buckets_and_the_plan(self):
        """``spec_buckets_for`` must cap what the server's ``SpecDecoder``
        captures at ``--spec-max-batch``, not the server's full (up to 512)
        bucket table -- and the plan must be sized off that capped set, not
        off the server's own ``max_batch``, or a low ``--spec-max-batch``
        would not save the memory it is supposed to."""
        p = self.serve.build_arg_parser()
        args = p.parse_args(["--model", "/nonexistent", "--spec-k", "3", "--spec-max-batch", "8"])
        rt = self.serve.runtime_config_from_args(args)
        buckets = self.serve.spec_buckets_for(rt, args.spec_max_batch)
        self.assertEqual(buckets[-1], 8)
        self.assertTrue(all(b <= 8 for b in buckets))
        self.assertLess(len(buckets), len(rt.buckets_for()))

        plan_narrow = self.serve.plan_memory_for(args, rt)
        args_wide = p.parse_args(["--model", "/nonexistent", "--spec-k", "3", "--spec-max-batch", "256"])
        plan_wide = self.serve.plan_memory_for(args_wide, rt)
        self.assertGreater(plan_wide["spec_gib"], plan_narrow["spec_gib"])

    def test_spec_buckets_for_always_covers_a_non_bucket_cap(self):
        p = self.serve.build_arg_parser()
        args = p.parse_args(["--model", "/nonexistent"])
        rt = self.serve.runtime_config_from_args(args)
        got = self.serve.spec_buckets_for(rt, 20)  # 20 is not one of RuntimeConfig's buckets
        self.assertEqual(got[-1], 20)
        self.assertTrue(all(b <= 20 for b in got))

    def test_serving_defaults_bound_the_gemm_repack_caches(self):
        """The repack-cache OOM in one assertion.

        ``--gemm-weight-cache multi`` lets two GEMM backends each memoise a
        full repacked copy of the FP8 linears. On the 27B that is +45.7 GiB,
        enough to turn a plan of 95.5 GiB into a process holding 139.8.
        Serving must default to one cache, and one
        cache must be what makes the config fit."""
        import argparse

        p = self.serve.add_runtime_args(argparse.ArgumentParser())
        args = p.parse_args([])
        rt = self.serve.runtime_config_from_args(args)
        self.assertEqual(rt.gemm_weight_cache, "single")

        def plan(cache):
            return self.serve.plan_memory(
                max_num_seqs=rt.max_num_seqs, max_model_len=args.max_model_len,
                page_size=rt.page_size, n_kv_pages=rt.n_kv_pages,
                max_pages_per_seq=rt.max_pages_per_seq,
                kv_cache_dtype=rt.kv_cache_dtype, ssm_state_dtype=rt.ssm_state_dtype,
                dtype=rt.dtype, gemm_weight_cache=cache,
                max_num_batched_tokens=rt.max_num_batched_tokens,
                n_graph_buckets=len(rt.buckets_for()), max_batch=rt.buckets_for()[-1],
            )

        none_, single, multi = plan("none"), plan("single"), plan("multi")
        self.assertAlmostEqual(none_["repack_gib"], 0.0)
        # each cache is ~1x the fp8 linears (marlin's qweight is byte-for-byte
        # the fp8 weight it replaces, plus fp16 group-128 scales)
        self.assertAlmostEqual(single["repack_gib"], none_["linear_fp8_gib"], delta=1.0)
        self.assertAlmostEqual(multi["repack_gib"], 2 * single["repack_gib"], places=4)
        # "multi" must be visibly over budget: it measured 139.78 GiB in use
        # on a 139.8 GiB device.
        self.assertGreater(multi["steady_gib"], 135.0)
        self.assertLess(multi["steady_gib"], 145.0)
        # ... and the shipped default must fit an H200 with real margin.
        self.assertLess(single["total_gib"], 125.0)

    def test_plan_models_the_kv_pool_retype(self):
        """``_make_kv_pool`` re-types a bf16 pool to the activation dtype when
        they differ. A plan that read
        ``kv_cache_dtype`` alone would be 2x light on ``--dtype fp32``."""
        kw = dict(max_num_seqs=4, max_model_len=64, page_size=4, n_kv_pages=64,
                  kv_cache_dtype="bf16", ssm_state_dtype="fp16")
        self.assertAlmostEqual(
            self.serve.plan_memory(dtype="fp32", **kw)["kv_kib_per_token"],
            2 * self.serve.plan_memory(dtype="bf16", **kw)["kv_kib_per_token"],
        )


# =========================================================================== #
# 6. the memory plan against a real (tiny, CPU) allocation
# =========================================================================== #
class TestMemoryPlanMatchesAllocation(unittest.TestCase):
    """``plan_memory`` arithmetic vs. tensors that actually exist.

    ``TestMemoryPlan`` above pins the plan against *stated numbers*, which
    cannot catch a plan that is wrong in the same way as those numbers:
    every assertion would agree with the same mistaken model of what gets
    allocated. This class asserts against ``FusedQwenForCausalLM.pool_nbytes()``
    -- ``numel * element_size`` of the live pools of a real (tiny, CPU) model
    -- so a dtype or shape error in the plan cannot cancel out on both sides.
    """

    GIB = 1024.0 ** 3

    def _arch_for(self, cfg) -> dict:
        n_attn = sum(1 for t in cfg.layer_types if t == "full_attention")
        n_gdn = sum(1 for t in cfg.layer_types if t == "linear_attention")
        return {
            "n_attn_layers": n_attn,
            "n_gdn_layers": n_gdn,
            "num_kv_heads": cfg.num_key_value_heads,
            "head_dim": cfg.head_dim,
            "gdn_v_heads": cfg.linear_num_value_heads,
            "gdn_head_k": cfg.linear_key_head_dim,
            "gdn_head_v": cfg.linear_value_head_dim,
            "conv_dim": cfg.conv_dim,
            "conv_width": cfg.linear_conv_kernel_dim,
            "hidden_size": cfg.hidden_size,
            "intermediate_size": cfg.intermediate_size,
            "vocab_size": cfg.vocab_size,
            "num_attention_heads": cfg.num_attention_heads,
            "gdn_z_dim": cfg.linear_num_value_heads * cfg.linear_value_head_dim,
        }

    def _plan_and_pools(self, **rt_overrides):
        from qwenfast.runtime import serve

        model, cfg = build_tiny_model(seed=13)
        rt = serving_rt(**rt_overrides)
        fm = FusedQwenForCausalLM.from_m0_module(model, rt)
        pools = fm.pool_nbytes()
        plan = serve.plan_memory(
            max_num_seqs=rt.max_num_seqs,
            max_model_len=rt.max_model_len,
            page_size=rt.page_size,
            n_kv_pages=rt.n_kv_pages,
            max_pages_per_seq=rt.max_pages_per_seq,
            kv_cache_dtype=rt.kv_cache_dtype,
            ssm_state_dtype=rt.ssm_state_dtype,
            dtype=rt.dtype,
            arch=self._arch_for(fm.config),
            # the tiny model is built `with_mtp=False`, so its KV pool has no
            # 17th layer -- exactly the `+1` the plan must not assume
            has_mtp=fm.fused.mtp is not None,
            gemm_weight_cache="none",
            use_cuda_graphs=False,
            weight_bytes=float(pools["weights"]),
            linear_fp8_bytes=0.0,
        )
        return plan, pools, fm

    def _assert_close(self, planned_gib: float, actual_bytes: int, label: str, rel=0.01):
        planned = planned_gib * self.GIB
        self.assertAlmostEqual(
            planned, actual_bytes, delta=max(actual_bytes * rel, 4096),
            msg=f"{label}: planned {planned:.0f} B vs allocated {actual_bytes} B",
        )

    def test_kv_ssm_conv_terms_match_the_allocated_pools(self):
        plan, pools, _fm = self._plan_and_pools()
        self._assert_close(plan["kv_gib"], pools["kv_pool"], "kv_pool")
        self._assert_close(plan["ssm_gib"], pools["ssm_state"], "ssm_state")
        self._assert_close(plan["conv_gib"], pools["conv_state"], "conv_state")

    def test_terms_track_the_knobs_that_drive_them(self):
        """Doubling a knob must double the term it drives -- against measured
        bytes, so a plan that ignored a knob entirely would still fail here."""
        base, base_pools, _ = self._plan_and_pools()
        wide, wide_pools, _ = self._plan_and_pools(max_num_seqs=16)
        deep, deep_pools, _ = self._plan_and_pools(n_kv_pages=1024)
        self._assert_close(wide["ssm_gib"], wide_pools["ssm_state"], "ssm(16 seqs)")
        self._assert_close(deep["kv_gib"], deep_pools["kv_pool"], "kv(1024 pages)")
        self.assertGreater(wide_pools["ssm_state"], base_pools["ssm_state"])
        self.assertGreater(deep_pools["kv_pool"], base_pools["kv_pool"])

    def test_fp16_state_halves_the_measured_pool(self):
        p32, pools32, _ = self._plan_and_pools(ssm_state_dtype="fp32")
        p16, pools16, _ = self._plan_and_pools(ssm_state_dtype="fp16")
        self.assertEqual(pools32["ssm_state"], 2 * pools16["ssm_state"])
        self._assert_close(p16["ssm_gib"], pools16["ssm_state"], "ssm fp16")
        self._assert_close(p32["ssm_gib"], pools32["ssm_state"], "ssm fp32")

    def test_steady_state_sums_the_pools_it_claims_to(self):
        """The whole-plan check: strip the terms a CPU model has no analogue
        for (CUDA context, graph pool, FlashInfer workspaces, the fp32 logits
        buffer) and what is left must be the measured pool bytes within 5%."""
        plan, pools, _fm = self._plan_and_pools()
        modelled = (
            plan["weights_gib"] + plan["repack_gib"] + plan["kv_gib"]
            + plan["ssm_gib"] + plan["conv_gib"]
        ) * self.GIB
        actual = (
            pools["weights"] + pools["repack_caches"] + pools["kv_pool"]
            + pools["ssm_state"] + pools["conv_state"]
        )
        self.assertLessEqual(
            abs(modelled - actual) / actual, 0.05,
            f"plan {modelled:.0f} B vs allocated {actual} B",
        )

    def test_repack_caches_are_measured_not_assumed(self):
        """``repack_cache_nbytes()`` is the term the old plan had no name for.
        It must be readable off the model and be zero before any GEMM runs."""
        _plan, pools, fm = self._plan_and_pools()
        self.assertEqual(pools["repack_caches"], 0)
        self.assertEqual(fm.repack_cache_nbytes(), 0)


# =========================================================================== #
# 7. engine liveness -- a dead engine must not look healthy
# =========================================================================== #
@unittest.skipUnless(HAS_HTTPX, "httpx is required for the ASGI end-to-end tests")
class TestEngineDeathIsVisible(unittest.IsolatedAsyncioTestCase):
    """A dead engine thread must be visible to clients.

    If the engine thread dies (e.g. of a CUDA OOM) while ``/health`` keeps
    answering ``{"status": "ok"}`` and the HTTP server keeps accepting
    requests, clients such as ``bench_serve.py`` sit on streams that will
    never produce a token. These tests kill the engine loop -- an exception
    out of ``scheduler.step()`` -- and pin the three things that must then
    happen.
    """

    async def asyncSetUp(self):
        self.engine, self.model, self.rt = build_engine(seed=17)
        self.tokenizer = TinyTokenizer()
        self.app = create_app(self.engine, self.tokenizer, model_name="qwen3.8-27b")
        await self.engine.start()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://qwenfast.test", timeout=30.0
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.engine.shutdown()

    async def _kill_engine(self) -> None:
        """Make the next scheduler step raise, then wait for the loop to die --
        exactly the shape of the real failure (an OOM inside `step()`)."""
        def boom():
            raise torch.OutOfMemoryError("CUDA out of memory (simulated)")

        self.engine.scheduler.step = boom
        self.engine.scheduler.has_work = lambda: True
        for _ in range(200):
            if self.engine.health() is not None:
                return
            await asyncio.sleep(0.01)
        self.fail("engine thread did not report its death")

    async def test_health_is_ok_while_the_engine_lives(self):
        r = await self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")
        self.assertIsNone(self.engine.health())

    async def test_health_turns_503_when_the_engine_thread_dies(self):
        await self._kill_engine()
        r = await self.client.get("/health")
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json()["status"], "unhealthy")
        self.assertIn("out of memory", r.json()["detail"].lower())

    async def test_completions_5xx_instead_of_hanging(self):
        await self._kill_engine()
        r = await self.client.post(
            "/v1/completions",
            json={"model": "qwen3.8-27b", "prompt": "hello", "max_tokens": 8},
        )
        self.assertEqual(r.status_code, 503)

    async def test_chat_completions_5xx_instead_of_hanging(self):
        await self._kill_engine()
        r = await self.client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3.8-27b",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 8,
            },
        )
        self.assertEqual(r.status_code, 503)

    async def test_die_releases_registered_waiters(self):
        """The mechanism, deterministically: every registered output queue gets
        the shutdown sentinel, so nobody is left on ``await out_q.get()``."""
        q: "asyncio.Queue" = asyncio.Queue()
        self.engine._out_queues["stuck"] = q
        self.engine._die(RuntimeError("simulated"))
        item = await asyncio.wait_for(q.get(), timeout=5.0)
        self.assertIsNone(item)
        self.engine._out_queues.pop("stuck", None)

    async def test_stream_terminates_after_engine_death(self):
        """End to end: a stream in flight when the engine dies must *end*
        (raise or stop) rather than hang until the client's own timeout."""
        # 120, not 4096: 3 + 4096 is past the tiny engine's 128-token context,
        # which `QwenFastEngine._add_request` rejects up front (see the
        # context-cap tests below) -- the stream would end before it ever started, and this
        # test needs one that is *still running* when the engine dies.
        gen = self.engine.add_request(
            "doomed", [3, 4, 5], SamplingParams(max_tokens=120, temperature=0.0, ignore_eos=True)
        )

        async def drain():
            try:
                async for _ in gen:
                    pass
            except Exception:
                return "raised"
            return "ended"

        task = asyncio.ensure_future(drain())
        await asyncio.sleep(0.02)
        await self._kill_engine()
        outcome = await asyncio.wait_for(task, timeout=10.0)
        self.assertIn(outcome, ("raised", "ended"))


# =========================================================================== #
# 8. the context cap
# =========================================================================== #
class TestRotaryTableIsUnguarded(unittest.TestCase):
    """*Why* every test below this one exists.

    ``RotaryTable.lookup`` is ``self.cos[positions.long()]`` -- a plain gather
    with no clamp, no wrap and no bounds check. On CPU an out-of-range position
    is an ``IndexError``; on CUDA it is
    ``IndexKernel.cu:111 ... index out of bounds``, a **device-side assert**,
    which poisons the CUDA context: every subsequent kernel launch on that
    process fails, so one bad request does not fail one request, it kills the
    engine. This test pins that property at the exact production geometry so
    nobody later assumes the gather is defensive.
    """

    def test_lookup_at_max_positions_is_out_of_bounds(self):
        from qwenfast.attn.rope import RotaryTable

        table = RotaryTable(max_positions=2560, rotary_dim=64, dtype=torch.float32, device=CPU)
        last = table.lookup(torch.tensor([2559], dtype=torch.int32))
        self.assertEqual(last[0].shape, (1, 64))
        with self.assertRaises(IndexError):
            table.lookup(torch.tensor([2560], dtype=torch.int32))


class TestContextCap(unittest.TestCase):
    """The scheduler must never hand the model a position it cannot index.

    Geometry of the tiny fixture, chosen to mirror the production failure:
    ``rt.max_model_len=256`` but ``cfg.max_position_embeddings=128``, so the
    rotary table has 128 rows while the page table holds ``64 x 4 = 256``
    tokens. The *binding* limit is therefore 128 and it is **not**
    ``max_model_len`` -- the same shape as a production config where 2560
    rotary rows bound a 165-page (2640-token) page table.
    """

    def test_max_context_len_is_the_min_of_rotary_and_pages(self):
        sched, fm, _dec, rt = build_scheduler()
        self.assertEqual(fm.rotary.max_positions, 128)  # min(max_model_len, max_position_embeddings)
        self.assertEqual(rt.max_pages_per_seq * rt.page_size, 256)
        self.assertEqual(fm.max_context_len, 128)
        self.assertEqual(sched.max_context_len, 128)
        self.assertLess(
            fm.max_context_len, rt.max_model_len,
            "the fixture must keep the rotary table as the binding limit -- that is the bug's shape",
        )

    def test_max_context_len_can_be_bound_by_the_page_table_instead(self):
        # The other side of the `min`: 8 pages x 4 tokens = 32 < 128 rotary rows.
        _sched, fm, _dec, _rt = build_scheduler(
            page_size=4, n_kv_pages=64, max_pages_per_seq=8, max_num_seqs=4
        )
        self.assertEqual(fm.max_context_len, 32)

    def test_context_length_error_boundaries(self):
        sched, _fm, _dec, _rt = build_scheduler()
        cap = sched.max_context_len
        self.assertIsNone(sched.context_length_error(cap - 10, 10))  # exactly the cap: fits
        self.assertIsNone(sched.context_length_error(cap, 0))  # prefill only, no decode
        msg = sched.context_length_error(cap - 10, 11)  # one token over
        self.assertIsNotNone(msg)
        self.assertIn(str(cap), msg)
        self.assertIn(str(cap + 1), msg)
        self.assertIsNotNone(sched.context_length_error(cap + 1, 0))  # prompt alone too long

    def test_decode_stops_at_the_cap_instead_of_indexing_out_of_range(self):
        """The reproduction, in miniature.

        At production scale: prompt 2158 + ``max_tokens`` 500 against a
        2560-row rotary table would be accepted, and 402 decode steps later
        position 2560 would be gathered and kill the engine. Here: prompt 100
        + ``max_tokens`` 200 against a 128-row table. Without the cap this
        raises ``IndexError`` out of ``RotaryTable.lookup`` at step 28 (the
        CPU spelling of the CUDA assert); with it, the request stops cleanly
        at the cap.
        """
        sched, fm, _dec, _rt = build_scheduler()
        cap = sched.max_context_len
        prompt = [3 + (i % 20) for i in range(100)]
        req = Request("long", prompt, gen_params(max_tokens=200))

        seen_positions = []
        real_lookup = fm.rotary.lookup

        def spy(positions):
            seen_positions.append(int(positions.max()))
            return real_lookup(positions)

        fm.rotary.lookup = spy  # type: ignore[assignment]
        sched.add_request(req)
        steps = 0
        while sched.has_work() and steps < 500:
            sched.step()  # must not raise
            steps += 1

        self.assertTrue(req.is_finished)
        self.assertEqual(req.finish_reason, "length")
        self.assertLess(
            max(seen_positions), cap,
            "a position >= max_context_len was gathered from the rotary table -- "
            "on CUDA that is a device-side assert, not an exception",
        )
        # It stopped *because* of the cap, not because max_tokens ran out:
        # 128 committed tokens, well short of the 200 that were asked for.
        self.assertEqual(req.num_computed_tokens, cap)
        # 29, not 28: the last token is *sampled* from the step that filled the
        # context and emitted, but never written back into KV -- so it costs no
        # position. (`num_computed_tokens` is the committed length; the count of
        # emitted tokens is always one ahead of the decode steps that ran.)
        self.assertEqual(len(req.output_token_ids), cap - len(prompt) + 1)

    def test_prompt_exactly_at_the_cap_generates_one_token_and_stops(self):
        sched, _fm, _dec, _rt = build_scheduler()
        cap = sched.max_context_len
        req = Request("edge", [3 + (i % 20) for i in range(cap)], gen_params(max_tokens=50))
        sched.add_request(req)
        steps = 0
        while sched.has_work() and steps < 400:
            sched.step()
            steps += 1
        self.assertTrue(req.is_finished)
        self.assertEqual(req.finish_reason, "length")
        # The token sampled off the last prefill chunk is emitted (it is a
        # legitimate continuation); it simply can never be fed back in.
        self.assertEqual(len(req.output_token_ids), 1)

    def test_prompt_over_the_cap_fails_the_request_not_the_engine(self):
        """Defence in depth: the HTTP layer 400s this, but a direct scheduler
        caller must still get a failed *request* rather than a prefill that
        indexes the rotary table at ``len(prompt) - 1 >= max_positions``."""
        sched, _fm, _dec, _rt = build_scheduler()
        cap = sched.max_context_len
        over = Request("over", [3 + (i % 20) for i in range(cap + 5)], gen_params(max_tokens=4))
        fits = Request("fits", [3, 4, 5, 6], gen_params(max_tokens=4))
        sched.add_request(over)
        sched.add_request(fits)
        steps = 0
        while sched.has_work() and steps < 200:
            sched.step()  # must not raise
            steps += 1
        self.assertTrue(over.is_finished)
        self.assertEqual(over.finish_reason, "length")
        self.assertEqual(over.output_token_ids, [])
        self.assertIsNone(over.slot, "an over-long request must not leak a slot")
        # ... and it must not have wedged the queue behind it.
        self.assertTrue(fits.is_finished)
        self.assertEqual(len(fits.output_token_ids), 4)
        self.assertEqual(sched.slots.num_free, sched.model.n_slots)

    def test_cap_is_respected_when_the_page_table_is_the_binding_limit(self):
        """Same guarantee with a small page count and positions crossing many
        page boundaries: 4-token pages, 8 pages per sequence => a 32-token
        context, so a 20-token prompt at ``max_tokens=50`` crosses 8 page
        boundaries and then stops at the cap -- and, importantly, stops with
        ``"length"`` rather than the ``"abort"`` that
        ``_ensure_capacity_with_preemption`` used to hand out at the last page.
        """
        sched, _fm, _dec, _rt = build_scheduler(
            page_size=4, n_kv_pages=64, max_pages_per_seq=8, max_num_seqs=4, max_num_batched_tokens=8
        )
        self.assertEqual(sched.max_context_len, 32)
        req = Request("pages", [3 + (i % 20) for i in range(20)], gen_params(max_tokens=50))
        sched.add_request(req)
        steps = 0
        while sched.has_work() and steps < 300:
            sched.step()
            steps += 1
        self.assertTrue(req.is_finished)
        self.assertEqual(req.finish_reason, "length")
        self.assertEqual(req.num_computed_tokens, 32)
        self.assertEqual(len(req.output_token_ids), 32 - 20 + 1)


@unittest.skipUnless(HAS_HTTPX, "httpx is required for the ASGI end-to-end tests")
class TestOverLongRequestIsA400(unittest.IsolatedAsyncioTestCase):
    """What the client sees. vLLM returns 400 for prompt+completion past
    ``max_model_len``. Without the context cap a server would accept the
    request, stream tokens until the rotary table ran out and then 503 --
    for that request and every request after it, because the engine thread
    would be gone."""

    async def asyncSetUp(self):
        self.engine, self.model, self.rt = build_engine()
        self.tokenizer = TinyTokenizer()
        self.app = create_app(
            self.engine, self.tokenizer, model_name="qwenfast-tiny", default_max_tokens=8, detok_workers=2
        )
        await self.engine.start()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://qwenfast.test", timeout=60.0
        )
        self.cap = self.engine.max_context_len

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.engine.shutdown()

    def _prompt(self, n_tokens: int) -> str:
        return "a" * n_tokens  # TinyTokenizer is character-level

    async def test_completions_over_length_clamps_and_the_engine_survives(self):
        """The invariant is about what reaches the engine: a request whose
        prompt+completion runs off the end of the context must never decode
        until it asserts inside ``RotaryTable.lookup`` -- a device-side
        assert, which kills the CUDA context and with it every later request.

        The client contract is to clamp rather than refuse. Refusing the
        whole request with a 400 is one way to prevent it; clamping
        ``max_tokens`` to the room the prompt leaves is the better one,
        because ``max_tokens`` is a ceiling in the OpenAI API rather than a
        reservation, and a short prompt with ``max_tokens`` set to the whole
        context is a common real-world request.

        So: served, clamped to the 10 tokens that fit, ``finish_reason:
        "length"``, and the engine is still alive afterwards.
        """
        r = await self.client.post(
            "/v1/completions",
            json={"model": "qwenfast-tiny", "prompt": self._prompt(self.cap - 10),
                  "max_tokens": 500, "temperature": 0.0, "ignore_eos": True},
        )
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["usage"]["completion_tokens"], 10)
        self.assertEqual(r.json()["choices"][0]["finish_reason"], "length")
        self.assertEqual(r.headers["X-Qwenfast-Max-Tokens"], "10")
        # prompt + completion landed exactly on the context, never past it.
        usage = r.json()["usage"]
        self.assertLessEqual(usage["prompt_tokens"] + usage["completion_tokens"], self.cap)

        # The whole point of the context cap: the next request still works.
        self.assertEqual((await self.client.get("/health")).status_code, 200)
        ok = await self.client.post(
            "/v1/completions",
            json={"model": "qwenfast-tiny", "prompt": "hello", "max_tokens": 4,
                  "temperature": 0.0, "ignore_eos": True},
        )
        self.assertEqual(ok.status_code, 200, ok.text)

    async def test_a_prompt_longer_than_the_context_is_still_a_400(self):
        """Clamping only helps while there is something to clamp to.

        A prompt that does not fit *by itself* leaves no room for any answer,
        so there is nothing to serve and it is refused -- in OpenAI's error
        envelope, with the code an SDK branches on.
        """
        r = await self.client.post(
            "/v1/completions",
            json={"model": "qwenfast-tiny", "prompt": self._prompt(self.cap + 50),
                  "max_tokens": 4, "temperature": 0.0, "ignore_eos": True},
        )
        self.assertEqual(r.status_code, 400, r.text)
        err = r.json()["error"]
        self.assertEqual(err["code"], "context_length_exceeded")
        self.assertIn(str(self.cap), err["message"])
        self.assertEqual((await self.client.get("/health")).status_code, 200)

    async def test_streaming_over_length_is_400_before_any_sse_frame(self):
        """It has to be rejected *before* the response starts: once a
        ``text/event-stream`` body is open the status is already 200 and the
        client (bench_serve) counts it as a successful request with zero
        tokens."""
        r = await self.client.post(
            "/v1/completions",
            json={"model": "qwenfast-tiny", "prompt": self._prompt(self.cap + 50),
                  "max_tokens": 500, "temperature": 0.0, "stream": True},
        )
        self.assertEqual(r.status_code, 400, r.text)
        self.assertNotIn("text/event-stream", r.headers.get("content-type", ""))
        self.assertEqual(r.json()["error"]["code"], "context_length_exceeded")

    async def test_chat_completions_over_length_is_400(self):
        r = await self.client.post(
            "/v1/chat/completions",
            json={"model": "qwenfast-tiny",
                  "messages": [{"role": "user", "content": self._prompt(self.cap + 50)}],
                  "max_tokens": 500, "temperature": 0.0},
        )
        self.assertEqual(r.status_code, 400, r.text)
        self.assertIn("maximum context length", r.json()["error"]["message"])
        self.assertEqual(r.json()["error"]["param"], "messages")

    async def test_chat_completions_over_length_clamps_when_the_prompt_fits(self):
        r = await self.client.post(
            "/v1/chat/completions",
            json={"model": "qwenfast-tiny",
                  "messages": [{"role": "user", "content": self._prompt(self.cap - 200)}],
                  "max_tokens": 4096, "temperature": 0.0},
        )
        self.assertEqual(r.status_code, 200, r.text)
        usage = r.json()["usage"]
        self.assertLessEqual(usage["prompt_tokens"] + usage["completion_tokens"], self.cap)

    async def test_a_request_that_exactly_fills_the_context_is_served(self):
        """The boundary must be inclusive -- rejecting a request that fits
        would silently shrink the usable context by one token per release."""
        n = self.cap - 4
        r = await self.client.post(
            "/v1/completions",
            json={"model": "qwenfast-tiny", "prompt": self._prompt(n), "max_tokens": 4,
                  "temperature": 0.0, "ignore_eos": True},
        )
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["usage"]["completion_tokens"], 4)
        self.assertEqual(r.json()["choices"][0]["finish_reason"], "length")


class TestServingDefaultsCoverTheSweep(unittest.TestCase):
    """The config half of the context-cap guarantee.

    ``--max-model-len 2560`` looks like "2000 in + 500 out + 60 slack", on
    the assumption that ``bench_serve.py --input-len 2000`` sends 2000-token
    prompts. It does not: it decodes 2000 random ids to text and the server
    re-encodes that text, which round-trips to **2090-2195** tokens on
    Qwen3.8's 248k vocab (all 928 seed-0 prompts measured). 2195 + 500 =
    2695 > 2560, so the sweep cannot complete at that setting.
    """

    #: Measured, not assumed. Update only alongside a fresh measurement.
    BENCH_MAX_PROMPT_TOKENS = 2195
    BENCH_OUTPUT_LEN = 500

    def test_default_max_model_len_covers_the_bench_workload(self):
        from qwenfast.runtime import serve

        need = self.BENCH_MAX_PROMPT_TOKENS + self.BENCH_OUTPUT_LEN
        self.assertGreaterEqual(
            serve.DEFAULT_MAX_MODEL_LEN, need,
            f"the serving sweep sends up to {need} tokens per request; a smaller "
            f"--max-model-len 400s (or, pre-fix, kills) its very first warmup request",
        )

    def test_default_page_geometry_can_hold_a_full_context_sequence(self):
        import argparse

        from qwenfast.runtime import serve

        args = serve.add_runtime_args(argparse.ArgumentParser()).parse_args([])
        rt = serve.runtime_config_from_args(args)
        # Both bounds of `max_context_len` must clear the workload, or the
        # binding one silently becomes something other than --max-model-len.
        self.assertGreaterEqual(rt.max_pages_per_seq * rt.page_size, args.max_model_len)
        self.assertGreaterEqual(
            args.max_model_len, self.BENCH_MAX_PROMPT_TOKENS + self.BENCH_OUTPUT_LEN
        )

    def test_default_config_still_fits_the_budget(self):
        import argparse

        from qwenfast.runtime import serve

        args = serve.add_runtime_args(argparse.ArgumentParser()).parse_args([])
        rt = serve.runtime_config_from_args(args)
        plan = serve.plan_memory(
            max_num_seqs=rt.max_num_seqs, max_model_len=args.max_model_len,
            page_size=rt.page_size, n_kv_pages=rt.n_kv_pages,
            max_pages_per_seq=rt.max_pages_per_seq,
            kv_cache_dtype=rt.kv_cache_dtype, ssm_state_dtype=rt.ssm_state_dtype,
            dtype=rt.dtype, gemm_weight_cache=rt.gemm_weight_cache,
            max_num_batched_tokens=rt.max_num_batched_tokens,
            n_graph_buckets=len(rt.buckets_for()), max_batch=rt.buckets_for()[-1],
        )
        # The longer context costs ~3.2 GiB of extra KV pool; the point of this
        # assertion is that the raise was priced, not waved through. 130.9 GiB
        # is the real ceiling (--gpu-memory-utilization 0.94 of an H200).
        self.assertLess(plan["total_gib"], 126.0, "must stay inside the 130.9 GiB serving budget")
        self.assertGreaterEqual(plan["kv_tokens_capacity"], plan["kv_tokens_needed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
