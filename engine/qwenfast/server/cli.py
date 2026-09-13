"""`python -m qwenfast.server` — run the OpenAI-compatible server.

    python -m qwenfast.server --engine mock --model <tokenizer-dir-or-repo-id> --port 8000
    python -m qwenfast.server --engine qwenfast --model /path/to/Qwen3.8-27B --port 8000 --api-key secret

`--engine mock` needs only the tokenizer (weights are never touched) and is what
`tests/test_server.py`, and a local `benchmarks/bench_serve.py` / `evals/run_eval.py` smoke run,
exercise. `--engine qwenfast` runs the real GPU runtime (`qwenfast.runtime`) behind
the same `AsyncEngine` interface -- nothing in this server subpackage changed for that to happen
beyond replacing the old "not available yet" stub with the two-line construction below.

The runtime's `--` flags (`--ssm-state-dtype`, `--kv-cache-dtype`, `--max-model-len`,
`--norm-backend`, `--no-graphs`, ...) are *the same objects* as `python -m qwenfast.runtime.serve`'s:
`runtime/serve.py::add_runtime_args` installs them into this parser too, and
`runtime/serve.py::build_engine_from_args` consumes them, so the two entry points cannot drift.
Importing `runtime.serve` is deferred to inside `main()` so `--engine mock` (and the CPU test
suite) never pays for importing torch.
"""

from __future__ import annotations

import argparse
import sys

import uvicorn

from .app import create_app
from .config import add_public_api_args, describe_public_config, public_api_kwargs
from .mock_engine import MockEngine


def _load_tokenizer(model: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "transformers is required to load the tokenizer (`pip install transformers`)."
        ) from exc
    return AutoTokenizer.from_pretrained(model, trust_remote_code=True)


def build_arg_parser(*, with_runtime: bool = False) -> argparse.ArgumentParser:
    """``with_runtime=True`` also installs ``runtime/serve.py``'s flags.

    Off by default because ``add_runtime_args`` lives in a module that
    imports torch; ``main()`` turns it on only after a first
    ``parse_known_args`` pass has established that ``--engine qwenfast`` was
    actually requested, so ``--engine mock`` and the CPU test suite never
    import the runtime at all.
    """
    p = argparse.ArgumentParser(
        prog="python -m qwenfast.server",
        description="OpenAI-compatible server for qwenfast.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--engine", choices=["mock", "qwenfast"], default="mock", help="serving backend")
    p.add_argument(
        "--model",
        required=True,
        help="tokenizer repo id or local directory (e.g. a snapshot_download of just the "
        "tokenizer files); for --engine qwenfast this is also the weights directory",
    )
    p.add_argument("--served-model-name", default=None, help="name reported by /v1/models (default: --model)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--api-key", default=None, help="if set, require `Authorization: Bearer <key>` on /v1/*")
    p.add_argument("--default-max-tokens", type=int, default=512, help="max_tokens when a request omits it")
    p.add_argument("--detok-workers", type=int, default=4, help="thread-pool size for tokenizer.decode calls")
    p.add_argument("--log-level", default="info")

    add_public_api_args(p)

    mock = p.add_argument_group("--engine mock options")
    mock.add_argument("--mock-decode-tokens-per-second", type=float, default=200.0)
    mock.add_argument("--mock-prefill-tokens-per-second", type=float, default=100_000.0)
    mock.add_argument("--mock-prefill-base-delay-s", type=float, default=0.005)
    mock.add_argument("--mock-max-concurrent-requests", type=int, default=256)
    mock.add_argument("--mock-ssm-slots-total", type=int, default=512)
    mock.add_argument("--mock-kv-slots-total", type=int, default=200_000)

    if with_runtime:
        from ..runtime.serve import add_runtime_args

        add_runtime_args(p)

    return p


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    pre, _unknown = build_arg_parser().parse_known_args(argv)
    args = build_arg_parser(with_runtime=(pre.engine == "qwenfast")).parse_args(argv)

    tokenizer = _load_tokenizer(args.model)
    model_name = args.served_model_name or args.model

    if args.engine == "mock":
        engine = MockEngine(
            tokenizer=tokenizer,
            decode_tokens_per_second=args.mock_decode_tokens_per_second,
            prefill_tokens_per_second=args.mock_prefill_tokens_per_second,
            prefill_base_delay_s=args.mock_prefill_base_delay_s,
            max_concurrent_requests=args.mock_max_concurrent_requests,
            ssm_slots_total=args.mock_ssm_slots_total,
            kv_slots_total=args.mock_kv_slots_total,
        )
    else:
        from ..runtime.serve import build_engine_from_args

        engine = build_engine_from_args(args, tokenizer)

    public = public_api_kwargs(args)
    app = create_app(
        engine,
        tokenizer,
        model_name=model_name,
        api_key=args.api_key,
        default_max_tokens=args.default_max_tokens,
        detok_workers=args.detok_workers,
        **public,
    )
    print(describe_public_config(public, app.state.key_store.names()), flush=True)

    # `timeout_graceful_shutdown` is the *hard* half of the drain: `create_app`'s
    # SIGTERM handler stops accepting immediately, and this bounds how long
    # uvicorn will then wait for the streams already in flight.
    uvicorn.run(
        app, host=args.host, port=args.port, log_level=args.log_level, http="h11",
        timeout_graceful_shutdown=int(args.drain_timeout) or None,
    )


if __name__ == "__main__":
    main()
