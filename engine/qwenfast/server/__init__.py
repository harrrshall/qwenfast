"""`qwenfast.server` — the OpenAI-compatible HTTP server.

Public surface:

    from qwenfast.server import AsyncEngine, SamplingParams, StepOutput  # engine_api.py
    from qwenfast.server import MockEngine                              # mock_engine.py
    from qwenfast.server import create_app                              # app.py

`python -m qwenfast.server ...` runs `cli.py`'s entrypoint (see `cli.py` / `README.md`).
"""

from .engine_api import (
    AsyncEngine,
    EngineStats,
    HistogramSnapshot,
    RequestStats,
    SamplingParams,
    StepOutput,
)
from .mock_engine import MockEngine
from .app import create_app
from .auth import ApiKey, KeyStore, RateLimiter
from .usage import UsageRecord, UsageRecorder

__all__ = [
    "AsyncEngine",
    "EngineStats",
    "HistogramSnapshot",
    "RequestStats",
    "SamplingParams",
    "StepOutput",
    "MockEngine",
    "create_app",
    "ApiKey",
    "KeyStore",
    "RateLimiter",
    "UsageRecord",
    "UsageRecorder",
]
