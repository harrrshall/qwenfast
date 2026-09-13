"""Shared fixtures for `test_server.py`.

Tokenizer resolution order:

1. `QWENFAST_TOKENIZER_DIR` env var, if set (a local directory with just the tokenizer files).
2. `Qwen/Qwen3.8-27B` from the Hugging Face Hub (or its cached snapshot).
3. `FakeTokenizer` (`../fake_tokenizer.py`), fully offline, no `transformers`/network needed.
"""

from __future__ import annotations

import os

import pytest

def _load_real_tokenizer_or_none():
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None

    candidates = []
    env_dir = os.environ.get("QWENFAST_TOKENIZER_DIR")
    if env_dir:
        candidates.append(env_dir)
    candidates.append("Qwen/Qwen3.8-27B")

    for candidate in candidates:
        try:
            return AutoTokenizer.from_pretrained(candidate, trust_remote_code=True)
        except Exception:
            continue
    return None


@pytest.fixture(scope="session")
def tokenizer():
    tok = _load_real_tokenizer_or_none()
    if tok is not None:
        return tok
    from qwenfast.server.fake_tokenizer import FakeTokenizer

    return FakeTokenizer.shared()


@pytest.fixture(scope="session")
def using_real_tokenizer(tokenizer) -> bool:
    return type(tokenizer).__name__ != "FakeTokenizer"
