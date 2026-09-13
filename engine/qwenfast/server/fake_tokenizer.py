"""A tiny, dependency-free, offline fallback tokenizer.

Used only when `transformers.AutoTokenizer` / the downloaded Qwen3.8-27B tokenizer files are not
available (e.g. running the test-suite with no network and no cached snapshot). It implements just
enough of the HF tokenizer surface (`encode`, `decode`, `convert_tokens_to_ids`,
`apply_chat_template`, `eos_token_id`, `pad_token_id`, `vocab_size`) for `mock_engine.py`,
`tokenization.py`, and `app.py` to run unmodified against it.

It is **not** a linguistic tokenizer — it is an open-vocabulary, whitespace-preserving splitter
that is lossless (`decode(encode(text)) == text`) so the server's incremental-detokenization and
`<think>`/`<tool_call>` streaming parsers can be exercised end-to-end without the real vocab.
"""

from __future__ import annotations

import re
import threading
from typing import Optional

_SPECIALS: list[tuple[str, int]] = [
    ("<|endoftext|>", 0),
    ("<|im_start|>", 1),
    ("<|im_end|>", 2),
    ("<think>", 3),
    ("</think>", 4),
    ("<tool_call>", 5),
    ("</tool_call>", 6),
]
_SPECIAL_TEXT_TO_ID = dict(_SPECIALS)
_SPECIAL_ID_TO_TEXT = {v: k for k, v in _SPECIALS}
_FIRST_DYNAMIC_ID = 1000

# Each "piece" is a run of non-whitespace plus any whitespace immediately following it (or a bare
# whitespace run at the very start of a segment) -- this keeps concatenation lossless without
# needing a separate space-joining convention.
_WORD_SPLIT_RE = re.compile(r"\S+\s*|\s+")
_SPECIAL_SPLIT_RE = re.compile("(" + "|".join(re.escape(t) for t, _ in _SPECIALS) + ")")


class FakeTokenizer:
    def __init__(self) -> None:
        self._piece_to_id: dict[str, int] = dict(_SPECIAL_TEXT_TO_ID)
        self._id_to_piece: dict[int, str] = dict(_SPECIAL_ID_TO_TEXT)
        self._next_id = _FIRST_DYNAMIC_ID
        self._lock = threading.Lock()

        self.eos_token_id = _SPECIAL_TEXT_TO_ID["<|im_end|>"]
        self.pad_token_id = _SPECIAL_TEXT_TO_ID["<|endoftext|>"]
        self.vocab_size = 50_000  # nominal only; the open vocab may grow past this harmlessly

    _shared_lock = threading.Lock()
    _shared_instance: Optional["FakeTokenizer"] = None

    @classmethod
    def shared(cls) -> "FakeTokenizer":
        with cls._shared_lock:
            if cls._shared_instance is None:
                cls._shared_instance = cls()
            return cls._shared_instance

    def _id_for_piece(self, piece: str) -> int:
        with self._lock:
            tid = self._piece_to_id.get(piece)
            if tid is None:
                tid = self._next_id
                self._next_id += 1
                self._piece_to_id[piece] = tid
                self._id_to_piece[tid] = piece
            return tid

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._id_for_piece(token)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids: list[int] = []
        for part in _SPECIAL_SPLIT_RE.split(text):
            if not part:
                continue
            if part in _SPECIAL_TEXT_TO_ID:
                ids.append(_SPECIAL_TEXT_TO_ID[part])
                continue
            for piece in _WORD_SPLIT_RE.findall(part):
                ids.append(self._id_for_piece(piece))
        return ids

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        out: list[str] = []
        with self._lock:
            for tid in ids:
                piece = self._id_to_piece.get(int(tid))
                if piece is None:
                    continue
                if int(tid) in _SPECIAL_ID_TO_TEXT and skip_special_tokens:
                    continue
                out.append(piece)
        return "".join(out)

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
        reasoning_effort: Optional[str] = None,
        tools=None,
        **_kwargs,
    ):
        parts = []
        for m in messages:
            parts.append(f"<|im_start|>{m.get('role', 'user')}\n{m.get('content', '') or ''}<|im_end|>\n")
        text = "".join(parts)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
            if enable_thinking:
                text += "<think>\n"
        if not tokenize:
            return text
        ids = self.encode(text, add_special_tokens=False)
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}
