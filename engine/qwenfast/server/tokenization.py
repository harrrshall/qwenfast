"""Incremental tokenization/detokenization and streaming-safe output parsing.

Three independent pieces, composed by `app.py` into one pipeline per request:

    raw token ids --[IncrementalDetokenizer]--> raw text deltas
                  --[ThinkTagParser]--> (reasoning_content deltas, content deltas)
                  --[ToolCallStreamParser]--> (plain content deltas, ToolCallDelta events)

All three are pure, allocation-light state machines that accept text/tokens in arbitrarily small
increments (down to one token, or even a partial UTF-8 byte sequence) and never emit output that
would need to be retracted later — which is what "streaming-safe" means here: whatever has been
handed to the client is final.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

# --------------------------------------------------------------------------
# Incremental detokenization (vLLM's prefix-offset technique)
# --------------------------------------------------------------------------


class IncrementalDetokenizer:
    """Turns a stream of new token ids into a stream of *safe* text deltas.

    A naive `tokenizer.decode([new_token_id])` per token is wrong for two reasons: (1) BPE/byte-
    level tokenizers don't have a 1:1 token->substring mapping independent of context (e.g. a
    token may only render a leading space when preceded by another token), and (2) a single token
    can be part of a multi-byte UTF-8 sequence, so decoding it alone can yield an invalid/partial
    character (surfaces as U+FFFD).

    The fix (used by vLLM, SGLang, TGI): keep a small window of already-emitted token ids, decode
    `window + [new_token]` as one string, decode just `window` as another, and the delta is
    whatever new suffix appeared -- *unless* that suffix ends in the UTF-8 replacement character,
    in which case we hold everything back and wait for the next token to complete the sequence.
    `prefix_offset`/`read_offset` then slide forward so the decode window stays small (vLLM's
    `detokenizer.py` naming, kept here for anyone cross-referencing).
    """

    def __init__(self, tokenizer, prompt_token_ids: list[int], *, skip_special_tokens: bool = True) -> None:
        self.tokenizer = tokenizer
        self.skip_special_tokens = skip_special_tokens
        self.all_ids: list[int] = list(prompt_token_ids)
        # Start the window at the generation boundary -- we don't need prompt text back out of
        # this class, only what's newly generated.
        self.prefix_offset = len(prompt_token_ids)
        self.read_offset = len(prompt_token_ids)

    def add_token(self, token_id: int) -> str:
        return self.add_tokens((token_id,))

    def add_tokens(self, token_ids: tuple[int, ...] | list[int]) -> str:
        """Feed one or more new token ids (in order). Returns the newly safe-to-emit text delta,
        which may be `""` if a multi-byte character is still incomplete."""
        if not token_ids:
            return ""
        self.all_ids.extend(token_ids)

        prefix_text = self.tokenizer.decode(
            self.all_ids[self.prefix_offset : self.read_offset],
            skip_special_tokens=self.skip_special_tokens,
        )
        new_text = self.tokenizer.decode(
            self.all_ids[self.prefix_offset :],
            skip_special_tokens=self.skip_special_tokens,
        )

        if len(new_text) > len(prefix_text) and not new_text.endswith("�"):
            delta = new_text[len(prefix_text) :]
            self.prefix_offset = self.read_offset
            self.read_offset = len(self.all_ids)
            return delta
        return ""

    def add_tokens_split(self, token_ids: tuple[int, ...] | list[int]) -> list[str]:
        """Per-token text deltas for a batch of new token ids.

        Produces exactly the same total text as :meth:`add_tokens` --
        ``"".join(add_tokens_split(ids)) == add_tokens(ids)`` -- but split so the caller can
        emit one stream chunk per *token* instead of one per engine step. Speculative
        decoding commits several tokens in one step, so a client that counts stream chunks
        to show a live tokens/sec figure would otherwise under-read by the accept length
        (2-4x) until the final `usage` arrives.

        Entries are ``""`` for tokens that don't complete a character yet; that text lands on
        the token that does complete it. Batched into a single call (rather than N
        :meth:`add_token` calls driven from the event loop) so the server still pays exactly
        one thread-pool hop per step.
        """
        return [self.add_tokens((tid,)) for tid in token_ids]


# --------------------------------------------------------------------------
# Shared "don't leak a partial tag" helper
# --------------------------------------------------------------------------


def _safe_emit_len(buffer: str, tag: str) -> int:
    """Length of `buffer`'s prefix that is guaranteed *not* to be (part of) the start of `tag`,
    i.e. how much of `buffer` can be emitted right now without risking having to retract it once
    more text arrives. Holds back the longest suffix of `buffer` that is also a proper (non-full)
    prefix of `tag`.
    """
    max_check = min(len(buffer), len(tag) - 1)
    for k in range(max_check, 0, -1):
        if buffer.endswith(tag[:k]):
            return len(buffer) - k
    return len(buffer)


# --------------------------------------------------------------------------
# Stop-string matching (streaming-safe: holds back a possible partial match at the tail)
# --------------------------------------------------------------------------


class StopStringMatcher:
    """Watches a stream of raw text deltas for the earliest occurrence of any string in `stops`.

    `feed(text)` returns `(forward, stopped)`: `forward` is the prefix of newly-arrived text that
    is safe to pass downstream (i.e. provably not part of, or before, a stop-string match), and
    `stopped` is True exactly once, on the call where a stop string is confirmed -- at which point
    `forward` excludes the stop string and everything after it, and the caller must stop
    generation. Safe against a stop string straddling two `feed()` calls, the same way the tag
    parsers above are.
    """

    def __init__(self, stops: list[str]) -> None:
        self.stops = [s for s in stops if s]
        self.buffer = ""

    def feed(self, text: str) -> tuple[str, bool]:
        if not self.stops:
            return text, False
        self.buffer += text

        earliest: Optional[int] = None
        for s in self.stops:
            idx = self.buffer.find(s)
            if idx != -1 and (earliest is None or idx < earliest):
                earliest = idx
        if earliest is not None:
            forward, self.buffer = self.buffer[:earliest], ""
            return forward, True

        hold = 0
        for s in self.stops:
            max_check = min(len(self.buffer), len(s) - 1)
            for k in range(max_check, 0, -1):
                if self.buffer.endswith(s[:k]):
                    hold = max(hold, k)
                    break
        safe_len = len(self.buffer) - hold
        forward, self.buffer = self.buffer[:safe_len], self.buffer[safe_len:]
        return forward, False


# --------------------------------------------------------------------------
# <think>...</think> streaming parser
# --------------------------------------------------------------------------


class ThinkTagParser:
    """Splits a raw text stream into `(reasoning_content, content)` deltas around `<think>` /
    `</think>`.

    Important model-specific detail (verified against the model's chat template):
    the chat template itself appends the literal `<think>\\n` to the *prompt* when
    `enable_thinking` is true, so the model's generated text normally contains only the closing
    `</think>` tag, never the opening one. This class therefore defaults to starting already
    inside a think block (`start_in_think=True`) when thinking was requested. It still recognizes
    an explicit `<think>` tag if one shows up in-stream (defensive: models occasionally echo
    instructions, and `enable_thinking=False` doesn't grammar-constrain the output), so behavior
    degrades gracefully either way.
    """

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self, *, start_in_think: bool = True) -> None:
        self._state = "in_think" if start_in_think else "content"
        self._buffer = ""
        self._newlines_to_strip = 0  # budget of leading '\n's to drop right after a close tag

    def _strip_pending(self, piece: str) -> str:
        if self._newlines_to_strip <= 0 or not piece:
            return piece
        i = 0
        while i < len(piece) and piece[i] == "\n" and self._newlines_to_strip > 0:
            i += 1
            self._newlines_to_strip -= 1
        if i < len(piece):
            self._newlines_to_strip = 0  # hit a non-newline (or ran dry) -- stop stripping
        return piece[i:]

    def feed(self, text: str) -> tuple[str, str]:
        """Returns `(reasoning_delta, content_delta)` for this increment of raw text."""
        self._buffer += text
        reasoning_out: list[str] = []
        content_out: list[str] = []
        while True:
            tag = self.CLOSE if self._state == "in_think" else self.OPEN
            idx = self._buffer.find(tag)
            if idx == -1:
                safe_len = _safe_emit_len(self._buffer, tag)
                if safe_len:
                    piece, self._buffer = self._buffer[:safe_len], self._buffer[safe_len:]
                    if self._state == "in_think":
                        reasoning_out.append(piece)
                    else:
                        content_out.append(self._strip_pending(piece))
                break
            piece, self._buffer = self._buffer[:idx], self._buffer[idx + len(tag) :]
            if self._state == "in_think":
                reasoning_out.append(piece)
                self._state = "content"
                self._newlines_to_strip = 2  # the template's own "</think>\n\n" separator
            else:
                content_out.append(self._strip_pending(piece))
                self._state = "in_think"
        return "".join(reasoning_out), "".join(content_out)

    def flush(self) -> tuple[str, str]:
        """Call once when the stream ends: anything still buffered was a tag prefix that never
        completed, so it is just literal text of whatever the current state is."""
        text, self._buffer = self._buffer, ""
        if not text:
            return "", ""
        return (text, "") if self._state == "in_think" else ("", text)


# --------------------------------------------------------------------------
# Hermes-style <tool_call>{"name": ..., "arguments": {...}}</tool_call> streaming parser
# --------------------------------------------------------------------------


@dataclass
class ToolCallDelta:
    index: int
    id: Optional[str] = None
    name: Optional[str] = None
    arguments_delta: str = ""


_NAME_RE = re.compile(r'"name"\s*:\s*"((?:[^"\\]|\\.)*)"')
_ARGS_KEY_RE = re.compile(r'"arguments"\s*:\s*')
_WS = " \t\r\n"


def _json_unescape(s: str) -> str:
    try:
        return json.loads('"' + s + '"')
    except Exception:
        return s


class ToolCallStreamParser:
    """Parses `<tool_call>{"name": "...", "arguments": {...}}</tool_call>` blocks out of a
    (post-thinking) content stream, Hermes-style.

    `function.arguments` is streamed as raw JSON *text* fragments -- concatenating every
    `arguments_delta` for a given call index yields the exact JSON text the model produced (this
    mirrors OpenAI's own wire format: `arguments` is always a string, parsed by the caller). Since
    generation is append-only, once a character of the arguments value has arrived it is final, so
    fragments are forwarded immediately; the only thing ever held back is a partial `<tool_call>`/
    `</tool_call>` tag straddling a chunk boundary.
    """

    OPEN = "<tool_call>"
    CLOSE = "</tool_call>"

    def __init__(self) -> None:
        self._state = "content"  # "content" | "in_call"
        self._buffer = ""
        self.has_tool_calls = False
        self._call_counter = -1
        self._reset_call_state()

    def _reset_call_state(self) -> None:
        self._call_buffer = ""
        self._call_index = -1
        self._name_emitted = False
        self._args_value_start: Optional[int] = None
        self._args_pos = 0
        self._args_depth = 0
        self._in_string = False
        self._escape = False
        self._args_done = False

    def feed(self, text: str) -> tuple[str, list[ToolCallDelta]]:
        self._buffer += text
        content_out: list[str] = []
        deltas: list[ToolCallDelta] = []
        while True:
            if self._state == "content":
                idx = self._buffer.find(self.OPEN)
                if idx == -1:
                    safe_len = _safe_emit_len(self._buffer, self.OPEN)
                    if safe_len:
                        content_out.append(self._buffer[:safe_len])
                        self._buffer = self._buffer[safe_len:]
                    break
                content_out.append(self._buffer[:idx])
                self._buffer = self._buffer[idx + len(self.OPEN) :]
                self._state = "in_call"
                self._reset_call_state()
                continue
            else:  # in_call
                idx = self._buffer.find(self.CLOSE)
                if idx == -1:
                    safe_len = _safe_emit_len(self._buffer, self.CLOSE)
                    if safe_len:
                        self._call_buffer += self._buffer[:safe_len]
                        self._buffer = self._buffer[safe_len:]
                        deltas.extend(self._advance_call(final=False))
                    break
                self._call_buffer += self._buffer[:idx]
                self._buffer = self._buffer[idx + len(self.CLOSE) :]
                deltas.extend(self._advance_call(final=True))
                self._state = "content"
                continue
        return "".join(content_out), deltas

    def flush(self) -> tuple[str, list[ToolCallDelta]]:
        if self._state == "content":
            text, self._buffer = self._buffer, ""
            return text, []
        # Truncated mid-tool-call (e.g. hit max_tokens): best-effort finalize with what we have.
        self._call_buffer += self._buffer
        self._buffer = ""
        deltas = self._advance_call(final=True)
        self._state = "content"
        return "", deltas

    # -- internal: incremental progress on the current <tool_call> block -----------------

    def _advance_call(self, *, final: bool) -> list[ToolCallDelta]:
        deltas: list[ToolCallDelta] = []
        if self._call_index == -1:
            self._call_index = self._next_call_index()
            self.has_tool_calls = True

        if not self._name_emitted:
            m = _NAME_RE.search(self._call_buffer)
            name: Optional[str] = None
            if m:
                name = _json_unescape(m.group(1))
            elif final:
                name = self._fallback_name()
            if name is not None:
                self._name_emitted = True
                deltas.append(ToolCallDelta(index=self._call_index, id=f"call_{self._call_index}", name=name))

        if self._name_emitted and not self._args_done:
            if self._args_value_start is None:
                m = _ARGS_KEY_RE.search(self._call_buffer)
                if m:
                    start = m.end()
                    while start < len(self._call_buffer) and self._call_buffer[start] in _WS:
                        start += 1
                    if start < len(self._call_buffer) and self._call_buffer[start] == "{":
                        self._args_value_start = start
                        self._args_pos = start
                        self._args_depth = 0
                        self._in_string = False
                        self._escape = False
            if self._args_value_start is not None:
                delta = self._scan_args()
                if delta is not None:
                    deltas.append(delta)
            if final and not self._args_done:
                self._args_done = True  # truncated: stop trying, don't hang forever

        return deltas

    def _scan_args(self) -> Optional[ToolCallDelta]:
        buf = self._call_buffer
        pos = self._args_pos
        new_chars: list[str] = []
        while pos < len(buf):
            c = buf[pos]
            if self._in_string:
                if self._escape:
                    self._escape = False
                elif c == "\\":
                    self._escape = True
                elif c == '"':
                    self._in_string = False
            else:
                if c == '"':
                    self._in_string = True
                elif c == "{":
                    self._args_depth += 1
                elif c == "}":
                    self._args_depth -= 1
            new_chars.append(c)
            pos += 1
            if not self._in_string and self._args_depth == 0:
                self._args_done = True
                break
        self._args_pos = pos
        if not new_chars:
            return None
        return ToolCallDelta(index=self._call_index, arguments_delta="".join(new_chars))

    def _fallback_name(self) -> str:
        try:
            obj = json.loads(self._call_buffer)
            return str(obj.get("name", ""))
        except Exception:
            return ""

    def _next_call_index(self) -> int:
        self._call_counter += 1
        return self._call_counter


# --------------------------------------------------------------------------
# Chat template
# --------------------------------------------------------------------------


def render_chat_prompt(
    tokenizer,
    messages: list[dict],
    *,
    tools: Optional[list[dict]] = None,
    enable_thinking: bool = True,
    reasoning_effort: Optional[str] = None,
    preserve_thinking: Optional[bool] = None,
    add_generation_prompt: bool = True,
) -> list[int]:
    """Apply the HF tokenizer's own chat template (never a hand-rolled string)."""
    kwargs: dict = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": enable_thinking,
    }
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    if preserve_thinking is not None:
        kwargs["preserve_thinking"] = preserve_thinking
    if tools:
        kwargs["tools"] = tools

    encoded = tokenizer.apply_chat_template(messages, **kwargs)
    return _extract_input_ids(encoded)


def _extract_input_ids(encoded) -> list[int]:
    """`apply_chat_template(tokenize=True)` returns a plain list on some tokenizer/transformers
    versions and a `BatchEncoding`-like dict (`{"input_ids": [...], ...}`) on others. Normalize."""
    if isinstance(encoded, dict):
        return list(encoded["input_ids"])
    if hasattr(encoded, "input_ids"):
        return list(encoded.input_ids)
    return list(encoded)
