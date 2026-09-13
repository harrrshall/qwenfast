"""OpenAI-compatible request schemas (Pydantic v2).

Only request bodies get real models — responses are built as plain dicts in `app.py`, since the
OpenAI wire format has enough optional/variant shape (streaming vs not, chat vs completions,
tool_calls vs content) that hand-building them is clearer than fighting a response model through
every branch.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, field_validator


class ChatTemplateKwargs(BaseModel):
    model_config = ConfigDict(extra="allow")

    enable_thinking: Optional[bool] = None
    reasoning_effort: Optional[Literal["xhigh", "medium", "low"]] = None
    preserve_thinking: Optional[bool] = None


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Optional[Union[str, list[dict[str, Any]]]] = None
    name: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

    def text(self) -> str:
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        # Multimodal-shaped content (list of {"type": "text", "text": ...} parts): join the text
        # parts. Non-text parts (images) are out of scope (the engine is text-only).
        parts = []
        for part in self.content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
        return "".join(parts)


class _SamplingFields(BaseModel):
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None
    stop: Optional[Union[str, list[str]]] = None
    seed: Optional[int] = None
    n: int = 1
    ignore_eos: bool = False
    logprobs: Optional[Union[bool, int]] = None
    top_logprobs: Optional[int] = None
    stream: bool = False
    stream_options: Optional[StreamOptions] = None

    @field_validator("n")
    @classmethod
    def _n_must_be_one(cls, v: int) -> int:
        if v != 1:
            raise ValueError("n != 1 is not supported")
        return v

    def stop_list(self) -> list[str]:
        if self.stop is None:
            return []
        if isinstance(self.stop, str):
            return [self.stop]
        return list(self.stop)

    def logprobs_count(self) -> Optional[int]:
        if self.logprobs is None or self.logprobs is False:
            return None
        if self.logprobs is True:
            return self.top_logprobs or 1
        return int(self.logprobs)  # legacy /v1/completions int form


class ChatCompletionRequest(_SamplingFields):
    model: str
    messages: list[ChatMessage]
    tools: Optional[list[dict[str, Any]]] = None
    tool_choice: Optional[Union[str, dict[str, Any]]] = None
    chat_template_kwargs: Optional[ChatTemplateKwargs] = None
    # Top-level convenience alias for chat_template_kwargs.reasoning_effort.
    reasoning_effort: Optional[Literal["xhigh", "medium", "low"]] = None

    def resolve_enable_thinking(self) -> bool:
        if self.chat_template_kwargs and self.chat_template_kwargs.enable_thinking is not None:
            return self.chat_template_kwargs.enable_thinking
        return False  # public-API default: no hidden reasoning unless asked (chat_template_kwargs.enable_thinking=true); see docs/api.md

    def resolve_reasoning_effort(self) -> Optional[str]:
        if self.chat_template_kwargs and self.chat_template_kwargs.reasoning_effort is not None:
            return self.chat_template_kwargs.reasoning_effort
        return self.reasoning_effort

    def resolve_preserve_thinking(self) -> Optional[bool]:
        if self.chat_template_kwargs:
            return self.chat_template_kwargs.preserve_thinking
        return None


class CompletionRequest(_SamplingFields):
    model: str
    prompt: Union[str, list[int], list[str]]
    echo: bool = False
    suffix: Optional[str] = None
