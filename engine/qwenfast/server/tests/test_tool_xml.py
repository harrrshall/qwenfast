"""qwen3-coder XML tool calls (the format Qwen3.8's chat template asks for) and the
message normalization an agent harness needs. CPU only."""

from __future__ import annotations

import json

import pytest

from qwenfast.server.tokenization import (
    ToolCallStreamParser,
    coerce_xml_param,
    normalize_messages_for_template,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "edit",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "edits": {"type": "array", "items": {"type": "object"}},
                    "line": {"type": "integer"},
                    "force": {"type": "boolean"},
                    "timeout": {"type": ["number", "null"]},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
        },
    },
]

XML_TWO_CALLS = (
    "I will edit the file.\n\n<tool_call>\n<function=edit>\n<parameter=path>\nsrc/a.py\n</parameter>\n"
    '<parameter=edits>\n[{"old": "x = 1", "new": "x = 2"}]\n</parameter>\n'
    "<parameter=line>\n42\n</parameter>\n<parameter=force>\ntrue\n</parameter>\n"
    "<parameter=timeout>\n1.5\n</parameter>\n</function>\n</tool_call>\n"
    "<tool_call>\n<function=bash>\n<parameter=command>\ncat <<'EOF' > f.txt\n  indented\nEOF\n\n</parameter>\n"
    "</function>\n</tool_call>"
)


def _run(parser: ToolCallStreamParser, text: str, chunk: int):
    content, names, args = [], {}, {}
    for i in range(0, len(text), chunk):
        c, deltas = parser.feed(text[i : i + chunk])
        content.append(c)
        for d in deltas:
            if d.name is not None:
                names[d.index] = d.name
            args.setdefault(d.index, []).append(d.arguments_delta)
    c, deltas = parser.flush()
    content.append(c)
    for d in deltas:
        if d.name is not None:
            names[d.index] = d.name
        args.setdefault(d.index, []).append(d.arguments_delta)
    return "".join(content), names, {k: "".join(v) for k, v in args.items()}


@pytest.mark.parametrize("chunk", [1, 2, 3, 7, 13, 10_000])
def test_xml_calls_stream_at_any_chunking(chunk):
    p = ToolCallStreamParser(TOOLS)
    content, names, args = _run(p, XML_TWO_CALLS, chunk)
    assert p.has_tool_calls
    assert content.strip() == "I will edit the file."
    assert names == {0: "edit", 1: "bash"}
    a0 = json.loads(args[0])
    assert a0 == {
        "path": "src/a.py",
        "edits": [{"old": "x = 1", "new": "x = 2"}],
        "line": 42,
        "force": True,
        "timeout": 1.5,
    }
    # exactly one leading and one trailing newline stripped; inner whitespace preserved
    assert json.loads(args[1]) == {"command": "cat <<'EOF' > f.txt\n  indented\nEOF\n"}


def test_bare_function_block_without_wrapper():
    p = ToolCallStreamParser(TOOLS)
    content, names, args = _run(p, "ok <function=bash>\n<parameter=command>\nls\n</parameter>\n</function> done", 4)
    assert names == {0: "bash"}
    assert json.loads(args[0]) == {"command": "ls"}
    assert content == "ok  done"


def test_xml_call_without_params():
    p = ToolCallStreamParser([])
    _, names, args = _run(p, "<tool_call>\n<function=noop>\n</function>\n</tool_call>", 5)
    assert names == {0: "noop"} and json.loads(args[0]) == {}


def test_truncated_xml_call_is_valid_json():
    p = ToolCallStreamParser(TOOLS)
    _, names, args = _run(p, "<tool_call>\n<function=bash>\n<parameter=command>\necho hi", 3)
    assert names == {0: "bash"}
    assert json.loads(args[0]) == {"command": "echo hi"}


def test_unknown_param_stays_string_and_hermes_json_still_works():
    p = ToolCallStreamParser(TOOLS)
    _, names, args = _run(
        p,
        '<tool_call>\n<function=bash>\n<parameter=extra>\n{"a": 1}\n</parameter>\n</function>\n</tool_call>'
        '<tool_call>\n{"name": "bash", "arguments": {"command": "pwd"}}\n</tool_call>',
        6,
    )
    assert json.loads(args[0]) == {"extra": '{"a": 1}'}
    assert names[1] == "bash" and json.loads(args[1]) == {"command": "pwd"}


def test_less_than_in_content_is_not_swallowed():
    p = ToolCallStreamParser(TOOLS)
    content, names, _ = _run(p, "if a < b and x <func> then <tool", 3)
    assert names == {}
    assert content == "if a < b and x <func> then <tool"


@pytest.mark.parametrize(
    "raw,types,expected",
    [
        ("\n5\n", {"integer"}, 5),
        ("\nfive\n", {"integer"}, "five"),
        ("\nnull\n", {"integer", "null"}, None),
        ("\nnull\n", {"string"}, "null"),
        ("\n{'a': 1}\n", {"object"}, {"a": 1}),
        ("\n  spaced  \n", {"string"}, "  spaced  "),
        ("\nFalse\n", {"boolean"}, False),
        ("\n3.0\n", {"number"}, 3.0),
    ],
)
def test_coerce(raw, types, expected):
    assert coerce_xml_param(raw, types) == expected


def test_normalize_messages():
    msgs = [
        {"role": "developer", "content": "be brief"},
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c0", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}},
                {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "not json"}},
            ],
        },
        {"role": "tool", "tool_call_id": "c0", "content": "a.py"},
        {"role": "system", "content": [{"type": "text", "text": "tools changed"}]},
        {"role": "user", "content": "go on"},
    ]
    out = normalize_messages_for_template(msgs)
    assert out[0] == {"role": "system", "content": "be brief\n\ntools changed"}
    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool", "user"]
    calls = out[2]["tool_calls"]
    assert calls[0]["function"]["arguments"] == {"command": "ls"}
    assert calls[1]["function"]["arguments"] == {"input": "not json"}
    # caller's objects untouched
    assert msgs[2]["tool_calls"][0]["function"]["arguments"] == '{"command": "ls"}'


def test_normalized_messages_render_with_real_template():
    """Renders through the actual Qwen3.8 template when transformers + the tokenizer are around."""
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B")
    except Exception:
        pytest.skip("tokenizer not available offline")
    msgs = normalize_messages_for_template(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "list files"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c0", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}]},
            {"role": "tool", "tool_call_id": "c0", "content": "a.py"},
            {"role": "system", "content": "late system"},
        ]
    )
    text = tok.apply_chat_template(msgs, tools=TOOLS, tokenize=False, add_generation_prompt=True)
    assert "<function=bash>\n<parameter=command>\nls\n</parameter>" in text
    assert "late system" in text
