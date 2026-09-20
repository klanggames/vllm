# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Klang tests for the gemma4 structural-tag patch (vllm-project/vllm#50477,
# #53363). They follow the kimi_k3 tests in
# ``tests/tool_parsers/test_structural_tag_registry.py``. They fail on stock
# vLLM v0.28.0 and pass with the patch applied.

import pytest
from xgrammar import Grammar, StructuralTag
from xgrammar.testing import _is_grammar_accept_string

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedFunction,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.parser.gemma4 import Gemma4Parser
from vllm.tool_parsers.gemma4_engine_tool_parser import Gemma4EngineToolParser
from vllm.tool_parsers.structural_tag_registry import (
    VLLM_BUILTIN_STRUCTURAL_TAG_MODELS,
    get_model_structural_tag,
)


class _DummyTokenizer:
    def get_vocab(self):
        return {}

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


class _VocabTokenizer(_DummyTokenizer):
    def get_vocab(self):
        return {
            "<|tool_call>": 1,
            "<tool_call|>": 2,
            '<|"|>': 3,
            "<|channel>": 4,
            "<channel|>": 5,
        }

    def decode(self, token_ids, **kwargs):
        by_id = {tid: text for text, tid in self.get_vocab().items()}
        return "".join(by_id.get(tid, chr(tid)) for tid in token_ids)


def _gemma4_tools() -> list[ChatCompletionToolsParam]:
    return [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "days": {"type": "integer"},
                    },
                    "required": ["city"],
                },
            },
        ),
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "run_command",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        ),
    ]


def _gemma4_call(name: str, args: str) -> str:
    return f"<|tool_call>call:{name}{{{args}}}<tool_call|>"


def _gemma4_grammar(tool_choice, tools=None) -> Grammar:
    tag = get_model_structural_tag(
        model="gemma4",
        tools=tools if tools is not None else _gemma4_tools(),
        tool_choice=tool_choice,
        reasoning=False,
    )
    assert isinstance(tag, StructuralTag)
    return Grammar.from_structural_tag(tag)


def _request(tool_choice) -> ChatCompletionRequest:
    # Send the tools as raw dicts, the same shape as on the wire.
    return ChatCompletionRequest(
        model="gemma4",
        messages=[{"role": "user", "content": "hi"}],
        tools=[t.model_dump(exclude_none=True) for t in _gemma4_tools()],
        tool_choice=tool_choice,
    )


def test_gemma4_registered_as_vllm_builtin():
    assert "gemma4" in VLLM_BUILTIN_STRUCTURAL_TAG_MODELS
    assert Gemma4EngineToolParser.structural_tag_model == "gemma4"
    assert not Gemma4EngineToolParser.supports_required_and_named


def test_gemma4_auto_without_strict_is_unconstrained():
    tag = get_model_structural_tag(
        model="gemma4",
        tools=_gemma4_tools(),
        tool_choice="auto",
        reasoning=False,
    )
    assert tag is None


def test_gemma4_auto_with_strict_tool_allows_text_and_calls():
    tools = [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "get_weather",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        )
    ]
    grammar = _gemma4_grammar("auto", tools=tools)
    assert _is_grammar_accept_string(grammar, "Just answering.")
    assert _is_grammar_accept_string(
        grammar, "One moment. " + _gemma4_call("get_weather", 'city:<|"|>Paris<|"|>')
    )


@pytest.mark.parametrize(
    "body",
    [
        # single call, one string arg
        _gemma4_call("get_weather", 'city:<|"|>Paris<|"|>'),
        # two args, compact form
        _gemma4_call("get_weather", 'city:<|"|>Paris<|"|>,days:3'),
        # args in reverse order
        _gemma4_call("get_weather", 'days:3,city:<|"|>Paris<|"|>'),
        # string value with spaces and metacharacters
        _gemma4_call("run_command", 'command:<|"|>grep -E "a|b" x.py<|"|>'),
        # two calls back-to-back
        _gemma4_call("get_weather", 'city:<|"|>Paris<|"|>')
        + _gemma4_call("run_command", 'command:<|"|>ls -la<|"|>'),
        # trailing tool-response terminator
        _gemma4_call("get_weather", 'city:<|"|>Paris<|"|>') + "<|tool_response>",
    ],
)
def test_gemma4_required_accepts_valid_tool_calls(body: str):
    assert _is_grammar_accept_string(_gemma4_grammar("required"), body)


@pytest.mark.parametrize(
    "body",
    [
        # required but no call
        "no call here",
        # no free text before the call: with EOS masked, a permitted preamble
        # lets the model write text until max_tokens and never start a call
        "Let me check. " + _gemma4_call("get_weather", 'city:<|"|>Paris<|"|>'),
        # unknown tool name
        _gemma4_call("get_temperature", 'city:<|"|>x<|"|>'),
        # integer arg with a non-numeric value
        _gemma4_call("get_weather", 'city:<|"|>Paris<|"|>,days:abc'),
        # undeclared argument key
        _gemma4_call("get_weather", 'city:<|"|>Paris<|"|>,zzz:<|"|>x<|"|>'),
        # empty args while the schema has required properties
        _gemma4_call("get_weather", ""),
        # unterminated call
        '<|tool_call>call:get_weather{city:<|"|>Paris',
        # trailing junk after the call
        _gemma4_call("get_weather", 'city:<|"|>Paris<|"|>') + "and more text",
    ],
)
def test_gemma4_required_rejects_invalid(body: str):
    assert not _is_grammar_accept_string(_gemma4_grammar("required"), body)


def test_gemma4_schema_without_required_accepts_empty_call():
    tools = [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        )
    ]
    grammar = _gemma4_grammar("required", tools=tools)
    assert _is_grammar_accept_string(grammar, _gemma4_call("get_weather", ""))


def test_gemma4_object_property_falls_back_to_permissive_args():
    tools = [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "update_state",
                "parameters": {
                    "type": "object",
                    "properties": {"payload": {"type": "object"}},
                    "required": ["payload"],
                },
            },
        )
    ]
    grammar = _gemma4_grammar("required", tools=tools)
    body = _gemma4_call("update_state", 'payload:{inner:<|"|>v<|"|>}')
    assert _is_grammar_accept_string(grammar, body)


def test_gemma4_forced_tool_choice_builds_single_mandatory_call():
    grammar = _gemma4_grammar(
        ChatCompletionNamedToolChoiceParam(
            function=ChatCompletionNamedFunction(name="get_weather")
        )
    )
    ok = _gemma4_call("get_weather", 'city:<|"|>Paris<|"|>')
    assert _is_grammar_accept_string(grammar, ok)
    assert not _is_grammar_accept_string(
        grammar, _gemma4_call("run_command", 'command:<|"|>ls<|"|>')
    )
    # forced stops after the first call
    assert not _is_grammar_accept_string(grammar, ok + ok)
    assert not _is_grammar_accept_string(grammar, "just text")


def test_unified_parser_attaches_structural_tag_for_required(monkeypatch):
    monkeypatch.setenv("VLLM_ENFORCE_STRICT_TOOL_CALLING", "1")
    parser = Gemma4Parser(_DummyTokenizer())
    request = parser.adjust_request(_request("required"))
    assert request.structured_outputs is not None
    assert request.structured_outputs.structural_tag is not None
    assert request.skip_special_tokens is False


def test_unified_parser_attaches_structural_tag_for_named(monkeypatch):
    monkeypatch.setenv("VLLM_ENFORCE_STRICT_TOOL_CALLING", "1")
    parser = Gemma4Parser(_DummyTokenizer())
    request = parser.adjust_request(
        _request({"type": "function", "function": {"name": "get_weather"}})
    )
    assert request.structured_outputs is not None
    assert request.structured_outputs.structural_tag is not None


def test_unified_parser_leaves_none_tool_choice_unconstrained():
    parser = Gemma4Parser(_DummyTokenizer())
    request = parser.adjust_request(_request("none"))
    assert request.structured_outputs is None


def test_tool_parser_builds_structural_tag_for_delegating_path(monkeypatch):
    monkeypatch.setenv("VLLM_ENFORCE_STRICT_TOOL_CALLING", "1")
    parser = Gemma4EngineToolParser(_DummyTokenizer())
    tag = parser.get_structural_tag(_request("required"))
    assert isinstance(tag, StructuralTag)


def test_parse_extracts_text_spelled_markers_with_token_ids(monkeypatch):
    # The structural tag forces the markers as text, and xgrammar lets the
    # model spell them with regular tokens instead of the special tokens.
    # The parser must lex the markers from text even when the vocabulary
    # has the special tokens and the stream carries token ids.
    monkeypatch.setenv("VLLM_ENFORCE_STRICT_TOOL_CALLING", "1")
    parser = Gemma4Parser(_VocabTokenizer())
    request = _request("required")
    parser.adjust_request(request)
    out = '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>'
    ids = [ord(c) for c in out]
    reasoning, content, tool_calls = parser.parse(
        out, request, enable_auto_tools=True, model_output_token_ids=ids
    )
    assert tool_calls is not None and len(tool_calls) == 1
    assert tool_calls[0].name == "get_weather"
    assert content is None
