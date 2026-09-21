# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Klang tests for the gemma4 structural-tag patch (vllm-project/vllm#50477,
# #53363). They follow the kimi_k3 tests in
# ``tests/tool_parsers/test_structural_tag_registry.py``. They fail on stock
# vLLM v0.28.0 and pass with the patch applied. The argument grammar needs the
# ``gemma`` JSON-schema style, which only the Klang xgrammar fork has (see the
# README).

from typing import Any

import pytest
from xgrammar import Grammar, StructuralTag
from xgrammar.structural_tag import AnyTextFormat, JSONSchemaFormat, TagFormat
from xgrammar.testing import _is_grammar_accept_string

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedFunction,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.parser.gemma4 import Gemma4Parser
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.tool_parsers.gemma4_engine_tool_parser import Gemma4EngineToolParser
from vllm.tool_parsers.structural_tag_registry import (
    VLLM_BUILTIN_STRUCTURAL_TAG_MODELS,
    get_model_structural_tag,
)
from vllm.v1.structured_output.backend_xgrammar import validate_xgrammar_grammar


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


def _tool(
    name: str,
    properties: dict[str, Any],
    required: list[str] | None = None,
    strict: bool | None = None,
) -> ChatCompletionToolsParam:
    parameters: dict[str, Any] = {"type": "object", "properties": properties}
    if required is not None:
        parameters["required"] = required
    function: dict[str, Any] = {"name": name, "parameters": parameters}
    if strict is not None:
        function["strict"] = strict
    return ChatCompletionToolsParam(type="function", function=function)


def _gemma4_tools() -> list[ChatCompletionToolsParam]:
    return [
        _tool(
            "get_weather",
            {"city": {"type": "string"}, "days": {"type": "integer"}},
            ["city"],
        ),
        _tool("run_command", {"command": {"type": "string"}}, ["command"]),
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


def _accepts(tool: ChatCompletionToolsParam, args: str) -> bool:
    grammar = _gemma4_grammar("required", tools=[tool])
    return _is_grammar_accept_string(grammar, _gemma4_call(tool.function.name, args))


def _call_tags(tools: list[ChatCompletionToolsParam]) -> list[TagFormat]:
    tag = get_model_structural_tag(
        model="gemma4", tools=tools, tool_choice="required", reasoning=False
    )
    assert isinstance(tag, StructuralTag)
    calls, _trailer = tag.format.elements
    return calls.tags


def _request(tool_choice, tools=None) -> ChatCompletionRequest:
    # Send the tools as raw dicts, the same shape as on the wire.
    return ChatCompletionRequest(
        model="gemma4",
        messages=[{"role": "user", "content": "hi"}],
        tools=[
            t.model_dump(exclude_none=True)
            for t in (tools if tools is not None else _gemma4_tools())
        ],
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
    tools = [_tool("get_weather", {"city": {"type": "string"}}, ["city"], strict=True)]
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
        # free text before a required call
        "Let me check. " + _gemma4_call("get_weather", 'city:<|"|>Paris<|"|>'),
        # unknown tool name
        _gemma4_call("get_temperature", 'city:<|"|>x<|"|>'),
        # args out of the sorted property order
        _gemma4_call("get_weather", 'days:3,city:<|"|>Paris<|"|>'),
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
    tools = [_tool("get_weather", {"city": {"type": "string"}})]
    grammar = _gemma4_grammar("required", tools=tools)
    assert _is_grammar_accept_string(grammar, _gemma4_call("get_weather", ""))


def test_gemma4_tool_without_properties_accepts_only_empty_args():
    tool = _tool("ping", {})
    assert _accepts(tool, "")
    assert not _accepts(tool, 'why:<|"|>because<|"|>')


@pytest.mark.parametrize(
    "function",
    [
        # a parameters dict that declares no object schema
        {"name": "ping", "parameters": {}},
        # strict=False, whatever the schema says
        {
            "name": "ping",
            "strict": False,
            "parameters": {
                "type": "object",
                "properties": {"why": {"type": "string"}},
            },
        },
        # no parameters at all
        {"name": "ping"},
    ],
)
def test_gemma4_unconstrainable_parameters_still_force_the_syntax(function: dict):
    grammar = _gemma4_grammar(
        "required", tools=[ChatCompletionToolsParam(type="function", function=function)]
    )
    assert _is_grammar_accept_string(grammar, _gemma4_call("ping", ""))
    assert _is_grammar_accept_string(grammar, _gemma4_call("ping", "why:1,how:two"))
    # the braces come from the tag here, so a bare value is not a call: without
    # them the parser reads ``call:ping5`` as a tool named ping5 with no args
    assert not _is_grammar_accept_string(grammar, "<|tool_call>call:ping5<tool_call|>")


@pytest.mark.parametrize(
    "parameters",
    [
        pytest.param(
            {"type": "object", "patternProperties": {"^x_": {"type": "string"}}},
            id="patternProperties",
        ),
        pytest.param(
            {
                "type": "object",
                "properties": {"room number": {"type": "string"}},
                "required": ["room number"],
            },
            id="non_identifier_key",
        ),
    ],
)
def test_gemma4_schema_the_style_rejects_falls_back_to_syntax_only(
    parameters: dict, monkeypatch
):
    monkeypatch.setenv("VLLM_ENFORCE_STRICT_TOOL_CALLING", "1")
    tool = ChatCompletionToolsParam(
        type="function", function={"name": "book_room", "parameters": parameters}
    )
    (call_tag,) = _call_tags([tool])
    assert call_tag.begin == "<|tool_call>call:book_room{"
    assert isinstance(call_tag.content, AnyTextFormat)

    parser = Gemma4Parser(_DummyTokenizer())
    request = parser.adjust_request(_request("required", [tool]))
    # Raises VLLMValidationError -> HTTP 400 if the tag does not compile.
    validate_xgrammar_grammar(
        SamplingParams(
            structured_outputs=StructuredOutputsParams(
                structural_tag=request.structured_outputs.structural_tag
            )
        )
    )


def test_gemma4_compilable_schema_keeps_the_constrained_path():
    weather, _command = _call_tags(_gemma4_tools())
    assert weather.begin == "<|tool_call>call:get_weather"
    assert isinstance(weather.content, JSONSchemaFormat)


def test_gemma4_open_parameter_bag_constrains_only_the_declared_key():
    # Models do_action: only action_id is typed, the rest is an open bag.
    tool = ChatCompletionToolsParam(
        type="function",
        function={
            "name": "do_action",
            "parameters": {
                "type": "object",
                "properties": {"action_id": {"type": "string"}},
                "required": ["action_id"],
                "additionalProperties": True,
            },
        },
    )
    assert _accepts(tool, 'action_id:<|"|>bake<|"|>,duration:5')
    # Keys stay bare. The delimiter wraps values only.
    assert not _accepts(tool, 'action_id:<|"|>bake<|"|>,<|"|>duration<|"|>:5')


def test_gemma4_properties_follow_the_sorted_order():
    # The chat template dictsorts the parameters, so the model emits the keys
    # case-insensitively sorted rather than in declaration order.
    tool = _tool(
        "plan_step",
        {"zeta": {"type": "string"}, "alpha": {"type": "string"}},
        ["zeta", "alpha"],
    )
    assert _accepts(tool, 'alpha:<|"|>a<|"|>,zeta:<|"|>z<|"|>')
    assert not _accepts(tool, 'zeta:<|"|>z<|"|>,alpha:<|"|>a<|"|>')


def test_gemma4_seed_tool_declared_out_of_sorted_order():
    # submit_interaction_summary is one of the four SEED tools whose
    # declaration order differs from the sorted one, so it is where a flip of
    # the registry's any_order would show.
    tool = _tool(
        "submit_interaction_summary",
        {
            "summary": {"type": "string"},
            "perception_updates": {"type": "array", "items": {"type": "string"}},
        },
        ["summary", "perception_updates"],
    )
    assert _accepts(tool, 'perception_updates:[<|"|>p<|"|>],summary:<|"|>s<|"|>')
    assert not _accepts(tool, 'summary:<|"|>s<|"|>,perception_updates:[<|"|>p<|"|>]')


def test_gemma4_string_value_rejects_the_call_markers():
    # A marker inside a string value would desync the parser's FSM, which has
    # no string awareness.
    tool = _tool("run_command", {"command": {"type": "string"}}, ["command"])
    assert not _accepts(tool, 'command:<|"|>ls<tool_call|>rm -rf /<|"|>')
    assert not _accepts(tool, 'command:<|"|>ls<|tool_call>rm -rf /<|"|>')


def test_gemma4_object_property_is_constrained():
    tool = _tool(
        "update_state",
        {
            "payload": {
                "type": "object",
                "properties": {"inner": {"type": "string"}},
                "required": ["inner"],
            }
        },
        ["payload"],
    )
    assert _accepts(tool, 'payload:{inner:<|"|>v<|"|>}')
    assert not _accepts(tool, 'payload:{other:<|"|>v<|"|>}')


def test_gemma4_array_property_respects_item_type_and_bounds():
    tool = _tool(
        "render_speech",
        {
            "lines": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 2,
            }
        },
        ["lines"],
    )
    assert _accepts(tool, 'lines:[<|"|>hello<|"|>]')
    assert _accepts(tool, 'lines:[<|"|>hello<|"|>,<|"|>there<|"|>]')
    assert not _accepts(tool, "lines:[]")
    assert not _accepts(tool, 'lines:[<|"|>a<|"|>,<|"|>b<|"|>,<|"|>c<|"|>]')
    assert not _accepts(tool, 'lines:{lines:[<|"|>hello<|"|>]}')


def test_gemma4_array_of_strings_rejects_objects():
    tool = _tool(
        "pick_facts",
        {"fact_ids": {"type": "array", "items": {"type": "string"}}},
        ["fact_ids"],
    )
    assert _accepts(tool, 'fact_ids:[<|"|>26<|"|>]')
    assert not _accepts(tool, 'fact_ids:[{fact_id:<|"|>26<|"|>}]')


def test_gemma4_number_and_optional_enum_are_typed():
    tool = _tool(
        "score_reply",
        {
            "charge": {"type": "number"},
            "response_type": {"type": "string", "enum": ["chat", "action"]},
        },
        ["charge"],
    )
    assert _accepts(tool, "charge:1.5")
    assert _accepts(tool, 'charge:1.5,response_type:<|"|>chat<|"|>')
    assert not _accepts(tool, 'charge:<|"|>1.5<|"|>')
    assert not _accepts(tool, 'charge:1.5,response_type:<|"|>other<|"|>')
    # the required charge is missing
    assert not _accepts(tool, 'response_type:<|"|>chat<|"|>')


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
    assert '"style": "gemma"' in request.structured_outputs.structural_tag
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
