"""Value objects, both directions.

The reference can write these and cannot read them back. Both halves live here,
so every row asserts the serialised bytes AND that rebuilding from them and
re-serialising returns the identical bytes.
"""

from __future__ import annotations

from typing import Any

import pytest

from prism import (
    AssistantMessage,
    Meta,
    PrismError,
    ProviderTool,
    ProviderToolCall,
    SystemMessage,
    Text,
    ToolCall,
    ToolResult,
    ToolResultMessage,
    Usage,
    UserMessage,
    canonical,
    message_from_dict,
)


def assert_roundtrips(subject: Any, expected: str) -> None:
    serialised = canonical.encode(subject.to_dict())
    assert serialised == expected

    rebuilt = type(subject).from_dict(subject.to_dict())
    assert canonical.encode(rebuilt.to_dict()) == expected


# -- user message -----------------------------------------------------------


def test_user_message_roundtrips() -> None:
    assert_roundtrips(
        UserMessage("Who are you?", additional_attributes={"turn": "1"}),
        '{"type":"user","content":"Who are you?",'
        '"additional_content":[{"text":"Who are you?"}],'
        '"additional_attributes":{"turn":"1"}}',
    )


def test_rebuilding_a_user_message_does_not_double_its_text() -> None:
    # The constructor appends a Text part built from content, so the stored
    # additional_content already contains it. Handing that list straight back
    # appends a SECOND copy and the message text doubles on every save-and-load
    # cycle — silently, with nothing but a growing conversation to show for it.
    message = UserMessage("Who are you?")

    for _ in range(5):
        message = UserMessage.from_dict(message.to_dict())

    assert message.text() == "Who are you?"
    assert len(message.additional_content) == 1


def test_a_user_message_with_extra_text_parts_roundtrips() -> None:
    message = UserMessage("world", [Text("hello ")])

    assert message.text() == "hello world"
    assert UserMessage.from_dict(message.to_dict()).text() == "hello world"


def test_the_constructor_does_not_mutate_the_callers_list() -> None:
    parts = [Text("a")]
    UserMessage("b", parts)

    assert parts == [Text("a")]


# -- assistant message ------------------------------------------------------


def test_assistant_message_with_tool_calls_roundtrips() -> None:
    assert_roundtrips(
        AssistantMessage(
            "Checking the weather.",
            [ToolCall("fc_1", "weather", {"city": "Paris"}, result_id="call_1")],
            {"reasoningSummaries": ["Consider the two cases."]},
        ),
        '{"type":"assistant","content":"Checking the weather.",'
        '"tool_calls":[{"id":"fc_1","name":"weather","arguments":{"city":"Paris"},'
        '"result_id":"call_1","reasoning_id":null,"reasoning_summary":null}],'
        '"additional_content":{"reasoningSummaries":["Consider the two cases."]},'
        '"tool_approval_requests":[]}',
    )


def test_an_empty_assistant_message_roundtrips() -> None:
    assert_roundtrips(
        AssistantMessage("Done."),
        '{"type":"assistant","content":"Done.","tool_calls":[],'
        '"additional_content":{},"tool_approval_requests":[]}',
    )


# -- tool call --------------------------------------------------------------


def test_tool_call_keeps_string_arguments_as_a_string() -> None:
    # The arguments field is a union: providers stream it as a string and
    # callers build it as a mapping. Normalising either way changes what gets
    # re-sent.
    assert_roundtrips(
        ToolCall("fc_2", "weather", '{"city":"Paris"}', result_id="call_2"),
        '{"id":"fc_2","name":"weather","arguments":"{\\"city\\":\\"Paris\\"}",'
        '"result_id":"call_2","reasoning_id":null,"reasoning_summary":null}',
    )


def test_tool_call_decodes_string_arguments_on_demand() -> None:
    assert ToolCall("fc", "t", '{"a":1}').decoded_arguments() == {"a": 1}


def test_tool_call_decodes_mapping_arguments_unchanged() -> None:
    assert ToolCall("fc", "t", {"a": 1}).decoded_arguments() == {"a": 1}


@pytest.mark.parametrize("arguments", ["", "0"])
def test_falsy_argument_strings_decode_to_no_arguments(arguments: str) -> None:
    assert ToolCall("fc", "t", arguments).decoded_arguments() == {}


def test_malformed_arguments_raise_with_a_code() -> None:
    with pytest.raises(PrismError) as raised:
        ToolCall("fc", "weather", '{"city": ').decoded_arguments()

    assert raised.value.code == "malformed_tool_call_arguments"


def test_raw_control_characters_inside_strings_are_recovered() -> None:
    # Some providers emit them unescaped; escaping in place beats stripping,
    # which would corrupt intentional newlines.
    assert ToolCall("fc", "t", '{"a":"x\ny"}').decoded_arguments() == {"a": "x\ny"}


def test_arguments_that_decode_to_a_non_object_become_no_arguments() -> None:
    assert ToolCall("fc", "t", "[1,2]").decoded_arguments() == {}


# -- usage ------------------------------------------------------------------


def test_usage_with_every_optional_field_unset_roundtrips() -> None:
    assert_roundtrips(
        Usage(11, 56),
        '{"prompt_tokens":11,"completion_tokens":56,"cache_write_input_tokens":null,'
        '"cache_read_input_tokens":null,"thought_tokens":null,"cost":null}',
    )


def test_usage_with_every_optional_field_set_roundtrips() -> None:
    assert_roundtrips(
        Usage(128, 12, 64, 896, 4, 0.00125),
        '{"prompt_tokens":128,"completion_tokens":12,"cache_write_input_tokens":64,'
        '"cache_read_input_tokens":896,"thought_tokens":4,"cost":0.00125}',
    )


# -- the rest ---------------------------------------------------------------


def test_system_message_roundtrips() -> None:
    assert_roundtrips(
        SystemMessage("You are terse."),
        '{"type":"system","content":"You are terse."}',
    )


def test_tool_result_message_roundtrips() -> None:
    assert_roundtrips(
        ToolResultMessage([ToolResult("fc_1", "weather", {"city": "Paris"}, "sunny", "call_1")]),
        '{"type":"tool_result","tool_results":[{"tool_call_id":"fc_1","tool_name":"weather",'
        '"args":{"city":"Paris"},"result":"sunny","tool_call_result_id":"call_1",'
        '"artifacts":[]}],"tool_approval_responses":[]}',
    )


def test_meta_with_no_service_tier_roundtrips() -> None:
    assert_roundtrips(
        Meta("resp_1", "gpt-4o"),
        '{"id":"resp_1","model":"gpt-4o","rate_limits":[],"service_tier":null}',
    )


def test_text_part_roundtrips() -> None:
    assert_roundtrips(Text("hi"), '{"text":"hi"}')


def test_provider_tool_roundtrips() -> None:
    assert_roundtrips(
        ProviderTool("web_search_preview", options={"depth": 2}),
        '{"type":"web_search_preview","name":null,"options":{"depth":2}}',
    )


def test_provider_tool_call_roundtrips() -> None:
    assert_roundtrips(
        ProviderToolCall("ws_1", "web_search_call", "completed", {"id": "ws_1"}),
        '{"id":"ws_1","type":"web_search_call","status":"completed","data":{"id":"ws_1"}}',
    )


# -- dispatch ---------------------------------------------------------------


@pytest.mark.parametrize(
    "subject",
    [
        UserMessage("hi"),
        AssistantMessage("hi"),
        SystemMessage("hi"),
        ToolResultMessage(),
    ],
)
def test_message_from_dict_rebuilds_the_right_class(subject: Any) -> None:
    rebuilt = message_from_dict(subject.to_dict())

    assert type(rebuilt) is type(subject)
    assert rebuilt.to_dict() == subject.to_dict()


def test_message_from_dict_refuses_an_unknown_type_with_a_code() -> None:
    with pytest.raises(PrismError) as raised:
        message_from_dict({"type": "nonesuch"})

    assert raised.value.code == "unknown_message_type"
