"""Mapping messages onto the OpenAI Responses ``input`` array."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from prism import (
    AssistantMessage,
    Message,
    PrismError,
    SystemMessage,
    Text,
    ToolCall,
    ToolResult,
    ToolResultMessage,
    UserMessage,
)
from prism.providers.openai.maps import map_messages


def test_system_prompts_are_prepended_to_the_message_list() -> None:
    mapped = map_messages([UserMessage("hi")], [SystemMessage("You are terse.")])

    assert mapped[0] == {"role": "system", "content": "You are terse."}
    assert mapped[1]["role"] == "user"


def test_system_prompts_keep_their_order() -> None:
    mapped = map_messages([], [SystemMessage("A"), SystemMessage("B")])

    assert [item["content"] for item in mapped] == ["A", "B"]


def test_a_system_message_maps_to_a_bare_string_and_a_user_message_to_parts() -> None:
    mapped = map_messages([UserMessage("hi")], [SystemMessage("terse")])

    assert mapped[0]["content"] == "terse"
    assert mapped[1]["content"] == [{"type": "input_text", "text": "hi"}]


def test_user_additional_attributes_are_spread_at_item_level() -> None:
    mapped = map_messages([UserMessage("hi", additional_attributes={"id": "msg_local_1"})])

    # Siblings of role and content, not nested under a key.
    assert mapped[0] == {
        "role": "user",
        "content": [{"type": "input_text", "text": "hi"}],
        "id": "msg_local_1",
    }


def test_user_text_concatenates_every_text_part() -> None:
    message = UserMessage("world", [Text("hello ")])

    assert map_messages([message])[0]["content"] == [{"type": "input_text", "text": "hello world"}]


def test_an_empty_prompt_is_still_a_user_turn() -> None:
    assert map_messages([UserMessage("")])[0]["content"] == [{"type": "input_text", "text": ""}]


def test_assistant_message_maps_to_an_output_text_part() -> None:
    mapped = map_messages([AssistantMessage("4")])

    assert mapped == [{"role": "assistant", "content": [{"type": "output_text", "text": "4"}]}]


def test_assistant_message_with_empty_content_emits_no_assistant_item() -> None:
    # An empty output_text part is rejected by OpenAI.
    assert map_messages([AssistantMessage("")]) == []


def test_tool_calls_expand_into_their_own_top_level_items() -> None:
    message = AssistantMessage(
        "",
        [ToolCall(id="fc_1", name="weather", arguments={"city": "Paris"}, result_id="call_1")],
    )

    assert map_messages([message]) == [
        {
            "id": "fc_1",
            "call_id": "call_1",
            "type": "function_call",
            "name": "weather",
            # A JSON string, not an object.
            "arguments": '{"city":"Paris"}',
        }
    ]


def test_tool_call_arguments_are_re_encoded_as_a_json_string() -> None:
    message = AssistantMessage(
        "",
        [ToolCall(id="fc_1", name="w", arguments='{"city": "Paris"}', result_id="call_1")],
    )

    assert map_messages([message])[0]["arguments"] == '{"city":"Paris"}'


def test_tool_calls_with_no_arguments_send_an_empty_object() -> None:
    message = AssistantMessage("", [ToolCall(id="fc_1", name="ping", arguments={})])

    assert map_messages([message])[0]["arguments"] == "{}"


def test_malformed_tool_call_arguments_fail_at_mapping_time() -> None:
    message = AssistantMessage("", [ToolCall(id="fc_1", name="weather", arguments='{"city": ')])

    with pytest.raises(PrismError) as raised:
        map_messages([message])

    assert raised.value.code == "malformed_tool_call_arguments"


def test_a_reasoning_id_emits_its_own_item_ahead_of_the_calls() -> None:
    message = AssistantMessage(
        "",
        [
            ToolCall(
                id="fc_1",
                name="w",
                arguments={},
                result_id="call_1",
                reasoning_id="rs_1",
                reasoning_summary=[{"type": "summary_text", "text": "think"}],
            )
        ],
    )

    mapped = map_messages([message])

    assert mapped[0] == {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [{"type": "summary_text", "text": "think"}],
    }
    assert mapped[1]["type"] == "function_call"


def test_tool_result_is_keyed_by_the_result_id_not_the_call_id() -> None:
    message = ToolResultMessage(
        [
            ToolResult(
                tool_call_id="fc_1",
                tool_name="weather",
                args={"city": "Paris"},
                result="sunny",
                tool_call_result_id="call_1",
            )
        ]
    )

    assert map_messages([message]) == [
        {"type": "function_call_output", "call_id": "call_1", "output": "sunny"}
    ]


def test_a_string_tool_result_passes_through_untouched() -> None:
    message = ToolResultMessage(
        [ToolResult("fc_1", "t", {}, "plain text", tool_call_result_id="call_1")]
    )

    assert map_messages([message])[0]["output"] == "plain text"


def test_a_structured_tool_result_is_encoded_as_json() -> None:
    message = ToolResultMessage(
        [ToolResult("fc_1", "t", {}, {"a": 1}, tool_call_result_id="call_1")]
    )

    assert map_messages([message])[0]["output"] == '{"a":1}'


def test_a_numeric_tool_result_is_stringified() -> None:
    message = ToolResultMessage([ToolResult("fc_1", "t", {}, 42, tool_call_result_id="call_1")])

    assert map_messages([message])[0]["output"] == "42"


def test_an_unknown_message_type_is_refused_with_a_code() -> None:
    class Odd(Message):
        TYPE = "odd"

        def to_dict(self) -> dict[str, object]:
            return {}

        @classmethod
        def from_dict(cls, data: Mapping[str, object]) -> Odd:
            return cls()

    with pytest.raises(PrismError) as raised:
        map_messages([Odd()])

    assert raised.value.code == "unknown_message_type"


def test_message_order_is_preserved() -> None:
    mapped = map_messages(
        [UserMessage("a"), AssistantMessage("b"), UserMessage("c")],
        [SystemMessage("s")],
    )

    assert [item.get("role") for item in mapped] == ["system", "user", "assistant", "user"]
