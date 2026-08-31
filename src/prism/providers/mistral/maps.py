"""Mapping between Prism's value objects and Mistral's chat-completions shapes.

This is the OpenAI CHAT-COMPLETIONS format, not the Responses format the OpenAI
provider in this package speaks. The two look similar enough that sharing a
mapper is tempting and wrong: roles differ from typed content parts, the finish
reason sits on the CHOICE rather than the root, and tool-call arguments are a
JSON string rather than an object. Every one of those fails quietly.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from prism._php import data_get
from prism.enums import FinishReason, ToolChoice
from prism.errors import PrismError
from prism.tool import Tool
from prism.value_objects.messages import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)

__all__ = [
    "map_finish_reason",
    "map_messages",
    "map_tool_choice",
    "map_tools",
]

_FINISH_REASONS = {
    "stop": FinishReason.STOP,
    "tool_calls": FinishReason.TOOL_CALLS,
    "length": FinishReason.LENGTH,
    "model_length": FinishReason.LENGTH,
    "content_filter": FinishReason.CONTENT_FILTER,
}


def map_finish_reason(data: Mapping[str, Any]) -> FinishReason:
    """Map ``finish_reason`` onto the shared enum.

    It sits on the CHOICE, not the top-level payload -- reading it off the root
    returns nothing and reports every generation as ``UNKNOWN``.

    An unrecognised reason becomes ``UNKNOWN`` rather than ``STOP``. Guessing
    ``STOP`` would present a truncated or filtered generation as a complete one.
    """
    reason = data_get(data, "choices.0.finish_reason", "")

    return _FINISH_REASONS.get(reason if isinstance(reason, str) else "", FinishReason.UNKNOWN)


def map_messages(
    messages: Sequence[Message],
    system_prompts: Sequence[SystemMessage],
) -> list[dict[str, Any]]:
    """System prompts FIRST, matching the reference.

    Mistral weights the earliest system turn most heavily, and a caller who set
    one expects it to lead.
    """
    mapped: list[dict[str, Any]] = []

    for message in [*system_prompts, *messages]:
        mapped.extend(_map_message(message))

    return mapped


def _map_message(message: Message) -> list[dict[str, Any]]:
    if isinstance(message, SystemMessage):
        return [{"role": "system", "content": message.content}]

    if isinstance(message, UserMessage):
        return [_map_user_message(message)]

    if isinstance(message, AssistantMessage):
        return [_map_assistant_message(message)]

    if isinstance(message, ToolResultMessage):
        # ONE message per result, not one carrying them all. Mistral matches
        # each back by `tool_call_id`, and a combined message has nowhere to put
        # the second id.
        return [
            {
                "role": "tool",
                "content": result.result,
                "tool_call_id": result.tool_call_id,
            }
            for result in message.tool_results
        ]

    raise PrismError.unknown_message_type(type(message).__name__)


def _map_user_message(message: UserMessage) -> dict[str, Any]:
    """A user turn, always as a content ARRAY.

    Even for plain text, matching the reference. Mistral accepts a bare string
    too, but sending the array shape unconditionally means adding a part later
    does not change the shape of every other message in a transcript -- so a
    stored conversation stays comparable with itself.

    IMAGE AND DOCUMENT PARTS ARE NOT MAPPED, because this port's ``UserMessage``
    carries text parts and nothing else. The reference maps ``image_url`` and
    ``document_url`` here; adding them without the message types to feed them
    would be a branch nothing can reach. Recorded in the port gaps register
    rather than stubbed.
    """
    return {
        "role": "user",
        "content": [{"type": "text", "text": message.text()}],
        **message.additional_attributes,
    }


def _map_assistant_message(message: AssistantMessage) -> dict[str, Any]:
    body: dict[str, Any] = {"role": "assistant", "content": message.content}

    if message.tool_calls:
        body["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    # A JSON STRING, not an object -- the chat-completions
                    # shape. An object is accepted by some gateways and rejected
                    # by Mistral, which is the worst kind of difference to
                    # carry. Already-string arguments pass through rather than
                    # being re-encoded into a quoted string.
                    "arguments": call.arguments
                    if isinstance(call.arguments, str)
                    else json.dumps(call.arguments),
                },
            }
            for call in message.tool_calls
        ]

    return body


def map_tools(tools: Sequence[Tool]) -> list[dict[str, Any]]:
    """Map tools onto Mistral's ``tools`` array.

    The chat-completions shape: a ``function`` object nested under a ``type``,
    where the Responses API flattens the same fields to the top level.
    ``parameters`` is always sent, even empty -- Mistral rejects a function
    declaration without one, unlike OpenAI which treats it as optional.

    Declaration order is preserved. Tool order reaches the model and influences
    which tool it picks.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": tool.parameters_as_dict() if tool.has_parameters() else {},
                    "required": list(tool.required_parameters),
                },
            },
        }
        for tool in tools
    ]


def map_tool_choice(choice: str | ToolChoice | None) -> str | dict[str, Any] | None:
    """Map the shared tool choice onto Mistral's form.

    ``auto`` / ``any`` / ``none`` are BARE STRINGS and a named tool is an
    object, so the return type is a union. Mistral spells "the model must call
    something" as ``any``, like Anthropic and unlike OpenAI's ``required``.
    """
    if choice is None:
        return None

    if choice is ToolChoice.AUTO:
        return "auto"

    if choice is ToolChoice.ANY:
        return "any"

    if choice is ToolChoice.NONE:
        return "none"

    return {"type": "function", "function": {"name": str(choice)}}
