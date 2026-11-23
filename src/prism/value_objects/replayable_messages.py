"""The conversation as a provider can be sent it."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from prism.value_objects.messages import AssistantMessage, Message, ToolResultMessage

__all__ = ["replayable_messages"]


def replayable_messages(messages: Sequence[Message]) -> list[Message]:
    """Replace unusable arguments on every tool call a later message answered.

    A tool call whose arguments could not be parsed cannot go back as written,
    and a provider requires the call beside its result. Once answered, the error
    is on record and the call is sent with no arguments. A call nobody answered
    is left alone, so that mapping it still fails with
    ``malformed_tool_call_arguments``.
    """
    replayable = list(messages)
    answered: set[str] = set()

    for index in range(len(replayable) - 1, -1, -1):
        message = replayable[index]

        if isinstance(message, ToolResultMessage):
            answered.update(result.tool_call_id for result in message.tool_results)
            continue

        if not isinstance(message, AssistantMessage) or not answered:
            continue

        calls = [
            call.without_malformed_arguments() if call.id in answered else call
            for call in message.tool_calls
        ]

        if any(new is not old for new, old in zip(calls, message.tool_calls, strict=True)):
            replayable[index] = replace(message, tool_calls=calls)

    return replayable
