"""Mapping Mistral's chat-completions stream onto shared events."""

from __future__ import annotations

from typing import Any

from prism.enums import FinishReason
from prism.streaming.events import (
    StreamEndEvent,
    StreamEvent,
    StreamStartEvent,
    TextCompleteEvent,
    TextDeltaEvent,
    TextStartEvent,
    ToolCallEvent,
)
from prism.value_objects import ToolCall
from prism.value_objects.usage import Usage

__all__ = ["MistralStreamMapper"]

_FINISH_REASONS = {
    "stop": FinishReason.STOP,
    "tool_calls": FinishReason.TOOL_CALLS,
    "length": FinishReason.LENGTH,
    "model_length": FinishReason.LENGTH,
    "content_filter": FinishReason.CONTENT_FILTER,
}


class MistralStreamMapper:
    """Mistral's chat-completions stream, which needs MEMORY.

    Every chunk is a ``chat.completion.chunk`` carrying a ``delta`` and nothing
    else -- no message id after the first chunk, no accumulated text, and
    tool-call arguments arrive SPLIT ACROSS CHUNKS keyed only by an index. So a
    chunk cannot be mapped in isolation, the way an OpenAI Responses event can.

    That puts this closer to the Anthropic mapper than to the OpenAI one, for a
    different provider-side reason, and it carries the same consequence: ONE
    INSTANCE PER STREAM. A shared mapper would let two concurrent generations
    read each other's accumulated text, which surfaces under load and looks like
    the model hallucinating.
    """

    def __init__(self) -> None:
        self._message_id = ""
        self._model = ""
        self._started = False
        self._text_started = False
        self._text = ""
        self._usage: Usage | None = None
        self._tool_calls: dict[int, dict[str, str]] = {}

    def map(self, payload: dict[str, Any]) -> list[StreamEvent]:
        """The events for one chunk -- zero, one, or SEVERAL.

        Several, unlike the other mappers' one-or-none, because a single chunk
        can both start the stream and carry the first token, and the final chunk
        can complete the text, flush a tool call and end the stream at once.
        Returning only the first would drop the rest silently.
        """
        events: list[StreamEvent] = []

        if not self._started:
            self._started = True
            self._message_id = _string(payload.get("id"))
            self._model = _string(payload.get("model"))
            events.append(StreamStartEvent(model=self._model))

        usage = payload.get("usage")

        if isinstance(usage, dict):
            # Usage arrives on the LAST chunk on Mistral, not alongside the
            # deltas, so it is stored rather than emitted and read at the end.
            self._usage = Usage(
                prompt_tokens=_number(usage.get("prompt_tokens")),
                completion_tokens=_number(usage.get("completion_tokens")),
            )

        choices = payload.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else None

        if not isinstance(choice, dict):
            return events

        delta = choice.get("delta")
        delta = delta if isinstance(delta, dict) else {}
        text = _delta_text(delta)

        if text:
            if not self._text_started:
                self._text_started = True
                events.append(TextStartEvent(message_id=self._message_id))

            self._text += text
            events.append(TextDeltaEvent(delta=text, message_id=self._message_id))

        self._accumulate_tool_calls(delta)

        finish_reason = choice.get("finish_reason")

        if isinstance(finish_reason, str) and finish_reason:
            if self._text_started:
                events.append(TextCompleteEvent(text=self._text, message_id=self._message_id))

            for call in self._tool_calls.values():
                # Flushed at the END, once the arguments are whole. Emitting a
                # tool call while its JSON is still arriving hands a consumer a
                # fragment that will not parse.
                events.append(
                    ToolCallEvent(
                        tool_call=ToolCall(
                            id=call["id"],
                            name=call["name"],
                            arguments=call["args"] or "{}",
                        ),
                        message_id=self._message_id,
                    )
                )

            # Usage stays None when Mistral reported none, rather than becoming
            # a zeroed Usage -- zero tokens claims the generation was free.
            events.append(
                StreamEndEvent(
                    finish_reason=_FINISH_REASONS.get(finish_reason, FinishReason.UNKNOWN),
                    usage=self._usage,
                )
            )

        return events

    def _accumulate_tool_calls(self, delta: dict[str, Any]) -> None:
        """Merge tool-call fragments by index.

        The id and name arrive on the FIRST fragment only and the arguments
        arrive a few characters at a time, so each field is written only when
        the chunk actually carries it -- overwriting the name with an empty
        string on the second fragment is how a tool call ends up nameless.
        """
        calls = delta.get("tool_calls")

        if not isinstance(calls, list):
            return

        for raw in calls:
            if not isinstance(raw, dict):
                continue

            index = _number(raw.get("index"))
            function = raw.get("function")
            function = function if isinstance(function, dict) else {}
            existing = self._tool_calls.get(index, {"id": "", "name": "", "args": ""})

            self._tool_calls[index] = {
                "id": _string(raw.get("id")) or existing["id"],
                "name": _string(function.get("name")) or existing["name"],
                "args": existing["args"] + _string(function.get("arguments")),
            }


def _delta_text(delta: dict[str, Any]) -> str:
    """A delta's text, from either shape.

    ``content`` is a string on ordinary models and a list of typed chunks on
    reasoning ones. A list stringified is a Python repr, which reaches the
    consumer as tokens the model never produced.
    """
    content = delta.get("content")

    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        return ""

    return "".join(
        chunk.get("text", "")
        for chunk in content
        if isinstance(chunk, dict) and chunk.get("type") == "text"
    )


def _string(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _number(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0
