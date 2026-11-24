"""Mapping an OpenAI chat-completions stream onto shared events.

Ported from the reference's ``Handlers/ChatCompletions/Stream``. Each chunk is a
``delta``; text, refusal fragments and tool-call arguments all arrive split
across chunks, so the mapper carries state and there is ONE INSTANCE PER STREAM.

Core runs no tools. A stream that finishes on tool calls hands them over as
:class:`ToolCallEvent` s and ends with ``FinishReason.TOOL_CALLS``, and nothing
else is sent: the caller decides whether any of them runs.
"""

from __future__ import annotations

import uuid
from typing import Any

from prism._php import data_get
from prism.enums import FinishReason
from prism.errors import PrismError
from prism.providers.openai.chat_completions import chat_usage, map_chat_finish_reason
from prism.streaming.events import (
    ErrorEvent,
    StreamEndEvent,
    StreamEvent,
    StreamStartEvent,
    TextCompleteEvent,
    TextDeltaEvent,
    TextStartEvent,
    ToolCallEvent,
)
from prism.streaming.failure_usage import StreamFailureUsageTracker
from prism.value_objects import ToolCall, Usage
from prism.value_objects.stream_failure_usage import StreamFailureUsage

__all__ = ["ChatCompletionsStreamMapper"]


class ChatCompletionsStreamMapper:
    def __init__(self, model: str) -> None:
        self._model = model
        self._message_id = ""
        self._started = False
        self._text_started = False
        self._text_completed = False
        self._text = ""
        self._refusal = ""
        self._finish_reason: FinishReason | None = None
        self._usage: Usage | None = None
        self._tool_calls: dict[int, dict[str, str]] = {}
        self._failure = StreamFailureUsageTracker("openai")

    def map(self, payload: dict[str, Any]) -> list[StreamEvent]:
        """The events for one chunk: zero, one or several.

        :raises PrismError: ``provider_response_error`` once a chunk carrying a
            finish reason arrives after any refusal fragment. Earlier text stays
            provisional; a refusal is never turned into a completed answer.
        """
        self._failure.record(payload)
        events: list[StreamEvent] = []

        if not self._started:
            self._started = True
            self._message_id = _string(payload.get("id")) or f"msg_{uuid.uuid4().hex}"
            events.append(StreamStartEvent(model=_string(payload.get("model")) or self._model))

        error = payload.get("error")
        if error is not None:
            events.append(
                ErrorEvent(
                    code=_string(data_get(error, "type")) or "unknown_error",
                    message=_string(data_get(error, "message")) or "No error message provided",
                    recoverable=False,
                    failure_usage=self.failure_usage(),
                )
            )
            return events

        usage = payload.get("usage")
        if isinstance(usage, dict) and usage:
            # Usage may arrive on its own chunk AFTER the finish chunk, with an
            # empty choices list, so it is held and read at the end.
            self._usage = chat_usage(usage)

        choices = payload.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else None
        if not isinstance(choice, dict):
            return events

        delta = choice.get("delta")
        delta = delta if isinstance(delta, dict) else {}
        finish = choice.get("finish_reason")

        refusal = delta.get("refusal")
        if isinstance(refusal, str):
            self._refusal += refusal
        if self._refusal and finish is not None:
            self._raise_refusal()

        self._accumulate_tool_calls(delta)

        content = delta.get("content")
        if isinstance(content, str) and content != "":
            if not self._text_started:
                self._text_started = True
                events.append(TextStartEvent(message_id=self._message_id))
            self._text += content
            events.append(TextDeltaEvent(delta=content, message_id=self._message_id))

        if finish is not None:
            self._finish_reason = map_chat_finish_reason(finish)
            if self._text_started and not self._text_completed:
                self._text_completed = True
                events.append(TextCompleteEvent(text=self._text, message_id=self._message_id))

        return events

    def finish(self) -> list[StreamEvent]:
        """The closing events, once the stream has ended (``[DONE]`` or end of body).

        A stream that ended WITHOUT a finish chunk reports ``UNKNOWN`` rather
        than ``STOP``: a cut-off stream is not a completed answer.
        """
        if self._refusal:
            self._raise_refusal()

        events: list[StreamEvent] = []

        if self._text_started and not self._text_completed:
            self._text_completed = True
            events.append(TextCompleteEvent(text=self._text, message_id=self._message_id))

        for index in sorted(self._tool_calls):
            call = self._tool_calls[index]
            events.append(
                ToolCallEvent(
                    tool_call=ToolCall(
                        id=call["id"] or f"call_{uuid.uuid4().hex}",
                        name=call["name"],
                        # Kept as the text the model sent. Reading it through
                        # decoded_arguments() is where a cut-off or non-object
                        # value is refused, never read as "no arguments".
                        arguments=call["args"],
                    ),
                    message_id=self._message_id,
                )
            )

        events.append(
            StreamEndEvent(
                finish_reason=self._finish_reason or FinishReason.UNKNOWN,
                usage=self._usage,
            )
        )
        return events

    def failure_usage(self) -> StreamFailureUsage | None:
        return self._failure.snapshot()

    def decode_failure(self) -> ErrorEvent:
        return ErrorEvent(
            code="stream_decode_error",
            message="The stream contained invalid JSON.",
            recoverable=False,
            failure_usage=self.failure_usage(),
        )

    def _raise_refusal(self) -> None:
        raise PrismError.provider_response_error(
            f"OpenAI refused the request: {self._refusal}"
        ).with_stream_failure_usage(self.failure_usage())

    def _accumulate_tool_calls(self, delta: dict[str, Any]) -> None:
        """Merge fragments by ``index``, as the reference does.

        A fragment carrying a NAME starts that call over, arguments included.
        The id is the first non-empty one seen. A fragment that is not text
        spoils the whole value (see ``ToolCall.append_argument_fragment``).
        """
        calls = delta.get("tool_calls")
        if not isinstance(calls, list):
            return

        for position, raw in enumerate(calls):
            if not isinstance(raw, dict):
                continue

            index = raw.get("index", position)
            if not isinstance(index, int) or isinstance(index, bool):
                index = position

            current = self._tool_calls.get(index, {"id": "", "name": "", "args": ""})
            name = _string(data_get(raw, "function.name"))
            if name:
                current = {**current, "name": name, "args": ""}

            current["id"] = current["id"] or _string(raw.get("id"))
            current["args"] = ToolCall.append_argument_fragment(
                current["args"], data_get(raw, "function.arguments")
            )
            self._tool_calls[index] = current


def _string(value: Any) -> str:
    return value if isinstance(value, str) else ""
