"""Anthropic's stream, which needs MEMORY where OpenAI's does not."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from prism._php import data_get
from prism.enums import FinishReason
from prism.providers.anthropic.maps import map_finish_reason
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
from prism.value_objects.tool_call import ToolCall
from prism.value_objects.usage import Usage

__all__ = ["AnthropicStreamMapper"]


@dataclass
class _Block:
    type: str
    text: str = ""
    id: str = ""
    name: str = ""


class AnthropicStreamMapper:
    """Maps Anthropic's SSE payloads, carrying the state they assume.

    OpenAI repeats the identifiers on every event, so a payload can be mapped on
    its own and its mapper is a pure function. Anthropic does not: a
    ``content_block_delta`` carries only an INDEX, the message id arrived back
    at ``message_start``, the stop reason arrives at ``message_delta`` and the
    stream ends at ``message_stop``. Mapping one payload in isolation is
    impossible.

    So this is an object with state, and that difference is the provider's, not
    a design choice worth hiding. The alternative -- emitting events with empty
    ids, or an end event with no reason -- would keep the shape uniform by
    making the contents wrong.
    """

    def __init__(self) -> None:
        self._message_id = ""
        self._stop: dict[str, Any] = {}
        self._usage: Usage | None = None
        self._blocks: dict[int, _Block] = {}

    def map(self, payload: dict[str, Any]) -> StreamEvent | None:
        kind = _str(payload.get("type"))

        if kind == "message_start":
            return self._message_start(payload)
        if kind == "content_block_start":
            return self._block_start(payload)
        if kind == "content_block_delta":
            return self._block_delta(payload)
        if kind == "content_block_stop":
            return self._block_stop(payload)
        if kind == "message_delta":
            self._message_delta(payload)
            return None
        if kind == "message_stop":
            return StreamEndEvent(finish_reason=self._finish_reason(), usage=self._usage)
        if kind == "error":
            return ErrorEvent(
                code=_str(data_get(payload, "error.type")) or "unknown_error",
                message=_str(data_get(payload, "error.message"))
                or "The provider reported an error mid-stream.",
            )

        # `ping`, and anything Anthropic adds later. Ignored rather than fatal,
        # for the same reason the OpenAI mapper ignores what it does not know:
        # a provider's additive change must not become an outage.
        return None

    def _message_start(self, payload: dict[str, Any]) -> StreamEvent:
        self._message_id = _str(data_get(payload, "message.id"))
        usage = data_get(payload, "message.usage")

        if isinstance(usage, dict):
            self._usage = Usage(
                prompt_tokens=_int(usage.get("input_tokens")),
                completion_tokens=_int(usage.get("output_tokens")),
            )

        return StreamStartEvent(model=_str(data_get(payload, "message.model")))

    def _block_start(self, payload: dict[str, Any]) -> StreamEvent | None:
        index = _int(payload.get("index"))
        block = payload.get("content_block")
        block = block if isinstance(block, dict) else {}
        kind = _str(block.get("type"))

        self._blocks[index] = _Block(
            type=kind,
            id=_str(block.get("id")),
            name=_str(block.get("name")),
        )

        # A text block opening is worth announcing; a tool block opening is not,
        # because the tool call is only meaningful once its arguments arrive.
        return TextStartEvent(message_id=self._message_id) if kind == "text" else None

    def _block_delta(self, payload: dict[str, Any]) -> StreamEvent | None:
        block = self._blocks.get(_int(payload.get("index")))
        delta = payload.get("delta")

        if block is None or not isinstance(delta, dict):
            return None

        # Text and tool arguments arrive through the same event with different
        # delta types. Both accumulate; only text is worth emitting per chunk,
        # because a half-parsed JSON fragment is not something a caller can use.
        if _str(delta.get("type")) == "text_delta":
            text = _str(delta.get("text"))
            block.text += text
            return TextDeltaEvent(delta=text, message_id=self._message_id)

        if _str(delta.get("type")) == "input_json_delta":
            block.text += _str(delta.get("partial_json"))

        return None

    def _block_stop(self, payload: dict[str, Any]) -> StreamEvent | None:
        block = self._blocks.get(_int(payload.get("index")))

        if block is None:
            return None

        # Truthful because it was accumulated. Anthropic's `content_block_stop`
        # carries no text of its own, so emitting a complete event without the
        # memory above would mean emitting an empty one.
        if block.type == "text":
            return TextCompleteEvent(text=block.text, message_id=self._message_id)

        return ToolCallEvent(
            tool_call=ToolCall(id=block.id, name=block.name, arguments=block.text),
            message_id=self._message_id,
        )

    def _message_delta(self, payload: dict[str, Any]) -> None:
        delta = payload.get("delta")

        if isinstance(delta, dict):
            self._stop = delta

        # Anthropic reports output tokens here rather than at message_stop, and
        # reports them CUMULATIVELY, so the last one wins rather than summing.
        usage = payload.get("usage")

        if isinstance(usage, dict):
            self._usage = Usage(
                prompt_tokens=self._usage.prompt_tokens if self._usage else 0,
                completion_tokens=_int(usage.get("output_tokens")),
            )

    def _finish_reason(self) -> FinishReason:
        if "stop_reason" not in self._stop:
            return FinishReason.STOP

        return map_finish_reason(self._stop)


def _str(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _int(value: Any) -> int:
    return value if isinstance(value, int) else 0
