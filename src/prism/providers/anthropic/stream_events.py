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
from prism.streaming.failure_usage import StreamFailureUsageTracker
from prism.value_objects.stream_failure_usage import StreamFailureUsage
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
        self._message_stopped = False
        self._failed = False
        self._finished = False
        self._final_usage_reported = False
        self._failure = StreamFailureUsageTracker("anthropic")

    def map(self, payload: dict[str, Any]) -> StreamEvent | None:
        self._failure.record(payload)
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
            self._message_stopped = True
            return None
        if kind == "error":
            self._failed = True
            return ErrorEvent(
                code=_str(data_get(payload, "error.type")) or "unknown_error",
                message=_str(data_get(payload, "error.message"))
                or "The provider reported an error mid-stream.",
                recoverable=False,
                failure_usage=self.failure_usage(),
            )

        # `ping`, and anything Anthropic adds later. Ignored rather than fatal,
        # for the same reason the OpenAI mapper ignores what it does not know:
        # a provider's additive change must not become an outage.
        return None

    def finish(self) -> list[StreamEvent]:
        """EOF is terminal; an unfinished tool block never becomes a call."""
        if self._finished:
            return []
        self._finished = True
        events: list[StreamEvent] = []
        if not self._message_stopped and not self._failed:
            self._failed = True
            events.append(
                ErrorEvent(
                    code="incomplete_stream",
                    message="The stream ended before Anthropic finished the message.",
                    recoverable=False,
                    failure_usage=self.failure_usage(),
                )
            )
        for block in self._blocks.values():
            if block.type == "text":
                events.append(TextCompleteEvent(text=block.text, message_id=self._message_id))
        self._blocks.clear()
        events.append(
            StreamEndEvent(
                finish_reason=self._finish_reason(),
                usage=self._usage if self._final_usage_reported else None,
            )
        )
        return events

    def failure_usage(self) -> StreamFailureUsage | None:
        return self._failure.snapshot()

    def decode_failure(self) -> ErrorEvent:
        self._failed = True
        return ErrorEvent(
            code="stream_decode_error",
            message="The stream contained invalid JSON.",
            recoverable=False,
            failure_usage=self.failure_usage(),
        )

    def _message_start(self, payload: dict[str, Any]) -> StreamEvent:
        self._message_id = _str(data_get(payload, "message.id"))
        usage = data_get(payload, "message.usage")

        if isinstance(usage, dict):
            self._usage = Usage(
                prompt_tokens=_int(usage.get("input_tokens")),
                completion_tokens=_int(usage.get("output_tokens")),
                cache_write_input_tokens=usage.get("cache_creation_input_tokens"),
                cache_read_input_tokens=usage.get("cache_read_input_tokens"),
                thought_tokens=data_get(usage, "output_tokens_details.thinking_tokens"),
                cache_write_5m_input_tokens=data_get(
                    usage, "cache_creation.ephemeral_5m_input_tokens"
                ),
                cache_write_1h_input_tokens=data_get(
                    usage, "cache_creation.ephemeral_1h_input_tokens"
                ),
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

        del self._blocks[_int(payload.get("index"))]

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
            # Restated counters replace cumulative values; sparse deltas retain
            # earlier reports, including cache lifetimes from message_start.
            if isinstance(usage.get("output_tokens"), int):
                self._final_usage_reported = True
            carried = self._usage

            def counter(key: str, previous: int | None) -> int | None:
                value = data_get(usage, key)
                return previous if value is None else _int(value)

            cache_write_total = counter(
                "cache_creation_input_tokens", carried.cache_write_input_tokens if carried else None
            )
            cache_write_report = (
                carried
                if carried is not None and cache_write_total == carried.cache_write_input_tokens
                else None
            )

            self._usage = Usage(
                prompt_tokens=counter("input_tokens", carried.prompt_tokens if carried else 0) or 0,
                completion_tokens=counter(
                    "output_tokens", carried.completion_tokens if carried else 0
                )
                or 0,
                cache_write_input_tokens=cache_write_total,
                cache_read_input_tokens=counter(
                    "cache_read_input_tokens", carried.cache_read_input_tokens if carried else None
                ),
                thought_tokens=counter(
                    "output_tokens_details.thinking_tokens",
                    carried.thought_tokens if carried else None,
                ),
                cost=carried.cost if carried else None,
                cache_write_5m_input_tokens=counter(
                    "cache_creation.ephemeral_5m_input_tokens",
                    cache_write_report.cache_write_5m_input_tokens if cache_write_report else None,
                ),
                cache_write_1h_input_tokens=counter(
                    "cache_creation.ephemeral_1h_input_tokens",
                    cache_write_report.cache_write_1h_input_tokens if cache_write_report else None,
                ),
            )

    def _finish_reason(self) -> FinishReason:
        if self._failed:
            return FinishReason.ERROR

        if "stop_reason" not in self._stop:
            return FinishReason.STOP

        return map_finish_reason(self._stop)


def _str(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _int(value: Any) -> int:
    return value if isinstance(value, int) else 0
