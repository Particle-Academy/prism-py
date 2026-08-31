"""Map one OpenAI Responses SSE payload to zero or one port events."""

from __future__ import annotations

from typing import Any

from prism._php import data_get
from prism.enums import FinishReason
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

__all__ = ["map_stream_event"]


def map_stream_event(payload: dict[str, Any]) -> StreamEvent | None:
    """Return the event this payload means, or ``None``.

    RETURNS NONE FOR ANYTHING IT DOES NOT RECOGNISE, on purpose. OpenAI adds
    event types without warning, and a mapper that raised on an unknown
    ``type`` would turn a provider's additive change into an outage for every
    consumer. A stream that silently ignores an event it cannot use still
    delivers the text; one that raises delivers nothing.
    """
    kind = _str(payload.get("type"))

    if kind == "response.created":
        model = _str(data_get(payload, "response.model")) or _str(payload.get("model"))
        return StreamStartEvent(model=model)

    if kind == "response.output_item.added":
        message_id = _str(payload.get("item_id")) or _str(payload.get("id"))
        return TextStartEvent(message_id=message_id)

    if kind == "response.output_text.delta":
        return TextDeltaEvent(
            delta=_str(payload.get("delta")),
            message_id=_str(payload.get("item_id")),
        )

    if kind == "response.output_text.done":
        return TextCompleteEvent(
            text=_str(payload.get("text")),
            message_id=_str(payload.get("item_id")),
        )

    if kind == "response.output_item.done":
        return _tool_call(payload)

    if kind in ("error", "response.failed"):
        return ErrorEvent(
            code=_str(data_get(payload, "error.code")) or "unknown_error",
            message=_str(data_get(payload, "error.message"))
            or "The provider reported an error mid-stream.",
        )

    if kind in ("response.completed", "response.incomplete"):
        return StreamEndEvent(
            finish_reason=_finish_reason(payload),
            usage=_usage(payload),
        )

    return None


def _tool_call(payload: dict[str, Any]) -> StreamEvent | None:
    item = payload.get("item")

    # Only function calls become tool-call events. `output_item.done` also
    # closes ordinary message items, and treating those as tool calls would
    # invent one per assistant turn.
    if not isinstance(item, dict) or item.get("type") != "function_call":
        return None

    return ToolCallEvent(
        tool_call=ToolCall(
            id=_str(item.get("call_id")) or _str(item.get("id")),
            name=_str(item.get("name")),
            arguments=_str(item.get("arguments")),
        ),
        message_id=_str(payload.get("item_id")) or _str(item.get("id")),
    )


def _finish_reason(payload: dict[str, Any]) -> FinishReason:
    # `incomplete_details` carries its own reason; the status does not say why.
    if _str(data_get(payload, "response.incomplete_details.reason")) == "max_output_tokens":
        return FinishReason.LENGTH

    status = _str(data_get(payload, "response.status"))

    if status == "completed":
        return FinishReason.STOP
    if status == "incomplete":
        return FinishReason.LENGTH

    return FinishReason.UNKNOWN


def _usage(payload: dict[str, Any]) -> Usage | None:
    raw = data_get(payload, "response.usage")

    if not isinstance(raw, dict):
        return None

    return Usage(
        prompt_tokens=_int(raw.get("input_tokens")),
        completion_tokens=_int(raw.get("output_tokens")),
    )


def _str(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _int(value: Any) -> int:
    return value if isinstance(value, int) else 0
