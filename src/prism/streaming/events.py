"""What a stream emits."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from itertools import count
from typing import Any

from prism.enums import FinishReason
from prism.value_objects.tool_call import ToolCall
from prism.value_objects.usage import Usage

__all__ = [
    "ErrorEvent",
    "StreamEndEvent",
    "StreamEvent",
    "StreamEventType",
    "StreamStartEvent",
    "TextCompleteEvent",
    "TextDeltaEvent",
    "TextStartEvent",
    "ToolCallEvent",
    "event_id",
]


class StreamEventType(str, Enum):
    """The events this port can emit.

    The reference defines twenty types; this lists the seven OpenAI actually
    produces through the implemented path. A member nothing can ever emit reads
    to a consumer as a case they must handle, and writing a branch for an event
    that never arrives is worse than not knowing it exists. They are added when
    a provider can emit them.
    """

    STREAM_START = "stream_start"
    TEXT_START = "text_start"
    TEXT_DELTA = "text_delta"
    TEXT_COMPLETE = "text_complete"
    TOOL_CALL = "tool_call"
    ERROR = "error"
    STREAM_END = "stream_end"


_counter = count(1)


def event_id(prefix: str = "evt") -> str:
    """Unique within a process, and cheap. Streams correlate by run, not by this."""
    return f"{prefix}_{int(time.time() * 1000):x}{next(_counter):x}"


@dataclass(frozen=True)
class StreamEvent:
    """The base every event shares."""

    id: str = field(default_factory=event_id)
    timestamp: int = field(default_factory=lambda: int(time.time()))

    def type(self) -> StreamEventType:  # pragma: no cover - overridden
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "timestamp": self.timestamp, "type": self.type().value}


@dataclass(frozen=True)
class StreamStartEvent(StreamEvent):
    model: str = ""

    def type(self) -> StreamEventType:
        return StreamEventType.STREAM_START

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "model": self.model}


@dataclass(frozen=True)
class TextStartEvent(StreamEvent):
    message_id: str = ""

    def type(self) -> StreamEventType:
        return StreamEventType.TEXT_START

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "message_id": self.message_id}


@dataclass(frozen=True)
class TextDeltaEvent(StreamEvent):
    """The one a consumer renders. Everything else is bookkeeping around it."""

    delta: str = ""
    message_id: str = ""

    def type(self) -> StreamEventType:
        return StreamEventType.TEXT_DELTA

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "delta": self.delta, "message_id": self.message_id}


@dataclass(frozen=True)
class TextCompleteEvent(StreamEvent):
    text: str = ""
    message_id: str = ""

    def type(self) -> StreamEventType:
        return StreamEventType.TEXT_COMPLETE

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "text": self.text, "message_id": self.message_id}


@dataclass(frozen=True)
class ToolCallEvent(StreamEvent):
    tool_call: ToolCall | None = None
    message_id: str = ""

    def type(self) -> StreamEventType:
        return StreamEventType.TOOL_CALL

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "tool_call": None if self.tool_call is None else self.tool_call.to_dict(),
            "message_id": self.message_id,
        }


@dataclass(frozen=True)
class ErrorEvent(StreamEvent):
    """The provider reported a failure MID-STREAM.

    Emitted rather than raised, and the distinction is the reason this type
    exists: by the time it arrives the consumer has already rendered text, and
    raising would discard a partial answer the user watched appear. The stream
    ends after it; the caller decides what the partial answer was worth.
    """

    code: str = "unknown_error"
    message: str = ""

    def type(self) -> StreamEventType:
        return StreamEventType.ERROR

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "code": self.code, "message": self.message}


@dataclass(frozen=True)
class StreamEndEvent(StreamEvent):
    finish_reason: FinishReason = FinishReason.UNKNOWN
    usage: Usage | None = None

    def type(self) -> StreamEventType:
        return StreamEventType.STREAM_END

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "finish_reason": self.finish_reason.value,
            "usage": None if self.usage is None else self.usage.to_dict(),
        }
