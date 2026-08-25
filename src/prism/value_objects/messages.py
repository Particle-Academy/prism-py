"""The four message kinds a text conversation is made of."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from prism.errors import PrismError
from prism.value_objects._opaque import opaque_list
from prism.value_objects.media import Text
from prism.value_objects.tool_call import ToolCall
from prism.value_objects.tool_result import ToolResult

__all__ = [
    "AssistantMessage",
    "Message",
    "SystemMessage",
    "ToolResultMessage",
    "UserMessage",
    "message_from_dict",
]


class Message(ABC):
    """A turn in a conversation.

    ``TYPE`` is the discriminator that appears in the serialised form and the
    key :func:`message_from_dict` dispatches on.
    """

    TYPE: ClassVar[str]

    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Message: ...


@dataclass
class UserMessage(Message):
    """A turn from the user.

    The constructor APPENDS a :class:`Text` part built from ``content`` to
    ``additional_content``, mirroring the reference — which means the stored
    ``additional_content`` already contains that part. :meth:`from_dict` drops
    it again before rebuilding. Handing the stored list straight back to the
    constructor would append a SECOND copy and the message text would double on
    every save-and-load cycle, silently, with nothing to catch it but a
    conversation that grows.
    """

    TYPE: ClassVar[str] = "user"

    content: str
    additional_content: list[Text] = field(default_factory=list)
    additional_attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # A new list rather than an in-place append: the caller's list is theirs.
        self.additional_content = [*self.additional_content, Text(self.content)]

    def text(self) -> str:
        """Every text part, concatenated — the text the provider is sent."""
        return "".join(part.text for part in self.additional_content if isinstance(part, Text))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.TYPE,
            "content": self.content,
            "additional_content": [part.to_dict() for part in self.additional_content],
            "additional_attributes": self.additional_attributes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> UserMessage:
        content: str = data["content"]
        parts = [Text.from_dict(part) for part in data.get("additional_content") or []]

        # Drop the part the constructor will re-append. It is always last and
        # always equal to the content.
        if parts and parts[-1].text == content:
            parts = parts[:-1]

        return cls(
            content=content,
            additional_content=parts,
            additional_attributes=dict(data.get("additional_attributes") or {}),
        )


@dataclass
class AssistantMessage(Message):
    """A turn from the model, with any tool calls it asked for."""

    TYPE: ClassVar[str] = "assistant"

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    additional_content: dict[str, Any] = field(default_factory=dict)
    tool_approval_requests: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.TYPE,
            "content": self.content,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "additional_content": self.additional_content,
            "tool_approval_requests": opaque_list(self.tool_approval_requests),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AssistantMessage:
        return cls(
            content=data["content"],
            tool_calls=[ToolCall.from_dict(call) for call in data.get("tool_calls") or []],
            additional_content=dict(data.get("additional_content") or {}),
            tool_approval_requests=list(data.get("tool_approval_requests") or []),
        )


@dataclass
class SystemMessage(Message):
    """A system instruction. Keeps its discriminator; never collapses to a bare string."""

    TYPE: ClassVar[str] = "system"

    content: str

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.TYPE, "content": self.content}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SystemMessage:
        return cls(content=data["content"])


@dataclass
class ToolResultMessage(Message):
    """The results of the tools the previous assistant turn asked for."""

    TYPE: ClassVar[str] = "tool_result"

    tool_results: list[ToolResult] = field(default_factory=list)
    tool_approval_responses: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.TYPE,
            "tool_results": [result.to_dict() for result in self.tool_results],
            "tool_approval_responses": opaque_list(self.tool_approval_responses),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ToolResultMessage:
        return cls(
            tool_results=[ToolResult.from_dict(r) for r in data.get("tool_results") or []],
            tool_approval_responses=list(data.get("tool_approval_responses") or []),
        )


_MESSAGE_TYPES: dict[str, type[Message]] = {
    UserMessage.TYPE: UserMessage,
    AssistantMessage.TYPE: AssistantMessage,
    SystemMessage.TYPE: SystemMessage,
    ToolResultMessage.TYPE: ToolResultMessage,
}


def message_from_dict(data: Mapping[str, Any]) -> Message:
    """Rebuild whichever message the stored ``type`` names.

    :raises PrismError: with code ``unknown_message_type`` for anything else.
    """
    message_type = data.get("type")
    factory = _MESSAGE_TYPES.get(message_type) if isinstance(message_type, str) else None

    if factory is None:
        raise PrismError.unknown_message_type(repr(message_type))

    return factory.from_dict(data)
