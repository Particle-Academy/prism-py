"""One provider round-trip."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from prism.enums import FinishReason
from prism.value_objects import (
    Message,
    Meta,
    ProviderToolCall,
    SystemMessage,
    ToolCall,
    ToolResult,
    Usage,
)
from prism.value_objects._opaque import opaque_list

__all__ = ["Step"]


@dataclass
class Step:
    """A single request/response exchange with the provider.

    A response is made of one or more of these; this slice only ever produces
    one, because the tool-execution loop that produces more is out of scope.
    """

    text: str
    finish_reason: FinishReason
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    provider_tool_calls: list[ProviderToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=lambda: Usage(0, 0))
    meta: Meta = field(default_factory=lambda: Meta("", ""))
    messages: list[Message] = field(default_factory=list)
    system_prompts: list[SystemMessage] = field(default_factory=list)
    additional_content: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] | None = None
    tool_approval_requests: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "finish_reason": self.finish_reason.value,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "tool_results": [result.to_dict() for result in self.tool_results],
            "provider_tool_calls": [call.to_dict() for call in self.provider_tool_calls],
            "usage": self.usage.to_dict(),
            "meta": self.meta.to_dict(),
            "messages": [message.to_dict() for message in self.messages],
            "system_prompts": [prompt.to_dict() for prompt in self.system_prompts],
            "additional_content": self.additional_content,
            "raw": self.raw,
            "tool_approval_requests": opaque_list(self.tool_approval_requests),
        }
