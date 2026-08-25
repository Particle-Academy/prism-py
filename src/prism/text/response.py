"""What the caller gets back."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from prism.enums import FinishReason
from prism.text.step import Step
from prism.value_objects import Message, Meta, ToolCall, ToolResult, Usage

__all__ = ["Response"]


@dataclass
class Response:
    """The finished text generation.

    ``messages`` is the conversation the request went in with PLUS the
    assistant turn that came back, so it can be handed straight to the next
    request.
    """

    steps: list[Step]
    text: str
    finish_reason: FinishReason
    tool_calls: list[ToolCall]
    tool_results: list[ToolResult]
    usage: Usage
    meta: Meta
    messages: list[Message]
    additional_content: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [step.to_dict() for step in self.steps],
            "text": self.text,
            "finish_reason": self.finish_reason.value,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "tool_results": [result.to_dict() for result in self.tool_results],
            "usage": self.usage.to_dict(),
            "meta": self.meta.to_dict(),
            "messages": [message.to_dict() for message in self.messages],
            "additional_content": self.additional_content,
            "raw": self.raw,
        }
