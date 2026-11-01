"""A structured round trip and its completed conversation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from prism.text.step import Step

__all__ = ["StructuredStep"]


@dataclass
class StructuredStep(Step):
    structured: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_step(cls, step: Step) -> StructuredStep:
        return cls(
            text=step.text,
            finish_reason=step.finish_reason,
            tool_calls=list(step.tool_calls),
            tool_results=list(step.tool_results),
            provider_tool_calls=list(step.provider_tool_calls),
            usage=step.usage,
            meta=step.meta,
            messages=list(step.messages),
            system_prompts=list(step.system_prompts),
            additional_content=dict(step.additional_content),
            raw=step.raw,
            tool_approval_requests=list(step.tool_approval_requests),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "finish_reason": self.finish_reason.value,
            "usage": self.usage.to_dict(),
            "meta": self.meta.to_dict(),
            "messages": [message.to_dict() for message in self.messages],
            "system_prompts": [prompt.to_dict() for prompt in self.system_prompts],
            "additional_content": dict(self.additional_content),
            "structured": dict(self.structured),
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "tool_results": [result.to_dict() for result in self.tool_results],
            "provider_tool_calls": [call.to_dict() for call in self.provider_tool_calls],
            "raw": self.raw,
        }
