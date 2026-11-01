"""The answer, and the answer parsed."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from prism.enums import FinishReason
from prism.structured.step import StructuredStep
from prism.value_objects import ProviderToolCall, ToolCall, ToolResult
from prism.value_objects.meta import Meta
from prism.value_objects.usage import Usage

__all__ = ["StructuredResponse"]


@dataclass(frozen=True)
class StructuredResponse:
    """The parsed result and its native conversation state.

    OpenAI/Mistral retain JSON in text. Anthropic native format returns empty
    text; its original JSON remains in the assistant message and raw response.
    Check the optional structured value before reading fields from it.
    """

    steps: tuple[Any, ...]
    text: str
    structured: dict[str, Any] | None
    finish_reason: FinishReason
    usage: Usage
    meta: Meta
    additional_content: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    provider_tool_calls: tuple[ProviderToolCall, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "steps",
            tuple(
                step if isinstance(step, StructuredStep) else StructuredStep.from_step(step)
                for step in self.steps
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [step.to_dict() for step in self.steps],
            "text": self.text,
            "structured": None if self.structured is None else dict(self.structured),
            "finish_reason": self.finish_reason.value,
            "usage": self.usage.to_dict(),
            "meta": self.meta.to_dict(),
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "provider_tool_calls": [call.to_dict() for call in self.provider_tool_calls],
            "tool_results": [result.to_dict() for result in self.tool_results],
            "additional_content": dict(self.additional_content),
            "raw": None if self.raw is None else dict(self.raw),
        }
