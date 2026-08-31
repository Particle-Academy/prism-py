"""The answer, and the answer parsed."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from prism.enums import FinishReason
from prism.value_objects.meta import Meta
from prism.value_objects.usage import Usage

__all__ = ["StructuredResponse"]


@dataclass(frozen=True)
class StructuredResponse:
    """What the model said, and what could be made of it.

    ``text`` and ``structured`` are BOTH kept, and the pairing is the point:
    ``text`` is what the model actually said, ``structured`` is what parsed.

    ``structured`` IS OPTIONAL, AND ``None`` IS NOT AN ERROR. A model asked for
    JSON can return prose -- a refusal, an apology, a fenced block with
    commentary around it -- and the reference reports that by leaving
    ``structured`` unset while ``text`` still carries what came back.
    Collapsing the two, or raising on unparseable output, would take away the
    one artifact that explains why it did not parse.
    """

    steps: tuple[Any, ...]
    text: str
    structured: dict[str, Any] | None
    finish_reason: FinishReason
    usage: Usage
    meta: Meta
    additional_content: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [step.to_dict() for step in self.steps],
            "text": self.text,
            "structured": None if self.structured is None else dict(self.structured),
            "finish_reason": self.finish_reason.value,
            "usage": self.usage.to_dict(),
            "meta": self.meta.to_dict(),
            "additional_content": dict(self.additional_content),
            "raw": None if self.raw is None else dict(self.raw),
        }
