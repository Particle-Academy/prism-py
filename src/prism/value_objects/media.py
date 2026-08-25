"""Media parts of a user message. This slice carries text only."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

__all__ = ["Text"]


@dataclass(frozen=True)
class Text:
    """A plain-text part of a user message."""

    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Text:
        return cls(text=data["text"])
