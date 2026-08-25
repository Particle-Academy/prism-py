"""A call the provider made to one of its own tools."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["ProviderToolCall"]


@dataclass
class ProviderToolCall:
    """A provider-executed tool call, reported back on the response."""

    id: str
    type: str
    status: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "status": self.status,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProviderToolCall:
        return cls(
            id=data["id"],
            type=data["type"],
            status=data["status"],
            data=dict(data.get("data") or {}),
        )
