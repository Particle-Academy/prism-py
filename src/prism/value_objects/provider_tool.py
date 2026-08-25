"""A tool the provider runs itself."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["ProviderTool"]


@dataclass
class ProviderTool:
    """A provider-native tool, e.g. OpenAI's ``web_search_preview``.

    ``options`` is spread alongside ``type`` when the tool is mapped onto the
    wire, so anything the provider accepts on the tool object goes here.
    """

    type: str
    name: str | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "name": self.name, "options": self.options}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProviderTool:
        return cls(
            type=data["type"],
            name=data.get("name"),
            options=dict(data.get("options") or {}),
        )
