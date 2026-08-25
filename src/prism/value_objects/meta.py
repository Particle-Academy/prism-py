"""Provider metadata attached to a response."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from prism.value_objects._opaque import opaque_list

__all__ = ["Meta"]


@dataclass
class Meta:
    """What the provider said about the response itself.

    ``service_tier`` is a field the provider may omit; it becomes an explicit
    ``None`` rather than disappearing, which is the mirror of the request side's
    always-present ``max_output_tokens``.

    ``rate_limits`` comes from response headers, which are outside this slice —
    it is present, empty, and carried opaquely.
    """

    id: str
    model: str
    rate_limits: list[Any] = field(default_factory=list)
    service_tier: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model": self.model,
            "rate_limits": opaque_list(self.rate_limits),
            "service_tier": self.service_tier,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Meta:
        return cls(
            id=data["id"],
            model=data["model"],
            rate_limits=list(data.get("rate_limits") or []),
            service_tier=data.get("service_tier"),
        )
