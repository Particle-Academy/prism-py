"""Provider metadata attached to a response."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from prism.value_objects.provider_rate_limit import ProviderRateLimit

__all__ = ["Meta"]


@dataclass
class Meta:
    """What the provider said about the response itself.

    ``service_tier`` is a field the provider may omit; it becomes an explicit
    ``None`` rather than disappearing, which is the mirror of the request side's
    always-present ``max_output_tokens``.

    ``rate_limits`` is what the provider's response HEADERS said about quota,
    parsed per provider. It used to be a list of anything, carried opaquely and
    always empty, which is why the ports disagreed about rate limits for a
    release without anything failing (G-15).
    """

    id: str
    model: str
    rate_limits: list[ProviderRateLimit] = field(default_factory=list)
    service_tier: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model": self.model,
            "rate_limits": [limit.to_dict() for limit in self.rate_limits],
            "service_tier": self.service_tier,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Meta:
        return cls(
            id=data["id"],
            model=data["model"],
            rate_limits=[
                limit
                if isinstance(limit, ProviderRateLimit)
                else ProviderRateLimit.from_dict(limit)
                for limit in (data.get("rate_limits") or [])
            ],
            service_tier=data.get("service_tier"),
        )
