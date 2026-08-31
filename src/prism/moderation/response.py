"""What the moderation endpoint said."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from prism.value_objects.meta import Meta
from prism.value_objects.moderation_result import ModerationResult

__all__ = ["ModerationResponse"]


@dataclass(frozen=True)
class ModerationResponse:
    results: tuple[ModerationResult, ...] = ()
    meta: Meta = field(default_factory=lambda: Meta(id="", model=""))
    raw: dict[str, Any] | None = None

    def is_flagged(self) -> bool:
        """Whether ANY input was flagged.

        The question nearly every caller actually asks, and the one most likely
        to be got wrong by hand: a caller checking ``results[0]`` alone passes a
        batch whose second input was the problem.
        """
        return any(result.flagged for result in self.results)

    def first_flagged(self) -> ModerationResult | None:
        return next((result for result in self.results if result.flagged), None)

    def flagged(self) -> tuple[ModerationResult, ...]:
        return tuple(result for result in self.results if result.flagged)

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [result.to_dict() for result in self.results],
            "meta": self.meta.to_dict(),
            "raw": None if self.raw is None else dict(self.raw),
        }
