"""One input's moderation verdict."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["ModerationResult"]


@dataclass(frozen=True)
class ModerationResult:
    """``flagged`` is the answer; the categories are why.

    All three are kept because a caller that acts on the boolean alone cannot
    explain the decision to the person it was made about, and "your message was
    blocked" with no reason is the worst version of this feature.
    """

    flagged: bool = False
    categories: dict[str, bool] = field(default_factory=dict)
    category_scores: dict[str, float] = field(default_factory=dict)

    @staticmethod
    def from_dict(data: Any) -> ModerationResult:
        source = data if isinstance(data, dict) else {}

        return ModerationResult(
            # `is True` rather than truthiness: a provider that omitted the
            # field has not told us the content is safe, but reporting True on
            # absence would block everything a malformed reply touched.
            flagged=source.get("flagged") is True,
            categories=_booleans(source.get("categories")),
            category_scores=_numbers(source.get("category_scores")),
        )

    def flagged_categories(self) -> list[str]:
        """The categories that tripped, without the ones that did not."""
        return [name for name, tripped in self.categories.items() if tripped]

    def to_dict(self) -> dict[str, Any]:
        return {
            "flagged": self.flagged,
            "categories": dict(self.categories),
            "category_scores": dict(self.category_scores),
        }


def _booleans(value: Any) -> dict[str, bool]:
    """Non-boolean members are DROPPED, not coerced.

    A category whose value arrived as a string would become ``True`` under
    coercion -- including the string ``"false"`` -- and this is the one value
    object in the port where a wrong ``True`` means content gets blocked.
    """
    source = value if isinstance(value, dict) else {}

    return {key: entry for key, entry in source.items() if isinstance(entry, bool)}


def _numbers(value: Any) -> dict[str, float]:
    source = value if isinstance(value, dict) else {}

    return {
        key: float(entry)
        for key, entry in source.items()
        if isinstance(entry, (int, float)) and not isinstance(entry, bool)
    }
