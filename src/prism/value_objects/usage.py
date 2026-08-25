"""Token accounting."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

__all__ = ["Usage"]


@dataclass
class Usage:
    """The token counts for a request.

    Every optional counter serialises as an explicit ``null`` when unset rather
    than vanishing from the payload. This is the object an application bills
    against, and a stored row that omits unset counters rebuilds with different
    arithmetic.
    """

    prompt_tokens: int
    completion_tokens: int
    cache_write_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    thought_tokens: int | None = None
    cost: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cache_write_input_tokens": self.cache_write_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "thought_tokens": self.thought_tokens,
            "cost": self.cost,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Usage:
        return cls(
            prompt_tokens=data["prompt_tokens"],
            completion_tokens=data["completion_tokens"],
            cache_write_input_tokens=data.get("cache_write_input_tokens"),
            cache_read_input_tokens=data.get("cache_read_input_tokens"),
            thought_tokens=data.get("thought_tokens"),
            cost=data.get("cost"),
        )
