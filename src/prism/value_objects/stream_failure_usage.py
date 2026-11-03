"""Reported usage for one failed attempt, with no mutable aliases to ordinary Usage."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from prism.value_objects.usage import Usage

_USAGE_FIELDS = frozenset(
    {
        "prompt_tokens",
        "completion_tokens",
        "cache_write_input_tokens",
        "cache_read_input_tokens",
        "thought_tokens",
        "cost",
        "cache_write_5m_input_tokens",
        "cache_write_1h_input_tokens",
    }
)

_UsageValues = tuple[
    int, int, int | None, int | None, int | None, float | None, int | None, int | None
]


@dataclass(frozen=True, init=False)
class StreamFailureUsage:
    """A provisional report excludes any previously completed steps."""

    _usage_values: _UsageValues
    step_index: int
    reported_fields: tuple[str, ...]
    status: ClassVar[Literal["provisional"]] = "provisional"
    scope: ClassVar[Literal["attempt"]] = "attempt"
    includes_completed_steps: ClassVar[Literal[False]] = False

    def __init__(self, usage: Usage, step_index: int, reported_fields: Sequence[str]) -> None:
        if (
            type(step_index) is not int
            or step_index < 0
            or isinstance(reported_fields, (str, bytes))
            or not reported_fields
            or any(
                not isinstance(field, str) or field not in _USAGE_FIELDS
                for field in reported_fields
            )
            or len(set(reported_fields)) != len(reported_fields)
        ):
            raise ValueError("Invalid stream failure usage fields.")
        values: _UsageValues = (
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.cache_write_input_tokens,
            usage.cache_read_input_tokens,
            usage.thought_tokens,
            usage.cost,
            usage.cache_write_5m_input_tokens,
            usage.cache_write_1h_input_tokens,
        )
        object.__setattr__(self, "_usage_values", values)
        object.__setattr__(self, "step_index", step_index)
        object.__setattr__(self, "reported_fields", tuple(reported_fields))

    @property
    def usage(self) -> Usage:
        """Return a copy; the pre-existing mutable Usage class remains unchanged."""
        return Usage(*self._usage_values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "usage": self.usage.to_dict(),
            "status": self.status,
            "scope": self.scope,
            "step_index": self.step_index,
            "reported_fields": list(self.reported_fields),
            "includes_completed_steps": self.includes_completed_steps,
        }
