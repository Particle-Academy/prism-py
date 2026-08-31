"""A frozen moderation request."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["ModerationRequest"]


@dataclass
class ModerationRequest:
    model: str = ""
    provider_key: str = ""
    #: Always a list, so a result index maps to an input index.
    inputs: list[str] = field(default_factory=list)
    client_options: dict[str, Any] = field(default_factory=dict)
    provider_options: dict[str, Any] = field(default_factory=dict)
