"""A frozen image-generation request."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["ImagesRequest"]


@dataclass
class ImagesRequest:
    model: str = ""
    provider_key: str = ""
    prompt: str = ""
    client_options: dict[str, Any] = field(default_factory=dict)
    provider_options: dict[str, Any] = field(default_factory=dict)
