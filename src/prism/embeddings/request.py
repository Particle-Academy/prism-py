"""A frozen embeddings request."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["EmbeddingsRequest"]


@dataclass
class EmbeddingsRequest:
    """Everything a provider needs to embed some text.

    ``inputs`` is ALWAYS a list, even for one input. The reference accumulates
    every ``from_input`` / ``from_array`` / ``from_file`` call into one list, so
    a single input is a list of one rather than a special case -- and keeping
    the shape constant means a response index maps to an input index without a
    branch in every provider.
    """

    model: str = ""
    provider_key: str = ""
    inputs: list[str] = field(default_factory=list)
    client_options: dict[str, Any] = field(default_factory=dict)
    provider_options: dict[str, Any] = field(default_factory=dict)
