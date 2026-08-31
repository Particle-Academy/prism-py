"""What a provider gave back."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from prism.value_objects.embedding import Embedding, EmbeddingsUsage
from prism.value_objects.meta import Meta

__all__ = ["EmbeddingsResponse"]


@dataclass(frozen=True)
class EmbeddingsResponse:
    embeddings: tuple[Embedding, ...] = ()
    usage: EmbeddingsUsage = field(default_factory=lambda: EmbeddingsUsage(None))
    meta: Meta = field(default_factory=lambda: Meta(id="", model=""))
    raw: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "embeddings": [embedding.to_dict() for embedding in self.embeddings],
            "usage": self.usage.to_dict(),
            "meta": self.meta.to_dict(),
            "raw": None if self.raw is None else dict(self.raw),
        }
