"""Vectors, and what an embeddings call cost."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["Embedding", "EmbeddingsUsage"]


@dataclass(frozen=True)
class Embedding:
    """One vector.

    A type rather than a bare ``list[float]``, matching the reference, and the
    reason survives the port: an embedding and an arbitrary list of numbers are
    the same shape and different things. A function taking ``list[float]``
    accepts a list of token counts by mistake; one taking ``Embedding`` does not.
    """

    embedding: tuple[float, ...]

    @staticmethod
    def from_list(values: list[Any]) -> Embedding:
        """Non-numeric members are DROPPED rather than coerced.

        ``float(None)`` raises and ``float("")`` raises, but a port that
        defended by coercing to 0.0 would push a zero into the vector and shift
        every distance computed against it. A provider that sent a null in a
        vector has malfunctioned, and a shorter vector is a visible fault where
        a zeroed one is not.

        ``bool`` is excluded explicitly: it is a subclass of ``int`` in Python,
        so ``True`` would otherwise become ``1.0`` in a vector.
        """
        return Embedding(
            tuple(
                float(value)
                for value in values
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {"embedding": list(self.embedding)}


@dataclass(frozen=True)
class EmbeddingsUsage:
    """What an embeddings call cost.

    Separate from :class:`~prism.value_objects.usage.Usage` rather than reusing
    it: an embeddings response has no completion tokens, and a shared type would
    report ``completion_tokens=0`` as though none had been generated rather than
    because the concept does not apply. Optional for the same reason -- a
    provider that reports nothing differs from one that reports zero.
    """

    tokens: int | None

    def to_dict(self) -> dict[str, Any]:
        return {"tokens": self.tokens}
