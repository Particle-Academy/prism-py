"""The entry point."""

from __future__ import annotations

from prism.embeddings.pending_request import EmbeddingsPendingRequest
from prism.structured.pending_request import StructuredPendingRequest
from prism.text.pending_request import PendingRequest

__all__ = ["Prism"]


class Prism:
    """Where a request starts.

    >>> from prism import Prism
    >>> pending = Prism.text().using("openai", "gpt-4o").with_prompt("Who are you?")
    >>> pending.to_request().model
    'gpt-4o'
    """

    @staticmethod
    def text() -> PendingRequest:
        """Begin a text generation."""
        return PendingRequest()

    @staticmethod
    def structured() -> StructuredPendingRequest:
        """Begin a generation that must come back shaped."""
        return StructuredPendingRequest()

    @staticmethod
    def embeddings() -> EmbeddingsPendingRequest:
        """Begin an embeddings request."""
        return EmbeddingsPendingRequest()
