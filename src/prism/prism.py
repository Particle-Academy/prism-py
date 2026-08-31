"""The entry point."""

from __future__ import annotations

from prism.audio.pending_request import AudioPendingRequest
from prism.embeddings.pending_request import EmbeddingsPendingRequest
from prism.images.pending_request import ImagesPendingRequest
from prism.moderation.pending_request import ModerationPendingRequest
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

    @staticmethod
    def images() -> ImagesPendingRequest:
        """Begin an image generation."""
        return ImagesPendingRequest()

    @staticmethod
    def audio() -> AudioPendingRequest:
        """Begin an audio request, in either direction."""
        return AudioPendingRequest()

    @staticmethod
    def moderation() -> ModerationPendingRequest:
        """Begin a moderation check."""
        return ModerationPendingRequest()
