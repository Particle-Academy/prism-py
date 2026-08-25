"""The provider contract.

Every capability has a default that refuses. A provider implements the ones it
supports and inherits a clear, coded failure for the rest, so calling an
unsupported capability is a named error rather than a missing attribute.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NoReturn

from prism.errors import PrismError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from prism.text.request import Request
    from prism.text.response import Response

__all__ = ["Provider"]


class Provider:
    """Base class for every provider."""

    def text(self, request: Request) -> Response:
        self._unsupported("text")

    def structured(self, request: Any) -> Any:
        self._unsupported("structured")

    def embeddings(self, request: Any) -> Any:
        self._unsupported("embeddings")

    def images(self, request: Any) -> Any:
        self._unsupported("images")

    def moderation(self, request: Any) -> Any:
        self._unsupported("moderation")

    def text_to_speech(self, request: Any) -> Any:
        self._unsupported("text_to_speech")

    def speech_to_text(self, request: Any) -> Any:
        self._unsupported("speech_to_text")

    def stream(self, request: Any) -> Any:
        self._unsupported("stream")

    def _unsupported(self, action: str) -> NoReturn:
        raise PrismError.unsupported_provider_action(action, type(self).__name__)
