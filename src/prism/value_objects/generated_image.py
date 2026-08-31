"""One generated image."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["GeneratedImage"]


@dataclass(frozen=True)
class GeneratedImage:
    """EITHER a url or base64, never both, and the type cannot say so.

    OpenAI returns whichever the request asked for, and a caller has to check.
    The reference models this as a ``Media`` subclass; this port has no Media
    hierarchy yet, so it stands alone rather than inheriting a base that does
    not exist. If Media arrives later, this is what moves under it.
    """

    url: str | None = None
    base64: str | None = None
    #: What the provider actually generated from.
    #:
    #: OpenAI rewrites prompts for safety and quality, often substantially. Kept
    #: because a caller comparing an image against its prompt is comparing it
    #: against THIS one, not the one they typed.
    revised_prompt: str | None = None
    mime_type: str | None = None

    def has_revised_prompt(self) -> bool:
        return self.revised_prompt is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "base64": self.base64,
            "revised_prompt": self.revised_prompt,
            "mime_type": self.mime_type,
        }
