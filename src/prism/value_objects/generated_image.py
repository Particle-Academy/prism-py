"""An image a provider generated, and what it generated from."""

from __future__ import annotations

from typing import Any

from prism.value_objects.media_file import Image

__all__ = ["GeneratedImage"]


class GeneratedImage(Image):
    """EXTENDS :class:`~prism.value_objects.media_file.Image` now that a Media
    base exists.

    It previously restated the url / base64 / mime_type triple itself, which was
    correct while it was the only binary type in the port and wrong the moment
    audio arrived -- three standalone copies is how a package ends up with three
    answers to "is this a file". This was the first, so it moved first.

    What it adds is the one thing a generated image has and a supplied one does
    not: the prompt the provider actually used.
    """

    def __init__(
        self,
        url: str | None = None,
        base64: str | None = None,
        revised_prompt: str | None = None,
        mime_type: str | None = None,
    ) -> None:
        super().__init__(url=url, base64_data=base64, mime_type=mime_type)
        #: What the provider actually generated from.
        #:
        #: OpenAI rewrites prompts for safety and quality, often substantially.
        #: Kept because a caller comparing an image against its prompt is
        #: comparing it against THIS one, not the one they typed.
        self.revised_prompt = revised_prompt

    def has_revised_prompt(self) -> bool:
        return self.revised_prompt is not None

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "revised_prompt": self.revised_prompt}
