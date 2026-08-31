"""Speech a provider generated."""

from __future__ import annotations

from typing import Any

from prism.value_objects.media_file import Audio

__all__ = ["GeneratedAudio"]


class GeneratedAudio(Audio):
    """EXTENDS :class:`~prism.value_objects.media_file.Audio`.

    So it answers the same questions every other payload does -- ``base64()``,
    ``raw_content()``, ``mime_type()``. It exists as its own type only because
    the reference has one, and because ``type`` (the provider's own format name,
    ``mp3`` / ``wav`` / ``opus``) is not the same thing as a mime type and
    should not be quietly stored as one.
    """

    def __init__(
        self,
        base64: str | None = None,
        type: str | None = None,
        mime_type: str | None = None,
    ) -> None:
        super().__init__(base64_data=base64, mime_type=mime_type)
        #: The provider's format name -- ``mp3``, ``wav``, ``opus``.
        self.type = type

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "type": self.type}
