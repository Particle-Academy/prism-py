"""What a provider gave back, in each direction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from prism.value_objects.generated_audio import GeneratedAudio
from prism.value_objects.usage import Usage

__all__ = ["AudioResponse", "AudioTextResponse"]


@dataclass(frozen=True)
class AudioResponse:
    """What ``as_audio()`` returns: speech, and whatever else the provider said."""

    audio: GeneratedAudio = field(default_factory=GeneratedAudio)
    additional_content: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "audio": self.audio.to_dict(),
            "additional_content": dict(self.additional_content),
        }


@dataclass(frozen=True)
class AudioTextResponse:
    """What ``as_text()`` returns: a transcript.

    ``usage`` is NULLABLE because transcription is billed by audio duration on
    most providers and they report no tokens at all. Zero would claim it was
    free.
    """

    text: str = ""
    usage: Usage | None = None
    additional_content: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "usage": None if self.usage is None else self.usage.to_dict(),
            "additional_content": dict(self.additional_content),
        }
