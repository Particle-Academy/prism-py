"""Frozen audio requests, one per direction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from prism.value_objects.media_file import Audio

__all__ = ["SpeechToTextRequest", "TextToSpeechRequest"]


@dataclass
class TextToSpeechRequest:
    model: str = ""
    provider_key: str = ""
    input: str = ""
    voice: str | None = None
    client_options: dict[str, Any] = field(default_factory=dict)
    provider_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpeechToTextRequest:
    model: str = ""
    provider_key: str = ""
    #: The recording. A default is needed because the fields above have one,
    #: and an empty payload is refused at build time, not here.
    input: Audio = field(default_factory=Audio)
    client_options: dict[str, Any] = field(default_factory=dict)
    provider_options: dict[str, Any] = field(default_factory=dict)
