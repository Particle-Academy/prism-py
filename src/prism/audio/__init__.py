"""Audio: text spoken aloud, and speech written down."""

from __future__ import annotations

from prism.audio.pending_request import AudioPendingRequest
from prism.audio.request import SpeechToTextRequest, TextToSpeechRequest
from prism.audio.response import AudioResponse, AudioTextResponse

__all__ = [
    "AudioPendingRequest",
    "AudioResponse",
    "AudioTextResponse",
    "SpeechToTextRequest",
    "TextToSpeechRequest",
]
