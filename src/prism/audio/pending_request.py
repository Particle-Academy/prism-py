"""The fluent builder for both audio directions."""

from __future__ import annotations

from typing import Any

from prism.audio.request import SpeechToTextRequest, TextToSpeechRequest
from prism.audio.response import AudioResponse, AudioTextResponse
from prism.enums import Provider as ProviderName
from prism.errors import ErrorCode, PrismError
from prism.providers.base import Provider
from prism.registry import resolve_provider
from prism.value_objects.media_file import Audio

__all__ = ["AudioPendingRequest"]


class AudioPendingRequest:
    """One builder, two terminals.

    Matching the reference: ``with_input()`` takes either a string to speak or
    an :class:`~prism.value_objects.media_file.Audio` to transcribe, and the
    terminal method decides which request is built. The alternative -- two
    builders -- would force a caller to know the direction before they have the
    input, which is backwards from how audio work actually arrives.
    """

    def __init__(self) -> None:
        self._provider: Provider | None = None
        self._provider_key = ""
        self._model = ""
        self._input: str | Audio | None = None
        self._voice: str | None = None
        self._provider_options: dict[str, Any] = {}
        self._client_options: dict[str, Any] = {}

    def using(
        self,
        provider: str | ProviderName,
        model: str = "",
        provider_config: dict[str, Any] | None = None,
    ) -> AudioPendingRequest:
        self._provider_key = provider.value if isinstance(provider, ProviderName) else provider
        self._model = model
        self._provider = resolve_provider(self._provider_key, provider_config or {})
        return self

    def with_input(self, value: str | Audio) -> AudioPendingRequest:
        """Text to speak, or a recording to transcribe."""
        self._input = value
        return self

    def with_voice(self, voice: str) -> AudioPendingRequest:
        self._voice = voice
        return self

    def with_provider_options(self, options: dict[str, Any]) -> AudioPendingRequest:
        self._provider_options = dict(options)
        return self

    def with_client_options(self, options: dict[str, Any]) -> AudioPendingRequest:
        self._client_options = dict(options)
        return self

    def to_text_to_speech_request(self) -> TextToSpeechRequest:
        """Freeze the builder for the speaking direction.

        :raises PrismError: with code ``wrong_audio_input`` when the input is a
            recording. Refused rather than coerced: there is no sensible reading
            of "speak this recording", and stringifying it would hand a provider
            something like ``<prism...Audio object>`` to read aloud.
        """
        if not isinstance(self._input, str):
            raise PrismError(
                ErrorCode.WRONG_AUDIO_INPUT,
                "Text to speech needs text to speak; it was given an audio payload.",
            )

        return TextToSpeechRequest(
            model=self._model,
            provider_key=self._provider_key,
            input=self._input,
            voice=self._voice,
            client_options=dict(self._client_options),
            provider_options=dict(self._provider_options),
        )

    def to_speech_to_text_request(self) -> SpeechToTextRequest:
        """:raises PrismError: code ``wrong_audio_input`` when given text."""
        if not isinstance(self._input, Audio):
            raise PrismError(
                ErrorCode.WRONG_AUDIO_INPUT,
                "Speech to text needs an audio payload; it was given text.",
            )

        return SpeechToTextRequest(
            model=self._model,
            provider_key=self._provider_key,
            input=self._input,
            client_options=dict(self._client_options),
            provider_options=dict(self._provider_options),
        )

    def as_audio(self) -> AudioResponse:
        result: AudioResponse = self._require_provider().text_to_speech(
            self.to_text_to_speech_request()
        )
        return result

    def as_text(self) -> AudioTextResponse:
        result: AudioTextResponse = self._require_provider().speech_to_text(
            self.to_speech_to_text_request()
        )
        return result

    def _require_provider(self) -> Provider:
        if self._provider is None:
            raise PrismError(
                ErrorCode.UNSUPPORTED_PROVIDER_ACTION,
                "No provider configured. Call using(<provider>, <model>) first.",
            )

        return self._provider
