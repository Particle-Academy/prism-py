"""OpenAI's two audio endpoints: /audio/speech and /audio/transcriptions."""

from __future__ import annotations

import base64 as _base64
from typing import Any

from prism.audio.request import SpeechToTextRequest, TextToSpeechRequest
from prism.audio.response import AudioResponse, AudioTextResponse
from prism.errors import ErrorCode, PrismError
from prism.http import MultipartBody, MultipartFile
from prism.value_objects.generated_audio import GeneratedAudio
from prism.value_objects.usage import Usage

__all__ = [
    "build_speech_body",
    "build_transcription_form",
    "parse_speech_response",
    "parse_transcription_response",
]

_SPEECH_OPTIONS = ("response_format", "speed", "instructions")
_TRANSCRIPTION_OPTIONS = ("language", "prompt", "response_format", "temperature")

_EXTENSIONS = {
    "audio/mpeg": "mp3",
    "audio/mp4": "m4a",
    "audio/wav": "wav",
    "audio/webm": "webm",
    "audio/ogg": "ogg",
    "audio/flac": "flac",
}


def build_speech_body(request: TextToSpeechRequest) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": request.model,
        "input": request.input,
        # `alloy` is OpenAI's own default and the endpoint REQUIRES a voice, so
        # omitting it fails the call. Defaulted here rather than in the builder,
        # so a provider with different voices can choose its own.
        "voice": request.voice or "alloy",
    }

    for key in _SPEECH_OPTIONS:
        value = request.provider_options.get(key)
        if value is not None:
            body[key] = value

    return body


def parse_speech_response(
    content: bytes,
    content_type: str | None,
    request: TextToSpeechRequest,
) -> AudioResponse:
    """Speech comes back as BYTES, not JSON.

    So the format has to come from the request or the content type -- the
    payload itself carries no field naming it, and guessing ``mp3`` would
    mislabel every caller who asked for ``wav``.
    """
    requested = request.provider_options.get("response_format")
    audio_type = requested if isinstance(requested, str) else "mp3"

    return AudioResponse(
        audio=GeneratedAudio(
            base64=_base64.b64encode(content).decode("ascii"),
            type=audio_type,
            mime_type=content_type or f"audio/{audio_type}",
        )
    )


def build_transcription_form(request: SpeechToTextRequest) -> MultipartBody:
    audio = request.input
    content = audio.raw_content()

    if content is None:
        # The payload holds only a url or a file id, and this port does not
        # fetch implicitly. An empty upload comes back as a transcript of
        # silence, which looks like a working call.
        raise PrismError(
            ErrorCode.NO_AUDIO_CONTENT,
            "The audio payload has no content to upload. Give it bytes, base64, or a local path.",
        )

    fields = {"model": request.model}

    for key in _TRANSCRIPTION_OPTIONS:
        value = request.provider_options.get(key)
        if value is not None:
            # Multipart carries strings; a number is spelled out here rather
            # than left for an encoder to stringify however it likes.
            fields[key] = str(value)

    mime_type = audio.mime_type()

    return MultipartBody(
        fields=fields,
        files=(
            MultipartFile(
                field="file",
                # A filename is REQUIRED: OpenAI infers the audio format from
                # the extension, and an unnamed part is rejected as an
                # unsupported format rather than as a missing name.
                filename=audio.filename() or f"audio.{_extension_for(mime_type)}",
                content=content,
                content_type=mime_type,
            ),
        ),
    )


def parse_transcription_response(raw_body: Any) -> AudioTextResponse:
    if not isinstance(raw_body, dict):
        raise PrismError.provider_response_error(
            "OpenAI returned an empty or non-object transcription response."
        )

    usage = raw_body.get("usage")
    text = raw_body.get("text")

    return AudioTextResponse(
        text=text if isinstance(text, str) else "",
        # None, not zero. Transcription is billed by audio duration on most
        # providers and they report no tokens at all; zero would claim it was
        # free.
        usage=None
        if not isinstance(usage, dict)
        else Usage(
            prompt_tokens=_number(usage.get("input_tokens", usage.get("prompt_tokens"))),
            completion_tokens=_number(usage.get("output_tokens")),
        ),
        additional_content=raw_body,
    )


def _extension_for(mime_type: str | None) -> str:
    return "mp3" if mime_type is None else _EXTENSIONS.get(mime_type, "mp3")


def _number(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0
