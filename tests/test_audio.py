"""Audio: text spoken aloud, and speech written down."""

from __future__ import annotations

import base64
import json

import pytest

from prism import HttpRequest, HttpResponse, Prism, PrismError, canonical
from prism.http import MultipartBody, encode_multipart
from prism.value_objects.media_file import Audio


class RecordingTransport:
    """Hands back bytes, because that is what these two endpoints trade in."""

    def __init__(
        self,
        content: bytes,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.content = content
        self.status = status
        self.headers = headers or {}
        self.sent: HttpRequest | None = None

    def send(self, request: HttpRequest) -> HttpResponse:
        self.sent = request
        return HttpResponse(status=self.status, body=self.content, headers=self.headers)


def _json(value: object) -> bytes:
    return canonical.encode(value).encode("utf-8")


def _sent_body(transport: RecordingTransport) -> dict[str, object]:
    assert transport.sent is not None
    assert transport.sent.body is not None
    parsed = json.loads(transport.sent.body.decode("utf-8"))
    assert isinstance(parsed, dict)
    return parsed


def _sent_form(transport: RecordingTransport) -> bytes:
    assert transport.sent is not None
    assert transport.sent.body is not None
    return transport.sent.body


# -- text to speech --------------------------------------------------------


def test_the_text_is_posted_and_the_bytes_come_back_as_audio() -> None:
    transport = RecordingTransport(b"MP3BYTES", headers={"content-type": "audio/mpeg"})

    response = (
        Prism.audio()
        .using("openai", "gpt-4o-mini-tts", {"api_key": "sk-test", "transport": transport})
        .with_input("hello there")
        .with_voice("nova")
        .as_audio()
    )

    assert transport.sent is not None
    assert transport.sent.url.endswith("/audio/speech")
    assert _sent_body(transport) == {
        "model": "gpt-4o-mini-tts",
        "input": "hello there",
        "voice": "nova",
    }
    assert response.audio.base64() == base64.b64encode(b"MP3BYTES").decode("ascii")
    assert response.audio.mime_type() == "audio/mpeg"


def test_the_voice_is_defaulted_because_the_endpoint_requires_one() -> None:
    # Omitting it fails the call, so a default beats a provider error naming a
    # field the caller never saw.
    transport = RecordingTransport(b"x")

    (
        Prism.audio()
        .using("openai", "gpt-4o-mini-tts", {"api_key": "sk-test", "transport": transport})
        .with_input("hi")
        .as_audio()
    )

    assert _sent_body(transport)["voice"] == "alloy"


def test_the_audio_is_labelled_with_the_format_that_was_asked_for() -> None:
    # The bytes carry no field naming the format, so guessing mp3 would mislabel
    # every caller who asked for wav.
    transport = RecordingTransport(b"x")

    response = (
        Prism.audio()
        .using("openai", "gpt-4o-mini-tts", {"api_key": "sk-test", "transport": transport})
        .with_input("hi")
        .with_provider_options({"response_format": "wav"})
        .as_audio()
    )

    assert response.audio.type == "wav"


def test_an_error_body_is_read_as_json_even_though_the_success_body_is_not() -> None:
    transport = RecordingTransport(_json({"error": {"message": "no such voice"}}), status=400)

    with pytest.raises(PrismError, match="no such voice"):
        (
            Prism.audio()
            .using("openai", "gpt-4o-mini-tts", {"api_key": "sk-test", "transport": transport})
            .with_input("hi")
            .as_audio()
        )


def test_speaking_an_audio_payload_is_refused() -> None:
    # There is no sensible reading of "speak this recording", and stringifying
    # it would hand the provider a repr to read aloud.
    with pytest.raises(PrismError) as error:
        (
            Prism.audio()
            .using("openai", "gpt-4o-mini-tts", {"api_key": "sk-test"})
            .with_input(Audio.from_base64("aGk="))
            .to_text_to_speech_request()
        )

    assert error.value.code == "wrong_audio_input"


# -- speech to text --------------------------------------------------------


def test_the_audio_is_uploaded_as_multipart_and_the_transcript_comes_back() -> None:
    transport = RecordingTransport(_json({"text": "hello there"}))

    response = (
        Prism.audio()
        .using("openai", "whisper-1", {"api_key": "sk-test", "transport": transport})
        .with_input(Audio.from_raw_content(b"WAVBYTES", "audio/wav"))
        .as_text()
    )

    form = _sent_form(transport)

    assert transport.sent is not None
    assert transport.sent.url.endswith("/audio/transcriptions")
    assert transport.sent.headers["Content-Type"].startswith("multipart/form-data; boundary=")
    assert b'name="model"' in form
    assert b"whisper-1" in form
    assert b"WAVBYTES" in form
    # OpenAI infers the format from the extension, so an unnamed part is
    # rejected as an unsupported format rather than as a missing name.
    assert b'filename="audio.wav"' in form
    assert response.text == "hello there"


def test_a_filename_the_caller_chose_is_kept() -> None:
    transport = RecordingTransport(_json({"text": "x"}))

    (
        Prism.audio()
        .using("openai", "whisper-1", {"api_key": "sk-test", "transport": transport})
        .with_input(Audio.from_base64("aGk=", "audio/mpeg").as_("interview.mp3"))
        .as_text()
    )

    assert b'filename="interview.mp3"' in _sent_form(transport)


def test_an_input_with_no_bytes_is_refused_rather_than_posted_empty() -> None:
    # A url payload is not fetched implicitly, so it has nothing to upload, and
    # an empty file comes back as a transcript of silence.
    with pytest.raises(PrismError) as error:
        (
            Prism.audio()
            .using("openai", "whisper-1", {"api_key": "sk-test"})
            .with_input(Audio.from_url("https://example.test/a.mp3"))
            .as_text()
        )

    assert error.value.code == "no_audio_content"


def test_no_usage_is_reported_as_none_rather_than_zero_tokens() -> None:
    # Transcription is billed by audio duration on most providers and they
    # report no tokens at all; zero would claim it was free.
    transport = RecordingTransport(_json({"text": "x"}))

    response = (
        Prism.audio()
        .using("openai", "whisper-1", {"api_key": "sk-test", "transport": transport})
        .with_input(Audio.from_base64("aGk=", "audio/wav"))
        .as_text()
    )

    assert response.usage is None


def test_usage_is_read_when_the_provider_does_report_it() -> None:
    transport = RecordingTransport(
        _json({"text": "x", "usage": {"input_tokens": 11, "output_tokens": 4}})
    )

    response = (
        Prism.audio()
        .using("openai", "whisper-1", {"api_key": "sk-test", "transport": transport})
        .with_input(Audio.from_base64("aGk=", "audio/wav"))
        .as_text()
    )

    assert response.usage is not None
    assert response.usage.prompt_tokens == 11


def test_transcribing_a_string_is_refused() -> None:
    with pytest.raises(PrismError) as error:
        (
            Prism.audio()
            .using("openai", "whisper-1", {"api_key": "sk-test"})
            .with_input("hi")
            .to_speech_to_text_request()
        )

    assert error.value.code == "wrong_audio_input"


# -- the encoder itself ----------------------------------------------------


def test_the_multipart_boundary_is_different_every_time() -> None:
    # The format has no escaping: a boundary that also appears inside a part
    # silently splits the file, and a random one per call is the only defence.
    first, _ = encode_multipart(MultipartBody(fields={"a": "b"}))
    second, _ = encode_multipart(MultipartBody(fields={"a": "b"}))

    assert first != second


def test_the_encoder_uses_crlf_because_the_format_requires_it() -> None:
    # A bare LF is tolerated by some servers and rejected by others, and the
    # failure is a rejected upload with no useful message.
    _, body = encode_multipart(MultipartBody(fields={"model": "whisper-1"}))

    assert b"\r\n" in body
    assert body.replace(b"\r\n", b"").find(b"\n") == -1
    assert body.endswith(b"--\r\n")
