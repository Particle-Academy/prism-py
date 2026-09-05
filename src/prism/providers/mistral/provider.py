"""The Mistral provider, chat-completions API."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

from prism import canonical
from prism._php import data_get
from prism.audio.request import SpeechToTextRequest
from prism.audio.response import AudioTextResponse
from prism.embeddings.request import EmbeddingsRequest
from prism.embeddings.response import EmbeddingsResponse
from prism.errors import PrismError
from prism.fim.request import FimRequest
from prism.fim.response import FimResponse
from prism.http import (
    DEFAULT_TIMEOUT,
    HttpRequest,
    StreamTransport,
    Transport,
    UrllibStreamTransport,
    UrllibTransport,
    encode_multipart,
)
from prism.providers.base import Provider
from prism.providers.mistral.embeddings import build_embeddings_body, parse_embeddings_response
from prism.providers.mistral.fim import parse_fim_response
from prism.providers.mistral.rate_limits import parse_rate_limits
from prism.providers.mistral.request_body import (
    build_fim_body,
    build_request_body,
    build_structured_body,
)
from prism.providers.mistral.response import parse_text_response
from prism.providers.mistral.stream_events import MistralStreamMapper
from prism.providers.openai.audio import build_transcription_form, parse_transcription_response
from prism.streaming.events import StreamEvent
from prism.streaming.sse import sse_data
from prism.structured.from_text import structured_from_text_response
from prism.structured.request import StructuredRequest
from prism.structured.response import StructuredResponse
from prism.text.request import Request
from prism.text.response import Response

__all__ = ["Mistral"]

DEFAULT_URL = "https://api.mistral.ai/v1"


class Mistral(Provider):
    """Talks to Mistral's chat-completions API.

    The third provider in this port, and the first that is neither OpenAI's
    Responses API nor Anthropic's Messages API: Mistral speaks the OpenAI
    CHAT-COMPLETIONS shape. That is close enough to be worth stating plainly,
    because the temptation to share the OpenAI mapper is real and wrong -- the
    two disagree on the request envelope (``messages`` vs ``input``), on where
    the finish reason lives (the choice vs the root), and on the tool-call
    argument encoding (a JSON string vs a dict). Every one of those fails
    quietly.

    It exists because ``fim`` does. Fill-in-the-middle is a Mistral-only
    capability in the reference, so the twelfth capability could not land here
    without it -- but a provider serving only ``fim`` would be a provider nobody
    could use for anything else, so the surface the reference gives Mistral is
    ported whole: text, structured, stream, embeddings, fim, speech_to_text.

    Configuration is explicit first and environment second, falling back to
    ``MISTRAL_API_KEY`` / ``MISTRAL_URL``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        url: str | None = None,
        transport: Transport | None = None,
        stream_transport: StreamTransport | None = None,
        timeout: float | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("MISTRAL_API_KEY", "")
        self.url = (url or os.environ.get("MISTRAL_URL") or DEFAULT_URL).rstrip("/")
        self.timeout = timeout
        self._transport: Transport = transport or UrllibTransport()
        self._stream_transport: StreamTransport = stream_transport or UrllibStreamTransport()

    def text(self, request: Request) -> Response:
        decoded, headers = self._send("chat/completions", build_request_body(request))

        return parse_text_response(request, decoded, parse_rate_limits(headers))

    def structured(self, request: StructuredRequest) -> StructuredResponse:
        """Structured output through Mistral's OWN strict schema mode.

        So this is enforced, not requested -- the same guarantee the OpenAI path
        gives and stronger than the Anthropic one, which can only ask. G-08
        records that ``structured`` means different things per provider; Mistral
        lands on the strong side of it.
        """
        decoded, headers = self._send("chat/completions", build_structured_body(request))

        return structured_from_text_response(
            parse_text_response(request, decoded, parse_rate_limits(headers))
        )

    def fim(self, request: FimRequest) -> FimResponse:
        """Fill in the middle: a prefix, an optional suffix, and the gap.

        A DIFFERENT ENDPOINT from chat, not a mode of it -- ``fim/completions``
        takes a prompt and a suffix and has no messages at all. That is why the
        capability has its own builder rather than a flag on the text one.
        """
        return parse_fim_response(self._post("fim/completions", build_fim_body(request)), request)

    def embeddings(self, request: EmbeddingsRequest) -> EmbeddingsResponse:
        return parse_embeddings_response(self._post("embeddings", build_embeddings_body(request)))

    def speech_to_text(self, request: SpeechToTextRequest) -> AudioTextResponse:
        """Transcription, sharing the OpenAI multipart form.

        SHARED DELIBERATELY, unlike the chat mapping. Mistral's
        ``audio/transcriptions`` is the Whisper endpoint shape field for field,
        so a second copy would be a copy and the two would drift on whichever
        one someone edited. The chat endpoints are not shared for the opposite
        reason: they only look alike.
        """
        content_type, body = encode_multipart(build_transcription_form(request))
        headers = {**self._headers(len(body)), "Content-Type": content_type}

        response = self._transport.send(
            HttpRequest(
                method="POST",
                url=f"{self.url}/audio/transcriptions",
                headers=headers,
                body=body,
                timeout=self.timeout if self.timeout is not None else DEFAULT_TIMEOUT,
            )
        )

        decoded = self._decode(response.status, response.body)

        if response.status >= 400:
            raise PrismError.provider_response_error(
                self._describe(response.status, decoded),
                status=response.status,
                body=response.body.decode("utf-8", errors="replace"),
            )

        return parse_transcription_response(decoded)

    def stream(self, request: Request) -> Iterator[StreamEvent]:
        """The same generation, delivered as it arrives.

        ``stream: True`` is added to the SAME body the non-streamed path builds,
        so the two cannot drift apart into different requests that merely look
        alike.
        """
        payload = {**build_request_body(request), "stream": True}
        body = canonical.encode(payload).encode("utf-8")
        headers = {**self._headers(len(body)), "Accept": "text/event-stream"}

        response = self._stream_transport.stream(
            HttpRequest(
                method="POST",
                url=f"{self.url}/chat/completions",
                headers=headers,
                body=body,
                timeout=self.timeout if self.timeout is not None else DEFAULT_TIMEOUT,
            )
        )

        if response.status >= 400:
            # An error response is not an event stream; the message inside it is
            # the only useful thing about this call.
            text = "".join(response.chunks)
            raise PrismError.provider_response_error(
                self._describe(response.status, _loads(text)),
                status=response.status,
                body=text,
            )

        return self._events(response.chunks)

    def _events(self, chunks: Iterator[str]) -> Iterator[StreamEvent]:
        """One mapper per stream.

        It carries the message id, the accumulated text and half-assembled
        tool-call arguments; a shared instance would let two concurrent
        generations read each other's fragments.
        """
        mapper = MistralStreamMapper()

        for payload in sse_data(chunks):
            # Mistral closes with a literal `[DONE]`, which is not JSON. Parsing
            # it yields an empty dict, which the mapper would read as a chunk.
            if payload == "[DONE]":
                return

            yield from mapper.map(_loads(payload))

    # -- internals ---------------------------------------------------------

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """The body alone, for the paths that carry no quota metadata."""
        return self._send(path, payload)[0]

    def _send(self, path: str, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
        """The decoded body AND the response headers.

        Two methods rather than one returning a tuple everywhere: only the text
        and structured paths read rate-limit headers, and widening every caller
        to a tuple it discards would put the headers in reach of code that has
        no business with them.
        """
        body = canonical.encode(payload).encode("utf-8")

        response = self._transport.send(
            HttpRequest(
                method="POST",
                url=f"{self.url}/{path}",
                headers=self._headers(len(body)),
                body=body,
                timeout=self.timeout if self.timeout is not None else DEFAULT_TIMEOUT,
            )
        )

        decoded = self._decode(response.status, response.body)

        if response.status >= 400:
            raise PrismError.provider_response_error(
                self._describe(response.status, decoded),
                status=response.status,
                body=response.body.decode("utf-8", errors="replace"),
            )

        return decoded, response.headers

    def _headers(self, content_length: int) -> dict[str, str]:
        """A bearer token, like OpenAI and unlike Anthropic's ``x-api-key``."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Content-Length": str(content_length),
        }

        # Omitted entirely when not configured, rather than sent empty.
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        return headers

    def _describe(self, status: int, decoded: dict[str, Any]) -> str:
        """A message out of either error shape Mistral uses.

        A nested ``error.message`` on most failures, and a BARE ``message`` on
        validation errors -- the second is what a malformed request actually
        gets, so reading only the first reports "unknown" for the most common
        mistake.
        """
        detail = data_get(decoded, "error.message") or decoded.get("message") or "unknown"

        return f"Mistral error [{status}]: {detail}"

    def _decode(self, status: int, body: bytes) -> dict[str, Any]:
        text = body.decode("utf-8", errors="replace")

        try:
            decoded = json.loads(text) if text else None
        except ValueError:
            decoded = None

        if not isinstance(decoded, dict):
            raise PrismError.provider_response_error(
                f"Mistral returned a body that is not a JSON object (status {status}).",
                status=status,
                body=text,
            )

        return decoded


def _loads(text: str) -> dict[str, Any]:
    """An SSE payload that is not JSON is skipped rather than fatal."""
    try:
        parsed = json.loads(text)
    except ValueError:
        return {}

    return parsed if isinstance(parsed, dict) else {}
