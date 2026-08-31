"""The OpenAI provider, Responses API only."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

from prism import canonical
from prism._php import data_get
from prism.embeddings.request import EmbeddingsRequest
from prism.embeddings.response import EmbeddingsResponse
from prism.errors import PrismError
from prism.http import (
    DEFAULT_TIMEOUT,
    HttpRequest,
    StreamTransport,
    Transport,
    UrllibStreamTransport,
    UrllibTransport,
)
from prism.images.request import ImagesRequest
from prism.images.response import ImagesResponse
from prism.providers.base import Provider
from prism.providers.openai.embeddings import build_embeddings_body, parse_embeddings_response
from prism.providers.openai.images import build_images_body, parse_images_response
from prism.providers.openai.request_body import build_request_body
from prism.providers.openai.response import parse_text_response
from prism.providers.openai.stream_events import map_stream_event
from prism.providers.openai.structured_body import build_structured_body
from prism.streaming.events import StreamEvent
from prism.streaming.sse import sse_data
from prism.structured.from_text import structured_from_text_response
from prism.structured.request import StructuredRequest
from prism.structured.response import StructuredResponse
from prism.text.request import Request
from prism.text.response import Response

__all__ = ["OpenAI"]

DEFAULT_URL = "https://api.openai.com/v1"


class OpenAI(Provider):
    """Talks to OpenAI's Responses API.

    Configuration comes from explicit constructor arguments, falling back to
    ``OPENAI_API_KEY`` / ``OPENAI_URL`` / ``OPENAI_ORGANIZATION`` /
    ``OPENAI_PROJECT``.

    ``transport`` is injectable so nothing here has to reach the network in a
    test.
    """

    def __init__(
        self,
        api_key: str | None = None,
        url: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        api_format: str = "responses",
        transport: Transport | None = None,
        stream_transport: StreamTransport | None = None,
        timeout: float | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        self.url = (url or os.environ.get("OPENAI_URL") or DEFAULT_URL).rstrip("/")
        self.organization = (
            organization if organization is not None else os.environ.get("OPENAI_ORGANIZATION")
        )
        self.project = project if project is not None else os.environ.get("OPENAI_PROJECT")
        self.api_format = api_format
        self.timeout = timeout
        self._transport: Transport = transport or UrllibTransport()
        self._stream_transport: StreamTransport = stream_transport or UrllibStreamTransport()

    def text(self, request: Request) -> Response:
        return self._send(build_request_body(request), request, "text")

    def stream(self, request: Request) -> Iterator[StreamEvent]:
        """The same generation, delivered as it arrives.

        ``stream: True`` is added to the SAME body the non-streamed path
        builds, so the two cannot drift apart into different requests that
        merely look alike.
        """
        if self.api_format != "responses":
            self._unsupported(f"stream via the {self.api_format} API")

        payload = {**build_request_body(request), "stream": True}
        body = canonical.encode(payload).encode("utf-8")
        headers = {**self._headers(len(body)), "Accept": "text/event-stream"}

        response = self._stream_transport.stream(
            HttpRequest(
                method="POST",
                url=f"{self.url}/responses",
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
                f"OpenAI error [{response.status}]: "
                f"{data_get(_loads(text), 'error.message', 'unknown')}",
                status=response.status,
                body=text,
            )

        return self._events(response.chunks)

    def _events(self, chunks: Iterator[str]) -> Iterator[StreamEvent]:
        for payload in sse_data(chunks):
            event = map_stream_event(_loads(payload))

            if event is not None:
                yield event

    def embeddings(self, request: EmbeddingsRequest) -> EmbeddingsResponse:
        """Vectors for one or more inputs.

        A different endpoint from the rest of this provider, so it does not go
        through ``_send``: that helper posts to ``/responses`` and parses a text
        reply, and bending it to also mean ``/embeddings`` would make both
        harder to read than two short methods.
        """
        body = canonical.encode(build_embeddings_body(request)).encode("utf-8")
        response = self._transport.send(
            HttpRequest(
                method="POST",
                url=f"{self.url}/embeddings",
                headers=self._headers(len(body)),
                body=body,
                timeout=self.timeout if self.timeout is not None else DEFAULT_TIMEOUT,
            )
        )

        decoded = self._decode(response.status, response.body)

        if response.status >= 400:
            raise PrismError.provider_response_error(
                f"OpenAI error [{response.status}]: "
                f"{data_get(decoded, 'error.message', 'unknown')}",
                status=response.status,
                body=response.body.decode("utf-8", errors="replace"),
            )

        return parse_embeddings_response(decoded)

    def images(self, request: ImagesRequest) -> ImagesResponse:
        body = canonical.encode(build_images_body(request)).encode("utf-8")
        response = self._transport.send(
            HttpRequest(
                method="POST",
                url=f"{self.url}/images/generations",
                headers=self._headers(len(body)),
                body=body,
                timeout=self.timeout if self.timeout is not None else DEFAULT_TIMEOUT,
            )
        )

        decoded = self._decode(response.status, response.body)

        if response.status >= 400:
            raise PrismError.provider_response_error(
                f"OpenAI error [{response.status}]: "
                f"{data_get(decoded, 'error.message', 'unknown')}",
                status=response.status,
                body=response.body.decode("utf-8", errors="replace"),
            )

        return parse_images_response(decoded, request.model)

    def _send(
        self,
        payload: dict[str, Any],
        request: Request,
        action: str = "structured",
    ) -> Response:
        """One round trip, shared by every capability that posts to /responses.

        Extracted so a structured call cannot drift from a text one: the
        endpoint, the headers, the timeout and the error mapping are decided
        once.
        """
        if self.api_format != "responses":
            self._unsupported(f"{action} via the {self.api_format} API")

        body = canonical.encode(payload).encode("utf-8")
        response = self._transport.send(
            HttpRequest(
                method="POST",
                url=f"{self.url}/responses",
                headers=self._headers(len(body)),
                body=body,
                timeout=self.timeout if self.timeout is not None else DEFAULT_TIMEOUT,
            )
        )

        decoded = self._decode(response.status, response.body)

        if response.status >= 400:
            raise PrismError.provider_response_error(
                f"OpenAI error [{response.status}]: "
                f"{data_get(decoded, 'error.message', 'unknown')}",
                status=response.status,
                body=response.body.decode("utf-8", errors="replace"),
            )

        return parse_text_response(request, decoded)

    def structured(self, request: StructuredRequest) -> StructuredResponse:
        """Structured output through the Responses API's schema format.

        The reply is parsed by the TEXT parser and then given its structured
        reading, so finish reasons, token-limit failures, usage and rate limits
        behave identically on both paths by construction.
        """
        return structured_from_text_response(self._send(build_structured_body(request), request))

    # -- internals ---------------------------------------------------------

    def _headers(self, content_length: int) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Content-Length": str(content_length),
        }

        # Omitted entirely when not configured, rather than sent empty.
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        if self.project:
            headers["OpenAI-Project"] = self.project

        return headers

    def _decode(self, status: int, body: bytes) -> dict[str, Any]:
        text = body.decode("utf-8", errors="replace")

        try:
            decoded = json.loads(text) if text else None
        except ValueError:
            decoded = None

        if not isinstance(decoded, dict):
            raise PrismError.provider_response_error(
                f"OpenAI returned a body that is not a JSON object (status {status}).",
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
