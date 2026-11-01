"""The Anthropic provider, Messages API."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

from prism import canonical
from prism._php import data_get
from prism.errors import PrismError
from prism.http import (
    DEFAULT_TIMEOUT,
    HttpRequest,
    StreamTransport,
    Transport,
    UrllibStreamTransport,
    UrllibTransport,
)
from prism.providers.anthropic.rate_limits import parse_rate_limits
from prism.providers.anthropic.request_body import build_request_body, build_structured_body
from prism.providers.anthropic.response import parse_text_response
from prism.providers.anthropic.stream_events import AnthropicStreamMapper
from prism.providers.base import Provider
from prism.streaming.events import StreamEvent
from prism.streaming.sse import sse_data
from prism.structured.from_text import structured_from_text_response
from prism.structured.request import StructuredRequest
from prism.structured.response import StructuredResponse
from prism.text.request import Request
from prism.text.response import Response

__all__ = ["Anthropic"]

DEFAULT_URL = "https://api.anthropic.com/v1"

#: Pinned rather than tracking latest.
#:
#: The version header decides the response SHAPE. Floating it would let a
#: provider-side release change what this parser receives without a line of code
#: changing here — the drift the conformance corpus exists to catch, arriving
#: through a door the corpus cannot see.
DEFAULT_API_VERSION = "2023-06-01"


class Anthropic(Provider):
    """Talks to Anthropic's Messages API.

    Configuration comes from explicit constructor arguments, falling back to
    ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_URL`` / ``ANTHROPIC_API_VERSION`` /
    ``ANTHROPIC_BETA_FEATURES``.

    ``transport`` is injectable so nothing here has to reach the network in a
    test.
    """

    def __init__(
        self,
        api_key: str | None = None,
        url: str | None = None,
        api_version: str | None = None,
        beta_features: str | None = None,
        transport: Transport | None = None,
        stream_transport: StreamTransport | None = None,
        timeout: float | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY", "")
        self.url = (url or os.environ.get("ANTHROPIC_URL") or DEFAULT_URL).rstrip("/")
        self.api_version = (
            api_version or os.environ.get("ANTHROPIC_API_VERSION") or DEFAULT_API_VERSION
        )
        self.beta_features = (
            beta_features
            if beta_features is not None
            else os.environ.get("ANTHROPIC_BETA_FEATURES")
        )
        self.timeout = timeout
        self._transport: Transport = transport or UrllibTransport()
        self._stream_transport: StreamTransport = stream_transport or UrllibStreamTransport()

    def text(self, request: Request) -> Response:
        return self._send(request, build_request_body(request))

    def _send(self, request: Request, payload: dict[str, Any]) -> Response:
        body = canonical.encode(payload).encode("utf-8")
        response = self._transport.send(
            HttpRequest(
                method="POST",
                url=f"{self.url}/messages",
                headers=self._headers(len(body)),
                body=body,
                timeout=self.timeout if self.timeout is not None else DEFAULT_TIMEOUT,
            )
        )

        decoded = self._decode(response.status, response.body)

        if response.status >= 400:
            raise PrismError.provider_response_error(
                f"Anthropic error [{response.status}]: "
                f"{data_get(decoded, 'error.message', 'unknown')}",
                status=response.status,
                body=response.body.decode("utf-8", errors="replace"),
            )

        return parse_text_response(request, decoded, parse_rate_limits(response.headers))

    def stream(self, request: Request) -> Iterator[StreamEvent]:
        """The same generation, delivered as it arrives.

        The mapper is constructed PER CALL, not shared. It carries the message
        id, the accumulated text and the stop reason for one stream; a shared
        instance would let two concurrent generations read each other's blocks,
        which is the kind of bug that only appears under load and looks like the
        model hallucinating.
        """
        payload = {**build_request_body(request), "stream": True}
        body = canonical.encode(payload).encode("utf-8")
        headers = {**self._headers(len(body)), "Accept": "text/event-stream"}

        response = self._stream_transport.stream(
            HttpRequest(
                method="POST",
                url=f"{self.url}/messages",
                headers=headers,
                body=body,
                timeout=self.timeout if self.timeout is not None else DEFAULT_TIMEOUT,
            )
        )

        if response.status >= 400:
            text = "".join(response.chunks)
            raise PrismError.provider_response_error(
                f"Anthropic error [{response.status}]: "
                f"{data_get(_loads(text), 'error.message', 'unknown')}",
                status=response.status,
                body=text,
            )

        return self._events(response.chunks)

    def _events(self, chunks: Iterator[str]) -> Iterator[StreamEvent]:
        mapper = AnthropicStreamMapper()

        for payload in sse_data(chunks):
            event = mapper.map(_loads(payload))

            if event is not None:
                yield event

        yield from mapper.finish()

    def structured(self, request: StructuredRequest) -> StructuredResponse:
        """Structured output through Anthropic's native output_config.format."""
        return structured_from_text_response(
            self._send(request, build_structured_body(request)), native_output_format=True
        )

    # -- internals ---------------------------------------------------------

    def _headers(self, content_length: int) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Content-Length": str(content_length),
            "anthropic-version": self.api_version,
        }

        # Anthropic authenticates with x-api-key, not a bearer token. Omitted
        # entirely when not configured, rather than sent empty.
        if self.api_key:
            headers["x-api-key"] = self.api_key
        if self.beta_features:
            headers["anthropic-beta"] = self.beta_features

        return headers

    def _decode(self, status: int, body: bytes) -> dict[str, Any]:
        text = body.decode("utf-8", errors="replace")

        try:
            decoded = json.loads(text) if text else None
        except ValueError:
            decoded = None

        if not isinstance(decoded, dict):
            raise PrismError.provider_response_error(
                f"Anthropic returned a body that is not a JSON object (status {status}).",
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
