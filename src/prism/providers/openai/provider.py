"""The OpenAI provider, Responses API only."""

from __future__ import annotations

import json
import os
from typing import Any

from prism import canonical
from prism._php import data_get
from prism.errors import PrismError
from prism.http import DEFAULT_TIMEOUT, HttpRequest, Transport, UrllibTransport
from prism.providers.base import Provider
from prism.providers.openai.request_body import build_request_body
from prism.providers.openai.response import parse_text_response
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

    def text(self, request: Request) -> Response:
        if self.api_format != "responses":
            self._unsupported(f"text via the {self.api_format} API")

        body = canonical.encode(build_request_body(request)).encode("utf-8")
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
