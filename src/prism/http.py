"""HTTP, kept small and injectable.

The package has ZERO runtime dependencies, so the default transport is built on
:mod:`urllib.request` from the standard library. It is behind a
:class:`Transport` protocol so a test — or a conformance runner — can substitute
a transport and never touch the network.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol

__all__ = ["HttpRequest", "HttpResponse", "Transport", "UrllibTransport"]

DEFAULT_TIMEOUT = 30.0


@dataclass(frozen=True)
class HttpRequest:
    """One outbound request."""

    method: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes | None = None
    timeout: float = DEFAULT_TIMEOUT


@dataclass(frozen=True)
class HttpResponse:
    """One inbound response."""

    status: int
    body: bytes
    headers: dict[str, str] = field(default_factory=dict)

    def text(self) -> str:
        return self.body.decode("utf-8")


class Transport(Protocol):
    """Anything that can carry an :class:`HttpRequest` and bring back a response."""

    def send(self, request: HttpRequest) -> HttpResponse: ...


class UrllibTransport:
    """The default transport: :mod:`urllib.request`, no third-party code.

    An HTTP error status is returned as a response rather than raised, because
    the provider reads the error body to build its own coded failure.
    """

    def send(self, request: HttpRequest) -> HttpResponse:
        raw = urllib.request.Request(
            url=request.url,
            data=request.body,
            headers=request.headers,
            method=request.method,
        )

        try:
            with urllib.request.urlopen(raw, timeout=request.timeout) as response:
                return HttpResponse(
                    status=response.status,
                    body=response.read(),
                    headers={key.lower(): value for key, value in response.headers.items()},
                )
        except urllib.error.HTTPError as error:
            return HttpResponse(
                status=error.code,
                body=error.read(),
                headers={key.lower(): value for key, value in error.headers.items()},
            )
