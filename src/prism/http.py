"""HTTP, kept small and injectable.

The package has ZERO runtime dependencies, so the default transport is built on
:mod:`urllib.request` from the standard library. It is behind a
:class:`Transport` protocol so a test — or a conformance runner — can substitute
a transport and never touch the network.
"""

from __future__ import annotations

import codecs
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

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


@dataclass
class HttpStreamResponse:
    """A response whose body arrives in pieces.

    Separate from :class:`HttpResponse` rather than a mode of it, because the
    two have genuinely different contracts: one has a body you can read twice,
    the other has a body you can read once. Modelling them as one type would
    let a caller hold a ``body`` that is sometimes bytes and sometimes already
    consumed.

    ``chunks`` yields whatever the network handed over -- NOT lines and NOT
    events. A transport that promised lines would have to buffer, and buffering
    is where streaming bugs live; the parser owns reassembly instead, where a
    test can split a payload at a deliberately awkward place.
    """

    status: int
    headers: dict[str, str]
    chunks: Iterator[str]


class StreamTransport(Protocol):
    """Anything that can carry a request and stream the response back."""

    def stream(self, request: HttpRequest) -> HttpStreamResponse: ...


class UrllibStreamTransport:
    """The default streaming transport.

    Reads the socket in blocks and decodes INCREMENTALLY. A multi-byte
    character can straddle two reads, so the decoder is kept across iterations
    and flushed at the end rather than decoding each block on its own -- doing
    that would corrupt any non-ASCII character unlucky enough to land on a
    boundary, which is a bug that only shows up in other people's languages.
    """

    def __init__(self, block_size: int = 8192) -> None:
        self.block_size = block_size

    def stream(self, request: HttpRequest) -> HttpStreamResponse:
        raw = urllib.request.Request(
            url=request.url,
            data=request.body,
            headers=request.headers,
            method=request.method,
        )

        try:
            response = urllib.request.urlopen(raw, timeout=request.timeout)
        except urllib.error.HTTPError as error:
            # An error response is not an event stream. Handed back whole so the
            # provider can read the message out of it.
            body = error.read()
            return HttpStreamResponse(
                status=error.code,
                headers={key.lower(): value for key, value in error.headers.items()},
                chunks=iter([body.decode("utf-8", errors="replace")]),
            )

        return HttpStreamResponse(
            status=response.status,
            headers={key.lower(): value for key, value in response.headers.items()},
            chunks=self._read(response),
        )

    def _read(self, response: Any) -> Iterator[str]:
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

        try:
            while True:
                block = response.read(self.block_size)

                if not block:
                    tail = decoder.decode(b"", final=True)
                    if tail:
                        yield tail
                    return

                text = decoder.decode(block)
                if text:
                    yield text
        finally:
            response.close()
