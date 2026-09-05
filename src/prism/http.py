"""HTTP, kept small and injectable.

The package has ZERO runtime dependencies, so the default transport is built on
:mod:`urllib.request` from the standard library. It is behind a
:class:`Transport` protocol so a test — or a conformance runner — can substitute
a transport and never touch the network.
"""

from __future__ import annotations

import codecs
import secrets
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

__all__ = [
    "HttpRequest",
    "HttpResponse",
    "MultipartBody",
    "MultipartFile",
    "Transport",
    "UrllibTransport",
    "encode_multipart",
    "fold_header_name",
]

DEFAULT_TIMEOUT = 30.0

#: The 26 ASCII letters, and nothing else. **Deliberately not** :meth:`str.lower`
#: or :meth:`str.casefold`, both of which are Unicode-aware: ``K`` (U+212A KELVIN
#: SIGN) folds to a plain ``k``, so a field name carrying it would be compared as
#: though it were the ASCII name the provider actually sends — and a rate-limit
#: reader that derives a BUCKET NAME from the header would then manufacture a
#: ``tokens`` bucket out of a header nobody sent. ``İ`` (U+0130) folds to TWO
#: codepoints, changing the string's length under any offset arithmetic over it.
#:
#: An HTTP field name is an RFC 9110 ``token``: ASCII by grammar. A 1:1 map over
#: the ASCII letters is both the correct fold and the only one ``prism`` (PHP)
#: and ``prism-ts`` can reproduce byte for byte.
_ASCII_FOLD = str.maketrans(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "abcdefghijklmnopqrstuvwxyz",
)


def fold_header_name(name: str) -> str:
    """One HTTP field name, folded to lower case as ASCII and nothing else.

    Field names are case-insensitive (RFC 9110 §5.1), so every response header
    is re-keyed through this before anything looks one up. See
    :data:`_ASCII_FOLD` for why the fold is not :meth:`str.lower`.
    """
    return name.translate(_ASCII_FOLD)


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
                    headers={
                        fold_header_name(key): value for key, value in response.headers.items()
                    },
                )
        except urllib.error.HTTPError as error:
            return HttpResponse(
                status=error.code,
                body=error.read(),
                headers={fold_header_name(key): value for key, value in error.headers.items()},
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
                headers={fold_header_name(key): value for key, value in error.headers.items()},
                chunks=iter([body.decode("utf-8", errors="replace")]),
            )

        return HttpStreamResponse(
            status=response.status,
            headers={fold_header_name(key): value for key, value in response.headers.items()},
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


@dataclass(frozen=True)
class MultipartFile:
    """One file part of a form upload."""

    field: str
    filename: str
    content: bytes
    content_type: str | None = None


@dataclass(frozen=True)
class MultipartBody:
    """A form upload: named string fields, plus files."""

    fields: dict[str, str] = field(default_factory=dict)
    files: tuple[MultipartFile, ...] = ()


#: The format's line ending. A bare LF is tolerated by some servers and
#: rejected by others, and the failure is a rejected upload with no useful
#: message, so it is spelled out once here rather than typed per line.
CRLF = b"\r\n"


def encode_multipart(body: MultipartBody) -> tuple[str, bytes]:
    """Encode a form upload, returning its Content-Type and its bytes.

    A FUNCTION rather than a transport, which is where this port diverges from
    the TypeScript one. There, ``fetch`` owns the encoder and sets the boundary
    itself, so a form has to be handed to the transport intact; here nothing in
    the standard library encodes one for us, so we encode it and the existing
    :class:`Transport` carries the result like any other body. That keeps the
    transport protocol at one method, and keeps every test double already
    written for it working unchanged.

    The BOUNDARY is random per call. The format has no escaping: a boundary that
    also appears inside a part silently splits the file, and the only defence is
    picking one that will not. 32 hex characters from :mod:`secrets` is the
    standard answer.
    """
    boundary = secrets.token_hex(16)
    marker = f"--{boundary}".encode("ascii")
    out = bytearray()

    for name, value in body.fields.items():
        out += marker + CRLF
        out += f'Content-Disposition: form-data; name="{name}"'.encode()
        out += CRLF + CRLF
        out += value.encode("utf-8")
        out += CRLF

    for part in body.files:
        out += marker + CRLF
        out += (
            f'Content-Disposition: form-data; name="{part.field}"; filename="{part.filename}"'
        ).encode()
        out += CRLF

        if part.content_type is not None:
            out += f"Content-Type: {part.content_type}".encode("ascii") + CRLF

        out += CRLF
        out += part.content
        out += CRLF

    out += marker + b"--" + CRLF

    return f"multipart/form-data; boundary={boundary}", bytes(out)
