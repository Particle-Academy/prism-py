"""HTTP, kept small and injectable.

The package has ZERO runtime dependencies, so the default transport is built on
:mod:`urllib.request` from the standard library. It is behind a
:class:`Transport` protocol so a test — or a conformance runner — can substitute
a transport and never touch the network.
"""

from __future__ import annotations

import codecs
import ipaddress
import math
import secrets
import urllib.error
import urllib.request
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol
from urllib.parse import urlsplit

from prism._http_deadline import NetworkDeadline, opener

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
DEFAULT_STREAM_IDLE_TIMEOUT = 60.0

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
    # Guarded media requests inspect each redirect before sending the next hop.
    # A Transport implementation must honor this flag when it is false.
    follow_redirects: bool = True
    # When set, read at most this many bytes of the body plus ONE, so the caller
    # can tell "over the cap" without the transport holding all of it; and read
    # nothing when a declared Content-Length is already over it. Guarded media
    # relies on this to bound an answer it does not trust (G-90). A Transport
    # that ignores it is still refused by the caller, after the memory is spent.
    max_body_bytes: int | None = None
    # Native per-request options have a deliberately narrow contract. They
    # cannot replace a guarded destination, headers, redirect policy or cap.
    client_options: Mapping[str, Any] = field(default_factory=dict)
    connect_timeout: float | None = None
    # Guarded media pins its checked addresses; a transport must not re-resolve.
    resolved_addresses: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if any(key not in ("timeout", "connect_timeout") for key in self.client_options):
            raise ValueError("Client options support only timeout and connect_timeout.")
        options = dict(self.client_options)
        connect = self.connect_timeout if self.connect_timeout is not None else self.timeout
        timeout = options.get("timeout", self.timeout)
        connect = options.get("connect_timeout", connect)
        for value in (timeout, connect):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError("Timeouts must be finite nonnegative seconds.")
        object.__setattr__(self, "timeout", float(timeout))
        object.__setattr__(self, "connect_timeout", float(connect))
        object.__setattr__(self, "client_options", MappingProxyType(options))
        if self.resolved_addresses is not None:
            addresses = tuple(self.resolved_addresses)
            if not addresses:
                raise ValueError("Pinned destinations need numeric addresses.")
            for address in addresses:
                ipaddress.ip_address(address)
            object.__setattr__(self, "resolved_addresses", addresses)


@dataclass(frozen=True)
class HttpResponse:
    """One inbound response."""

    status: int
    body: bytes
    headers: dict[str, str] = field(default_factory=dict)

    def text(self) -> str:
        return self.body.decode("utf-8")


class Transport(Protocol):
    """Carry a request and return its response, honoring ``follow_redirects``.

    When false, return the redirect itself without requesting its destination.
    Guarded media relies on this contract to check each hop before dispatch.
    """

    def send(self, request: HttpRequest) -> HttpResponse: ...


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> None:
        return None


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

        budget = NetworkDeadline(
            request.timeout,
            request.connect_timeout,
            urlsplit(request.url).hostname,
            request.resolved_addresses,
        )
        try:
            open_request = opener(budget, [] if request.follow_redirects else [_NoRedirect()]).open
            try:
                response = open_request(raw, timeout=budget.connection_timeout())
            except urllib.error.HTTPError as error:
                response = error
            with response:
                headers = {fold_header_name(key): value for key, value in response.headers.items()}
                body = _bounded_read(response, headers, request.max_body_bytes)
                budget.remaining()
                return HttpResponse(status=response.status, body=body, headers=headers)
        except Exception:
            budget.remaining()  # A deadline-caused EOF is a timeout, not success.
            raise
        finally:
            budget.close()


def _bounded_read(source: Any, headers: dict[str, str], limit: int | None) -> bytes:
    """The body, or at most ``limit + 1`` bytes of it.

    Nothing is read when a declared Content-Length is already over ``limit``.
    """
    if limit is None:
        return bytes(source.read())
    declared = headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        return b""
    return bytes(source.read(limit + 1))


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

        budget = NetworkDeadline(
            request.timeout,
            request.connect_timeout,
            urlsplit(request.url).hostname,
            request.resolved_addresses,
        )
        response: Any = None
        try:
            try:
                response = opener(budget, [] if request.follow_redirects else [_NoRedirect()]).open(
                    raw, timeout=budget.connection_timeout()
                )
            except urllib.error.HTTPError as error:
                response = error
            budget.stream_body(DEFAULT_STREAM_IDLE_TIMEOUT)
            return HttpStreamResponse(
                status=response.status,
                headers={fold_header_name(key): value for key, value in response.headers.items()},
                chunks=_ResponseChunks(response, budget, self.block_size),
            )
        except BaseException:
            try:
                budget.remaining()
            finally:
                if response is not None:
                    response.close()
                budget.close()
            raise


class _ResponseChunks(Iterator[str]):
    """Closing an unstarted iterator also closes its already-open response."""

    def __init__(self, response: Any, budget: NetworkDeadline, block_size: int) -> None:
        self.response = response
        self.budget = budget
        self.block_size = block_size
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.closed = False

    def __next__(self) -> str:
        if self.closed:
            raise StopIteration
        try:
            while True:
                block = self.response.read1(self.block_size)
                if not block:
                    tail = self.decoder.decode(b"", final=True)
                    self.close()
                    if tail:
                        return str(tail)
                    raise StopIteration
                text = self.decoder.decode(block)
                if text:
                    return str(text)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            try:
                self.response.close()
            finally:
                self.budget.close()


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
