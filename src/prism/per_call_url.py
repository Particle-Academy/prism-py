"""Refuse a per-call provider url that points at a private address.

A provider url passed for a single call -- ``using(provider, model, {"url":
...})`` -- may not choose a loopback, private, link-local or metadata address.
CONFIG urls (``<PREFIX>_URL``, the provider default, a ``configure_providers``
entry's ``url``) are not checked here: an operator may point those anywhere.

The rules are spelled identically in the PHP reference
(``Support\\PerCallUrl``) and prism-ts (``src/internal/per-call-url.ts``). The
check is synchronous and makes no network or DNS call, so a NAME that resolves
to a private address is not caught here; a literal address in any spelling an
OS resolver accepts is. Neither message names the url, host or address.
"""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlsplit

from prism.errors import ErrorCode, PrismError
from prism.public_url import is_public_address

__all__ = ["assert_per_call_url"]

PRIVATE_MESSAGE = (
    "A per-call provider url may not point at a loopback, private, link-local or metadata"
    " address. Allow it in the provider config with allow_private_per_call_url."
)
SCHEME_MESSAGE = "A per-call provider url must be an absolute http or https url."

# re.ASCII: under Unicode IGNORECASE, U+017F (long s) would match "s".
_HTTP = re.compile(r"^https?://", re.IGNORECASE | re.ASCII)
_NUMERIC_LAST = re.compile(r"(0x[0-9a-f]*|[0-9]+)\Z", re.ASCII)
_HEX = re.compile(r"0x[0-9a-f]*\Z", re.ASCII)
_OCTAL = re.compile(r"0[0-7]+\Z", re.ASCII)
_DECIMAL = re.compile(r"(0|[1-9][0-9]*)\Z", re.ASCII)

_MAX_LABEL = 4294967295

_NAMES = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "ip6-localhost",
        "ip6-loopback",
        "metadata",
        "metadata.google.internal",
        "instance-data",
    }
)

_ASCII_LOWER = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")


def _private() -> PrismError:
    return PrismError(ErrorCode.PRIVATE_ADDRESS_REFUSED, PRIVATE_MESSAGE)


def _scheme() -> PrismError:
    return PrismError(ErrorCode.SCHEME_NOT_ALLOWED, SCHEME_MESSAGE)


def assert_per_call_url(url: str) -> None:
    """Refuse ``url`` when its host is a non-public address or a local name.

    :raises PrismError: ``scheme_not_allowed`` for a url that is not absolute
        http/https with a host; ``private_address_refused`` for a loopback,
        private, link-local, metadata or otherwise non-public address, a local
        name, or a host that reads as an address but cannot be decoded.
    """
    # 1. Scheme.
    if not _HTTP.match(url):
        raise _scheme()

    # 2. Parse. The parser's host drops any userinfo.
    try:
        parts = urlsplit(url)
        host = parts.hostname
        parts.port  # noqa: B018 - an unparseable port is an unparseable url
    except ValueError:
        # Not chained: the parser's message quotes the url.
        raise _scheme() from None
    if not host:
        raise _scheme()

    # 3. Normalize.
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    if not host.isascii():
        try:
            host = host.encode("idna").decode("ascii")
        except UnicodeError:
            raise _private() from None
    host = host.translate(_ASCII_LOWER).rstrip(".")
    if not host:
        raise _scheme()

    # 4. IPv6 literal.
    if ":" in host:
        if "%" in host:
            raise _private()
        try:
            ipaddress.IPv6Address(host)
        except ValueError:
            raise _private() from None
        if not is_public_address(host):
            raise _private()
        return

    # 5. IPv4, in every form an OS resolver accepts.
    labels = host.split(".")
    if _NUMERIC_LAST.match(labels[-1]):
        if not is_public_address(_dotted_quad(labels)):
            raise _private()
        return

    # 6. Local and metadata names.
    if host in _NAMES or host.endswith(".localhost"):
        raise _private()

    # 7. Any other name is allowed; what it resolves to is not checked here.


def _dotted_quad(labels: list[str]) -> str:
    """The address an IPv4 attempt names, or a refusal when it cannot be read."""
    count = len(labels)
    if count > 4:
        raise _private()
    values = [_label_value(label) for label in labels]
    for value in values[:-1]:
        if value > 255:
            raise _private()
    if values[-1] > 256 ** (5 - count) - 1:
        raise _private()
    address = values[-1]
    for index, value in enumerate(values[:-1]):
        address += value * 256 ** (3 - index)
    return str(ipaddress.IPv4Address(address))


def _label_value(label: str) -> int:
    if _HEX.match(label):
        base, digits = 16, label[2:]
    elif _OCTAL.match(label):
        base, digits = 8, label[1:]
    elif _DECIMAL.match(label):
        base, digits = 10, label
    else:
        raise _private()
    value = 0
    for digit in digits:
        value = value * base + int(digit, 16)
        if value > _MAX_LABEL:
            raise _private()
    return value
