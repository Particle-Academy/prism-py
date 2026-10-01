"""Validate media destinations before sending; DNS is not connection-pinned."""

from __future__ import annotations

import ipaddress
import re
import socket
from collections.abc import Sequence
from typing import Protocol
from urllib.parse import urlsplit

from prism.errors import ErrorCode, PrismError


class HostResolver(Protocol):
    """Resolve a hostname; every returned address must pass the guard."""

    def resolve(self, host: str) -> Sequence[str]: ...


class DnsHostResolver:
    def resolve(self, host: str) -> Sequence[str]:
        try:
            return list(
                dict.fromkeys(str(record[4][0]) for record in socket.getaddrinfo(host, None))
            )
        except socket.gaierror:
            return []


# Match the reference ranges rather than Python's version-dependent is_global.
# As in TypeScript, refuse every IPv4-compatible IPv6 spelling consistently;
# PHP admits some compressed forms of the same address (e.g. ::7f00:1).
_BLOCKED = tuple(
    ipaddress.ip_network(cidr)
    for cidr in (
        "0.0.0.0/8",
        "10.0.0.0/8",
        "127.0.0.0/8",
        "169.254.0.0/16",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "240.0.0.0/4",
        "::/96",
        "::ffff:0:0/96",
        "fc00::/7",
        "fe80::/10",
    )
)


def _assert_public_address(address: str) -> None:
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        parsed = None
    if parsed is None or "%" in address or any(parsed in network for network in _BLOCKED):
        raise PrismError(
            ErrorCode.PRIVATE_ADDRESS_REFUSED,
            "A guarded fetch refuses private, reserved or invalid addresses.",
        )


def assert_public_url(url: str, resolver: HostResolver) -> None:
    """Check scheme, literal and all DNS answers; rebinding remains possible."""
    try:
        if not re.match(r"^https?://", url, re.IGNORECASE):
            raise ValueError("Not an absolute HTTP URL")
        parsed = urlsplit(url)
        host = parsed.hostname
        # Validate the port too, before passing malformed input to transport.
        _ = parsed.port
        if not host or any(ord(char) <= 32 for char in url):
            raise ValueError("Invalid HTTP URL")
    except ValueError:
        raise PrismError(
            ErrorCode.SCHEME_NOT_ALLOWED, "A guarded fetch needs an absolute http or https URL."
        ) from None
    try:
        ipaddress.ip_address(host)
    except ValueError:
        # libc also accepts shortened, integer, octal and hex IPv4 literals.
        # inet_aton performs no DNS; check their actual address before a socket.
        try:
            literal = socket.inet_ntoa(socket.inet_aton(host))
        except OSError:
            addresses = resolver.resolve(host)
            if not addresses:
                raise PrismError(
                    ErrorCode.HOST_DID_NOT_RESOLVE,
                    "The hostname did not resolve to a verifiable public address.",
                ) from None
            for address in addresses:
                _assert_public_address(address)
        else:
            _assert_public_address(literal)
    else:
        _assert_public_address(host)
