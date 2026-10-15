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


# Every range a guarded fetch refuses, spelled identically in all three languages
# rather than delegated to a language's IP helpers (G-89): Python's is_global is
# version-dependent, and PHP's filter_var flags missed CGNAT, where Alibaba
# Cloud's metadata endpoint lives. As in TypeScript, every IPv4-compatible IPv6
# spelling is refused consistently.
_BLOCKED = tuple(
    ipaddress.ip_network(cidr)
    for cidr in (
        "0.0.0.0/8",
        "10.0.0.0/8",
        "100.64.0.0/10",
        "127.0.0.0/8",
        "169.254.0.0/16",
        "172.16.0.0/12",
        "192.0.0.0/24",
        "192.0.2.0/24",
        "192.168.0.0/16",
        "198.18.0.0/15",
        "198.51.100.0/24",
        "203.0.113.0/24",
        "224.0.0.0/4",
        "240.0.0.0/4",
        "::/96",
        "::ffff:0:0/96",
        "64:ff9b:1::/48",
        "100::/64",
        "2001::/32",
        "2001:db8::/32",
        "fc00::/7",
        "fe80::/10",
        "fec0::/10",
        "ff00::/8",
    )
)

# NAT64 and 6to4 carry an IPv4 address inside an IPv6 one, and the embedded
# address is where the packet goes: 64:ff9b::a9fe:a9fe IS the metadata endpoint
# on a NAT64 network. The embedded address is judged. Not refused outright,
# because a DNS64 resolver hands an IPv6-only host 64:ff9b:: addresses for
# perfectly public sites. (network, byte offset of the embedded IPv4)
_EMBEDDING = (
    (ipaddress.ip_network("64:ff9b::/96"), 12),
    (ipaddress.ip_network("2002::/16"), 2),
)


def _assert_public_address(address: str) -> None:
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        parsed = None
    if parsed is not None and "%" not in address and parsed.version == 6:
        for network, offset in _EMBEDDING:
            if parsed in network:
                embedded = ipaddress.IPv4Address(parsed.packed[offset : offset + 4])
                _assert_public_address(str(embedded))
                return
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
