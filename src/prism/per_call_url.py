"""Refuse a per-call provider url that points at a private address.

A provider url passed for a single call -- ``using(provider, model, {"url":
...})`` -- may not choose a loopback, private, link-local or metadata address.
CONFIG urls (``<PREFIX>_URL``, the provider default, a ``configure_providers``
entry's ``url``) are not checked here: an operator may point those anywhere.

The rules are spelled identically in the PHP reference
(``Support\\PerCallUrl``) and prism-ts (``src/internal/per-call-url.ts``). The
check is synchronous and makes no network or DNS call, so a NAME that resolves
to a private address is not caught here: this checks what the url says, and
does not protect against a caller who chooses a hostname. A literal address in
the spellings an OS resolver reads is caught. No message names the url, host or
address.

The host is judged AS WRITTEN and never decoded. Some HTTP clients
percent-decode a host before they connect, so a check that read the text and a
client that decoded it would judge different hosts. A host holding anything but
letters, digits, ``-`` and ``.`` (and, inside brackets, an IPv6 literal's
``:``) is refused, ``%`` included, and so is a backslash anywhere in the
authority, which URL parsers split in different places. Trailing dots are
removed before the address step, because a resolver reads ``localhost.`` as
``localhost``.

The host must be ASCII. IDNA converters disagree on which name some characters
spell (deviation characters, joiners, characters that map to one), so no
conversion is trusted: an international name is given in its punycode
(``xn--``) form, which every client sends unchanged, and a non-ASCII host is
refused with ``non_ascii_host_use_punycode``. A format, invisible or deviation
character anywhere in the authority is refused first, as
``private_address_refused``.
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
HOST_MESSAGE = (
    "A per-call provider url host may hold only letters, digits, hyphens and dots,"
    " or an IPv6 address in brackets."
)
PUNYCODE_MESSAGE = (
    "A per-call provider url host must be ASCII. Give an international name in its"
    " punycode (xn--) form."
)

# As written: a name (non-ASCII is refused after the invisible check), or a
# bracketed IPv6 literal.
_WRITTEN_HOST = re.compile(r"(?:[A-Za-z0-9.\-\u0080-\U0010ffff]+|\[[A-Za-z0-9.:\-]+\])\Z")
# After lowercasing and trailing dots: ASCII only.
_CANONICAL_HOST = re.compile(r"[a-z0-9.:\-]+\Z", re.ASCII)

# re.ASCII: under Unicode IGNORECASE, U+017F (long s) would match "s".
_HTTP = re.compile(r"^https?://", re.IGNORECASE | re.ASCII)
_NUMERIC_LAST = re.compile(r"(0x[0-9a-f]*|[0-9]+)\Z", re.ASCII)
_HEX = re.compile(r"0x[0-9a-f]*\Z", re.ASCII)
_OCTAL = re.compile(r"0[0-7]+\Z", re.ASCII)
_DECIMAL = re.compile(r"(0|[1-9][0-9]*)\Z", re.ASCII)

_MAX_LABEL = 4294967295

# Names that reach a non-public address without DNS: RFC 6761's ``localhost``,
# the names the stock hosts files map to loopback or to an IPv6 local or
# multicast address, and the documented cloud metadata names. Hosts files:
# Debian, Ubuntu and Docker (``ip6-*``); RHEL, CentOS, Fedora and Amazon Linux
# (``localhost4``, ``localhost6`` and their ``.localdomainN`` forms); openSUSE
# (``ipv6-*``); Alpine (``localhost.localdomain``); macOS (``broadcasthost``).
# Windows ships no active entries.
_NAMES = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "localhost4",
        "localhost4.localdomain4",
        "localhost6",
        "localhost6.localdomain6",
        "ip6-localhost",
        "ip6-loopback",
        "ip6-localnet",
        "ip6-mcastprefix",
        "ip6-allnodes",
        "ip6-allrouters",
        "ipv6-localhost",
        "ipv6-loopback",
        "ipv6-localnet",
        "ipv6-mcastprefix",
        "ipv6-allnodes",
        "ipv6-allrouters",
        "ipv6-allhosts",
        "broadcasthost",
        "metadata",
        "metadata.google.internal",
        "instance-data",
    }
)

# Every name under these reaches loopback without DNS (``*.localhost.localdomain``
# is nss-myhostname's).
_SUFFIXES = (".localhost", ".localhost.localdomain")

# Refused anywhere in the authority, userinfo included: every format or
# invisible character (Unicode's default-ignorable code points, the Cf format
# controls, the C1 controls and the non-ASCII spaces), the four UTS46 deviation
# characters (U+00DF, U+03C2, U+200C, U+200D) and U+1E9E, which UTS46 maps to
# U+00DF. IDNA implementations disagree on these: an HTTP client that deletes a
# zero-width joiner contacts ``localhost`` where a nontransitional conversion
# reads a public name. Spelled code point for code point in the reference
# (``PerCallUrl::INVISIBLE``) and prism-ts (``INVISIBLE``). Inclusive ranges.
_INVISIBLE_RANGES = (
    (0x0080, 0x00A0),
    (0x00AD, 0x00AD),
    (0x00DF, 0x00DF),
    (0x034F, 0x034F),
    (0x03C2, 0x03C2),
    (0x0600, 0x0605),
    (0x061C, 0x061C),
    (0x06DD, 0x06DD),
    (0x070F, 0x070F),
    (0x0890, 0x0891),
    (0x08E2, 0x08E2),
    (0x115F, 0x1160),
    (0x1680, 0x1680),
    (0x17B4, 0x17B5),
    (0x1806, 0x1806),
    (0x180B, 0x180F),
    (0x1E9E, 0x1E9E),
    (0x2000, 0x200F),
    (0x2028, 0x202F),
    (0x205F, 0x206F),
    (0x2800, 0x2800),
    (0x3000, 0x3000),
    (0x3164, 0x3164),
    (0xFE00, 0xFE0F),
    (0xFEFF, 0xFEFF),
    (0xFFA0, 0xFFA0),
    (0xFFF0, 0xFFFB),
    (0x110BD, 0x110BD),
    (0x110CD, 0x110CD),
    (0x13430, 0x1343F),
    (0x1BCA0, 0x1BCA3),
    (0x1D173, 0x1D17A),
    (0xE0000, 0xE0FFF),
)
# Lone surrogates (U+D800..U+DFFF) cannot be read as text, so they are refused too.
_INVISIBLE = re.compile(
    "["
    + "".join(f"{chr(low)}-{chr(high)}" for low, high in (*_INVISIBLE_RANGES, (0xD800, 0xDFFF)))
    + "]"
)

_ASCII_LOWER = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")


def _private() -> PrismError:
    return PrismError(ErrorCode.PRIVATE_ADDRESS_REFUSED, PRIVATE_MESSAGE)


def _scheme() -> PrismError:
    return PrismError(ErrorCode.SCHEME_NOT_ALLOWED, SCHEME_MESSAGE)


def _host() -> PrismError:
    """The private-address code: a host that cannot be read safely is refused
    under the rule it might be evading."""
    return PrismError(ErrorCode.PRIVATE_ADDRESS_REFUSED, HOST_MESSAGE)


def _authority(url: str) -> str:
    """The authority as written: from after ``//`` to the first ``/``, ``?`` or ``#``."""
    rest = url[url.index("//") + 2 :]
    return re.split(r"[/?#]", rest, maxsplit=1)[0]


def _written_host(url: str) -> str | None:
    """The host exactly as written, or None when the authority holds a backslash.

    The host follows the authority's last ``@`` and ends at the port's ``:``
    (or, when bracketed, at the first ``]``). Nothing is decoded.
    """
    authority = _authority(url)
    if "\\" in authority:
        return None
    host_port = authority.rpartition("@")[2]
    if host_port.startswith("["):
        end = host_port.find("]")
        return host_port if end == -1 else host_port[: end + 1]
    return host_port.partition(":")[0]


def assert_per_call_url(url: str, *, allow_private: bool = False) -> None:
    """Refuse ``url`` when its host is a non-public address or a local name.

    ``allow_private=True`` (the provider config's opt-out) skips only the
    address and local-name rules. The scheme, the host as written, invisible
    characters and the ASCII rule hold either way: the opt-out lets a call name
    a private address, never another scheme or a host that cannot be read.

    :raises PrismError: ``scheme_not_allowed`` for a url that is not absolute
        http/https with a host; ``private_address_refused`` for a loopback,
        private, link-local, metadata or otherwise non-public address, a local
        name, or a host that reads as an address but cannot be decoded;
        ``non_ascii_host_use_punycode`` for a host that is not ASCII.
    """
    # 1. Scheme.
    if not _HTTP.match(url):
        raise _scheme()

    # 2. The host as written, before any parser touches it.
    written = _written_host(url)
    if written == "":
        raise _scheme()
    if written is None or not _WRITTEN_HOST.match(written):
        raise _host()
    # An address literal has no trailing dot; only a name does.
    if written.endswith(".]"):
        raise _host()
    if _INVISIBLE.search(_authority(url)):
        raise _host()
    if not written.isascii():
        raise PrismError(ErrorCode.NON_ASCII_HOST_USE_PUNYCODE, PUNYCODE_MESSAGE)

    # 3. Parse. The parser's host drops any userinfo.
    try:
        parts = urlsplit(url)
        host = parts.hostname
        parts.port  # noqa: B018 - an unparseable port is an unparseable url
    except ValueError:
        # Not chained: the parser's message quotes the url.
        raise _scheme() from None
    if not host:
        raise _scheme()

    # 4. Normalize.
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    host = host.translate(_ASCII_LOWER).rstrip(".")
    if not host:
        raise _scheme()
    # The written host was ASCII, so anything else here is a parser surprise.
    if not _CANONICAL_HOST.match(host):
        raise _host()
    if allow_private:
        return

    # 5. IPv6 literal.
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

    # 6. IPv4, in every form an OS resolver accepts.
    labels = host.split(".")
    if _NUMERIC_LAST.match(labels[-1]):
        if not is_public_address(_dotted_quad(labels)):
            raise _private()
        return

    # 7. Local and metadata names.
    if host in _NAMES or host.endswith(_SUFFIXES):
        raise _private()

    # 8. Any other name is allowed; what it resolves to is not checked here.


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
