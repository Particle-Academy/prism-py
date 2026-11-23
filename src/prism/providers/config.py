"""Provider configuration presence, separate from empty and explicit null."""

from __future__ import annotations

import math
import os
import re
import threading
import warnings
from decimal import Decimal
from enum import Enum
from typing import Any
from urllib.parse import urlsplit


class Unset(Enum):
    VALUE = 0


UNSET = Unset.VALUE
_warned_legacy_beta = False
_legacy_beta_warning_lock = threading.Lock()


def _environment_value(name: str, fallback: Any) -> Any:
    value = os.environ.get(name)
    if value is None:
        return fallback
    if value.lower() in ("null", "(null)"):
        return None
    if value.lower() in ("true", "(true)"):
        return True
    if value.lower() in ("false", "(false)"):
        return False
    if value.lower() in ("empty", "(empty)"):
        return ""
    quoted = re.fullmatch(r"(['\"])(.*)\1", value)
    return quoted[2] if quoted else value


def configured_string(
    value: str | Unset | None, name: str, fallback: str, null_fallback: str | None = None
) -> str:
    resolved = _environment_value(name, fallback) if value is UNSET else value
    if resolved is None and null_fallback is not None:
        return null_fallback
    if not isinstance(resolved, str):
        raise TypeError(f"Invalid string configuration for {name}.")
    return resolved


def nullable_string(value: str | Unset | None, name: str) -> str | None:
    resolved = _environment_value(name, None) if value is UNSET else value
    if resolved is not None and not isinstance(resolved, str):
        raise TypeError(f"Invalid nullable string configuration for {name}.")
    return resolved


# Codepoints a provider url may not contain anywhere, spelled by codepoint and
# identically in the reference and prism-ts: every Cc control (U+0000..U+001F,
# U+007F..U+009F), every Unicode White_Space codepoint, and the G-36
# zero-widths U+200B..U+200D and U+FEFF. str.isspace() / str.strip() disagree
# with the other two languages' built-ins, so neither is used.
_URL_FORBIDDEN = frozenset(
    [
        *range(0x0000, 0x001F + 1),
        0x0020,
        *range(0x007F, 0x009F + 1),
        0x00A0,
        0x1680,
        *range(0x2000, 0x200D + 1),
        0x2028,
        0x2029,
        0x202F,
        0x205F,
        0x3000,
        0xFEFF,
    ]
)


def assert_clean_url(url: object) -> None:
    """Refuse a url that carries whitespace or a control character anywhere.

    It is never trimmed: where the key goes must not depend on how tolerant an
    HTTP stack is of a padded url. The message names neither the url nor any
    credential. A value that is not a string is left to the caller's own type
    check.

    :raises ValueError: for a url containing a forbidden codepoint.
    """
    if isinstance(url, str) and any(ord(char) in _URL_FORBIDDEN for char in url):
        raise ValueError(
            "That is not a valid provider url: a provider url must not contain"
            " whitespace or control characters."
        )


def configured_url(
    value: str | Unset | None, name: str, fallback: str, null_fallback: str | None = None
) -> str:
    """The provider url, with the environment's and the per-call value both checked.

    A padded environment url is a misconfiguration even when this call
    overrides it, so both are checked by :func:`assert_clean_url` first.
    """
    assert_clean_url(os.environ.get(name))
    assert_clean_url(value)
    return configured_string(value, name, fallback, null_fallback)


_DEFAULT_PORTS = {"https": 443, "http": 80}


def _endpoint_identity(url: str) -> tuple[str, ...] | None:
    """Endpoint identity: case-insensitive scheme/host, default port, no trailing slash."""
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return None
    host = parts.hostname
    if not parts.scheme or not host:
        return None
    scheme = parts.scheme.lower()
    if port is None:
        port = _DEFAULT_PORTS.get(scheme)
    return (
        scheme,
        parts.username or "",
        parts.password or "",
        host.lower(),
        "" if port is None else str(port),
        parts.path.rstrip("/"),
        parts.query,
    )


def same_endpoint(a: str, b: str) -> bool:
    """True when two URLs name the same endpoint.

    A trailing slash or a difference in the case of the scheme or host does not
    make a different endpoint; a different scheme, userinfo, host, port, path or
    query does. An unparseable URL is never the same endpoint.
    """
    left = _endpoint_identity(a)
    return left is not None and left == _endpoint_identity(b)


def inherits_environment_credentials(
    url: str, url_name: str, fallback: str, null_fallback: str | None = None
) -> bool:
    """Whether environment credentials apply to ``url``.

    Only when it is the URL the environment configures, or the provider default
    when none is configured.
    """
    try:
        configured_url = configured_string(UNSET, url_name, fallback, null_fallback)
    except TypeError:
        # An unusable environment URL configures no endpoint to scope a key to.
        return False
    return url == configured_url or same_endpoint(url, configured_url)


def scoped_string(
    value: str | Unset | None,
    name: str,
    inherit: bool,
    fallback: str,
    null_fallback: str | None = None,
) -> str:
    """``configured_string``, reading the environment only when ``inherit`` is true."""
    if value is UNSET and not inherit:
        return fallback
    return configured_string(value, name, fallback, null_fallback)


def scoped_nullable_string(value: str | Unset | None, name: str, inherit: bool) -> str | None:
    """``nullable_string``, reading the environment only when ``inherit`` is true."""
    if value is UNSET and not inherit:
        return None
    return nullable_string(value, name)


def anthropic_beta(value: str | Unset | None) -> str | None:
    global _warned_legacy_beta
    if value is not UNSET or "ANTHROPIC_BETA" in os.environ:
        return nullable_string(value, "ANTHROPIC_BETA")
    legacy = os.environ.get("ANTHROPIC_BETA_FEATURES")
    if legacy is not None:
        with _legacy_beta_warning_lock:
            if not _warned_legacy_beta:
                _warned_legacy_beta = True
                warnings.warn(
                    "ANTHROPIC_BETA_FEATURES is deprecated; use ANTHROPIC_BETA.",
                    FutureWarning,
                    stacklevel=3,
                )
    return legacy


def thinking_budget_default(value: Any = UNSET) -> int:
    """The reference casts the config default, not a noninteger request budget."""
    resolved = (
        _environment_value("ANTHROPIC_DEFAULT_THINKING_BUDGET", 1024) if value is UNSET else value
    )
    if resolved is None:
        return 0
    if isinstance(resolved, (bool, int)):
        return int(resolved)
    if isinstance(resolved, float):
        return int(resolved) if math.isfinite(resolved) else 0
    if isinstance(resolved, str):
        prefix = re.match(r"\s*[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", resolved)
        return int(Decimal(prefix[0])) if prefix else 0
    raise TypeError("Invalid ANTHROPIC_DEFAULT_THINKING_BUDGET configuration.")
