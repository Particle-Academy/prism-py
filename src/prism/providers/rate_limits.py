"""Header reading shared by every provider's rate-limit parser.

A MODULE, not a package, for the same reason ``providers/support.py`` is one: a
directory here is read as a fourth provider by anything that lists
``prism/providers/``.

Each provider's own ``rate_limits`` module owns the part that is genuinely
per-provider — which headers exist, what a reset MEANS — because those are
three different answers and pretending otherwise is how a shared helper starts
lying about one of them. Anthropic sends an instant, OpenAI sends a duration
string, Mistral sends a count of seconds. Only the integer reading is common.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

__all__ = ["header_int", "lowercased"]

#: A leading integer, the way both references read one. PHP's ``(int)`` cast and
#: JavaScript's ``parseInt`` both take the digits off the front and both answer
#: ``0`` for a value with no digits at all — so ``"many"`` becomes ``0`` rather
#: than raising, which is what :func:`int` would do here.
_LEADING_INT = re.compile(r"\s*([+-]?\d+)")


def header_int(value: str | None) -> int:
    """A header's integer value, or ``0`` when it does not carry one.

    Zero rather than ``None`` deliberately: this is the value a provider DID
    send and could not be read, which both references also collapse to zero. A
    header that is absent never reaches here — the caller decides that, and
    absence is what :class:`~prism.value_objects.ProviderRateLimit`'s ``None``
    is for.
    """
    if value is None:
        return 0

    match = _LEADING_INT.match(value)

    return int(match.group(1)) if match is not None else 0


def lowercased(headers: Mapping[str, str]) -> dict[str, str]:
    """The headers with lower-cased names.

    HTTP field names are case-insensitive (RFC 9110 §5.1), and a proxy that
    title-cases them is ordinary rather than hostile. **The reference matches
    its prefix case-sensitively**, so one such proxy makes it report no rate
    limits at all — silently, since an empty list is what a response without
    the headers legitimately looks like. This port's transport already
    lower-cases, so this is belt and braces for a caller passing headers from
    somewhere else; it costs one dict comprehension and removes a failure that
    is invisible when it happens.
    """
    return {name.lower(): value for name, value in headers.items()}
