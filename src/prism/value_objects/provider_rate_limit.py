"""What a provider said about the quota this response drew on.

The reset is an INSTANT, not a rendering of one. A rate-limit header is the one
place in this port where three languages could each be "right" and still store
three different strings for the same moment, so the type is a timezone-aware
:class:`~datetime.datetime` and the serialised form is normalised to UTC. The
reference keeps whatever offset the wire used, which means the same instant
stores as different bytes depending on which language wrote the record; that is
recorded as a divergence rather than copied.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

__all__ = ["ProviderRateLimit"]

#: Fractional seconds, so a wire timestamp with any number of digits can be cut
#: down to the six :func:`datetime.datetime.fromisoformat` accepts on 3.10.
_FRACTION = re.compile(r"\.(\d+)")


@dataclass(frozen=True)
class ProviderRateLimit:
    """One quota bucket, as the provider reported it.

    ``limit`` and ``remaining`` are ``None`` when the provider did not send
    them. That is not the same as zero: zero means "you have none left" and
    ``None`` means "the provider did not say", and a caller that backs off on
    the second is throttling itself on a fact it never received.
    """

    name: str
    limit: int | None = None
    remaining: int | None = None
    resets_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "limit": self.limit,
            "remaining": self.remaining,
            "resets_at": iso8601(self.resets_at),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProviderRateLimit:
        return cls(
            name=str(data.get("name", "")),
            limit=_optional_int(data.get("limit")),
            remaining=_optional_int(data.get("remaining")),
            resets_at=parse_instant(data.get("resets_at")),
        )


def iso8601(moment: datetime | None) -> str | None:
    """ISO-8601 in UTC, with an explicit offset and no fractional seconds.

    ``2026-08-25T11:15:00+00:00`` — the shape the reference's date library emits
    and the shape prism-ts emits, rather than Python's default ``...+00:00``
    with microseconds attached. Microseconds are TRUNCATED, matching both, which
    matters for OpenAI's millisecond resets: ``6ms`` from now renders as the
    current second in all three languages rather than as one of three values.

    A naive datetime is read as UTC. Guessing the machine's zone instead would
    make the serialised form depend on where the process runs.
    """
    if moment is None:
        return None

    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)

    utc = moment.astimezone(timezone.utc).replace(microsecond=0)

    return utc.strftime("%Y-%m-%dT%H:%M:%S+00:00")


def parse_instant(value: Any) -> datetime | None:
    """A wire timestamp as an aware UTC datetime, or ``None`` when it is not one.

    Accepts the two shapes a provider actually sends: an epoch in seconds, and
    an RFC 3339 timestamp. Anything else — including a value a proxy mangled —
    is ``None``, never an exception. **The reference raises here**, so a single
    unparseable header aborts the parse of an otherwise successful response;
    a header is chosen by whatever is in front of the provider, and letting it
    decide whether a paid-for completion reaches the caller is not a trade this
    port makes.
    """
    if isinstance(value, bool) or value is None:
        return None

    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)

    if not isinstance(value, str) or value.strip() == "":
        return None

    text = value.strip()

    if _is_numeric(text):
        return datetime.fromtimestamp(float(text), tz=timezone.utc)

    try:
        parsed = datetime.fromisoformat(_normalise(text))
    except ValueError:
        return None

    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _normalise(text: str) -> str:
    """RFC 3339 spellings :func:`datetime.fromisoformat` rejects before 3.11.

    A trailing ``Z`` and a fractional part that is not exactly three or six
    digits are both legal on the wire and both rejected on 3.10, which this
    package still supports. Normalising here rather than raising the floor keeps
    the failure out of the one Python version most likely to be in production.
    """
    if text.endswith(("Z", "z")):
        text = f"{text[:-1]}+00:00"

    return _FRACTION.sub(lambda match: f".{match.group(1)[:6]:0<6}", text, count=1)


def _is_numeric(text: str) -> bool:
    """An epoch, the way the reference's ``is_numeric`` reads one."""
    return re.fullmatch(r"[+-]?\d+(\.\d+)?", text) is not None


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None
