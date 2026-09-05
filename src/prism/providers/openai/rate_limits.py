"""OpenAI's ``x-ratelimit-*`` headers, read into quota buckets.

Two buckets, ``requests`` and ``tokens``, ENUMERATED rather than discovered —
which is what the reference does here and the opposite of what it does for
Anthropic. The difference is not an inconsistency to tidy up: OpenAI's names put
the bucket LAST (``x-ratelimit-limit-requests``), so a generic walk cannot tell
a bucket name from a field name without already knowing the field names.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone

from prism.providers.rate_limits import header_int, lowercased
from prism.value_objects.provider_rate_limit import ProviderRateLimit

__all__ = ["BUCKETS", "parse_rate_limits"]

BUCKETS = ("requests", "tokens")

#: ``6ms``, ``30s``, ``5m``, ``1h``. Anchored at both ends, so OpenAI's compound
#: form (``1m30s``) does NOT match and is reported as no known reset. That is
#: what both references do, and it is a gap both of them have rather than a
#: choice this port made — recorded in the ``provider-rate-limits`` suite.
_DURATION = re.compile(r"^(\d+)(ms|s|m|h)$")

_UNITS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}


def parse_rate_limits(
    headers: Mapping[str, str],
    now: datetime | None = None,
) -> list[ProviderRateLimit]:
    """Read the two buckets OpenAI reports.

    ``now`` is injectable because a reset here is a DURATION and the value
    produced therefore depends on the clock. A parser that reads the clock
    itself cannot be pinned by a fixture, and a rate-limit reset is exactly the
    kind of value where the three languages would otherwise be compared while
    each held a different ``now``.
    """
    moment = now if now is not None else datetime.now(tz=timezone.utc)
    found = lowercased(headers)
    limits: list[ProviderRateLimit] = []

    for bucket in BUCKETS:
        limit = found.get(f"x-ratelimit-limit-{bucket}")
        remaining = found.get(f"x-ratelimit-remaining-{bucket}")

        # Both or neither. A bucket with a limit and no remaining tells a caller
        # nothing actionable, and reporting the missing half as zero would say
        # the quota is exhausted when nothing said so.
        if limit is None or remaining is None:
            continue

        limits.append(
            ProviderRateLimit(
                name=bucket,
                limit=header_int(limit),
                remaining=header_int(remaining),
                resets_at=_resets_at(found.get(f"x-ratelimit-reset-{bucket}"), moment),
            )
        )

    return limits


def _resets_at(value: str | None, now: datetime) -> datetime | None:
    """A duration from now, or ``None`` when the header says nothing usable.

    ``"0"`` is treated as absent rather than as "resets immediately", matching
    the reference: a zero-second reset would have a caller retry straight back
    into the same limit.
    """
    if value is None or value in ("", "0"):
        return None

    match = _DURATION.match(value)

    if match is None:
        return None

    return now + timedelta(seconds=int(match.group(1)) * _UNITS[match.group(2)])
