"""Mistral's ``ratelimitbysize-*`` headers, read into one quota bucket.

One word, no separators, and NO BUCKET SEGMENT — ``ratelimitbysize-limit``, not
``ratelimitbysize-limit-tokens``. Mistral meters by size, so the single bucket
it reports is named ``tokens``, which is what the reference names it and what
its own tests assert against.

prism-ts reads ``ratelimitbysize-limit-requests`` and
``ratelimitbysize-limit-tokens`` instead, and therefore returns NOTHING for the
headers the reference handles. That is the largest disagreement the
``provider-rate-limits`` suite found and it is a defect in that port, not a
choice available to this one.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta, timezone

from prism.providers.rate_limits import header_int, lowercased
from prism.value_objects.provider_rate_limit import ProviderRateLimit

__all__ = ["parse_rate_limits"]


def parse_rate_limits(
    headers: Mapping[str, str],
    now: datetime | None = None,
) -> list[ProviderRateLimit]:
    """The ``tokens`` bucket, when Mistral reported one.

    **Only when it reported one.** The reference emits this bucket
    unconditionally, so a response carrying no rate-limit headers at all comes
    back saying limit 0, remaining 0, resets now — a provider that said nothing
    reported as exhausted, from a cast over an absent header. A caller cannot
    tell that apart from a real exhaustion, and both readings of it are wrong:
    back off, or retry immediately into a limit that was never there.
    """
    found = lowercased(headers)
    limit = found.get("ratelimitbysize-limit")
    remaining = found.get("ratelimitbysize-remaining")

    if limit is None or remaining is None:
        return []

    return [
        ProviderRateLimit(
            name="tokens",
            limit=header_int(limit),
            remaining=header_int(remaining),
            resets_at=_resets_at(found.get("ratelimitbysize-reset"), now),
        )
    ]


def _resets_at(value: str | None, now: datetime | None) -> datetime | None:
    """SECONDS from now — not a timestamp, and not OpenAI's ``1m30s`` string.

    Absent means unknown, so it stays ``None``. The reference casts the missing
    header to ``0`` and reports a reset of *now*, which reads as "already
    reset" and sends a caller straight back at the provider.
    """
    if value is None or value.strip() == "":
        return None

    moment = now if now is not None else datetime.now(tz=timezone.utc)

    return moment + timedelta(seconds=header_int(value))
