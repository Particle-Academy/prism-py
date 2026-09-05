"""Anthropic's ``anthropic-ratelimit-*`` headers, read into quota buckets.

**Generic, not enumerated.** Every header carrying the prefix is walked, split
into ``{bucket}-{field}``, and grouped — so ``requests``, ``tokens``,
``input-tokens`` and ``output-tokens`` all arrive without anyone listing them,
and a bucket Anthropic ships next quarter arrives for free. That is what the
reference does. prism-ts enumerates four names instead, which is the same answer
for today's four buckets and silently drops the fifth; the divergence is
recorded in the ``provider-rate-limits`` conformance suite rather than copied.
"""

from __future__ import annotations

from collections.abc import Mapping

from prism.providers.rate_limits import header_int, lowercased
from prism.value_objects.provider_rate_limit import ProviderRateLimit, parse_instant

__all__ = ["PREFIX", "parse_rate_limits"]

PREFIX = "anthropic-ratelimit-"


def parse_rate_limits(headers: Mapping[str, str]) -> list[ProviderRateLimit]:
    """Group the prefixed headers into one bucket each, in the order they arrive.

    ORDER IS THE HEADER ORDER, not a fixed list. Nothing errors when two
    languages order these differently — a caller reading ``rate_limits[0]``
    simply gets a different bucket, which is the hardest kind of divergence to
    notice.

    A bucket needs only ONE of its fields to exist. The reference reports a
    bucket whose ``limit`` arrived and whose ``remaining`` did not, with
    ``remaining`` left null; that is honest, and it is strictly more information
    than dropping the bucket. It is also the opposite of what prism-ts does.
    """
    buckets: dict[str, dict[str, str]] = {}

    for name, value in lowercased(headers).items():
        if not name.startswith(PREFIX):
            continue

        rest = name[len(PREFIX) :]
        bucket, separator, field = rest.rpartition("-")

        # A prefixed header with no second segment at all. The reference's
        # `beforeLast`/`afterLast` both return the whole string when the
        # delimiter is missing, so `anthropic-ratelimit-limit` becomes a bucket
        # NAMED "limit" carrying a `limit` field. Matched here rather than
        # tidied up: it is reachable from a malformed header, and a port that
        # quietly disagreed about it would disagree about a real response.
        if separator == "":
            bucket, field = rest, rest

        buckets.setdefault(bucket, {})[field] = value

    return [
        ProviderRateLimit(
            name=bucket,
            limit=header_int(fields["limit"]) if "limit" in fields else None,
            remaining=header_int(fields["remaining"]) if "remaining" in fields else None,
            # Anthropic sends RFC 3339, and has also sent a unix epoch; both are
            # accepted, and anything else is "no known reset" rather than an
            # exception that takes the whole response down with it.
            resets_at=parse_instant(fields.get("reset")),
        )
        for bucket, fields in buckets.items()
    ]
