"""The cross-language provider-rate-limits corpus from ``prism-parity``.

Quota is a fact about an ACCOUNT, not about a process. A PHP worker and a Python
worker drawing on one key both read these headers to decide whether to send the
next request, and a record of what the provider said is routinely written by one
service and read back by another. A disagreement never errors: one language
throttles on a bucket the other cannot see, or retries into a limit it believes
has lifted.

This suite was written while closing G-15 -- this port had no rate-limit reader
at all -- and its first run found that the two REFERENCES disagree with each
other on nine of sixteen rows, which the register had not predicted. Ten rows
disagreed in total; SEVEN do now. They are recorded rather than reconciled, and
pinned here so that closing any of them is a deliberate act.

Three closed, and this column moved for none of them: prism-ts learned the names
Mistral actually sends (G-41), and both references learned to compare a field
name case-insensitively (G-43, prl-0008) and to keep a response whose reset
header it cannot read (G-43, prl-0006). This port had all three from the start,
which is what decision 0002 buys.

Mirrors prism-ts/tests/rate-limits-corpus.test.ts case for case.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from prism.providers.anthropic.rate_limits import parse_rate_limits as anthropic_limits
from prism.providers.mistral.rate_limits import parse_rate_limits as mistral_limits
from prism.providers.openai.rate_limits import parse_rate_limits as openai_limits

CORPUS: dict[str, Any] = json.loads(
    (Path(__file__).parent / "fixtures" / "provider-rate-limits.json").read_text(encoding="utf-8")
)
CASES: list[dict[str, Any]] = CORPUS["cases"]


def _id(case: dict[str, Any]) -> str:
    return str(case["id"])


def _read(case: dict[str, Any]) -> dict[str, Any]:
    given = case["given"]
    headers = given["headers"]
    # Frozen per row: two of the three providers report a reset as a DURATION,
    # so an unfrozen clock would turn every one of those rows into a measurement
    # of elapsed time rather than a comparison.
    now = datetime.fromtimestamp(given["now"], tz=timezone.utc)

    try:
        # Header names are handed over with their case INTACT. Lower-casing here
        # would answer prl-0008 in the test instead of in the package.
        if given["provider"] == "anthropic":
            limits = anthropic_limits(headers)
        elif given["provider"] == "openai":
            limits = openai_limits(headers, now=now)
        else:
            limits = mistral_limits(headers, now=now)
    except Exception:  # that it raised at all is the record; the class and message are outside 0004
        return {"outcome": "raised", "buckets": None}

    return {"outcome": "ok", "buckets": [limit.to_dict() for limit in limits]}


def test_the_corpus_is_whole_not_a_subset_someone_trimmed_to_green() -> None:
    assert len(CASES) == 16


@pytest.mark.parametrize("case", CASES, ids=_id)
def test_reads_exactly_what_the_corpus_recorded(case: dict[str, Any]) -> None:
    assert _read(case) == case["result"]["py"]


def test_the_rows_that_disagree_are_the_ones_the_corpus_says_disagree() -> None:
    # Asserted from BOTH directions. A test that only checked the agreeing rows
    # would pass just as happily if a divergence were quietly closed, and this
    # suite's whole value is that closing one is deliberate.
    assert [case["id"] for case in CASES if not case["agrees"]] == [
        "prl-0001",
        "prl-0002",
        "prl-0003",
        "prl-0004",
        "prl-0005",
        "prl-0015",
        "prl-0016",
    ]


def test_follows_the_reference_where_the_reference_is_right() -> None:
    # The generic prefix walk, the epoch reset, the partial bucket and the
    # malformed bucket name. On all four this port agrees with the reference and
    # prism-ts does not, so a future "tidy-up" toward the other port would be a
    # regression rather than a convergence.
    for case_id in ("prl-0002", "prl-0003", "prl-0004", "prl-0016"):
        case = next(entry for entry in CASES if entry["id"] == case_id)
        assert case["result"]["py"] == case["result"]["php"], case_id
        assert case["result"]["py"] != case["result"]["ts"], case_id


def test_the_mistral_names_that_prism_ts_used_to_miss_now_agree_in_three() -> None:
    # prl-0014 was the largest single disagreement in the suite: Mistral sends
    # `ratelimitbysize-limit` with no bucket segment, prism-ts read
    # `ratelimitbysize-limit-tokens`, and every Mistral call in that port
    # therefore reported no quota at all. Closed in prism-ts (G-41) by moving to
    # the names this port and the reference already read -- so this row is now
    # evidence of convergence rather than of a gap, and it is asserted from the
    # agreeing side so that a regression in either direction fails.
    case = next(entry for entry in CASES if entry["id"] == "prl-0014")

    assert case["agrees"] is True
    assert case["result"]["py"] == case["result"]["php"] == case["result"]["ts"]


def test_diverges_from_the_reference_only_where_the_reference_is_wrong() -> None:
    # Decision 0002: the port contract is the golden, not the defect. Four rows
    # started here; TWO remain, because on the other two the REFERENCE was
    # brought to this port rather than the other way round.
    #
    #   prl-0005  the reference keeps the wire offset, so one instant stores as
    #             two different strings depending on who wrote the record
    #   prl-0015  the reference invents an exhausted bucket out of headers that
    #             were never sent
    #
    # Closed in the reference, 2026-09-05 (G-43):
    #
    #   prl-0006  the reference RAISED on an unparseable reset, so one mangled
    #             header aborted an already-paid-for completion
    #   prl-0008  the reference matched its prefix case-sensitively, so a
    #             title-casing proxy silently erased every rate limit
    divergent = {"prl-0005", "prl-0015"}

    for case in CASES:
        agrees_with_reference = case["result"]["py"] == case["result"]["php"]
        assert agrees_with_reference is (case["id"] not in divergent), case["id"]
