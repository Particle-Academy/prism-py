"""Reading a provider's rate-limit headers, and what this port refuses to copy.

G-15: this port had no ``ProviderRateLimit`` at all and every provider reported
an empty list, while prism-ts parsed headers on all three. Closing it meant
choosing between two references that DISAGREE with each other, so the choices
are pinned here rather than left to whichever module is read first.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from prism import HttpRequest, HttpResponse, Prism, canonical
from prism.providers.anthropic.rate_limits import parse_rate_limits as anthropic_limits
from prism.providers.mistral.rate_limits import parse_rate_limits as mistral_limits
from prism.providers.openai.rate_limits import parse_rate_limits as openai_limits
from prism.value_objects import Meta, ProviderRateLimit

NOW = datetime(2026, 8, 25, 11, 15, 0, tzinfo=timezone.utc)


def _named(limits: list[ProviderRateLimit]) -> list[str]:
    return [limit.name for limit in limits]


# -- the value object ------------------------------------------------------


def test_serialises_a_reset_as_utc_with_an_explicit_offset_and_no_fraction() -> None:
    # The instant is the value; the rendering is normalised so two records of
    # the same moment are the same bytes whoever wrote them.
    limit = ProviderRateLimit(
        "tokens", 100, 40, datetime(2026, 8, 25, 11, 15, 0, 123456, tzinfo=timezone.utc)
    )

    assert limit.to_dict() == {
        "name": "tokens",
        "limit": 100,
        "remaining": 40,
        "resets_at": "2026-08-25T11:15:00+00:00",
    }


def test_renders_a_non_utc_instant_in_utc_rather_than_in_its_own_offset() -> None:
    berlin = timezone(timedelta(hours=2))
    limit = ProviderRateLimit("tokens", resets_at=datetime(2026, 8, 25, 13, 15, 0, tzinfo=berlin))

    assert limit.to_dict()["resets_at"] == "2026-08-25T11:15:00+00:00"


def test_round_trips_through_a_dict() -> None:
    limit = ProviderRateLimit("requests", 1000, 500, NOW)

    assert ProviderRateLimit.from_dict(limit.to_dict()) == limit


def test_a_bucket_the_provider_said_nothing_about_is_none_and_not_zero() -> None:
    # Zero means "you have none left" and None means "the provider did not say".
    # A caller that backs off on the second is throttling itself on a fact it
    # never received.
    assert ProviderRateLimit("tokens").to_dict() == {
        "name": "tokens",
        "limit": None,
        "remaining": None,
        "resets_at": None,
    }


def test_meta_carries_rate_limits_through_serialisation_and_back() -> None:
    meta = Meta(id="msg_1", model="m", rate_limits=[ProviderRateLimit("tokens", 8, 7, NOW)])
    rebuilt = Meta.from_dict(meta.to_dict())

    assert rebuilt.rate_limits == meta.rate_limits
    assert meta.to_dict()["rate_limits"] == [
        {"name": "tokens", "limit": 8, "remaining": 7, "resets_at": "2026-08-25T11:15:00+00:00"}
    ]


# -- Anthropic -------------------------------------------------------------


def test_reads_every_anthropic_bucket_in_the_order_the_headers_arrived() -> None:
    limits = anthropic_limits(
        {
            "anthropic-ratelimit-requests-limit": "1000",
            "anthropic-ratelimit-requests-remaining": "500",
            "anthropic-ratelimit-requests-reset": "2026-08-25T11:15:30Z",
            "anthropic-ratelimit-input-tokens-limit": "80000",
            "anthropic-ratelimit-input-tokens-remaining": "0",
            "anthropic-ratelimit-output-tokens-limit": "16000",
            "anthropic-ratelimit-output-tokens-remaining": "15000",
            "anthropic-ratelimit-tokens-limit": "96000",
            "anthropic-ratelimit-tokens-remaining": "15000",
        }
    )

    # Order is the HEADER order, not a fixed list. Nothing errors when two
    # languages order these differently -- a caller reading rate_limits[0] just
    # gets a different bucket.
    assert _named(limits) == ["requests", "input-tokens", "output-tokens", "tokens"]
    assert limits[0].limit == 1000
    assert limits[0].remaining == 500
    assert limits[0].resets_at == datetime(2026, 8, 25, 11, 15, 30, tzinfo=timezone.utc)


def test_discovers_a_bucket_nobody_enumerated() -> None:
    # The whole reason the Anthropic reader walks the prefix instead of listing
    # four names: a bucket the provider ships next quarter arrives for free.
    # prism-ts enumerates and would report nothing here.
    limits = anthropic_limits(
        {
            "anthropic-ratelimit-web-search-requests-limit": "60",
            "anthropic-ratelimit-web-search-requests-remaining": "59",
        }
    )

    assert _named(limits) == ["web-search-requests"]
    assert limits[0].limit == 60


def test_reports_a_bucket_whose_remaining_never_arrived() -> None:
    limits = anthropic_limits({"anthropic-ratelimit-tokens-limit": "96000"})

    assert limits == [ProviderRateLimit("tokens", limit=96000, remaining=None, resets_at=None)]


def test_accepts_a_reset_sent_as_a_unix_epoch() -> None:
    limits = anthropic_limits(
        {
            "anthropic-ratelimit-requests-limit": "1000",
            "anthropic-ratelimit-requests-reset": "1735689600",
        }
    )

    assert limits[0].resets_at == datetime(2025, 1, 1, tzinfo=timezone.utc)


def test_a_malformed_reset_loses_the_reset_and_not_the_response() -> None:
    # THE ADVERSARIAL ONE. The reference raises here, so a single unreadable
    # header aborts the parse of an otherwise successful, already-paid-for
    # completion -- and a header is chosen by whatever proxy sits in front of
    # the provider.
    limits = anthropic_limits(
        {
            "anthropic-ratelimit-requests-limit": "1000",
            "anthropic-ratelimit-requests-remaining": "500",
            "anthropic-ratelimit-requests-reset": "soon",
        }
    )

    assert limits == [ProviderRateLimit("requests", 1000, 500, None)]


def test_a_limit_that_is_not_a_number_reads_as_zero_rather_than_raising() -> None:
    limits = anthropic_limits({"anthropic-ratelimit-requests-limit": "many"})

    assert limits[0].limit == 0


def test_matches_the_prefix_whatever_case_a_proxy_used() -> None:
    # HTTP field names are case-insensitive (RFC 9110 5.1). Both references used
    # to compare their prefix case-sensitively, so one title-casing proxy made
    # them report no rate limits at all -- invisibly, since that is also what a
    # response without the headers looks like. This port had it from the start;
    # the other two were brought here on 2026-09-05.
    limits = anthropic_limits(
        {
            "Anthropic-RateLimit-Requests-Limit": "1000",
            "Anthropic-RateLimit-Requests-Remaining": "500",
        }
    )

    assert limits == [ProviderRateLimit("requests", 1000, 500, None)]


def test_the_fold_is_ascii_only_and_will_not_invent_a_bucket_from_a_lookalike() -> None:
    # THE REASON THE FOLD IS NOT str.lower(). U+212A KELVIN SIGN lower-cases to a
    # plain ASCII `k`, so this header would come back as a bucket named `tokens`
    # -- the name a caller matches on to decide whether it has token quota left,
    # manufactured out of a field the provider never sent. An HTTP field name is
    # an RFC 9110 `token` and ASCII by grammar, so a name carrying this codepoint
    # is a DIFFERENT name and stays one.
    name = "anthropic-ratelimit-to\u212aens-limit"
    assert name.lower() == "anthropic-ratelimit-tokens-limit"

    assert _named(anthropic_limits({name: "96000"})) == ["to\u212aens"]


def test_the_fold_never_changes_the_length_of_a_name() -> None:
    # U+0130 lower-cases to TWO codepoints (`i` + U+0307), which would make the
    # bucket name this reader derives a different string in each of the three
    # languages implementing the parser -- a cross-language divergence produced
    # by the fix rather than removed by it.
    name = "anthropic-ratelimit-\u0130nput-tokens-limit"
    assert len(name.lower()) > len(name)

    assert _named(anthropic_limits({name: "80000"})) == ["\u0130nput-tokens"]


def test_a_response_with_no_rate_limit_headers_reports_no_buckets() -> None:
    assert anthropic_limits({"content-type": "application/json"}) == []


# -- OpenAI ----------------------------------------------------------------


def test_reads_both_openai_buckets_and_resolves_the_duration_against_now() -> None:
    limits = openai_limits(
        {
            "x-ratelimit-limit-requests": "10000",
            "x-ratelimit-remaining-requests": "9999",
            "x-ratelimit-reset-requests": "6ms",
            "x-ratelimit-limit-tokens": "200000",
            "x-ratelimit-remaining-tokens": "199000",
            "x-ratelimit-reset-tokens": "1h",
        },
        now=NOW,
    )

    assert _named(limits) == ["requests", "tokens"]
    assert limits[0].resets_at == NOW + timedelta(milliseconds=6)
    assert limits[1].resets_at == NOW + timedelta(hours=1)


def test_a_zero_reset_is_no_known_reset_rather_than_reset_now() -> None:
    limits = openai_limits(
        {
            "x-ratelimit-limit-tokens": "1",
            "x-ratelimit-remaining-tokens": "0",
            "x-ratelimit-reset-tokens": "0",
        },
        now=NOW,
    )

    assert limits[0].resets_at is None


def test_a_compound_duration_is_not_matched_by_either_reference_or_by_this_port() -> None:
    # OpenAI really does send `1m30s`. Both references anchor the pattern at both
    # ends and report no reset; this port agrees rather than quietly becoming the
    # only language that understands it.
    limits = openai_limits(
        {
            "x-ratelimit-limit-tokens": "1",
            "x-ratelimit-remaining-tokens": "0",
            "x-ratelimit-reset-tokens": "1m30s",
        },
        now=NOW,
    )

    assert limits[0].resets_at is None


def test_an_openai_bucket_needs_both_halves_or_it_is_not_reported() -> None:
    assert openai_limits({"x-ratelimit-limit-tokens": "200000"}, now=NOW) == []


# -- Mistral ---------------------------------------------------------------


def test_reads_mistrals_single_bucket_from_the_headers_it_actually_sends() -> None:
    # `ratelimitbysize-limit`, with NO bucket segment. prism-ts reads
    # `ratelimitbysize-limit-tokens` and therefore returns nothing at all here.
    limits = mistral_limits(
        {
            "ratelimitbysize-limit": "500000",
            "ratelimitbysize-remaining": "499900",
            "ratelimitbysize-reset": "28",
        },
        now=NOW,
    )

    assert limits == [ProviderRateLimit("tokens", 500000, 499900, NOW + timedelta(seconds=28))]


def test_a_mistral_response_that_said_nothing_is_not_reported_as_exhausted() -> None:
    # The reference emits this bucket unconditionally, so an absent header set
    # comes back as limit 0, remaining 0, resets now: a provider that said
    # nothing, reported as out of quota and ready to retry immediately.
    assert mistral_limits({"content-type": "application/json"}, now=NOW) == []


# -- through a provider ----------------------------------------------------


class HeaderTransport:
    """Answers with a canned body and a canned set of response headers."""

    def __init__(self, body: dict[str, Any], headers: dict[str, str]) -> None:
        self.body = body
        self.headers = headers

    def send(self, request: HttpRequest) -> HttpResponse:
        return HttpResponse(
            status=200,
            body=canonical.encode(self.body).encode("utf-8"),
            headers=self.headers,
        )


def test_an_anthropic_call_carries_the_quota_it_was_told_about() -> None:
    transport = HeaderTransport(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [{"type": "text", "text": "Hello."}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 2},
        },
        {
            "anthropic-ratelimit-requests-limit": "1000",
            "anthropic-ratelimit-requests-remaining": "500",
            "anthropic-ratelimit-requests-reset": "2026-08-25T11:15:30Z",
        },
    )

    response = (
        Prism.text()
        .using("anthropic", "claude-sonnet-4-5", {"transport": transport, "api_key": "sk-test"})
        .with_prompt("Hi")
        .with_max_tokens(64)
        .as_text()
    )

    assert response.meta.rate_limits == [
        ProviderRateLimit(
            "requests", 1000, 500, datetime(2026, 8, 25, 11, 15, 30, tzinfo=timezone.utc)
        )
    ]
    # And it survives the trip through storage, which is the only reason a
    # downstream service ever sees it.
    stored = json.loads(canonical.encode(response.to_dict()))
    assert stored["steps"][0]["meta"]["rate_limits"][0]["resets_at"] == "2026-08-25T11:15:30+00:00"


def test_a_mistral_call_carries_the_quota_it_was_told_about() -> None:
    transport = HeaderTransport(
        {
            "id": "cmpl_1",
            "model": "mistral-small-latest",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hi."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        },
        {"ratelimitbysize-limit": "500000", "ratelimitbysize-remaining": "499900"},
    )

    response = (
        Prism.text()
        .using("mistral", "mistral-small-latest", {"transport": transport, "api_key": "sk-test"})
        .with_prompt("Hi")
        .as_text()
    )

    assert [(limit.name, limit.limit, limit.remaining) for limit in response.meta.rate_limits] == [
        ("tokens", 500000, 499900)
    ]
