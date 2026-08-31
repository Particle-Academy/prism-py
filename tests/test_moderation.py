"""Moderation: the one capability where failing open is the dangerous failure."""

from __future__ import annotations

import json
from typing import Any

import pytest

from prism import HttpRequest, HttpResponse, Prism, PrismError, canonical
from prism.providers.openai.moderation import parse_moderation_response
from prism.value_objects.moderation_result import ModerationResult

OK = {
    "id": "modr_1",
    "model": "omni-moderation-latest",
    "results": [
        {"flagged": False, "categories": {"violence": False}, "category_scores": {"violence": 0.01}},
        {"flagged": True, "categories": {"violence": True, "hate": False}, "category_scores": {"violence": 0.98}},
    ],
}


class RecordingTransport:
    def __init__(self, body: dict[str, Any], status: int = 200) -> None:
        self.body = body
        self.status = status
        self.sent: HttpRequest | None = None

    def send(self, request: HttpRequest) -> HttpResponse:
        self.sent = request
        return HttpResponse(status=self.status, body=canonical.encode(self.body).encode("utf-8"))


def _moderate(body: dict[str, Any], *inputs: str) -> Any:
    transport = RecordingTransport(body)
    response = (
        Prism.moderation()
        .using("openai", "omni-moderation-latest", {"api_key": "sk-test", "transport": transport})
        .with_input(*inputs)
        .as_moderation()
    )
    return response, transport


def test_every_input_is_posted_and_gets_a_verdict() -> None:
    response, transport = _moderate(OK, "safe", "not safe")

    assert transport.sent is not None
    assert transport.sent.url.endswith("/moderations")
    assert json.loads(transport.sent.body.decode("utf-8"))["input"] == ["safe", "not safe"]
    assert len(response.results) == 2


def test_flagged_is_true_when_any_input_was_flagged_not_just_the_first() -> None:
    # The question nearly every caller asks, and the one most likely to be got
    # wrong by hand: checking results[0] alone passes a batch whose SECOND input
    # was the problem.
    response, _ = _moderate(OK, "safe", "not safe")

    assert response.is_flagged() is True
    first = response.first_flagged()
    assert first is not None
    assert first.flagged_categories() == ["violence"]
    assert len(response.flagged()) == 1


def test_a_request_with_no_input_is_refused_rather_than_failing_open() -> None:
    # An empty call returns no results, is_flagged() is then False, and a caller
    # gating on it lets everything through.
    with pytest.raises(PrismError, match="at least one input"):
        Prism.moderation().using("openai", "omni-moderation-latest").to_request()


def test_a_malformed_reply_is_refused_rather_than_reporting_nothing_flagged() -> None:
    with pytest.raises(PrismError):
        parse_moderation_response(None, "omni-moderation-latest")


def test_a_non_boolean_category_is_dropped_rather_than_coerced() -> None:
    # Coercion would make the STRING "false" into True, and this is the one
    # value object where a wrong True means content gets blocked.
    result = ModerationResult.from_dict(
        {
            "flagged": False,
            "categories": {"violence": "false", "hate": True},
            "category_scores": {"violence": "high", "hate": 0.9},
        }
    )

    assert result.categories == {"hate": True}
    assert result.category_scores == {"hate": 0.9}


def test_a_missing_flagged_field_reads_as_not_flagged() -> None:
    # `is True` rather than truthiness: a provider that omitted the field has
    # not told us the content is safe, but reporting True on absence would block
    # everything a malformed reply touched.
    assert ModerationResult.from_dict({}).flagged is False


def test_the_requested_model_is_used_when_the_response_omits_it() -> None:
    parsed = parse_moderation_response({"results": []}, "omni-moderation-latest")

    assert parsed.meta.model == "omni-moderation-latest"
