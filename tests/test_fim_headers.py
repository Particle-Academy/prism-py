"""A06: quota headers must survive an actual FIM transport round trip."""

from __future__ import annotations

import json

import pytest

from prism.fim.request import FimRequest
from prism.http import HttpRequest, HttpResponse
from prism.providers.mistral.provider import Mistral


class FimTransport:
    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers
        self.calls: list[HttpRequest] = []

    def send(self, request: HttpRequest) -> HttpResponse:
        self.calls.append(request)
        return HttpResponse(
            status=200,
            headers=self.headers,
            body=json.dumps(
                {
                    "id": "fixture",
                    "model": "fixture",
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }
            ).encode(),
        )


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        (
            {
                "ratelimitbysize-limit": "100",
                "ratelimitbysize-remaining": "75",
                "ratelimitbysize-reset": "5",
            },
            (100, 75),
        ),
        (
            {
                "RateLimitBySize-LIMIT": "100",
                "RATELIMITBYSIZE-remaining": "75",
                "RateLimitBySize-RESET": "5",
            },
            (100, 75),
        ),
        ({}, None),
        ({"ratelimitbysize-remaining": "75", "ratelimitbysize-reset": "5"}, None),
        ({"ratelimitbysize-limit": "100", "ratelimitbysize-reset": "5"}, None),
        (
            {
                "ratelimitbysize-limit": "0",
                "ratelimitbysize-remaining": "0",
                "ratelimitbysize-reset": "5",
            },
            (0, 0),
        ),
    ],
)
def test_fim_preserves_actual_rate_headers(
    headers: dict[str, str], expected: tuple[int, int] | None
) -> None:
    transport = FimTransport(headers)
    response = Mistral(
        api_key="synthetic", url="https://offline.invalid/v1", transport=transport
    ).fim(FimRequest(model="fixture", prompt="prefix"))
    assert len(transport.calls) == 1
    assert transport.calls[0].url == "https://offline.invalid/v1/fim/completions"
    assert response.text == "ok"
    if expected is None:
        assert response.meta.rate_limits == []
    else:
        assert len(response.meta.rate_limits) == 1
        rate = response.meta.rate_limits[0]
        assert rate.name == "tokens"
        assert (rate.limit, rate.remaining) == expected
        assert rate.resets_at is not None
