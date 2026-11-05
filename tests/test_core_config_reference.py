"""Independent expected values checked against normally installed PHP 0.146.0."""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from prism.fim.request import FimRequest
from prism.http import HttpRequest, HttpResponse
from prism.providers.anthropic.provider import Anthropic
from prism.providers.mistral.provider import Mistral
from prism.providers.openai.provider import OpenAI
from prism.text.request import Request

FIXTURE = json.loads((Path(__file__).parent / "fixtures/provider-configuration.json").read_text())
PROVIDERS: dict[str, Any] = {"anthropic": Anthropic, "openai": OpenAI, "mistral": Mistral}
NAMES = {"version": "api_version", "anthropic_beta": "beta_features"}
PROPERTIES = {
    "apiKey": "api_key",
    "apiVersion": "api_version",
    "betaFeatures": "beta_features",
    "apiFormat": "api_format",
}


class Capture:
    def __init__(self) -> None:
        self.calls: list[HttpRequest] = []

    def send(self, request: HttpRequest) -> HttpResponse:
        self.calls.append(request)
        return HttpResponse(
            status=200,
            body=json.dumps(
                {
                    "id": "fixture",
                    "model": "fixture",
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    "usage": {},
                }
            ).encode(),
        )


@pytest.mark.parametrize("entry", FIXTURE["cases"], ids=lambda entry: entry["id"])
def test_installed_php_configuration(
    entry: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse_network(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("Unowned network in configuration corpus.")

    monkeypatch.setattr("socket.getaddrinfo", refuse_network)
    monkeypatch.setattr("socket.socket.connect", refuse_network)
    row = {**entry["given"], "expected": entry["result"]["php"]}
    for name in FIXTURE["environment_names"]:
        monkeypatch.delenv(name, raising=False)
    for name, value in row.get("env", {}).items():
        monkeypatch.setenv(name, value)
    if row["kind"] == "config":
        config = {NAMES.get(name, name): value for name, value in row.get("config", {}).items()}
        if "error" in row["expected"]:
            with pytest.raises(TypeError):
                PROVIDERS[row["provider"]](**config)
        else:
            provider = PROVIDERS[row["provider"]](**config)
            assert (
                getattr(provider, PROPERTIES.get(row["property"], row["property"]))
                == row["expected"]["value"]
            )
        return
    if row["kind"] == "fim":
        from prism.providers.mistral import rate_limits

        calls: list[HttpRequest] = []

        class FimCapture:
            def send(self, request: HttpRequest) -> HttpResponse:
                calls.append(request)
                return HttpResponse(
                    status=200,
                    headers=row["headers"],
                    body=json.dumps(
                        {
                            "id": "fixture",
                            "model": "fixture",
                            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                        }
                    ).encode(),
                )

        class FrozenDateTime(datetime):
            @classmethod
            def now(cls, tz: Any = None) -> datetime:
                return datetime.fromtimestamp(row["now"], tz=timezone.utc).astimezone(tz)

        monkeypatch.setattr(rate_limits, "datetime", FrozenDateTime)
        response = Mistral(
            api_key="synthetic", url="https://offline.invalid/v1", transport=FimCapture()
        ).fim(FimRequest(model="fixture", prompt="prefix"))
        assert len(calls) == 1
        assert calls[0].url == "https://offline.invalid/v1/fim/completions"
        assert {
            "count": len(calls),
            "text": response.text,
            "rates": [rate.to_dict() for rate in response.meta.rate_limits],
        } == row["expected"]
        return
    capture = Capture()
    config = {"api_key": "synthetic", "transport": capture}
    if "defaultConfig" in row:
        config["default_thinking_budget"] = row["defaultConfig"]
    provider = Anthropic(**config)
    provider.text(
        Request(
            model="fixture",
            provider_options={"thinking": row["thinking"]},
            reasoning_enabled=row.get("reasoning"),
        )
    )
    assert len(capture.calls) == 1
    assert json.loads(capture.calls[0].body or b"{}").get("thinking") == row["expected"]["thinking"]


def test_explicit_null_beta_does_not_consume_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_BETA", "canonical-secret")
    monkeypatch.setenv("ANTHROPIC_BETA_FEATURES", "private-legacy")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert Anthropic(beta_features=None).beta_features is None
    assert caught == []


def test_legacy_beta_warns_once_only_if_consumed(monkeypatch: pytest.MonkeyPatch) -> None:
    from prism.providers import config as provider_config

    monkeypatch.setattr(provider_config, "_warned_legacy_beta", False)
    monkeypatch.setenv("ANTHROPIC_BETA", "")
    monkeypatch.setenv("ANTHROPIC_BETA_FEATURES", "private-legacy-value")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert Anthropic().beta_features == ""
        assert caught == []
        monkeypatch.delenv("ANTHROPIC_BETA")
        assert Anthropic().beta_features == "private-legacy-value"
        assert Anthropic().beta_features == "private-legacy-value"
    assert len(caught) == 1
    assert "ANTHROPIC_BETA" in str(caught[0].message)
    assert "private-legacy-value" not in str(caught[0].message)
