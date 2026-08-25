"""The OpenAI provider and the contract it implements."""

from __future__ import annotations

import json
from typing import Any

import pytest

from prism import (
    HttpRequest,
    HttpResponse,
    OpenAI,
    Prism,
    PrismError,
    Provider,
    canonical,
)


class RecordingTransport:
    """Answers with a canned body and remembers what it was asked to send."""

    def __init__(self, body: dict[str, Any] | None = None, status: int = 200) -> None:
        self.body = body if body is not None else _completed_body()
        self.status = status
        self.sent: HttpRequest | None = None

    def send(self, request: HttpRequest) -> HttpResponse:
        self.sent = request
        return HttpResponse(status=self.status, body=canonical.encode(self.body).encode("utf-8"))


def _completed_body() -> dict[str, Any]:
    return {
        "id": "resp_1",
        "status": "completed",
        "model": "gpt-4o-2024-08-06",
        "output": [
            {
                "id": "msg_1",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "I am a language model."}],
            }
        ],
        "usage": {
            "input_tokens": 11,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 56,
            "output_tokens_details": {"reasoning_tokens": 0},
        },
        "service_tier": "default",
    }


def test_as_text_sends_the_canonical_body_and_parses_the_reply() -> None:
    transport = RecordingTransport()

    response = (
        Prism.text()
        .using("openai", "gpt-4o", {"api_key": "sk-test", "transport": transport})
        .with_prompt("Who are you?")
        .as_text()
    )

    assert response.text == "I am a language model."

    assert transport.sent is not None
    assert transport.sent.method == "POST"
    assert transport.sent.url == "https://api.openai.com/v1/responses"
    assert transport.sent.body is not None
    assert transport.sent.body.decode("utf-8") == (
        '{"model":"gpt-4o","input":[{"role":"user","content":'
        '[{"type":"input_text","text":"Who are you?"}]}],"max_output_tokens":null}'
    )


def test_the_api_key_becomes_a_bearer_token() -> None:
    transport = RecordingTransport()
    OpenAI(api_key="sk-test", transport=transport).text(
        Prism.text().using("openai", "gpt-4o").with_prompt("hi").to_request()
    )

    assert transport.sent is not None
    assert transport.sent.headers["Authorization"] == "Bearer sk-test"
    assert transport.sent.headers["Content-Type"] == "application/json"


def test_organisation_and_project_headers_are_omitted_when_not_configured() -> None:
    transport = RecordingTransport()
    OpenAI(api_key="sk-test", transport=transport).text(
        Prism.text().using("openai", "gpt-4o").with_prompt("hi").to_request()
    )

    assert transport.sent is not None
    assert "OpenAI-Organization" not in transport.sent.headers
    assert "OpenAI-Project" not in transport.sent.headers


def test_organisation_and_project_headers_are_sent_when_configured() -> None:
    transport = RecordingTransport()
    OpenAI(
        api_key="sk-test",
        organization="org_1",
        project="proj_1",
        transport=transport,
    ).text(Prism.text().using("openai", "gpt-4o").with_prompt("hi").to_request())

    assert transport.sent is not None
    assert transport.sent.headers["OpenAI-Organization"] == "org_1"
    assert transport.sent.headers["OpenAI-Project"] == "proj_1"


def test_configuration_falls_back_to_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    monkeypatch.setenv("OPENAI_URL", "https://proxy.example.com/v1")
    monkeypatch.setenv("OPENAI_ORGANIZATION", "org_env")
    monkeypatch.setenv("OPENAI_PROJECT", "proj_env")

    provider = OpenAI()

    assert provider.api_key == "sk-env"
    assert provider.url == "https://proxy.example.com/v1"
    assert provider.organization == "org_env"
    assert provider.project == "proj_env"


def test_an_explicit_url_wins_over_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_URL", "https://proxy.example.com/v1")

    assert OpenAI(url="https://direct.example.com/v1/").url == "https://direct.example.com/v1"


def test_the_default_url_is_the_public_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_URL", raising=False)

    assert OpenAI().url == "https://api.openai.com/v1"


def test_an_http_error_status_is_refused_with_a_code() -> None:
    transport = RecordingTransport(
        body={"error": {"type": "invalid_request_error", "message": "bad model"}},
        status=400,
    )

    with pytest.raises(PrismError) as raised:
        OpenAI(api_key="sk-test", transport=transport).text(
            Prism.text().using("openai", "gpt-4o").with_prompt("hi").to_request()
        )

    assert raised.value.code == "provider_response_error"
    assert raised.value.status == 400
    assert json.loads(raised.value.body or "{}")["error"]["message"] == "bad model"


def test_a_non_json_body_is_refused_with_a_code() -> None:
    class BrokenTransport:
        def send(self, request: HttpRequest) -> HttpResponse:
            return HttpResponse(status=200, body=b"<html>gateway timeout</html>")

    with pytest.raises(PrismError) as raised:
        OpenAI(api_key="sk-test", transport=BrokenTransport()).text(
            Prism.text().using("openai", "gpt-4o").with_prompt("hi").to_request()
        )

    assert raised.value.code == "provider_response_error"


def test_the_chat_completions_format_is_refused_with_a_code() -> None:
    with pytest.raises(PrismError) as raised:
        OpenAI(api_key="sk-test", api_format="chat_completions").text(
            Prism.text().using("openai", "gpt-4o").with_prompt("hi").to_request()
        )

    assert raised.value.code == "unsupported_provider_action"


@pytest.mark.parametrize(
    "action",
    ["text", "structured", "embeddings", "images", "moderation", "stream"],
)
def test_the_base_contract_refuses_every_capability_with_a_code(action: str) -> None:
    with pytest.raises(PrismError) as raised:
        getattr(Provider(), action)(None)

    assert raised.value.code == "unsupported_provider_action"


def test_an_unimplemented_capability_on_openai_still_refuses() -> None:
    with pytest.raises(PrismError) as raised:
        OpenAI(api_key="sk-test").embeddings(None)

    assert raised.value.code == "unsupported_provider_action"
