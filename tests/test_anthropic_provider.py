"""The Anthropic provider and the ways its wire shape differs from OpenAI's."""

from __future__ import annotations

import json
from typing import Any

import pytest

from prism import HttpRequest, HttpResponse, Prism, PrismError, canonical
from prism.providers.anthropic.provider import Anthropic


class RecordingTransport:
    """Answers with a canned body and remembers what it was asked to send."""

    def __init__(self, body: dict[str, Any] | None = None, status: int = 200) -> None:
        self.body = body if body is not None else _ok_body()
        self.status = status
        self.sent: HttpRequest | None = None

    def send(self, request: HttpRequest) -> HttpResponse:
        self.sent = request
        return HttpResponse(status=self.status, body=canonical.encode(self.body).encode("utf-8"))


def _ok_body() -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5",
        "content": [{"type": "text", "text": "Hello."}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 2},
    }


def _sent_body(transport: RecordingTransport) -> dict[str, Any]:
    assert transport.sent is not None
    assert transport.sent.body is not None
    decoded: dict[str, Any] = json.loads(transport.sent.body.decode("utf-8"))
    return decoded


def test_posts_to_the_messages_endpoint_and_parses_the_reply() -> None:
    transport = RecordingTransport()

    response = (
        Prism.text()
        .using("anthropic", "claude-sonnet-4-5", {"transport": transport, "api_key": "sk-test"})
        .with_prompt("Hi")
        .with_max_tokens(64)
        .as_text()
    )

    assert transport.sent is not None
    assert transport.sent.url.endswith("/messages")
    assert response.text == "Hello."
    assert response.usage.prompt_tokens == 10


def test_authenticates_with_x_api_key_and_pins_the_version() -> None:
    # Anthropic does not take a bearer token, and the version header decides the
    # response SHAPE — a floating one would let a provider release change what
    # the parser receives without a line of code changing here.
    transport = RecordingTransport()

    Prism.text().using(
        "anthropic", "claude-sonnet-4-5", {"transport": transport, "api_key": "sk-test"}
    ).with_prompt("Hi").as_text()

    assert transport.sent is not None
    assert transport.sent.headers["x-api-key"] == "sk-test"
    assert "Authorization" not in transport.sent.headers
    assert transport.sent.headers["anthropic-version"] == "2023-06-01"


def test_always_sends_max_tokens() -> None:
    # The OpenAI body sends an explicit null for an unset limit. Doing that here
    # is a 400, so a default has to be chosen rather than omitted.
    transport = RecordingTransport()

    Prism.text().using("anthropic", "claude-sonnet-4-5", {"transport": transport}).with_prompt(
        "Hi"
    ).as_text()

    assert _sent_body(transport)["max_tokens"] == 4096


def test_system_prompts_go_in_the_top_level_field() -> None:
    transport = RecordingTransport()

    Prism.text().using(
        "anthropic", "claude-sonnet-4-5", {"transport": transport}
    ).with_system_prompt("Be brief.").with_prompt("Hi").as_text()

    body = _sent_body(transport)

    assert body["system"] == "Be brief."
    assert "Be brief." not in json.dumps(body["messages"])


def test_omits_tools_rather_than_sending_an_empty_list() -> None:
    # An empty list is falsy in Python but the key would still be emitted by a
    # naive mapping; sending `tools: []` changes tool_choice defaults on some
    # models and is rejected outright by others.
    transport = RecordingTransport()

    Prism.text().using("anthropic", "claude-sonnet-4-5", {"transport": transport}).with_prompt(
        "Hi"
    ).as_text()

    assert "tools" not in _sent_body(transport)


def test_joins_every_text_block_rather_than_taking_the_first() -> None:
    # Anthropic splits a reply across blocks when thinking or tool use
    # interleaves. Taking content[0] returns a truncated answer that looks
    # complete.
    body = _ok_body()
    body["content"] = [
        {"type": "thinking", "thinking": "considering"},
        {"type": "text", "text": "First. "},
        {"type": "text", "text": "Second."},
    ]
    transport = RecordingTransport(body)

    response = (
        Prism.text()
        .using("anthropic", "claude-sonnet-4-5", {"transport": transport})
        .with_prompt("Hi")
        .as_text()
    )

    assert response.text == "First. Second."


def test_reports_cache_tokens_without_subtracting_them() -> None:
    # Anthropic reports cache tokens SEPARATELY from input_tokens, unlike OpenAI
    # which nests them inside. Subtracting here would under-report the prompt.
    body = _ok_body()
    body["usage"] = {
        "input_tokens": 10,
        "output_tokens": 2,
        "cache_read_input_tokens": 7,
        "cache_creation_input_tokens": 3,
    }
    transport = RecordingTransport(body)

    response = (
        Prism.text()
        .using("anthropic", "claude-sonnet-4-5", {"transport": transport})
        .with_prompt("Hi")
        .as_text()
    )

    assert response.usage.prompt_tokens == 10
    assert response.usage.cache_read_input_tokens == 7


def test_reports_the_thinking_tokens_anthropic_sends() -> None:
    # Reported by the Moic Suite team against the live API. Anthropic puts
    # reasoning at usage.output_tokens_details.thinking_tokens, and this mapping
    # hardcoded thought_tokens=None -- while PHP and TypeScript simply never set
    # it. All three agreed, so no cross-language check could see it.
    #
    # The numbers matter as much as the field: 1240 thinking tokens INSIDE 2820
    # output tokens. A consumer pricing completion + thought bills the reasoning
    # twice, which is the expensive half.
    body = _ok_body()
    body["usage"] = {
        "input_tokens": 11,
        "output_tokens": 2820,
        "output_tokens_details": {"thinking_tokens": 1240},
    }
    transport = RecordingTransport(body)

    response = (
        Prism.text()
        .using("anthropic", "claude-sonnet-4-5", {"transport": transport})
        .with_prompt("Hi")
        .as_text()
    )

    assert response.usage.thought_tokens == 1240
    assert response.usage.completion_tokens == 2820
    # The breakdown claim, asserted rather than left to the comment.
    assert response.usage.thought_tokens < response.usage.completion_tokens


def test_leaves_thought_tokens_none_when_anthropic_reports_no_thinking() -> None:
    # The control. Without it the test above passes against a mapping that
    # hardcodes 1240, or one that invents a number when none was sent -- which
    # would make "the model did not reason" unreadable.
    body = _ok_body()
    body["usage"] = {"input_tokens": 11, "output_tokens": 2820}
    transport = RecordingTransport(body)

    response = (
        Prism.text()
        .using("anthropic", "claude-sonnet-4-5", {"transport": transport})
        .with_prompt("Hi")
        .as_text()
    )

    assert response.usage.thought_tokens is None


def test_raises_on_an_error_body_even_with_a_success_status() -> None:
    # Anthropic reports some failures with type: "error" and a 200.
    transport = RecordingTransport(
        {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
    )

    with pytest.raises(PrismError):
        Prism.text().using("anthropic", "claude-sonnet-4-5", {"transport": transport}).with_prompt(
            "Hi"
        ).as_text()


def test_raises_when_generation_was_cut_short() -> None:
    body = _ok_body()
    body["stop_reason"] = "max_tokens"
    transport = RecordingTransport(body)

    with pytest.raises(PrismError):
        Prism.text().using("anthropic", "claude-sonnet-4-5", {"transport": transport}).with_prompt(
            "Hi"
        ).as_text()


def test_an_unrecognised_stop_reason_is_unknown_not_stop() -> None:
    # Guessing STOP would present a truncated or refused generation as complete.
    body = _ok_body()
    body["stop_reason"] = "something_new"
    transport = RecordingTransport(body)

    response = (
        Prism.text()
        .using("anthropic", "claude-sonnet-4-5", {"transport": transport})
        .with_prompt("Hi")
        .as_text()
    )

    assert response.finish_reason.value == "unknown"


def test_the_provider_is_registered_under_its_key() -> None:
    from prism.registry import resolve_provider

    assert isinstance(resolve_provider("anthropic"), Anthropic)
