"""The Mistral provider: chat completions, FIM, embeddings, streaming."""

from __future__ import annotations

import json
from typing import Any

import pytest

from prism import (
    FinishReason,
    HttpRequest,
    HttpResponse,
    Mistral,
    Prism,
    PrismError,
    Tool,
    ToolChoice,
    canonical,
)
from prism.providers.mistral.fim import parse_fim_response

CHAT: dict[str, Any] = {
    "id": "cmpl-1",
    "object": "chat.completion",
    "model": "mistral-large-latest",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Bonjour"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12},
}

FIM: dict[str, Any] = {
    "id": "fim-1",
    "model": "codestral-latest",
    "choices": [{"index": 0, "message": {"content": "    return a + b"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 12, "completion_tokens": 6},
}


class RecordingTransport:
    def __init__(self, body: dict[str, Any], status: int = 200) -> None:
        self.body = body
        self.status = status
        self.sent: HttpRequest | None = None

    def send(self, request: HttpRequest) -> HttpResponse:
        self.sent = request
        return HttpResponse(status=self.status, body=canonical.encode(self.body).encode("utf-8"))


def _sent(transport: RecordingTransport) -> dict[str, Any]:
    assert transport.sent is not None
    assert transport.sent.body is not None
    parsed = json.loads(transport.sent.body.decode("utf-8"))
    assert isinstance(parsed, dict)
    return parsed


# -- text ------------------------------------------------------------------


def test_the_chat_completions_shape_is_posted_and_the_reply_read() -> None:
    transport = RecordingTransport(CHAT)

    response = (
        Prism.text()
        .using("mistral", "mistral-large-latest", {"api_key": "sk-test", "transport": transport})
        .with_prompt("Say hello in French")
        .as_text()
    )

    assert transport.sent is not None
    assert transport.sent.url == "https://api.mistral.ai/v1/chat/completions"
    assert transport.sent.headers["Authorization"] == "Bearer sk-test"
    # `messages`, not the Responses API's `input`. Sharing the OpenAI mapper
    # would send the wrong envelope to an endpoint that looks compatible.
    assert _sent(transport)["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "Say hello in French"}]}
    ]
    assert response.text == "Bonjour"
    assert response.finish_reason is FinishReason.STOP
    assert response.usage.prompt_tokens == 9


def test_system_prompts_come_first() -> None:
    # Mistral weights the earliest system turn most heavily, and a caller who
    # set one expects it to lead.
    transport = RecordingTransport(CHAT)

    (
        Prism.text()
        .using("mistral", "mistral-large-latest", {"api_key": "sk-test", "transport": transport})
        .with_system_prompt("You are terse.")
        .with_prompt("Hi")
        .as_text()
    )

    messages = _sent(transport)["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_the_finish_reason_is_read_off_the_choice_not_the_root() -> None:
    # The chat-completions shape puts it there. Reading the root returns nothing
    # and reports every generation as Unknown.
    transport = RecordingTransport(
        {
            **CHAT,
            "choices": [
                {"index": 0, "message": {"content": "x"}, "finish_reason": "content_filter"}
            ],
        }
    )

    response = (
        Prism.text()
        .using("mistral", "mistral-large-latest", {"api_key": "sk-test", "transport": transport})
        .with_prompt("Hi")
        .as_text()
    )

    assert response.finish_reason is FinishReason.CONTENT_FILTER


def test_typed_content_chunks_are_joined_rather_than_stringified() -> None:
    # Reasoning models return a list. Stringifying it yields a Python repr,
    # which reaches the caller as text the model never produced.
    transport = RecordingTransport(
        {
            **CHAT,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "content": [
                            {"type": "thinking", "thinking": [{"type": "text", "text": "hmm"}]},
                            {"type": "text", "text": "the answer"},
                        ]
                    },
                    "finish_reason": "stop",
                }
            ],
        }
    )

    response = (
        Prism.text()
        .using("mistral", "magistral-medium-latest", {"api_key": "sk-test", "transport": transport})
        .with_prompt("Hi")
        .as_text()
    )

    assert response.text == "the answer"
    assert response.additional_content["thinking"] == "hmm"


def test_an_empty_tool_list_is_omitted_rather_than_sent() -> None:
    transport = RecordingTransport(CHAT)

    (
        Prism.text()
        .using("mistral", "mistral-large-latest", {"api_key": "sk-test", "transport": transport})
        .with_prompt("Hi")
        .as_text()
    )

    body = _sent(transport)
    assert "tools" not in body
    assert "tool_choice" not in body


def test_tools_map_into_the_nested_chat_completions_shape() -> None:
    transport = RecordingTransport(CHAT)

    (
        Prism.text()
        .using("mistral", "mistral-large-latest", {"api_key": "sk-test", "transport": transport})
        .with_prompt("Weather?")
        .with_tools(
            [
                Tool()
                .as_("weather")
                .for_("Look up weather")
                .with_string_parameter("city", "The city")
            ]
        )
        .with_tool_choice(ToolChoice.ANY)
        .as_text()
    )

    body = _sent(transport)
    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Look up weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"description": "The city", "type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    # A bare string, not OpenAI's `required` and not Anthropic's object.
    assert body["tool_choice"] == "any"


def test_a_bare_message_error_is_read() -> None:
    # That is what a malformed request actually gets back; reading only the
    # nested `error.message` reports "unknown" for the most common mistake.
    transport = RecordingTransport(
        {"object": "error", "message": "Invalid model", "type": "invalid_request_error"}, status=422
    )

    with pytest.raises(PrismError, match="Invalid model"):
        (
            Prism.text()
            .using("mistral", "nope", {"api_key": "sk-test", "transport": transport})
            .with_prompt("Hi")
            .as_text()
        )


# -- fim -------------------------------------------------------------------


def test_a_prompt_and_a_suffix_are_posted_to_the_fim_endpoint() -> None:
    transport = RecordingTransport(FIM)

    response = (
        Prism.fim()
        .using("mistral", "codestral-latest", {"api_key": "sk-test", "transport": transport})
        .with_prompt("def add(a, b):\n")
        .with_suffix("\n\nprint(add(1, 2))")
        .as_text()
    )

    assert transport.sent is not None
    # A DIFFERENT endpoint from chat, not a mode of it.
    assert transport.sent.url == "https://api.mistral.ai/v1/fim/completions"
    assert _sent(transport) == {
        "model": "codestral-latest",
        "prompt": "def add(a, b):\n",
        "suffix": "\n\nprint(add(1, 2))",
    }
    assert response.text == "    return a + b"
    assert response.finish_reason is FinishReason.STOP
    assert response.usage.completion_tokens == 6


def test_the_suffix_is_omitted_when_none_was_set() -> None:
    # No suffix means "complete to the end", a different request rather than a
    # degraded version of a suffixed one.
    transport = RecordingTransport(FIM)

    (
        Prism.fim()
        .using("mistral", "codestral-latest", {"api_key": "sk-test", "transport": transport})
        .with_prompt("x")
        .as_text()
    )

    assert "suffix" not in _sent(transport)


def test_a_temperature_of_zero_survives_and_an_empty_stop_list_does_not() -> None:
    # 0 is a real setting; an empty list sent as `"stop": []` reads as a caller
    # who chose no stop sequences rather than one who never set any.
    transport = RecordingTransport(FIM)

    (
        Prism.fim()
        .using("mistral", "codestral-latest", {"api_key": "sk-test", "transport": transport})
        .with_prompt("x")
        .with_temperature(0)
        .as_text()
    )

    body = _sent(transport)
    assert body["temperature"] == 0
    assert "stop" not in body


def test_a_single_stop_string_is_wrapped() -> None:
    transport = RecordingTransport(FIM)

    (
        Prism.fim()
        .using("mistral", "codestral-latest", {"api_key": "sk-test", "transport": transport})
        .with_prompt("x")
        .with_stop("\n\n")
        .as_text()
    )

    assert _sent(transport)["stop"] == ["\n\n"]


def test_a_length_finish_does_not_raise() -> None:
    # Unlike a chat call. Hitting the ceiling is an ordinary outcome for a
    # completion: the caller wanted as much of the gap as the budget bought, and
    # the partial text is useful.
    transport = RecordingTransport(
        {
            **FIM,
            "choices": [
                {"index": 0, "message": {"content": "    return a"}, "finish_reason": "length"}
            ],
        }
    )

    response = (
        Prism.fim()
        .using("mistral", "codestral-latest", {"api_key": "sk-test", "transport": transport})
        .with_prompt("x")
        .with_max_tokens(4)
        .as_text()
    )

    assert response.finish_reason is FinishReason.LENGTH
    assert response.text == "    return a"


def test_an_unrecognised_fim_finish_reason_is_unknown_not_stop() -> None:
    # A truncated completion reported as complete is how an editor silently
    # inserts half a function.
    parsed = parse_fim_response(
        {
            **FIM,
            "choices": [{"index": 0, "message": {"content": "x"}, "finish_reason": "reticulating"}],
        },
        Prism.fim().using("mistral", "codestral-latest", {"api_key": "sk-test"}).to_request(),
    )

    assert parsed.finish_reason is FinishReason.UNKNOWN


def test_fim_is_refused_by_a_provider_that_has_no_such_endpoint() -> None:
    with pytest.raises(PrismError) as error:
        (Prism.fim().using("openai", "gpt-4o", {"api_key": "sk-test"}).with_prompt("x").as_text())

    assert error.value.code == "unsupported_provider_action"


# -- embeddings ------------------------------------------------------------


def test_embeddings_send_model_and_input_only_and_order_by_index() -> None:
    # Unknown keys are rejected outright by this endpoint rather than ignored,
    # so OpenAI's `dimensions` is not forwarded.
    transport = RecordingTransport(
        {
            "id": "emb-1",
            "model": "mistral-embed",
            "data": [
                {"index": 1, "embedding": [0.3, 0.4]},
                {"index": 0, "embedding": [0.1, 0.2]},
            ],
            "usage": {"total_tokens": 7},
        }
    )

    response = (
        Prism.embeddings()
        .using("mistral", "mistral-embed", {"api_key": "sk-test", "transport": transport})
        .from_array(["first", "second"])
        .with_provider_options({"dimensions": 512})
        .as_embeddings()
    )

    assert _sent(transport) == {"model": "mistral-embed", "input": ["first", "second"]}
    assert response.embeddings[0].embedding == (0.1, 0.2)
    assert response.usage.tokens == 7


# -- configuration ---------------------------------------------------------


def test_the_default_url_is_used_when_none_is_configured() -> None:
    assert Mistral(api_key="sk-test").url == "https://api.mistral.ai/v1"


def test_a_trailing_slash_is_stripped_so_paths_do_not_double_up() -> None:
    assert (
        Mistral(api_key="sk-test", url="https://gateway.test/v1/").url == "https://gateway.test/v1"
    )


def test_the_authorization_header_is_omitted_entirely_when_no_key_is_set() -> None:
    assert "Authorization" not in Mistral(api_key="")._headers(0)
