"""A tool call whose arguments could not be parsed, as a request carries it."""

from __future__ import annotations

import json
from typing import Any

import pytest

from prism import (
    AssistantMessage,
    HttpRequest,
    HttpResponse,
    Prism,
    PrismError,
    ToolCall,
    ToolResult,
    ToolResultMessage,
    UserMessage,
    canonical,
)

MALFORMED = '{"city": '

RESPONSES: dict[str, dict[str, Any]] = {
    "openai": {
        "id": "r1",
        "model": "m",
        "status": "completed",
        "output": [
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "ok"}],
            }
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    },
    "anthropic": {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "m",
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    },
    "mistral": {
        "id": "c1",
        "model": "m",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    },
}


class RecordingTransport:
    def __init__(self, provider: str) -> None:
        self.body = RESPONSES[provider]
        self.sent: HttpRequest | None = None

    def send(self, request: HttpRequest) -> HttpResponse:
        self.sent = request
        return HttpResponse(status=200, body=canonical.encode(self.body).encode("utf-8"))


def _answered(arguments: str) -> list[Any]:
    return [
        UserMessage("Weather?"),
        AssistantMessage("", [ToolCall("call_1", "weather", arguments, "call_1")]),
        ToolResultMessage(
            [
                ToolResult(
                    "call_1", "weather", {}, "Tool call arguments are not valid JSON", "call_1"
                )
            ]
        ),
        UserMessage("Again"),
    ]


def _failure_of(pending: Any) -> str | None:
    """A request that fails to build fails its test on an assertion."""
    try:
        pending.as_text()
    except PrismError as error:
        return str(error)
    return None


def _sent_body(transport: RecordingTransport) -> dict[str, Any]:
    assert transport.sent is not None
    body = transport.sent.body
    decoded: dict[str, Any] = json.loads(body) if isinstance(body, (str, bytes)) else body
    return decoded


def _replayed(provider: str, transport: RecordingTransport) -> Any:
    body = _sent_body(transport)

    if provider == "openai":
        return next(i for i in body["input"] if i.get("type") == "function_call")["arguments"]

    if provider == "mistral":
        call = next(m for m in body["messages"] if m.get("tool_calls"))
        return call["tool_calls"][0]["function"]["arguments"]

    assistant = next(m for m in body["messages"] if m["role"] == "assistant")
    return next(b for b in assistant["content"] if b["type"] == "tool_use")["input"]


@pytest.mark.parametrize(
    ("provider", "empty"), [("openai", "{}"), ("anthropic", {}), ("mistral", "{}")]
)
def test_replays_empty_once_a_tool_result_has_answered_it(provider: str, empty: Any) -> None:
    transport = RecordingTransport(provider)
    history = _answered(MALFORMED)

    pending = Prism.text().using(provider, "m", {"transport": transport}).with_messages(history)

    assert _failure_of(pending) is None
    assert _replayed(provider, transport) == empty
    assert history[1].tool_calls[0].arguments == MALFORMED


@pytest.mark.parametrize("provider", ["openai", "anthropic", "mistral"])
def test_leaves_well_formed_arguments_alone(provider: str) -> None:
    transport = RecordingTransport(provider)

    pending = (
        Prism.text()
        .using(provider, "m", {"transport": transport})
        .with_messages(_answered('{"city":"Paris"}'))
    )

    assert _failure_of(pending) is None

    sent = _replayed(provider, transport)
    assert (json.loads(sent) if isinstance(sent, str) else sent) == {"city": "Paris"}


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_still_fails_at_mapping_when_nothing_has_answered_it(provider: str) -> None:
    transport = RecordingTransport(provider)
    pending = (
        Prism.text()
        .using(provider, "m", {"transport": transport})
        .with_messages(
            [
                UserMessage("Weather?"),
                AssistantMessage("", [ToolCall("call_1", "weather", MALFORMED, "call_1")]),
            ]
        )
    )

    with pytest.raises(PrismError) as raised:
        pending.as_text()

    assert raised.value.code == "malformed_tool_call_arguments"
    assert transport.sent is None


def test_replays_a_stored_output_item_empty_once_answered() -> None:
    transport = RecordingTransport("openai")
    item = {
        "id": "fc_1",
        "type": "function_call",
        "call_id": "call_1",
        "name": "weather",
        "arguments": MALFORMED,
    }

    pending = (
        Prism.text()
        .using("openai", "m", {"transport": transport})
        .with_messages(
            [
                UserMessage("Weather?"),
                AssistantMessage(
                    "",
                    [ToolCall("call_1", "weather", MALFORMED, "call_1")],
                    {"output_items": [item]},
                ),
                ToolResultMessage([ToolResult("call_1", "weather", {}, "failed", "call_1")]),
            ]
        )
    )

    assert _failure_of(pending) is None
    assert _replayed("openai", transport) == "{}"


@pytest.mark.parametrize(
    ("arguments", "malformed"),
    [
        (MALFORMED, True),
        ("5", True),
        ('"x"', True),
        ("[1,2]", True),
        ("true", True),
        ('{"0":"x"}', True),
        ('{"-5":"x"}', True),
        ('{"city":"Paris"}', False),
        ('{"01":"x"}', False),
        ('{"9223372036854775808":"x"}', False),
        ("{}", False),
        ("[]", False),
        ("null", False),
        ("", False),
        ("0", False),
    ],
)
def test_classifies_arguments(arguments: str, malformed: bool) -> None:
    call = ToolCall("c", "t", arguments)

    assert hasattr(call, "has_malformed_arguments")
    assert call.has_malformed_arguments() is malformed


@pytest.mark.parametrize(
    "arguments",
    ["5", '"x"', "true", "false", "[1,2]", '{"0":"x"}', MALFORMED, {"0": "x"}],
)
def test_decoded_arguments_refuses_anything_but_named_values(arguments: Any) -> None:
    returned: Any = "nothing returned"
    code: Any = None

    try:
        returned = ToolCall("c", "weather", arguments).decoded_arguments()
    except PrismError as error:
        code = error.code

    assert returned == "nothing returned"
    assert code == "malformed_tool_call_arguments"


@pytest.mark.parametrize("arguments", ["", "0", "null", "[]", "{}"])
def test_decoded_arguments_still_reads_no_arguments(arguments: str) -> None:
    assert ToolCall("c", "weather", arguments).decoded_arguments() == {}


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_refuses_an_unanswered_non_object_call_at_mapping(provider: str) -> None:
    transport = RecordingTransport(provider)
    pending = (
        Prism.text()
        .using(provider, "m", {"transport": transport})
        .with_messages(
            [
                UserMessage("Weather?"),
                AssistantMessage("", [ToolCall("call_1", "weather", "5", "call_1")]),
            ]
        )
    )

    assert "not a JSON object" in (_failure_of(pending) or "")
    assert transport.sent is None


@pytest.mark.parametrize("provider", ["openai", "anthropic", "mistral"])
def test_still_replays_an_answered_non_object_call_empty(provider: str) -> None:
    transport = RecordingTransport(provider)
    pending = (
        Prism.text().using(provider, "m", {"transport": transport}).with_messages(_answered("5"))
    )

    assert _failure_of(pending) is None
    sent = _replayed(provider, transport)
    assert (json.loads(sent) if isinstance(sent, str) else sent) == {}
