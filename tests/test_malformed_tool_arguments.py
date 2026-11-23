"""A tool call whose arguments could not be parsed, as a request carries it."""

from __future__ import annotations

import json
import socket
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
from prism.http import HttpStreamResponse
from prism.streaming.events import ToolCallEvent

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


def _nested(depth: int) -> str:
    return '{"a":' * (depth - 1) + "{}" + "}" * (depth - 1)


@pytest.mark.parametrize(
    "arguments",
    [
        '{"a":NaN}',
        '{"a":Infinity}',
        '{"a":-Infinity}',
        '{"a":"\\ud800"}',
        '{"a":"\\udc00"}',
        '{"\\ud800":1}',
        _nested(501),
        _nested(100000),
    ],
    ids=["nan", "inf", "neg-inf", "high", "low", "key", "depth-501", "depth-huge"],
)
def test_refuses_what_the_other_languages_refuse(arguments: str) -> None:
    assert ToolCall("c", "t", arguments).has_malformed_arguments() is True


@pytest.mark.parametrize(
    "arguments", ['{"a":"\\ud83d\\ude00"}', _nested(500)], ids=["pair", "depth-500"]
)
def test_accepts_what_the_other_languages_accept(arguments: str) -> None:
    assert ToolCall("c", "t", arguments).has_malformed_arguments() is False


def test_mistral_refuses_an_unanswered_unusable_call_at_mapping() -> None:
    transport = RecordingTransport("mistral")
    pending = (
        Prism.text()
        .using("mistral", "m", {"transport": transport})
        .with_messages(
            [
                UserMessage("Weather?"),
                AssistantMessage("", [ToolCall("call_1", "weather", "5", "call_1")]),
            ]
        )
    )

    assert "not a JSON object" in (_failure_of(pending) or "")
    assert transport.sent is None


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test below injects a transport; opening a socket is a bug."""

    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("A test tried to use the network.")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


UNUSABLE = {
    "a number past the float range": '{"amount":1e999}',
    "a nested negative one": '{"a":{"b":[-1e999]}}',
    "an integer past the float range": '{"amount":1' + "0" * 400 + "}",
    "a key starting with NUL": '{"amount":"5","\\u0000a":1}',
    "a nested key starting with NUL": '{"a":[{"\\u0000b":1}]}',
    "a control character inside a number": '{"amount":1\x0100}',
    "a control character inside a literal": "nu\x01ll",
    "a control character between tokens": '{"amount"\x01:1}',
    "a control character after a backslash": '{"a":"b\\\x01"}',
}


@pytest.mark.parametrize("arguments", UNUSABLE.values(), ids=UNUSABLE.keys())
def test_refuses_arguments_a_request_could_not_carry(arguments: str, no_network: None) -> None:
    call = ToolCall("c", "probe", arguments)

    assert call.has_malformed_arguments() is True
    assert call.without_malformed_arguments().decoded_arguments() == {}


@pytest.mark.parametrize("provider", ["openai", "anthropic", "mistral"])
@pytest.mark.parametrize("arguments", UNUSABLE.values(), ids=UNUSABLE.keys())
def test_replays_unusable_arguments_as_empty(
    provider: str, arguments: str, no_network: None
) -> None:
    transport = RecordingTransport(provider)
    pending = (
        Prism.text()
        .using(provider, "m", {"transport": transport})
        .with_messages(_answered(arguments))
    )

    assert _failure_of(pending) is None
    assert transport.sent is not None
    sent = _replayed(provider, transport)
    assert (json.loads(sent) if isinstance(sent, str) else sent) == {}


def test_keeps_a_control_character_inside_a_string(no_network: None) -> None:
    call = ToolCall("c", "probe", '{"note":"a\x01b\nc","n":1}')

    assert call.has_malformed_arguments() is False
    assert call.decoded_arguments() == {"note": "a\x01b\nc", "n": 1}


def test_accepts_a_long_digit_key_and_an_inner_nul(no_network: None) -> None:
    key = "7" * 5000
    call = ToolCall("c", "probe", '{"' + key + '":1,"a\\u0000b":2}')

    assert call.has_malformed_arguments() is False
    assert list(call.decoded_arguments()) == [key, "a\x00b"]


@pytest.mark.parametrize(
    "arguments",
    [
        {"a": float("inf")},
        {"a": {"b": float("nan")}},
        {"\x00a": 1},
        {"a": {"\x00b": 1}},
        {1: "x"},
    ],
)
def test_refuses_a_decoded_map_no_request_can_carry(
    arguments: dict[Any, Any], no_network: None
) -> None:
    assert ToolCall("c", "probe", arguments).has_malformed_arguments() is True


@pytest.mark.parametrize(
    ("value", "refused"),
    [
        (5, True),
        (0, True),
        (0.0, True),
        (True, True),
        (False, True),
        ([1, 2], True),
        ({"0": "x"}, True),
        (None, False),
        ([], False),
    ],
)
def test_reads_a_provider_value_by_what_it_is(value: Any, refused: bool, no_network: None) -> None:
    stored = ToolCall.storable_arguments(value)

    assert ToolCall("c", "probe", stored).has_malformed_arguments() is refused


class FrameTransport:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self.body = "".join(f"data: {canonical.encode(e)}\n\n" for e in events) + "data: [DONE]\n\n"
        self.used = 0

    def stream(self, request: HttpRequest) -> HttpStreamResponse:
        self.used += 1
        return HttpStreamResponse(status=200, headers={}, chunks=iter([self.body]))


def _streamed_call(provider: str, events: list[dict[str, Any]]) -> ToolCall:
    transport = FrameTransport(events)
    seen = [
        event.tool_call
        for event in Prism.text()
        .using(provider, "m", {"api_key": "k", "stream_transport": transport})
        .with_prompt("p")
        .as_stream()
        if isinstance(event, ToolCallEvent)
    ]

    assert transport.used == 1
    assert len(seen) == 1
    return seen[0]


def _mistral_deltas(values: list[Any]) -> list[dict[str, Any]]:
    return [
        *[
            {
                "id": "c",
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {"name": "probe", "arguments": value},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }
            for value in values
        ],
        {
            "id": "c",
            "model": "m",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        },
    ]


NON_TEXT = {"a number": 5, "a list": [1, 2], "true": True, "an integer-keyed map": {"0": "x"}}


@pytest.mark.parametrize("value", NON_TEXT.values(), ids=NON_TEXT.keys())
def test_streams_refuse_a_non_text_value(value: Any, no_network: None) -> None:
    openai = _streamed_call(
        "openai",
        [
            {
                "type": "response.output_item.done",
                "item_id": "fc_1",
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "probe",
                    "arguments": value,
                },
            }
        ],
    )
    mistral = _streamed_call("mistral", _mistral_deltas([value]))
    anthropic = _streamed_call(
        "anthropic",
        [
            {
                "type": "message_start",
                "message": {
                    "id": "m1",
                    "model": "m",
                    "usage": {"input_tokens": 1, "output_tokens": 0},
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "call_1", "name": "probe", "input": {}},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": value},
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_stop"},
        ],
    )

    assert openai.has_malformed_arguments() is True
    assert mistral.has_malformed_arguments() is True
    assert anthropic.has_malformed_arguments() is True


def test_mistral_stream_refuses_a_number_completing_an_object(no_network: None) -> None:
    call = _streamed_call("mistral", _mistral_deltas(['{"amount":', 5, "}"]))

    assert call.has_malformed_arguments() is True


@pytest.mark.parametrize("value", NON_TEXT.values(), ids=NON_TEXT.keys())
@pytest.mark.parametrize("provider", ["openai", "anthropic", "mistral"])
def test_parsers_refuse_a_non_text_value(provider: str, value: Any, no_network: None) -> None:
    bodies: dict[str, dict[str, Any]] = {
        "openai": {
            "id": "r",
            "model": "m",
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "probe",
                    "arguments": value,
                }
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
        "anthropic": {
            "id": "m",
            "type": "message",
            "role": "assistant",
            "model": "m",
            "content": [{"type": "tool_use", "id": "call_1", "name": "probe", "input": value}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
        "mistral": {
            "id": "c",
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"id": "call_1", "function": {"name": "probe", "arguments": value}}
                        ],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    }
    transport = RecordingTransport(provider)
    transport.body = bodies[provider]

    response = (
        Prism.text()
        .using(provider, "m", {"api_key": "k", "transport": transport})
        .with_prompt("p")
        .as_text()
    )

    assert transport.sent is not None
    assert response.tool_calls[0].has_malformed_arguments() is True


@pytest.mark.parametrize("form", ["", "0", "null", "[]"])
def test_mistral_replays_an_empty_form_as_an_object(form: str, no_network: None) -> None:
    transport = RecordingTransport("mistral")
    pending = (
        Prism.text().using("mistral", "m", {"transport": transport}).with_messages(_answered(form))
    )

    assert _failure_of(pending) is None
    assert _replayed("mistral", transport) == "{}"


def test_accepts_500_levels_and_refuses_501(no_network: None) -> None:
    def nested(depth: int) -> str:
        return '{"a":' * depth + "1" + "}" * depth

    assert ToolCall("c", "probe", nested(500)).has_malformed_arguments() is False
    assert ToolCall("c", "probe", nested(501)).has_malformed_arguments() is True


POSITIONAL = [
    '{"0":"prod"}',
    '["prod"]',
    '{"0":"prod","note":"x"}',
    '{"note":"x","1":"prod"}',
    '{"-1":"prod"}',
]
NAMED = ['{"00":"prod"}', '{"1.0":"prod"}', '{"-0":"prod"}', '{"+1":"prod"}', '{" 1":"prod"}']


@pytest.mark.parametrize("arguments", POSITIONAL)
def test_refuses_a_key_that_could_bind_by_position(arguments: str, no_network: None) -> None:
    transport = RecordingTransport("openai")
    pending = (
        Prism.text()
        .using("openai", "m", {"transport": transport})
        .with_messages(
            [
                UserMessage("Go"),
                AssistantMessage("", [ToolCall("call_1", "probe", arguments, "call_1")]),
            ]
        )
    )

    assert _failure_of(pending) is not None
    assert transport.sent is None


@pytest.mark.parametrize("arguments", NAMED)
def test_keeps_a_numeric_looking_key_as_a_name(arguments: str, no_network: None) -> None:
    call = ToolCall("c", "probe", arguments)

    assert call.has_malformed_arguments() is False
    assert call.decoded_arguments() == json.loads(arguments)
