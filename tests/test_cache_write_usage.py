"""Reported cache lifetimes, independently asserted against Anthropic's wire shape."""

from __future__ import annotations

import json
from typing import Any

import pytest

from prism import HttpRequest, HttpResponse, Prism, Usage
from prism.providers.anthropic.stream_events import AnthropicStreamMapper
from prism.schema import ObjectSchema, StringSchema
from prism.streaming.events import StreamEndEvent

LIFETIMES = [(148, 0), (0, 100), (148, 100), (0, 0), (None, None)]


def reported(five: int | None, hour: int | None) -> dict[str, Any]:
    usage: dict[str, Any] = {
        "input_tokens": 2048,
        "output_tokens": 503,
        "cache_creation_input_tokens": 248 if five is None else five + (hour or 0),
        "cache_read_input_tokens": 1800,
    }
    if five is not None:
        usage["cache_creation"] = {
            "ephemeral_5m_input_tokens": five,
            "ephemeral_1h_input_tokens": hour,
        }
    return usage


def assert_usage(usage: Usage | None, five: int | None, hour: int | None) -> None:
    assert usage is not None
    assert usage.cache_write_5m_input_tokens == five
    assert usage.cache_write_1h_input_tokens == hour
    assert usage.cache_write_input_tokens == (248 if five is None else five + (hour or 0))
    assert usage.prompt_tokens == 2048
    assert usage.completion_tokens == 503
    assert usage.cache_read_input_tokens == 1800
    assert Usage.from_dict(usage.to_dict()).to_dict() == usage.to_dict()
    if five is not None and hour is not None:
        assert usage.cache_write_5m_input_tokens is not None
        assert usage.cache_write_1h_input_tokens is not None
        assert (
            usage.cache_write_input_tokens
            == usage.cache_write_5m_input_tokens + usage.cache_write_1h_input_tokens
        )


@pytest.mark.parametrize(("five", "hour"), LIFETIMES)
@pytest.mark.parametrize("mode", ["text", "structured"])
def test_response_cache_lifetimes(five: int | None, hour: int | None, mode: str) -> None:
    body = {
        "id": "msg_cache",
        "model": "claude-owned-fixture",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": '{"value":"ok"}'}],
        "stop_reason": "end_turn",
        "usage": reported(five, hour),
    }

    class Transport:
        def send(self, request: HttpRequest) -> HttpResponse:
            return HttpResponse(status=200, body=json.dumps(body).encode())

    config = {"api_key": "owned-test-value", "transport": Transport()}
    if mode == "text":
        response = (
            Prism.text()
            .using("anthropic", "claude-owned-fixture", config)
            .with_prompt("Owned fixture")
            .as_text()
        )
        assert_usage(response.usage, five, hour)
        assert_usage(response.steps[0].usage, five, hour)
    else:
        request = Prism.structured()
        request.with_schema(
            ObjectSchema("result", "Result", (StringSchema("value", "Value"),), ("value",))
        )
        request.using("anthropic", "claude-owned-fixture", config).with_prompt("Owned fixture")
        structured = request.as_structured()
        assert_usage(structured.usage, five, hour)
        assert_usage(structured.steps[0].usage, five, hour)


@pytest.mark.parametrize(("five", "hour"), LIFETIMES)
@pytest.mark.parametrize("at_end", [False, True])
def test_stream_cumulative_sparse_cache_lifetimes(
    five: int | None, hour: int | None, at_end: bool
) -> None:
    mapper = AnthropicStreamMapper()
    raw = reported(five, hour)
    replace_zero = not at_end and five == 0 and hour == 0
    restated = at_end or replace_zero
    mapper.map(
        {
            "type": "message_start",
            "message": {
                "id": "msg_cache",
                "model": "claude-owned-fixture",
                "usage": {"input_tokens": 0, "output_tokens": 0}
                if at_end
                else {**(reported(148, 100) if replace_zero else raw), "output_tokens": 0},
            },
        }
    )
    mapper.map(
        {
            "type": "message_delta",
            "delta": {},
            "usage": {**raw, "output_tokens": 100} if restated else {"output_tokens": 100},
        }
    )
    mapper.map(
        {"type": "message_delta", "delta": {}, "usage": raw if restated else {"output_tokens": 503}}
    )
    mapper.map(
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 503},
        }
    )
    mapper.map({"type": "message_stop"})
    end = next(event for event in mapper.finish() if isinstance(event, StreamEndEvent))
    assert_usage(end.usage, five, hour)


def test_legacy_usage_positional_arguments() -> None:
    usage = Usage(11, 23, 5, 7, 3, 0.02)
    assert usage.cost == 0.02
    assert usage.cache_write_5m_input_tokens is None
    assert usage.cache_write_1h_input_tokens is None
    assert (
        Usage.from_dict({"prompt_tokens": 11, "completion_tokens": 23}).cache_write_1h_input_tokens
        is None
    )


def test_changed_total_does_not_retain_an_earlier_unreported_split() -> None:
    mapper = AnthropicStreamMapper()
    mapper.map(
        {
            "type": "message_start",
            "message": {
                "id": "msg_cache",
                "model": "claude-owned-fixture",
                "usage": {**reported(148, 100), "output_tokens": 0},
            },
        }
    )
    mapper.map(
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 503, "cache_creation_input_tokens": 100},
        }
    )
    mapper.map({"type": "message_stop"})
    usage = next(event for event in mapper.finish() if isinstance(event, StreamEndEvent)).usage
    assert usage is not None
    assert usage.cache_write_input_tokens == 100
    assert usage.cache_write_5m_input_tokens is None
    assert usage.cache_write_1h_input_tokens is None
