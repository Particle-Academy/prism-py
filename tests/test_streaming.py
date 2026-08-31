"""Streaming: SSE reassembly, event mapping, and what happens when it fails."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from prism import HttpRequest, Prism, PrismError
from prism.enums import FinishReason
from prism.http import HttpStreamResponse
from prism.streaming import (
    ErrorEvent,
    StreamEndEvent,
    StreamEventType,
    TextDeltaEvent,
    sse_data,
)

SSE = [
    'data: {"type":"response.created","response":{"model":"gpt-4o"}}\n\n',
    'data: {"type":"response.output_text.delta","delta":"Hel","item_id":"m1"}\n\n',
    'data: {"type":"response.output_text.delta","delta":"lo","item_id":"m1"}\n\n',
    'data: {"type":"response.completed","response":{"status":"completed",'
    '"usage":{"input_tokens":5,"output_tokens":2}}}\n\n',
    "data: [DONE]\n\n",
]


class RecordingStreamTransport:
    def __init__(self, chunks: list[str], status: int = 200) -> None:
        self.chunks = chunks
        self.status = status
        self.sent: HttpRequest | None = None

    def stream(self, request: HttpRequest) -> HttpStreamResponse:
        self.sent = request
        return HttpStreamResponse(status=self.status, headers={}, chunks=iter(self.chunks))


def _stream(chunks: list[str], status: int = 200) -> tuple[Iterator[Any], RecordingStreamTransport]:
    transport = RecordingStreamTransport(chunks, status)
    pending = Prism.text().using(
        "openai", "gpt-4o", {"api_key": "sk-test", "stream_transport": transport}
    )
    return pending.with_prompt("Hi").as_stream(), transport


def test_a_payload_split_across_chunks_is_reassembled() -> None:
    # THE reason the transport yields chunks rather than lines. A provider does
    # not align its writes to the reader's convenience, and this split lands
    # mid-JSON -- the case a line-promising transport would handle internally
    # where no test could reach it.
    assert list(sse_data(['data: {"type":"resp', 'onse.created"}\n\n'])) == [
        '{"type":"response.created"}'
    ]


def test_the_done_sentinel_is_dropped_rather_than_handed_on() -> None:
    assert list(sse_data(["data: [DONE]\n\n"])) == []


def test_a_final_line_without_a_newline_is_kept() -> None:
    # Dropping it loses the last event of any stream a server closes without one.
    assert list(sse_data(['data: {"a":1}'])) == ['{"a":1}']


def test_crlf_line_endings_are_tolerated() -> None:
    assert list(sse_data(['data: {"a":1}\r\n\r\n'])) == ['{"a":1}']


def test_deltas_arrive_and_the_stream_ends_with_a_reason_and_usage() -> None:
    stream, transport = _stream(SSE)
    events = list(stream)

    deltas = [event for event in events if isinstance(event, TextDeltaEvent)]
    end = events[-1]

    assert [delta.delta for delta in deltas] == ["Hel", "lo"]
    assert isinstance(end, StreamEndEvent)
    assert end.finish_reason is FinishReason.STOP
    assert end.usage is not None and end.usage.prompt_tokens == 5

    # `stream: True` is added to the SAME body the non-streamed path sends.
    assert transport.sent is not None
    assert b'"stream":true' in transport.sent.body
    assert transport.sent.headers["Accept"] == "text/event-stream"


def test_an_unrecognised_event_type_is_ignored_rather_than_fatal() -> None:
    # OpenAI adds event types without warning. A mapper that raised would turn a
    # provider's additive change into an outage for every consumer.
    stream, _ = _stream(
        [
            'data: {"type":"response.created","response":{"model":"gpt-4o"}}\n\n',
            'data: {"type":"response.some_future_thing","payload":{}}\n\n',
            'data: {"type":"response.output_text.delta","delta":"Hi","item_id":"m1"}\n\n',
        ]
    )

    assert [event.type() for event in stream] == [
        StreamEventType.STREAM_START,
        StreamEventType.TEXT_DELTA,
    ]


def test_a_mid_stream_error_is_an_event_not_an_exception() -> None:
    # By the time this arrives the consumer has already rendered text. Raising
    # would discard a partial answer the user watched appear.
    stream, _ = _stream(
        [
            'data: {"type":"response.output_text.delta","delta":"Par","item_id":"m1"}\n\n',
            'data: {"type":"error","error":{"code":"rate_limit","message":"slow down"}}\n\n',
        ]
    )

    events = list(stream)

    assert [event.type() for event in events] == [StreamEventType.TEXT_DELTA, StreamEventType.ERROR]
    assert isinstance(events[1], ErrorEvent)
    assert events[1].code == "rate_limit"


def test_an_incomplete_response_reads_as_a_length_finish() -> None:
    stream, _ = _stream(
        [
            'data: {"type":"response.incomplete","response":{"status":"incomplete",'
            '"incomplete_details":{"reason":"max_output_tokens"}}}\n\n'
        ]
    )

    events = list(stream)
    assert isinstance(events[0], StreamEndEvent)
    assert events[0].finish_reason is FinishReason.LENGTH


def test_a_failed_start_reports_the_provider_message() -> None:
    with pytest.raises(PrismError, match="bad key"):
        stream, _ = _stream(['{"error":{"message":"bad key"}}'], status=401)
        list(stream)


ANTHROPIC_SSE = [
    'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_1",'
    '"model":"claude-sonnet-4-5","usage":{"input_tokens":7,"output_tokens":0}}}\n\n',
    'event: content_block_start\ndata: {"type":"content_block_start","index":0,'
    '"content_block":{"type":"text","text":""}}\n\n',
    'data: {"type":"ping"}\n\n',
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}\n\n',
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo"}}\n\n',
    'data: {"type":"content_block_stop","index":0}\n\n',
    'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":4}}\n\n',
    'data: {"type":"message_stop"}\n\n',
]


def _anthropic_stream(chunks: list[str], status: int = 200) -> Iterator[Any]:
    transport = RecordingStreamTransport(chunks, status)
    pending = Prism.text().using(
        "anthropic", "claude-sonnet-4-5", {"api_key": "sk-test", "stream_transport": transport}
    )
    return pending.with_prompt("Hi").as_stream()


def test_anthropic_accumulates_deltas_and_completes_with_what_it_saw() -> None:
    events = list(_anthropic_stream(ANTHROPIC_SSE))

    # `ping` produces nothing a consumer can use and is dropped as noise.
    assert [event.type() for event in events] == [
        StreamEventType.STREAM_START,
        StreamEventType.TEXT_START,
        StreamEventType.TEXT_DELTA,
        StreamEventType.TEXT_DELTA,
        StreamEventType.TEXT_COMPLETE,
        StreamEventType.STREAM_END,
    ]

    complete = events[4]
    end = events[-1]

    # Truthful because it was accumulated: `content_block_stop` carries no text
    # of its own, so without the mapper's memory this would be empty.
    assert complete.text == "Hello"
    assert end.finish_reason is FinishReason.STOP
    # Output tokens are reported cumulatively on message_delta: last one wins.
    assert end.usage is not None
    assert end.usage.completion_tokens == 4
    assert end.usage.prompt_tokens == 7


def test_anthropic_emits_a_tool_call_once_its_arguments_have_arrived() -> None:
    # A half-parsed JSON fragment is not something a consumer can use, so the
    # partials accumulate and only the finished call is emitted.
    events = list(
        _anthropic_stream(
            [
                'data: {"type":"message_start","message":{"id":"m","model":"claude"}}\n\n',
                'data: {"type":"content_block_start","index":0,"content_block":'
                '{"type":"tool_use","id":"tu_1","name":"lookup"}}\n\n',
                'data: {"type":"content_block_delta","index":0,"delta":'
                '{"type":"input_json_delta","partial_json":"{\\"q\\":"}}\n\n',
                'data: {"type":"content_block_delta","index":0,"delta":'
                '{"type":"input_json_delta","partial_json":"\\"ada\\"}"}}\n\n',
                'data: {"type":"content_block_stop","index":0}\n\n',
                'data: {"type":"message_stop"}\n\n',
            ]
        )
    )

    call = next(event for event in events if event.type() is StreamEventType.TOOL_CALL)

    assert call.tool_call is not None
    assert call.tool_call.name == "lookup"
    assert call.tool_call.decoded_arguments() == {"q": "ada"}
    # The tool block opening emitted nothing: a tool call is only meaningful
    # once its arguments exist.
    assert not [e for e in events if e.type() is StreamEventType.TEXT_START]


def test_each_anthropic_stream_gets_its_own_mapper() -> None:
    # The mapper carries a message id and accumulated text. A shared instance
    # would let two concurrent generations read each other's blocks.
    first = list(_anthropic_stream(ANTHROPIC_SSE))
    second = list(_anthropic_stream(ANTHROPIC_SSE))

    def text_of(events: list[Any]) -> str:
        return next(e for e in events if e.type() is StreamEventType.TEXT_COMPLETE).text

    assert text_of(first) == "Hello"
    assert text_of(second) == "Hello"


def test_anthropic_max_tokens_reads_as_a_length_finish() -> None:
    events = list(
        _anthropic_stream(
            [
                'data: {"type":"message_start","message":{"id":"m","model":"claude"}}\n\n',
                'data: {"type":"message_delta","delta":{"stop_reason":"max_tokens"}}\n\n',
                'data: {"type":"message_stop"}\n\n',
            ]
        )
    )

    assert events[-1].finish_reason is FinishReason.LENGTH
