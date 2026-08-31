"""Mistral's chat-completions stream."""

from __future__ import annotations

import json
from typing import Any

from prism import FinishReason, HttpRequest, Prism, canonical
from prism.http import HttpStreamResponse
from prism.providers.mistral.stream_events import MistralStreamMapper
from prism.streaming.events import (
    StreamEndEvent,
    StreamEventType,
    TextCompleteEvent,
    ToolCallEvent,
)


def chunk(
    delta: dict[str, Any],
    finish_reason: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "id": "cmpl-1",
        "object": "chat.completion.chunk",
        "model": "mistral-large-latest",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        **extra,
    }


class ScriptedStreamTransport:
    """Mistral's chunks, as SSE frames."""

    def __init__(self, chunks: list[dict[str, Any]], trailing: str = "data: [DONE]\n\n") -> None:
        self.body = "".join(f"data: {canonical.encode(item)}\n\n" for item in chunks) + trailing

    def stream(self, request: HttpRequest) -> HttpStreamResponse:
        return HttpStreamResponse(status=200, headers={}, chunks=iter([self.body]))


def collect(transport: ScriptedStreamTransport) -> list[Any]:
    return list(
        Prism.text()
        .using(
            "mistral", "mistral-large-latest", {"api_key": "sk-test", "stream_transport": transport}
        )
        .with_prompt("Hi")
        .as_stream()
    )


def test_the_stream_opens_deltas_completes_and_ends() -> None:
    events = collect(
        ScriptedStreamTransport(
            [
                chunk({"role": "assistant", "content": "Bon"}),
                chunk({"content": "jour"}),
                chunk({}, "stop", usage={"prompt_tokens": 5, "completion_tokens": 2}),
            ]
        )
    )

    assert [event.type() for event in events] == [
        StreamEventType.STREAM_START,
        StreamEventType.TEXT_START,
        StreamEventType.TEXT_DELTA,
        StreamEventType.TEXT_DELTA,
        StreamEventType.TEXT_COMPLETE,
        StreamEventType.STREAM_END,
    ]


def test_one_chunk_can_produce_several_events() -> None:
    # The first chunk both opens the stream and carries the first token.
    # Returning only the first event would drop tokens silently.
    events = collect(ScriptedStreamTransport([chunk({"content": "Hi"})]))

    assert len(events) == 3
    assert events[0].type() is StreamEventType.STREAM_START
    assert events[2].type() is StreamEventType.TEXT_DELTA


def test_the_full_text_is_accumulated_for_the_completion_event() -> None:
    events = collect(
        ScriptedStreamTransport(
            [chunk({"content": "Bon"}), chunk({"content": "jour"}), chunk({}, "stop")]
        )
    )
    complete = next(event for event in events if isinstance(event, TextCompleteEvent))

    assert complete.text == "Bonjour"


def test_done_is_not_mapped_as_a_chunk() -> None:
    # It is not JSON. Parsing it yields an empty dict, which the mapper would
    # read as a chunk with no choices.
    events = collect(ScriptedStreamTransport([chunk({"content": "x"}, "stop")]))

    assert len([event for event in events if isinstance(event, StreamEndEvent)]) == 1


def test_usage_is_none_rather_than_zero_when_the_provider_sent_none() -> None:
    # Zero tokens claims the generation was free.
    events = collect(ScriptedStreamTransport([chunk({"content": "x"}, "stop")]))
    end = events[-1]

    assert isinstance(end, StreamEndEvent)
    assert end.usage is None
    assert end.finish_reason is FinishReason.STOP


def test_usage_is_read_off_the_last_chunk_where_mistral_puts_it() -> None:
    events = collect(
        ScriptedStreamTransport(
            [
                chunk({"content": "x"}),
                chunk({}, "stop", usage={"prompt_tokens": 11, "completion_tokens": 4}),
            ]
        )
    )
    end = events[-1]

    assert isinstance(end, StreamEndEvent)
    assert end.usage is not None
    assert end.usage.prompt_tokens == 11


def test_tool_call_arguments_split_across_chunks_are_assembled_and_flushed_at_the_end() -> None:
    # Emitting a tool call while its JSON is still arriving hands a consumer a
    # fragment that will not parse.
    mapper = MistralStreamMapper()

    mapper.map(
        chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "function": {"name": "weather", "arguments": '{"ci'},
                    }
                ]
            }
        )
    )
    mapper.map(chunk({"tool_calls": [{"index": 0, "function": {"arguments": 'ty":"Paris"}'}}]}))

    events = mapper.map(chunk({}, "tool_calls"))
    call = next(event for event in events if isinstance(event, ToolCallEvent))

    assert call.tool_call is not None
    assert call.tool_call.id == "call_1"
    # The name arrives on the first fragment only; overwriting it with an empty
    # string on the second is how a tool call ends up nameless.
    assert call.tool_call.name == "weather"
    assert call.tool_call.arguments == '{"city":"Paris"}'
    assert json.loads(call.tool_call.arguments) == {"city": "Paris"}


def test_a_tool_call_with_no_arguments_sends_an_empty_object() -> None:
    mapper = MistralStreamMapper()

    mapper.map(chunk({"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "ping"}}]}))

    events = mapper.map(chunk({}, "tool_calls"))
    call = next(event for event in events if isinstance(event, ToolCallEvent))

    assert call.tool_call is not None
    assert call.tool_call.arguments == "{}"


def test_a_reasoning_delta_is_joined_rather_than_stringified() -> None:
    events = collect(
        ScriptedStreamTransport([chunk({"content": [{"type": "text", "text": "hi"}]}, "stop")])
    )
    complete = next(event for event in events if isinstance(event, TextCompleteEvent))

    assert complete.text == "hi"
