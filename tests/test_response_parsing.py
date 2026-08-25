"""Parsing an OpenAI Responses body."""

from __future__ import annotations

from typing import Any

import pytest

from prism import (
    AssistantMessage,
    FinishReason,
    PendingRequest,
    Prism,
    PrismError,
    Request,
    UserMessage,
    parse_text_response,
)


def pending() -> PendingRequest:
    return Prism.text().using("openai", "gpt-4o").with_messages([UserMessage("hi")])


def request() -> Request:
    return pending().to_request()


def body(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": "resp_1",
        "object": "response",
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
            "total_tokens": 67,
        },
        "service_tier": "default",
    }
    payload.update(overrides)
    return payload


def test_the_text_comes_from_the_last_output_item() -> None:
    response = parse_text_response(request(), body())

    assert response.text == "I am a language model."
    assert response.finish_reason is FinishReason.STOP


def test_cached_input_tokens_are_subtracted_from_the_prompt_count() -> None:
    # Passing input_tokens straight through double-counts the cache and inflates
    # the caller's cost reporting on every prompt-cached request.
    response = parse_text_response(
        request(),
        body(
            usage={
                "input_tokens": 1024,
                "input_tokens_details": {"cached_tokens": 896},
                "output_tokens": 12,
                "output_tokens_details": {"reasoning_tokens": 4},
            }
        ),
    )

    assert response.usage.prompt_tokens == 128
    assert response.usage.cache_read_input_tokens == 896
    assert response.usage.completion_tokens == 12
    assert response.usage.thought_tokens == 4


def test_meta_carries_the_provider_ids() -> None:
    response = parse_text_response(request(), body())

    assert response.meta.id == "resp_1"
    assert response.meta.model == "gpt-4o-2024-08-06"
    assert response.meta.service_tier == "default"


def test_an_absent_service_tier_becomes_an_explicit_none() -> None:
    payload = body()
    del payload["service_tier"]

    assert parse_text_response(request(), payload).meta.service_tier is None
    assert parse_text_response(request(), payload).to_dict()["meta"]["service_tier"] is None


def test_a_content_filter_has_its_own_finish_reason() -> None:
    # Not length — and length is the branch that raises, so getting this wrong
    # turns a filtered answer into an exception.
    response = parse_text_response(
        request(),
        body(
            status="incomplete",
            incomplete_details={"reason": "content_filter"},
            output=[
                {
                    "id": "msg_1",
                    "type": "message",
                    "status": "incomplete",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "I cannot help with that."}],
                }
            ],
        ),
    )

    assert response.finish_reason is FinishReason.CONTENT_FILTER
    assert response.text == "I cannot help with that."


def test_an_empty_output_array_parses_rather_than_crashing() -> None:
    response = parse_text_response(request(), body(output=[]))

    assert response.text == ""
    assert response.finish_reason is FinishReason.UNKNOWN


def test_reasoning_summaries_are_collected_and_do_not_become_the_answer() -> None:
    response = parse_text_response(
        request(),
        body(
            output=[
                {
                    "id": "rs_1",
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "Consider the two cases."}],
                },
                {
                    "id": "msg_1",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "The answer is four."}],
                },
            ]
        ),
    )

    assert response.text == "The answer is four."
    assert response.additional_content == {"reasoningSummaries": ["Consider the two cases."]}


def test_the_messages_include_the_trailing_assistant_turn() -> None:
    response = parse_text_response(request(), body())

    assert len(response.messages) == 2
    assert isinstance(response.messages[0], UserMessage)
    last = response.messages[1]
    assert isinstance(last, AssistantMessage)
    assert last.content == "I am a language model."


def test_a_single_step_is_recorded() -> None:
    response = parse_text_response(request(), body())

    assert len(response.steps) == 1
    assert response.steps[0].raw == body()


def test_a_length_finish_raises_with_a_code() -> None:
    with pytest.raises(PrismError) as raised:
        parse_text_response(
            request(),
            body(status="incomplete", incomplete_details={"reason": "max_output_tokens"}),
        )

    assert raised.value.code == "max_tokens_exceeded"


def test_a_tool_call_finish_is_refused_rather_than_half_implemented() -> None:
    with pytest.raises(PrismError) as raised:
        parse_text_response(
            request(),
            body(
                output=[
                    {
                        "id": "fc_1",
                        "call_id": "call_1",
                        "type": "function_call",
                        "status": "completed",
                        "name": "weather",
                        "arguments": '{"city":"Paris"}',
                    }
                ]
            ),
        )

    assert raised.value.code == "tool_loop_not_supported"


def test_an_error_body_is_refused_with_a_code() -> None:
    with pytest.raises(PrismError) as raised:
        parse_text_response(request(), {"error": {"type": "invalid_request", "message": "no"}})

    assert raised.value.code == "provider_response_error"


def test_an_empty_body_is_refused_with_a_code() -> None:
    with pytest.raises(PrismError) as raised:
        parse_text_response(request(), {})

    assert raised.value.code == "provider_response_error"


def test_provider_tool_calls_are_collected_from_the_output() -> None:
    response = parse_text_response(
        request(),
        body(
            output=[
                {"id": "ws_1", "type": "web_search_call", "status": "completed"},
                {
                    "id": "msg_1",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "done"}],
                },
            ]
        ),
    )

    assert [call.id for call in response.steps[0].provider_tool_calls] == ["ws_1"]


def test_web_search_actions_are_collected_uniquely() -> None:
    response = parse_text_response(
        request(),
        body(
            output=[
                {
                    "id": "ws_1",
                    "type": "web_search_call",
                    "status": "completed",
                    "action": {"type": "search", "query": "prism"},
                },
                {
                    "id": "ws_2",
                    "type": "web_search_call",
                    "status": "completed",
                    "action": {"type": "search", "query": "prism"},
                },
                {
                    "id": "msg_1",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "done"}],
                },
            ]
        ),
    )

    assert response.additional_content["searchQueries"] == ["prism"]


def test_the_serialised_shape_keeps_the_reference_key_order() -> None:
    result = parse_text_response(request(), body()).to_dict()

    assert list(result) == [
        "steps",
        "text",
        "finish_reason",
        "tool_calls",
        "tool_results",
        "usage",
        "meta",
        "messages",
        "additional_content",
        "raw",
    ]
    assert list(result["steps"][0]) == [
        "text",
        "finish_reason",
        "tool_calls",
        "tool_results",
        "provider_tool_calls",
        "usage",
        "meta",
        "messages",
        "system_prompts",
        "additional_content",
        "raw",
        "tool_approval_requests",
    ]
