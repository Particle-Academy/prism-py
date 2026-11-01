"""Native request mode, complete conversation state and before-send refusal."""

from __future__ import annotations

import json
from typing import Any

import pytest

from prism import HttpRequest, HttpResponse, Prism, PrismError, canonical
from prism.enums import FinishReason
from prism.providers.anthropic import Anthropic
from prism.schema import ObjectSchema, StringSchema
from prism.structured import StructuredPendingRequest
from prism.structured.from_text import structured_from_text_response
from prism.text.response_builder import ResponseBuilder
from prism.text.step import Step
from prism.value_objects import Meta, ProviderToolCall, ToolCall, ToolResult, Usage

SCHEMA = ObjectSchema("output", "The output", (StringSchema("result", "Result"),), ("result",))
JSON_TEXT = '{"result":"héllo"}'
FIXTURES: dict[str, dict[str, Any]] = {
    "openai": {
        "id": "structured",
        "model": "gpt-4o",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "status": "completed",
                "content": [{"type": "output_text", "text": JSON_TEXT}],
            }
        ],
        "usage": {"input_tokens": 12, "output_tokens": 7},
    },
    "anthropic": {
        "id": "structured",
        "model": "claude-sonnet-4-5",
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": JSON_TEXT}],
        "usage": {"input_tokens": 12, "output_tokens": 7},
    },
    "mistral": {
        "id": "structured",
        "model": "mistral-small-latest",
        "choices": [
            {"finish_reason": "stop", "message": {"role": "assistant", "content": JSON_TEXT}}
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 7},
    },
}


class Recorder:
    def __init__(self, body: dict[str, Any]) -> None:
        self.body = body
        self.calls: list[HttpRequest] = []

    def send(self, request: HttpRequest) -> HttpResponse:
        self.calls.append(request)
        return HttpResponse(status=200, body=canonical.encode(self.body).encode("utf-8"))


def structured_pending(
    provider: str, model: str, transport: Recorder | None = None
) -> StructuredPendingRequest:
    pending = Prism.structured()
    config: dict[str, Any] = {"api_key": "fixture-only"}
    if transport is not None:
        config["transport"] = transport
    pending.using(provider, model, config)
    pending.with_schema(SCHEMA)
    pending.with_prompt("Return JSON.")
    return pending


@pytest.mark.parametrize("provider", ["openai", "anthropic", "mistral"])
def test_complete_native_structured_shape(provider: str) -> None:
    fixture = FIXTURES[provider]
    transport = Recorder(fixture)
    response = structured_pending(provider, fixture["model"], transport).as_structured()
    obj = response.to_dict()
    assert len(transport.calls) == 1
    assert response.structured == {"result": "héllo"}
    assert (response.usage.prompt_tokens, response.usage.completion_tokens) == (12, 7)
    for key in ("tool_calls", "provider_tool_calls", "tool_results"):
        assert obj[key] == []
    step = obj["steps"][0]
    assert step["structured"] == ({} if provider == "mistral" else {"result": "héllo"})
    assert "tool_approval_requests" not in step
    assert len(step["messages"]) == 2
    assert step["messages"][-1]["type"] == "assistant"
    assert step["messages"][-1]["content"] == JSON_TEXT
    assert step["messages"][-1]["tool_calls"] == []
    assert response.text == ("" if provider == "anthropic" else JSON_TEXT)
    assert step["text"] == response.text
    assert step["raw"] == fixture
    assert obj["raw"] == fixture


@pytest.mark.parametrize("mode", [None, False])
def test_anthropic_native_schema_effort_and_unchanged_messages(mode: bool | None) -> None:
    transport = Recorder(FIXTURES["anthropic"])
    options: dict[str, Any] = {"effort": "high"}
    if mode is not None:
        options["use_tool_calling"] = mode
    pending = structured_pending("anthropic", "claude-sonnet-4-5")
    pending.with_provider_options(options)
    request = pending.to_request()
    before = [message.to_dict() for message in request.messages]
    provider = Anthropic(api_key="fixture-only", transport=transport)
    provider.structured(request)
    provider.structured(request)
    assert len(transport.calls) == 2
    assert transport.calls[0].body == transport.calls[1].body
    assert [message.to_dict() for message in request.messages] == before
    request_body = transport.calls[0].body
    assert request_body is not None
    body = json.loads(request_body)
    assert body["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "Return JSON."}]}
    ]
    assert body["output_config"] == {
        "effort": "high",
        "format": {"type": "json_schema", "schema": SCHEMA.to_dict()},
    }
    assert "tools" not in body
    assert "Respond with ONLY JSON" not in request_body.decode()


@pytest.mark.parametrize("mode", [True, 1, "enabled", {"mode": "tool"}])
def test_structured_tool_mode_refused_before_http(mode: Any) -> None:
    transport = Recorder(FIXTURES["anthropic"])
    pending = structured_pending("anthropic", "claude-sonnet-4-5", transport)
    pending.with_provider_options({"use_tool_calling": mode})
    with pytest.raises(PrismError) as caught:
        pending.as_structured()
    assert caught.value.code == "unsupported_provider_action"
    assert not transport.calls


def test_ordinary_anthropic_text_path_unchanged() -> None:
    transport = Recorder(FIXTURES["anthropic"])
    response = (
        Prism.text()
        .using(
            "anthropic", "claude-sonnet-4-5", {"api_key": "fixture-only", "transport": transport}
        )
        .with_prompt("Return JSON.")
        .as_text()
    )
    assert response.text == JSON_TEXT
    request_body = transport.calls[0].body
    assert request_body is not None
    assert "output_config" not in json.loads(request_body)
    assert "tool_approval_requests" in response.to_dict()["steps"][0]


def test_citations_refused_before_http_in_native_mode() -> None:
    transport = Recorder(FIXTURES["anthropic"])
    pending = structured_pending("anthropic", "claude-sonnet-4-5", transport)
    pending.with_provider_options({"citations": True})
    with pytest.raises(PrismError) as caught:
        pending.as_structured()
    assert caught.value.code == "unsupported_provider_action"
    assert not transport.calls


def test_nonempty_tool_fields_from_all_steps_and_original_text_unchanged() -> None:
    builder = ResponseBuilder()
    for call_id in ("one", "two"):
        builder.add_step(
            Step(
                text=JSON_TEXT,
                finish_reason=FinishReason.STOP,
                usage=Usage(12, 7),
                meta=Meta(call_id, "fixture"),
                tool_calls=[ToolCall(call_id, "lookup", {"id": call_id})],
                tool_results=[ToolResult(call_id, "lookup", {"id": call_id}, {"result": call_id})],
                provider_tool_calls=[
                    ProviderToolCall(call_id, "search", "completed", {"opaque": call_id})
                ],
                tool_approval_requests=[{"id": "approval-" + call_id}],
            )
        )
    text = builder.to_response()
    before = text.to_dict()
    structured = structured_from_text_response(text)
    assert text.to_dict() == before
    obj = structured.to_dict()
    assert [call["id"] for call in obj["tool_calls"]] == ["one", "two"]
    assert [call["id"] for call in obj["provider_tool_calls"]] == ["one", "two"]
    assert [result["tool_call_id"] for result in obj["tool_results"]] == ["one", "two"]
    steps = obj["steps"]
    assert steps[0]["tool_calls"] == obj["tool_calls"][:1]
    assert steps[1]["tool_results"] == obj["tool_results"][1:]
    assert all("tool_approval_requests" not in step for step in steps)
    assert steps[0]["messages"][0]["tool_approval_requests"] == [{"id": "approval-one"}]


def test_openai_structured_replay_and_text_reasoning_summary() -> None:
    reasoning = {
        "type": "reasoning",
        "id": "reasoning-fixture",
        "encrypted_content": "fixture-only",
        "summary": [{"type": "summary_text", "text": "Separate reasoning summary."}],
    }
    body = {**FIXTURES["openai"], "output": [reasoning, *FIXTURES["openai"]["output"]]}
    transport = Recorder(body)
    structured = structured_pending("openai", "gpt-4o", transport).as_structured()
    assert "reasoningSummaries" not in structured.additional_content
    assert structured.additional_content["output_items"] == body["output"]
    assert structured.raw == body
    text = (
        Prism.text()
        .using("openai", "gpt-4o", {"api_key": "fixture-only", "transport": transport})
        .with_prompt("Return JSON.")
        .as_text()
    )
    assert text.additional_content["reasoningSummaries"] == ["Separate reasoning summary."]
