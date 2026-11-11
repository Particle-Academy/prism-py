"""Independent strict-schema expectations through actual provider serialization."""

from __future__ import annotations

import json
from typing import Any

import pytest

from prism import HttpRequest, HttpResponse, Prism, PrismError, StringSchema, Tool
from prism.schema import ObjectSchema


class RecordingTransport:
    def __init__(self) -> None:
        self.calls: list[HttpRequest] = []

    def send(self, request: HttpRequest) -> HttpResponse:
        self.calls.append(request)
        body = {
            "id": "synthetic",
            "model": "synthetic",
            "status": "completed",
            "type": "message",
            "role": "assistant",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "Synthetic only."}],
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "Synthetic only."}]}
            ],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Synthetic only."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "input_tokens": 1,
                "output_tokens": 1,
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }
        return HttpResponse(status=200, body=json.dumps(body).encode("utf-8"))


def make_tool(shape: str, strict: bool | None = None) -> Tool:
    tool = Tool("synthetic_search", "Synthetic only.")
    if shape == "flat":
        tool.with_string_parameter("query", "Synthetic query.")
    if shape == "nested":
        tool.with_parameter(
            ObjectSchema(
                "profile",
                "Synthetic profile.",
                (StringSchema("region", "Synthetic region."),),
                required_fields=("region",),
            )
        )
    if strict is not None:
        tool.with_provider_options({"strict": strict})
    return tool


def send(provider: str, tool: Tool, transport: RecordingTransport) -> None:
    Prism.text().using(
        provider,
        "synthetic",
        {
            "api_key": "synthetic-not-a-provider-key",
            "url": "https://api.example.test/v1",
            "transport": transport,
        },
    ).with_prompt("Synthetic only.").with_tools([tool]).as_text()


def definition(transport: RecordingTransport) -> dict[str, Any]:
    assert len(transport.calls) == 1
    assert transport.calls[0].body is not None
    body: dict[str, Any] = json.loads(transport.calls[0].body)
    result: dict[str, Any] = body["tools"][0]
    return result


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize("shape", ["flat", "nested", "empty"])
def test_native_root_schema(provider: str, strict: bool, shape: str) -> None:
    tool = make_tool(shape, strict)
    transport = RecordingTransport()
    send(provider, tool, transport)
    mapped = definition(transport)
    assert mapped["name"] == "synthetic_search"
    assert "function" not in mapped
    if strict:
        assert mapped.get("strict") is True
    else:
        assert "strict" not in mapped
    key = "input_schema" if provider == "anthropic" else "parameters"
    if shape == "empty" and not strict and provider == "openai":
        assert key not in mapped
    else:
        assert key in mapped
        root = mapped[key]
        assert root["type"] == "object"
        assert root["required"] == (
            [] if shape == "empty" else ["query" if shape == "flat" else "profile"]
        )
        assert root["properties"] == tool.parameters_as_dict()
        if shape == "empty":
            assert json.dumps(root["properties"]) == "{}"
        if strict:
            assert root.get("additionalProperties") is False
        else:
            assert "additionalProperties" not in root


def test_optional_openai_strict_fields_refused_before_send() -> None:
    tool = (
        Tool("synthetic_optional", "Synthetic only.")
        .with_string_parameter("query", "Synthetic query.", False)
        .with_string_parameter("region", "Synthetic region.", False)
        .with_provider_options({"strict": True})
    )
    transport = RecordingTransport()
    with pytest.raises(PrismError) as caught:
        send("openai", tool, transport)
    assert caught.value.code == "invalid_tool_schema"
    for word in ["synthetic_optional", "query", "region", "required", "nullable", "strict"]:
        assert word in str(caught.value)
    assert transport.calls == []


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("strict", [True, False])
def test_nullable_and_declared_requirements(provider: str, strict: bool) -> None:
    tool = (
        Tool("nullable", "Synthetic only.")
        .with_parameter(StringSchema("query", "Synthetic query.", nullable=True), required=strict)
        .with_provider_options({"strict": strict})
    )
    transport = RecordingTransport()
    send(provider, tool, transport)
    root = definition(transport)["input_schema" if provider == "anthropic" else "parameters"]
    assert root["required"] == (["query"] if strict else [])
    assert root["properties"]["query"] == {
        "description": "Synthetic query.",
        "type": ["string", "null"],
    }


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_nested_caller_schema_not_rewritten(provider: str) -> None:
    nested = ObjectSchema(
        "profile",
        "Caller schema.",
        (StringSchema("note", "Optional note."),),
        required_fields=(),
        allow_additional_properties=True,
    )
    before = json.dumps(nested.to_dict())
    tool = (
        Tool("nested", "Synthetic only.")
        .with_parameter(nested)
        .with_provider_options({"strict": True})
    )
    transport = RecordingTransport()
    send(provider, tool, transport)
    root = definition(transport)["input_schema" if provider == "anthropic" else "parameters"]
    assert root["properties"]["profile"] == json.loads(before)
    assert root.get("additionalProperties") is False
    assert json.dumps(nested.to_dict()) == before


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("shape", ["flat", "empty"])
def test_shipped_absent_strict_default(provider: str, shape: str) -> None:
    transport = RecordingTransport()
    send(provider, make_tool(shape), transport)
    mapped = definition(transport)
    assert "strict" not in mapped
    key = "input_schema" if provider == "anthropic" else "parameters"
    if provider == "openai" and shape == "empty":
        assert key not in mapped
    else:
        assert "additionalProperties" not in mapped[key]


def test_anthropic_keeps_optional_strict_fields() -> None:
    tool = (
        Tool("optional", "Synthetic only.")
        .with_string_parameter("query", "Synthetic.", False)
        .with_provider_options({"strict": True})
    )
    transport = RecordingTransport()
    send("anthropic", tool, transport)
    root = definition(transport)["input_schema"]
    assert root["required"] == []
    assert root.get("additionalProperties") is False


def test_same_tool_recomputed_after_strict_disabled() -> None:
    tool = make_tool("empty", True)
    first = RecordingTransport()
    send("openai", tool, first)
    assert "parameters" in definition(first)
    tool.with_provider_options({"strict": False})
    second = RecordingTransport()
    send("openai", tool, second)
    assert "parameters" not in definition(second)
    assert "strict" not in definition(second)


@pytest.mark.parametrize("strict", [True, False])
def test_mistral_optional_schema_unchanged(strict: bool) -> None:
    tool = (
        Tool("optional", "Synthetic only.")
        .with_string_parameter("query", "Synthetic.", False)
        .with_provider_options({"strict": strict})
    )
    transport = RecordingTransport()
    send("mistral", tool, transport)
    mapped = definition(transport)
    fn = mapped["function"]
    assert "strict" not in mapped
    assert "strict" not in fn
    assert fn["parameters"]["required"] == []
    assert "additionalProperties" not in fn["parameters"]


@pytest.mark.parametrize("provider", ["azure", "deepseek"])
def test_unimplemented_providers_stay_unsupported(provider: str) -> None:
    transport = RecordingTransport()
    with pytest.raises(PrismError) as caught:
        send(provider, make_tool("empty", True), transport)
    assert caught.value.code == "unsupported_provider_action"
    assert transport.calls == []
