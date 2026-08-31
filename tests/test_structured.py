"""Structured output: the schema, the modes, and what happens when it fails."""

from __future__ import annotations

import json
from typing import Any

import pytest

from prism import HttpRequest, HttpResponse, Prism, PrismError, canonical
from prism.enums import StructuredMode
from prism.schema import ArraySchema, EnumSchema, ObjectSchema, StringSchema
from prism.structured import extract_structured

SCHEMA = ObjectSchema("person", "A person", (StringSchema("name", "Their name"),), ("name",))


class RecordingTransport:
    def __init__(self, text: str) -> None:
        self.text = text
        self.sent: HttpRequest | None = None

    def send(self, request: HttpRequest) -> HttpResponse:
        self.sent = request
        body = {
            "id": "resp_1",
            "status": "completed",
            "model": "gpt-4o",
            "output": [
                {
                    "type": "message",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": self.text}],
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 2},
        }
        return HttpResponse(status=200, body=canonical.encode(body).encode("utf-8"))


def _sent_body(transport: RecordingTransport) -> dict[str, Any]:
    assert transport.sent is not None
    return json.loads(transport.sent.body.decode("utf-8"))


def test_openai_is_asked_to_enforce_the_schema() -> None:
    transport = RecordingTransport('{"name":"Ada"}')

    response = (
        Prism.structured()
        .using("openai", "gpt-4o", {"api_key": "sk-test", "transport": transport})
        .with_schema(SCHEMA)
        .with_prompt("Who?")
        .as_structured()
    )

    fmt = _sent_body(transport)["text"]["format"]
    assert fmt["type"] == "json_schema"
    # strict is the whole reason to prefer this mode: without it a near-miss
    # that parses but omits a required field comes back looking fine.
    assert fmt["strict"] is True
    assert response.structured == {"name": "Ada"}
    assert response.text == '{"name":"Ada"}'


def test_a_model_without_schema_support_falls_back_to_json_mode() -> None:
    transport = RecordingTransport('{"name":"Ada"}')

    (
        Prism.structured()
        .using("openai", "gpt-3.5-turbo", {"api_key": "sk-test", "transport": transport})
        .with_schema(SCHEMA)
        .with_prompt("Who?")
        .as_structured()
    )

    assert _sent_body(transport)["text"]["format"] == {"type": "json_object"}


def test_a_model_that_cannot_do_structured_output_is_refused() -> None:
    with pytest.raises(PrismError, match="not supported for o1-mini"):
        (
            Prism.structured()
            .using(
                "openai", "o1-mini", {"api_key": "sk-test", "transport": RecordingTransport("{}")}
            )
            .with_schema(SCHEMA)
            .with_prompt("Who?")
            .as_structured()
        )


def test_a_fine_tune_inherits_its_base_model_capability() -> None:
    transport = RecordingTransport('{"name":"Ada"}')

    (
        Prism.structured()
        .using(
            "openai", "ft:gpt-4o:acme:tuned:abc123", {"api_key": "sk-test", "transport": transport}
        )
        .with_schema(SCHEMA)
        .with_prompt("Who?")
        .as_structured()
    )

    # Matching the prefix against the whole string would classify every
    # fine-tune as JSON-only, including ones built on gpt-4o.
    assert _sent_body(transport)["text"]["format"]["type"] == "json_schema"


def test_an_explicit_mode_beats_the_resolver() -> None:
    transport = RecordingTransport('{"name":"Ada"}')

    (
        Prism.structured()
        .using("openai", "gpt-4o", {"api_key": "sk-test", "transport": transport})
        .with_schema(SCHEMA)
        .using_structured_mode(StructuredMode.JSON)
        .with_prompt("Who?")
        .as_structured()
    )

    assert _sent_body(transport)["text"]["format"] == {"type": "json_object"}


def test_a_structured_request_without_a_schema_is_refused() -> None:
    with pytest.raises(PrismError, match="needs a schema"):
        Prism.structured().using("openai", "gpt-4o", {"api_key": "sk-test"}).with_prompt(
            "Who?"
        ).to_request()


def test_the_text_survives_when_the_answer_is_not_an_object() -> None:
    # A refusal is an answer, not a crash. Raising here would destroy the one
    # artifact that explains why it did not parse.
    transport = RecordingTransport("I am afraid I cannot help with that.")

    response = (
        Prism.structured()
        .using("openai", "gpt-4o", {"api_key": "sk-test", "transport": transport})
        .with_schema(SCHEMA)
        .with_prompt("Who?")
        .as_structured()
    )

    assert response.structured is None
    assert response.text == "I am afraid I cannot help with that."


def test_fenced_json_is_unfenced() -> None:
    assert extract_structured('```json\n{"name":"Ada"}\n```') == {"name": "Ada"}


def test_a_parsed_value_that_is_not_an_object_is_rejected() -> None:
    # `[1,2,3]` parses. Returning it as "structured" would satisfy the
    # annotation and break the first caller to read a key off it.
    assert extract_structured("[1,2,3]") is None
    assert extract_structured('"just a string"') is None


def test_object_schema_forbids_additional_properties_by_default() -> None:
    # OpenAI's strict mode REJECTS a schema that permits extra properties, so a
    # permissive default would silently disable the strongest structured mode.
    assert SCHEMA.to_dict()["additionalProperties"] is False


def test_a_nullable_enum_offers_null_as_an_option() -> None:
    # An enum that can be null but whose options omit null describes a value no
    # validator can satisfy.
    schema = EnumSchema("colour", "A colour", ("red", "blue"), nullable=True)
    assert schema.to_dict()["enum"] == ["red", "blue", None]
    assert schema.to_dict()["type"] == ["string", "null"]


def test_array_bounds_are_omitted_when_unset() -> None:
    schema = ArraySchema("tags", "Tags", StringSchema("tag", "A tag"))
    assert "minItems" not in schema.to_dict()
