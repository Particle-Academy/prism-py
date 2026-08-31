"""Parsing Mistral's chat-completions responses."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from prism._php import data_get, where_not_null
from prism.enums import FinishReason
from prism.errors import PrismError
from prism.providers.mistral.maps import map_finish_reason
from prism.text.request import Request
from prism.text.response import Response
from prism.text.response_builder import ResponseBuilder
from prism.text.step import Step
from prism.value_objects import ToolCall
from prism.value_objects.meta import Meta
from prism.value_objects.usage import Usage

__all__ = [
    "extract_text",
    "extract_thinking",
    "first_choice_message",
    "parse_text_response",
    "validate_response",
]


def parse_text_response(request: Request, raw_body: Any) -> Response:
    """Turn a raw payload into a :class:`~prism.text.response.Response`.

    No HTTP happens here, so a stored payload replays through exactly the code
    path a live call takes.

    :raises PrismError: ``provider_response_error`` when the payload is missing,
        empty, or carries an error; ``max_tokens_exceeded`` when generation was
        cut short; ``tool_loop_not_supported`` when it stopped on tool calls.
    """
    data = validate_response(raw_body)
    finish_reason = map_finish_reason(data)

    if finish_reason is FinishReason.TOOL_CALLS:
        raise PrismError.tool_loop_not_supported()

    if finish_reason is FinishReason.LENGTH:
        raise PrismError.max_tokens_exceeded("length", "chat.completion")

    builder = ResponseBuilder()
    builder.add_step(_build_step(data, request, finish_reason))

    return builder.to_response()


def validate_response(raw_body: Any) -> dict[str, Any]:
    """Refuse anything that is not a usable payload.

    Mistral reports failures TWO ways: an ``object: "error"`` envelope, and a
    bare ``{message, type}`` on validation errors. Both are checked, because
    only the first carries a recognisable marker and the second is what a
    malformed request actually gets back.
    """
    if not isinstance(raw_body, dict) or not raw_body:
        raise PrismError.provider_response_error(
            "Mistral returned an empty or non-object response body."
        )

    error = raw_body.get("error")

    if isinstance(error, dict):
        raise PrismError.provider_response_error(
            f"Mistral error [{data_get(error, 'type', 'unknown')}]: "
            f"{data_get(error, 'message', 'unknown')}"
        )

    if raw_body.get("object") == "error" or (
        isinstance(raw_body.get("message"), str) and "choices" not in raw_body
    ):
        raise PrismError.provider_response_error(
            f"Mistral error [{data_get(raw_body, 'type', 'unknown')}]: "
            f"{data_get(raw_body, 'message', 'unknown')}"
        )

    return raw_body


def first_choice_message(data: Mapping[str, Any]) -> dict[str, Any]:
    """The first choice's message, or an empty dict.

    Mistral returns one choice unless ``n`` was set.
    """
    message = data_get(data, "choices.0.message")

    return message if isinstance(message, dict) else {}


def extract_text(message: Mapping[str, Any]) -> str:
    """The reply text, from either shape Mistral uses.

    ``content`` is usually a string, but reasoning models return an ARRAY of
    typed chunks -- and stringifying the list yields a Python repr, which
    reaches the caller as text the model never produced.
    """
    content = message.get("content")

    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        return ""

    return "".join(
        chunk.get("text", "")
        for chunk in content
        if isinstance(chunk, dict) and chunk.get("type") == "text"
    )


def extract_thinking(message: Mapping[str, Any]) -> str | None:
    """Reasoning chunks joined, or None when the model did not think."""
    content = message.get("content")

    if not isinstance(content, list):
        return None

    parts = [
        part.get("text", "")
        for chunk in content
        if isinstance(chunk, dict) and chunk.get("type") == "thinking"
        for part in (chunk.get("thinking") or [])
        if isinstance(part, dict)
    ]
    joined = "".join(part for part in parts if part)

    return joined or None


def _build_step(data: dict[str, Any], request: Request, finish_reason: FinishReason) -> Step:
    message = first_choice_message(data)

    return Step(
        text=extract_text(message),
        finish_reason=finish_reason,
        tool_calls=_map_tool_calls(message),
        usage=Usage(
            prompt_tokens=_number(data_get(data, "usage.prompt_tokens")),
            completion_tokens=_number(data_get(data, "usage.completion_tokens")),
        ),
        # `rate_limits` stays EMPTY. This port does not parse rate-limit
        # headers on any provider -- prism-ts does, and that divergence is in
        # the gaps register. Parsing them for Mistral alone would make one
        # provider in this port answer a question the other two cannot.
        meta=Meta(id=_string(data.get("id")), model=_string(data.get("model"))),
        messages=list(request.messages),
        system_prompts=list(request.system_prompts),
        additional_content=where_not_null({"thinking": extract_thinking(message)}),
        raw=data,
    )


def _map_tool_calls(message: Mapping[str, Any]) -> list[ToolCall]:
    calls = message.get("tool_calls")

    if not isinstance(calls, list):
        return []

    mapped = []

    for call in calls:
        if not isinstance(call, dict):
            continue

        function = call.get("function")
        function = function if isinstance(function, dict) else {}
        arguments = function.get("arguments")

        mapped.append(
            ToolCall(
                id=_string(call.get("id")),
                name=_string(function.get("name")),
                # A JSON STRING on the wire, like OpenAI's chat completions and
                # unlike Anthropic's decoded dict. Kept as the string;
                # `parsed_arguments()` is the one place it is decoded.
                arguments=arguments if isinstance(arguments, (str, dict)) else "{}",
            )
        )

    return mapped


def _string(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _number(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0
