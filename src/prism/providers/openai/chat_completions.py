"""OpenAI's chat-completions format: request bodies and response parsing.

Selected by ``api_format="chat_completions"`` on the :class:`OpenAI` provider,
which is also the provider every named OpenAI-compatible entry builds (vLLM, LM
Studio, gateways). Ported from the reference's ``RequestPayloadMap::chat*``,
``ChatCompletions*Map`` and ``Handlers/ChatCompletions``.

The bodies are compared byte for byte with the reference's, so the KEY ORDER and
which keys are dropped when empty are part of the contract here, not style:

* text drops every null key, ``max_tokens`` included;
* structured and stream send ``max_tokens`` unconditionally, as an explicit
  null when unset;
* stream puts ``stream: true`` FIRST and sends no ``verbosity``.

This is the same wire shape as Mistral's, and still not shared with it: the two
differ on ``tool_choice`` (``required`` vs ``any``), on whether a tool choice is
sent without tools, on a strict flag, and on the key order of a tool result.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from prism import canonical
from prism._php import data_get, where_not_null
from prism.enums import FinishReason, ToolChoice
from prism.errors import PrismError
from prism.providers.openai.response import validate_response_body
from prism.providers.support import data_uri
from prism.structured.request import StructuredRequest
from prism.text.request import Request
from prism.text.response import Response
from prism.text.response_builder import ResponseBuilder
from prism.text.step import Step
from prism.tool import Tool
from prism.value_objects import Meta, ProviderRateLimit, ToolCall, Usage
from prism.value_objects.media_file import Image
from prism.value_objects.messages import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)
from prism.value_objects.replayable_messages import replayable_messages

__all__ = [
    "build_chat_request_body",
    "build_chat_stream_body",
    "build_chat_structured_body",
    "json_instruction",
    "map_chat_finish_reason",
    "map_chat_messages",
    "map_chat_tool_choice",
    "map_chat_tools",
    "parse_chat_response",
    "with_json_instruction",
]


# ---------------------------------------------------------------------------
# request bodies
# ---------------------------------------------------------------------------


def build_chat_request_body(request: Request) -> dict[str, Any]:
    """``RequestPayloadMap::chatText``: every key through the not-null filter."""
    return where_not_null(
        {
            "model": request.model,
            "messages": map_chat_messages(request.messages, request.system_prompts),
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "tools": map_chat_tools(request.tools),
            # Sent even without tools, as the reference does.
            "tool_choice": map_chat_tool_choice(request.tool_choice),
            "verbosity": request.provider_option("text_verbosity"),
        }
    )


def build_chat_structured_body(request: StructuredRequest) -> dict[str, Any]:
    """``RequestPayloadMap::chatStructured``.

    JSON mode, never a schema: the reference's chat-completions sender asks for
    ``json_object`` and carries the schema in a system message (see
    :func:`with_json_instruction`, which the caller applies first). Tools are not
    sent, so a structured call cannot stop on a tool call it was never offered.
    """
    return {
        "model": request.model,
        "messages": map_chat_messages(request.messages, request.system_prompts),
        "max_tokens": request.max_tokens,
        **where_not_null(
            {
                "temperature": request.temperature,
                "top_p": request.top_p,
                "response_format": {"type": "json_object"},
                "verbosity": request.provider_option("text_verbosity"),
            }
        ),
    }


def build_chat_stream_body(request: Request) -> dict[str, Any]:
    """``RequestPayloadMap::chatStream``.

    No ``stream_options``: the reference does not ask for usage, so a server
    that reports it unasked is read, and one that does not leaves usage unknown.
    """
    return {
        "stream": True,
        "model": request.model,
        "messages": map_chat_messages(request.messages, request.system_prompts),
        "max_tokens": request.max_tokens,
        **where_not_null(
            {
                "temperature": request.temperature,
                "top_p": request.top_p,
                "tools": map_chat_tools(request.tools),
                "tool_choice": map_chat_tool_choice(request.tool_choice),
            }
        ),
    }


def json_instruction(request: StructuredRequest) -> SystemMessage:
    """The system message carrying the schema, worded and encoded as the reference.

    The schema is pretty-printed as PHP's ``json_encode(..., JSON_PRETTY_PRINT)``
    renders it: four-space indent, ``\\/`` for a slash and ``\\uXXXX`` for
    anything outside ASCII. The message is part of the request body, so a
    different encoding is a different body.
    """
    schema = request.schema
    if schema is None:  # pragma: no cover - the builder refuses this first
        raise ValueError("A structured request reached the provider without a schema.")

    pretty = json.dumps(schema.to_dict(), indent=4, ensure_ascii=True).replace("/", "\\/")

    return SystemMessage(f"Respond with JSON that matches the following schema: \n {pretty}")


def with_json_instruction(request: StructuredRequest) -> StructuredRequest:
    """A copy of ``request`` with the instruction appended LAST, after the conversation."""
    return replace(request, messages=[*request.messages, json_instruction(request)])


# ---------------------------------------------------------------------------
# messages
# ---------------------------------------------------------------------------


def map_chat_messages(
    messages: Sequence[Message], system_prompts: Sequence[SystemMessage] = ()
) -> list[dict[str, Any]]:
    """System prompts first, then the conversation in order.

    The conversation goes through :func:`replayable_messages` first: a tool
    call a later result answered is sent with no arguments if its own could not
    be parsed, and one nobody answered still fails when it is mapped.
    """
    mapped: list[dict[str, Any]] = []

    for message in [*system_prompts, *replayable_messages(messages)]:
        if isinstance(message, SystemMessage):
            mapped.append({"role": "system", "content": message.content})
        elif isinstance(message, UserMessage):
            mapped.append(_map_user_message(message))
        elif isinstance(message, AssistantMessage):
            mapped.append(_map_assistant_message(message))
        elif isinstance(message, ToolResultMessage):
            # One message per result, keyed by the CALL id.
            mapped.extend(
                {"role": "tool", "tool_call_id": result.tool_call_id, "content": result.result}
                for result in message.tool_results
            )
        else:
            raise PrismError.unknown_message_type(type(message).__name__)

    return mapped


def _map_user_message(message: UserMessage) -> dict[str, Any]:
    """Text, then images. Always a content array, as the reference sends it.

    A document, audio or video part is REFUSED. The reference's chat map drops
    them without a word, which answers a question about an attachment as if
    there were none; failing names what was not sent.
    """
    if message.documents() or message.audios() or message.videos():
        raise PrismError.unsupported_media(
            "OpenAI chat completions", "message part", "text and images only"
        )

    return {
        "role": "user",
        "content": [
            {"type": "text", "text": message.text()},
            *(_map_image(image) for image in message.images()),
        ],
    }


def _map_image(image: Image) -> dict[str, Any]:
    """``image_url`` is an OBJECT here, wrapping a url or a ``data:`` uri."""
    if image.is_url():
        return {"type": "image_url", "image_url": {"url": image.url}}

    if not image.has_base64():
        raise PrismError.unsupported_media(
            "OpenAI chat completions", "image", "a url, or bytes it can send as base64"
        )

    return {"type": "image_url", "image_url": {"url": data_uri(image, "OpenAI", "image")}}


def _map_assistant_message(message: AssistantMessage) -> dict[str, Any]:
    """``Payload::compact``: an empty content or an empty tool list is left out."""
    tool_calls = [
        {
            "id": call.id,
            "type": "function",
            "function": {
                "name": call.name,
                # A JSON STRING, re-encoded from the decoded arguments as the
                # reference's argumentsAsJson() does. Decoding here is also the
                # refusal: arguments that are not a JSON object fail at mapping,
                # before anything is sent.
                "arguments": canonical.encode(call.decoded_arguments() or {}),
            },
        }
        for call in message.tool_calls
    ]

    return {
        key: value
        for key, value in {
            "role": "assistant",
            "content": message.content,
            "tool_calls": tool_calls,
        }.items()
        if value not in (None, "", [], False)
    }


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------


def map_chat_tools(tools: Sequence[Tool]) -> list[dict[str, Any]] | None:
    """Function declarations, or None for no tools so the key is dropped.

    ``parameters`` is always sent, and ``strict`` only when it is on.
    """
    if not tools:
        return None

    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                **({"strict": True} if bool(tool.provider_option("strict")) else {}),
                "parameters": tool.parameters_as_schema(require_all_properties=True),
            },
        }
        for tool in tools
    ]


def map_chat_tool_choice(choice: str | ToolChoice | None) -> str | dict[str, Any] | None:
    """``ANY`` is ``"required"`` on OpenAI; a named tool is an object."""
    if isinstance(choice, str):
        return {"type": "function", "function": {"name": choice}}
    if choice is ToolChoice.AUTO:
        return "auto"
    if choice is ToolChoice.ANY:
        return "required"
    if choice is ToolChoice.NONE:
        return "none"
    return None


# ---------------------------------------------------------------------------
# responses
# ---------------------------------------------------------------------------

_FINISH_REASONS = {
    "stop": FinishReason.STOP,
    "tool_calls": FinishReason.TOOL_CALLS,
    "length": FinishReason.LENGTH,
    "content_filter": FinishReason.CONTENT_FILTER,
}


def map_chat_finish_reason(reason: Any) -> FinishReason:
    """``ChatCompletionsFinishReasonMap``: null is UNKNOWN, anything unrecognised OTHER."""
    if reason is None:
        return FinishReason.UNKNOWN
    if not isinstance(reason, str):
        return FinishReason.OTHER
    return _FINISH_REASONS.get(reason, FinishReason.OTHER)


def parse_chat_response(
    request: Request,
    data: Mapping[str, Any],
    rate_limits: Sequence[ProviderRateLimit] = (),
    *,
    structured: bool = False,
) -> Response:
    """A chat-completions body as a :class:`Response`, with no HTTP involved.

    :raises PrismError: ``provider_response_error`` for an empty body, an
        ``error`` object, a refusal, or (text only) a finish reason other than
        stop or length -- the reference raises for those rather than returning a
        filtered answer as if it were complete; ``tool_loop_not_supported`` for
        a reply that finished on tool calls, before any second request.
    """
    validate_response_body(data)

    refusal = data_get(data, "choices.0.message.refusal")
    if isinstance(refusal, str) and refusal != "":
        raise PrismError.provider_response_error(f"OpenAI refused the request: {refusal}")

    finish_reason = map_chat_finish_reason(data_get(data, "choices.0.finish_reason", ""))

    if finish_reason is FinishReason.TOOL_CALLS:
        raise PrismError.tool_loop_not_supported()

    if not structured and finish_reason not in (FinishReason.STOP, FinishReason.LENGTH):
        raise PrismError.provider_response_error("OpenAI: unhandled finish reason")

    content = data_get(data, "choices.0.message.content")

    builder = ResponseBuilder()
    builder.add_step(
        Step(
            text=content if isinstance(content, str) else "",
            finish_reason=finish_reason,
            tool_calls=map_chat_tool_calls(data_get(data, "choices.0.message.tool_calls")),
            tool_results=[],
            provider_tool_calls=[],
            usage=chat_usage(data_get(data, "usage")),
            meta=Meta(
                id=_string(data.get("id")),
                model=_string(data.get("model")),
                rate_limits=list(rate_limits),
            ),
            messages=list(request.messages),
            system_prompts=list(request.system_prompts),
            additional_content={},
            raw=dict(data),
        )
    )

    return builder.to_response()


def map_chat_tool_calls(calls: Any) -> list[ToolCall]:
    """Tool calls as the model sent them: the arguments kept as the JSON string."""
    if not isinstance(calls, list):
        return []

    return [
        ToolCall(
            id=_string(call.get("id")),
            name=_string(data_get(call, "function.name")),
            arguments=ToolCall.storable_arguments(data_get(call, "function.arguments")),
        )
        for call in calls
        if isinstance(call, dict)
    ]


def chat_usage(usage: Any) -> Usage:
    """Cached prompt tokens are SUBTRACTED from the prompt count and reported apart."""
    cached = _int(data_get(usage, "prompt_tokens_details.cached_tokens"))

    return Usage(
        prompt_tokens=max(0, _int(data_get(usage, "prompt_tokens")) - cached),
        completion_tokens=_int(data_get(usage, "completion_tokens")),
        cache_read_input_tokens=cached or None,
    )


def _string(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0
