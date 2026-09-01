"""Mapping Prism's value objects onto the OpenAI Responses wire shapes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from prism import canonical
from prism._php import data_get, is_truthy, strval
from prism.enums import FinishReason, ToolChoice
from prism.errors import PrismError
from prism.providers.support import data_uri
from prism.tool import Tool
from prism.value_objects import (
    AssistantMessage,
    Message,
    ProviderToolCall,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from prism.value_objects.media_file import Document, Image

__all__ = [
    "map_finish_reason",
    "map_messages",
    "map_provider_tool_calls",
    "map_tool_calls",
    "map_tool_choice",
    "map_tools",
    "resolve_finish_reason",
]


# ---------------------------------------------------------------------------
# messages
# ---------------------------------------------------------------------------


def map_messages(
    messages: Sequence[Message],
    system_prompts: Sequence[SystemMessage] = (),
) -> list[dict[str, Any]]:
    """Build the ``input`` array.

    System prompts are PREPENDED to the message list rather than carried in a
    separate field, and the roles are shaped differently in the same payload: a
    system message maps to a bare string ``content``, a user message to a list
    of parts.
    """
    mapped: list[dict[str, Any]] = []

    for message in [*system_prompts, *messages]:
        if isinstance(message, UserMessage):
            mapped.append(_map_user_message(message))
        elif isinstance(message, AssistantMessage):
            mapped.extend(_map_assistant_message(message))
        elif isinstance(message, ToolResultMessage):
            mapped.extend(_map_tool_result_message(message))
        elif isinstance(message, SystemMessage):
            mapped.append({"role": "system", "content": message.content})
        else:
            raise PrismError.unknown_message_type(type(message).__name__)

    return mapped


def _map_user_message(message: UserMessage) -> dict[str, Any]:
    # Additional attributes are SPREAD at item level — siblings of role and
    # content, not nested under a key of their own.
    #
    # Text leads, then images, then documents, matching the reference's order.
    return {
        "role": "user",
        "content": [
            {"type": "input_text", "text": message.text()},
            *(_map_image(image) for image in message.images()),
            *(_map_document(document) for document in message.documents()),
        ],
        **message.additional_attributes,
    }


def _map_image(image: Image) -> dict[str, Any]:
    """The Responses API's image part.

    ``image_url`` here is a BARE STRING, where Mistral's chat-completions shape
    wraps the same value in an object. A file id goes in its own field rather
    than through the url, because a ``data:`` uri and a provider-side file are
    two different things to OpenAI even though both end up as ``input_image``.
    """
    if image.is_file_id():
        return {"type": "input_image", "file_id": image.file_id()}

    if image.is_url():
        return {"type": "input_image", "image_url": image.url}

    if not image.has_base64():
        raise PrismError.unsupported_media(
            "OpenAI", "image", "a file id, a url, or bytes it can send as base64"
        )

    return {"type": "input_image", "image_url": data_uri(image, "OpenAI", "image")}


def _map_document(document: Document) -> dict[str, Any]:
    """The Responses API's document part.

    Three shapes for one concept, and the inline one needs a FILENAME: OpenAI
    reads the extension off it to decide how to parse the bytes, so a document
    with no title is sent as ``document`` rather than omitting the field,
    matching the reference. The other two shapes carry the name in the url or
    the file.

    Chunks are rejected. They are text with no container, and only Anthropic has
    somewhere to put them.
    """
    if document.is_file_id():
        return {"type": "input_file", "file_id": document.file_id()}

    if document.is_url():
        return {"type": "input_file", "file_url": document.url}

    if document.is_chunks():
        raise PrismError.unsupported_media(
            "OpenAI", "document", "a file id, a url, or bytes -- not pre-split chunks"
        )

    if not document.has_base64():
        raise PrismError.unsupported_media(
            "OpenAI", "document", "a file id, a url, or bytes it can send as base64"
        )

    return {
        "type": "input_file",
        "filename": document.document_title() or "document",
        "file_data": data_uri(document, "OpenAI", "document"),
    }


def _map_assistant_message(message: AssistantMessage) -> list[dict[str, Any]]:
    mapped: list[dict[str, Any]] = []

    if message.tool_calls:
        for reasoning_id, calls in _group_by_reasoning_id(message.tool_calls):
            if reasoning_id != "":
                mapped.append(
                    {
                        "type": "reasoning",
                        "id": calls[0].reasoning_id,
                        "summary": calls[0].reasoning_summary,
                    }
                )

            for call in calls:
                mapped.append(
                    {
                        "id": call.id,
                        "call_id": call.result_id,
                        "type": "function_call",
                        "name": call.name,
                        # A JSON STRING, not an object: the arguments are
                        # double-encoded on this API. Missing that produces a
                        # request OpenAI accepts and misreads.
                        "arguments": canonical.encode(call.decoded_arguments() or {}),
                    }
                )

    # An assistant turn that carries only tool calls contributes no item at all.
    # An empty output_text part is rejected by OpenAI.
    if message.content != "":
        mapped.append(
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": message.content}],
            }
        )

    return mapped


def _group_by_reasoning_id(calls: Sequence[ToolCall]) -> list[tuple[str, list[ToolCall]]]:
    """Group calls by reasoning id, keeping first-seen group order."""
    groups: dict[str, list[ToolCall]] = {}

    for call in calls:
        groups.setdefault(call.reasoning_id or "", []).append(call)

    return list(groups.items())


def _map_tool_result_message(message: ToolResultMessage) -> list[dict[str, Any]]:
    mapped: list[dict[str, Any]] = []

    for result in message.tool_results:
        output = result.result

        if isinstance(output, str):
            # A string result passes through untouched rather than being
            # re-encoded as a JSON string.
            payload = output
        elif isinstance(output, (list, dict)):
            payload = canonical.encode(output)
        else:
            payload = strval(output)

        mapped.append(
            {
                "type": "function_call_output",
                # Keyed by the RESULT id, not the tool call id.
                "call_id": result.tool_call_result_id,
                "output": payload,
            }
        )

    return mapped


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------


def map_tools(tools: Sequence[Tool]) -> list[dict[str, Any]]:
    """Map tools onto function declarations.

    The reference filters each declaration on FALSINESS, not on nullity, which
    is why ``strict: False`` never reaches the wire while ``store: False`` — a
    provider option, filtered on nullity — does. That inconsistency is the
    reference's and is reproduced rather than tidied: tidying it would change
    what a caller's existing tool declarations send.
    """
    mapped: list[dict[str, Any]] = []

    for tool in tools:
        declaration: dict[str, Any] = {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
        }

        if tool.has_parameters():
            declaration["parameters"] = {
                "type": "object",
                "properties": tool.parameters_as_dict(),
                "required": tool.required_parameters,
            }

        declaration["strict"] = bool(tool.provider_option("strict"))

        mapped.append({key: value for key, value in declaration.items() if is_truthy(value)})

    return mapped


def map_tool_choice(tool_choice: str | ToolChoice | None) -> str | dict[str, Any] | None:
    """Map a tool choice onto its wire form.

    A named choice becomes an OBJECT; the enum members become strings — and
    ``ANY`` becomes ``"required"``, which is the one member whose wire name
    differs from its own. Lowercasing the member would send ``"any"``, which
    OpenAI rejects.
    """
    if isinstance(tool_choice, str):
        return {"type": "function", "name": tool_choice}

    if tool_choice is ToolChoice.AUTO:
        return "auto"
    if tool_choice is ToolChoice.ANY:
        return "required"
    if tool_choice is ToolChoice.NONE:
        return "none"

    return None


# ---------------------------------------------------------------------------
# responses
# ---------------------------------------------------------------------------


def resolve_finish_reason(status: str, item_type: str | None = None) -> FinishReason:
    """Map an output item's status and type onto a finish reason."""
    if status in ("incomplete", "length"):
        return FinishReason.LENGTH
    if status == "failed":
        return FinishReason.ERROR
    if status == "completed":
        if item_type == "function_call":
            return FinishReason.TOOL_CALLS
        if item_type == "message":
            return FinishReason.STOP
        return (
            FinishReason.TOOL_CALLS
            if str(item_type or "").endswith("_call")
            else FinishReason.UNKNOWN
        )

    return FinishReason.UNKNOWN


def map_finish_reason(data: Mapping[str, Any]) -> FinishReason:
    """Map a whole response body onto a finish reason.

    The TOP-LEVEL status is inspected first: an incomplete response caused by a
    content filter has its own reason, and getting that wrong turns a filtered
    answer into a length exception.
    """
    if data_get(data, "status", "") == "incomplete":
        if data_get(data, "incomplete_details.reason") == "content_filter":
            return FinishReason.CONTENT_FILTER
        return FinishReason.LENGTH

    return resolve_finish_reason(
        data_get(data, "output.{last}.status", ""),
        data_get(data, "output.{last}.type", ""),
    )


def map_tool_calls(
    items: Sequence[Mapping[str, Any]],
    reasonings: Sequence[Mapping[str, Any]] | None = None,
) -> list[ToolCall]:
    """Build tool calls out of the ``function_call`` items of an output list."""
    return [
        ToolCall(
            id=data_get(item, "id"),
            name=data_get(item, "name"),
            arguments=data_get(item, "arguments"),
            result_id=data_get(item, "call_id"),
            reasoning_id=data_get(reasonings, "0.id"),
            reasoning_summary=data_get(reasonings, "0.summary"),
        )
        for item in items
    ]


def map_provider_tool_calls(output: Sequence[Mapping[str, Any]]) -> list[ProviderToolCall]:
    """Build provider tool calls out of an output list.

    Anything whose type ends in ``_call`` and is not a ``function_call`` was run
    by the provider itself.
    """
    return [
        ProviderToolCall(
            id=data_get(item, "id"),
            type=data_get(item, "type"),
            status=data_get(item, "status"),
            data=dict(item),
        )
        for item in output
        if _is_provider_tool_call(item)
    ]


def _is_provider_tool_call(item: Mapping[str, Any]) -> bool:
    item_type = data_get(item, "type", "")

    return (
        isinstance(item_type, str) and item_type.endswith("_call") and item_type != "function_call"
    )
