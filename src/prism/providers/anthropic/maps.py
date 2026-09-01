"""Mapping between Prism's value objects and Anthropic's Messages wire shapes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from prism import canonical
from prism._php import data_get, where_not_null
from prism.enums import FinishReason, ToolChoice
from prism.errors import PrismError
from prism.tool import Tool
from prism.value_objects import ToolCall
from prism.value_objects.media_file import Document, Image
from prism.value_objects.messages import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)

__all__ = [
    "map_finish_reason",
    "map_messages",
    "map_system",
    "map_tool_calls",
    "map_tool_choice",
    "map_tools",
]

_FINISH_REASONS = {
    # Both mean the model chose to stop. The distinction is about HOW, which the
    # raw payload still carries for anyone who needs it.
    "end_turn": FinishReason.STOP,
    "stop_sequence": FinishReason.STOP,
    "tool_use": FinishReason.TOOL_CALLS,
    "max_tokens": FinishReason.LENGTH,
    "pause_turn": FinishReason.PAUSE,
    "refusal": FinishReason.REFUSAL,
}


def map_finish_reason(data: Mapping[str, Any]) -> FinishReason:
    """Map ``stop_reason`` onto the shared enum.

    An unrecognised reason becomes ``UNKNOWN`` rather than ``STOP``. Guessing
    ``STOP`` would present a truncated or refused generation as a complete one.
    """
    return _FINISH_REASONS.get(data_get(data, "stop_reason", ""), FinishReason.UNKNOWN)


def map_system(system_prompts: Sequence[SystemMessage]) -> str | None:
    """The top-level ``system`` field, or ``None`` when there is nothing to say.

    Several prompts join with a blank line between them. Anthropic accepts an
    array of blocks here too, but a joined string is what the reference sends
    and the bytes are the contract.
    """
    parts = [prompt.content for prompt in system_prompts if prompt.content]

    return "\n\n".join(parts) if parts else None


def map_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Map messages onto Anthropic's ``messages`` array.

    Four things differ from the Responses API and each is easy to get subtly
    wrong:

    1. System prompts are NOT messages. They are a top-level field, which is why
       :func:`map_system` exists separately.
    2. Tool calls stay ATTACHED to the assistant turn as ``tool_use`` blocks.
    3. A tool result is a USER turn keyed by ``tool_use_id`` — the id of the
       CALL, the opposite of the Responses API which keys on the result id.
    4. Turns must alternate, so consecutive tool results travel as blocks inside
       ONE user turn rather than as several user turns.

    :raises PrismError: ``unknown_message_type``
    """
    items: list[dict[str, Any]] = []

    for message in messages:
        if isinstance(message, SystemMessage):
            # Reaching here means a system message was passed as a MESSAGE.
            # Anthropic has no system role, and silently demoting it to a user
            # turn would change what the model was told.
            raise PrismError.unknown_message_type(
                "a system message in the message list; pass it as a system prompt"
            )

        if isinstance(message, UserMessage):
            items.append(
                {
                    "role": "user",
                    # Text leads, then images, then documents -- the reference's
                    # order.
                    "content": [
                        {"type": "text", "text": message.text()},
                        *(_map_image(image) for image in message.images()),
                        *(_map_document(document) for document in message.documents()),
                    ],
                    **message.additional_attributes,
                }
            )
        elif isinstance(message, AssistantMessage):
            _append_assistant(message, items)
        elif isinstance(message, ToolResultMessage):
            _append_tool_results(message, items)
        else:
            raise PrismError.unknown_message_type(type(message).__name__)

    return items


def _append_assistant(message: AssistantMessage, items: list[dict[str, Any]]) -> None:
    content: list[dict[str, Any]] = []

    # Text first. Anthropic reads blocks in order, and a tool_use ahead of the
    # reasoning that led to it reads as a model that decided first and explained
    # afterwards.
    if message.content:
        content.append({"type": "text", "text": message.content})

    for tool_call in message.tool_calls:
        content.append(
            {
                "type": "tool_use",
                # The CALL id. Anthropic echoes it back on the matching
                # tool_result and rejects a mismatch rather than ignoring it.
                "id": tool_call.id,
                "name": tool_call.name,
                # An OBJECT, not a JSON string. The Responses API takes a
                # document nested inside a document; Anthropic takes the
                # arguments themselves.
                "input": tool_call.decoded_arguments(),
            }
        )

    # A turn with neither text nor tool calls contributes nothing — Anthropic
    # rejects an empty content array.
    if content:
        items.append({"role": "assistant", "content": content})


def _append_tool_results(message: ToolResultMessage, items: list[dict[str, Any]]) -> None:
    blocks = [
        {
            "type": "tool_result",
            "tool_use_id": result.tool_call_id,
            "content": _stringify(result.result),
        }
        for result in message.tool_results
    ]

    if not blocks:
        return

    # Appended to the previous turn when that turn is already a user turn:
    # Anthropic requires roles to alternate and two user turns in a row is a 400.
    if items and items[-1]["role"] == "user" and isinstance(items[-1]["content"], list):
        items[-1]["content"] = [*items[-1]["content"], *blocks]
        return

    items.append({"role": "user", "content": blocks})


def _stringify(result: Any) -> str:
    """A tool's return value as text. Objects are canonicalised, not inspected."""
    return result if isinstance(result, str) else canonical.encode(result)


def map_tools(tools: Sequence[Tool]) -> list[dict[str, Any]]:
    """Map tools onto Anthropic's ``tools`` array.

    Anthropic names the schema field ``input_schema``, not ``parameters``, and
    it is NOT optional the way OpenAI's is: a tool without one is rejected. So a
    parameterless tool sends an empty object schema rather than omitting the key.

    Declaration order is preserved. Tool order reaches the model.
    """
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": {
                "type": "object",
                "properties": tool.parameters_as_dict() if tool.has_parameters() else {},
                "required": list(tool.required_parameters),
            },
        }
        for tool in tools
    ]


def map_tool_choice(choice: str | ToolChoice | None) -> dict[str, Any] | None:
    """Map the shared tool choice onto Anthropic's object form.

    Anthropic takes an OBJECT where OpenAI takes a string, and spells "the model
    must call something" as ``any`` rather than ``required``.
    """
    if choice is None:
        return None

    if choice is ToolChoice.AUTO:
        return {"type": "auto"}

    if choice is ToolChoice.ANY:
        return {"type": "any"}

    if choice is ToolChoice.NONE:
        return {"type": "none"}

    return {"type": "tool", "name": str(choice)}


def map_tool_calls(blocks: Sequence[Mapping[str, Any]]) -> list[ToolCall]:
    """Read ``tool_use`` content blocks into tool calls."""
    return [
        ToolCall(
            id=data_get(block, "id", ""),
            name=data_get(block, "name", ""),
            # An object on the wire, unlike the Responses API's JSON string.
            arguments=data_get(block, "input", {}) or {},
        )
        for block in blocks
    ]


def _map_image(image: Image) -> dict[str, Any]:
    """Anthropic's image block.

    The payload never appears at the top level: it goes in a ``source`` object
    whose OWN ``type`` says which of the three forms it is. So an image block
    carries two ``type`` keys at two depths meaning different things -- the
    outer one is the block kind, the inner one is where the bytes came from.
    """
    return {"type": "image", "source": _image_source(image)}


def _image_source(image: Image) -> dict[str, Any]:
    if image.is_file_id():
        return {"type": "file", "file_id": image.file_id()}

    if image.is_url():
        return {"type": "url", "url": image.url}

    encoded = image.base64()
    mime_type = image.mime_type()

    if encoded is None or mime_type is None:
        raise PrismError.unsupported_media(
            "Anthropic", "image", "a file id, a url, or bytes with a known mime type"
        )

    return {"type": "base64", "media_type": mime_type, "data": encoded}


def _map_document(document: Document) -> dict[str, Any]:
    """Anthropic's document block.

    The only provider here that takes a document five ways, and the only one
    that takes CHUNKS at all -- pre-split text as a ``content`` source, each
    chunk its own text block.

    Text documents go as ``text``, not base64: Anthropic reads a ``text/*``
    source directly and base64-wrapping it would make the model's citations
    point into an encoded blob.
    """
    return where_not_null(
        {
            "type": "document",
            "title": document.document_title(),
            "source": _document_source(document),
        }
    )


def _document_source(document: Document) -> dict[str, Any]:
    if document.is_file_id():
        return {"type": "file", "file_id": document.file_id()}

    if document.is_url():
        return {"type": "url", "url": document.url}

    chunks = document.chunks()

    if chunks is not None:
        return {
            "type": "content",
            "content": [{"type": "text", "text": chunk} for chunk in chunks],
        }

    mime_type = document.mime_type()
    raw_content = document.raw_content()

    if mime_type is not None and mime_type.startswith("text/"):
        if raw_content is None:
            raise PrismError.unsupported_media("Anthropic", "text document", "bytes it can decode")

        # Refused BY NAME rather than left to raise UnicodeDecodeError. A
        # document declared `text/plain` that is not actually UTF-8 is a caller
        # mistake, and a coded error says which part is wrong; an uncoded
        # decoder exception escaping a mapper says only that something failed
        # deep inside. `prism-ts` replaced the invalid bytes with U+FFFD and
        # sent the corruption on -- Anthropic cites into that content -- so both
        # ports now refuse it, which is the answer they have to share.
        try:
            decoded = raw_content.decode("utf-8")
        except UnicodeDecodeError as error:
            raise PrismError.unsupported_media(
                "Anthropic",
                "text document",
                f"valid UTF-8, or a mime type that is not text/* -- {error}",
            ) from error

        return {"type": "text", "media_type": mime_type, "data": decoded}

    encoded = document.base64()

    if encoded is None or mime_type is None:
        raise PrismError.unsupported_media(
            "Anthropic",
            "document",
            "a file id, a url, chunks, or bytes with a known mime type",
        )

    return {"type": "base64", "media_type": mime_type, "data": encoded}
