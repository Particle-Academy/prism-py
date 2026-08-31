"""Building Mistral's chat-completions request bodies."""

from __future__ import annotations

from typing import Any

from prism._php import where_not_null
from prism.providers.mistral.maps import map_messages, map_tool_choice, map_tools
from prism.structured.request import StructuredRequest
from prism.text.request import Request

__all__ = ["build_fim_body", "build_request_body", "build_structured_body"]


def build_request_body(request: Request) -> dict[str, Any]:
    """The chat-completions body.

    ``model``, ``messages`` and ``max_tokens`` go in UNCONDITIONALLY, matching
    the reference -- so ``max_tokens`` is present as an explicit None on a
    request that never set one. Everything else passes a NOT-NULL filter, so
    ``0`` and ``False`` survive and only None is dropped:
    ``with_temperature(0)`` is a real setting.

    An empty tool list collapses to None BEFORE the filter, so the key vanishes.
    An empty list sent as ``"tools": []`` changes ``tool_choice`` defaults on
    some models and is rejected outright by others.
    """
    tools = map_tools(request.tools)

    return {
        "model": request.model,
        "messages": map_messages(request.messages, request.system_prompts),
        "max_tokens": request.max_tokens,
        **where_not_null(
            {
                "temperature": request.temperature,
                "top_p": request.top_p,
                "reasoning_effort": request.provider_options.get("reasoning_effort"),
                "tools": tools or None,
                "tool_choice": map_tool_choice(request.tool_choice) if tools else None,
            }
        ),
    }


def build_structured_body(request: StructuredRequest) -> dict[str, Any]:
    """The same body, plus a schema -- and never both a schema and tools.

    MISTRAL REFUSES ``response_format`` AND ``tools`` IN ONE REQUEST. The
    reference works around it with a two-pass loop: send with tools, let the
    model call them, then re-send without tools and with the schema to get the
    final JSON.

    This port does not run tools (the execution loop is deferred -- see the
    parity manifest), so there is no loop to two-pass. Tools are DROPPED here and
    the schema is sent, which is the half that matters for ``structured``: a
    caller who declared tools on a structured request would otherwise get an
    error from the provider naming a conflict they did not create.
    """
    schema = request.schema

    return {
        "model": request.model,
        "messages": map_messages(request.messages, request.system_prompts),
        "max_tokens": request.max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "schema": {} if schema is None else schema.to_dict(),
                "name": "response" if schema is None else schema.name,
                # Mistral's own strict mode, so this IS enforced rather than
                # requested -- unlike the Anthropic path, which can only ask.
                # See G-08.
                "strict": True,
            },
        },
        **where_not_null(
            {
                "temperature": request.temperature,
                "top_p": request.top_p,
                "reasoning_effort": request.provider_options.get("reasoning_effort"),
            }
        ),
    }


def build_fim_body(request: Any) -> dict[str, Any]:
    """The fill-in-the-middle body.

    ``model`` and ``prompt`` are unconditional; everything else passes the
    NOT-NULL filter so ``with_temperature(0)`` survives.

    ``stop`` collapses to None when EMPTY, matching the reference. Sent as
    ``"stop": []`` it reads, to anyone inspecting the request, as a caller who
    asked for no stop sequences rather than one who never set any.
    """
    return {
        "model": request.model,
        "prompt": request.prompt,
        **where_not_null(
            {
                # The text AFTER the gap. Omitting it is legal and means
                # "complete to the end", which is a different request rather
                # than a degraded one.
                "suffix": request.suffix,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "stop": list(request.stop) if request.stop else None,
            }
        ),
    }
