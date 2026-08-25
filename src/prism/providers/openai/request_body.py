"""Building the OpenAI Responses request body."""

from __future__ import annotations

from typing import Any

from prism._php import is_truthy, where_not_null
from prism.providers.openai.maps import map_messages, map_tool_choice, map_tools
from prism.text.request import Request

__all__ = ["build_request_body", "build_tools"]


def build_tools(request: Request) -> list[dict[str, Any]]:
    """Merge provider-native tools IN FRONT of the caller's tools.

    Not the order the two lists were configured in — provider tools always come
    first, and tool order reaches the model.
    """
    tools = map_tools(request.tools)

    if not request.provider_tools:
        return tools

    provider_tools = [{"type": tool.type, **tool.options} for tool in request.provider_tools]

    return [*provider_tools, *tools]


def build_request_body(request: Request) -> dict[str, Any]:
    """Map a :class:`~prism.text.request.Request` onto the request body.

    Three keys — ``model``, ``input`` and ``max_output_tokens`` — are merged
    UNCONDITIONALLY. That is why ``max_output_tokens`` appears as an explicit
    ``null`` on a request that never set it; a port that models "unset" as
    "absent" drops the key and the bytes differ.

    Everything else goes through a not-null filter, so ``False`` and ``0``
    survive and only ``None`` is dropped. ``temperature=0`` is a deliberate
    request for deterministic sampling and reaches the wire.
    """
    body: dict[str, Any] = {
        "model": request.model,
        "input": map_messages(request.messages, request.system_prompts),
        "max_output_tokens": request.max_tokens,
    }

    text_verbosity = request.provider_option("text_verbosity")
    reasoning = request.provider_option("reasoning")

    optional: dict[str, Any] = {
        "temperature": request.temperature,
        "top_p": request.top_p,
        "metadata": request.provider_option("metadata"),
        # An EMPTY tool list collapses to None and the key vanishes. An empty
        # array is falsy in PHP and truthy in Python, so this is one of the
        # places a direct port silently sends `tools: []` — which OpenAI
        # rejects on some models and which changes tool_choice defaults on
        # others.
        "tools": build_tools(request) or None,
        "tool_choice": map_tool_choice(request.tool_choice),
        "parallel_tool_calls": request.provider_option("parallel_tool_calls"),
        "previous_response_id": request.provider_option("previous_response_id"),
        "service_tier": request.provider_option("service_tier"),
        "store": request.provider_option("store"),
        "text": {"verbosity": text_verbosity} if is_truthy(text_verbosity) else None,
        "truncation": request.provider_option("truncation"),
        # Reasoning set explicitly through provider options wins. Otherwise the
        # provider-agnostic toggle speaks only when it was turned OFF.
        "reasoning": reasoning
        if reasoning is not None
        else ({"effort": "minimal"} if request.reasoning_enabled is False else None),
    }

    body.update(where_not_null(optional))

    return body
