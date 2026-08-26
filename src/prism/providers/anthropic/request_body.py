"""Building the Anthropic Messages request body."""

from __future__ import annotations

from typing import Any

from prism._php import where_not_null
from prism.providers.anthropic.maps import (
    map_messages,
    map_system,
    map_tool_choice,
    map_tools,
)
from prism.text.request import Request

__all__ = ["DEFAULT_MAX_TOKENS", "build_request_body"]

#: Anthropic requires ``max_tokens`` and documents no default.
#:
#: The OpenAI body sends ``max_output_tokens: null`` for a request that never
#: set one; doing the same here is a 400. A number has to be chosen, so it is
#: named here rather than buried, and it is generous enough that hitting it
#: means the caller genuinely wanted a long answer and should say so.
DEFAULT_MAX_TOKENS = 4096


def build_request_body(request: Request) -> dict[str, Any]:
    """Map a :class:`~prism.text.request.Request` onto the request body.

    The same two rules as the OpenAI body decide every key: ``model``,
    ``messages`` and ``max_tokens`` are merged UNCONDITIONALLY, and everything
    else goes through a not-null filter so ``temperature=0`` survives and only
    ``None`` is dropped.

    What differs is ``system``: a top-level field here, not a message, so system
    prompts never enter ``messages`` at all.
    """
    body: dict[str, Any] = {
        "model": request.model,
        "messages": map_messages(request.messages),
        "max_tokens": request.max_tokens if request.max_tokens is not None else DEFAULT_MAX_TOKENS,
    }

    optional: dict[str, Any] = {
        "system": map_system(request.system_prompts),
        "temperature": request.temperature,
        "top_p": request.top_p,
        "top_k": request.top_k,
        # An EMPTY tool list collapses to None and the key vanishes. Sending
        # `tools: []` changes tool_choice defaults on some models and is
        # rejected outright by others.
        "tools": map_tools(request.tools) or None,
        "tool_choice": map_tool_choice(request.tool_choice),
        # Extended thinking. Asymmetric like OpenAI's reasoning and for the same
        # reason: enabling it emits nothing, because a budget is a per-provider
        # setting the toggle must not invent.
        "thinking": request.provider_option("thinking"),
        "metadata": request.provider_option("metadata"),
        "stop_sequences": request.provider_option("stop_sequences"),
    }

    body.update(where_not_null(optional))

    return body
