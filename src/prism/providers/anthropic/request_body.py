"""Building the Anthropic Messages request body."""

from __future__ import annotations

from collections.abc import Mapping
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

#: Anthropic's minimum thinking budget, and the reference's default.
DEFAULT_THINKING_BUDGET = 1024


def build_request_body(request: Request) -> dict[str, Any]:
    """Map a :class:`~prism.text.request.Request` onto the request body.

    The same two rules as the OpenAI body decide every key: ``model``,
    ``messages`` and ``max_tokens`` are merged UNCONDITIONALLY, and everything
    else goes through a not-null filter so ``temperature=0`` survives and only
    ``None`` is dropped.

    What differs is ``system``: a top-level field here, not a message, so system
    prompts never enter ``messages`` at all.
    """
    effort = request.provider_option("effort")

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
        "thinking": _thinking(request),
        "metadata": request.provider_option("metadata"),
        "stop_sequences": request.provider_option("stop_sequences"),
        # ``effort`` is Prism's name for it; Anthropic reads it from
        # output_config. This port used to drop it while the reference sent it
        # (G-57).
        "output_config": {"effort": effort} if effort is not None else None,
    }

    body.update(where_not_null(optional))

    return body


def _thinking(request: Request) -> Any:
    """The ``thinking`` field, spelled the way the reference spells it.

    Asymmetric like OpenAI's reasoning and for the same reason: enabling
    reasoning emits nothing, because a budget is a per-provider setting the
    toggle must not invent. ``with_reasoning(False)`` does win over a
    ``thinking`` option, as it does in the reference.

    ``{"enabled": True, "budgetTokens": n}`` is Prism's spelling, not
    Anthropic's, and becomes ``{"type": "enabled", "budget_tokens": n}``. Sent as
    given it was a 400, so a mode that worked in PHP failed here. A budget that
    is not an integer falls back to 1024, Anthropic's minimum, as in the
    reference.

    A map with ``enabled`` and no ``type`` that is not ``True`` asks for no
    thinking, and so does an empty map. Every other shape,
    ``{"type": "adaptive"}`` included, is sent as given, as the reference sends
    it.
    """
    if request.reasoning_enabled is False:
        return None

    thinking = request.provider_option("thinking")

    if not isinstance(thinking, Mapping):
        return thinking

    if not thinking:
        return None

    if thinking.get("type") == "adaptive":
        return thinking

    if thinking.get("enabled") is True:
        budget = thinking.get("budgetTokens")

        return {
            "type": "enabled",
            # bool is an int in Python and not in PHP, so True is not a budget.
            "budget_tokens": budget
            if isinstance(budget, int) and not isinstance(budget, bool)
            else DEFAULT_THINKING_BUDGET,
        }

    if "enabled" in thinking and "type" not in thinking:
        return None

    return thinking
