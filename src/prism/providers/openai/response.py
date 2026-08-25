"""Parsing an OpenAI Responses body into a :class:`~prism.text.response.Response`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from prism._php import data_get, where_not_null
from prism.enums import FinishReason
from prism.errors import PrismError
from prism.providers.openai.maps import (
    map_finish_reason,
    map_provider_tool_calls,
    map_tool_calls,
)
from prism.text.request import Request
from prism.text.response import Response
from prism.text.response_builder import ResponseBuilder
from prism.text.step import Step
from prism.value_objects import Meta, Usage

__all__ = ["parse_text_response"]


def parse_text_response(request: Request, raw_body: Mapping[str, Any]) -> Response:
    """Parse a raw Responses body, with no HTTP involved.

    :raises PrismError: ``provider_response_error`` when the body is empty or
        carries an ``error``; ``max_tokens_exceeded`` when the model stopped at
        the token limit; ``tool_loop_not_supported`` when the model finished on
        tool calls, which is a capability this slice does not implement — better
        a clearly-coded refusal than a half-executed loop.
    """
    validate_response_body(raw_body)

    finish_reason = map_finish_reason(raw_body)

    if finish_reason is FinishReason.TOOL_CALLS:
        raise PrismError.tool_loop_not_supported()

    if finish_reason is FinishReason.LENGTH:
        raise PrismError.max_tokens_exceeded(
            data_get(raw_body, "output.{last}.status", "n/a"),
            data_get(raw_body, "output.{last}.type", "n/a"),
        )

    builder = ResponseBuilder()
    builder.add_step(_build_step(raw_body, request, finish_reason))

    return builder.to_response()


def validate_response_body(raw_body: Mapping[str, Any]) -> None:
    """Refuse a body the provider filled with an error instead of an answer."""
    if not raw_body or data_get(raw_body, "error") is not None:
        raise PrismError.provider_response_error(
            "OpenAI error: "
            f"[{data_get(raw_body, 'error.type', 'unknown')}] "
            f"{data_get(raw_body, 'error.message', 'unknown')}"
        )


def _build_step(
    data: Mapping[str, Any],
    request: Request,
    finish_reason: FinishReason,
) -> Step:
    output: list[Mapping[str, Any]] = list(data_get(data, "output", []) or [])

    # The LAST output item holds the answer. On a reasoning model the earlier
    # items are the thinking, and on an empty output there is no item at all —
    # which is a response the provider is entitled to send.
    text = data_get(data, "output.{last}.content.0.text")

    return Step(
        text="" if text is None else text,
        finish_reason=finish_reason,
        tool_calls=map_tool_calls(
            [item for item in output if data_get(item, "type") == "function_call"]
        ),
        tool_results=[],
        provider_tool_calls=map_provider_tool_calls(output),
        usage=_build_usage(data),
        meta=Meta(
            id=data_get(data, "id", ""),
            model=data_get(data, "model", ""),
            rate_limits=[],
            service_tier=data_get(data, "service_tier"),
        ),
        messages=list(request.messages),
        system_prompts=list(request.system_prompts),
        additional_content=_build_additional_content(output),
        raw=dict(data),
    )


def _build_usage(data: Mapping[str, Any]) -> Usage:
    cached = data_get(data, "usage.input_tokens_details.cached_tokens")

    return Usage(
        # Cached tokens are SUBTRACTED from the prompt count and reported
        # separately. Passing input_tokens straight through double-counts the
        # cache on every prompt-cached request.
        prompt_tokens=data_get(data, "usage.input_tokens", 0) - (cached or 0),
        completion_tokens=data_get(data, "usage.output_tokens", 0),
        cache_write_input_tokens=None,
        cache_read_input_tokens=cached,
        thought_tokens=data_get(data, "usage.output_tokens_details.reasoning_tokens"),
    )


def _build_additional_content(output: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Everything the response carried besides the answer itself.

    Citation extraction would sit first in this mapping; it is outside this
    slice, and a null entry is dropped by the not-null filter anyway.
    """
    return where_not_null(
        {
            "searchQueries": _web_search_actions(output, "search", "query") or None,
            "openPageUrls": _web_search_actions(output, "open_page", "url") or None,
            "findInPagePatterns": _web_search_actions(output, "find_in_page", "pattern") or None,
            "reasoningSummaries": _reasoning_summaries(output),
        }
    )


def _web_search_actions(
    output: Sequence[Mapping[str, Any]],
    action_type: str,
    field: str,
) -> list[str]:
    values: list[str] = []

    for item in output:
        if data_get(item, "type") != "web_search_call":
            continue
        if data_get(item, "action.type") != action_type:
            continue

        value = data_get(item, f"action.{field}")
        if value:
            values.append(value)

    # Unique, order preserved.
    return list(dict.fromkeys(values))


def _reasoning_summaries(output: Sequence[Mapping[str, Any]]) -> list[str]:
    summaries: list[str] = []

    for item in output:
        if data_get(item, "type") != "reasoning":
            continue

        for entry in data_get(item, "summary", []) or []:
            text = data_get(entry, "text")
            if text:
                summaries.append(text)

    return summaries
