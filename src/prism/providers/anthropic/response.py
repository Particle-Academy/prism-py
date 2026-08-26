"""Parsing an Anthropic Messages body into a :class:`~prism.text.response.Response`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from prism._php import data_get, where_not_null
from prism.enums import FinishReason
from prism.errors import PrismError
from prism.providers.anthropic.maps import map_finish_reason, map_tool_calls
from prism.text.request import Request
from prism.text.response import Response
from prism.text.response_builder import ResponseBuilder
from prism.text.step import Step
from prism.value_objects import Meta, Usage

__all__ = ["parse_text_response"]


def parse_text_response(request: Request, raw_body: Mapping[str, Any]) -> Response:
    """Parse a raw Messages body, with no HTTP involved.

    :raises PrismError: ``provider_response_error`` when the body is empty or
        carries an error; ``max_tokens_exceeded`` when the model stopped at the
        token limit; ``tool_loop_not_supported`` when it finished on tool use,
        which this slice does not implement — better a clearly-coded refusal
        than a half-executed loop.
    """
    _validate(raw_body)

    finish_reason = map_finish_reason(raw_body)

    if finish_reason is FinishReason.TOOL_CALLS:
        raise PrismError.tool_loop_not_supported()

    if finish_reason is FinishReason.LENGTH:
        raise PrismError.max_tokens_exceeded(
            data_get(raw_body, "stop_reason", ""), data_get(raw_body, "type", "")
        )

    builder = ResponseBuilder()
    builder.add_step(_build_step(raw_body, request, finish_reason))

    return builder.to_response()


def _validate(data: Mapping[str, Any]) -> None:
    if not data:
        raise PrismError.provider_response_error(
            "Anthropic returned an empty or non-object response body."
        )

    # Anthropic reports some failures with ``type: "error"`` and a 200, so this
    # is checked on the BODY rather than left to the caller's status check.
    if data_get(data, "type") == "error" or data_get(data, "error"):
        raise PrismError.provider_response_error(
            f"Anthropic error [{data_get(data, 'error.type', 'unknown')}]: "
            f"{data_get(data, 'error.message', 'unknown')}"
        )


def _build_step(data: Mapping[str, Any], request: Request, finish_reason: FinishReason) -> Step:
    content: Sequence[Mapping[str, Any]] = data_get(data, "content", []) or []

    return Step(
        text=_output_text(content),
        finish_reason=finish_reason,
        tool_calls=map_tool_calls(
            [block for block in content if data_get(block, "type") == "tool_use"]
        ),
        tool_results=[],
        provider_tool_calls=[],
        usage=_build_usage(data),
        meta=Meta(
            id=data_get(data, "id", ""),
            model=data_get(data, "model", ""),
            rate_limits=[],
            service_tier=None,
        ),
        messages=list(request.messages),
        system_prompts=list(request.system_prompts),
        additional_content=where_not_null(
            {
                "thinking": _thinking(content),
                "stopSequence": data_get(data, "stop_sequence"),
            }
        ),
        raw=dict(data),
    )


def _output_text(content: Sequence[Mapping[str, Any]]) -> str:
    """Every text block joined, not just the first.

    Anthropic splits a reply across blocks when thinking or tool use interleaves
    with it, so taking ``content[0]`` returns a truncated answer that looks
    complete.
    """
    return "".join(
        data_get(block, "text", "") for block in content if data_get(block, "type") == "text"
    )


def _thinking(content: Sequence[Mapping[str, Any]]) -> str | None:
    """Extended-thinking blocks joined, or ``None`` when the model did not think."""
    parts = [
        data_get(block, "thinking", "")
        for block in content
        if data_get(block, "type") == "thinking"
    ]
    joined = "".join(part for part in parts if part)

    return joined or None


def _build_usage(data: Mapping[str, Any]) -> Usage:
    return Usage(
        # Anthropic reports cache tokens SEPARATELY from input_tokens rather
        # than inside them, so nothing is subtracted here. The OpenAI mapping
        # subtracts, and copying that would under-report every prompt.
        prompt_tokens=data_get(data, "usage.input_tokens", 0),
        completion_tokens=data_get(data, "usage.output_tokens", 0),
        cache_write_input_tokens=data_get(data, "usage.cache_creation_input_tokens"),
        cache_read_input_tokens=data_get(data, "usage.cache_read_input_tokens"),
        thought_tokens=None,
    )
