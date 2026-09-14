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
from prism.value_objects import Meta, ProviderRateLimit, Usage

__all__ = ["parse_text_response"]


def parse_text_response(
    request: Request,
    raw_body: Mapping[str, Any],
    rate_limits: Sequence[ProviderRateLimit] = (),
) -> Response:
    """Parse a raw Messages body, with no HTTP involved.

    :raises PrismError: ``provider_response_error`` when the body is empty or
        carries an error; ``tool_loop_not_supported`` when it finished on tool
        use, which this slice does not implement -- better a clearly-coded
        refusal than a half-executed loop.

    A Length finish does NOT raise. It returns the partial response with
    ``finish_reason`` set, matching the reference; see the note in the body.
    """
    _validate(raw_body)

    finish_reason = map_finish_reason(raw_body)

    if finish_reason is FinishReason.TOOL_CALLS:
        raise PrismError.tool_loop_not_supported()

    # A Length finish RETURNS the partial answer here, and does not raise.
    #
    # The reference's stance is per-PROVIDER rather than uniform, which is easy
    # to misread: Anthropic and Mistral return, OpenAI's Responses handler and
    # its structured handler raise, and Azure raises. This port previously
    # raised for all three providers, so it matched the reference on OpenAI and
    # diverged on the other two.
    #
    # Returning is the right behaviour to copy here. The model did write
    # something usable and you paid for the tokens, and running out of room is
    # exactly when the usage numbers matter most -- it is the expensive case and
    # the unfinished one at once, so raising discards the reasoning-token count
    # on the very call where a caller most wants it.
    #
    # The cost is real and is the caller's to manage: code that ignores
    # finish_reason will treat a truncated answer as a complete one. That is why
    # FinishReason.LENGTH is on the response rather than implied.
    #
    # Found by prism-parity's anthropic-text-response suite (G-50). The existing
    # seventeen OpenAI rows never caught it because none of them sends a
    # max_tokens finish.

    builder = ResponseBuilder()
    builder.add_step(_build_step(raw_body, request, finish_reason, rate_limits))

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


def _build_step(
    data: Mapping[str, Any],
    request: Request,
    finish_reason: FinishReason,
    rate_limits: Sequence[ProviderRateLimit],
) -> Step:
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
            rate_limits=list(rate_limits),
            service_tier=None,
        ),
        messages=list(request.messages),
        system_prompts=list(request.system_prompts),
        additional_content=where_not_null(
            {
                **_thinking(content),
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


def _thinking(content: Sequence[Mapping[str, Any]]) -> dict[str, str | None]:
    """The first thinking block's text and signature, as the reference keeps them.

    The signature is what lets the block be sent back on a later tool-use turn
    (G-57). It covers one block's text exactly, so the text kept beside it is
    that block's, not every block joined: joined text under one signature is a
    block Anthropic would refuse. A response with more than one thinking block
    keeps only the first, in all three languages.
    """
    block = next((block for block in content if data_get(block, "type") == "thinking"), None)
    text = data_get(block, "thinking") if block is not None else None
    signature = data_get(block, "signature") if block is not None else None

    return {
        "thinking": text if isinstance(text, str) else None,
        "thinking_signature": signature if isinstance(signature, str) else None,
    }


def _build_usage(data: Mapping[str, Any]) -> Usage:
    return Usage(
        # Anthropic reports cache tokens SEPARATELY from input_tokens rather
        # than inside them, so nothing is subtracted here. The OpenAI mapping
        # subtracts, and copying that would under-report every prompt.
        prompt_tokens=data_get(data, "usage.input_tokens", 0),
        completion_tokens=data_get(data, "usage.output_tokens", 0),
        cache_write_input_tokens=data_get(data, "usage.cache_creation_input_tokens"),
        cache_read_input_tokens=data_get(data, "usage.cache_read_input_tokens"),
        # Reasoning tokens, and a BREAKDOWN of completion_tokens rather than an
        # addition to them. Reported by the Moic Suite team against the live
        # API; this was hardcoded to None, and the other two languages simply
        # never set it -- so all three agreed, and agreement is not correctness.
        thought_tokens=data_get(data, "usage.output_tokens_details.thinking_tokens"),
    )
