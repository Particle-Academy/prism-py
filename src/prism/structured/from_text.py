"""Turn a parsed text response into a structured one."""

from __future__ import annotations

from prism.structured.extract import extract_structured
from prism.structured.response import StructuredResponse
from prism.structured.step import StructuredStep
from prism.text.response import Response
from prism.value_objects import AssistantMessage

__all__ = ["structured_from_text_response"]


def structured_from_text_response(
    response: Response,
    *,
    native_output_format: bool = False,
    parsed_step: bool = False,
    assistant_additional_content: bool = True,
) -> StructuredResponse:
    """Reuse provider parsing and retain the completed structured conversation."""
    steps = []
    for original in response.steps:
        step = StructuredStep.from_step(original)
        step.messages.append(
            AssistantMessage(
                original.text,
                list(original.tool_calls),
                dict(original.additional_content) if assistant_additional_content else {},
                list(original.tool_approval_requests),
            )
        )
        if native_output_format:
            step.text = ""
        if native_output_format or parsed_step:
            step.structured = extract_structured(original.text) or {}
        steps.append(step)
    return StructuredResponse(
        steps=tuple(steps),
        text="" if native_output_format else response.text,
        structured=extract_structured(response.text),
        finish_reason=response.finish_reason,
        usage=response.usage,
        meta=response.meta,
        additional_content=dict(response.additional_content),
        raw=None if response.raw is None else dict(response.raw),
        tool_calls=tuple(call for step in steps for call in step.tool_calls),
        provider_tool_calls=tuple(call for step in steps for call in step.provider_tool_calls),
        tool_results=tuple(result for step in steps for result in step.tool_results),
    )
