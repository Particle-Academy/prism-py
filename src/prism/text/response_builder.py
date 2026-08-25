"""Accumulates steps and turns them into a response."""

from __future__ import annotations

from typing import Any

from prism.text.response import Response
from prism.text.step import Step
from prism.value_objects import AssistantMessage, Message, Usage

__all__ = ["ResponseBuilder"]


class ResponseBuilder:
    """Collects the steps of a generation and assembles the final response."""

    def __init__(self) -> None:
        self.steps: list[Step] = []

    def add_step(self, step: Step) -> ResponseBuilder:
        self.steps.append(step)
        return self

    def to_response(self) -> Response:
        final = self.steps[-1]

        messages: list[Message] = [*final.messages]

        additional_content: dict[str, Any] = dict(final.additional_content)
        if final.provider_tool_calls:
            additional_content["provider_tool_calls"] = final.provider_tool_calls

        messages.append(
            AssistantMessage(
                content=final.text,
                tool_calls=final.tool_calls,
                additional_content=additional_content,
            )
        )

        return Response(
            steps=list(self.steps),
            text=final.text,
            finish_reason=final.finish_reason,
            tool_calls=final.tool_calls,
            tool_results=final.tool_results,
            usage=self._total_usage(),
            meta=final.meta,
            messages=messages,
            # Deliberately the step's own content, without the provider tool
            # calls folded in above — that fold is for the assistant message.
            additional_content=final.additional_content,
            raw=final.raw,
        )

    def _total_usage(self) -> Usage:
        """Sum the counters across steps.

        The nullable counters stay null unless at least ONE step reported them,
        so "no provider ever told us" stays distinguishable from "the provider
        told us zero".
        """
        usages = [step.usage for step in self.steps]

        def total(attribute: str) -> int | None:
            values = [getattr(usage, attribute) for usage in usages]
            if all(value is None for value in values):
                return None
            return sum(value or 0 for value in values)

        costs = [usage.cost for usage in usages]

        return Usage(
            prompt_tokens=sum(usage.prompt_tokens for usage in usages),
            completion_tokens=sum(usage.completion_tokens for usage in usages),
            cache_write_input_tokens=total("cache_write_input_tokens"),
            cache_read_input_tokens=total("cache_read_input_tokens"),
            thought_tokens=total("thought_tokens"),
            cost=None
            if all(cost is None for cost in costs)
            else float(sum(cost or 0.0 for cost in costs)),
        )
