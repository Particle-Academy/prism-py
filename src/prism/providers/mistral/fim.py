"""Parsing Mistral's fill-in-the-middle completions."""

from __future__ import annotations

from typing import Any

from prism._php import data_get
from prism.enums import FinishReason
from prism.fim.request import FimRequest
from prism.fim.response import FimResponse
from prism.providers.mistral.response import first_choice_message, validate_response
from prism.value_objects.meta import Meta
from prism.value_objects.usage import Usage

__all__ = ["parse_fim_response"]

#: DELIBERATELY NARROWER than the chat map, and matching the reference: FIM
#: answers only `stop` or `length`, so `tool_calls` and `content_filter` are not
#: special-cased here.
_FIM_FINISH_REASONS = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "model_length": FinishReason.LENGTH,
}


def parse_fim_response(raw_body: Any, request: FimRequest) -> FimResponse:
    """Parse a FIM completion.

    Unlike :func:`~prism.providers.mistral.response.parse_text_response`, a
    LENGTH finish does NOT raise. Hitting the token ceiling is an ordinary
    outcome for a completion: the caller wanted as much of the gap as the budget
    bought, and the partial text is useful. ``finish_reason`` carries the fact.

    An unrecognised reason becomes ``UNKNOWN`` rather than ``STOP`` -- a
    truncated completion reported as complete is how an editor silently inserts
    half a function.
    """
    data = validate_response(raw_body)
    message = first_choice_message(data)
    reason = data_get(data, "choices.0.finish_reason", "")

    return FimResponse(
        text=_string(message.get("content")),
        finish_reason=_FIM_FINISH_REASONS.get(
            reason if isinstance(reason, str) else "", FinishReason.UNKNOWN
        ),
        usage=Usage(
            prompt_tokens=_number(data_get(data, "usage.prompt_tokens")),
            completion_tokens=_number(data_get(data, "usage.completion_tokens")),
        ),
        meta=Meta(
            id=_string(data.get("id")),
            model=_string(data.get("model")) or request.model,
        ),
        raw=data,
    )


def _string(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _number(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0
