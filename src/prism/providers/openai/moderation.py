"""OpenAI's moderation endpoint."""

from __future__ import annotations

from typing import Any

from prism.errors import PrismError
from prism.moderation.request import ModerationRequest
from prism.moderation.response import ModerationResponse
from prism.value_objects.meta import Meta
from prism.value_objects.moderation_result import ModerationResult

__all__ = ["build_moderation_body", "parse_moderation_response"]


def build_moderation_body(request: ModerationRequest) -> dict[str, Any]:
    return {"model": request.model, "input": list(request.inputs)}


def parse_moderation_response(raw_body: Any, model: str) -> ModerationResponse:
    if not isinstance(raw_body, dict):
        # Refused rather than returning an empty response. An empty response has
        # `is_flagged() is False`, so a caller gating on it would let the content
        # through -- a safety check that fails open on a malformed reply.
        raise PrismError.provider_response_error(
            "OpenAI returned an empty or non-object moderation response."
        )

    results = raw_body.get("results")
    items = results if isinstance(results, list) else []

    return ModerationResponse(
        results=tuple(ModerationResult.from_dict(item) for item in items),
        meta=Meta(id=_str(raw_body.get("id")), model=_str(raw_body.get("model")) or model),
        raw=raw_body,
    )


def _str(value: Any) -> str:
    return value if isinstance(value, str) else ""
