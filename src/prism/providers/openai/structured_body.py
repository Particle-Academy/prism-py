"""The text body, plus the instruction that makes it structured."""

from __future__ import annotations

from typing import Any

from prism.enums import StructuredMode
from prism.providers.openai.request_body import build_request_body
from prism.providers.openai.structured_mode import resolve_structured_mode
from prism.structured.request import StructuredRequest

__all__ = ["build_structured_body"]


def build_structured_body(request: StructuredRequest) -> dict[str, Any]:
    """Build on top of the text body rather than beside it.

    Every rule that decides the text body -- the unconditional keys, the
    not-null filter, the empty-tools collapse -- applies identically here. A
    second builder would drift from the first the moment either changed.

    ``AUTO`` is resolved to a concrete mode against the MODEL, because "auto" is
    not something OpenAI accepts. Somebody has to decide, and doing it here
    means the decision is inspectable in one place rather than implied by the
    wire format.
    """
    body = build_request_body(request)
    mode = (
        resolve_structured_mode(request.model)
        if request.mode is StructuredMode.AUTO
        else request.mode
    )

    # `text` may already hold a verbosity from the text builder. Merged rather
    # than overwritten: asking for a schema is not a reason to silently drop the
    # caller's verbosity setting.
    text_option = body.get("text")
    existing: dict[str, Any] = text_option if isinstance(text_option, dict) else {}

    return {**body, "text": {**existing, "format": _format(request, mode)}}


def _format(request: StructuredRequest, mode: StructuredMode) -> dict[str, Any]:
    if mode is StructuredMode.JSON:
        # JSON mode guarantees syntactic validity and nothing about the shape.
        # The schema still reaches the model through the prompt the caller
        # wrote; it cannot be enforced here, and pretending otherwise would be
        # the dangerous half of this feature.
        return {"type": "json_object"}

    # The builder refuses a request without a schema, so this cannot be None by
    # the time a provider sees it.
    schema = request.schema
    if schema is None:  # pragma: no cover - defensive
        raise ValueError("A structured request reached the provider without a schema.")

    return {
        "type": "json_schema",
        "name": schema.name,
        "schema": schema.to_dict(),
        # The whole reason to prefer this mode: without it the model may return
        # a near-miss that parses and is missing a required field.
        "strict": True,
    }
