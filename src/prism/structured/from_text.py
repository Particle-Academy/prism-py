"""Turn a parsed text response into a structured one."""

from __future__ import annotations

from prism.structured.extract import extract_structured
from prism.structured.response import StructuredResponse
from prism.text.response import Response

__all__ = ["structured_from_text_response"]


def structured_from_text_response(response: Response) -> StructuredResponse:
    """Share the text path's parsing, and add the one thing that differs.

    Every provider that answers a structured request answers it as TEXT --
    OpenAI with a schema-constrained string, Anthropic with a message it was
    asked to keep to JSON. So finish-reason handling, usage and metadata are the
    text path's, and this adds the parse.

    Sharing it this way is what keeps a structured call from quietly diverging:
    when the text parser learns that a provider reports token limits
    differently, structured learns it in the same commit.
    """
    return StructuredResponse(
        steps=tuple(response.steps),
        text=response.text,
        structured=extract_structured(response.text),
        finish_reason=response.finish_reason,
        usage=response.usage,
        meta=response.meta,
        additional_content=dict(response.additional_content),
        raw=None if response.raw is None else dict(response.raw),
    )
