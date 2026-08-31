"""Mistral's embeddings endpoint."""

from __future__ import annotations

from typing import Any

from prism._php import data_get
from prism.embeddings.request import EmbeddingsRequest
from prism.embeddings.response import EmbeddingsResponse
from prism.providers.mistral.response import validate_response
from prism.value_objects.embedding import Embedding, EmbeddingsUsage
from prism.value_objects.meta import Meta

__all__ = ["build_embeddings_body", "parse_embeddings_response"]


def build_embeddings_body(request: EmbeddingsRequest) -> dict[str, Any]:
    """Narrower than OpenAI's on purpose.

    Mistral takes ``model`` and ``input`` and nothing else the reference sends.
    ``dimensions`` and ``encoding_format`` are OpenAI options and are not
    forwarded -- an unknown key is rejected outright by this endpoint rather
    than ignored.
    """
    return {
        "model": request.model,
        # A LIST even for one input, so a response index maps to an input index
        # without a special case for the single-input call.
        "input": list(request.inputs),
    }


def parse_embeddings_response(raw_body: Any) -> EmbeddingsResponse:
    data = validate_response(raw_body)
    items = data.get("data")
    items = [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []

    # ORDERED BY THE PROVIDER'S OWN INDEX, not by arrival, for the same reason
    # the OpenAI mapping does it: an embeddings caller zips the result against
    # the inputs it sent, so a silent reordering attaches every vector to the
    # wrong text.
    ordered = sorted(items, key=lambda item: _number(item.get("index")))

    return EmbeddingsResponse(
        embeddings=tuple(Embedding.from_list(item.get("embedding") or []) for item in ordered),
        # Mistral reports `total_tokens` only.
        usage=_usage(data_get(data, "usage.total_tokens")),
        meta=Meta(id=_string(data.get("id")), model=_string(data.get("model"))),
        raw=data,
    )


def _usage(value: Any) -> EmbeddingsUsage:
    # None rather than 0 when the provider said nothing: zero tokens claims the
    # call was free.
    return EmbeddingsUsage(
        tokens=value if isinstance(value, int) and not isinstance(value, bool) else None
    )


def _string(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _number(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0
