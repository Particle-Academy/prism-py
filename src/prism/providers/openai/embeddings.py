"""OpenAI's embeddings endpoint."""

from __future__ import annotations

from typing import Any

from prism.embeddings.request import EmbeddingsRequest
from prism.embeddings.response import EmbeddingsResponse
from prism.errors import PrismError
from prism.value_objects.embedding import Embedding, EmbeddingsUsage
from prism.value_objects.meta import Meta

__all__ = ["build_embeddings_body", "parse_embeddings_response"]


def build_embeddings_body(request: EmbeddingsRequest) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": request.model,
        # A LIST even for one input, which the API accepts either way -- and
        # keeping the shape constant means the response index maps to the input
        # index without a special case for the single-input call.
        "input": list(request.inputs),
    }

    for key in ("dimensions", "encoding_format", "user"):
        value = request.provider_options.get(key)
        if value is not None:
            body[key] = value

    return body


def parse_embeddings_response(raw_body: Any) -> EmbeddingsResponse:
    if not isinstance(raw_body, dict):
        raise PrismError.provider_response_error(
            "OpenAI returned an empty or non-object embeddings response."
        )

    data = raw_body.get("data")
    items = [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []

    # ORDERED BY THE PROVIDER'S OWN INDEX, not by arrival. The API documents
    # that `data` may come back out of order, and an embeddings caller almost
    # always zips the result against the inputs it sent -- so a silent
    # reordering here would attach every vector to the wrong text.
    items.sort(key=_index)

    return EmbeddingsResponse(
        embeddings=tuple(
            Embedding.from_list(
                item["embedding"] if isinstance(item.get("embedding"), list) else []
            )
            for item in items
        ),
        usage=EmbeddingsUsage(_tokens(raw_body)),
        meta=Meta(id=_str(raw_body.get("id")), model=_str(raw_body.get("model"))),
        raw=raw_body,
    )


def _index(item: dict[str, Any]) -> int:
    """A missing or non-integer index sorts first rather than raising.

    A provider that omitted it has malfunctioned, and stable order beats an
    exception here: the caller still gets its vectors and can see the raw body.
    """
    index = item.get("index")

    return index if isinstance(index, int) else 0


def _tokens(body: dict[str, Any]) -> int | None:
    """None when the provider said nothing, rather than 0.

    A caller totalling spend across calls needs to tell "this cost nothing"
    from "nobody told me what this cost"; folding them together makes the
    second silently understate the first.
    """
    usage = body.get("usage")

    if not isinstance(usage, dict):
        return None

    total = usage.get("total_tokens")

    return total if isinstance(total, int) else None


def _str(value: Any) -> str:
    return value if isinstance(value, str) else ""
