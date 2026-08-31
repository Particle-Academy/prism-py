"""Embeddings: ordering, coercion, and what a missing token count means."""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from prism import HttpRequest, HttpResponse, Prism, PrismError, canonical
from prism.providers.openai.embeddings import parse_embeddings_response
from prism.value_objects.embedding import Embedding

OK = {
    "id": "emb_1",
    "model": "text-embedding-3-small",
    "data": [
        {"index": 0, "embedding": [0.1, 0.2]},
        {"index": 1, "embedding": [0.3, 0.4]},
    ],
    "usage": {"total_tokens": 9},
}


class RecordingTransport:
    def __init__(self, body: dict[str, Any], status: int = 200) -> None:
        self.body = body
        self.status = status
        self.sent: HttpRequest | None = None

    def send(self, request: HttpRequest) -> HttpResponse:
        self.sent = request
        return HttpResponse(status=self.status, body=canonical.encode(self.body).encode("utf-8"))


def _sent(transport: RecordingTransport) -> dict[str, Any]:
    assert transport.sent is not None
    return json.loads(transport.sent.body.decode("utf-8"))


def test_every_input_is_posted_as_a_list_and_returns_one_vector_each() -> None:
    transport = RecordingTransport(OK)

    response = (
        Prism.embeddings()
        .using("openai", "text-embedding-3-small", {"api_key": "sk-test", "transport": transport})
        .from_input("first")
        .from_input("second")
        .as_embeddings()
    )

    assert transport.sent is not None
    assert transport.sent.url.endswith("/embeddings")
    # A list even for one input, so the response index maps to the input index.
    assert _sent(transport)["input"] == ["first", "second"]
    assert len(response.embeddings) == 2
    assert response.embeddings[0].embedding == (0.1, 0.2)
    assert response.usage.tokens == 9
    assert response.meta.model == "text-embedding-3-small"


def test_vectors_are_ordered_by_the_provider_index_not_by_arrival() -> None:
    # The API documents that `data` may come back out of order, and callers zip
    # the result against the inputs they sent -- so trusting arrival order would
    # attach every vector to the wrong text.
    transport = RecordingTransport(
        {
            **OK,
            "data": [
                {"index": 1, "embedding": [0.3, 0.4]},
                {"index": 0, "embedding": [0.1, 0.2]},
            ],
        }
    )

    response = (
        Prism.embeddings()
        .using("openai", "text-embedding-3-small", {"api_key": "sk-test", "transport": transport})
        .from_array(["first", "second"])
        .as_embeddings()
    )

    assert response.embeddings[0].embedding == (0.1, 0.2)
    assert response.embeddings[1].embedding == (0.3, 0.4)


def test_a_request_with_no_input_is_refused() -> None:
    # Billable, comes back empty, and reads as a provider that answered nothing.
    with pytest.raises(PrismError, match="at least one input"):
        Prism.embeddings().using("openai", "text-embedding-3-small").to_request()


def test_a_missing_token_count_is_none_rather_than_zero() -> None:
    # A caller totalling spend needs "this cost nothing" to differ from "nobody
    # told me what this cost".
    body = {key: value for key, value in OK.items() if key != "usage"}
    transport = RecordingTransport(body)

    response = (
        Prism.embeddings()
        .using("openai", "text-embedding-3-small", {"api_key": "sk-test", "transport": transport})
        .from_input("x")
        .as_embeddings()
    )

    assert response.usage.tokens is None


def test_a_non_numeric_member_is_dropped_rather_than_coerced() -> None:
    # A zero pushed into a vector shifts every distance computed against it. A
    # shorter vector is a visible fault; a zeroed one is not.
    assert Embedding.from_list([0.1, None, 0.2]).embedding == (0.1, 0.2)


def test_a_bool_is_not_treated_as_a_number() -> None:
    # bool subclasses int in Python, so True would otherwise become 1.0.
    assert Embedding.from_list([0.1, True, 0.2]).embedding == (0.1, 0.2)


def test_provider_options_pass_through_and_unset_ones_are_omitted() -> None:
    transport = RecordingTransport(OK)

    (
        Prism.embeddings()
        .using("openai", "text-embedding-3-small", {"api_key": "sk-test", "transport": transport})
        .from_input("x")
        .with_provider_options({"dimensions": 256})
        .as_embeddings()
    )

    sent = _sent(transport)
    assert sent["dimensions"] == 256
    assert "encoding_format" not in sent


def test_an_empty_provider_response_is_refused() -> None:
    with pytest.raises(PrismError):
        parse_embeddings_response(None)


def test_an_unreadable_input_file_names_the_path() -> None:
    # Escaped: the `.` is a path separator here, not a regex wildcard, and an
    # unescaped pattern would pass against a message naming a different file.
    with pytest.raises(PrismError, match=re.escape("does/not/exist.txt")):
        Prism.embeddings().using("openai", "text-embedding-3-small").from_file("does/not/exist.txt")
