"""Run both real embedding handlers against the loader's PHP reference rows."""

from __future__ import annotations

import base64
import math
from collections.abc import Callable
from typing import Any

import pytest
from prism_conformance import Corpus, compare

from prism import HttpRequest, HttpResponse, Prism, PrismError, canonical
from prism.embeddings.response import EmbeddingsResponse
from prism.providers.mistral.embeddings import parse_embeddings_response as mistral
from prism.providers.openai.embeddings import parse_embeddings_response as openai

# Missing loader/corpus is a collection failure, never an optional test skip.
CASES = Corpus.open().suite("embedding-decoding").cases("py")
ENCODED = "AACAPwAAIMAAAAA/AAAAAA=="
PARSERS = [openai, mistral]
Parser = Callable[[Any], EmbeddingsResponse]


def body(value: Any) -> dict[str, Any]:
    return {"data": [{"index": 0, "embedding": value}], "usage": {"total_tokens": 9}}


def test_all_nine_canonical_rows_execute_without_skips() -> None:
    assert len(CASES) == 9
    assert not any(row["skipped"] for row in CASES)
    assert sum("exception" in row["expected"]["php"] for row in CASES) == 7


@pytest.mark.parametrize("parse", PARSERS, ids=["openai", "mistral"])
@pytest.mark.parametrize("case", CASES, ids=[row["id"] for row in CASES])
def test_matches_the_executed_php_reference(parse: Parser, case: dict[str, Any]) -> None:
    try:
        response = parse(body(case["embedding"]))
        assert len(response.embeddings) == 1
        actual: dict[str, Any] = {"embedding": list(response.embeddings[0].embedding)}
    except PrismError as error:
        # An unrelated coded error must not count as malformed-vector refusal.
        assert error.code == "invalid_embedding"
        actual = {"exception": "Prism\\Prism\\Exceptions\\PrismException"}

    assert compare(canonical.encode(case["expected"]["php"]), canonical.encode(actual))


@pytest.mark.parametrize("parse", PARSERS, ids=["openai", "mistral"])
@pytest.mark.parametrize(
    "value",
    [
        ENCODED + "\n",
        ENCODED[:-2],
        ENCODED + "=",
        "____fw==",
        "é" + ENCODED,
        "AA==",
        "AAAA",
        "AACA/w==",
        False,
        42,
        {},
    ],
    ids=[
        "whitespace",
        "missing-padding",
        "extra-padding",
        "url-safe",
        "non-ascii",
        "one-byte",
        "three-bytes",
        "negative-infinity",
        "boolean",
        "number",
        "object",
    ],
)
def test_refuses_with_coded_error_and_no_raw_payload(parse: Parser, value: Any) -> None:
    with pytest.raises(PrismError) as raised:
        parse(body(value))

    error = raised.value
    assert error.code == "invalid_embedding"
    assert error.body is None
    assert error.__cause__ is None
    assert str(error).startswith("Invalid embedding: ")
    if isinstance(value, str):
        assert value not in str(error)


@pytest.mark.parametrize("parse", PARSERS, ids=["openai", "mistral"])
def test_missing_embedding_is_a_coded_refusal(parse: Parser) -> None:
    with pytest.raises(PrismError) as raised:
        parse({"data": [{"index": 0}]})
    assert raised.value.code == "invalid_embedding"


@pytest.mark.parametrize("parse", PARSERS, ids=["openai", "mistral"])
def test_finite_float32_extremes_and_signed_zero(parse: Parser) -> None:
    # Fixed IEEE-754 words, independent of the decoder's float packing path.
    encoded = base64.b64encode(bytes.fromhex("0000008001000000ffff7f7f")).decode("ascii")
    values = parse(body(encoded)).embeddings[0].embedding
    assert values[0] == 0 and math.copysign(1, values[0]) == -1
    assert values[1:] == (2**-149, (2 - 2**-23) * 2**127)


@pytest.mark.parametrize("parse", PARSERS, ids=["openai", "mistral"])
def test_existing_numeric_array_behavior_is_preserved(parse: Parser) -> None:
    assert parse(body([1, None, -2.5])).embeddings[0].embedding == (1, -2.5)
    assert parse(body([])).embeddings[0].embedding == ()


@pytest.mark.parametrize("parse", PARSERS, ids=["openai", "mistral"])
def test_one_corrupt_vector_refuses_the_whole_response(parse: Parser) -> None:
    with pytest.raises(PrismError) as raised:
        parse({"data": [{"embedding": ENCODED}, {"embedding": "not!base64"}]})
    assert raised.value.code == "invalid_embedding"


class VectorTransport:
    def send(self, request: HttpRequest) -> HttpResponse:
        return HttpResponse(
            status=200,
            body=canonical.encode(
                {
                    "data": [{"index": 1, "embedding": ENCODED}, {"index": 0, "embedding": [3]}],
                    "usage": {"total_tokens": 9},
                }
            ).encode("utf-8"),
        )


@pytest.mark.parametrize("provider", ["openai", "mistral"])
def test_public_request_path_preserves_ordering_and_usage(provider: str) -> None:
    response = (
        Prism.embeddings()
        .using(provider, "test-embedding", {"api_key": "test-only", "transport": VectorTransport()})
        .from_array(["first", "second"])
        .as_embeddings()
    )
    assert [item.embedding for item in response.embeddings] == [(3,), (1, -2.5, 0.5, 0)]
    assert response.usage.tokens == 9
