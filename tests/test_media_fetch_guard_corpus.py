"""Execute the shipped corpus; an absent loader must fail, never skip rows."""

from typing import Any

import pytest

from prism import ErrorCode, HttpRequest, HttpResponse, Image, PrismError

try:
    from prism_conformance import Corpus
except ImportError as error:
    # A skip here would be green without exercising a single shared row.
    raise RuntimeError(
        "Cannot load prism_conformance. Clone prism-parity into ./.parity and run "
        "`python -m pip install ./.parity/loaders/py` before pytest."
    ) from error

SUITE = Corpus.open().suite("media-fetch-guard")
ROWS = SUITE.cases("py")
CONTENT = b"public image bytes"
REFUSALS = {
    ErrorCode.SCHEME_NOT_ALLOWED,
    ErrorCode.PRIVATE_ADDRESS_REFUSED,
    ErrorCode.HOST_DID_NOT_RESOLVE,
    ErrorCode.REDIRECT_REFUSED,
    ErrorCode.TOO_MANY_REDIRECTS,
}


def test_inventory_and_explicit_legacy_path_skip() -> None:
    assert [row["id"] for row in ROWS] == [f"url-{i:04}" for i in range(1, 12)]
    assert SUITE.skipped_ids("py") == ["url-0009"]
    assert SUITE.skipped_ids("php") == []
    assert SUITE.skipped_ids("ts") == []
    skipped = next(row for row in ROWS if row["id"] == "url-0009")
    assert "no unguarded fetch" in skipped["skip_reason"]
    assert not hasattr(Image, "fetch")


@pytest.mark.parametrize("row", ROWS, ids=lambda row: row["id"])
def test_recorded_refusal_and_requests(row: dict[str, Any]) -> None:
    if row["skipped"]:
        # This is the corpus's explicit per-language applicability decision.
        assert row["id"] == "url-0009"
        pytest.skip(row["skip_reason"])
    assert row.get("guarded", True) is True
    sent: list[HttpRequest] = []

    class Resolver:
        def resolve(self, host: str) -> list[str]:
            return row["resolves"].get(host, [])

    class FakeTransport:
        def send(self, request: HttpRequest) -> HttpResponse:
            # Fresh fake per row: a previous control cannot swallow a redirect.
            sent.append(request)
            assert request.follow_redirects is False
            if row.get("redirects_to") and request.url != row["redirects_to"]:
                return HttpResponse(302, b"", {"location": row["redirects_to"]})
            return HttpResponse(200, CONTENT, {"content-type": "image/png"})

    image = Image.from_url(row["url"])
    refusal = None
    try:
        image.fetch_public(transport=FakeTransport(), resolver=Resolver())
    except PrismError as error:
        if error.code not in REFUSALS:
            raise
        refusal = error.code
    assert refusal == row["refusal"]["php"]
    assert refusal == row["refusal"]["ts"]
    if row["refusal"]["py"] is not None:
        assert refusal == row["refusal"]["py"]
    if refusal is None:
        assert [request.url for request in sent] == [row["url"]]
        assert image.raw_content() == CONTENT
        assert image.mime_type() == "image/png"
    elif row.get("redirects_to"):
        assert [request.url for request in sent] == [row["url"]]
        assert row["redirects_to"] not in [request.url for request in sent]
        assert not image.has_raw_content()
    else:
        assert sent == []
        assert not image.has_raw_content()
