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
ALL_ROWS = SUITE.cases("py")
# A `transport: default` row (corpus 0.1.44, G-386) names `{port}` and is
# fetched over the REAL default transport against a loopback server, reading
# what arrived off the server: prism-parity's tools/media-fetch-guard.mjs runs
# those against this repository's default branch. A fake transport here
# records what the guard decided, not where the client would have connected,
# which is the blind spot the wire rows close. Every other row runs here.
WIRE = "default"
ROWS = [row for row in ALL_ROWS if row.get("transport") != WIRE]
WIRE_ROWS = [row for row in ALL_ROWS if row.get("transport") == WIRE]
CONTENT = b"public image bytes"
REFUSALS = {
    ErrorCode.SCHEME_NOT_ALLOWED,
    ErrorCode.PRIVATE_ADDRESS_REFUSED,
    ErrorCode.HOST_DID_NOT_RESOLVE,
    ErrorCode.REDIRECT_REFUSED,
    ErrorCode.TOO_MANY_REDIRECTS,
}


def test_inventory_and_explicit_legacy_path_skip() -> None:
    # The row count is pinned to the corpus version, so a loader that drops rows
    # fails here instead of running fewer. conformance/parity.lock pins a corpus
    # from before 0.1.29, which has 17 rows; the six G-366 rows (url-0018..0023)
    # arrive when that pin advances, and their address forms are asserted
    # natively in tests/test_guarded_fetch.py meanwhile.
    version = tuple(int(part) for part in Corpus.open().version.split("."))
    assert len(ROWS) == (23 if version >= (0, 1, 29) else 17)
    assert [row["id"] for row in ALL_ROWS] == [f"url-{i:04}" for i in range(1, len(ALL_ROWS) + 1)]
    assert all(row.get("transport") in (None, WIRE) for row in ALL_ROWS)
    # Wire rows follow the 23 transport-free rows; none before corpus 0.1.44.
    assert not WIRE_ROWS or len(ROWS) == 23
    for row in WIRE_ROWS:
        assert row["url"].count("{port}") == 1
        assert row.get("guarded", True) is True or "py" in row.get("skip", {})
    assert [i for i in SUITE.skipped_ids("py") if i not in {row["id"] for row in WIRE_ROWS}] == [
        "url-0009"
    ]
    assert SUITE.skipped_ids("php") == []
    assert SUITE.skipped_ids("ts") == []
    skipped = next(row for row in ROWS if row["id"] == "url-0009")
    assert "no unguarded fetch" in skipped["skip_reason"]
    assert not hasattr(Image, "fetch")
    for row in ROWS:
        if "redirects_always" in row:
            assert isinstance(row["redirects_always"], bool)
        if row.get("redirects_always") is True:
            assert isinstance(row.get("redirects_to"), str) and row["redirects_to"]
            assert row.get("guarded", True) is True


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
            if row.get("redirects_to") and (
                row.get("redirects_always") is True or request.url != row["redirects_to"]
            ):
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
    elif row.get("redirects_always") is True:
        # Six public requests pin the default allowance to five redirect hops.
        assert [request.url for request in sent] == [row["url"]] * 6
        assert not image.has_raw_content()
    elif row.get("redirects_to"):
        assert [request.url for request in sent] == [row["url"]]
        assert row["redirects_to"] not in [request.url for request in sent]
        assert not image.has_raw_content()
    else:
        assert sent == []
        assert not image.has_raw_content()
