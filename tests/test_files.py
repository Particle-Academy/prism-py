"""Provider-side file storage: five operations on one store."""

from __future__ import annotations

import pytest

from prism import HttpRequest, HttpResponse, Prism, PrismError, canonical
from prism.files.pending_request import FilesPendingRequest
from prism.providers.openai.files import parse_file_data

FILE = {
    "id": "file-abc",
    "object": "file",
    "bytes": 1234,
    "created_at": 1735689600,
    "filename": "notes.pdf",
    "purpose": "assistants",
}


class RecordingTransport:
    def __init__(self, content: bytes, status: int = 200) -> None:
        self.content = content
        self.status = status
        self.sent: HttpRequest | None = None

    def send(self, request: HttpRequest) -> HttpResponse:
        self.sent = request
        return HttpResponse(status=self.status, body=self.content)


def _json(value: object) -> bytes:
    return canonical.encode(value).encode("utf-8")


def _files(transport: RecordingTransport) -> FilesPendingRequest:
    return Prism.files().using("openai", {"api_key": "sk-test", "transport": transport})


def _sent(transport: RecordingTransport) -> HttpRequest:
    assert transport.sent is not None
    return transport.sent


# -- upload ----------------------------------------------------------------


def test_the_bytes_are_posted_as_multipart_and_the_file_comes_back() -> None:
    transport = RecordingTransport(_json(FILE))

    file = _files(transport).upload(b"%PDF-1.4", "notes.pdf")

    sent = _sent(transport)
    assert sent.method == "POST"
    assert sent.url.endswith("/files")
    assert sent.headers["Content-Type"].startswith("multipart/form-data; boundary=")
    assert sent.body is not None
    assert b'filename="notes.pdf"' in sent.body
    assert b"%PDF-1.4" in sent.body
    assert file.id == "file-abc"
    assert file.size_bytes == 1234


def test_the_purpose_is_defaulted_because_openai_requires_one() -> None:
    # `assistants` accepts the widest set of file types, so it is the choice
    # that fails least often for a caller who did not know they had to make it.
    transport = RecordingTransport(_json(FILE))

    _files(transport).upload(b"x", "a.txt")

    body = _sent(transport).body
    assert body is not None
    assert b'name="purpose"' in body
    assert b"assistants" in body


def test_a_purpose_the_caller_chose_is_kept() -> None:
    transport = RecordingTransport(_json(FILE))

    (
        Prism.files()
        .using("openai", {"api_key": "sk-test", "transport": transport})
        .with_provider_options({"purpose": "batch"})
        .upload(b"x", "a.jsonl")
    )

    body = _sent(transport).body
    assert body is not None
    assert b"batch" in body


# -- list ------------------------------------------------------------------


def test_only_the_pagination_fields_that_were_set_are_sent() -> None:
    transport = RecordingTransport(
        _json({"data": [FILE], "has_more": False, "first_id": "file-abc"})
    )

    result = _files(transport).list(10, "file-zzz")

    assert "limit=10" in _sent(transport).url
    assert "after=file-zzz" in _sent(transport).url
    assert len(result.data) == 1
    assert result.has_more is False
    assert result.first_id == "file-abc"


def test_before_id_is_dropped_because_openai_has_no_such_parameter() -> None:
    # Sending it would be ignored silently, which reads to a caller like a
    # working backwards page.
    transport = RecordingTransport(_json({"data": []}))

    _files(transport).list(None, None, "file-aaa")

    assert "before" not in _sent(transport).url
    assert _sent(transport).url.endswith("/files")


# -- metadata, delete, download --------------------------------------------


def test_one_file_is_read_by_id() -> None:
    transport = RecordingTransport(_json(FILE))

    file = _files(transport).get_metadata("file-abc")

    assert _sent(transport).url.endswith("/files/file-abc")
    assert file.filename == "notes.pdf"


def test_the_delete_verdict_comes_from_the_body_not_the_status_code() -> None:
    # OpenAI answers 200 with `deleted: false` for a file it declined to remove,
    # and treating 200 as the verdict reports a success that did not happen.
    transport = RecordingTransport(_json({"id": "file-abc", "object": "file", "deleted": False}))

    result = _files(transport).delete("file-abc")

    assert result.deleted is False
    assert result.id == "file-abc"


def test_download_returns_bytes_rather_than_decoded_text() -> None:
    # A PDF decoded as text is corrupt, and the caller has no way to tell.
    transport = RecordingTransport(b"%PDF")

    content = _files(transport).download("file-abc")

    assert _sent(transport).url.endswith("/files/file-abc/content")
    assert content == b"%PDF"


def test_a_file_id_is_escaped_rather_than_pasted_into_the_path() -> None:
    transport = RecordingTransport(_json(FILE))

    _files(transport).get_metadata("a/../b")

    assert "a%2F..%2Fb" in _sent(transport).url


# -- failures --------------------------------------------------------------


def test_an_http_error_raises() -> None:
    transport = RecordingTransport(_json({"error": {"message": "no such file"}}), status=404)

    with pytest.raises(PrismError, match="no such file"):
        _files(transport).get_metadata("file-nope")


def test_an_error_inside_a_200_raises_too() -> None:
    # The files endpoints do this, so the status alone is not the verdict.
    transport = RecordingTransport(
        _json({"error": {"type": "invalid_request_error", "message": "bad purpose"}})
    )

    with pytest.raises(PrismError, match="bad purpose"):
        _files(transport).upload(b"x", "a.txt")


# -- parsing ---------------------------------------------------------------


def test_created_at_is_rendered_as_iso_8601_in_utc() -> None:
    # The reference uses PHP `date('c')`, which renders in the server's local
    # zone, so the same file reports a different creation time on two machines.
    assert parse_file_data(FILE).created_at == "2025-01-01T00:00:00Z"


def test_mime_type_stays_none_because_openai_never_reports_one() -> None:
    # The field exists because the reference has it; empty is not a parse bug.
    assert parse_file_data(FILE).mime_type is None


def test_a_response_that_is_not_an_object_parses_to_an_empty_file() -> None:
    assert parse_file_data(None).id == ""
