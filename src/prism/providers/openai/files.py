"""OpenAI's files endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from prism.files.file_data import DeleteFileResult, FileData, FileListResult
from prism.files.request import ListFilesRequest, UploadFileRequest
from prism.http import MultipartBody, MultipartFile

__all__ = [
    "build_list_query",
    "build_upload_form",
    "parse_delete_response",
    "parse_file_data",
    "parse_file_list_response",
]


def build_upload_form(request: UploadFileRequest) -> MultipartBody:
    purpose = request.provider_options.get("purpose")

    return MultipartBody(
        fields={
            # REQUIRED by OpenAI, and it has no default. `assistants` accepts
            # the widest set of file types, so it is the one that fails least
            # often for a caller who did not know they had to choose.
            "purpose": purpose if isinstance(purpose, str) else "assistants",
        },
        files=(
            MultipartFile(
                field="file",
                filename=request.filename,
                content=request.content,
                content_type=request.mime_type,
            ),
        ),
    )


def build_list_query(request: ListFilesRequest) -> dict[str, str]:
    """The list query.

    ``before_id`` is deliberately dropped: OpenAI's files endpoint paginates
    with ``after`` only. Sending ``before`` would be ignored silently, which
    reads to a caller like a working backwards page.
    """
    query: dict[str, str] = {}
    purpose = request.provider_options.get("purpose")

    if request.limit is not None:
        query["limit"] = str(request.limit)

    if request.after_id is not None:
        query["after"] = request.after_id

    if isinstance(purpose, str):
        query["purpose"] = purpose

    return query


def parse_file_data(raw_body: Any) -> FileData:
    if not isinstance(raw_body, dict):
        return FileData()

    size = raw_body.get("bytes")

    return FileData(
        id=_string(raw_body.get("id")) or "",
        filename=_string(raw_body.get("filename")),
        mime_type=_string(raw_body.get("mime_type")),
        size_bytes=size if isinstance(size, int) and not isinstance(size, bool) else None,
        created_at=_created_at(raw_body.get("created_at")),
        purpose=_string(raw_body.get("purpose")),
        raw=raw_body,
    )


def parse_file_list_response(raw_body: Any) -> FileListResult:
    if not isinstance(raw_body, dict):
        return FileListResult()

    data = raw_body.get("data")
    items = data if isinstance(data, list) else []

    return FileListResult(
        data=tuple(parse_file_data(item) for item in items),
        has_more=raw_body.get("has_more") is True,
        first_id=_string(raw_body.get("first_id")),
        last_id=_string(raw_body.get("last_id")),
    )


def parse_delete_response(raw_body: Any) -> DeleteFileResult:
    body = raw_body if isinstance(raw_body, dict) else {}

    return DeleteFileResult(
        id=_string(body.get("id")) or "",
        # The provider's own verdict, not the status code. OpenAI answers 200
        # with `deleted: false` for a file it declined to remove.
        deleted=body.get("deleted") is True,
    )


def _created_at(value: Any) -> str | None:
    """A unix timestamp, rendered as ISO 8601 in UTC.

    The reference uses PHP's ``date('c')``, which renders in the SERVER's local
    zone -- so the same file reports a different creation time on two machines.
    UTC here, deliberately: a timestamp that means something different depending
    on who read it is not a timestamp.
    """
    if not isinstance(value, int) or isinstance(value, bool):
        return None

    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _string(value: Any) -> str | None:
    """None rather than an empty string.

    A missing filename and an empty one are different answers, and a caller
    checking truthiness would treat them the same.
    """
    return value if isinstance(value, str) and value != "" else None
