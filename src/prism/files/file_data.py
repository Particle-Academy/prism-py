"""What a provider says about a stored file."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["DeleteFileResult", "FileData", "FileListResult"]


@dataclass(frozen=True)
class FileData:
    """One file, as the provider describes it."""

    id: str = ""
    filename: str | None = None
    #: NONE on OpenAI, always. Its files endpoint reports ``id``, ``object``,
    #: ``bytes``, ``created_at``, ``filename`` and ``purpose`` -- and no content
    #: type at all. The field is here because the reference has it and another
    #: provider may fill it; it is not a parsing bug when it is empty.
    mime_type: str | None = None
    size_bytes: int | None = None
    #: ISO 8601, in UTC. See :func:`~prism.providers.openai.files.parse_file_data`.
    created_at: str | None = None
    purpose: str | None = None
    raw: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "filename": self.filename,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at,
            "purpose": self.purpose,
        }


@dataclass(frozen=True)
class FileListResult:
    """Files that came back from a list call, and where the page ends."""

    data: tuple[FileData, ...] = ()
    has_more: bool = False
    first_id: str | None = None
    last_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "data": [file.to_dict() for file in self.data],
            "has_more": self.has_more,
            "first_id": self.first_id,
            "last_id": self.last_id,
        }


@dataclass(frozen=True)
class DeleteFileResult:
    """What a delete call reported.

    ``deleted`` is the provider's own answer, not an inference from the status
    code: OpenAI answers 200 with ``deleted: false`` for a file it declined to
    remove, and treating the status as the verdict would report a success that
    did not happen.
    """

    id: str = ""
    deleted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "deleted": self.deleted}
