"""The fluent builder for provider-side file storage."""

from __future__ import annotations

from typing import Any

from prism.enums import Provider as ProviderName
from prism.errors import ErrorCode, PrismError
from prism.files.file_data import DeleteFileResult, FileData, FileListResult
from prism.files.request import (
    DeleteFileRequest,
    DownloadFileRequest,
    GetFileMetadataRequest,
    ListFilesRequest,
    UploadFileRequest,
)
from prism.providers.base import Provider
from prism.registry import resolve_provider

__all__ = ["FilesPendingRequest"]


class FilesPendingRequest:
    """FIVE terminals, not one.

    These are five different operations on a store, not five renderings of one
    request. ``using()`` is the only thing they share -- there is no model, no
    prompt and nothing to accumulate between calls, so the builder is thin by
    nature rather than by omission.

    ``using()`` also takes no model, unlike every other capability's, because
    the file endpoints take none.
    """

    def __init__(self) -> None:
        self._provider: Provider | None = None
        self._provider_key = ""
        self._provider_options: dict[str, Any] = {}
        self._client_options: dict[str, Any] = {}

    def using(
        self,
        provider: str | ProviderName,
        provider_config: dict[str, Any] | None = None,
    ) -> FilesPendingRequest:
        self._provider_key = provider.value if isinstance(provider, ProviderName) else provider
        self._provider = resolve_provider(self._provider_key, provider_config or {})
        return self

    def with_provider_options(self, options: dict[str, Any]) -> FilesPendingRequest:
        self._provider_options = dict(options)
        return self

    def with_client_options(self, options: dict[str, Any]) -> FilesPendingRequest:
        self._client_options = dict(options)
        return self

    # -- terminals ---------------------------------------------------------

    def upload(self, content: bytes, filename: str, mime_type: str | None = None) -> FileData:
        result: FileData = self._require_provider().upload_file(
            self.to_upload_request(content, filename, mime_type)
        )
        return result

    def list(
        self,
        limit: int | None = None,
        after_id: str | None = None,
        before_id: str | None = None,
    ) -> FileListResult:
        result: FileListResult = self._require_provider().list_files(
            self.to_list_request(limit, after_id, before_id)
        )
        return result

    def get_metadata(self, file_id: str) -> FileData:
        result: FileData = self._require_provider().get_file_metadata(
            self.to_get_metadata_request(file_id)
        )
        return result

    def delete(self, file_id: str) -> DeleteFileResult:
        result: DeleteFileResult = self._require_provider().delete_file(
            self.to_delete_request(file_id)
        )
        return result

    def download(self, file_id: str) -> bytes:
        """The file's bytes.

        The reference returns a PHP string, which is a byte array; ``bytes``
        says the same thing in a language where ``str`` would mean text and
        would corrupt a PDF on the way through.
        """
        result: bytes = self._require_provider().download_file(self.to_download_request(file_id))
        return result

    # -- freezing ----------------------------------------------------------

    def to_upload_request(
        self, content: bytes, filename: str, mime_type: str | None = None
    ) -> UploadFileRequest:
        return UploadFileRequest(
            filename=filename,
            content=content,
            mime_type=mime_type,
            **self._base(),
        )

    def to_list_request(
        self,
        limit: int | None = None,
        after_id: str | None = None,
        before_id: str | None = None,
    ) -> ListFilesRequest:
        return ListFilesRequest(
            limit=limit,
            after_id=after_id,
            before_id=before_id,
            **self._base(),
        )

    def to_get_metadata_request(self, file_id: str) -> GetFileMetadataRequest:
        return GetFileMetadataRequest(file_id=file_id, **self._base())

    def to_delete_request(self, file_id: str) -> DeleteFileRequest:
        return DeleteFileRequest(file_id=file_id, **self._base())

    def to_download_request(self, file_id: str) -> DownloadFileRequest:
        return DownloadFileRequest(file_id=file_id, **self._base())

    # -- internals ---------------------------------------------------------

    def _base(self) -> dict[str, Any]:
        return {
            "provider_key": self._provider_key,
            "client_options": dict(self._client_options),
            "provider_options": dict(self._provider_options),
        }

    def _require_provider(self) -> Provider:
        if self._provider is None:
            raise PrismError(
                ErrorCode.UNSUPPORTED_PROVIDER_ACTION,
                "No provider configured. Call using(<provider>) first.",
            )

        return self._provider
