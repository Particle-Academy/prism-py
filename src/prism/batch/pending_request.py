"""The fluent builder for batch jobs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from prism.batch.batch_job import BatchJob, BatchListResult, BatchResultItem
from prism.batch.request import (
    BatchRequest,
    BatchRequestItem,
    CancelBatchRequest,
    GetBatchResultsRequest,
    ListBatchesRequest,
    RetrieveBatchRequest,
)
from prism.enums import Provider as ProviderName
from prism.errors import ErrorCode, PrismError
from prism.providers.base import Provider
from prism.registry import resolve_provider

__all__ = ["BatchPendingRequest"]


class BatchPendingRequest:
    """Five terminals, like :class:`~prism.files.pending_request.FilesPendingRequest`.

    Submitting a batch, polling one, listing them, reading results and
    cancelling are five operations on a queue rather than five renderings of one
    request.
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
    ) -> BatchPendingRequest:
        self._provider_key = provider.value if isinstance(provider, ProviderName) else provider
        self._provider = resolve_provider(self._provider_key, provider_config or {})
        return self

    def with_provider_options(self, options: dict[str, Any]) -> BatchPendingRequest:
        self._provider_options = dict(options)
        return self

    def with_client_options(self, options: dict[str, Any]) -> BatchPendingRequest:
        self._client_options = dict(options)
        return self

    # -- terminals ---------------------------------------------------------

    def create(
        self,
        items: Sequence[BatchRequestItem] | None = None,
        input_file_id: str | None = None,
    ) -> BatchJob:
        result: BatchJob = self._require_provider().batch(
            self.to_create_request(items, input_file_id)
        )
        return result

    def retrieve(self, batch_id: str) -> BatchJob:
        result: BatchJob = self._require_provider().retrieve_batch(
            self.to_retrieve_request(batch_id)
        )
        return result

    def list(
        self,
        limit: int | None = None,
        after_id: str | None = None,
        before_id: str | None = None,
    ) -> BatchListResult:
        result: BatchListResult = self._require_provider().list_batches(
            self.to_list_request(limit, after_id, before_id)
        )
        return result

    def get_results(self, batch_id: str) -> tuple[BatchResultItem, ...]:
        result: tuple[BatchResultItem, ...] = self._require_provider().get_batch_results(
            self.to_get_results_request(batch_id)
        )
        return result

    def cancel(self, batch_id: str) -> BatchJob:
        result: BatchJob = self._require_provider().cancel_batch(self.to_cancel_request(batch_id))
        return result

    # -- freezing ----------------------------------------------------------

    def to_create_request(
        self,
        items: Sequence[BatchRequestItem] | None = None,
        input_file_id: str | None = None,
    ) -> BatchRequest:
        return BatchRequest(
            items=None if items is None else tuple(items),
            input_file_id=input_file_id,
            **self._base(),
        )

    def to_retrieve_request(self, batch_id: str) -> RetrieveBatchRequest:
        return RetrieveBatchRequest(batch_id=batch_id, **self._base())

    def to_list_request(
        self,
        limit: int | None = None,
        after_id: str | None = None,
        before_id: str | None = None,
    ) -> ListBatchesRequest:
        return ListBatchesRequest(
            limit=limit, after_id=after_id, before_id=before_id, **self._base()
        )

    def to_get_results_request(self, batch_id: str) -> GetBatchResultsRequest:
        return GetBatchResultsRequest(batch_id=batch_id, **self._base())

    def to_cancel_request(self, batch_id: str) -> CancelBatchRequest:
        return CancelBatchRequest(batch_id=batch_id, **self._base())

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
