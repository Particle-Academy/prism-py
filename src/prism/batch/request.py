"""Frozen batch requests, one per operation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from prism.text.request import Request as TextRequest

__all__ = [
    "BatchRequest",
    "BatchRequestItem",
    "CancelBatchRequest",
    "GetBatchResultsRequest",
    "ListBatchesRequest",
    "RetrieveBatchRequest",
]


@dataclass(frozen=True)
class BatchRequestItem:
    """One request inside a batch, and the id the caller will match it back by.

    ``custom_id`` is the CALLER's, not the provider's: results come back in
    whatever order the provider finished them, so this is the only thing tying
    a result to the request that produced it.
    """

    custom_id: str
    request: TextRequest


@dataclass
class BatchRequest:
    """A batch to create, described either way.

    ``items`` and ``input_file_id`` are alternatives, not a pair. Items mean the
    provider mapping builds a JSONL file and uploads it; a file id means one was
    uploaded already. Which is set is checked at SEND time rather than here,
    because it is a provider rule -- a provider that accepted inline items would
    have nothing to complain about.
    """

    items: tuple[BatchRequestItem, ...] | None = None
    input_file_id: str | None = None
    provider_key: str = ""
    client_options: dict[str, Any] = field(default_factory=dict)
    provider_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class ListBatchesRequest:
    limit: int | None = None
    after_id: str | None = None
    #: Carried and dropped by the OpenAI mapper, exactly as on
    #: :class:`~prism.files.request.ListFilesRequest`.
    before_id: str | None = None
    provider_key: str = ""
    client_options: dict[str, Any] = field(default_factory=dict)
    provider_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class _BatchIdRequest:
    """The three requests that are nothing but a batch id."""

    batch_id: str = ""
    provider_key: str = ""
    client_options: dict[str, Any] = field(default_factory=dict)
    provider_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrieveBatchRequest(_BatchIdRequest):
    pass


@dataclass
class GetBatchResultsRequest(_BatchIdRequest):
    pass


@dataclass
class CancelBatchRequest(_BatchIdRequest):
    pass
