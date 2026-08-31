"""What a batch is, where it is, and what came of each request in it."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from prism.value_objects.usage import Usage

__all__ = [
    "BatchJob",
    "BatchJobRequestCounts",
    "BatchListResult",
    "BatchResultItem",
    "BatchResultStatus",
    "BatchStatus",
]


class BatchStatus(str, Enum):
    """Where a batch is in its lifecycle, as the provider names it."""

    VALIDATING = "validating"
    IN_PROGRESS = "in_progress"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class BatchResultStatus(str, Enum):
    """How one item inside a batch turned out."""

    SUCCEEDED = "succeeded"
    ERRORED = "errored"
    CANCELED = "canceled"
    EXPIRED = "expired"


@dataclass(frozen=True)
class BatchJobRequestCounts:
    """How many requests are in each state.

    FIVE counts and a total, matching the reference -- but OpenAI reports only
    three (``total``, ``completed``, ``failed``), so ``canceled`` and
    ``expired`` stay zero on that provider. They are not dropped, because a
    batch cancelled mid-flight is a real thing another provider does report.
    """

    processing: int = 0
    succeeded: int = 0
    failed: int = 0
    canceled: int = 0
    expired: int = 0
    total: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "processing": self.processing,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "canceled": self.canceled,
            "expired": self.expired,
            "total": self.total,
        }


@dataclass(frozen=True)
class BatchJobError:
    code: str = ""
    message: str = ""
    line: int | None = None
    param: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "line": self.line,
            "param": self.param,
        }


@dataclass(frozen=True)
class BatchJob:
    """A submitted batch."""

    id: str = ""
    status: BatchStatus = BatchStatus.VALIDATING
    request_counts: BatchJobRequestCounts = field(default_factory=BatchJobRequestCounts)
    #: ISO 8601, in UTC. See :func:`~prism.providers.openai.batch.parse_batch_job`.
    created_at: str | None = None
    expires_at: str | None = None
    ended_at: str | None = None
    results_url: str | None = None
    input_file_id: str | None = None
    output_file_id: str | None = None
    error_file_id: str | None = None
    errors: tuple[BatchJobError, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status.value,
            "request_counts": self.request_counts.to_dict(),
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "ended_at": self.ended_at,
            "results_url": self.results_url,
            "input_file_id": self.input_file_id,
            "output_file_id": self.output_file_id,
            "error_file_id": self.error_file_id,
            "errors": [error.to_dict() for error in self.errors],
        }


@dataclass(frozen=True)
class BatchListResult:
    """One page of batches."""

    data: tuple[BatchJob, ...] = ()
    has_more: bool = False
    last_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "data": [job.to_dict() for job in self.data],
            "has_more": self.has_more,
            "last_id": self.last_id,
        }


@dataclass(frozen=True)
class BatchResultItem:
    """What one request in a batch produced.

    ``text`` and ``usage`` are None on a failure and ``error_type`` /
    ``error_message`` are None on a success -- the four are read together with
    ``status``, which is the field that says which pair means anything.
    """

    custom_id: str = ""
    status: BatchResultStatus = BatchResultStatus.SUCCEEDED
    text: str | None = None
    usage: Usage | None = None
    message_id: str | None = None
    model: str | None = None
    error_type: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "custom_id": self.custom_id,
            "status": self.status.value,
            "text": self.text,
            "usage": None if self.usage is None else self.usage.to_dict(),
            "message_id": self.message_id,
            "model": self.model,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }
