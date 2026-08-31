"""OpenAI's batch endpoints."""

from __future__ import annotations

import json as _json
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from prism import canonical
from prism.batch.batch_job import (
    BatchJob,
    BatchJobError,
    BatchJobRequestCounts,
    BatchListResult,
    BatchResultItem,
    BatchResultStatus,
    BatchStatus,
)
from prism.batch.request import BatchRequestItem, ListBatchesRequest
from prism.errors import PrismError
from prism.providers.openai.request_body import build_request_body
from prism.value_objects.usage import Usage

__all__ = [
    "batch_status_from_value",
    "build_batch_body",
    "build_batch_input_file",
    "build_batch_list_query",
    "parse_batch_job",
    "parse_batch_list_response",
    "parse_batch_result_item",
    "parse_batch_results",
]


def build_batch_input_file(items: Sequence[BatchRequestItem]) -> str:
    """The JSONL body OpenAI's batch endpoint expects, one request per line."""
    return "\n".join(
        canonical.encode(
            {
                "custom_id": item.custom_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": build_request_body(item.request),
            }
        )
        for item in items
    )


def build_batch_body(input_file_id: str, completion_window: Any) -> dict[str, Any]:
    return {
        "input_file_id": input_file_id,
        "endpoint": "/v1/responses",
        # Required by the endpoint, and `24h` is OpenAI's only supported window
        # today -- so a default here beats an error naming a field the caller
        # never knew existed.
        "completion_window": completion_window if isinstance(completion_window, str) else "24h",
    }


def build_batch_list_query(request: ListBatchesRequest) -> dict[str, str]:
    query: dict[str, str] = {}

    if request.limit is not None:
        query["limit"] = str(request.limit)

    if request.after_id is not None:
        query["after"] = request.after_id

    return query


def batch_status_from_value(value: Any) -> BatchStatus:
    """A wire status, or a refusal.

    Matching the reference, an unrecognised status RAISES rather than mapping to
    a plausible member. It reads harsh inside a polling loop, and it is still
    right: a new status means the provider changed something about the
    lifecycle, and quietly calling it in-progress would leave a caller waiting
    forever on a batch that had already stopped.
    """
    try:
        return BatchStatus(value)
    except ValueError as error:
        raise PrismError.provider_response_error(f"Unknown OpenAI batch status: {value}") from error


def parse_batch_job(raw_body: Any) -> BatchJob:
    if not isinstance(raw_body, dict):
        raise PrismError.provider_response_error(
            "OpenAI returned an empty or non-object batch response."
        )

    counts = raw_body.get("request_counts")
    counts = counts if isinstance(counts, dict) else {}
    total = _number(counts.get("total"))
    succeeded = _number(counts.get("completed"))
    failed = _number(counts.get("failed"))

    return BatchJob(
        id=_string(raw_body.get("id")) or "",
        status=batch_status_from_value(raw_body.get("status")),
        request_counts=BatchJobRequestCounts(
            # OpenAI does not report an in-flight count, so it is derived.
            # Clamped at zero: a partial response -- total missing while
            # completed is not -- produces a negative, and a negative number of
            # in-flight requests is not a state anything can be in.
            processing=max(0, total - succeeded - failed),
            succeeded=succeeded,
            failed=failed,
            # OpenAI reports neither, so they stay zero rather than guessed.
            canceled=0,
            expired=0,
            total=total,
        ),
        created_at=_timestamp(raw_body.get("created_at")),
        expires_at=_timestamp(raw_body.get("expires_at")),
        # OpenAI's field is `completed_at`; the port's is `ended_at`, matching
        # the reference, because a cancelled batch also ends.
        ended_at=_timestamp(raw_body.get("completed_at")),
        results_url=_string(raw_body.get("results_url")),
        input_file_id=_string(raw_body.get("input_file_id")),
        output_file_id=_string(raw_body.get("output_file_id")),
        error_file_id=_string(raw_body.get("error_file_id")),
        errors=_errors(raw_body.get("errors")),
    )


def parse_batch_list_response(raw_body: Any) -> BatchListResult:
    if not isinstance(raw_body, dict):
        return BatchListResult()

    data = raw_body.get("data")
    items = data if isinstance(data, list) else []

    return BatchListResult(
        data=tuple(parse_batch_job(entry) for entry in items),
        has_more=raw_body.get("has_more") is True,
        last_id=_string(raw_body.get("last_id")),
    )


def parse_batch_results(body: str) -> tuple[BatchResultItem, ...]:
    """The results file, one JSON object per line.

    A line that will not parse is SKIPPED rather than fatal. The file is written
    by the provider a request at a time, and one malformed record should not
    cost a caller the other nine hundred.
    """
    items: list[BatchResultItem] = []

    for line in body.split("\n"):
        trimmed = line.strip()

        if trimmed == "":
            continue

        try:
            decoded = _json.loads(trimmed)
        except ValueError:
            continue

        if isinstance(decoded, dict):
            items.append(parse_batch_result_item(decoded))

    return tuple(items)


def parse_batch_result_item(data: dict[str, Any]) -> BatchResultItem:
    custom_id = _string(data.get("custom_id")) or ""
    error = data.get("error")

    if isinstance(error, dict):
        code = _string(error.get("code")) or ""

        return BatchResultItem(
            custom_id=custom_id,
            # `batch_expired` is its own outcome, not an error: the request was
            # never run, so reporting it as a failure would blame the request.
            status=BatchResultStatus.EXPIRED
            if code == "batch_expired"
            else BatchResultStatus.ERRORED,
            error_type=code,
            error_message=_string(error.get("message")),
        )

    response = data.get("response")
    body = response.get("body") if isinstance(response, dict) else None
    body = body if isinstance(body, dict) else {}

    return BatchResultItem(
        custom_id=custom_id,
        status=BatchResultStatus.SUCCEEDED,
        text=_extract_text(body),
        usage=_extract_usage(body),
        message_id=_string(body.get("id")),
        model=_string(body.get("model")),
    )


def _extract_text(body: dict[str, Any]) -> str:
    """The assistant's text, from either API shape.

    Responses output is searched from the END: a turn that used tools has the
    message last, and taking the first would return whatever preceded the tool
    call.
    """
    output = body.get("output")
    output = output if isinstance(output, list) else []

    for item in reversed(output):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue

        content = item.get("content")
        content = content if isinstance(content, list) else []

        for part in content:
            if (
                isinstance(part, dict)
                and part.get("type") == "output_text"
                and isinstance(part.get("text"), str)
            ):
                text: str = part["text"]
                return text

    # The chat-completions shape, for a batch submitted against that endpoint.
    choices = body.get("choices")
    choices = choices if isinstance(choices, list) else []

    if choices:
        first = choices[0]
        message = first.get("message") if isinstance(first, dict) else None

        if isinstance(message, dict) and isinstance(message.get("content"), str):
            content_text: str = message["content"]
            return content_text

    return ""


def _extract_usage(body: dict[str, Any]) -> Usage:
    usage = body.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    input_details = usage.get("input_tokens_details")
    input_details = input_details if isinstance(input_details, dict) else {}
    output_details = usage.get("output_tokens_details")
    output_details = output_details if isinstance(output_details, dict) else {}

    return Usage(
        prompt_tokens=_number(usage.get("input_tokens", usage.get("prompt_tokens"))),
        completion_tokens=_number(usage.get("output_tokens", usage.get("completion_tokens"))),
        # Cache WRITE is not reported here. `cached_tokens` is what was READ
        # from the cache -- the two are billed differently, and putting one in
        # the other's slot misreports the cost in the direction that flatters it.
        cache_write_input_tokens=None,
        cache_read_input_tokens=_optional_number(input_details.get("cached_tokens")),
        thought_tokens=_optional_number(output_details.get("reasoning_tokens")),
    )


def _errors(value: Any) -> tuple[BatchJobError, ...]:
    data = value.get("data") if isinstance(value, dict) else None
    entries = data if isinstance(data, list) else []

    return tuple(
        BatchJobError(
            code=_string(entry.get("code")) or "",
            message=_string(entry.get("message")) or "",
            line=_optional_number(entry.get("line")),
            param=_string(entry.get("param")),
        )
        for entry in entries
        if isinstance(entry, dict)
    )


def _timestamp(value: Any) -> str | None:
    """Unix seconds to ISO 8601 in UTC. Same reasoning as ``files``."""
    if not isinstance(value, int) or isinstance(value, bool):
        return None

    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value != "" else None


def _number(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _optional_number(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
