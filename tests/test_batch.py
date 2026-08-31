"""Batch jobs: submit, poll, list, read results, cancel."""

from __future__ import annotations

import json
from typing import Any

import pytest

from prism import (
    BatchPendingRequest,
    BatchRequestItem,
    BatchResultStatus,
    BatchStatus,
    HttpRequest,
    HttpResponse,
    Prism,
    PrismError,
    canonical,
)
from prism.providers.openai.batch import parse_batch_job, parse_batch_results

JOB = {
    "id": "batch_1",
    "status": "in_progress",
    "request_counts": {"total": 10, "completed": 4, "failed": 1},
    "created_at": 1735689600,
    "input_file_id": "file-in",
}


class ScriptedTransport:
    """Answers each call in turn, so a two-hop operation runs end to end."""

    def __init__(self, responses: list[bytes] | list[tuple[bytes, int]]) -> None:
        self.responses: list[Any] = list(responses)
        self.calls: list[HttpRequest] = []

    def send(self, request: HttpRequest) -> HttpResponse:
        self.calls.append(request)
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        entry = self.responses[index]
        content, status = entry if isinstance(entry, tuple) else (entry, 200)
        return HttpResponse(status=status, body=content)


def _json(value: object) -> bytes:
    return canonical.encode(value).encode("utf-8")


def _batch(transport: ScriptedTransport) -> BatchPendingRequest:
    return Prism.batch().using("openai", {"api_key": "sk-test", "transport": transport})


# -- create ----------------------------------------------------------------


def test_items_are_uploaded_as_jsonl_and_then_the_batch_is_created() -> None:
    transport = ScriptedTransport([_json({"id": "file-in"}), _json(JOB)])

    item = BatchRequestItem(
        "row-1",
        Prism.text().using("openai", "gpt-4o").with_prompt("Who?").to_request(),
    )

    job = _batch(transport).create([item])

    # Two hops: the file, then the batch that points at it.
    assert len(transport.calls) == 2
    assert transport.calls[0].url.endswith("/files")
    upload = transport.calls[0].body
    assert upload is not None
    assert b"batch" in upload

    jsonl_start = upload.index(b'{"custom_id"')
    line = json.loads(upload[jsonl_start:].split(b"\r\n")[0].decode("utf-8"))
    assert line["custom_id"] == "row-1"
    assert line["url"] == "/v1/responses"
    assert line["body"]["model"] == "gpt-4o"

    assert transport.calls[1].url.endswith("/batches")
    sent = transport.calls[1].body
    assert sent is not None
    assert json.loads(sent.decode("utf-8")) == {
        "input_file_id": "file-in",
        "endpoint": "/v1/responses",
        "completion_window": "24h",
    }
    assert job.id == "batch_1"


def test_the_upload_is_skipped_when_an_input_file_id_was_given() -> None:
    transport = ScriptedTransport([_json(JOB)])

    _batch(transport).create(None, "file-already-there")

    assert len(transport.calls) == 1
    body = transport.calls[0].body
    assert body is not None
    assert json.loads(body.decode("utf-8"))["input_file_id"] == "file-already-there"


def test_both_an_input_file_id_and_items_is_refused() -> None:
    # They are alternatives, and sending both would mean silently ignoring one.
    transport = ScriptedTransport([_json(JOB)])
    item = BatchRequestItem(
        "a", Prism.text().using("openai", "gpt-4o").with_prompt("x").to_request()
    )

    with pytest.raises(PrismError, match="not both"):
        _batch(transport).create([item], "file-x")


def test_neither_is_refused() -> None:
    transport = ScriptedTransport([_json(JOB)])

    with pytest.raises(PrismError, match="either"):
        _batch(transport).create()


# -- retrieve, list, cancel ------------------------------------------------


def test_the_in_flight_count_openai_does_not_report_is_derived() -> None:
    transport = ScriptedTransport([_json(JOB)])

    job = _batch(transport).retrieve("batch_1")

    assert job.request_counts.total == 10
    assert job.request_counts.succeeded == 4
    assert job.request_counts.failed == 1
    assert job.request_counts.processing == 5


def test_the_in_flight_count_is_never_negative() -> None:
    # A partial response -- total missing while completed is not -- would
    # produce one, and a negative number of in-flight requests is not a state
    # anything can be in.
    job = parse_batch_job({"id": "b", "status": "completed", "request_counts": {"completed": 5}})

    assert job.request_counts.processing == 0


def test_an_unrecognised_status_is_refused() -> None:
    # Quietly calling it in-progress would leave a caller waiting forever on a
    # batch that had already stopped.
    with pytest.raises(PrismError, match="reticulating"):
        parse_batch_job({"id": "b", "status": "reticulating"})


def test_listing_sends_only_the_parameters_that_were_set() -> None:
    transport = ScriptedTransport([_json({"data": [JOB], "has_more": True, "last_id": "batch_1"})])

    result = _batch(transport).list(5)

    assert "limit=5" in transport.calls[0].url
    assert "after" not in transport.calls[0].url
    assert result.data[0].status is BatchStatus.IN_PROGRESS
    assert result.has_more is True
    assert result.last_id == "batch_1"


def test_cancelling_posts_to_the_cancel_path() -> None:
    transport = ScriptedTransport([_json({**JOB, "status": "cancelling"})])

    job = _batch(transport).cancel("batch_1")

    assert transport.calls[0].method == "POST"
    assert transport.calls[0].url.endswith("/batches/batch_1/cancel")
    assert job.status is BatchStatus.CANCELLING


# -- results ---------------------------------------------------------------


def test_both_the_output_and_the_error_file_are_read() -> None:
    # A batch where some requests succeeded and others failed writes to both,
    # and reading only the output file reports a clean run.
    transport = ScriptedTransport(
        [
            _json(
                {
                    **JOB,
                    "status": "completed",
                    "output_file_id": "file-out",
                    "error_file_id": "file-err",
                }
            ),
            json.dumps(
                {
                    "custom_id": "row-1",
                    "response": {
                        "body": {
                            "id": "resp_1",
                            "model": "gpt-4o",
                            "output": [
                                {
                                    "type": "message",
                                    "content": [{"type": "output_text", "text": "hello"}],
                                }
                            ],
                        }
                    },
                }
            ).encode("utf-8"),
            json.dumps(
                {"custom_id": "row-2", "error": {"code": "invalid_request", "message": "bad"}}
            ).encode("utf-8"),
        ]
    )

    results = _batch(transport).get_results("batch_1")

    assert len(transport.calls) == 3
    assert transport.calls[1].url.endswith("/files/file-out/content")
    assert transport.calls[2].url.endswith("/files/file-err/content")
    assert len(results) == 2
    assert results[0].status is BatchResultStatus.SUCCEEDED
    assert results[0].text == "hello"
    assert results[1].status is BatchResultStatus.ERRORED
    assert results[1].error_message == "bad"


def test_a_batch_that_has_written_no_file_yet_returns_nothing() -> None:
    # Nothing went wrong; there is just nothing yet.
    transport = ScriptedTransport([_json(JOB)])

    assert _batch(transport).get_results("batch_1") == ()


def test_an_expired_item_is_expired_not_errored() -> None:
    # The request was never run, so reporting it as a failure blames the request
    # for something the queue did.
    items = parse_batch_results(
        json.dumps({"custom_id": "row-1", "error": {"code": "batch_expired", "message": "late"}})
    )

    assert items[0].status is BatchResultStatus.EXPIRED


def test_a_line_that_will_not_parse_is_skipped_rather_than_losing_the_file() -> None:
    body = "\n".join(
        [
            '{"custom_id":"a","response":{"body":{}}}',
            "not json at all",
            "",
            '{"custom_id":"b","response":{"body":{}}}',
        ]
    )

    assert [item.custom_id for item in parse_batch_results(body)] == ["a", "b"]


def test_the_last_message_is_taken_from_a_responses_output_not_the_first() -> None:
    # A turn that used tools has the message last; taking the first returns
    # whatever preceded the tool call.
    items = parse_batch_results(
        json.dumps(
            {
                "custom_id": "a",
                "response": {
                    "body": {
                        "output": [
                            {
                                "type": "message",
                                "content": [{"type": "output_text", "text": "thinking"}],
                            },
                            {
                                "type": "message",
                                "content": [{"type": "output_text", "text": "the answer"}],
                            },
                        ]
                    }
                },
            }
        )
    )

    assert items[0].text == "the answer"


def test_the_chat_completions_shape_is_read_too() -> None:
    items = parse_batch_results(
        json.dumps(
            {"custom_id": "a", "response": {"body": {"choices": [{"message": {"content": "hi"}}]}}}
        )
    )

    assert items[0].text == "hi"


def test_cached_tokens_go_in_the_cache_read_slot() -> None:
    # They are billed differently from cache writes, and the wrong slot
    # misreports the cost in the direction that flatters it.
    items = parse_batch_results(
        json.dumps(
            {
                "custom_id": "a",
                "response": {
                    "body": {
                        "usage": {
                            "input_tokens": 100,
                            "output_tokens": 20,
                            "input_tokens_details": {"cached_tokens": 80},
                            "output_tokens_details": {"reasoning_tokens": 5},
                        }
                    }
                },
            }
        )
    )

    usage = items[0].usage
    assert usage is not None
    assert usage.cache_read_input_tokens == 80
    assert usage.cache_write_input_tokens is None
    assert usage.thought_tokens == 5
