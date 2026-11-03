"""Independent native failure and compatibility controls; no provider network calls."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from prism import Prism, PrismError, StreamFailureUsage, Usage
from prism.http import HttpRequest, HttpStreamResponse
from prism.streaming.events import ErrorEvent

LEGACY = json.loads((Path(__file__).parent / "fixtures/stream-failure-legacy.json").read_text())
PROVIDERS = ("anthropic", "openai", "mistral")


def native(
    provider: str, reported: bool, failure: Exception | None = None, malformed: bool = False
) -> tuple[Any, dict[str, bool]]:
    state = {"closed": False}
    usage = (
        (
            {"prompt_tokens": 11, "completion_tokens": 0}
            if provider == "mistral"
            else {"input_tokens": 11, "output_tokens": 0}
        )
        if reported
        else None
    )
    payload: dict[str, Any]
    if provider == "anthropic":
        payload = {
            "type": "message_start",
            "message": {"id": "owned", "model": "owned", "usage": usage},
        }
    elif provider == "openai":
        payload = {"type": "response.created", "response": {"model": "owned", "usage": usage}}
    else:
        payload = {
            "id": "owned",
            "model": "owned",
            "usage": usage,
            "choices": [{"delta": {"content": "owned"}}],
        }

    class Transport:
        def stream(self, request: HttpRequest) -> HttpStreamResponse:
            def chunks() -> Iterator[str]:
                try:
                    yield "data: " + json.dumps(payload) + "\n\n"
                    if malformed:
                        yield "data: {broken\n\n"
                    if failure is not None:
                        raise failure
                finally:
                    state["closed"] = True

            return HttpStreamResponse(200, {}, chunks())

    stream = (
        Prism.text()
        .using(provider, "owned", {"api_key": "", "stream_transport": Transport()})
        .with_prompt("owned")
        .as_stream()
    )
    return stream, state


def test_exact_published_legacy_event_and_exception_contract() -> None:
    event = ErrorEvent(
        code="owned_code", message="owned-message", id="owned-event", timestamp=0, recoverable=False
    )
    cause = RuntimeError("owned-cause")
    error = PrismError("provider_response_error", "owned-message", status=429, body="owned-body")
    error.__cause__ = cause
    assert event.to_dict() == LEGACY["error_event_object"]
    assert json.dumps(event.to_dict()) == LEGACY["error_event_json"]
    assert {
        "class_name": type(error).__name__,
        "code": error.code,
        "message": str(error),
        "status": error.status,
        "body": error.body,
        "args": list(error.args),
        "representation": repr(error),
    } == LEGACY["sdk_error_fields"]
    assert (error.__cause__ is cause) == LEGACY["cause_preserved"]
    assert error.stream_failure_usage() is None
    assert event.failure_usage is None
    report = StreamFailureUsage(
        Usage(0, 0, cost=0), 0, ["prompt_tokens", "completion_tokens", "cost"]
    )
    assert error.with_stream_failure_usage(report) is error
    assert error.to_dict()["failure_usage"]["usage"]["cost"] == 0
    error.with_stream_failure_usage(None)
    assert "failure_usage" not in error.to_dict()


def test_failure_snapshot_has_no_mutable_aliases_and_keeps_zero() -> None:
    usage = Usage(0, 0, cost=0)
    mask = ["prompt_tokens", "completion_tokens", "cost"]
    report = StreamFailureUsage(usage, 2, mask)
    usage.prompt_tokens = 99
    mask.append("thought_tokens")
    report.usage.prompt_tokens = 99
    serialized = report.to_dict()
    serialized["usage"]["prompt_tokens"] = 99
    serialized["reported_fields"].clear()
    assert report.usage.prompt_tokens == 0
    assert report.usage.cost == 0
    assert report.reported_fields == ("prompt_tokens", "completion_tokens", "cost")
    with pytest.raises(FrozenInstanceError):
        report.__setattr__("step_index", 99)
    assert report.to_dict()["step_index"] == 2
    assert report.to_dict()["includes_completed_steps"] is False


@pytest.mark.parametrize(
    "mask", [[], ["prompt_tokens", "prompt_tokens"], ["not_usage"], "prompt_tokens"]
)
def test_invalid_report_mask(mask: Any) -> None:
    with pytest.raises(ValueError):
        StreamFailureUsage(Usage(0, 0), 0, mask)


@pytest.mark.parametrize("step", [-1, 0.5, True])
def test_invalid_step(step: Any) -> None:
    with pytest.raises(ValueError):
        StreamFailureUsage(Usage(0, 0), step, ["prompt_tokens"])


def test_usage_subclass_cannot_extend_the_closed_report_mask() -> None:
    class ExtendedUsage(Usage):
        def to_dict(self) -> dict[str, Any]:
            return {**super().to_dict(), "private_field": 1}

    with pytest.raises(ValueError):
        StreamFailureUsage(ExtendedUsage(0, 0), 0, ["private_field"])


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("reported", [False, True])
def test_foreign_read_identity(provider: str, reported: bool) -> None:
    failure = OSError("owned-foreign")
    stream, _ = native(provider, reported, failure)
    events = []
    with pytest.raises(OSError) as caught:
        events.extend(stream)
    assert caught.value is failure
    errors = [event for event in events if isinstance(event, ErrorEvent)]
    assert len(errors) == (1 if reported else 0)
    if reported:
        report = errors[0].failure_usage
        assert report is not None
        assert report.usage.prompt_tokens == 11


@pytest.mark.parametrize("provider", PROVIDERS)
def test_sdk_read_error_keeps_identity_status_and_cause(provider: str) -> None:
    cause = RuntimeError("owned-cause")
    failure = PrismError("provider_response_error", "owned-sdk", status=503)
    failure.__cause__ = cause
    stream, _ = native(provider, True, failure)
    events = []
    with pytest.raises(PrismError) as caught:
        events.extend(stream)
    assert caught.value is failure
    assert failure.__cause__ is cause
    assert failure.status == 503
    report = failure.stream_failure_usage()
    assert report is not None
    assert report.usage.prompt_tokens == 11
    assert not any(isinstance(event, ErrorEvent) for event in events)


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("reported", [False, True])
def test_malformed_json_observable_once(provider: str, reported: bool) -> None:
    stream, _ = native(provider, reported, malformed=True)
    events = list(stream)
    errors = [event for event in events if isinstance(event, ErrorEvent)]
    assert len(errors) == 1
    assert errors[0].code == "stream_decode_error"
    assert (errors[0].failure_usage is not None) == reported
    ends = [event for event in events if event.type().value == "stream_end"]
    assert len(ends) == (1 if provider == "anthropic" else 0)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_consumer_throw_is_not_a_provider_failure(provider: str) -> None:
    stream, state = native(provider, True)
    next(stream)
    user = PrismError("provider_response_error", "owned-user")
    with pytest.raises(PrismError) as caught:
        stream.throw(user)
    assert caught.value is user
    assert user.stream_failure_usage() is None
    assert state["closed"]


@pytest.mark.parametrize("provider", PROVIDERS)
def test_consumer_close_releases_native_iterator(provider: str) -> None:
    stream, state = native(provider, True)
    next(stream)
    stream.close()
    assert state["closed"]
