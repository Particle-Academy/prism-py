from __future__ import annotations

from typing import Any

from conformance.stream_driver import violations


def fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        {
            "events": [
                {"type": "text_start", "payload": {}},
                {"type": "text_delta", "payload": {"delta": "Hé"}},
                {"type": "text_complete", "payload": {"text": "Hé"}},
                {"type": "stream_end", "payload": {"finish_reason": "stop", "usage": None}},
            ],
            "terminal": {"status": "returned"},
            "requests": 1,
        },
        {
            "text": "Hé",
            "normal_completion": True,
            "error_observable": False,
            "terminal_events": 1,
            "no_completed_tools": True,
            "usage_known": False,
        },
    )


def test_completed_text_unknown_accounting_control() -> None:
    trace, properties = fixture()
    assert violations(trace, properties) == []


def test_retained_deltas_do_not_excuse_an_unclosed_text_block() -> None:
    trace, properties = fixture()
    trace["events"] = [event for event in trace["events"] if event["type"] != "text_complete"]
    assert "text_completion" in violations(trace, properties)


def test_failed_stream_requires_error_finish_not_merely_non_stop() -> None:
    trace, properties = fixture()
    trace["events"].insert(0, {"type": "error", "payload": {}})
    trace["events"][-1]["payload"]["finish_reason"] = "unknown"
    properties.update(normal_completion=False, error_observable=True)
    assert "normal_completion" in violations(trace, properties)
