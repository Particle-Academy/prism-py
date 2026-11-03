"""Owned offline byte/fault adapter; all events and exceptions are native SDK output."""

from __future__ import annotations

import base64
import codecs
import copy
import json
import math
from typing import Any

import prism
from prism import Prism
from prism.errors import PrismError
from prism.http import HttpStreamResponse
from prism.providers.anthropic import Anthropic
from prism.providers.mistral import Mistral
from prism.providers.openai import OpenAI
from prism.streaming.events import ErrorEvent
from prism.value_objects.usage import Usage


def parse_failure_stream(case: dict[str, Any]) -> dict[str, Any]:
    spec = case["failure_stream"]
    expected_builder = [
        {"call": "using", "args": [spec.get("provider"), "owned"]},
        {"call": "withPrompt", "args": ["private-prompt-sentinel"]},
        {"call": "withMaxTokens", "args": [32]},
        {"call": "withMaxSteps", "args": [2 if spec.get("mode") == "continuation" else 1]},
    ]
    if case["builder"] != expected_builder:
        raise ValueError("Only the fixed owned failure builder is executable.")
    if spec.get("provider") not in {"anthropic", "openai", "mistral"}:
        raise ValueError("Unsupported failure fixture provider.")
    if spec["mode"] == "serialization":
        reported = spec["serialization"] == "reported-zero-cost"
        if spec["serialization"] not in {"legacy-error", "reported-zero-cost"}:
            raise ValueError("Unknown native serialization control.")
        failure = (
            prism.StreamFailureUsage(
                Usage(0, 0, cost=0), 0, ["prompt_tokens", "completion_tokens", "cost"]
            )
            if reported
            else None
        )
        event = ErrorEvent(
            code="owned_failure",
            message="owned-fixture",
            id="owned-error",
            timestamp=0,
            recoverable=False,
            failure_usage=failure,
        )
        exception = PrismError("provider_response_error", "private-exception-sentinel")
        exception.with_stream_failure_usage(failure)
        return {
            "event": event.to_dict(),
            "event_json": json.loads(json.dumps(event.to_dict())),
            "exception_json": json.loads(json.dumps(exception.to_dict())),
            "requests": 0,
        }
    if spec["mode"] not in {"single", "reuse", "interleave"}:
        raise ValueError("Native continuation/telemetry has no declared port implementation.")
    if not 1 <= len(spec["attempts"]) <= 3:
        raise ValueError("Invalid native attempt count.")
    states: list[dict[str, Any]] = [
        {
            "events": [],
            "requests": 0,
            "owned_tool_calls": 0,
            "injected": None,
            "terminal": {
                "status": "returned",
                "exception": None,
                "injected_exception_preserved": None,
                "failure_usage": None,
                "serialized_error": None,
            },
        }
        for _ in spec["attempts"]
    ]
    active = 0

    class Transport:
        def stream(self, request: Any) -> HttpStreamResponse:
            if not request.url.startswith("https://conformance.invalid/v1/"):
                raise RuntimeError("Unexpected owned native request.")
            if json.loads(request.body).get("stream") is not True:
                raise RuntimeError("Owned request must be streamed.")
            state = states[active]
            if state["requests"] >= len(spec["attempts"][active]["responses"]):
                raise RuntimeError("Unexpected extra native request.")
            response = spec["attempts"][active]["responses"][state["requests"]]
            state["requests"] += 1
            chunks = _decode(response)
            if response["fault"] == "transport":
                state["injected"] = OSError("private-transport-sentinel")

            def body() -> Any:
                decoder = codecs.getincrementaldecoder("utf-8")()
                for chunk in chunks:
                    text = decoder.decode(chunk)
                    if text:
                        yield text
                rest = decoder.decode(b"", final=True)
                if rest:
                    yield rest
                if response["fault"] == "transport":
                    raise state["injected"]

            return HttpStreamResponse(200, {"content-type": "text/event-stream"}, body())

    kind = {"anthropic": Anthropic, "openai": OpenAI, "mistral": Mistral}[spec["provider"]]
    transport = Transport()
    config: dict[str, Any] = {
        "api_key": "owned-fixture-only",
        "url": "https://conformance.invalid/v1",
        "stream_transport": transport,
    }
    reused = kind(**config) if spec["mode"] == "reuse" else None
    event_ids: dict[str, str] = {}
    opaque_ids: dict[str, str] = {}

    def normalize(payload: dict[str, Any]) -> dict[str, Any]:
        payload = copy.deepcopy(payload)
        if "id" in payload:
            payload["id"] = _alias(event_ids, payload["id"], "event")
        if "timestamp" in payload:
            payload["timestamp"] = 0
        normalized: dict[str, Any] = _opaque(payload, opaque_ids)
        return normalized

    def walk(native: Any, state: dict[str, Any]) -> Any:
        try:
            for event in native:
                row = {"type": event.type().value, "payload": normalize(event.to_dict())}
                if isinstance(event, ErrorEvent):
                    row["serialized_error"] = normalize(json.loads(json.dumps(event.to_dict())))
                state["events"].append(row)
                yield None
        except Exception as error:
            sdk = isinstance(error, PrismError)
            failure = (
                error.stream_failure_usage()
                if sdk and hasattr(error, "stream_failure_usage")
                else None
            )
            state["terminal"] = {
                "status": "raised",
                "exception": type(error).__name__,
                "injected_exception_preserved": None
                if state["injected"] is None
                else error is state["injected"],
                "failure_usage": failure.to_dict() if failure is not None else None,
                "serialized_error": error.to_dict() if sdk and hasattr(error, "to_dict") else None,
            }

    iterators = []
    names = {
        "using": "using",
        "withPrompt": "with_prompt",
        "withMaxTokens": "with_max_tokens",
        "withMaxSteps": "with_max_steps",
    }
    for index, _attempt in enumerate(spec["attempts"]):
        pending = Prism.text()
        for step in case["builder"]:
            args = step.get("args", [])
            pending = (
                pending.using(args[0], args[1], provider_config=config)
                if step["call"] == "using"
                else getattr(pending, names[step["call"]])(*args)
            )
        provider = reused if reused is not None else kind(**config)

        # Some Python providers open the transport before returning their generator.
        # Defer that native call until the correct interleaved attempt is active.
        def invoke(provider: Any = provider, pending: Any = pending) -> Any:
            yield from provider.stream(pending.to_request())

        iterators.append(walk(invoke(), states[index]))
    if spec["mode"] == "interleave":
        done: set[int] = set()
        while len(done) < len(iterators):
            for index, iterator in enumerate(iterators):
                if index in done:
                    continue
                active = index
                try:
                    next(iterator)
                except StopIteration:
                    done.add(index)
    else:
        for index, iterator in enumerate(iterators):
            active = index
            for _ in iterator:
                pass
    return {
        "attempts": [
            {key: value for key, value in state.items() if key != "injected"} for state in states
        ],
        "requests": sum(state["requests"] for state in states),
    }


def _decode(response: dict[str, Any]) -> list[bytes]:
    chunks = response.get("chunks")
    if (
        response.get("encoding") != "base64"
        or response.get("fault") not in {"none", "transport"}
        or not isinstance(chunks, list)
        or len(chunks) > 100
        or (not chunks and response["fault"] != "transport")
    ):
        raise ValueError("Invalid native byte fixture.")
    result = []
    size = 0
    for encoded in chunks:
        if not isinstance(encoded, str) or len(encoded) > 1398104:
            raise ValueError("Oversized encoded native byte fixture.")
        chunk = base64.b64decode(encoded, validate=True)
        size += len(chunk)
        if not chunk or base64.b64encode(chunk).decode() != encoded or size > 1048576:
            raise ValueError("Invalid or oversized native byte fixture.")
        result.append(chunk)
    return result


def _alias(aliases: dict[str, str], value: str, prefix: str) -> str:
    aliases.setdefault(value, f"{prefix}-{len(aliases) + 1}")
    return aliases[value]


def _opaque(value: Any, aliases: dict[str, str]) -> Any:
    if isinstance(value, list):
        return [_opaque(child, aliases) for child in value]
    if not isinstance(value, dict):
        return value
    return {
        key: _alias(aliases, child, "opaque")
        if key in {"message_id", "reasoning_id"} and isinstance(child, str) and child
        else _opaque(child, aliases)
        for key, child in value.items()
    }


def failure_usage_violations(trace: dict[str, Any], properties: dict[str, Any]) -> list[str]:
    """Independent Python checks on native output; language-specific channels are declared."""
    failed = []

    def check(label: str, ok: bool) -> None:
        if not ok:
            failed.append(label)

    usage_keys = {
        "prompt_tokens",
        "completion_tokens",
        "cache_write_input_tokens",
        "cache_read_input_tokens",
        "thought_tokens",
        "cost",
        "cache_write_5m_input_tokens",
        "cache_write_1h_input_tokens",
    }

    def payload(actual: Any, expected: Any, label: str) -> None:
        check(
            label + ".shape",
            isinstance(actual, dict)
            and set(actual)
            == {
                "usage",
                "status",
                "scope",
                "step_index",
                "reported_fields",
                "includes_completed_steps",
            },
        )
        if not isinstance(actual, dict):
            return
        check(
            label + ".scope",
            actual.get("status") == "provisional"
            and actual.get("scope") == "attempt"
            and actual.get("includes_completed_steps") is False
            and type(actual.get("step_index")) is int
            and actual["step_index"] >= 0,
        )
        numbers = actual.get("usage", {})
        check(label + ".usage-shape", isinstance(numbers, dict) and set(numbers) == usage_keys)
        if not isinstance(numbers, dict):
            return
        for key in usage_keys:
            value = numbers.get(key)
            valid = value is None and key not in {"prompt_tokens", "completion_tokens"}
            valid = valid or (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                and 0 <= value <= 9007199254740991
                and (key == "cost" or type(value) is int)
            )
            check(label + ".numeric." + key, valid)
        fields = actual.get("reported_fields", [])
        check(
            label + ".mask",
            isinstance(fields, list)
            and bool(fields)
            and all(isinstance(field, str) and field in usage_keys for field in fields)
            and len(set(fields)) == len(fields),
        )
        check(
            label + ".reported-values",
            all(type(numbers.get(field)) in {int, float} for field in fields),
        )
        check(label + ".expected", actual == expected)
        check(
            label + ".privacy",
            all(sentinel not in json.dumps(actual) for sentinel in properties["privacy_sentinels"]),
        )

    def serialized(actual: Any, expected: Any, label: str) -> None:
        check(label + ".object", isinstance(actual, dict))
        if not isinstance(actual, dict):
            return
        check(label + ".no-private-field", "failureUsage" not in actual)
        check(label + ".presence", ("failure_usage" in actual) == (expected is not None))
        if expected is not None and "failure_usage" in actual:
            payload(actual["failure_usage"], expected, label + ".failure_usage")

    check("contract", properties.get("contract") == "failure-usage")
    check("requests", trace.get("requests") == properties["requests"])
    if "serialization" in properties:
        for field in ["event", "event_json", "exception_json"]:
            serialized(trace.get(field), properties["serialization"]["failure_usage"], field)
        check(
            "exception.privacy",
            "private-exception-sentinel" not in json.dumps(trace.get("exception_json")),
        )
        return failed
    check("attempt-count", len(trace.get("attempts", [])) == len(properties["attempts"]))
    for index, common in enumerate(properties["attempts"]):
        expected = {**common, **properties.get("native_outcomes", {}).get("py", {})}
        actual = trace.get("attempts", [])[index] if index < len(trace.get("attempts", [])) else {}
        label = f"attempt.{index}"
        terminal = actual.get("terminal", {})
        check(label + ".requests", actual.get("requests") == expected["requests"])
        check(label + ".terminal", terminal.get("status") == expected["terminal"])
        if expected["terminal"] == "raised":
            check(
                label + ".exception-class",
                isinstance(terminal.get("exception"), str) and bool(terminal["exception"]),
            )
        check(
            label + ".identity",
            terminal.get("injected_exception_preserved") is expected["transport_identity"],
        )
        check(
            label + ".owned-tools", actual.get("owned_tool_calls") == expected["owned_tool_calls"]
        )
        events = actual.get("events", [])
        errors = [event for event in events if event["type"] == "error"]
        ends = [event for event in events if event["type"] == "stream_end"]
        check(label + ".error-count", len(errors) == expected["error_events"])
        check(label + ".stream-ends", len(ends) == expected["stream_ends"])
        check(
            label + ".failure-only-metadata",
            all(
                event["type"] == "error" or "failure_usage" not in event["payload"]
                for event in events
            ),
        )
        channels = set()
        for offset, event in enumerate(errors):
            present = "failure_usage" in event["payload"]
            check(label + ".event-presence", present == (expected["failure_usage"] is not None))
            if present:
                channels.add("event")
                payload(
                    event["payload"]["failure_usage"],
                    expected["failure_usage"],
                    f"{label}.error.{offset}",
                )
            serialized(
                event.get("serialized_error"), expected["failure_usage"], label + ".event-json"
            )
        if terminal.get("failure_usage") is not None:
            channels.add("exception")
            payload(terminal["failure_usage"], expected["failure_usage"], label + ".exception")
        check(label + ".channels", channels == set(expected["channels"]))
        if "exception" in expected["channels"]:
            check(
                label + ".exception-json-present",
                isinstance(terminal.get("serialized_error"), dict),
            )
        if terminal.get("serialized_error") is not None:
            serialized(
                terminal["serialized_error"],
                expected["failure_usage"] if "exception" in expected["channels"] else None,
                label + ".exception-json",
            )
        stops = [
            event for event in ends if event["payload"].get("finish_reason") in {"Stop", "stop"}
        ]
        if expected["terminal"] == "raised" or expected["error_events"] > 0:
            check(label + ".no-success-end", not stops)
        if expected.get("normal_completion"):
            check(label + ".successful-baseline", len(ends) == 1 and len(stops) == 1)
        for field, value in expected.get("final_usage", {}).items():
            check(
                label + ".final-usage." + field,
                len(ends) == 1
                and isinstance(ends[0]["payload"].get("usage"), dict)
                and field in ends[0]["payload"]["usage"]
                and ends[0]["payload"]["usage"][field] == value,
            )
        if expected["owned_tool_calls"] == 0:
            check(
                label + ".no-completed-tool",
                not any(event["type"] in {"tool_call", "tool_result"} for event in events),
            )
    return failed
