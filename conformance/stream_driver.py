"""Native published urllib transport + SSE parser, without provider network I/O."""

from __future__ import annotations

import base64
from typing import Any
from unittest.mock import patch

from prism import Prism
from prism.providers.anthropic import Anthropic


class ChunkResponse:
    status = 200

    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = list(chunks)
        self.headers = {"Content-Type": "text/event-stream"}

    def read(self, size: int) -> bytes:
        if not self.chunks:
            return b""
        chunk = self.chunks.pop(0)
        if len(chunk) > size:
            self.chunks.insert(0, chunk[size:])
            return chunk[:size]
        return chunk

    def close(self) -> None:
        pass


def parse_stream(case: dict[str, Any]) -> dict[str, Any]:
    stream = case["stream"]
    if stream.get("encoding") != "base64" or not stream.get("chunks"):
        raise ValueError("A stream fixture needs nonempty base64 chunks.")
    chunks = [base64.b64decode(encoded, validate=True) for encoded in stream["chunks"]]
    if any(not chunk for chunk in chunks):
        raise ValueError("Empty stream chunk.")
    requests = 0

    def open_owned(request: Any, **_options: Any) -> ChunkResponse:
        nonlocal requests
        requests += 1
        if request.full_url != "https://api.anthropic.com/v1/messages":
            raise RuntimeError("Unexpected request in owned stream fixture.")
        return ChunkResponse(chunks)

    pending = Prism.text()
    names = {"using": "using", "withPrompt": "with_prompt", "withMaxSteps": "with_max_steps"}
    for step in case["builder"]:
        pending = getattr(pending, names[step["call"]])(*step.get("args", []))
    events = []
    event_ids: dict[str, str] = {}
    opaque_ids: dict[str, str] = {}
    terminal = {"status": "returned"}
    with patch("urllib.request.urlopen", side_effect=open_owned):
        try:
            for event in Anthropic(api_key="owned-g115", url="https://api.anthropic.com/v1").stream(
                pending.to_request()
            ):
                payload = event.to_dict()
                if "id" in payload:
                    event_ids.setdefault(payload["id"], f"event-{len(event_ids) + 1}")
                    payload["id"] = event_ids[payload["id"]]
                if "timestamp" in payload:
                    payload["timestamp"] = 0
                events.append(
                    {"type": event.type().value, "payload": alias_opaque(payload, opaque_ids)}
                )
        except Exception as error:
            terminal = {"status": "raised", "exception": type(error).__name__}
    return {"events": events, "terminal": terminal, "requests": requests}


def alias_opaque(value: Any, aliases: dict[str, str]) -> Any:
    if isinstance(value, list):
        return [alias_opaque(child, aliases) for child in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, child in value.items():
        if key in {"message_id", "reasoning_id"} and isinstance(child, str) and child:
            aliases.setdefault(child, f"opaque-{len(aliases) + 1}")
            result[key] = aliases[child]
        else:
            result[key] = alias_opaque(child, aliases)
    return result


def violations(trace: dict[str, Any], properties: dict[str, Any]) -> list[str]:
    failed = []
    events = trace["events"]
    ends = [event for event in events if event["type"] == "stream_end"]
    text = "".join(event["payload"]["delta"] for event in events if event["type"] == "text_delta")
    if text != properties["text"]:
        failed.append("text")
    starts = [event for event in events if event["type"] == "text_start"]
    completes = [event for event in events if event["type"] == "text_complete"]
    if (
        len(starts) != len(completes)
        or "".join(event["payload"]["text"] for event in completes) != properties["text"]
    ):
        failed.append("text_completion")
    if (
        len(ends) != properties["terminal_events"]
        or not events
        or events[-1]["type"] != "stream_end"
    ):
        failed.append("terminal_events")
    if trace["terminal"]["status"] != "returned" or trace["requests"] != 1:
        failed.append("native_execution")
    errors = [event for event in events if event["type"] == "error"]
    if bool(errors) != properties["error_observable"]:
        failed.append("error_observable")
    if properties.get("error_nonrecoverable") and any(
        event["payload"].get("recoverable") is not False for event in errors
    ):
        failed.append("error_nonrecoverable")
    if (
        "error_codes" in properties
        and [event["payload"].get("error_type", event["payload"].get("code")) for event in errors]
        != properties["error_codes"]
    ):
        failed.append("error_codes")
    if properties["no_completed_tools"] and any(
        event["type"] in {"tool_call", "tool_result"} for event in events
    ):
        failed.append("no_completed_tools")
    if len(ends) == 1:
        payload = ends[0]["payload"]
        accepted = {"Stop", "stop"} if properties["normal_completion"] else {"Error", "error"}
        if payload.get("finish_reason") not in accepted:
            failed.append("normal_completion")
        if (
            "usage_known" in properties
            and (payload.get("usage") is not None) != properties["usage_known"]
        ):
            failed.append("usage_known")
        for field, expected in properties.get("final_usage", {}).items():
            if (payload.get("usage") or {}).get(field) != expected:
                failed.append(f"final_usage.{field}")
    return failed
