"""Opaque replay survives the actual stored parsed assistant message."""

import json
from typing import Any

import pytest

from prism import AssistantMessage, Prism
from prism.providers.anthropic import parse_text_response as parse_anthropic
from prism.providers.anthropic.request_body import build_request_body as build_anthropic
from prism.providers.openai import build_request_body, parse_text_response


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_stored_parsed_assistant_replays_opaque_state(provider: str) -> None:
    request = Prism.text().using(provider, "owned-fixture").with_prompt("fixture").to_request()
    opaque = "synthetic-owned-opaque-λ+/="
    blocks: list[dict[str, Any]] = (
        [
            {"type": "reasoning", "id": "rs_owned", "summary": [], "encrypted_content": opaque},
            {
                "type": "message",
                "id": "msg_owned",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "fixture"}],
            },
        ]
        if provider == "openai"
        else [
            {"type": "thinking", "thinking": "fixture", "signature": opaque},
            {"type": "text", "text": "fixture"},
            {"type": "redacted_thinking", "data": opaque},
        ]
    )
    raw: dict[str, Any] = {
        "id": "owned",
        "model": "owned-fixture",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    if provider == "openai":
        raw.update(status="completed", output=blocks)
        response = parse_text_response(request, raw)
    else:
        raw.update(stop_reason="end_turn", content=blocks)
        response = parse_anthropic(request, raw)
    assistant = next(
        message for message in response.messages if isinstance(message, AssistantMessage)
    )
    stored = AssistantMessage.from_dict(json.loads(json.dumps(assistant.to_dict())))
    following = Prism.text().using(provider, "owned-fixture").with_messages([stored]).to_request()
    body = build_request_body(following) if provider == "openai" else build_anthropic(following)
    replay = body["input"] if provider == "openai" else body["messages"][0]["content"]
    # Keep opaque values out of failure diagnostics.
    assert bool(replay == blocks)
