"""A reply that finished on tool calls is refused on every provider.

G-352 in the PHP reference: non-streaming handlers dropped the pending approval
requests from their response, so a resume denied every approval. That gap needs
a tool loop that stops for an approval. This port has none, and its tools carry
no handler: every provider's text and structured path refuses a reply that
finished on tool calls, before any second request. These cases check that for
every provider here, so the claim cannot go stale when a provider is added or
one of them starts running tools.
"""

from __future__ import annotations

from typing import Any

import pytest

from prism import HttpRequest, HttpResponse, Prism, PrismError, Tool, canonical
from prism.schema import ObjectSchema, StringSchema

TOOL_CALL_BODIES: dict[str, dict[str, Any]] = {
    "anthropic": {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "test-model",
        "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "delete_file", "input": {"path": "a.txt"}}
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    },
    "mistral": {
        "id": "cmpl-1",
        "object": "chat.completion",
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "delete_file", "arguments": '{"path":"a.txt"}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    },
    "openai": {
        "id": "resp_1",
        "object": "response",
        "status": "completed",
        "model": "test-model",
        "output": [
            {
                "id": "fc_1",
                "type": "function_call",
                "status": "completed",
                "call_id": "call_1",
                "name": "delete_file",
                "arguments": '{"path":"a.txt"}',
            }
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    },
}

SCHEMA = ObjectSchema("answer", "The answer", (StringSchema("answer", "The answer"),), ("answer",))


class CountingTransport:
    def __init__(self, body: dict[str, Any]) -> None:
        self.body = body
        self.calls = 0

    def send(self, request: HttpRequest) -> HttpResponse:
        self.calls += 1
        return HttpResponse(status=200, body=canonical.encode(self.body).encode("utf-8"))


def _tool() -> Tool:
    return Tool().as_("delete_file").for_("Delete a file").with_string_parameter("path", "Path")


@pytest.mark.parametrize("provider", sorted(TOOL_CALL_BODIES))
def test_text_path_refuses_a_tool_call_reply(provider: str) -> None:
    transport = CountingTransport(TOOL_CALL_BODIES[provider])

    with pytest.raises(PrismError) as raised:
        (
            Prism.text()
            .using(provider, "test-model", {"api_key": "sk-test", "transport": transport})
            .with_tools([_tool()])
            .with_max_steps(3)
            .with_prompt("Delete a.txt.")
            .as_text()
        )

    assert raised.value.code == "tool_loop_not_supported"
    assert transport.calls == 1


@pytest.mark.parametrize("provider", sorted(TOOL_CALL_BODIES))
def test_structured_path_refuses_a_tool_call_reply(provider: str) -> None:
    transport = CountingTransport(TOOL_CALL_BODIES[provider])
    request = Prism.structured()
    request.with_schema(SCHEMA)
    request.using(provider, "test-model", {"api_key": "sk-test", "transport": transport})
    request.with_tools([_tool()]).with_max_steps(3).with_prompt("Delete a.txt.")

    with pytest.raises(PrismError) as raised:
        request.as_structured()

    assert raised.value.code == "tool_loop_not_supported"
    assert transport.calls == 1
