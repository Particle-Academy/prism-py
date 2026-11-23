"""Core does not execute tools, so it has no approval check to fail open."""

from __future__ import annotations

from typing import Any

import pytest

from prism import HttpRequest, HttpResponse, Prism, PrismError, Tool, canonical

TOOL_CALL: dict[str, Any] = {
    "id": "cmpl-1",
    "object": "chat.completion",
    "model": "mistral-large-latest",
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
    "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12},
}


class Transport:
    def send(self, request: HttpRequest) -> HttpResponse:
        return HttpResponse(status=200, body=canonical.encode(TOOL_CALL).encode("utf-8"))


def test_has_no_approval_check_or_handler_that_could_fail_open() -> None:
    # If one is added, its failure behaviour needs its own test here.
    for name in ("requires_approval", "needs_approval", "using", "handle"):
        assert not hasattr(Tool, name)


def test_refuses_a_model_tool_call_rather_than_executing_it() -> None:
    tool = Tool().as_("delete_file").for_("Delete a file").with_string_parameter("path", "Path")

    with pytest.raises(PrismError, match="Executing tools is not part of this slice"):
        (
            Prism.text()
            .using(
                "mistral",
                "mistral-large-latest",
                {"api_key": "sk-test", "transport": Transport()},
            )
            .with_prompt("Delete a.txt.")
            .with_tools([tool])
            .as_text()
        )
