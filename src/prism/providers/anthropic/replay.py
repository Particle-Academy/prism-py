"""Reference replay and fallback rules without interpreting signed content."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def thinking_blocks(blocks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fallback = max(
        (i for i, block in enumerate(blocks) if block.get("type") == "fallback"), default=-1
    )
    return [
        dict(block)
        for block in blocks[fallback + 1 :]
        if block.get("type") in ("thinking", "redacted_thinking")
    ]


def _result_ids(blocks: Sequence[Mapping[str, Any]]) -> list[Any]:
    return [
        block["tool_use_id"]
        for block in blocks
        if str(block.get("type", "")).endswith("_tool_result")
        and block.get("tool_use_id") is not None
    ]


def response_content_blocks(
    blocks: Sequence[Mapping[str, Any]], keep_server_tool_use: bool = False
) -> list[dict[str, Any]]:
    answered = _result_ids(blocks)
    return [
        dict(block)
        for block in blocks
        if block.get("type") != "tool_use"
        and (
            keep_server_tool_use
            or block.get("type") not in ("server_tool_use", "mcp_tool_use")
            or block.get("id") in answered
        )
    ]


def replay_content_blocks(blocks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fallback = max(
        (i for i, block in enumerate(blocks) if block.get("type") == "fallback"), default=-1
    )
    answered = _result_ids(blocks)
    replayed = []
    for index, block in enumerate(blocks):
        if index < fallback and (
            block.get("type") in ("thinking", "redacted_thinking", "connector_text", "tool_use")
            or (block.get("type") == "server_tool_use" and block.get("id") not in answered)
        ):
            continue
        normalized = dict(block)
        if normalized.get("input") == []:
            normalized["input"] = {}
        replayed.append(normalized)
    return replayed
