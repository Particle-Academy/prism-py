"""Ordered replay state; provider reasoning values remain opaque."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def replayable_items(output: Sequence[Any]) -> list[dict[str, Any]]:
    items = []
    for item in output:
        if not isinstance(item, Mapping) or item.get("type") not in (
            "message",
            "reasoning",
            "function_call",
        ):
            continue
        normalized = dict(item)
        if item.get("type") == "reasoning" and not isinstance(item.get("summary"), (list, dict)):
            normalized["summary"] = []
        items.append(normalized)
    return items


def response_replay_items(output: Sequence[Any]) -> list[dict[str, Any]]:
    kept = []
    followed_by_kept_item = False
    for item in reversed(replayable_items(output)):
        if item.get("type") == "function_call":
            followed_by_kept_item = False
            continue
        if item.get("type") == "reasoning" and not followed_by_kept_item:
            continue
        followed_by_kept_item = True
        kept.append(item)
    return list(reversed(kept))
