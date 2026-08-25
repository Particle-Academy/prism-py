"""The outcome of running a tool."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from prism.value_objects._opaque import opaque_list

__all__ = ["ToolResult"]

ToolResultValue = int | float | str | dict[str, Any] | list[Any] | None


@dataclass
class ToolResult:
    """One tool's result.

    Two ids, not one: ``tool_call_id`` identifies the call and
    ``tool_call_result_id`` is the id the provider expects the result to be
    keyed by on the NEXT request. Losing the second breaks the follow-up call,
    not this one.
    """

    tool_call_id: str
    tool_name: str
    args: dict[str, Any]
    result: ToolResultValue
    tool_call_result_id: str | None = None
    artifacts: list[Any] = field(default_factory=list)

    def has_artifacts(self) -> bool:
        return self.artifacts != []

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "args": self.args,
            "result": self.result,
            "tool_call_result_id": self.tool_call_result_id,
            "artifacts": opaque_list(self.artifacts),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ToolResult:
        return cls(
            tool_call_id=data["tool_call_id"],
            tool_name=data["tool_name"],
            args=dict(data.get("args") or {}),
            result=data.get("result"),
            tool_call_result_id=data.get("tool_call_result_id"),
            artifacts=list(data.get("artifacts") or []),
        )
