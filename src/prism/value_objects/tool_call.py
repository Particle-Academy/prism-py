"""A tool call the model asked for."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from prism.errors import PrismError

__all__ = ["ToolCall"]

_INTEGER_KEY = re.compile(r"0|-?[1-9][0-9]*", re.ASCII)

_CONTROL_ESCAPES = {
    "\x08": "\\b",
    "\x09": "\\t",
    "\x0a": "\\n",
    "\x0c": "\\f",
    "\x0d": "\\r",
}


@dataclass
class ToolCall:
    """A single call, as the model emitted it.

    ``arguments`` is a union on purpose. Providers stream it as a JSON string
    and callers build it as a mapping, and the stored form keeps whichever it
    was given: normalising on the way in loses the provider's exact bytes, and
    normalising on the way out changes what gets re-sent.
    """

    id: str
    name: str
    arguments: str | dict[str, Any]
    result_id: str | None = None
    reasoning_id: str | None = None
    reasoning_summary: list[Any] | None = field(default=None)

    def decoded_arguments(self) -> dict[str, Any]:
        """The arguments as a mapping, decoding the string form if needed.

        :raises PrismError: with code ``malformed_tool_call_arguments`` when the
            string form cannot be decoded. Failing here, at mapping time, is the
            point: the alternative is a malformed function call reaching the
            provider.
        """
        if not isinstance(self.arguments, str):
            return self.arguments

        # PHP treats "0" as falsy and the reference short-circuits on it; both
        # it and the empty string decode to no arguments rather than failing.
        if self.arguments in ("", "0"):
            return {}

        decoded = self._decode(self.arguments)

        return decoded if isinstance(decoded, dict) else {}

    def has_malformed_arguments(self) -> bool:
        """Whether the arguments are unusable: not JSON, or not named values.

        A JSON ``null``, an empty list and an empty string all mean none were
        given. Integer-like keys count as unusable because the reference binds
        them to a handler by position, and all three languages classify a call
        the same way.
        """
        if self.arguments in ("", "0"):
            return False

        decoded: Any = self.arguments

        if isinstance(self.arguments, str):
            try:
                decoded = self._decode(self.arguments)
            except PrismError:
                return True

        if decoded is None or decoded == []:
            return False

        return not isinstance(decoded, dict) or any(_is_integer_key(key) for key in decoded)

    def without_malformed_arguments(self) -> ToolCall:
        """This call with unusable arguments replaced by none."""
        if not self.has_malformed_arguments():
            return self

        return replace(self, arguments={})

    def _decode(self, text: str) -> Any:
        try:
            return json.loads(text)
        except ValueError:
            # Some providers (DeepSeek when streaming, notably) emit raw control
            # characters inside string values, which RFC 8259 requires to be
            # escaped. Escape them in place — stripping them would corrupt
            # intentional newlines and tabs — and decode again.
            try:
                return json.loads(_escape_control_characters_in_strings(text))
            except ValueError as exc:
                raise PrismError.malformed_tool_call_arguments(self.name) from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
            "result_id": self.result_id,
            "reasoning_id": self.reasoning_id,
            "reasoning_summary": self.reasoning_summary,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ToolCall:
        return cls(
            id=data["id"],
            name=data["name"],
            arguments=data["arguments"],
            result_id=data.get("result_id"),
            reasoning_id=data.get("reasoning_id"),
            reasoning_summary=data.get("reasoning_summary"),
        )


def _is_integer_key(key: object) -> bool:
    """A key PHP would store as an integer: canonical decimal, within 64 bits."""
    if not isinstance(key, str) or _INTEGER_KEY.fullmatch(key) is None:
        return False

    return -(2**63) <= int(key) <= 2**63 - 1


def _escape_control_characters_in_strings(raw: str) -> str:
    """Escape raw control characters inside JSON string literals.

    Control characters outside a string can never be valid and are dropped,
    except for tab, newline and carriage return, which are legal whitespace
    between tokens and are kept.
    """
    result: list[str] = []
    in_string = False
    escaped = False

    for char in raw:
        if ord(char) <= 0x1F:
            if in_string:
                result.append(_CONTROL_ESCAPES.get(char, f"\\u{ord(char):04x}"))
            elif char in ("\t", "\n", "\r"):
                result.append(char)
            escaped = False
            continue

        result.append(char)

        if escaped:
            escaped = False
        elif in_string and char == "\\":
            escaped = True
        elif char == '"':
            in_string = not in_string

    return "".join(result)
