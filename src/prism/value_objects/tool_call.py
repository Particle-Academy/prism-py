"""A tool call the model asked for."""

from __future__ import annotations

import json
import math
import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, TypeVar

from prism import canonical
from prism.errors import ErrorCode, PrismError

__all__ = ["ToolCall"]

# The deepest nesting accepted. A provider wraps the arguments in its own request
# envelope, and the reference encodes that request with a depth limit of 512, so
# the limit leaves room for the deepest envelope.
_MAX_ARGUMENT_DEPTH = 500

# What a record holds in place of unusable arguments: text that is refused if the
# record is ever read back into a call.
_UNUSABLE_ARGUMENTS_RECORD = "]"

_T = TypeVar("_T")

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

        A JSON ``null`` and an empty list mean none were given. Anything else
        that is not a map of named values is refused rather than read as no
        arguments, so a tool is never run on an empty mapping the model did not
        send.

        :raises PrismError: with code ``malformed_tool_call_arguments`` when the
            arguments cannot be decoded or are not named values. Failing here,
            at mapping time, is the point: the alternative is a malformed
            function call reaching the provider.
        """
        # PHP treats "0" as falsy and the reference short-circuits on it; both
        # it and the empty string decode to no arguments rather than failing.
        if isinstance(self.arguments, str) and self.arguments in ("", "0"):
            return {}

        decoded: Any = (
            self._decode(self.arguments) if isinstance(self.arguments, str) else self.arguments
        )

        if decoded is None or decoded == []:
            return {}

        # Integer-like keys are refused because the reference binds them to a
        # handler by position, and all three languages classify a call the same
        # way.
        if not isinstance(decoded, dict) or any(_is_integer_key(key) for key in decoded):
            raise PrismError.tool_call_arguments_not_an_object(self.name)

        # Admitted only if a follow-up request can carry it, in any language.
        if not _can_be_sent_back(decoded):
            raise PrismError.malformed_tool_call_arguments(self.name)

        return decoded

    @staticmethod
    def storable_arguments(value: Any) -> str | dict[str, Any]:
        """A provider's value as this class stores it.

        ``None`` means none were given. Any other value that is neither text nor
        a map is held as a one-item list, the one form that is refused whatever
        the value was: ``0`` and ``False`` must not turn into "no arguments" on
        the way in.
        """
        if isinstance(value, (str, dict)):
            return value
        if value is None:
            return ""
        if value == [] and isinstance(value, list):
            return "[]"

        try:
            return json.dumps([value])
        except (TypeError, ValueError):
            return "[null]"

    @staticmethod
    def append_argument_fragment(text: str, fragment: Any) -> str:
        """The accumulated text once a streamed fragment is added.

        A fragment that is not text makes the whole value unreadable, wherever
        it arrived: a leading ``]`` can never become valid JSON, whatever is
        appended after it.
        """
        if fragment is None:
            return text

        return text + fragment if isinstance(fragment, str) else f"]{text}"

    def replayable_arguments_text(self) -> str:
        """The arguments as text for a provider field that carries them as written.

        The original text when it is valid JSON naming values, otherwise the
        same value encoded, and ``{}`` for no arguments or unusable ones.
        """
        if (
            isinstance(self.arguments, str)
            and not self.has_malformed_arguments()
            and self.decoded_arguments()
        ):
            try:
                json.loads(self.arguments)
            except ValueError:
                pass  # Valid only after repair, so it is sent encoded.
            else:
                return self.arguments

        return canonical.encode(self.without_malformed_arguments().decoded_arguments())

    def has_malformed_arguments(self) -> bool:
        """Whether :meth:`decoded_arguments` would refuse these arguments."""
        try:
            self.decoded_arguments()
        except PrismError as error:
            if error.code == ErrorCode.MALFORMED_TOOL_CALL_ARGUMENTS:
                return True
            raise

        return False

    def without_malformed_arguments(self) -> ToolCall:
        """This call with unusable arguments replaced by none."""
        if not self.has_malformed_arguments():
            return self

        return replace(self, arguments={})

    def _decode(self, text: str) -> Any:
        try:
            return _independent_of_caller_stack(lambda: _loads(text))
        except (ValueError, RuntimeError) as exc:
            # RuntimeError covers RecursionError, and a thread that could not start.
            raise PrismError.malformed_tool_call_arguments(self.name) from exc

    def to_dict(self) -> dict[str, Any]:
        """The stored record.

        Unusable arguments are never kept: the record holds a fixed text that is
        refused if it is ever read back into a call, and says so with
        ``arguments_malformed``, as the reference does.
        """
        malformed = self.has_malformed_arguments()

        return {
            "id": self.id,
            "name": self.name,
            "arguments": _UNUSABLE_ARGUMENTS_RECORD if malformed else self.arguments,
            **({"arguments_malformed": True} if malformed else {}),
            "result_id": self.result_id,
            "reasoning_id": self.reasoning_id,
            "reasoning_summary": self.reasoning_summary,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ToolCall:
        # A record marked unusable stays unusable, whatever arguments it carries.
        malformed = data.get("arguments_malformed") is True

        return cls(
            id=data["id"],
            name=data["name"],
            arguments=_UNUSABLE_ARGUMENTS_RECORD if malformed else data["arguments"],
            result_id=data.get("result_id"),
            reasoning_id=data.get("reasoning_id"),
            reasoning_summary=data.get("reasoning_summary"),
        )


def _loads(text: str) -> Any:
    try:
        return json.loads(text, parse_constant=_refuse_constant)
    except ValueError:
        # Some providers (DeepSeek when streaming, notably) emit raw control
        # characters inside string values, which RFC 8259 requires to be
        # escaped. Escape them in place — stripping them would corrupt
        # intentional newlines and tabs — and decode again.
        return json.loads(
            _escape_control_characters_in_strings(text), parse_constant=_refuse_constant
        )


def _independent_of_caller_stack(work: Callable[[], _T]) -> _T:
    """Run ``work`` so that its answer does not depend on the caller's stack depth.

    The decoder and the encoder recurse once per nesting level, and Python
    counts those levels against the same limit as the caller's own frames. A
    caller already deep in its stack would otherwise see valid arguments refused
    that a shallow caller sees accepted. When the first attempt runs out of
    room, the work is done again on a new thread, whose stack starts empty, so
    the answer is the one any caller gets. Input that is still too deep there is
    too deep for every caller.
    """
    try:
        return work()
    except RecursionError:
        pass

    outcome: list[tuple[bool, Any]] = []

    def run() -> None:
        # Any failure is handed back and raised on the caller's thread.
        try:
            outcome.append((True, work()))
        except BaseException as error:
            outcome.append((False, error))

    thread = threading.Thread(target=run, name="prism-tool-call-arguments", daemon=True)
    thread.start()
    thread.join()

    succeeded, value = outcome[0]
    if not succeeded:
        raise value

    return value  # type: ignore[no-any-return]


def _refuse_constant(name: str) -> Any:
    """``NaN`` and ``Infinity`` are not JSON; Python's decoder accepts them."""
    raise ValueError(f"{name} is not valid JSON")


def _has_lone_surrogate(text: str) -> bool:
    return not text.isascii() and any("\ud800" <= char <= "\udfff" for char in text)


def _is_finite_number(value: int | float) -> bool:
    """Whether the number fits a double, which is what the other languages hold."""
    if isinstance(value, float):
        return math.isfinite(value)

    try:
        float(value)
    except OverflowError:
        return False

    return True


def _can_be_sent_back(root: dict[str, Any]) -> bool:
    """Whether a decoded value survives the encode a follow-up request uses.

    In every language: no non-finite number, no key starting with NUL, no lone
    surrogate, and no deeper than the limit. Walked a level at a time, so deep
    input cannot overflow the call stack, and only nested values are held.
    """
    level: list[Any] = [root]
    depth = 1

    while level:
        following: list[Any] = []

        for value in level:
            if isinstance(value, dict):
                for key in value:
                    if (
                        not isinstance(key, str)
                        or key.startswith("\x00")
                        or _has_lone_surrogate(key)
                    ):
                        return False
                items = value.values()
            else:
                items = value

            for item in items:
                if isinstance(item, bool):
                    continue
                if isinstance(item, (int, float)):
                    if not _is_finite_number(item):
                        return False
                elif isinstance(item, str):
                    if _has_lone_surrogate(item):
                        return False
                elif isinstance(item, (list, dict)):
                    if depth >= _MAX_ARGUMENT_DEPTH:
                        return False
                    following.append(item)

        level = following
        depth += 1

    try:
        _independent_of_caller_stack(lambda: canonical.encode(root))
    except (TypeError, ValueError, RuntimeError, PrismError):
        # RuntimeError covers RecursionError, and a thread that could not start.
        return False

    return True


def _is_integer_key(key: object) -> bool:
    """A key PHP would store as an integer: canonical decimal, within 64 bits."""
    if isinstance(key, int):
        return True

    # Twenty characters is the longest 64-bit integer, with its sign.
    if not isinstance(key, str) or len(key) > 20 or _INTEGER_KEY.fullmatch(key) is None:
        return False

    return -(2**63) <= int(key) <= 2**63 - 1


def _escape_control_characters_in_strings(raw: str) -> str:
    """Escape raw control characters inside JSON string literals.

    Anything else is left as it is, so that the decode refuses it: a control
    character outside a string, or straight after a backslash, has no value it
    can be repaired to.
    """
    result: list[str] = []
    in_string = False
    escaped = False

    for char in raw:
        if ord(char) <= 0x1F:
            if in_string and not escaped:
                result.append(_CONTROL_ESCAPES.get(char, f"\\u{ord(char):04x}"))
            else:
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
