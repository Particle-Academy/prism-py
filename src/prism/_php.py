"""The handful of PHP semantics the request contract actually depends on.

The reference filters some payload keys on *nullity* and others on *falsiness*,
and PHP's notion of falsy is not Python's — most notably the string ``"0"``,
which PHP treats as false and Python treats as true. Those filters are
observable in the goldens, so the difference is reproduced here rather than
smoothed over. Isolated in one private module so it is obvious where the
foreign semantics live, and so nothing else has to think about them.
"""

from __future__ import annotations

import math
from typing import Any

__all__ = ["data_get", "is_truthy", "strval", "where_not_null"]


def is_truthy(value: Any) -> bool:
    """PHP's truthiness.

    Identical to Python's except for ``"0"``, which PHP considers falsy. Both
    agree that ``None``, ``False``, ``0``, ``0.0``, ``""``, ``[]`` and ``{}``
    are falsy.
    """
    if isinstance(value, str):
        return value not in ("", "0")
    return bool(value)


def strval(value: Any) -> str:
    """PHP's ``strval``.

    Used where the reference stringifies a non-array tool result. Differs from
    :func:`str` on ``None`` (``""``), booleans (``"1"`` / ``""``) and integral
    floats (``"1"``, not ``"1.0"``).
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else ""
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return str(int(value))
    return str(value)


def data_get(target: Any, path: str, default: Any = None) -> Any:
    """A narrow stand-in for Laravel's ``data_get``.

    Supports dotted paths, numeric segments into lists, and the ``{last}``
    segment the OpenAI handler uses to reach the final output item. Returns
    ``default`` when any segment is missing or the value found is ``None``,
    which is what the reference's ``data_get($data, $path, $default)`` does.
    """
    current = target

    for segment in path.split("."):
        if current is None:
            return default

        if segment == "{last}":
            if isinstance(current, (list, tuple)) and current:
                current = current[-1]
                continue
            return default

        if isinstance(current, dict):
            if segment not in current:
                return default
            current = current[segment]
            continue

        if isinstance(current, (list, tuple)):
            try:
                index = int(segment)
            except ValueError:
                return default
            if index >= len(current) or index < -len(current):
                return default
            current = current[index]
            continue

        return default

    return default if current is None else current


def where_not_null(values: dict[str, Any]) -> dict[str, Any]:
    """Laravel's ``Arr::whereNotNull``: drop null values, keep ``False`` and ``0``."""
    return {key: value for key, value in values.items() if value is not None}
