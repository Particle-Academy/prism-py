"""Make an object out of what the model said, or report that you could not."""

from __future__ import annotations

import json
import re
from typing import Any

__all__ = ["extract_structured"]

_FENCE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_structured(text: str) -> dict[str, Any] | None:
    """Parse ``text`` into a dict, or return ``None``.

    RETURNS NONE, NEVER RAISES. A model asked for JSON can answer with a
    refusal, an apology, or a correct object wrapped in commentary, and none of
    those is an exception -- they are answers of a shape the caller did not
    want. Raising here would destroy ``text``, which is the only evidence of
    what happened and the only thing that explains the failure to whoever reads
    the log.

    Two attempts, in order: the whole string, then the contents of the first
    fenced block. Models fence JSON even when told not to, and the reference's
    own Anthropic prompt pleads against it -- a plea is not a guarantee.

    A JSON array or bare scalar parses and is still rejected: the schema
    describes an object, and returning ``[1, 2, 3]`` as "structured" would
    satisfy the annotation and break the first caller to read a key off it.
    """
    direct = _parse_object(text)

    if direct is not None:
        return direct

    fenced = _FENCE.search(text)

    return None if fenced is None else _parse_object(fenced.group(1))


def _parse_object(candidate: str) -> dict[str, Any] | None:
    stripped = candidate.strip()

    if not stripped:
        return None

    try:
        parsed = json.loads(stripped)
    except ValueError:
        return None

    return parsed if isinstance(parsed, dict) else None
