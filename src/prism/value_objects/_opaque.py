"""Carrying values from outside this slice without pretending to model them.

A few serialised fields belong to capabilities this port does not implement —
tool approvals, tool-call artifacts, provider rate limits. Their keys are part
of the stored shape and must still appear, so the fields exist and round-trip,
but their contents are carried opaquely: anything that can serialise itself is
asked to, and anything else passes through untouched.
"""

from __future__ import annotations

from typing import Any

__all__ = ["opaque_list"]


def opaque_list(items: list[Any]) -> list[Any]:
    serialised: list[Any] = []

    for item in items:
        to_dict = getattr(item, "to_dict", None)
        serialised.append(to_dict() if callable(to_dict) else item)

    return serialised
