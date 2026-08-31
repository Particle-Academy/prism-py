"""A frozen fill-in-the-middle request."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["FimRequest"]


@dataclass
class FimRequest:
    """A prefix, an optional suffix, and a gap.

    No messages and no tools, unlike every other generation request. FIM is a
    COMPLETION, not a conversation -- the model is given code either side of a
    hole and writes what goes in it, which is why an editor is the natural
    caller and a chat transcript is not.
    """

    model: str = ""
    provider_key: str = ""
    #: The text BEFORE the gap.
    prompt: str = ""
    #: The text AFTER the gap, or None to complete to the end.
    suffix: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: tuple[str, ...] = ()
    client_options: dict[str, Any] = field(default_factory=dict)
    provider_options: dict[str, Any] = field(default_factory=dict)
