"""What the model wrote into the gap."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from prism.enums import FinishReason
from prism.value_objects.meta import Meta
from prism.value_objects.usage import Usage

__all__ = ["FimResponse"]


@dataclass(frozen=True)
class FimResponse:
    """Flat, with no steps and no messages, unlike :class:`~prism.text.response.Response`.

    A FIM call is one round trip that cannot call a tool, so there is nothing to
    accumulate and modelling it with a step list would advertise a loop that
    does not exist.
    """

    text: str = ""
    finish_reason: FinishReason = FinishReason.UNKNOWN
    usage: Usage = field(default_factory=lambda: Usage(prompt_tokens=0, completion_tokens=0))
    meta: Meta = field(default_factory=lambda: Meta(id="", model=""))
    raw: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "finish_reason": self.finish_reason.value,
            "usage": self.usage.to_dict(),
            "meta": self.meta.to_dict(),
        }
