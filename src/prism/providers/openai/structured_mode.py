"""Which structured method an OpenAI model can actually honour."""

from __future__ import annotations

import re

from prism.enums import StructuredMode
from prism.errors import ErrorCode, PrismError

__all__ = ["resolve_structured_mode"]

# CAPABILITY INFERRED FROM THE MODEL NAME, which is a compromise worth naming
# rather than hiding: OpenAI publishes no endpoint that answers "does this model
# support strict schemas", so the reference infers it and so does this.
#
# A list alone goes stale in the unsafe direction: the newest models fall back
# to JSON mode, where the schema is not enforced, and nothing says so. gpt-6.x
# and o1/o3/o4 did exactly that until prism#83. So families are also recognised
# by OpenAI's naming, gpt-N for N >= 5 and the oN reasoning series. A name only
# has to LOOK like a family to be caught, so custom models served through this
# provider under other names keep the JSON-mode default.
_STRUCTURED_PREFIXES = ("gpt-4o", "gpt-4.1", "gpt-4.5", "gpt-5", "chatgpt-4o", "o3-mini")
_GPT_FAMILY = re.compile(r"^gpt-(\d+)(?:[.\-]|$)")
_O_SERIES = re.compile(r"^o\d+(?:-|$)")

_UNSUPPORTED = (
    "o1-mini",
    "o1-mini-2024-09-12",
    "o1-preview",
    "o1-preview-2024-09-12",
)


def resolve_structured_mode(model: str) -> StructuredMode:
    base = _base_model(model)

    if base in _UNSUPPORTED:
        raise PrismError(
            ErrorCode.UNSUPPORTED_STRUCTURED_MODEL,
            f"Structured output is not supported for {model}",
        )

    if any(base.startswith(prefix) for prefix in _STRUCTURED_PREFIXES):
        return StructuredMode.STRUCTURED

    family = _GPT_FAMILY.match(base)
    if family is not None:
        return StructuredMode.STRUCTURED if int(family.group(1)) >= 5 else StructuredMode.JSON

    # o1-mini and o1-preview never supported it; _UNSUPPORTED refused them above.
    if _O_SERIES.match(base) is not None:
        return StructuredMode.STRUCTURED

    return StructuredMode.JSON


def _base_model(model: str) -> str:
    """A fine-tune is named ``ft:<base>:<org>:<name>:<hash>``.

    Its capability is the BASE model's. Matching prefixes against the whole
    string would classify every fine-tune as JSON-only, including gpt-4o ones.
    """
    if not model.startswith("ft:"):
        return model

    parts = model.split(":", 2)

    return parts[1] if len(parts) > 1 else model
