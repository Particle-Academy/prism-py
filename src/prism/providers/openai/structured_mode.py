"""Which structured method an OpenAI model can actually honour."""

from __future__ import annotations

from prism.enums import StructuredMode
from prism.errors import ErrorCode, PrismError

__all__ = ["resolve_structured_mode"]

# CAPABILITY INFERRED FROM THE MODEL NAME, which is a compromise worth naming
# rather than hiding: OpenAI publishes no endpoint that answers "does this model
# support strict schemas", so the reference matches prefixes and so does this.
# The cost is that a model released tomorrow is treated as JSON-only until this
# list learns about it -- a conservative failure, but a failure.
#
# prism-provider-watch treats changes to this list as actionable drift.
_STRUCTURED_PREFIXES = ("gpt-4o", "gpt-4.1", "gpt-4.5", "gpt-5", "chatgpt-4o", "o3-mini")

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
