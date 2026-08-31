"""Structured output: a text generation that must come back shaped."""

from __future__ import annotations

from prism.structured.extract import extract_structured
from prism.structured.pending_request import StructuredPendingRequest
from prism.structured.request import StructuredRequest
from prism.structured.response import StructuredResponse

__all__ = [
    "StructuredPendingRequest",
    "StructuredRequest",
    "StructuredResponse",
    "extract_structured",
]
