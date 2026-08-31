"""A text request that must come back shaped."""

from __future__ import annotations

from dataclasses import dataclass, field

from prism.enums import StructuredMode
from prism.schema import Schema
from prism.text.request import Request

__all__ = ["StructuredRequest"]


@dataclass
class StructuredRequest(Request):
    """A :class:`~prism.text.request.Request` plus the shape it must satisfy.

    SUBCLASSES the text request rather than restating it. The reference keeps
    two request classes carrying the same twenty fields, which is defensible in
    PHP where the duplication is visible and reviewed. Here it would be a second
    copy nobody diffs: a field added to one and forgotten in the other produces
    a request that builds fine and silently drops ``top_k`` on structured calls
    only. Everything a text request carries, a structured request carries -- by
    construction, not by agreement.
    """

    # Defaulted because every field on the base is defaulted; a structured
    # request without a schema is refused by the builder, not by the type.
    schema: Schema | None = None
    mode: StructuredMode = field(default=StructuredMode.AUTO)
