"""The fluent builder for structured output."""

from __future__ import annotations

from prism.enums import StructuredMode
from prism.errors import ErrorCode, PrismError
from prism.schema import Schema
from prism.structured.request import StructuredRequest
from prism.structured.response import StructuredResponse
from prism.text.pending_request import PendingRequest

__all__ = ["StructuredPendingRequest"]


class StructuredPendingRequest(PendingRequest):
    """Extends the text builder, so every request-shaping method is the same one.

    ``using``, ``with_messages``, ``using_temperature``, ``with_tools`` -- all of
    them keep their spelling rather than becoming a parallel set that drifts. A
    caller moving a call from :meth:`Prism.text` to :meth:`Prism.structured`
    changes two lines.
    """

    def __init__(self) -> None:
        super().__init__()
        self._schema: Schema | None = None
        self._mode: StructuredMode = StructuredMode.AUTO

    def with_schema(self, schema: Schema) -> StructuredPendingRequest:
        self._schema = schema

        return self

    def using_structured_mode(self, mode: StructuredMode) -> StructuredPendingRequest:
        self._mode = mode

        return self

    def to_request(self) -> StructuredRequest:
        """Freeze this builder into a :class:`StructuredRequest`.

        :raises PrismError: with code ``missing_schema`` when none was set. A
            structured request without a schema is a text request that has not
            said so, and defaulting to "any object" would return something that
            parses and means nothing.
        """
        if self._schema is None:
            raise PrismError(
                ErrorCode.MISSING_SCHEMA,
                "A structured request needs a schema. Call with_schema() before as_structured().",
            )

        base = super().to_request()

        # Rebuilt from the base request's own fields rather than from the
        # builder's, so the two can never disagree about what was collected.
        return StructuredRequest(
            **{key: getattr(base, key) for key in base.__dataclass_fields__},
            schema=self._schema,
            mode=self._mode,
        )

    def as_structured(self) -> StructuredResponse:
        """Send the request and return the parsed response."""
        if self._provider is None:
            raise PrismError(
                ErrorCode.UNSUPPORTED_PROVIDER_ACTION,
                "No provider configured. Call using(<provider>, <model>) first.",
            )

        # The base Provider contract types every capability as `Any`, because
        # most of them are unported. Narrowed here rather than there, so adding
        # a capability does not have to touch the contract every provider
        # implements.
        result = self._provider.structured(self.to_request())

        if not isinstance(result, StructuredResponse):  # pragma: no cover - defensive
            raise PrismError(
                ErrorCode.PROVIDER_RESPONSE_ERROR,
                "The provider returned something other than a structured response.",
            )

        return result
