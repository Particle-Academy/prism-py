"""A tool offered to the model."""

from __future__ import annotations

from typing import Any

from prism._php import data_get
from prism.errors import ErrorCode, PrismError
from prism.schema import BooleanSchema, NumberSchema, Schema, StringSchema

__all__ = ["Tool"]


class Tool:
    """A function the model may call.

    Declaration order matters and is preserved: the order tools are declared in
    reaches the model and influences which one it picks.

    Constructible either way — pass ``name`` and ``description`` up front, or
    build fluently. ``as_`` and ``for_`` carry trailing underscores because
    ``as`` and ``for`` are Python keywords; the reference spells them ``as()``
    and ``for()``.

    >>> Tool("weather", "Get the weather").with_string_parameter("city", "The city")
    """

    def __init__(self, name: str = "", description: str = "") -> None:
        self.name = name
        self.description = description
        self._parameters: dict[str, Schema] = {}
        self._required_parameters: list[str] = []
        self._provider_options: dict[str, Any] = {}

    # -- fluent configuration ----------------------------------------------

    def as_(self, name: str) -> Tool:
        self.name = name
        return self

    def for_(self, description: str) -> Tool:
        self.description = description
        return self

    def with_parameter(self, parameter: Schema, required: bool = True) -> Tool:
        self._parameters[parameter.name] = parameter

        if required:
            self._required_parameters.append(parameter.name)

        return self

    def with_string_parameter(self, name: str, description: str, required: bool = True) -> Tool:
        return self.with_parameter(StringSchema(name, description), required)

    def with_number_parameter(self, name: str, description: str, required: bool = True) -> Tool:
        return self.with_parameter(NumberSchema(name, description), required)

    def with_boolean_parameter(self, name: str, description: str, required: bool = True) -> Tool:
        return self.with_parameter(BooleanSchema(name, description), required)

    def with_provider_options(self, options: dict[str, Any] | None = None) -> Tool:
        self._provider_options = dict(options or {})
        return self

    # -- reading -----------------------------------------------------------

    @property
    def parameters(self) -> dict[str, Schema]:
        return self._parameters

    @property
    def required_parameters(self) -> list[str]:
        return self._required_parameters

    def has_parameters(self) -> bool:
        return bool(self._parameters)

    def parameters_as_dict(self) -> dict[str, Any]:
        return {name: schema.to_dict() for name, schema in self._parameters.items()}

    def parameters_as_schema(self, require_all_properties: bool = False) -> dict[str, Any]:
        """Close the generated root without rewriting caller-owned nested schemas."""
        strict = bool(self.provider_option("strict"))
        if strict and require_all_properties:
            optional = [name for name in self._parameters if name not in self._required_parameters]
            if optional:
                raise PrismError(
                    ErrorCode.INVALID_TOOL_SCHEMA,
                    f"Strict tool [{self.name}] has optional parameters [{', '.join(optional)}]. "
                    "This provider requires every property to be required. "
                    "Make them required (nullable if needed) or turn strict off.",
                )

        return {
            "type": "object",
            "properties": self.parameters_as_dict(),
            "required": list(self._required_parameters),
            **({"additionalProperties": False} if strict else {}),
        }

    def provider_option(self, path: str | None = None, default: Any = None) -> Any:
        if path is None:
            return self._provider_options

        return data_get(self._provider_options, path, default)
