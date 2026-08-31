"""JSON Schema fragments for tool parameters.

Key ORDER inside a fragment is decided here, not by the tool mapper, and it is
observable in the goldens: ``description`` comes before ``type``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

__all__ = [
    "ArraySchema",
    "BooleanSchema",
    "EnumSchema",
    "NumberSchema",
    "ObjectSchema",
    "Schema",
    "StringSchema",
]


class Schema(ABC):
    """A named parameter and the JSON Schema fragment describing it."""

    name: str
    description: str
    nullable: bool

    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...

    def _type(self, type_name: str) -> str | list[str]:
        return [type_name, "null"] if self.nullable else type_name


@dataclass(frozen=True)
class StringSchema(Schema):
    """A string parameter."""

    name: str
    description: str
    nullable: bool = False
    pattern: str | None = None
    format: str | None = None

    def to_dict(self) -> dict[str, Any]:
        schema: dict[str, Any] = {
            "description": self.description,
            "type": self._type("string"),
        }

        if self.pattern is not None:
            schema["pattern"] = self.pattern
        if self.format is not None:
            schema["format"] = self.format

        return schema


@dataclass(frozen=True)
class NumberSchema(Schema):
    """A numeric parameter."""

    name: str
    description: str
    nullable: bool = False
    multiple_of: float | None = None
    maximum: float | None = None
    exclusive_maximum: float | None = None
    minimum: float | None = None
    exclusive_minimum: float | None = None

    def to_dict(self) -> dict[str, Any]:
        schema: dict[str, Any] = {
            "description": self.description,
            "type": self._type("number"),
        }

        # The wire names are JSON Schema's camelCase, not the field names.
        for key, value in (
            ("multipleOf", self.multiple_of),
            ("maximum", self.maximum),
            ("exclusiveMaximum", self.exclusive_maximum),
            ("minimum", self.minimum),
            ("exclusiveMinimum", self.exclusive_minimum),
        ):
            if value is not None:
                schema[key] = value

        return schema


@dataclass(frozen=True)
class BooleanSchema(Schema):
    """A boolean parameter."""

    name: str
    description: str
    nullable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "type": self._type("boolean"),
        }


@dataclass(frozen=True)
class ObjectSchema(Schema):
    """The schema a structured response is shaped by.

    ``allow_additional_properties`` defaults to ``False``, matching the
    reference, and the default matters more than it looks: OpenAI's strict
    schema mode REJECTS a schema that permits extra properties, so a permissive
    default would make the strongest structured mode unusable and silently push
    every request down to plain JSON mode.
    """

    name: str
    description: str
    properties: tuple[Schema, ...] = ()
    required_fields: tuple[str, ...] = ()
    allow_additional_properties: bool = False
    nullable: bool = False

    def to_dict(self) -> dict[str, Any]:
        schema: dict[str, Any] = {
            "description": self.description,
            "type": self._type("object"),
        }

        # Dropped when empty rather than sent as {}, matching the reference's
        # not-null filter. `required` and `additionalProperties` are always
        # sent: [] and False are meaningful answers, not absences.
        if self.properties:
            schema["properties"] = {p.name: p.to_dict() for p in self.properties}

        schema["required"] = list(self.required_fields)
        schema["additionalProperties"] = self.allow_additional_properties

        return schema


@dataclass(frozen=True)
class ArraySchema(Schema):
    """A list of one item shape."""

    name: str
    description: str
    items: Schema = None  # type: ignore[assignment]
    min_items: int | None = None
    max_items: int | None = None
    nullable: bool = False

    def to_dict(self) -> dict[str, Any]:
        schema: dict[str, Any] = {
            "description": self.description,
            "type": self._type("array"),
            "items": self.items.to_dict(),
        }

        # Present only when set. The reference appends these conditionally, and
        # a `minItems: null` is rejected by strict mode.
        if self.min_items is not None:
            schema["minItems"] = self.min_items
        if self.max_items is not None:
            schema["maxItems"] = self.max_items

        return schema


@dataclass(frozen=True)
class EnumSchema(Schema):
    """A closed set of allowed values.

    When nullable, ``None`` joins the OPTIONS as well as the type union: a
    nullable enum whose options omit null describes a value that can be null and
    may not be, which no validator can satisfy.
    """

    name: str
    description: str
    options: tuple[str | int | float | None, ...] = ()
    nullable: bool = False

    def to_dict(self) -> dict[str, Any]:
        options = [*self.options, None] if self.nullable else list(self.options)

        return {
            "description": self.description,
            "enum": options,
            "type": self._type("string"),
        }
