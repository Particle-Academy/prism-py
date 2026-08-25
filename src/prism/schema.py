"""JSON Schema fragments for tool parameters.

Key ORDER inside a fragment is decided here, not by the tool mapper, and it is
observable in the goldens: ``description`` comes before ``type``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

__all__ = ["BooleanSchema", "NumberSchema", "Schema", "StringSchema"]


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
