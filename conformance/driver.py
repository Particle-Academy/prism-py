"""Driving this port from a corpus case.

The corpus names builder calls in the canonical (PHP) spelling and names value
objects by construct tag. Translating both into this port's idiom is the
runner's job, not the corpus's: the call SEQUENCE is the contract, the spelling
is not.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from conformance import mutations
from prism import canonical
from prism.enums import ToolChoice
from prism.errors import PrismError
from prism.providers.openai import build_request_body, parse_text_response
from prism.schema import BooleanSchema, NumberSchema, Schema, StringSchema
from prism.text import PendingRequest
from prism.tool import Tool
from prism.value_objects import (
    AssistantMessage,
    Meta,
    ProviderTool,
    SystemMessage,
    ToolCall,
    ToolResult,
    ToolResultMessage,
    Usage,
    UserMessage,
)

__all__ = ["Attempt", "DriverError", "build_pending", "hydrate", "run_case"]


class DriverError(Exception):
    """The corpus asked for something this runner cannot express."""


# Constructs whose non-``$`` keys are constructor arguments under their
# canonical names. Anything needing more than that is special-cased below.
_CONSTRUCTS: dict[str, type[Any]] = {
    "UserMessage": UserMessage,
    "AssistantMessage": AssistantMessage,
    "SystemMessage": SystemMessage,
    "ToolResultMessage": ToolResultMessage,
    "ToolCall": ToolCall,
    "ToolResult": ToolResult,
    "ProviderTool": ProviderTool,
    "Usage": Usage,
    "Meta": Meta,
}

_TOOL_CHOICES: dict[str, ToolChoice] = {
    "Auto": ToolChoice.AUTO,
    "Any": ToolChoice.ANY,
    "None": ToolChoice.NONE,
}

# Typed as the call the driver actually makes — every schema in the corpus is a
# name and a description — rather than as `type[Schema]`, whose base has no
# constructor to check the call against.
_SCHEMAS: dict[str, Callable[[str, str], Schema]] = {
    "string": StringSchema,
    "number": NumberSchema,
    "boolean": BooleanSchema,
}

Attempt = tuple[str, str]
"""One (expected, actual) pair. A case may make more than one — a roundtrip row
asserts the serialised bytes AND that rebuilding from them returns them."""


def _snake(name: str) -> str:
    """``withSystemPrompt`` becomes ``with_system_prompt``."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def hydrate(value: Any) -> Any:
    """Turn a corpus fixture into this port's objects, depth first."""
    if isinstance(value, list):
        return [hydrate(item) for item in value]

    if not isinstance(value, dict):
        return value

    tag = value.get("$")

    if tag is None:
        return {key: hydrate(item) for key, item in value.items()}

    if tag == "ToolChoice":
        case = value["case"]
        if case not in _TOOL_CHOICES:
            raise DriverError(f"Unknown ToolChoice case {case!r}.")
        return _TOOL_CHOICES[case]

    if tag == "Tool":
        return _tool(value)

    factory = _CONSTRUCTS.get(str(tag))

    if factory is None:
        raise DriverError(f"Unknown construct {tag!r}.")

    return factory(**{_snake(key): hydrate(item) for key, item in value.items() if key != "$"})


def _tool(spec: Mapping[str, Any]) -> Tool:
    # `as` and `for` are Python keywords, so the port spells them `as_`/`for_`
    # and the parameter list is a builder call rather than a constructor
    # argument. One of the two constructs the generic path cannot carry.
    tool = Tool().as_(spec["as"]).for_(spec["for"])

    for parameter in spec.get("parameters") or []:
        schema = _SCHEMAS.get(parameter["type"])

        if schema is None:
            raise DriverError(f"Unknown parameter type {parameter['type']!r}.")

        tool.with_parameter(
            schema(parameter["name"], parameter["description"]),
            parameter.get("required", True),
        )

    if "providerOptions" in spec:
        tool.with_provider_options(spec["providerOptions"])

    return tool


def build_pending(script: Sequence[Mapping[str, Any]]) -> PendingRequest:
    """Replay a builder script against a fresh pending request."""
    from prism import Prism

    pending = Prism.text()

    for step in script:
        name = _snake(str(step["call"]))
        method = getattr(pending, name, None)

        if method is None:
            raise DriverError(
                f"The builder script calls {step['call']!r}; this port has no {name!r}."
            )

        pending = method(*[hydrate(argument) for argument in step.get("args") or []])

    return pending


def run_case(kind: str, case: Mapping[str, Any], mutation: mutations.Mutation) -> list[Attempt]:
    """Execute one case and return every (expected, actual) pair it asserts."""
    if kind == "request-payload":
        return _run_request_payload(case, mutation)
    if kind == "response-parse":
        return _run_response_parse(case, mutation)
    if kind == "roundtrip":
        return _run_roundtrip(case, mutation)
    if kind == "error-code":
        return _run_error_code(case, mutation)
    if kind == "container-identity":
        return _run_container_identity(case)

    raise DriverError(f"Unknown suite kind {kind!r}.")


def _request_body(case: Mapping[str, Any], mutation: mutations.Mutation) -> dict[str, Any]:
    request = build_pending(case["builder"]).to_request()

    return mutation.request_body(build_request_body(request))


def _run_request_payload(case: Mapping[str, Any], mutation: mutations.Mutation) -> list[Attempt]:
    return [(case["expect"]["body_json"], canonical.encode(_request_body(case, mutation)))]


def _run_response_parse(case: Mapping[str, Any], mutation: mutations.Mutation) -> list[Attempt]:
    request = build_pending(case["builder"]).to_request()
    parsed = parse_text_response(request, case["response"])

    return [(case["expect"]["result_json"], canonical.encode(mutation.parsed(parsed.to_dict())))]


def _run_roundtrip(case: Mapping[str, Any], mutation: mutations.Mutation) -> list[Attempt]:
    expected: str = case["expect"]["serialized_json"]
    subject = hydrate(case["subject"])

    stored = mutation.serialize(subject.to_dict())
    attempts: list[Attempt] = [(expected, canonical.encode(stored))]

    if not case["expect"].get("rehydrates"):
        return attempts

    # Rebuild from what was just written, not from the golden: a port that can
    # write its value objects and cannot read them back is the gap that shipped
    # a defect downstream, and only this half catches it.
    rebuild = mutation.rehydrate or (lambda cls, data: cls.from_dict(data))
    rebuilt = rebuild(type(subject), stored)
    attempts.append((expected, canonical.encode(mutation.serialize(rebuilt.to_dict()))))

    return attempts


def _run_error_code(case: Mapping[str, Any], mutation: mutations.Mutation) -> list[Attempt]:
    # CODES only. The prose is worded idiomatically per language and pinning it
    # would hold every implementation to a translation.
    try:
        _request_body(case, mutation)
    except PrismError as error:
        actual = error.code
    else:
        actual = "no_error"

    return [(case["expect"]["error_code"], actual)]


def _run_container_identity(case: Mapping[str, Any]) -> list[Attempt]:
    # No decoded input at all: the row can only be answered by parsing the raw
    # text, which is what proves the raw-string channel is wired to something.
    left = json.loads(case["left_raw"])
    right = json.loads(case["right_raw"])

    return [
        (
            canonical.encode(case["expect"]["equal_after_parse"]),
            canonical.encode(left == right),
        )
    ]
