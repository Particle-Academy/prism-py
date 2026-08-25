"""Discrimination probes: deliberately wrong implementations of this port.

A conformance table every plausible implementation passes proves nothing. Each
mutant here injects one named defect, and the corpus declares the EXACT set of
case ids it must fail — which is what turns the table into a measurement rather
than decoration.

Two rules shape the design:

* **Nothing here lives in the library.** ``src/prism`` has no probe switch. Each
  mutant is installed from outside by rebinding a module global at a real
  decision point — Python resolves globals at call time, so rebinding
  ``request_body.map_tool_choice`` genuinely changes which mapper
  ``build_request_body`` consults — or by post-processing a value the driver
  already holds. Every install returns its own undo and the driver restores it.

* **A probe is installed only for the suite kinds its declared scope names.**
  ``probes.json`` states a ``scope`` per probe, and the scope is normative: it is
  what makes ``omit-null-keys`` a request-body defect rather than a defect in
  every serializer this port owns. Without that it would fail rows in suites its
  declaration never mentions, and the expectation could not be exact.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from prism.enums import ToolChoice
from prism.providers.openai import request_body as request_body_module
from prism.providers.openai import response as response_module
from prism.value_objects import Text, UserMessage

__all__ = ["FAITHFUL", "Mutation", "for_probe", "installed", "known_ids", "scope_to_kinds"]

FAITHFUL = "faithful"

# A probe's declared scope, mapped onto the suite kinds it is installed for.
# `error-code` shares the request-body path, so a request-body defect is
# installed there too; none of today's mutants changes an error code, and that
# is a measured result rather than an assumption.
_SCOPE_KINDS: dict[str, frozenset[str]] = {
    "request body": frozenset({"request-payload", "error-code"}),
    "response parsing": frozenset({"response-parse"}),
    "value object serialization": frozenset({"roundtrip"}),
    "value object rehydration": frozenset({"roundtrip"}),
}


def scope_to_kinds(scope: str) -> frozenset[str]:
    """The suite kinds a probe's declared scope covers."""
    if scope not in _SCOPE_KINDS:
        raise KeyError(f"probes.json declares an unrecognised scope {scope!r}.")

    return _SCOPE_KINDS[scope]


def _no_op() -> Callable[[], None]:
    return lambda: None


def _identity(value: dict[str, Any]) -> dict[str, Any]:
    return value


@dataclass(frozen=True)
class Mutation:
    """One injected defect.

    ``install`` rebinds library globals and hands back its own undo.
    ``request_body``, ``serialize`` and ``rehydrate`` are hooks the driver
    consults around values it already holds, for defects that live after the
    library has finished rather than inside it.
    """

    id: str
    install: Callable[[], Callable[[], None]] = _no_op
    request_body: Callable[[dict[str, Any]], dict[str, Any]] = _identity
    parsed: Callable[[dict[str, Any]], dict[str, Any]] = _identity
    serialize: Callable[[dict[str, Any]], dict[str, Any]] = _identity
    rehydrate: Callable[[type[Any], Mapping[str, Any]], Any] | None = None
    kinds: frozenset[str] = field(default_factory=frozenset)


def _rebind(module: Any, name: str, replacement: Any) -> Callable[[], None]:
    """Swap one module global and hand back the undo."""
    original = getattr(module, name)
    setattr(module, name, replacement)

    def undo() -> None:
        setattr(module, name, original)

    return undo


def _installer(module: Any, name: str, replacement: Any) -> Callable[[], Callable[[], None]]:
    return lambda: _rebind(module, name, replacement)


def _wrapping_installer(
    module: Any,
    name: str,
    wrap: Callable[[Any], Any],
) -> Callable[[], Callable[[], None]]:
    """An install that replaces ``module.name`` with ``wrap(original)``."""
    return lambda: _rebind(module, name, wrap(getattr(module, name)))


# ---------------------------------------------------------------------------
# the mutants
# ---------------------------------------------------------------------------


def _without_nulls(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _without_nulls(item) for key, item in value.items() if item is not None}

    if isinstance(value, list):
        return [_without_nulls(item) for item in value]

    return value


def _omit_nulls(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop null-valued keys, all the way down."""
    return {key: _without_nulls(item) for key, item in payload.items() if item is not None}


def _falsy_filter(values: dict[str, Any]) -> dict[str, Any]:
    """The optional-key filter, on falsiness instead of nullity."""
    return {key: value for key, value in values.items() if value}


class _TruthyWhenEmpty(list[Any]):
    """A list that is truthy even when empty.

    This IS the hazard, stated as a type: an empty array is falsy in PHP and
    truthy in JavaScript and Python, so ``build_tools(request) or None`` stops
    collapsing and the ``tools`` key survives — in the position the faithful
    body would have used it, which is what keeps the mutant honest.
    """

    def __bool__(self) -> bool:
        return True


def _keep_empty_tools(original: Callable[[Any], list[Any]]) -> Callable[[Any], list[Any]]:
    def build_tools(request: Any) -> list[Any]:
        return _TruthyWhenEmpty(original(request))

    return build_tools


def _provider_tools_last(
    original: Callable[[Any], list[dict[str, Any]]],
) -> Callable[[Any], list[dict[str, Any]]]:
    """The merge, the other way round — as a rotation of the real output.

    Provider tools map one-to-one onto the leading items, so moving the first
    ``len(provider_tools)`` entries to the back reverses the merge without
    rebuilding either list.
    """

    def build_tools(request: Any) -> list[dict[str, Any]]:
        built = original(request)
        provider_count = len(request.provider_tools)

        return [*built[provider_count:], *built[:provider_count]]

    return build_tools


def _tool_choice_any_verbatim(original: Callable[[Any], Any]) -> Callable[[Any], Any]:
    def map_tool_choice(tool_choice: Any) -> Any:
        mapped = original(tool_choice)

        return "any" if tool_choice is ToolChoice.ANY else mapped

    return map_tool_choice


def _tool_arguments_as_object(
    original: Callable[..., list[dict[str, Any]]],
) -> Callable[..., list[dict[str, Any]]]:
    def map_messages(
        messages: Sequence[Any],
        system_prompts: Sequence[Any] = (),
    ) -> list[dict[str, Any]]:
        mapped = original(messages, system_prompts)

        for item in mapped:
            if item.get("type") == "function_call" and isinstance(item.get("arguments"), str):
                item["arguments"] = json.loads(item["arguments"])

        return mapped

    return map_messages


def _system_prompts_last(
    original: Callable[..., list[dict[str, Any]]],
) -> Callable[..., list[dict[str, Any]]]:
    """Append the system prompts instead of prepending them.

    Also a rotation of the real output: a system prompt maps to exactly one
    leading item, so the count is the offset.
    """

    def map_messages(
        messages: Sequence[Any],
        system_prompts: Sequence[Any] = (),
    ) -> list[dict[str, Any]]:
        mapped = original(messages, system_prompts)
        prompt_count = len(system_prompts)

        return [*mapped[prompt_count:], *mapped[:prompt_count]]

    return map_messages


def _prompt_tokens_unadjusted(
    original: Callable[[Mapping[str, Any]], Any],
) -> Callable[[Mapping[str, Any]], Any]:
    def build_usage(data: Mapping[str, Any]) -> Any:
        usage = original(data)
        usage.prompt_tokens += usage.cache_read_input_tokens or 0

        return usage

    return build_usage


def _rehydrate_reappends_text(subject_type: type[Any], data: Mapping[str, Any]) -> Any:
    """Hand the stored content parts straight back to the constructor.

    The constructor appends a text part built from its own content argument, so
    the stored list already holds one — passing it back appends a second, and
    the message text doubles on every save-and-load cycle. Nothing errors. A
    real consumer shipped exactly this.
    """
    if subject_type is not UserMessage:
        return subject_type.from_dict(data)

    return UserMessage(
        data["content"],
        [Text.from_dict(part) for part in data.get("additional_content") or []],
        dict(data.get("additional_attributes") or {}),
    )


_MUTATIONS: dict[str, Mutation] = {
    FAITHFUL: Mutation(id=FAITHFUL),
    "omit-null-keys": Mutation(id="omit-null-keys", request_body=_omit_nulls),
    "falsy-filter": Mutation(
        id="falsy-filter",
        install=_installer(request_body_module, "where_not_null", _falsy_filter),
    ),
    "keep-empty-tools": Mutation(
        id="keep-empty-tools",
        install=_wrapping_installer(request_body_module, "build_tools", _keep_empty_tools),
    ),
    "tool-choice-any-verbatim": Mutation(
        id="tool-choice-any-verbatim",
        install=_wrapping_installer(
            request_body_module, "map_tool_choice", _tool_choice_any_verbatim
        ),
    ),
    "tool-arguments-as-object": Mutation(
        id="tool-arguments-as-object",
        install=_wrapping_installer(request_body_module, "map_messages", _tool_arguments_as_object),
    ),
    "provider-tools-last": Mutation(
        id="provider-tools-last",
        install=_wrapping_installer(request_body_module, "build_tools", _provider_tools_last),
    ),
    "system-prompts-last": Mutation(
        id="system-prompts-last",
        install=_wrapping_installer(request_body_module, "map_messages", _system_prompts_last),
    ),
    "prompt-tokens-unadjusted": Mutation(
        id="prompt-tokens-unadjusted",
        install=_wrapping_installer(response_module, "_build_usage", _prompt_tokens_unadjusted),
    ),
    "omit-null-on-serialize": Mutation(id="omit-null-on-serialize", serialize=_omit_nulls),
    "omit-null-on-parse": Mutation(id="omit-null-on-parse", parsed=_omit_nulls),
    "rehydrate-reappends-text": Mutation(
        id="rehydrate-reappends-text", rehydrate=_rehydrate_reappends_text
    ),
}


def known_ids() -> frozenset[str]:
    return frozenset(_MUTATIONS)


def for_probe(probe: Mapping[str, Any]) -> Mutation:
    """The mutation implementing one probe declaration from the corpus.

    Raises rather than degrading if the corpus grows a probe this port has not
    implemented — a probe that silently does nothing is a probe that passes.
    """
    probe_id = str(probe["id"])
    mutation = _MUTATIONS.get(probe_id)

    if mutation is None:
        raise KeyError(f"No mutation implements probe {probe_id!r}.")

    if probe_id == FAITHFUL:
        return mutation

    return Mutation(
        id=mutation.id,
        install=mutation.install,
        request_body=mutation.request_body,
        parsed=mutation.parsed,
        serialize=mutation.serialize,
        rehydrate=mutation.rehydrate,
        kinds=scope_to_kinds(str(probe["scope"])),
    )


@contextmanager
def installed(mutation: Mutation, suite_kind: str) -> Iterator[Mutation]:
    """Install ``mutation`` for the duration, if this suite kind is in its scope.

    Yields the mutation the driver should consult: the real one inside its
    scope, and the no-op control outside it.
    """
    if mutation.id == FAITHFUL or suite_kind not in mutation.kinds:
        yield _MUTATIONS[FAITHFUL]
        return

    undo = mutation.install()
    try:
        yield mutation
    finally:
        undo()
