"""Resolving a provider key to a provider instance."""

from __future__ import annotations

import unicodedata
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from prism.errors import ErrorCode, PrismError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from prism.providers.base import Provider

__all__ = ["register_provider", "resolve_provider"]

ProviderFactory = Callable[..., "Provider"]

_FACTORIES: dict[str, ProviderFactory] = {}

_BUILT_INS = ("anthropic", "mistral", "openai")


def register_provider(key: str, factory: ProviderFactory, *, replace: bool = False) -> None:
    """Teach the builder about a provider key.

    A key that is already registered -- a built-in such as ``openai``, or one
    registered earlier, compared case-insensitively -- is refused unless
    ``replace=True`` is passed. Replacing a provider changes what every caller
    of that key gets, process-wide, so it has to be asked for.

    The key itself must be readable: see :func:`_is_valid_provider_name`. An
    invalid key is refused whether or not ``replace`` is set, and never trimmed.

    :raises ValueError: when the key is not a valid provider name, or is taken
        and ``replace`` is not set (a programming error; it carries no Prism
        error code).
    """
    if not _is_valid_provider_name(key):
        raise ValueError(
            "That is not a valid provider name: a provider name must be non-empty"
            " and contain no whitespace, control, format or invisible characters."
        )
    if not replace and _is_registered(key):
        raise ValueError(
            f"The provider '{key}' is already registered. Pass replace=True to replace it."
        )
    _FACTORIES[key] = factory


# Codepoints that are invisible in a name: the G-36 set, spelled by codepoint
# and identically to the human-plus confirmation guard and to the reference and
# prism-ts. It covers every Unicode White_Space codepoint plus U+0000,
# U+200B..U+200D and U+FEFF. str.isspace() / str.strip() disagree with the other
# two languages' built-ins, so none of them is used.
_INVISIBLE = frozenset(
    [
        0x0000,
        *range(0x0009, 0x000D + 1),
        0x0020,
        0x0085,
        0x00A0,
        0x1680,
        *range(0x2000, 0x200D + 1),
        0x2028,
        0x2029,
        0x202F,
        0x205F,
        0x3000,
        0xFEFF,
    ]
)

_REFUSED_CATEGORIES = frozenset({"Cc", "Cf", "Cs"})


def _is_valid_provider_name(key: str) -> bool:
    """Non-empty, with no Cc, Cf or Cs codepoint and none from the G-36 set.

    `` openai``, ``openai `` and ``openai`` + U+200B are distinct keys that read
    as the built-in in a config file or a log, so they are refused rather than
    trimmed into another name. Same rule as the reference's extend() and
    prism-ts registerProvider().
    """
    if not isinstance(key, str) or key == "":
        return False
    return not any(
        ord(char) in _INVISIBLE or unicodedata.category(char) in _REFUSED_CATEGORIES for char in key
    )


def _ascii_lower(value: str) -> str:
    """ASCII-only case folding, as the reference's strtolower() does."""
    return value.translate(_ASCII_LOWER)


_ASCII_LOWER = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")


def _is_registered(key: str) -> bool:
    name = _ascii_lower(key)
    return name in _BUILT_INS or any(_ascii_lower(known) == name for known in _FACTORIES)


def resolve_provider(key: str, config: dict[str, Any] | None = None) -> Provider:
    """Build the provider registered under ``key``.

    :raises PrismError: with code ``unsupported_provider_action`` for a key no
        provider is registered under.
    """
    factory = _FACTORIES.get(key) or _default_factory(key)

    if factory is None:
        raise PrismError(
            ErrorCode.UNSUPPORTED_PROVIDER_ACTION,
            f"No provider is registered for '{key}'.",
        )

    return factory(**(config or {}))


def _default_factory(key: str) -> ProviderFactory | None:
    """Built-in providers, imported on first use.

    Deferred rather than imported at module level so the builder and the
    providers do not have to import each other at start-up.
    """
    if key == "anthropic":
        from prism.providers.anthropic.provider import Anthropic

        _FACTORIES["anthropic"] = Anthropic
        return Anthropic

    if key == "mistral":
        from prism.providers.mistral.provider import Mistral

        _FACTORIES["mistral"] = Mistral
        return Mistral

    if key == "openai":
        from prism.providers.openai.provider import OpenAI

        _FACTORIES["openai"] = OpenAI
        return OpenAI

    return None
