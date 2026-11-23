"""Resolving a provider key to a provider instance."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from prism.errors import ErrorCode, PrismError
from prism.providers.config import assert_clean_url

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
            "That is not a valid provider name: a provider name must start with an"
            ' ASCII letter or digit, contain only ASCII letters, digits, ".", "_"'
            ' and "-", and be at most 64 characters long.'
        )
    if not replace and _is_registered(key):
        raise ValueError(
            f"The provider '{key}' is already registered. Pass replace=True to replace it."
        )
    _FACTORIES[key] = factory


# A provider key: ASCII, starting with a letter or digit, at most 64
# characters. A positive allow-list rather than a list of forbidden characters,
# so it needs no Unicode tables and reads the same as the reference's extend()
# and prism-ts registerProvider(). match() with \Z, not $: $ would admit a
# trailing newline. The class is spelled out, so no Unicode table is consulted.
_VALID_PROVIDER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z", re.ASCII)


def _is_valid_provider_name(key: str) -> bool:
    """Whether ``key`` matches ``^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$`` in ASCII.

    A homoglyph (Cyrillic U+043E for the first ``o``), a full-width ``openai``,
    a padded one or one carrying an invisible codepoint is a second key that
    reads as the built-in in a config file or a log, so it is refused, never
    trimmed or folded.
    """
    return isinstance(key, str) and _VALID_PROVIDER_NAME.match(key) is not None


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

    config = config or {}
    # Before the factory runs, so a custom provider is held to the same rule.
    assert_clean_url(config.get("url"))
    return factory(**config)


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
