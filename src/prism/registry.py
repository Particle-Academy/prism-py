"""Resolving a provider key to a provider instance."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from prism.errors import ErrorCode, PrismError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from prism.providers.base import Provider

__all__ = ["register_provider", "resolve_provider"]

ProviderFactory = Callable[..., "Provider"]

_FACTORIES: dict[str, ProviderFactory] = {}


def register_provider(key: str, factory: ProviderFactory) -> None:
    """Teach the builder about a provider key."""
    _FACTORIES[key] = factory


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

        register_provider("anthropic", Anthropic)
        return Anthropic

    if key == "mistral":
        from prism.providers.mistral.provider import Mistral

        register_provider("mistral", Mistral)
        return Mistral

    if key == "openai":
        from prism.providers.openai.provider import OpenAI

        register_provider("openai", OpenAI)
        return OpenAI

    return None
