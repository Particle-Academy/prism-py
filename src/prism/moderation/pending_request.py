"""The fluent builder for moderation."""

from __future__ import annotations

from typing import Any

from prism.enums import Provider as ProviderName
from prism.errors import ErrorCode, PrismError
from prism.moderation.request import ModerationRequest
from prism.moderation.response import ModerationResponse
from prism.providers.base import Provider
from prism.registry import resolve_provider

__all__ = ["ModerationPendingRequest"]


class ModerationPendingRequest:
    """Is this content acceptable, and why not."""

    def __init__(self) -> None:
        self._provider: Provider | None = None
        self._provider_key = ""
        self._model = ""
        self._inputs: list[str] = []
        self._provider_options: dict[str, Any] = {}
        self._client_options: dict[str, Any] = {}

    def using(
        self,
        provider: str | ProviderName,
        model: str = "",
        provider_config: dict[str, Any] | None = None,
    ) -> ModerationPendingRequest:
        self._provider_key = provider.value if isinstance(provider, ProviderName) else provider
        self._model = model
        self._provider = resolve_provider(self._provider_key, provider_config or {})
        return self

    def with_input(self, *inputs: str) -> ModerationPendingRequest:
        """APPENDS, so several calls moderate several inputs in one request."""
        self._inputs.extend(inputs)
        return self

    def with_provider_options(self, options: dict[str, Any]) -> ModerationPendingRequest:
        self._provider_options = dict(options)
        return self

    def with_client_options(self, options: dict[str, Any]) -> ModerationPendingRequest:
        self._client_options = dict(options)
        return self

    def to_request(self) -> ModerationRequest:
        """Freeze the builder.

        :raises PrismError: with code ``no_moderation_input`` when nothing was
            given. Refused rather than sent, and this matters more than the
            other empty-input guards: an empty call returns no results,
            ``is_flagged()`` is then False, and a caller gating on it lets
            everything through. A safety check that fails OPEN because it was
            called wrong is the worst shape in the package.
        """
        if not self._inputs:
            raise PrismError(
                ErrorCode.NO_MODERATION_INPUT,
                "A moderation request needs at least one input. Call with_input().",
            )

        return ModerationRequest(
            model=self._model,
            provider_key=self._provider_key,
            inputs=list(self._inputs),
            client_options=dict(self._client_options),
            provider_options=dict(self._provider_options),
        )

    def as_moderation(self) -> ModerationResponse:
        if self._provider is None:
            raise PrismError(
                ErrorCode.UNSUPPORTED_PROVIDER_ACTION,
                "No provider configured. Call using(<provider>, <model>) first.",
            )

        result: ModerationResponse = self._provider.moderation(self.to_request())

        return result
