"""The fluent builder for fill-in-the-middle."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from prism.enums import Provider as ProviderName
from prism.errors import ErrorCode, PrismError
from prism.fim.request import FimRequest
from prism.fim.response import FimResponse
from prism.providers.base import Provider
from prism.registry import resolve_provider

__all__ = ["FimPendingRequest"]


class FimPendingRequest:
    """A prefix in, the gap out.

    Deliberately NOT a subclass of the text builder, for the same reason the
    embeddings one is not: a FIM request has no messages, no tools and no step
    budget, and inheriting those would advertise controls that do nothing. What
    it does have that no other builder does is a SUFFIX.
    """

    def __init__(self) -> None:
        self._provider: Provider | None = None
        self._provider_key = ""
        self._model = ""
        self._prompt = ""
        self._suffix: str | None = None
        self._max_tokens: int | None = None
        self._temperature: float | None = None
        self._top_p: float | None = None
        self._stop: tuple[str, ...] = ()
        self._provider_options: dict[str, Any] = {}
        self._client_options: dict[str, Any] = {}

    def using(
        self,
        provider: str | ProviderName,
        model: str = "",
        provider_config: dict[str, Any] | None = None,
    ) -> FimPendingRequest:
        self._provider_key = provider.value if isinstance(provider, ProviderName) else provider
        self._model = model
        self._provider = resolve_provider(self._provider_key, provider_config or {})
        return self

    def with_prompt(self, prompt: str) -> FimPendingRequest:
        """The text BEFORE the gap. REPLACES; a FIM call has one prompt."""
        self._prompt = prompt
        return self

    def with_suffix(self, suffix: str | None) -> FimPendingRequest:
        """The text AFTER the gap. Omit it and the model completes to the end."""
        self._suffix = suffix
        return self

    def with_max_tokens(self, max_tokens: int) -> FimPendingRequest:
        self._max_tokens = max_tokens
        return self

    def with_temperature(self, temperature: float) -> FimPendingRequest:
        self._temperature = temperature
        return self

    def with_top_p(self, top_p: float) -> FimPendingRequest:
        self._top_p = top_p
        return self

    def with_stop(self, stop: str | Sequence[str]) -> FimPendingRequest:
        """A single stop string or several. A string is wrapped, matching the reference."""
        self._stop = (stop,) if isinstance(stop, str) else tuple(stop)
        return self

    def with_provider_options(self, options: dict[str, Any]) -> FimPendingRequest:
        self._provider_options = dict(options)
        return self

    def with_client_options(self, options: dict[str, Any]) -> FimPendingRequest:
        self._client_options = dict(options)
        return self

    def as_text(self) -> FimResponse:
        if self._provider is None:
            raise PrismError(
                ErrorCode.UNSUPPORTED_PROVIDER_ACTION,
                "No provider configured. Call using(<provider>, <model>) first.",
            )

        result: FimResponse = self._provider.fim(self.to_request())
        return result

    def to_request(self) -> FimRequest:
        return FimRequest(
            model=self._model,
            provider_key=self._provider_key,
            prompt=self._prompt,
            suffix=self._suffix,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            top_p=self._top_p,
            stop=self._stop,
            client_options=dict(self._client_options),
            provider_options=dict(self._provider_options),
        )
