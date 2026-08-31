"""The fluent builder for image generation."""

from __future__ import annotations

from typing import Any

from prism.enums import Provider as ProviderName
from prism.errors import ErrorCode, PrismError
from prism.images.request import ImagesRequest
from prism.images.response import ImagesResponse
from prism.providers.base import Provider
from prism.registry import resolve_provider

__all__ = ["ImagesPendingRequest"]


class ImagesPendingRequest:
    """A prompt in, pictures out.

    Like the embeddings builder and unlike the structured one, it does not
    subclass the text builder: an image request has no messages, no tools and no
    step budget, and inheriting those would advertise controls that do nothing.
    """

    def __init__(self) -> None:
        self._provider: Provider | None = None
        self._provider_key = ""
        self._model = ""
        self._prompt: str | None = None
        self._provider_options: dict[str, Any] = {}
        self._client_options: dict[str, Any] = {}

    def using(
        self,
        provider: str | ProviderName,
        model: str = "",
        provider_config: dict[str, Any] | None = None,
    ) -> ImagesPendingRequest:
        self._provider_key = provider.value if isinstance(provider, ProviderName) else provider
        self._model = model
        self._provider = resolve_provider(self._provider_key, provider_config or {})
        return self

    def with_prompt(self, prompt: str) -> ImagesPendingRequest:
        """REPLACES. An image request has one prompt, not a conversation."""
        self._prompt = prompt
        return self

    def with_provider_options(self, options: dict[str, Any]) -> ImagesPendingRequest:
        self._provider_options = dict(options)
        return self

    def with_client_options(self, options: dict[str, Any]) -> ImagesPendingRequest:
        self._client_options = dict(options)
        return self

    def to_request(self) -> ImagesRequest:
        """Freeze the builder.

        :raises PrismError: with code ``no_image_prompt`` when none was set.
            Absence, not emptiness: an empty prompt is refused by the provider
            with its own message, which is more useful than one invented here.
            This catches the caller who forgot the line entirely.
        """
        if self._prompt is None:
            raise PrismError(
                ErrorCode.NO_IMAGE_PROMPT,
                "An images request needs a prompt. Call with_prompt().",
            )

        return ImagesRequest(
            model=self._model,
            provider_key=self._provider_key,
            prompt=self._prompt,
            client_options=dict(self._client_options),
            provider_options=dict(self._provider_options),
        )

    def generate(self) -> ImagesResponse:
        if self._provider is None:
            raise PrismError(
                ErrorCode.UNSUPPORTED_PROVIDER_ACTION,
                "No provider configured. Call using(<provider>, <model>) first.",
            )

        result: ImagesResponse = self._provider.images(self.to_request())

        return result
