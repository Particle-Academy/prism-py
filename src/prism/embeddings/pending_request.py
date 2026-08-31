"""The fluent builder for embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from prism.embeddings.request import EmbeddingsRequest
from prism.embeddings.response import EmbeddingsResponse
from prism.enums import Provider as ProviderName
from prism.errors import ErrorCode, PrismError
from prism.providers.base import Provider
from prism.registry import resolve_provider

__all__ = ["EmbeddingsPendingRequest"]


class EmbeddingsPendingRequest:
    """Text in, vectors out.

    NOT a subclass of the text builder, unlike the structured one. An embeddings
    request has no system prompt, no tools, no temperature and no step budget --
    inheriting twenty methods that all raise or silently do nothing would be a
    larger lie than the small duplication of the four that apply.
    """

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
    ) -> EmbeddingsPendingRequest:
        self._provider_key = provider.value if isinstance(provider, ProviderName) else provider
        self._model = model
        self._provider = resolve_provider(self._provider_key, provider_config or {})
        return self

    def from_input(self, text: str) -> EmbeddingsPendingRequest:
        """APPENDS, so several calls embed several inputs in one request."""
        self._inputs.append(text)
        return self

    def from_array(self, inputs: list[str]) -> EmbeddingsPendingRequest:
        self._inputs.extend(inputs)
        return self

    def from_file(self, path: str) -> EmbeddingsPendingRequest:
        """The file's contents as one input.

        Read EAGERLY rather than at send time: a caller that mistypes a path
        should find out on the line that names it, not inside a provider call
        already billed for the inputs that came before it.
        """
        try:
            self._inputs.append(Path(path).read_text(encoding="utf-8"))
        except OSError as error:
            raise PrismError(
                ErrorCode.UNREADABLE_INPUT_FILE,
                f"Could not read the embeddings input file [{path}].",
            ) from error

        return self

    def with_provider_options(self, options: dict[str, Any]) -> EmbeddingsPendingRequest:
        self._provider_options = dict(options)
        return self

    def with_client_options(self, options: dict[str, Any]) -> EmbeddingsPendingRequest:
        self._client_options = dict(options)
        return self

    def to_request(self) -> EmbeddingsRequest:
        """Freeze the builder.

        :raises PrismError: with code ``no_embedding_input`` when nothing was
            given. The call is billable, comes back with an empty list, and
            reads to the caller as a provider that answered nothing.
        """
        if not self._inputs:
            raise PrismError(
                ErrorCode.NO_EMBEDDING_INPUT,
                "An embeddings request needs at least one input. "
                "Call from_input(), from_array() or from_file().",
            )

        return EmbeddingsRequest(
            model=self._model,
            provider_key=self._provider_key,
            inputs=list(self._inputs),
            client_options=dict(self._client_options),
            provider_options=dict(self._provider_options),
        )

    def as_embeddings(self) -> EmbeddingsResponse:
        if self._provider is None:
            raise PrismError(
                ErrorCode.UNSUPPORTED_PROVIDER_ACTION,
                "No provider configured. Call using(<provider>, <model>) first.",
            )

        result: EmbeddingsResponse = self._provider.embeddings(self.to_request())

        return result
