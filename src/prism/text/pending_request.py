"""The fluent builder behind ``Prism.text()``."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from prism.enums import Provider as ProviderName
from prism.enums import ToolChoice
from prism.errors import ErrorCode, PrismError
from prism.providers.base import Provider
from prism.registry import resolve_provider
from prism.text.request import Request
from prism.text.response import Response
from prism.tool import Tool
from prism.value_objects import Message, ProviderTool, SystemMessage, Text, UserMessage

__all__ = ["PendingRequest"]


class PendingRequest:
    """Collects configuration, then freezes it into a :class:`Request`.

    Every configuration method returns ``self``, so calls chain:

    >>> from prism import Prism
    >>> request = (
    ...     Prism.text()
    ...     .using("openai", "gpt-4o")
    ...     .with_prompt("Who are you?")
    ...     .to_request()
    ... )
    >>> request.model
    'gpt-4o'
    """

    def __init__(self) -> None:
        self._provider: Provider | None = None
        self._provider_key: str | None = None
        self._model: str = ""
        self._prompt: str | None = None
        self._additional_content: list[Text] = []
        self._system_prompts: list[SystemMessage] = []
        self._messages: list[Message] = []
        self._max_steps: int = 1
        self._max_tokens: int | None = None
        self._temperature: float | None = None
        self._top_p: float | None = None
        self._top_k: int | None = None
        self._tools: list[Tool] = []
        self._client_options: dict[str, Any] = {}
        self._tool_choice: str | ToolChoice | None = None
        self._provider_options: dict[str, Any] = {}
        self._provider_tools: list[ProviderTool] = []
        self._reasoning_enabled: bool | None = None

    # -- provider ----------------------------------------------------------

    def using(
        self,
        provider: str | ProviderName,
        model: str = "",
        provider_config: dict[str, Any] | None = None,
    ) -> PendingRequest:
        """Pick the provider and model."""
        self._provider_key = provider.value if isinstance(provider, ProviderName) else provider
        self._model = model
        self._provider = resolve_provider(self._provider_key, provider_config or {})
        return self

    @property
    def provider(self) -> Provider | None:
        return self._provider

    @property
    def provider_key(self) -> str | None:
        return self._provider_key

    @property
    def model(self) -> str:
        return self._model

    # -- prompts and messages ---------------------------------------------

    def with_prompt(
        self,
        prompt: str,
        additional_content: Sequence[Text] | None = None,
    ) -> PendingRequest:
        """Set the single user turn this request is asking about."""
        self._prompt = prompt
        self._additional_content = list(additional_content or [])
        return self

    def with_system_prompt(self, message: str | SystemMessage) -> PendingRequest:
        """APPEND a system prompt. Calling this twice gives two system messages."""
        self._system_prompts.append(
            message if isinstance(message, SystemMessage) else SystemMessage(message)
        )
        return self

    def with_system_prompts(self, messages: Sequence[SystemMessage]) -> PendingRequest:
        """Replace the whole system-prompt list."""
        self._system_prompts = list(messages)
        return self

    def with_messages(self, messages: Sequence[Message]) -> PendingRequest:
        """Supply the conversation explicitly, instead of a prompt."""
        self._messages = list(messages)
        return self

    # -- generation settings ----------------------------------------------

    def with_max_tokens(self, max_tokens: int | None) -> PendingRequest:
        self._max_tokens = max_tokens
        return self

    def using_temperature(self, temperature: float | None) -> PendingRequest:
        self._temperature = temperature
        return self

    def using_top_p(self, top_p: float) -> PendingRequest:
        self._top_p = top_p
        return self

    def using_top_k(self, top_k: int) -> PendingRequest:
        self._top_k = top_k
        return self

    def with_max_steps(self, steps: int) -> PendingRequest:
        self._max_steps = steps
        return self

    def with_reasoning(self, enabled: bool = True) -> PendingRequest:
        """Toggle reasoning output, provider-agnostically.

        Asymmetric on purpose, and the reference is the same: ``False`` asks the
        provider to skip reasoning where it can (OpenAI: ``reasoning.effort =
        minimal``), while ``True`` is reserved for symmetry and emits nothing,
        so it cannot override a per-provider setting the caller made
        deliberately.
        """
        self._reasoning_enabled = enabled
        return self

    # -- tools -------------------------------------------------------------

    def with_tools(self, tools: Sequence[Tool]) -> PendingRequest:
        self._tools = list(tools)
        return self

    def with_tool_choice(self, tool_choice: str | ToolChoice | Tool) -> PendingRequest:
        self._tool_choice = tool_choice.name if isinstance(tool_choice, Tool) else tool_choice
        return self

    def with_provider_tools(self, provider_tools: Sequence[ProviderTool]) -> PendingRequest:
        self._provider_tools = list(provider_tools)
        return self

    # -- transport and provider options ------------------------------------

    def with_provider_options(self, options: dict[str, Any] | None = None) -> PendingRequest:
        self._provider_options = dict(options or {})
        return self

    def with_client_options(self, options: dict[str, Any] | None = None) -> PendingRequest:
        self._client_options = dict(options or {})
        return self

    # -- terminals ---------------------------------------------------------

    def to_request(self) -> Request:
        """Freeze this builder into a :class:`Request`.

        :raises PrismError: with code ``prompt_and_messages`` when both a prompt
            and an explicit message list were set. There is no defensible order
            to merge them in, and silently picking one would send a conversation
            the caller did not write.
        """
        if self._messages and self._prompt is not None:
            raise PrismError.prompt_and_messages()

        messages: list[Message] = [*self._messages]

        # `is not None`, not truthiness. An empty prompt and the prompt "0" are
        # ordinary model input, and both are things a caller can arrive at by
        # interpolation. The reference gates this on PHP truthiness and drops
        # them, producing a request with no user turn at all; the conformance
        # goldens (trq-0021, trq-0022) encode the correct behaviour and this
        # port follows the goldens rather than the reference.
        if self._prompt is not None:
            messages.append(UserMessage(self._prompt, self._additional_content))

        return Request(
            model=self._model,
            provider_key=self._provider_key,
            system_prompts=list(self._system_prompts),
            prompt=self._prompt,
            messages=messages,
            max_steps=self._max_steps,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            top_p=self._top_p,
            top_k=self._top_k,
            tools=list(self._tools),
            client_options=dict(self._client_options),
            tool_choice=self._tool_choice,
            provider_options=dict(self._provider_options),
            provider_tools=list(self._provider_tools),
            reasoning_enabled=self._reasoning_enabled,
        )

    def as_text(self) -> Response:
        """Send the request and return the parsed response."""
        if self._provider is None:
            raise PrismError(
                ErrorCode.UNSUPPORTED_PROVIDER_ACTION,
                "No provider configured. Call using(<provider>, <model>) first.",
            )

        return self._provider.text(self.to_request())
