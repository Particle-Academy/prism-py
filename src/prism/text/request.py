"""The resolved, provider-agnostic text request."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from prism._php import data_get
from prism.enums import ToolChoice
from prism.tool import Tool
from prism.value_objects import Message, ProviderTool, SystemMessage

__all__ = ["Request"]


@dataclass
class Request:
    """Everything a provider needs to build one request.

    This is what the pending-request builder freezes into, and what a provider's
    request-body mapper reads. Nothing here is provider-specific: the OpenAI
    spellings (``max_output_tokens``, ``top_p``) are introduced by the mapper.
    """

    model: str
    provider_key: str | None = None
    system_prompts: list[SystemMessage] = field(default_factory=list)
    prompt: str | None = None
    messages: list[Message] = field(default_factory=list)
    max_steps: int = 1
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    tools: list[Tool] = field(default_factory=list)
    client_options: dict[str, Any] = field(default_factory=dict)
    tool_choice: str | ToolChoice | None = None
    provider_options: dict[str, Any] = field(default_factory=dict)
    provider_tools: list[ProviderTool] = field(default_factory=list)
    reasoning_enabled: bool | None = None

    def provider_option(self, path: str | None = None, default: Any = None) -> Any:
        """Read a provider option by dotted path, or the whole mapping."""
        if path is None:
            return self.provider_options

        return data_get(self.provider_options, path, default)

    def add_message(self, message: Message) -> Request:
        self.messages = [*self.messages, message]
        return self
