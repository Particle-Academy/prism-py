"""Failures, identified by a stable code.

The PHP reference identifies every failure by an English sentence and nothing
else, so any consumer that needs to branch on a failure ends up matching on
prose and every wording improvement becomes a silent breaking change. This port
carries a code on every error instead. The CODE is the contract; the prose is
not, and is free to change.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

__all__ = ["ErrorCode", "PrismError"]


class ErrorCode(str, Enum):
    """The stable identity of a failure."""

    PROMPT_AND_MESSAGES = "prompt_and_messages"
    MAX_TOKENS_EXCEEDED = "max_tokens_exceeded"
    PROVIDER_RESPONSE_ERROR = "provider_response_error"
    UNSUPPORTED_PROVIDER_ACTION = "unsupported_provider_action"
    MALFORMED_TOOL_CALL_ARGUMENTS = "malformed_tool_call_arguments"
    UNKNOWN_MESSAGE_TYPE = "unknown_message_type"
    TOOL_LOOP_NOT_SUPPORTED = "tool_loop_not_supported"
    MISSING_SCHEMA = "missing_schema"
    UNSUPPORTED_STRUCTURED_MODEL = "unsupported_structured_model"
    NO_EMBEDDING_INPUT = "no_embedding_input"
    UNREADABLE_INPUT_FILE = "unreadable_input_file"
    NO_IMAGE_PROMPT = "no_image_prompt"
    NO_MODERATION_INPUT = "no_moderation_input"


class PrismError(Exception):
    """Every failure this package raises.

    ``code`` is a plain string drawn from :class:`ErrorCode`, so callers can
    compare against either the enum member or the literal.
    """

    def __init__(
        self,
        code: ErrorCode | str,
        message: str,
        *,
        status: int | None = None,
        body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code: str = code.value if isinstance(code, ErrorCode) else code
        self.message = message
        self.status = status
        self.body = body

    def __repr__(self) -> str:
        return f"PrismError(code={self.code!r}, message={self.message!r})"

    # -- factories ---------------------------------------------------------

    @classmethod
    def prompt_and_messages(cls) -> PrismError:
        return cls(
            ErrorCode.PROMPT_AND_MESSAGES,
            "A request may carry a prompt or a message list, not both.",
        )

    @classmethod
    def max_tokens_exceeded(cls, status: Any = None, item_type: Any = None) -> PrismError:
        return cls(
            ErrorCode.MAX_TOKENS_EXCEEDED,
            "The provider stopped at the token limit "
            f"(status: {status or 'n/a'}, type: {item_type or 'n/a'}). "
            "On a reasoning model, raise max tokens to cover the internal reasoning budget.",
        )

    @classmethod
    def provider_response_error(
        cls,
        message: str,
        *,
        status: int | None = None,
        body: str | None = None,
    ) -> PrismError:
        return cls(ErrorCode.PROVIDER_RESPONSE_ERROR, message, status=status, body=body)

    @classmethod
    def unsupported_provider_action(cls, action: str, provider: str) -> PrismError:
        return cls(
            ErrorCode.UNSUPPORTED_PROVIDER_ACTION,
            f"{action} is not supported by {provider}.",
        )

    @classmethod
    def malformed_tool_call_arguments(cls, tool_name: str) -> PrismError:
        return cls(
            ErrorCode.MALFORMED_TOOL_CALL_ARGUMENTS,
            f"Tool call arguments for tool {tool_name} are not valid JSON.",
        )

    @classmethod
    def unknown_message_type(cls, message_type: str) -> PrismError:
        return cls(
            ErrorCode.UNKNOWN_MESSAGE_TYPE,
            f"Could not map message type {message_type}.",
        )

    @classmethod
    def tool_loop_not_supported(cls) -> PrismError:
        return cls(
            ErrorCode.TOOL_LOOP_NOT_SUPPORTED,
            "The response finished on tool calls. Executing tools is not part of this slice.",
        )
