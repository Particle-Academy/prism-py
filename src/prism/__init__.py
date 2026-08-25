"""prism — a unified API layer over LLM providers.

This is the Python port of the Prism text capability against OpenAI's Responses
API:

>>> from prism import Prism
>>> response = (                             # doctest: +SKIP
...     Prism.text()
...     .using("openai", "gpt-4o")
...     .with_prompt("Who are you?")
...     .as_text()
... )
"""

from __future__ import annotations

from prism import canonical
from prism.canonical import encode as canonical_encode
from prism.enums import FinishReason, ToolChoice
from prism.enums import Provider as ProviderName
from prism.errors import ErrorCode, PrismError
from prism.http import HttpRequest, HttpResponse, Transport, UrllibTransport
from prism.prism import Prism
from prism.providers.base import Provider
from prism.providers.openai import OpenAI, build_request_body, build_tools, parse_text_response
from prism.registry import register_provider, resolve_provider
from prism.schema import BooleanSchema, NumberSchema, Schema, StringSchema
from prism.text import PendingRequest, Request, Response, ResponseBuilder, Step
from prism.tool import Tool
from prism.value_objects import (
    AssistantMessage,
    Message,
    Meta,
    ProviderTool,
    ProviderToolCall,
    SystemMessage,
    Text,
    ToolCall,
    ToolResult,
    ToolResultMessage,
    Usage,
    UserMessage,
    message_from_dict,
)

__version__ = "0.1.0"

__all__ = [
    "AssistantMessage",
    "BooleanSchema",
    "ErrorCode",
    "FinishReason",
    "HttpRequest",
    "HttpResponse",
    "Message",
    "Meta",
    "NumberSchema",
    "OpenAI",
    "PendingRequest",
    "Prism",
    "PrismError",
    "Provider",
    "ProviderName",
    "ProviderTool",
    "ProviderToolCall",
    "Request",
    "Response",
    "ResponseBuilder",
    "Schema",
    "Step",
    "StringSchema",
    "SystemMessage",
    "Text",
    "Tool",
    "ToolCall",
    "ToolChoice",
    "ToolResult",
    "ToolResultMessage",
    "Transport",
    "UrllibTransport",
    "Usage",
    "UserMessage",
    "__version__",
    "build_request_body",
    "build_tools",
    "canonical",
    "canonical_encode",
    "message_from_dict",
    "parse_text_response",
    "register_provider",
    "resolve_provider",
]
