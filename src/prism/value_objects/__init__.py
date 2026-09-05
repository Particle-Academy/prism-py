"""The value objects the text capability is built from.

Every one of them serialises with :meth:`to_dict` and rebuilds with
:meth:`from_dict`. The reference has only the writing half — no ``fromArray()``
for messages — which forced a downstream package to invent its own rehydration
and ship a defect with it. Both directions live here on purpose.
"""

from __future__ import annotations

from prism.value_objects.media import Part, Text, part_from_dict
from prism.value_objects.messages import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
    message_from_dict,
)
from prism.value_objects.meta import Meta
from prism.value_objects.provider_rate_limit import ProviderRateLimit
from prism.value_objects.provider_tool import ProviderTool
from prism.value_objects.provider_tool_call import ProviderToolCall
from prism.value_objects.tool_call import ToolCall
from prism.value_objects.tool_result import ToolResult
from prism.value_objects.usage import Usage

__all__ = [
    "AssistantMessage",
    "Message",
    "Meta",
    "Part",
    "ProviderRateLimit",
    "ProviderTool",
    "ProviderToolCall",
    "SystemMessage",
    "Text",
    "ToolCall",
    "ToolResult",
    "ToolResultMessage",
    "Usage",
    "UserMessage",
    "message_from_dict",
    "part_from_dict",
]
