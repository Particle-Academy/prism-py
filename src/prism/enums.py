"""Provider-agnostic enums."""

from __future__ import annotations

from enum import Enum, auto

__all__ = ["FinishReason", "Provider", "ToolChoice"]


class FinishReason(str, Enum):
    """Why the model stopped generating.

    The values are the wire/storage spellings the reference serialises, hyphens
    included — ``content-filter``, not ``content_filter``.
    """

    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content-filter"
    TOOL_CALLS = "tool-calls"
    PAUSE = "pause"
    REFUSAL = "refusal"
    ERROR = "error"
    OTHER = "other"
    UNKNOWN = "unknown"


class ToolChoice(Enum):
    """How the model should pick among the offered tools.

    Deliberately without backing values, exactly as the reference: the mapping
    to a provider's wire spelling lives in the provider, not on the enum. That
    is what keeps ``ANY`` from serialising as ``"any"`` — OpenAI calls it
    ``"required"``.
    """

    AUTO = auto()
    ANY = auto()
    NONE = auto()


class Provider(str, Enum):
    """Known provider keys. Only :attr:`OPENAI` is implemented in this slice."""

    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    OLLAMA = "ollama"
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    REQUESTY = "requesty"
    MISTRAL = "mistral"
    GROQ = "groq"
    XAI = "xai"
    GEMINI = "gemini"
    VOYAGEAI = "voyageai"
    ELEVENLABS = "elevenlabs"
    REPLICATE = "replicate"
    QWEN = "qwen"
    AZURE = "azure"
    PERPLEXITY = "perplexity"
    VERTEX = "vertex"
    Z = "z"
