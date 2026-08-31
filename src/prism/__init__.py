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
from prism.audio import (
    AudioPendingRequest,
    AudioResponse,
    AudioTextResponse,
    SpeechToTextRequest,
    TextToSpeechRequest,
)
from prism.batch import (
    BatchJob,
    BatchJobError,
    BatchJobRequestCounts,
    BatchListResult,
    BatchPendingRequest,
    BatchRequest,
    BatchRequestItem,
    BatchResultItem,
    BatchResultStatus,
    BatchStatus,
    CancelBatchRequest,
    GetBatchResultsRequest,
    ListBatchesRequest,
    RetrieveBatchRequest,
)
from prism.canonical import encode as canonical_encode
from prism.enums import FinishReason, ToolChoice
from prism.enums import Provider as ProviderName
from prism.errors import ErrorCode, PrismError
from prism.files import (
    DeleteFileRequest,
    DeleteFileResult,
    DownloadFileRequest,
    FileData,
    FileListResult,
    FilesPendingRequest,
    GetFileMetadataRequest,
    ListFilesRequest,
    UploadFileRequest,
)
from prism.http import (
    HttpRequest,
    HttpResponse,
    MultipartBody,
    MultipartFile,
    Transport,
    UrllibTransport,
    encode_multipart,
)
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
from prism.value_objects.generated_audio import GeneratedAudio
from prism.value_objects.media_file import Audio, Document, Image, Media, Video

__version__ = "0.1.0"

__all__ = [
    "AssistantMessage",
    "Audio",
    "AudioPendingRequest",
    "AudioResponse",
    "AudioTextResponse",
    "BatchJob",
    "BatchJobError",
    "BatchJobRequestCounts",
    "BatchListResult",
    "BatchPendingRequest",
    "BatchRequest",
    "BatchRequestItem",
    "BatchResultItem",
    "BatchResultStatus",
    "BatchStatus",
    "BooleanSchema",
    "CancelBatchRequest",
    "DeleteFileRequest",
    "DeleteFileResult",
    "Document",
    "DownloadFileRequest",
    "ErrorCode",
    "FileData",
    "FileListResult",
    "FilesPendingRequest",
    "FinishReason",
    "GeneratedAudio",
    "GetBatchResultsRequest",
    "GetFileMetadataRequest",
    "HttpRequest",
    "HttpResponse",
    "Image",
    "ListBatchesRequest",
    "ListFilesRequest",
    "Media",
    "Message",
    "Meta",
    "MultipartBody",
    "MultipartFile",
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
    "RetrieveBatchRequest",
    "Schema",
    "SpeechToTextRequest",
    "Step",
    "StringSchema",
    "SystemMessage",
    "Text",
    "TextToSpeechRequest",
    "Tool",
    "ToolCall",
    "ToolChoice",
    "ToolResult",
    "ToolResultMessage",
    "Transport",
    "UploadFileRequest",
    "UrllibTransport",
    "Usage",
    "UserMessage",
    "Video",
    "__version__",
    "build_request_body",
    "build_tools",
    "canonical",
    "canonical_encode",
    "encode_multipart",
    "message_from_dict",
    "parse_text_response",
    "register_provider",
    "resolve_provider",
]
