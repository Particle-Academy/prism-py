"""The OpenAI provider: the Responses API, or chat completions by ``api_format``."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from contextlib import closing
from dataclasses import replace
from typing import Any
from urllib.parse import quote, urlencode

from prism import canonical
from prism._php import data_get
from prism.audio.request import SpeechToTextRequest, TextToSpeechRequest
from prism.audio.response import AudioResponse, AudioTextResponse
from prism.batch.batch_job import BatchJob, BatchListResult, BatchResultItem
from prism.batch.request import (
    BatchRequest,
    CancelBatchRequest,
    GetBatchResultsRequest,
    ListBatchesRequest,
    RetrieveBatchRequest,
)
from prism.embeddings.request import EmbeddingsRequest
from prism.embeddings.response import EmbeddingsResponse
from prism.errors import PrismError
from prism.files.file_data import DeleteFileResult, FileData, FileListResult
from prism.files.request import (
    DeleteFileRequest,
    DownloadFileRequest,
    GetFileMetadataRequest,
    ListFilesRequest,
    UploadFileRequest,
)
from prism.http import (
    DEFAULT_TIMEOUT,
    HttpRequest,
    HttpResponse,
    HttpStreamResponse,
    MultipartBody,
    StreamTransport,
    Transport,
    UrllibStreamTransport,
    UrllibTransport,
    encode_multipart,
)
from prism.images.request import ImagesRequest
from prism.images.response import ImagesResponse
from prism.moderation.request import ModerationRequest
from prism.moderation.response import ModerationResponse
from prism.providers.base import Provider
from prism.providers.config import (
    UNSET,
    Unset,
    configured_string,
    configured_url,
    inherits_environment_credentials,
    scoped_nullable_string,
    scoped_string,
)
from prism.providers.openai.audio import (
    build_speech_body,
    build_transcription_form,
    parse_speech_response,
    parse_transcription_response,
)
from prism.providers.openai.batch import (
    build_batch_body,
    build_batch_input_file,
    build_batch_list_query,
    parse_batch_job,
    parse_batch_list_response,
    parse_batch_results,
)
from prism.providers.openai.chat_completions import (
    build_chat_request_body,
    build_chat_stream_body,
    build_chat_structured_body,
    parse_chat_response,
    with_json_instruction,
)
from prism.providers.openai.chat_stream import ChatCompletionsStreamMapper
from prism.providers.openai.embeddings import build_embeddings_body, parse_embeddings_response
from prism.providers.openai.files import (
    build_list_query,
    build_upload_form,
    parse_delete_response,
    parse_file_data,
    parse_file_list_response,
)
from prism.providers.openai.images import build_images_body, parse_images_response
from prism.providers.openai.moderation import build_moderation_body, parse_moderation_response
from prism.providers.openai.rate_limits import parse_rate_limits
from prism.providers.openai.request_body import build_request_body
from prism.providers.openai.response import parse_text_response
from prism.providers.openai.stream_events import map_stream_event
from prism.providers.openai.structured_body import build_structured_body
from prism.streaming.events import ErrorEvent, StreamEvent
from prism.streaming.failure_usage import StreamFailureUsageTracker, failure_aware_sse
from prism.structured.from_text import structured_from_text_response
from prism.structured.request import StructuredRequest
from prism.structured.response import StructuredResponse
from prism.text.request import Request
from prism.text.response import Response

__all__ = ["OpenAI"]

DEFAULT_URL = "https://api.openai.com/v1"

#: The formats this provider speaks. Anything else is refused when it is used.
CHAT_COMPLETIONS = "chat_completions"
RESPONSES = "responses"


class OpenAI(Provider):
    """Talks to OpenAI's Responses API, or to chat completions.

    ``api_format`` picks the wire format for text, structured and stream:
    ``responses`` (the default) or ``chat_completions``. Every other capability
    has one endpoint and ignores it.

    Configuration comes from explicit constructor arguments, falling back to
    ``OPENAI_API_KEY`` / ``OPENAI_URL`` / ``OPENAI_ORGANIZATION`` /
    ``OPENAI_PROJECT`` / ``OPENAI_API_FORMAT``. A named OpenAI-compatible
    provider is built by :meth:`from_entry` instead, which reads NONE of them.

    ``transport`` is injectable so nothing here has to reach the network in a
    test.
    """

    def __init__(
        self,
        api_key: str | Unset | None = UNSET,
        url: str | Unset | None = UNSET,
        organization: str | Unset | None = UNSET,
        project: str | Unset | None = UNSET,
        api_format: str | Unset | None = UNSET,
        transport: Transport | None = None,
        stream_transport: StreamTransport | None = None,
        timeout: float | None = None,
    ) -> None:
        resolved_url = configured_url(url, "OPENAI_URL", DEFAULT_URL)
        # Environment credentials belong to the environment's endpoint only.
        inherit = url is UNSET or inherits_environment_credentials(
            resolved_url, "OPENAI_URL", DEFAULT_URL
        )
        self.url = resolved_url.rstrip("/")
        self.api_key = scoped_string(api_key, "OPENAI_API_KEY", inherit, "", "")
        self.organization = scoped_nullable_string(organization, "OPENAI_ORGANIZATION", inherit)
        self.project = scoped_nullable_string(project, "OPENAI_PROJECT", inherit)
        self.api_format = configured_string(
            api_format, "OPENAI_API_FORMAT", "responses", "responses"
        )
        self.timeout = timeout
        self._transport: Transport = transport or UrllibTransport()
        self._stream_transport: StreamTransport = stream_transport or UrllibStreamTransport()

    @classmethod
    def from_entry(
        cls,
        *,
        url: str,
        api_key: str | None,
        organization: str | None,
        project: str | None,
        api_format: str,
        transport: Transport | None = None,
        stream_transport: StreamTransport | None = None,
        timeout: float | None = None,
    ) -> OpenAI:
        """A provider configured by these values ALONE, never by the environment.

        The constructor fills every absent value from ``OPENAI_*``, and an
        explicit url equal to ``OPENAI_URL`` inherits the environment's key.
        Both are right for ``openai`` and wrong for a named compatible endpoint,
        whose credentials must be its own entry's and nothing else -- so this
        path does not go through the constructor at all. A missing or empty
        key, organization or project sends no header.

        Redirects are NOT followed. urllib re-sends the Authorization header to
        wherever a 301, 302 or 303 points, so following one would hand the
        entry's key to a host nobody configured. A redirect comes back as a
        response and is reported as an error.
        """
        provider = cls.__new__(cls)
        provider.url = url.rstrip("/")
        provider.api_key = api_key or ""
        provider.organization = organization or None
        provider.project = project or None
        provider.api_format = api_format
        provider.timeout = timeout
        provider._transport = _NoRedirects(transport or UrllibTransport())
        provider._stream_transport = _NoStreamRedirects(stream_transport or UrllibStreamTransport())
        return provider

    def text(self, request: Request) -> Response:
        if self.api_format == CHAT_COMPLETIONS:
            decoded, headers = self._chat_send(build_chat_request_body(request), request)
            return parse_chat_response(request, decoded, parse_rate_limits(headers))

        return self._send(build_request_body(request), request, "text")

    def stream(self, request: Request) -> Iterator[StreamEvent]:
        """The same generation, delivered as it arrives.

        ``stream: True`` is added to the SAME body the non-streamed path
        builds, so the two cannot drift apart into different requests that
        merely look alike.
        """
        chat = self.api_format == CHAT_COMPLETIONS
        if not chat and self.api_format != RESPONSES:
            self._unsupported(f"stream via the {self.api_format} API")

        payload = (
            build_chat_stream_body(request)
            if chat
            else {**build_request_body(request), "stream": True}
        )
        body = canonical.encode(payload).encode("utf-8")
        headers = {**self._headers(len(body)), "Accept": "text/event-stream"}

        response = self._stream_transport.stream(
            HttpRequest(
                method="POST",
                url=f"{self.url}/chat/completions" if chat else f"{self.url}/responses",
                headers=headers,
                body=body,
                timeout=self.timeout if self.timeout is not None else DEFAULT_TIMEOUT,
                client_options=request.client_options,
            )
        )

        # 3xx too: a provider that does not follow redirects sees one as a response.
        if response.status >= 300:
            # An error response is not an event stream; the message inside it is
            # the only useful thing about this call.
            text = "".join(response.chunks)
            raise PrismError.provider_response_error(
                f"OpenAI error [{response.status}]: "
                f"{data_get(_loads(text), 'error.message', 'unknown')}",
                status=response.status,
                body=text,
            )

        if chat:
            return self._chat_events(response.chunks, request.model)

        return self._events(response.chunks)

    def _chat_events(self, chunks: Iterator[str], model: str) -> Iterator[StreamEvent]:
        """One mapper per stream; it holds half-assembled text and tool calls."""
        mapper = ChatCompletionsStreamMapper(model)

        with closing(failure_aware_sse(chunks, mapper.failure_usage)) as payloads:
            for payload in payloads:
                if isinstance(payload, ErrorEvent):
                    # The transport failed mid-stream: there is no end to report.
                    yield payload
                    return
                # The stream closes with a literal `[DONE]`, which is not JSON.
                if payload == "[DONE]":
                    break
                parsed = _stream_loads(payload)
                if parsed is None:
                    yield mapper.decode_failure()
                    return
                yield from mapper.map(parsed)

        yield from mapper.finish()

    def _events(self, chunks: Iterator[str]) -> Iterator[StreamEvent]:
        failure = StreamFailureUsageTracker("openai")
        with closing(failure_aware_sse(chunks, failure.snapshot)) as payloads:
            for payload in payloads:
                if isinstance(payload, ErrorEvent):
                    yield payload
                    continue
                parsed = _stream_loads(payload)
                if parsed is None:
                    yield ErrorEvent(
                        code="stream_decode_error",
                        message="The stream contained invalid JSON.",
                        recoverable=False,
                        failure_usage=failure.snapshot(),
                    )
                    break
                failure.record(parsed)
                event = map_stream_event(parsed, failure.snapshot())

                if event is not None:
                    yield event

    def embeddings(self, request: EmbeddingsRequest) -> EmbeddingsResponse:
        """Vectors for one or more inputs.

        A different endpoint from the rest of this provider, so it does not go
        through ``_send``: that helper posts to ``/responses`` and parses a text
        reply, and bending it to also mean ``/embeddings`` would make both
        harder to read than two short methods.
        """
        body = canonical.encode(build_embeddings_body(request)).encode("utf-8")
        response = self._transport.send(
            HttpRequest(
                method="POST",
                url=f"{self.url}/embeddings",
                headers=self._headers(len(body)),
                body=body,
                timeout=self.timeout if self.timeout is not None else DEFAULT_TIMEOUT,
                client_options=request.client_options,
            )
        )

        decoded = self._decode(response.status, response.body)

        if response.status >= 400:
            raise PrismError.provider_response_error(
                f"OpenAI error [{response.status}]: "
                f"{data_get(decoded, 'error.message', 'unknown')}",
                status=response.status,
                body=response.body.decode("utf-8", errors="replace"),
            )

        return parse_embeddings_response(decoded)

    def images(self, request: ImagesRequest) -> ImagesResponse:
        body = canonical.encode(build_images_body(request)).encode("utf-8")
        response = self._transport.send(
            HttpRequest(
                method="POST",
                url=f"{self.url}/images/generations",
                headers=self._headers(len(body)),
                body=body,
                timeout=self.timeout if self.timeout is not None else DEFAULT_TIMEOUT,
                client_options=request.client_options,
            )
        )

        decoded = self._decode(response.status, response.body)

        if response.status >= 400:
            raise PrismError.provider_response_error(
                f"OpenAI error [{response.status}]: "
                f"{data_get(decoded, 'error.message', 'unknown')}",
                status=response.status,
                body=response.body.decode("utf-8", errors="replace"),
            )

        return parse_images_response(decoded, request.model)

    def moderation(self, request: ModerationRequest) -> ModerationResponse:
        body = canonical.encode(build_moderation_body(request)).encode("utf-8")
        response = self._transport.send(
            HttpRequest(
                method="POST",
                url=f"{self.url}/moderations",
                headers=self._headers(len(body)),
                body=body,
                timeout=self.timeout if self.timeout is not None else DEFAULT_TIMEOUT,
                client_options=request.client_options,
            )
        )

        decoded = self._decode(response.status, response.body)

        if response.status >= 400:
            raise PrismError.provider_response_error(
                f"OpenAI error [{response.status}]: "
                f"{data_get(decoded, 'error.message', 'unknown')}",
                status=response.status,
                body=response.body.decode("utf-8", errors="replace"),
            )

        return parse_moderation_response(decoded, request.model)

    def text_to_speech(self, request: TextToSpeechRequest) -> AudioResponse:
        """Speech, returned as BYTES.

        Nothing new is needed on the transport for this: :class:`HttpResponse`
        already carries a ``bytes`` body, because ``urllib`` hands one over. The
        TypeScript port needed a second transport here, since ``fetch`` decodes
        to text; this one does not, which is the divergence rather than an
        omission.

        The ERROR path is still JSON, so a failure is decoded as such rather
        than reported as an unreadable body.
        """
        body = canonical.encode(build_speech_body(request)).encode("utf-8")
        response = self._transport.send(
            HttpRequest(
                method="POST",
                url=f"{self.url}/audio/speech",
                headers=self._headers(len(body)),
                body=body,
                timeout=self.timeout if self.timeout is not None else DEFAULT_TIMEOUT,
                client_options=request.client_options,
            )
        )

        if response.status >= 400:
            decoded = self._decode(response.status, response.body)
            raise PrismError.provider_response_error(
                f"OpenAI error [{response.status}]: "
                f"{data_get(decoded, 'error.message', 'unknown')}",
                status=response.status,
                body=response.body.decode("utf-8", errors="replace"),
            )

        return parse_speech_response(response.body, response.headers.get("content-type"), request)

    def speech_to_text(self, request: SpeechToTextRequest) -> AudioTextResponse:
        """A transcript, uploaded as multipart.

        The only capability here that does not post JSON. The Content-Type
        carries the boundary the encoder chose, so it REPLACES the JSON one
        rather than sitting alongside it -- a form sent under
        ``application/json`` is rejected as malformed, naming the wrong thing.
        """
        content_type, body = encode_multipart(build_transcription_form(request))
        headers = {**self._headers(len(body)), "Content-Type": content_type}

        response = self._transport.send(
            HttpRequest(
                method="POST",
                url=f"{self.url}/audio/transcriptions",
                headers=headers,
                body=body,
                timeout=self.timeout if self.timeout is not None else DEFAULT_TIMEOUT,
                client_options=request.client_options,
            )
        )

        decoded = self._decode(response.status, response.body)

        if response.status >= 400:
            raise PrismError.provider_response_error(
                f"OpenAI error [{response.status}]: "
                f"{data_get(decoded, 'error.message', 'unknown')}",
                status=response.status,
                body=response.body.decode("utf-8", errors="replace"),
            )

        return parse_transcription_response(decoded)

    # -- files -------------------------------------------------------------

    def upload_file(self, request: UploadFileRequest) -> FileData:
        return parse_file_data(self._file("POST", "files", request, build_upload_form(request)))

    def list_files(self, request: ListFilesRequest) -> FileListResult:
        query = urlencode(build_list_query(request))
        path = "files" if query == "" else f"files?{query}"

        return parse_file_list_response(self._file("GET", path, request))

    def get_file_metadata(self, request: GetFileMetadataRequest) -> FileData:
        return parse_file_data(
            self._file("GET", f"files/{quote(request.file_id, safe='')}", request)
        )

    def delete_file(self, request: DeleteFileRequest) -> DeleteFileResult:
        return parse_delete_response(
            self._file("DELETE", f"files/{quote(request.file_id, safe='')}", request)
        )

    def download_file(self, request: DownloadFileRequest) -> bytes:
        """The file's bytes, returned WITHOUT being decoded.

        The only file operation that does not go through ``_file``, because that
        helper decodes JSON and this one must not: a downloaded PDF run through
        a JSON parse raises, and the failure would name the decode rather than
        the download.
        """
        response = self._transport.send(
            HttpRequest(
                method="GET",
                url=f"{self.url}/files/{quote(request.file_id, safe='')}/content",
                headers=self._headers(0),
                timeout=self.timeout if self.timeout is not None else DEFAULT_TIMEOUT,
                client_options=request.client_options,
            )
        )

        if response.status >= 400:
            decoded = self._decode(response.status, response.body)
            raise PrismError.provider_response_error(
                f"OpenAI error [{response.status}]: "
                f"{data_get(decoded, 'error.message', 'unknown')}",
                status=response.status,
                body=response.body.decode("utf-8", errors="replace"),
            )

        return response.body

    # -- batch -------------------------------------------------------------

    def batch(self, request: BatchRequest) -> BatchJob:
        """Submit a batch, uploading its items as a file when they were given.

        The either/or is checked HERE rather than on the request, because it is
        a provider rule: a provider that accepted inline items would have
        nothing to complain about.
        """
        if request.input_file_id is not None and request.items is not None:
            raise PrismError.provider_response_error(
                "OpenAI batch takes either an input file id or items, not both."
            )

        if request.input_file_id is None and request.items is None:
            raise PrismError.provider_response_error(
                "OpenAI batch needs either an input file id or items."
            )

        input_file_id = request.input_file_id

        if input_file_id is None:
            uploaded = self.upload_file(
                UploadFileRequest(
                    filename=f"prism-batch-{uuid.uuid4()}.jsonl",
                    content=build_batch_input_file(request.items or ()).encode("utf-8"),
                    mime_type="application/jsonl",
                    provider_key="openai",
                    client_options=dict(request.client_options),
                    # `purpose` MUST be `batch` for a file the batch endpoint
                    # will read, so it is set here rather than inherited from
                    # whatever the caller passed for the batch itself.
                    provider_options={"purpose": "batch"},
                )
            )
            input_file_id = uploaded.id

        return parse_batch_job(
            self._file(
                "POST",
                "batches",
                request,
                json_body=build_batch_body(
                    input_file_id, request.provider_options.get("completion_window")
                ),
            )
        )

    def retrieve_batch(self, request: RetrieveBatchRequest) -> BatchJob:
        return parse_batch_job(
            self._file("GET", f"batches/{quote(request.batch_id, safe='')}", request)
        )

    def list_batches(self, request: ListBatchesRequest) -> BatchListResult:
        query = urlencode(build_batch_list_query(request))
        path = "batches" if query == "" else f"batches?{query}"

        return parse_batch_list_response(self._file("GET", path, request))

    def cancel_batch(self, request: CancelBatchRequest) -> BatchJob:
        return parse_batch_job(
            self._file("POST", f"batches/{quote(request.batch_id, safe='')}/cancel", request)
        )

    def get_batch_results(self, request: GetBatchResultsRequest) -> tuple[BatchResultItem, ...]:
        """Every result in a batch, from the files it wrote.

        BOTH files are read, output and error, because a batch where some
        requests succeeded and others failed writes to both -- reading only the
        output file would silently drop every failure and report a clean run.

        An unfinished batch has neither, and answers with an empty tuple rather
        than an error: nothing went wrong, there is just nothing yet.
        """
        batch = self.retrieve_batch(
            RetrieveBatchRequest(
                batch_id=request.batch_id,
                provider_key=request.provider_key,
                client_options=dict(request.client_options),
                provider_options=dict(request.provider_options),
            )
        )

        items: list[BatchResultItem] = []

        for file_id in (batch.output_file_id, batch.error_file_id):
            if file_id is None:
                continue

            content = self.download_file(
                DownloadFileRequest(
                    file_id=file_id,
                    provider_key=request.provider_key,
                    client_options=dict(request.client_options),
                )
            )
            items.extend(parse_batch_results(content.decode("utf-8", errors="replace")))

        return tuple(items)

    def _file(
        self,
        method: str,
        path: str,
        request: Any,
        multipart: MultipartBody | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """One round trip for every file and batch operation answering with JSON.

        Routed together so their headers, url building and error mapping are
        decided once. The alternative was upload on its own path because it
        sends a form, which is how two paths that should agree stop agreeing.
        Batch joined them rather than growing a second copy: it has the same
        error-inside-a-200 behaviour, and creating a batch uploads a file.
        """
        headers = self._headers(0)
        body: bytes | None = None

        if json_body is not None:
            body = canonical.encode(json_body).encode("utf-8")
            headers = self._headers(len(body))

        if multipart is not None:
            content_type, body = encode_multipart(multipart)
            # REPLACES the JSON content type rather than sitting alongside it: a
            # form sent under application/json is rejected as malformed, naming
            # the wrong thing.
            headers = {**self._headers(len(body)), "Content-Type": content_type}

        response = self._transport.send(
            HttpRequest(
                method=method,
                url=f"{self.url}/{path}",
                headers=headers,
                body=body,
                timeout=self.timeout if self.timeout is not None else DEFAULT_TIMEOUT,
                client_options=request.client_options,
            )
        )

        decoded = self._decode(response.status, response.body)

        if response.status >= 400:
            raise PrismError.provider_response_error(
                f"OpenAI error [{response.status}]: "
                f"{data_get(decoded, 'error.message', 'unknown')}",
                status=response.status,
                body=response.body.decode("utf-8", errors="replace"),
            )

        # OpenAI reports some file failures with a 200 and an `error` object, so
        # the status alone is not the verdict.
        if isinstance(decoded.get("error"), dict):
            raise PrismError.provider_response_error(
                f"OpenAI error: {data_get(decoded, 'error.message', 'unknown')}",
                status=response.status,
                body=response.body.decode("utf-8", errors="replace"),
            )

        return decoded

    def _send(
        self,
        payload: dict[str, Any],
        request: Request,
        action: str = "structured",
    ) -> Response:
        """One round trip, shared by every capability that posts to /responses.

        Extracted so a structured call cannot drift from a text one: the
        endpoint, the headers, the timeout and the error mapping are decided
        once.
        """
        if self.api_format != RESPONSES:
            self._unsupported(f"{action} via the {self.api_format} API")

        body = canonical.encode(payload).encode("utf-8")
        response = self._transport.send(
            HttpRequest(
                method="POST",
                url=f"{self.url}/responses",
                headers=self._headers(len(body)),
                body=body,
                timeout=self.timeout if self.timeout is not None else DEFAULT_TIMEOUT,
                client_options=request.client_options,
            )
        )

        decoded = self._decode(response.status, response.body)

        if response.status >= 400:
            raise PrismError.provider_response_error(
                f"OpenAI error [{response.status}]: "
                f"{data_get(decoded, 'error.message', 'unknown')}",
                status=response.status,
                body=response.body.decode("utf-8", errors="replace"),
            )

        return parse_text_response(
            request, decoded, parse_rate_limits(response.headers), structured=action == "structured"
        )

    def structured(self, request: StructuredRequest) -> StructuredResponse:
        """Structured output through the Responses API's schema format.

        The reply is parsed by the TEXT parser and then given its structured
        reading, so finish reasons, token-limit failures, usage and rate limits
        behave identically on both paths by construction.

        On chat completions it is JSON mode with the schema in a system message
        appended after the conversation, as the reference sends it.
        """
        if self.api_format == CHAT_COMPLETIONS:
            instructed = with_json_instruction(request)
            decoded, headers = self._chat_send(build_chat_structured_body(instructed), instructed)
            return structured_from_text_response(
                parse_chat_response(
                    instructed, decoded, parse_rate_limits(headers), structured=True
                ),
                assistant_additional_content=False,
            )

        return structured_from_text_response(
            self._send(build_structured_body(request), request), parsed_step=True
        )

    # -- internals ---------------------------------------------------------

    def _chat_send(
        self, payload: dict[str, Any], request: Request
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """One round trip to ``chat/completions``: the decoded body and the headers.

        The body is built by the caller BEFORE this runs, so a conversation that
        cannot be mapped (unusable tool-call arguments) fails with nothing sent.
        """
        body = canonical.encode(payload).encode("utf-8")
        response = self._transport.send(
            HttpRequest(
                method="POST",
                url=f"{self.url}/chat/completions",
                headers=self._headers(len(body)),
                body=body,
                timeout=self.timeout if self.timeout is not None else DEFAULT_TIMEOUT,
                client_options=request.client_options,
            )
        )

        # 3xx too, and before decoding: a provider that does not follow
        # redirects sees one as a response, and its body is not an answer.
        if 300 <= response.status < 400:
            raise PrismError.provider_response_error(
                f"OpenAI error [{response.status}]: the endpoint answered with a redirect,"
                " which is not followed.",
                status=response.status,
            )

        decoded = self._decode(response.status, response.body)

        if response.status >= 400:
            raise PrismError.provider_response_error(
                f"OpenAI error [{response.status}]: "
                f"{data_get(decoded, 'error.message', 'unknown')}",
                status=response.status,
                body=response.body.decode("utf-8", errors="replace"),
            )

        return decoded, response.headers

    def _headers(self, content_length: int) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Content-Length": str(content_length),
        }

        # Omitted entirely when not configured, rather than sent empty.
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        if self.project:
            headers["OpenAI-Project"] = self.project

        return headers

    def _decode(self, status: int, body: bytes) -> dict[str, Any]:
        text = body.decode("utf-8", errors="replace")

        try:
            decoded = json.loads(text) if text else None
        except ValueError:
            decoded = None

        if not isinstance(decoded, dict):
            raise PrismError.provider_response_error(
                f"OpenAI returned a body that is not a JSON object (status {status}).",
                status=status,
                body=text,
            )

        return decoded


class _NoRedirects:
    """Sends every request with ``follow_redirects=False``."""

    def __init__(self, inner: Transport) -> None:
        self._inner = inner

    def send(self, request: HttpRequest) -> HttpResponse:
        return self._inner.send(replace(request, follow_redirects=False))


class _NoStreamRedirects:
    """Streams every request with ``follow_redirects=False``."""

    def __init__(self, inner: StreamTransport) -> None:
        self._inner = inner

    def stream(self, request: HttpRequest) -> HttpStreamResponse:
        return self._inner.stream(replace(request, follow_redirects=False))


def _loads(text: str) -> dict[str, Any]:
    return _stream_loads(text) or {}


def _stream_loads(text: str) -> dict[str, Any] | None:
    """Distinguish malformed JSON from an unknown, valid payload."""
    try:
        parsed = json.loads(text)
    except ValueError:
        return None

    return parsed if isinstance(parsed, dict) else {}
