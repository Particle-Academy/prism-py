"""Options cross the provider/transport seam; no option can change guarded routes."""

from __future__ import annotations

from typing import Any

import pytest

from prism.audio.request import TextToSpeechRequest
from prism.files.request import DownloadFileRequest, ListFilesRequest
from prism.http import HttpRequest, HttpResponse, HttpStreamResponse
from prism.providers.anthropic.provider import Anthropic
from prism.providers.mistral.provider import Mistral
from prism.providers.openai.provider import OpenAI
from prism.text.request import Request


class ReachedTransportError(Exception):
    pass


class Capture:
    def __init__(self) -> None:
        self.calls: list[HttpRequest] = []

    def send(self, request: HttpRequest) -> HttpResponse:
        self.calls.append(request)
        raise ReachedTransportError

    def stream(self, request: HttpRequest) -> HttpStreamResponse:
        self.calls.append(request)
        raise ReachedTransportError


@pytest.mark.parametrize("provider_type", [Anthropic, OpenAI, Mistral])
@pytest.mark.parametrize("action", ["text", "stream"])
def test_json_and_stream_forward_options(provider_type: Any, action: str) -> None:
    capture = Capture()
    options = {"timeout": 2.5, "connect_timeout": 0.5}
    provider = provider_type(
        api_key="synthetic",
        url="https://offline.invalid/v1",
        transport=capture,
        stream_transport=capture,
        timeout=9,
    )
    request = Request(model="fixture", client_options=options)
    with pytest.raises(ReachedTransportError):
        result = getattr(provider, action)(request)
        if action == "stream":
            list(result)
    sent = capture.calls[0]
    assert sent.timeout == 2.5
    assert sent.connect_timeout == 0.5
    assert dict(sent.client_options) == options
    assert sent.url.startswith("https://offline.invalid/v1/")


@pytest.mark.parametrize("action", ["text_to_speech", "download_file", "list_files"])
def test_binary_and_file_dispatch_forward_options(action: str) -> None:
    capture = Capture()
    options = {"timeout": 2.5, "connect_timeout": 0.5}
    request: Any
    if action == "download_file":
        request = DownloadFileRequest(file_id="fixture", client_options=options)
    elif action == "list_files":
        request = ListFilesRequest(client_options=options)
    else:
        request = TextToSpeechRequest(
            model="fixture", input="hello", voice="fixture", client_options=options
        )
    with pytest.raises(ReachedTransportError):
        getattr(
            OpenAI(api_key="synthetic", url="https://offline.invalid/v1", transport=capture), action
        )(request)
    assert capture.calls[0].timeout == 2.5
    assert capture.calls[0].connect_timeout == 0.5


@pytest.mark.parametrize(
    "name", ["url", "headers", "body", "follow_redirects", "max_body_bytes", "read_timeout"]
)
def test_options_cannot_override_the_guarded_request(name: str) -> None:
    with pytest.raises(ValueError, match="timeout and connect_timeout"):
        HttpRequest(
            method="GET",
            url="https://offline.invalid/",
            follow_redirects=False,
            max_body_bytes=8,
            client_options={name: "forbidden"},
        )


@pytest.mark.parametrize("value", [-1, True, float("inf"), float("nan"), "2", None])
def test_timeout_values_refuse_invalid_shapes_without_echoing_them(value: Any) -> None:
    with pytest.raises(ValueError, match="finite nonnegative"):
        HttpRequest(method="GET", url="https://offline.invalid/", client_options={"timeout": value})
