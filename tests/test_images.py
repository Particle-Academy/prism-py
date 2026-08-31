"""Image generation: revised prompts, two usage spellings, and empty answers."""

from __future__ import annotations

import json
from typing import Any

import pytest

from prism import HttpRequest, HttpResponse, Prism, PrismError, canonical
from prism.providers.openai.images import parse_images_response

OK = {
    "id": "img_1",
    "model": "gpt-image-1",
    "data": [{"b64_json": "aGk=", "revised_prompt": "a cat, photographic, soft light"}],
    "usage": {"input_tokens": 12, "output_tokens": 300},
}


class RecordingTransport:
    def __init__(self, body: dict[str, Any], status: int = 200) -> None:
        self.body = body
        self.status = status
        self.sent: HttpRequest | None = None

    def send(self, request: HttpRequest) -> HttpResponse:
        self.sent = request
        return HttpResponse(status=self.status, body=canonical.encode(self.body).encode("utf-8"))


def _sent(transport: RecordingTransport) -> dict[str, Any]:
    assert transport.sent is not None
    return json.loads(transport.sent.body.decode("utf-8"))


def test_the_prompt_is_posted_and_the_image_comes_back() -> None:
    transport = RecordingTransport(OK)

    response = (
        Prism.images()
        .using("openai", "gpt-image-1", {"api_key": "sk-test", "transport": transport})
        .with_prompt("a cat")
        .generate()
    )

    assert transport.sent is not None
    assert transport.sent.url.endswith("/images/generations")
    assert _sent(transport) == {"model": "gpt-image-1", "prompt": "a cat"}

    image = response.first_image()
    assert image is not None
    # `base64()` is a method now that GeneratedImage extends Media: the base
    # computes it lazily from raw content when only bytes are known.
    assert image.base64() == "aGk="
    assert image.url is None


def test_the_revised_prompt_is_kept() -> None:
    # OpenAI rewrites prompts, often substantially. A caller comparing an image
    # against its prompt is comparing it against THIS one, not the one they typed.
    transport = RecordingTransport(OK)

    response = (
        Prism.images()
        .using("openai", "gpt-image-1", {"api_key": "sk-test", "transport": transport})
        .with_prompt("a cat")
        .generate()
    )

    image = response.first_image()
    assert image is not None
    assert image.has_revised_prompt()
    assert image.revised_prompt == "a cat, photographic, soft light"


def test_both_usage_spellings_are_read() -> None:
    # gpt-image-1 reports input/output tokens; DALL-E reports prompt/completion.
    # Reading only one reports zero cost for the other.
    dalle = parse_images_response(
        {
            "id": "i",
            "model": "dall-e-3",
            "data": [{"url": "https://x/y.png"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 0},
        },
        "dall-e-3",
    )

    assert dalle.usage.prompt_tokens == 7
    assert parse_images_response(OK, "gpt-image-1").usage.prompt_tokens == 12


def test_a_missing_url_is_none_rather_than_an_empty_string() -> None:
    response = parse_images_response({"data": [{"b64_json": "aGk="}]}, "gpt-image-1")

    image = response.first_image()
    assert image is not None
    assert image.url is None


def test_first_image_is_none_when_the_provider_returned_none() -> None:
    # The provider answered; it just answered with nothing. `raw` says why.
    assert parse_images_response({"data": []}, "gpt-image-1").first_image() is None


def test_a_request_with_no_prompt_is_refused() -> None:
    with pytest.raises(PrismError, match="needs a prompt"):
        Prism.images().using("openai", "gpt-image-1").to_request()


def test_the_requested_model_is_used_when_the_response_omits_it() -> None:
    assert parse_images_response({"data": []}, "gpt-image-1").meta.model == "gpt-image-1"


def test_provider_options_pass_through_and_unset_ones_are_omitted() -> None:
    transport = RecordingTransport(OK)

    (
        Prism.images()
        .using("openai", "gpt-image-1", {"api_key": "sk-test", "transport": transport})
        .with_prompt("a cat")
        .with_provider_options({"size": "1024x1024", "n": 2})
        .generate()
    )

    sent = _sent(transport)
    assert sent["size"] == "1024x1024"
    assert sent["n"] == 2
    assert "quality" not in sent
