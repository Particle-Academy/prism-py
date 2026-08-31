"""OpenAI's image-generation endpoint."""

from __future__ import annotations

from typing import Any

from prism._php import data_get
from prism.errors import PrismError
from prism.images.request import ImagesRequest
from prism.images.response import ImagesResponse
from prism.value_objects.generated_image import GeneratedImage
from prism.value_objects.meta import Meta
from prism.value_objects.usage import Usage

__all__ = ["build_images_body", "parse_images_response"]

_OPTIONS = (
    "n",
    "size",
    "quality",
    "style",
    "response_format",
    "background",
    "output_format",
    "user",
)


def build_images_body(request: ImagesRequest) -> dict[str, Any]:
    body: dict[str, Any] = {"model": request.model, "prompt": request.prompt}

    for key in _OPTIONS:
        value = request.provider_options.get(key)
        if value is not None:
            body[key] = value

    return body


def parse_images_response(raw_body: Any, model: str) -> ImagesResponse:
    if not isinstance(raw_body, dict):
        raise PrismError.provider_response_error(
            "OpenAI returned an empty or non-object images response."
        )

    data = raw_body.get("data")
    items = [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []

    return ImagesResponse(
        images=tuple(
            GeneratedImage(
                url=_optional(item.get("url")),
                base64=_optional(item.get("b64_json")),
                revised_prompt=_optional(item.get("revised_prompt")),
            )
            for item in items
        ),
        # Two spellings, because the image endpoints disagree with each other:
        # gpt-image-1 reports input/output tokens and DALL-E reports
        # prompt/completion. Reading only one reports zero cost for the other,
        # which is worse than reporting nothing -- a zero looks like an answer.
        usage=Usage(
            prompt_tokens=_number(raw_body, "usage.input_tokens", "usage.prompt_tokens"),
            completion_tokens=_number(raw_body, "usage.output_tokens", "usage.completion_tokens"),
        ),
        meta=Meta(id=_str(raw_body.get("id")), model=_str(raw_body.get("model")) or model),
        raw=raw_body,
    )


def _optional(value: Any) -> str | None:
    """None rather than an empty string.

    A missing url and an empty one are different answers, and a caller checking
    truthiness would treat them the same.
    """
    return value if isinstance(value, str) and value != "" else None


def _str(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _number(body: dict[str, Any], *paths: str) -> int:
    for path in paths:
        value = data_get(body, path)
        if isinstance(value, int) and not isinstance(value, bool):
            return value

    return 0
