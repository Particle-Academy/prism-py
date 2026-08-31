"""What a provider gave back."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from prism.value_objects.generated_image import GeneratedImage
from prism.value_objects.meta import Meta
from prism.value_objects.usage import Usage

__all__ = ["ImagesResponse"]


@dataclass(frozen=True)
class ImagesResponse:
    images: tuple[GeneratedImage, ...] = ()
    usage: Usage = field(default_factory=lambda: Usage(prompt_tokens=0, completion_tokens=0))
    meta: Meta = field(default_factory=lambda: Meta(id="", model=""))
    additional_content: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] | None = None

    def first_image(self) -> GeneratedImage | None:
        """The first image, or ``None``.

        Most callers ask for one image and want it without indexing. Optional
        rather than raising: a provider that returned none has answered, and the
        caller can read ``raw`` to find out why.
        """
        return self.images[0] if self.images else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "images": [image.to_dict() for image in self.images],
            "usage": self.usage.to_dict(),
            "meta": self.meta.to_dict(),
            "additional_content": dict(self.additional_content),
            "raw": None if self.raw is None else dict(self.raw),
        }
