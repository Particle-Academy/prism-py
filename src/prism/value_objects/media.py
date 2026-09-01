"""The parts a user message is made of: text, and payloads."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from prism.errors import PrismError
from prism.value_objects.media_file import Audio, Document, Image, Media, Video

__all__ = ["Part", "Text", "part_from_dict"]


@dataclass(frozen=True)
class Text:
    """A plain-text part of a user message.

    SERIALISES AS ``{"text": ...}`` AND NOTHING ELSE -- no ``kind``
    discriminator, unlike every other part. That asymmetry is deliberate and
    load-bearing: the conformance corpus pins this exact form (``rtp-0001``, and
    every row of ``openai-text-response``), and the PHP reference emits it too,
    so a text part is one of the few places all three implementations are byte
    for byte identical. Adding a key for symmetry would take both ports out of
    parity with the reference on the only part type the reference can currently
    produce, and would make every message a consumer has already stored
    unreadable.

    A part carrying no ``kind`` therefore MEANS text, which is what
    :func:`part_from_dict` relies on.
    """

    KIND: ClassVar[str] = "text"

    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Text:
        return cls(text=data["text"])


#: One part of a user turn: text, or a payload.
#:
#: :class:`~prism.value_objects.media_file.Media` rather than the four
#: subclasses by name, so a payload type added later is a part without this
#: alias having to be edited to allow it.
Part = Text | Media

_KINDS: dict[str, type[Media]] = {
    Image.KIND: Image,
    Document.KIND: Document,
    Audio.KIND: Audio,
    Video.KIND: Video,
}


def part_from_dict(data: Mapping[str, Any]) -> Part:
    """Rebuild a part from its serialised form.

    NO ``kind`` MEANS TEXT. That is the specified form rather than a tolerated
    one: a text part serialises as ``{"text": ...}`` in this port, in
    ``prism-ts`` and in the PHP reference alike, pinned by the conformance
    corpus -- see :class:`Text`. So every user message any consumer has already
    stored reads back unchanged, and a discriminator is only ever needed to tell
    the payload kinds apart from each other, which is the one thing their keys
    cannot do (an image, an audio file and a video serialise identically).

    :raises PrismError: with code ``unknown_message_type``.
    """
    kind = data.get("kind")

    if kind is None:
        if isinstance(data.get("text"), str):
            return Text.from_dict(data)

        raise PrismError.unknown_message_type(
            f"media part {dict(data)!r} (no `kind`, and no text to fall back to)"
        )

    factory = _KINDS.get(kind) if isinstance(kind, str) else None

    if factory is None:
        raise PrismError.unknown_message_type(f"media part of kind {kind!r}")

    return factory.from_dict(data)
