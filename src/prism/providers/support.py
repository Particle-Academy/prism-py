"""Helpers shared by more than one provider's media mapping.

A MODULE, not a package. A directory here would be read as a fourth provider by
anything that lists ``prism/providers/`` -- which the agent's ``describe_port``
did until it was pointed at the registry instead.
"""

from __future__ import annotations

from prism.errors import PrismError
from prism.value_objects.media_file import Media

__all__ = ["data_uri"]


def data_uri(media: Media, provider: str, part: str) -> str:
    """A payload as a ``data:`` uri.

    Shared by the two providers that inline bytes into a url string rather than
    into a field of their own. ``provider`` and ``part`` are only carried so a
    failure here names the same provider the caller's mapper would have.

    The mime type is REQUIRED and never defaulted: a ``data:;base64,`` uri is
    accepted by both providers and then fails as the wrong kind of file, which
    is a worse failure than being told the type is missing. ``guess_mime_type``
    reads the extension, so a payload built from raw content is the case that
    arrives here without one.

    :raises PrismError: with code ``unsupported_media``.
    """
    encoded = media.base64()

    if encoded is None:
        raise PrismError.unsupported_media(provider, part, "a url, or bytes to encode")

    mime_type = media.mime_type()

    if mime_type is None:
        raise PrismError.unsupported_media(
            provider,
            part,
            "bytes with a known mime type -- pass one to the factory when it cannot be "
            "read from a file extension",
        )

    return f"data:{mime_type};base64,{encoded}"
