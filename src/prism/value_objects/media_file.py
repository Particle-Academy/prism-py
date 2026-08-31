"""A binary payload, however it was given to us."""

from __future__ import annotations

import base64 as _base64
import mimetypes
from pathlib import Path
from typing import Any

from prism.errors import ErrorCode, PrismError

__all__ = ["Audio", "Document", "Image", "Media", "Video", "guess_mime_type"]


class Media:
    """One type behind five ways of naming the same bytes.

    A url, a base64 string, a local file, raw content, or a provider-side file
    id. Subclasses add nothing but a name; :class:`Image` and :class:`Audio`
    differ in what a provider does with them, not in what they are.

    The alternative was three standalone classes each restating url / base64 /
    mime_type, which is how a package ends up with three subtly different
    answers to "is this a file". ``GeneratedImage`` was the first of those and
    is now :class:`Image`'s sibling under this base.

    TWO DELIBERATE DIVERGENCES FROM THE REFERENCE, both in the port gaps
    register:

    - ``from_storage_path`` is absent. It resolves through Laravel's filesystem,
      and there is nothing here to resolve through.
    - A url is NEVER fetched implicitly. The reference reads url content on
      demand inside ``rawContent()``, so touching a property performs an
      outbound request; here fetching is explicit and separate. A value object
      that reaches the network when you read it turns a stored locator into a
      request at replay time, which is the hazard prism-harness documents about
      replaying threads.
    """

    def __init__(
        self,
        url: str | None = None,
        base64_data: str | None = None,
        mime_type: str | None = None,
    ) -> None:
        self.url = url
        self._base64 = base64_data
        self._mime_type = mime_type
        self._raw_content: bytes | None = None
        self._local_path: str | None = None
        self._file_id: str | None = None
        self._filename: str | None = None

    # -- constructors ------------------------------------------------------

    @classmethod
    def from_file_id(cls, file_id: str) -> Any:
        media = cls()
        media._file_id = file_id
        return media

    @classmethod
    def from_url(cls, url: str, mime_type: str | None = None) -> Any:
        return cls(url=url, mime_type=mime_type)

    @classmethod
    def from_base64(cls, data: str, mime_type: str | None = None) -> Any:
        return cls(base64_data=data, mime_type=mime_type)

    @classmethod
    def from_raw_content(cls, content: bytes, mime_type: str | None = None) -> Any:
        media = cls()
        media._raw_content = content
        if mime_type is not None:
            media._mime_type = mime_type
        return media

    @classmethod
    def from_local_path(cls, path: str, mime_type: str | None = None) -> Any:
        """Read a file from disk.

        Refuses an EMPTY file, matching the reference. A zero-byte upload is
        almost always a mistake upstream, and a provider's error for it is far
        less useful than one naming the path.
        """
        try:
            content = Path(path).read_bytes()
        except OSError as error:
            raise PrismError(
                ErrorCode.UNREADABLE_MEDIA_FILE, f"Could not read the media file [{path}]."
            ) from error

        if not content:
            raise PrismError(ErrorCode.UNREADABLE_MEDIA_FILE, f"The media file [{path}] is empty.")

        media = cls()
        media._raw_content = content
        media._mime_type = mime_type or guess_mime_type(path)
        media._local_path = path
        return media

    @classmethod
    def from_path(cls, path: str) -> Any:
        """Alias, matching the reference's spelling."""
        return cls.from_local_path(path)

    # -- accessors ---------------------------------------------------------

    def as_(self, name: str) -> Media:
        """Name this payload for a provider that wants a filename.

        ``as`` is a Python keyword, so the trailing underscore is forced rather
        than chosen; the reference spells it ``as``.
        """
        self._filename = name
        return self

    def filename(self) -> str | None:
        return self._filename

    def file_id(self) -> str | None:
        return self._file_id

    def local_path(self) -> str | None:
        return self._local_path

    def is_file_id(self) -> bool:
        return self._file_id is not None

    def is_file(self) -> bool:
        return self._local_path is not None

    def is_url(self) -> bool:
        return self.url is not None

    def has_base64(self) -> bool:
        return self._base64 is not None or self._raw_content is not None

    def has_mime_type(self) -> bool:
        return self._mime_type is not None

    def has_raw_content(self) -> bool:
        return self._raw_content is not None

    def mime_type(self) -> str | None:
        return self._mime_type

    def raw_content(self) -> bytes | None:
        if self._raw_content is not None:
            return self._raw_content

        if self._base64 is not None:
            return _base64.b64decode(self._base64)

        # A url is NOT fetched here. See the class docstring.
        return None

    def base64(self) -> str | None:
        """The payload as base64, or None when only a url or file id is known.

        Computed once and kept, because encoding a large file on every access
        is a cost a caller cannot see.
        """
        if self._base64 is not None:
            return self._base64

        if self._raw_content is None:
            return None

        self._base64 = _base64.b64encode(self._raw_content).decode("ascii")
        return self._base64

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "base64": self._base64,
            "mime_type": self._mime_type,
            "file_id": self._file_id,
            "filename": self._filename,
        }


class Image(Media):
    """An image.

    Adds nothing to :class:`Media` but a name, deliberately: an image and an
    audio file differ in what a provider does with them, not in what they are.
    """


class Audio(Media):
    """A recording to transcribe, or speech that was generated."""


class Document(Media):
    """A document."""


class Video(Media):
    """A video."""


def guess_mime_type(path: str) -> str | None:
    """Mime type from the file extension.

    The reference sniffs the CONTENT with finfo. ``mimetypes`` reads the
    extension, and an unknown one returns None rather than a plausible default:
    ``application/octet-stream`` would be accepted by a provider and then
    rejected as the wrong kind of file, which is a worse failure than being
    asked for the type.
    """
    guessed, _ = mimetypes.guess_type(path)
    return guessed
