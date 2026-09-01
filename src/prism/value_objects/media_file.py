"""A binary payload, however it was given to us."""

from __future__ import annotations

import base64 as _base64
import mimetypes
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, cast

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

    #: The discriminator in the serialised form, under the key ``kind``.
    #:
    #: NOT ``type``. :class:`~prism.value_objects.generated_audio.GeneratedAudio`
    #: already serialises ``type`` as the provider's own format name (``mp3``,
    #: ``wav``), so a discriminator spelled ``type`` would be overwritten by it
    #: on exactly one subclass -- the kind of collision that round-trips fine in
    #: every test that does not happen to use that class.
    #:
    #: A subclass of a subclass keeps its parent's kind: a ``GeneratedImage`` is
    #: an ``image``, and reading one back as :class:`Image` loses only the
    #: revised prompt, which is not something a user message carries.
    KIND: ClassVar[str] = "media"

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
        """The serialised form.

        ``base64()``, not the base64 ATTRIBUTE: a payload built from raw content
        or from a local file has bytes and an empty ``_base64`` until something
        asks for it, so reading the attribute would serialise a full image as
        ``base64: None`` and the round trip would silently return an empty one.
        Encoding here costs one pass over bytes the caller has already decided
        to persist.
        """
        return {
            "kind": self.KIND,
            "url": self.url,
            "base64": self.base64(),
            "mime_type": self._mime_type,
            "file_id": self._file_id,
            "filename": self._filename,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Media:
        """Rebuild a payload from its serialised form.

        The local path is NOT restored: it names a file on the machine that
        serialised this, and a path that resolves to a different file elsewhere
        is worse than no path at all. The bytes travel as base64 instead.

        Only ever reached for the four kinds :func:`part_from_dict` dispatches
        to. ``GeneratedImage`` and ``GeneratedAudio`` take different constructor
        arguments and are not message parts, so they are not read back here.

        Every field is TYPE-CHECKED, and a value that is not a string becomes
        ``None`` rather than being carried. What arrives here is whatever a
        consumer stored -- a database row, a replayed thread -- so it is not
        this package's own output by the time it comes back. Without the check a
        stored ``file_id`` of the wrong type reaches the provider payload
        unexamined, and ``prism-ts`` already nulled it, so the two ports
        disagreed on the same stored message.
        """

        def as_str(key: str) -> str | None:
            value = data.get(key)
            return value if isinstance(value, str) else None

        media = cls(
            url=as_str("url"),
            base64_data=as_str("base64"),
            mime_type=as_str("mime_type"),
        )
        media._file_id = as_str("file_id")
        media._filename = as_str("filename")
        return media


class Image(Media):
    """An image.

    Adds nothing to :class:`Media` but a name, deliberately: an image and an
    audio file differ in what a provider does with them, not in what they are.
    """

    KIND: ClassVar[str] = "image"


class Audio(Media):
    """A recording to transcribe, or speech that was generated."""

    KIND: ClassVar[str] = "audio"


class Video(Media):
    """A video."""

    KIND: ClassVar[str] = "video"


class Document(Media):
    """A document -- a PDF, a text file, or text supplied directly as chunks.

    Two things a plain :class:`Media` does not have, because every provider that
    accepts a document asks for them:

    - a TITLE, sent as ``document_name`` by Mistral, ``filename`` by OpenAI and
      ``title`` by Anthropic;
    - CHUNKS, pre-split text sent as a ``content`` source. Anthropic is the only
      provider here that takes them.

    ONE DELIBERATE DIVERGENCE FROM THE REFERENCE. The reference threads the
    title through every factory as the SECOND positional argument --
    ``Document::fromUrl($url, $title)`` -- where the same position on the
    :class:`Media` base means the mime type. So
    ``Document::fromUrl($url, 'application/pdf')`` silently sets the title to
    "application/pdf" and leaves the mime type unset. Both are strings, so
    nothing catches it. The factories keep their base meaning here and the title
    is set by :meth:`titled`, which cannot be confused with anything.
    """

    KIND: ClassVar[str] = "document"

    def __init__(
        self,
        url: str | None = None,
        base64_data: str | None = None,
        mime_type: str | None = None,
    ) -> None:
        super().__init__(url, base64_data, mime_type)
        self._document_title: str | None = None
        self._chunks: list[str] | None = None

    @classmethod
    def from_chunks(cls, chunks: Sequence[str], title: str | None = None) -> Document:
        """A document supplied as pre-split text.

        Carries no bytes and no mime type -- the chunks ARE the document. Only
        Anthropic accepts this form; the other two mappers reject it by name
        rather than sending an empty payload.
        """
        document = cls()
        document._chunks = list(chunks)
        return document if title is None else document.titled(title)

    @classmethod
    def from_text(cls, text: str, title: str | None = None) -> Document:
        """Text as a document, rather than as part of the prompt."""
        document: Document = cls.from_raw_content(text.encode("utf-8"), "text/plain")
        return document if title is None else document.titled(title)

    def titled(self, title: str) -> Document:
        """Name this document for the provider. Returns itself, so it chains onto a factory."""
        self._document_title = title
        return self

    def document_title(self) -> str | None:
        return self._document_title

    def chunks(self) -> list[str] | None:
        return self._chunks

    def is_chunks(self) -> bool:
        return self._chunks is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "document_title": self._document_title,
            "chunks": None if self._chunks is None else list(self._chunks),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Document:
        # `cls` is Document, so this cast is sound; the base signature cannot say
        # so without `typing.Self`, which arrived in 3.11 and this package floors
        # at 3.10.
        document = cast("Document", super().from_dict(data))

        title = data.get("document_title")
        if isinstance(title, str):
            document.titled(title)

        chunks = data.get("chunks")
        if isinstance(chunks, list):
            document._chunks = [chunk for chunk in chunks if isinstance(chunk, str)]

        return document


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
