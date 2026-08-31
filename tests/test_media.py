"""The Media base: five ways of naming the same bytes, and one that never fetches."""

from __future__ import annotations

import base64
import re
from pathlib import Path

import pytest

from prism.errors import PrismError
from prism.value_objects.generated_image import GeneratedImage
from prism.value_objects.media_file import Audio, Image, guess_mime_type


def test_a_local_file_is_read_and_its_mime_type_derived(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    path.write_text("hello", encoding="utf-8")

    media = Image.from_local_path(str(path))

    assert media.is_file() is True
    assert media.mime_type() == "text/plain"
    assert media.base64() == base64.b64encode(b"hello").decode("ascii")


def test_an_empty_file_is_refused_naming_the_path(tmp_path: Path) -> None:
    # A zero-byte upload is almost always a mistake upstream, and a provider's
    # error for it is far less useful than one naming the path.
    path = tmp_path / "empty.png"
    path.write_bytes(b"")

    with pytest.raises(PrismError, match=re.escape("empty.png")):
        Image.from_local_path(str(path))


def test_a_missing_path_is_refused() -> None:
    with pytest.raises(PrismError):
        Image.from_local_path("nope/missing.png")


def test_a_url_is_never_fetched_just_because_something_read_it() -> None:
    # The reference reads url content on demand, so touching a property performs
    # an outbound request. A stored locator becoming a request the moment
    # something replays it is the hazard.
    media = Image.from_url("https://example.test/cat.png")

    assert media.is_url() is True
    assert media.raw_content() is None
    assert media.base64() is None


def test_base64_decodes_without_a_network_call() -> None:
    media = Image.from_base64(base64.b64encode(b"bytes").decode("ascii"), "image/png")

    assert media.raw_content() == b"bytes"
    assert media.mime_type() == "image/png"


def test_a_file_id_carries_no_content() -> None:
    media = Audio.from_file_id("file_123")

    assert media.is_file_id() is True
    assert media.file_id() == "file_123"
    assert media.has_base64() is False


def test_a_payload_can_be_named_for_providers_that_want_a_filename() -> None:
    assert Audio.from_base64("aGk=").as_("clip.mp3").filename() == "clip.mp3"


def test_an_unknown_extension_gives_none_rather_than_a_plausible_default() -> None:
    # application/octet-stream would be accepted by a provider and then rejected
    # as the wrong kind of file -- a worse failure than being asked for the type.
    assert guess_mime_type("thing.qqq") is None
    assert guess_mime_type("clip.mp3") == "audio/mpeg"


def test_generated_image_gains_the_media_surface_and_keeps_its_revised_prompt() -> None:
    # The move under Media is the point: a generated image is a file, and now it
    # answers the same questions every other file does.
    image = GeneratedImage(base64="aGk=", revised_prompt="a cat, photographic")

    assert isinstance(image, Image)
    assert image.has_base64() is True
    assert image.has_revised_prompt() is True
    assert image.to_dict()["revised_prompt"] == "a cat, photographic"
