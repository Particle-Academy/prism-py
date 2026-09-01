"""Images and documents on a user message, and the three ways they are spelled.

G-16: the gap was in the MESSAGE TYPES rather than in one provider, so closing
it lights up three providers at once. Each spells an image differently, which is
exactly why the three maps could not be shared.
"""

from __future__ import annotations

from typing import Any

import pytest

from prism import PrismError, Text, UserMessage, message_from_dict
from prism.providers.anthropic.maps import map_messages as map_anthropic
from prism.providers.mistral.maps import map_messages as map_mistral
from prism.providers.openai.maps import map_messages as map_openai
from prism.value_objects.media import part_from_dict
from prism.value_objects.media_file import Audio, Document, Image, Video

#: base64 of the five bytes "hello" -- the payload every part in this file
#: carries, whatever kind it is dressed as.
HELLO = "aGVsbG8="


def an_image() -> Image:
    image: Image = Image.from_base64(HELLO, "image/png")
    return image


def a_url_image() -> Image:
    image: Image = Image.from_url("https://example.test/a.png")
    return image


# -- the message types -------------------------------------------------------


def test_the_part_kinds_are_separated_and_text_reads_only_the_text() -> None:
    message = UserMessage(
        "look at this",
        [
            an_image(),
            Document.from_url("https://example.test/spec.pdf"),
            Audio.from_url("https://example.test/clip.mp3"),
            Video.from_url("https://example.test/clip.mp4"),
        ],
    )

    # The turn's own content is the only text part, appended last by the
    # constructor. Reading `.text` off every part would raise on the other four.
    assert message.text() == "look at this"
    assert len(message.images()) == 1
    assert len(message.documents()) == 1
    assert len(message.audios()) == 1
    assert len(message.videos()) == 1


def test_media_returns_every_non_text_part() -> None:
    # Reads narrower than it is, and matches the reference: its filter tests
    # `Audio || Video || Media` and the first two are redundant, because both
    # extend Media. So "media" means "not text".
    message = UserMessage("hi", [an_image(), Document.from_url("https://example.test/a.pdf")])

    assert len(message.media()) == 2
    assert not any(isinstance(part, Text) for part in message.media())


def test_several_text_parts_keep_their_order_with_the_content_last() -> None:
    assert UserMessage("and this", [Text("first "), an_image()]).text() == "first and this"


# -- serialisation -----------------------------------------------------------


def test_a_text_part_serialises_exactly_as_the_reference_and_the_corpus_pin_it() -> None:
    # No `kind`. The conformance corpus (rtp-0001, every openai-text-response
    # row) pins this byte for byte and the PHP reference emits the same, so a
    # discriminator here would break parity on the one part type all three
    # implementations share.
    assert Text("hi").to_dict() == {"text": "hi"}


def test_the_payload_kinds_are_discriminated() -> None:
    assert an_image().to_dict()["kind"] == "image"
    assert Audio.from_base64(HELLO, "audio/mpeg").to_dict()["kind"] == "audio"
    assert Video.from_base64(HELLO, "video/mp4").to_dict()["kind"] == "video"
    assert Document.from_url("https://example.test/a.pdf").to_dict()["kind"] == "document"


def test_a_part_with_no_kind_reads_as_text_so_stored_messages_still_load() -> None:
    assert isinstance(part_from_dict({"text": "stored before media existed"}), Text)


def test_a_part_with_neither_a_kind_nor_text_is_refused() -> None:
    with pytest.raises(PrismError):
        part_from_dict({"url": "https://example.test/a.png"})

    with pytest.raises(PrismError, match="hologram"):
        part_from_dict({"kind": "hologram"})


def test_raw_bytes_serialise_as_base64_so_a_local_file_survives_the_round_trip() -> None:
    # `to_dict()` calls `base64()` rather than reading the attribute. A payload
    # built from raw content has bytes and an empty `_base64` until something
    # asks, so reading the attribute would store a full image as `base64: None`
    # and rehydrate an empty one -- silently.
    raw: Image = Image.from_raw_content(b"hello", "image/png")

    assert raw.to_dict()["base64"] == HELLO


def test_a_user_message_round_trips_with_every_part_kind() -> None:
    original = UserMessage(
        "describe these",
        [
            Image.from_url("https://example.test/a.png", "image/png"),
            Document.from_chunks(["one", "two"], "Notes"),
            Audio.from_file_id("file_123"),
        ],
    )

    restored = message_from_dict(original.to_dict())

    assert isinstance(restored, UserMessage)
    assert restored.to_dict() == original.to_dict()
    assert restored.text() == "describe these"
    assert restored.images()[0].url == "https://example.test/a.png"
    assert restored.documents()[0].chunks() == ["one", "two"]
    assert restored.documents()[0].document_title() == "Notes"
    assert restored.audios()[0].file_id() == "file_123"


def test_the_turn_text_is_not_duplicated_when_a_media_part_is_last() -> None:
    # The constructor appends `Text(content)`, so `from_dict` drops a trailing
    # text part that matches. It must not drop a trailing MEDIA part, and the
    # old check compared `.text` on whatever was last -- which would raise on an
    # image rather than leave it alone.
    original = UserMessage("hi", [an_image()])
    restored = message_from_dict(original.to_dict())

    assert isinstance(restored, UserMessage)
    assert restored.text() == "hi"
    assert len(restored.images()) == 1
    assert len(restored.additional_content) == 2


def test_a_stored_field_that_is_not_a_string_is_dropped() -> None:
    # What arrives here is whatever a CONSUMER stored -- a database row, a
    # replayed thread -- so it is not this package's own output by the time it
    # comes back. An unchecked `file_id` of the wrong type would reach the
    # provider payload unexamined, and prism-ts already nulled it, so the two
    # ports disagreed on the same stored message.
    restored = part_from_dict(
        {"kind": "image", "url": {"evil": True}, "file_id": 42, "mime_type": ["image/png"]}
    )

    assert isinstance(restored, Image)
    assert restored.url is None
    assert restored.file_id() is None
    assert restored.mime_type() is None


# -- Document ----------------------------------------------------------------


def test_a_document_takes_its_title_through_titled_not_a_factory_argument() -> None:
    # The reference threads the title as the second positional argument of every
    # factory, where the Media base means the mime type -- so
    # `Document::fromUrl($url, 'application/pdf')` sets the TITLE to
    # "application/pdf". Both are strings, so nothing catches it.
    document: Document = Document.from_url("https://example.test/a.pdf", "application/pdf")
    document.titled("Spec")

    assert document.mime_type() == "application/pdf"
    assert document.document_title() == "Spec"


def test_a_text_document_carries_text_plain() -> None:
    document = Document.from_text("hello", "Greeting")

    assert document.mime_type() == "text/plain"
    assert document.base64() == HELLO
    assert document.is_chunks() is False


# -- Mistral -----------------------------------------------------------------


def _content(mapped: list[dict[str, Any]]) -> list[Any]:
    content = mapped[0]["content"]
    assert isinstance(content, list)
    return content


def test_mistral_wraps_an_image_url_in_an_object() -> None:
    mapped = map_mistral([UserMessage("what is this", [a_url_image()])], [])

    assert _content(mapped) == [
        {"type": "text", "text": "what is this"},
        {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}},
    ]


def test_mistral_inlines_bytes_as_a_data_uri() -> None:
    mapped = map_mistral([UserMessage("what is this", [an_image()])], [])

    assert {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{HELLO}"},
    } in _content(mapped)


def test_mistral_sends_a_document_as_a_url_and_its_name() -> None:
    document: Document = Document.from_url("https://example.test/spec.pdf")
    document.titled("Spec")
    mapped = map_mistral([UserMessage("read this", [document])], [])

    assert {
        "type": "document_url",
        "document_url": "https://example.test/spec.pdf",
        "document_name": "Spec",
    } in _content(mapped)


def test_mistral_refuses_a_document_that_is_only_bytes() -> None:
    # Mistral fetches the document itself, so there is no base64 fallback.
    with pytest.raises(PrismError, match="url only"):
        map_mistral([UserMessage("read this", [Document.from_text("hello", "Notes")])], [])


def test_mistral_refuses_an_image_with_no_mime_type() -> None:
    # `data:;base64,` is accepted by the provider and then rejected as the wrong
    # kind of file -- the failure lands far from its cause.
    untyped: Image = Image.from_raw_content(b"hello")

    with pytest.raises(PrismError, match="mime type"):
        map_mistral([UserMessage("what is this", [untyped])], [])


# -- OpenAI ------------------------------------------------------------------


def test_openai_sends_an_image_url_as_a_bare_string() -> None:
    mapped = map_openai([UserMessage("what is this", [a_url_image()])], [])

    assert _content(mapped) == [
        {"type": "input_text", "text": "what is this"},
        {"type": "input_image", "image_url": "https://example.test/a.png"},
    ]


def test_openai_prefers_a_file_id_over_a_url() -> None:
    mapped = map_openai([UserMessage("what is this", [Image.from_file_id("file_1")])], [])

    assert _content(mapped)[1] == {"type": "input_image", "file_id": "file_1"}


def test_openai_names_an_inline_document_because_it_parses_by_extension() -> None:
    mapped = map_openai([UserMessage("read this", [Document.from_text("hello", "notes.txt")])], [])

    assert _content(mapped)[1] == {
        "type": "input_file",
        "filename": "notes.txt",
        "file_data": f"data:text/plain;base64,{HELLO}",
    }


def test_openai_falls_back_to_the_name_document() -> None:
    mapped = map_openai([UserMessage("read this", [Document.from_text("hello")])], [])

    assert _content(mapped)[1]["filename"] == "document"


def test_openai_refuses_chunks_which_only_anthropic_can_carry() -> None:
    with pytest.raises(PrismError, match="not pre-split chunks"):
        map_openai([UserMessage("read this", [Document.from_chunks(["one"], "Notes")])], [])


# -- Anthropic ---------------------------------------------------------------


def test_anthropic_nests_the_payload_in_a_source_block() -> None:
    mapped = map_anthropic([UserMessage("what is this", [an_image()])])

    assert _content(mapped) == [
        {"type": "text", "text": "what is this"},
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": HELLO},
        },
    ]


def test_anthropic_sends_a_url_image_as_a_url_source_not_fetched_bytes() -> None:
    mapped = map_anthropic([UserMessage("what is this", [a_url_image()])])

    assert _content(mapped)[1] == {
        "type": "image",
        "source": {"type": "url", "url": "https://example.test/a.png"},
    }


def test_anthropic_sends_chunks_as_a_content_source() -> None:
    mapped = map_anthropic(
        [UserMessage("read this", [Document.from_chunks(["one", "two"], "Notes")])]
    )

    assert _content(mapped)[1] == {
        "type": "document",
        "title": "Notes",
        "source": {
            "type": "content",
            "content": [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}],
        },
    }


def test_anthropic_sends_a_text_document_as_text() -> None:
    # Anthropic reads a text/* source directly, and base64-wrapping it would
    # make the model's citations point into an encoded blob.
    mapped = map_anthropic([UserMessage("read this", [Document.from_text("hello", "Notes")])])

    assert _content(mapped)[1]["source"] == {
        "type": "text",
        "media_type": "text/plain",
        "data": "hello",
    }


def test_anthropic_refuses_a_text_document_that_is_not_valid_utf8_by_name() -> None:
    # Left alone this raised UnicodeDecodeError out of a mapper -- uncoded, and
    # saying only that something failed deep inside. `prism-ts` did the opposite
    # and replaced the invalid bytes with U+FFFD, sending corruption Anthropic
    # then cites into. Both ports refuse it now.
    # Built with bytes(), not with an escape inside a literal: this
    # machine's shell wrapper rewrites backslash escapes on the way to a
    # file, which the workspace notes already record as a trap.
    not_utf8: Document = Document.from_raw_content(bytes([0xFF, 0xFE, 0x00]), "text/plain")

    with pytest.raises(PrismError, match="valid UTF-8"):
        map_anthropic([UserMessage("read this", [not_utf8])])


def test_anthropic_omits_the_title_rather_than_sending_null() -> None:
    mapped = map_anthropic(
        [UserMessage("read this", [Document.from_url("https://example.test/a.pdf")])]
    )

    assert "title" not in _content(mapped)[1]


# -- all three ---------------------------------------------------------------


def test_one_image_reaches_all_three_providers_in_three_spellings() -> None:
    message = UserMessage("what is this", [a_url_image()])

    mistral = _content(map_mistral([message], []))[1]
    openai = _content(map_openai([message], []))[1]
    anthropic = _content(map_anthropic([message]))[1]

    assert mistral["image_url"] == {"url": "https://example.test/a.png"}
    assert openai["image_url"] == "https://example.test/a.png"
    assert anthropic["source"] == {"type": "url", "url": "https://example.test/a.png"}
