"""Server-sent events, reassembled from whatever the network handed over."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

__all__ = ["sse_data"]


def sse_data(chunks: Iterable[str]) -> Iterator[str]:
    """Yield the ``data:`` payloads out of a chunked SSE body.

    THE TRANSPORT YIELDS CHUNKS, NOT LINES, and this is where that costs
    something and pays for itself. A chunk can end mid-line, mid-JSON, or mid
    anything -- providers do not align their writes to the reader's
    convenience -- so the buffer below is the whole point of the split. A
    transport that promised lines would have to do this internally, where no
    test could split a payload at an awkward place on purpose.

    Only ``data:`` lines are surfaced. SSE also carries ``event:``, ``id:`` and
    comments; neither provider this port speaks to uses them for anything the
    payload does not already say, and inventing meaning for them would be
    guessing at a protocol rather than reading it.
    """
    buffer = ""

    for chunk in chunks:
        buffer += chunk

        # Split on "\n", not "\r\n": the spec allows either, and splitting on
        # the longer one leaves a stray "\r" on every line from a server that
        # uses it.
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            payload = _payload(line)

            if payload is not None:
                yield payload

    # A final line with no trailing newline is still a line. Dropping it loses
    # the last event of any stream a server closes without one.
    payload = _payload(buffer)

    if payload is not None:
        yield payload


def _payload(line: str) -> str | None:
    line = line.rstrip("\r")

    if not line.startswith("data:"):
        return None

    # One optional space after the colon is part of the format, not content.
    payload = line[5:]
    payload = payload[1:] if payload.startswith(" ") else payload

    # "[DONE]" is OpenAI's sentinel, not JSON. Filtered here rather than left
    # for the caller to parse and fail on.
    return None if payload == "" or payload == "[DONE]" else payload
