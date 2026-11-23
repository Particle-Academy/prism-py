"""What checking a long argument list costs."""

from __future__ import annotations

import json
import socket
import tracemalloc
from collections.abc import Callable
from typing import Any

import pytest

from prism import ToolCall, canonical


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("A test tried to use the network.")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def _peak(action: Callable[[], Any]) -> tuple[int, Any]:
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        result = action()

        return tracemalloc.get_traced_memory()[1] - before, result
    finally:
        tracemalloc.stop()


def test_checks_a_long_list_in_about_the_memory_of_reading_and_encoding_it() -> None:
    text = '{"a":[' + ",".join(["0"] * 100_000) + "]}"
    # The check must decode and encode the value. How much encoding holds
    # differs between Python versions, so the bound is measured on this one.
    necessary, _ = _peak(lambda: canonical.encode(json.loads(text)))
    checking, malformed = _peak(lambda: ToolCall("c", "t", text).has_malformed_arguments())

    assert malformed is False
    assert checking / necessary < 1.2
