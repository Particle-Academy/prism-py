"""What checking a long argument list costs."""

from __future__ import annotations

import json
import socket
import tracemalloc
from collections.abc import Callable
from typing import Any

import pytest

from prism import ToolCall


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


def test_checks_a_long_list_in_a_bounded_multiple_of_reading_it() -> None:
    text = '{"a":[' + ",".join(["0"] * 100_000) + "]}"
    reading, _ = _peak(lambda: json.loads(text))
    checking, malformed = _peak(lambda: ToolCall("c", "t", text).has_malformed_arguments())

    assert malformed is False
    assert checking / reading < 4
