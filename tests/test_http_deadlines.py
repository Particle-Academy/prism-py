"""Owned loopback measurements of deadlines, independent of provider fakes."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import prism.http as transport
from prism.http import HttpRequest, UrllibStreamTransport, UrllibTransport


@pytest.fixture
def server() -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *args: object) -> None:
            pass

        def do_GET(self) -> None:
            try:
                if self.path == "/headers":
                    time.sleep(0.25)
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                if self.path == "/idle":
                    self.wfile.flush()
                    time.sleep(0.25)
                count = 12 if self.path == "/heartbeat" else 4
                for _ in range(count):
                    self.wfile.write(b"x")
                    self.wfile.flush()
                    time.sleep(0.04)
            except (ConnectionError, OSError):
                pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    http.daemon_threads = True
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{http.server_port}"
    finally:
        http.shutdown()
        http.server_close()
        thread.join()


def test_shipped_deadline_defaults() -> None:
    assert transport.DEFAULT_TIMEOUT == 30
    assert transport.DEFAULT_STREAM_IDLE_TIMEOUT == 60


def test_nonstream_has_total_deadline_despite_heartbeats(server: str) -> None:
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        UrllibTransport().send(HttpRequest("GET", server + "/heartbeat", timeout=0.18))
    assert time.monotonic() - start < 0.45


def test_headers_honor_override(server: str) -> None:
    with pytest.raises((TimeoutError, OSError)):
        UrllibStreamTransport().stream(HttpRequest("GET", server + "/headers", timeout=0.05))


def test_stream_has_no_whole_lifetime_deadline(server: str) -> None:
    response = UrllibStreamTransport().stream(HttpRequest("GET", server + "/body", timeout=0.05))
    start = time.monotonic()
    assert "".join(response.chunks) == "xxxx"
    assert time.monotonic() - start > 0.1


def test_stream_idle_is_distinct_from_header_timeout(
    server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(transport, "DEFAULT_STREAM_IDLE_TIMEOUT", 0.05)
    response = UrllibStreamTransport().stream(HttpRequest("GET", server + "/idle", timeout=1))
    with pytest.raises(TimeoutError):
        list(response.chunks)


def test_zero_total_timeout_allows_completion(server: str) -> None:
    assert (
        UrllibTransport()
        .send(HttpRequest("GET", server + "/body", client_options={"timeout": 0}))
        .body
        == b"xxxx"
    )
