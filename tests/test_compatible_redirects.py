"""A compatible provider's key never follows a redirect to another host.

urllib re-sends the Authorization header to wherever a 301, 302 or 303 points,
re-issuing a POST as a GET. Following one would hand a named entry's key to a
host nobody configured. These cases run the REAL default transport against two
loopback servers: the configured endpoint answers with a redirect to the
second, and the second must never be asked anything.
"""

from __future__ import annotations

from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

import prism.registry
from prism import Prism, PrismError, configure_providers

KEY = "sk-test-redirect"


class Server:
    def __init__(self, redirect_to: str | None = None) -> None:
        self.seen: list[tuple[str, str, str | None]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _answer(self) -> None:
                outer.seen.append((self.command, self.path, self.headers.get("Authorization")))
                length = int(self.headers.get("Content-Length") or 0)
                if length:
                    self.rfile.read(length)
                if redirect_to is not None:
                    self.send_response(302)
                    self.send_header("Location", redirect_to)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                body = b'{"error":{"message":"thief"}}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            do_GET = _answer  # noqa: N815 - the stdlib handler name
            do_POST = _answer  # noqa: N815 - the stdlib handler name

            def log_message(self, format: str, *args: object) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_port}"
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()


@pytest.fixture
def servers(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Server, Server]]:
    for name in ("OPENAI_API_KEY", "OPENAI_URL", "OPENAI_ORGANIZATION", "OPENAI_PROJECT"):
        monkeypatch.delenv(name, raising=False)
    thief = Server()
    endpoint = Server(redirect_to=f"{thief.url}/steal")
    saved = dict(prism.registry._FACTORIES)
    try:
        configure_providers(
            {
                "my-vllm": {
                    "driver": "openai-compatible",
                    "url": f"{endpoint.url}/v1",
                    "api_key": KEY,
                }
            }
        )
        yield endpoint, thief
    finally:
        prism.registry._FACTORIES.clear()
        prism.registry._FACTORIES.update(saved)
        endpoint.close()
        thief.close()


def test_a_text_call_refuses_a_cross_origin_redirect(servers: tuple[Server, Server]) -> None:
    endpoint, thief = servers

    with pytest.raises(PrismError) as raised:
        Prism.text().using("my-vllm", "m").with_prompt("hi").as_text()

    assert raised.value.code == "redirect_refused"
    assert [(method, path) for method, path, _ in endpoint.seen] == [
        ("POST", "/v1/chat/completions")
    ]
    assert endpoint.seen[0][2] == f"Bearer {KEY}"
    assert thief.seen == []
    assert KEY not in str(raised.value)


def test_a_stream_refuses_a_cross_origin_redirect(servers: tuple[Server, Server]) -> None:
    endpoint, thief = servers

    with pytest.raises(PrismError) as raised:
        list(Prism.text().using("my-vllm", "m").with_prompt("hi").as_stream())

    assert raised.value.code == "redirect_refused"
    assert KEY not in str(raised.value)

    assert [(method, path) for method, path, _ in endpoint.seen] == [
        ("POST", "/v1/chat/completions")
    ]
    assert thief.seen == []
