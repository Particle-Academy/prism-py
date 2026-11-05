"""Resolver deadline, address binding and native TLS verification controls."""

from __future__ import annotations

import socket
import ssl
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from prism.http import HttpRequest, UrllibTransport
from prism.value_objects.media_file import Image


@pytest.fixture
def owned_server() -> Iterator[ThreadingHTTPServer]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *args: object) -> None:
            pass

        def do_GET(self) -> None:
            self.send_response(302 if self.path == "/redirect" else 200)
            if self.path == "/redirect":
                self.send_header("Location", "/ok")
            self.end_headers()
            self.wfile.write(b"ok")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_dns_wait_is_part_of_total_deadline(
    owned_server: ThreadingHTTPServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = socket.getaddrinfo
    completed = threading.Event()

    def delayed(*args: Any, **kwargs: Any) -> Any:
        try:
            time.sleep(0.2)
            return original(*args, **kwargs)
        finally:
            completed.set()

    monkeypatch.setattr(socket, "getaddrinfo", delayed)
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        UrllibTransport().send(
            HttpRequest("GET", f"http://127.0.0.1:{owned_server.server_port}/ok", timeout=0.05)
        )
    elapsed = time.monotonic() - start
    assert elapsed < 0.15
    assert completed.wait(1)  # The abandoned daemon ends; it never sends a request.


def test_followed_redirect_uses_remaining_budget(owned_server: ThreadingHTTPServer) -> None:
    response = UrllibTransport().send(
        HttpRequest("GET", f"http://127.0.0.1:{owned_server.server_port}/redirect", timeout=1)
    )
    assert response.status == 200 and response.body == b"ok"


def test_ipv6_refusal_falls_back_to_ipv4_within_budget(
    owned_server: ThreadingHTTPServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    port = owned_server.server_port
    records = [
        (socket.AF_INET6, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("::1", port, 0, 0)),
        (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("127.0.0.1", port)),
    ]
    monkeypatch.setattr(socket, "getaddrinfo", lambda *_args: records)
    original_connect = socket.socket.connect
    attempted: list[str] = []

    def connect(connection: socket.socket, address: Any) -> None:
        attempted.append(str(address[0]))
        if connection.family == socket.AF_INET6:
            raise ConnectionRefusedError("Owned simulated IPv6 refusal.")
        original_connect(connection, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    response = UrllibTransport().send(HttpRequest("GET", f"http://127.0.0.1:{port}/ok", timeout=1))
    assert response.body == b"ok"
    assert attempted == ["::1", "127.0.0.1"]


def test_guard_connects_only_to_its_single_validated_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RebindingResolver:
        def __init__(self) -> None:
            self.calls = 0

        def resolve(self, _host: str) -> Sequence[str]:
            self.calls += 1
            return ["93.184.216.34"] if self.calls == 1 else ["127.0.0.1"]

    resolver = RebindingResolver()
    attempted: list[str] = []
    lookups: list[str] = []

    def unexpected_lookup(host: str, *_args: Any) -> Any:
        lookups.append(host)
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("127.0.0.1", 80))]

    def refuse_socket(_socket: socket.socket, address: Any) -> None:
        attempted.append(str(address[0]))
        raise ConnectionRefusedError("Owned connection observer; no packet sent.")

    monkeypatch.setattr(socket, "getaddrinfo", unexpected_lookup)
    monkeypatch.setattr(socket.socket, "connect", refuse_socket)
    monkeypatch.setenv("http_proxy", "http://synthetic-proxy.invalid:8000")
    with pytest.raises(urllib.error.URLError):
        Image.from_url("http://public.test/image").fetch_public(resolver=resolver)
    assert resolver.calls == 1
    assert lookups == []
    assert attempted == ["93.184.216.34"]


def test_tls_keeps_original_sni_and_rejects_wrong_hostname(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = Path(__file__).parent / "fixtures/tls"
    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_context.load_cert_chain(fixture / "fixture.crt", fixture / "fixture.key")
    names: list[str | None] = []
    server_context.set_servername_callback(lambda _socket, name, _context: names.append(name))
    trusted = ssl.create_default_context(cafile=str(fixture / "fixture.crt"))
    assert trusted.check_hostname and trusted.verify_mode == ssl.CERT_REQUIRED
    monkeypatch.setattr(ssl, "_create_default_https_context", lambda: trusted)
    monkeypatch.setattr(urllib.request, "getproxies", lambda: {})
    requests: list[str | None] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *args: object) -> None:
            pass

        def do_GET(self) -> None:
            requests.append(self.headers.get("Host"))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"verified")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    server.socket = server_context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    original = socket.getaddrinfo

    def lookup(host: str, port: int, *_args: Any) -> Any:
        assert host in ("fixture.test", "wrong.test")
        return original("127.0.0.1", port, 0, socket.SOCK_STREAM)

    monkeypatch.setattr(socket, "getaddrinfo", lookup)
    try:
        response = UrllibTransport().send(
            HttpRequest("GET", f"https://fixture.test:{server.server_port}/", timeout=2)
        )
        assert response.body == b"verified"
        with pytest.raises(urllib.error.URLError) as rejected:
            UrllibTransport().send(
                HttpRequest("GET", f"https://wrong.test:{server.server_port}/", timeout=2)
            )
        assert isinstance(rejected.value.reason, ssl.SSLCertVerificationError)
        assert rejected.value.reason.verify_code == 62  # X509 hostname mismatch.
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert names == ["fixture.test", "wrong.test"]
    assert requests == [f"fixture.test:{server.server_port}"]
    assert trusted.check_hostname and trusted.verify_mode == ssl.CERT_REQUIRED
