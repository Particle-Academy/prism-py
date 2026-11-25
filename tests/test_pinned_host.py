"""A guarded fetch connects only to the host it checked.

The guarded media fetch checks a url's host, resolves it once and pins the
connection to the checked addresses. urllib percent-decodes the host it reads,
so a host written with escapes was checked as one name and connected to as
another, outside the pin. Escapes and backslashes in a guarded host are
refused before any lookup, the default transports refuse any url whose host
urllib would decode, and a pinned connection to any other host is refused
rather than resolved.

Name lookups are recorded and refused except for 127.x literals; connections
are recorded and refused except to loopback.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any

import pytest

from prism import HttpRequest, Image, PrismError
from prism._http_deadline import NetworkDeadline
from prism.http import UrllibStreamTransport, UrllibTransport


@pytest.fixture
def network(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    seen: dict[str, list[Any]] = {"lookups": [], "connects": []}
    real_getaddrinfo = socket.getaddrinfo
    real_connect = socket.socket.connect

    def getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        seen["lookups"].append(str(host))
        if isinstance(host, str) and host.startswith("127.") and host.replace(".", "").isdigit():
            return real_getaddrinfo(host, *args, **kwargs)
        raise OSError("name lookup blocked in tests")

    def connect(self: socket.socket, address: Any) -> Any:
        seen["connects"].append(address)
        host = address[0] if isinstance(address, tuple) else address
        if not str(host).startswith("127."):
            raise OSError("network blocked in tests")
        return real_connect(self, address)

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(socket.socket, "connect", connect)
    return seen


@pytest.fixture
def loopback() -> Iterator[int]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            body = b"INTERNAL-ONLY-BYTES"
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield int(server.server_address[1])
    server.shutdown()
    server.server_close()


class AnswersEverything:
    """A resolver that answers any name, as a wildcard record would."""

    def __init__(self, addresses: Sequence[str] = ("93.184.216.34",)) -> None:
        self.addresses = list(addresses)
        self.hosts: list[str] = []

    def resolve(self, host: str) -> Sequence[str]:
        self.hosts.append(host)
        return self.addresses


ESCAPED = {
    "escaped loopback digits": "http://%31%32%37.0.0.1:{port}/x.png",
    "escaped dot": "http://127.0.0.1%2eexample.test:{port}/x.png",
    "one escaped letter": "http://loc%61lhost:{port}/x.png",
    "backslash in the host": "http://127.0.0.1\\public.test:{port}/x.png",
}


@pytest.mark.parametrize("url", ESCAPED.values(), ids=ESCAPED.keys())
def test_a_guarded_fetch_refuses_an_escaped_host_before_any_lookup(
    url: str, loopback: int, network: dict[str, list[Any]]
) -> None:
    resolver = AnswersEverything()
    image = Image.from_url(url.format(port=loopback))

    with pytest.raises(PrismError) as refused:
        image.fetch_public(resolver=resolver)

    assert refused.value.code == "private_address_refused"
    assert resolver.hosts == []
    assert network["lookups"] == []
    assert network["connects"] == []


@pytest.mark.parametrize("call", ["send", "stream"])
def test_the_default_transports_refuse_a_host_urllib_would_decode(
    call: str, loopback: int, network: dict[str, list[Any]]
) -> None:
    request = HttpRequest("GET", f"http://%31%32%37.0.0.1:{loopback}/x.png")
    transport = UrllibTransport() if call == "send" else UrllibStreamTransport()

    with pytest.raises(PrismError) as refused:
        getattr(transport, call)(request)

    assert refused.value.code == "scheme_not_allowed"
    assert "%31" not in str(refused.value)
    assert network["lookups"] == []
    assert network["connects"] == []


def test_a_pinned_connection_to_another_host_is_refused_not_resolved(
    network: dict[str, list[Any]],
) -> None:
    budget = NetworkDeadline(5.0, 5.0, "media.example", ("93.184.216.34",))
    try:
        with pytest.raises(OSError):
            budget.create_connection(("other.example", 80))
    finally:
        budget.close()
    assert network["lookups"] == []
    assert network["connects"] == []


def test_a_plain_guarded_fetch_still_connects_to_its_pin(
    loopback: int, network: dict[str, list[Any]]
) -> None:
    # Control: the pin is still used for the host it was checked for. The
    # resolver says the public name is 127.0.0.1, which the guard refuses, so a
    # public literal stands in: the transport connects to the pinned address.
    image = Image.from_url(f"http://public.test:{loopback}/x.png")
    resolver = AnswersEverything(["93.184.216.34"])

    with pytest.raises(Exception):  # noqa: B017 - the pinned public address is blocked here
        image.fetch_public(resolver=resolver)

    assert resolver.hosts == ["public.test"]
    assert network["lookups"] == []
    assert network["connects"] == [("93.184.216.34", loopback)]
