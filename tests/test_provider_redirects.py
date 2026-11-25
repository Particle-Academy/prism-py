"""A provider call never carries its credentials to another origin.

urllib follows a 301, 302 or 303 (and a 307 or 308 on a GET) and re-sends
every header it was given, so a provider endpoint that answers with a redirect
would hand the key -- ``Authorization``, ``x-api-key``, ``OpenAI-Organization``
or any other header -- to wherever ``Location`` points.

The default transports therefore follow a redirect only while it stays on the
request's own origin (scheme, host and port), and refuse any other redirect
with ``redirect_refused`` before the second origin is asked anything, as the
PHP reference and prism-ts do. These cases run the REAL default transports, through
the same entry points an application uses, against loopback servers only:
every other connection is refused by the ``loopback_only`` fixture.
"""

from __future__ import annotations

import socket
import socketserver
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any

import pytest

from prism import Prism, PrismError
from prism.http import HttpRequest, UrllibStreamTransport, UrllibTransport

KEY = "sk-test-redirect-key"
ORGANIZATION = "org-test-redirect"
PROJECT = "proj-test-redirect"
SECRETS = (KEY, ORGANIZATION, PROJECT)

ENVIRONMENT_NAMES = (
    "OPENAI_API_KEY",
    "OPENAI_URL",
    "OPENAI_ORGANIZATION",
    "OPENAI_PROJECT",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_URL",
    "MISTRAL_API_KEY",
    "MISTRAL_URL",
)


class _LoopbackServer(ThreadingHTTPServer):
    """No reverse lookup of the bound address, which is slow for 127.0.0.2."""

    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = str(host)
        self.server_port = int(port)


class Server:
    """A loopback server that records every request, then redirects or answers."""

    def __init__(self, host: str = "127.0.0.1") -> None:
        self.seen: list[tuple[str, str, dict[str, str]]] = []
        self.redirect: tuple[int, str] | None = None
        # How many requests get the redirect before the server answers; None is all.
        self.redirects_left: int | None = None
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _answer(self) -> None:
                outer.seen.append((self.command, self.path, dict(self.headers.items())))
                length = int(self.headers.get("Content-Length") or 0)
                if length:
                    self.rfile.read(length)
                if (
                    outer.redirect is not None
                    and not self.path.startswith("/landed")
                    and (outer.redirects_left is None or outer.redirects_left > 0)
                ):
                    if outer.redirects_left is not None:
                        outer.redirects_left -= 1
                    status, location = outer.redirect
                    self.send_response(status)
                    self.send_header("Location", location)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                body = b'{"error":{"message":"landed"}}'
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            do_GET = _answer  # noqa: N815 - the stdlib handler name
            do_POST = _answer  # noqa: N815 - the stdlib handler name
            do_DELETE = _answer  # noqa: N815 - the stdlib handler name

            def log_message(self, format: str, *args: object) -> None:
                pass

        self._server = _LoopbackServer((host, 0), Handler)
        self.url = f"http://{host}:{self._server.server_port}"
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()


@pytest.fixture(autouse=True)
def loopback_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse every connection that is not to a loopback address."""
    original = socket.socket.connect

    def connect(self: socket.socket, address: Any) -> Any:
        host = address[0] if isinstance(address, tuple) else address
        if not str(host).startswith("127."):
            raise OSError(f"network blocked in tests: {host!r}")
        return original(self, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    for name in ENVIRONMENT_NAMES:
        monkeypatch.delenv(name, raising=False)
    # The servers are loopback and each case passes its url per call, which is
    # refused unless the operator opts in, in CONFIG. These cases are about
    # redirects, not about that refusal (tests/test_per_call_url.py).
    for prefix in ("OPENAI", "ANTHROPIC", "MISTRAL"):
        monkeypatch.setenv(f"{prefix}_ALLOW_PRIVATE_PER_CALL_URL", "true")


@pytest.fixture
def endpoint() -> Iterator[Server]:
    server = Server("127.0.0.1")
    yield server
    server.close()


# Another host, and the same host on another port: both are another origin.
@pytest.fixture(params=["127.0.0.2", "127.0.0.1"], ids=["other-host", "other-port"])
def other(request: pytest.FixtureRequest) -> Iterator[Server]:
    server = Server(request.param)
    yield server
    server.close()


def _config(provider: str, endpoint: Server) -> dict[str, Any]:
    config: dict[str, Any] = {"url": f"{endpoint.url}/v1", "api_key": KEY}
    if provider == "openai":
        config["organization"] = ORGANIZATION
        config["project"] = PROJECT
    return config


def _call(action: Callable[[], Any]) -> None:
    try:
        result = action()
        if hasattr(result, "__iter__") and not isinstance(result, (str, bytes, dict)):
            list(result)
    except PrismError as error:  # the call is expected to fail; the key must not
        assert all(secret not in str(error) for secret in SECRETS)


def _assert_nothing_reached(other: Server) -> None:
    for _, _, headers in other.seen:
        for name, value in headers.items():
            assert all(secret not in value for secret in SECRETS), name
    assert other.seen == []


PROVIDERS = ["openai", "anthropic", "mistral"]


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
@pytest.mark.parametrize("provider", PROVIDERS)
def test_a_text_call_keeps_its_credentials_off_another_origin(
    provider: str, status: int, endpoint: Server, other: Server
) -> None:
    endpoint.redirect = (status, f"{other.url}/elsewhere")

    with pytest.raises(PrismError) as raised:
        Prism.text().using(provider, "fixture-model", _config(provider, endpoint)).with_prompt(
            "hi"
        ).as_text()

    assert raised.value.code == "redirect_refused"
    assert all(secret not in str(raised.value) for secret in SECRETS)
    assert len(endpoint.seen) == 1
    _assert_nothing_reached(other)


@pytest.mark.parametrize("status", [301, 302, 303])
@pytest.mark.parametrize("provider", PROVIDERS)
def test_a_stream_keeps_its_credentials_off_another_origin(
    provider: str, status: int, endpoint: Server, other: Server
) -> None:
    endpoint.redirect = (status, f"{other.url}/elsewhere")

    _call(
        lambda: (
            Prism.text()
            .using(provider, "fixture-model", _config(provider, endpoint))
            .with_prompt("hi")
            .as_stream()
        )
    )

    assert len(endpoint.seen) == 1
    _assert_nothing_reached(other)


@pytest.mark.parametrize("provider", ["openai", "mistral"])
def test_an_embeddings_call_keeps_its_credentials_off_another_origin(
    provider: str, endpoint: Server, other: Server
) -> None:
    endpoint.redirect = (302, f"{other.url}/elsewhere")

    _call(
        lambda: (
            Prism.embeddings()
            .using(provider, "fixture-model", _config(provider, endpoint))
            .from_input("hi")
            .as_embeddings()
        )
    )

    assert len(endpoint.seen) == 1
    _assert_nothing_reached(other)


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_a_get_keeps_its_credentials_off_another_origin(
    status: int, endpoint: Server, other: Server
) -> None:
    endpoint.redirect = (status, f"{other.url}/elsewhere")

    _call(
        lambda: Prism.files().using("openai", _config("openai", endpoint)).get_metadata("file-abc")
    )

    assert len(endpoint.seen) == 1
    assert endpoint.seen[0][0] == "GET"
    _assert_nothing_reached(other)


# -- the transports themselves ----------------------------------------------------


def _request(url: str) -> HttpRequest:
    return HttpRequest(
        method="POST",
        url=url,
        headers={"Authorization": f"Bearer {KEY}", "x-api-key": KEY, "api-key": KEY},
        body=b"{}",
    )


def _refusal(action: Callable[[], Any]) -> str:
    with pytest.raises(PrismError) as raised:
        action()
    assert all(secret not in str(raised.value) for secret in SECRETS)
    return raised.value.code


def test_the_default_transport_refuses_a_cross_origin_redirect(
    endpoint: Server, other: Server
) -> None:
    endpoint.redirect = (302, f"{other.url}/elsewhere")

    code = _refusal(lambda: UrllibTransport().send(_request(f"{endpoint.url}/v1/x")))

    assert code == "redirect_refused"
    _assert_nothing_reached(other)


def test_the_default_stream_transport_refuses_a_cross_origin_redirect(
    endpoint: Server, other: Server
) -> None:
    endpoint.redirect = (302, f"{other.url}/elsewhere")

    code = _refusal(lambda: UrllibStreamTransport().stream(_request(f"{endpoint.url}/v1/x")))

    assert code == "redirect_refused"
    _assert_nothing_reached(other)


def test_a_scheme_change_is_another_origin(endpoint: Server) -> None:
    port = endpoint.url.rsplit(":", 1)[1]
    endpoint.redirect = (302, f"https://127.0.0.1:{port}/landed")

    code = _refusal(lambda: UrllibTransport().send(_request(f"{endpoint.url}/v1/x")))

    assert code == "redirect_refused"
    assert len(endpoint.seen) == 1


@pytest.mark.parametrize("location", ["ftp://127.0.0.1/landed", "file:///etc/passwd"])
def test_a_redirect_to_another_scheme_is_refused(endpoint: Server, location: str) -> None:
    endpoint.redirect = (302, location)

    code = _refusal(lambda: UrllibTransport().send(_request(f"{endpoint.url}/v1/x")))

    assert code == "redirect_refused"
    assert len(endpoint.seen) == 1


@pytest.mark.parametrize("status", [307, 308])
def test_a_same_origin_307_or_308_keeps_the_method_and_body(endpoint: Server, status: int) -> None:
    endpoint.redirect = (status, f"{endpoint.url}/landed")

    response = UrllibTransport().send(_request(f"{endpoint.url}/v1/x"))

    assert response.status == 400
    assert [(method, path) for method, path, _ in endpoint.seen] == [
        ("POST", "/v1/x"),
        ("POST", "/landed"),
    ]
    assert endpoint.seen[1][2].get("Content-Length") == "2"
    assert endpoint.seen[1][2].get("Authorization") == f"Bearer {KEY}"


@pytest.mark.parametrize("status", [301, 302, 303])
def test_a_same_origin_post_redirect_becomes_a_get_without_a_body(
    endpoint: Server, status: int
) -> None:
    endpoint.redirect = (status, f"{endpoint.url}/landed")

    response = UrllibTransport().send(_request(f"{endpoint.url}/v1/x"))

    assert response.status == 400
    assert [(method, path) for method, path, _ in endpoint.seen] == [
        ("POST", "/v1/x"),
        ("GET", "/landed"),
    ]
    assert "Content-Length" not in endpoint.seen[1][2]


def test_more_than_five_same_origin_redirects_are_refused(endpoint: Server) -> None:
    # The reference follows five redirects and refuses the sixth.
    endpoint.redirect = (307, "/v1/again")

    code = _refusal(lambda: UrllibTransport().send(_request(f"{endpoint.url}/v1/x")))

    assert code == "too_many_redirects"
    assert len(endpoint.seen) == 6


def test_the_stream_transport_stops_after_five_redirects(endpoint: Server) -> None:
    endpoint.redirect = (307, "/v1/again")

    code = _refusal(lambda: UrllibStreamTransport().stream(_request(f"{endpoint.url}/v1/x")))

    assert code == "too_many_redirects"
    assert len(endpoint.seen) == 6


def test_a_chain_of_exactly_five_same_origin_redirects_is_followed(endpoint: Server) -> None:
    endpoint.redirect = (307, "/v1/again")
    endpoint.redirects_left = 5

    response = UrllibTransport().send(_request(f"{endpoint.url}/v1/x"))

    assert response.status == 400
    assert len(endpoint.seen) == 6


@pytest.mark.parametrize("provider", PROVIDERS)
def test_a_text_call_stops_after_five_redirects(provider: str, endpoint: Server) -> None:
    endpoint.redirect = (307, "/v1/again")

    code = _refusal(
        lambda: (
            Prism.text()
            .using(provider, "fixture-model", _config(provider, endpoint))
            .with_prompt("hi")
            .as_text()
        )
    )

    assert code == "too_many_redirects"
    assert len(endpoint.seen) == 6


@pytest.mark.parametrize("status", [307, 308])
@pytest.mark.parametrize("provider", PROVIDERS)
def test_a_text_call_follows_a_same_origin_307_or_308(
    provider: str, status: int, endpoint: Server
) -> None:
    endpoint.redirect = (status, f"{endpoint.url}/landed")

    _call(
        lambda: (
            Prism.text()
            .using(provider, "fixture-model", _config(provider, endpoint))
            .with_prompt("hi")
            .as_text()
        )
    )

    assert [method for method, _, _ in endpoint.seen] == ["POST", "POST"]
    assert endpoint.seen[1][1] == "/landed"


def test_a_same_origin_redirect_is_still_followed(endpoint: Server) -> None:
    endpoint.redirect = (303, f"{endpoint.url}/landed")

    response = UrllibTransport().send(_request(f"{endpoint.url}/v1/x"))

    assert response.status == 400
    assert [(method, path) for method, path, _ in endpoint.seen] == [
        ("POST", "/v1/x"),
        ("GET", "/landed"),
    ]


def test_a_relative_redirect_stays_on_the_origin(endpoint: Server) -> None:
    endpoint.redirect = (302, "/landed")

    response = UrllibTransport().send(_request(f"{endpoint.url}/v1/x"))

    assert response.status == 400
    assert [path for _, path, _ in endpoint.seen] == ["/v1/x", "/landed"]


@pytest.mark.parametrize("userinfo", ["evil.example@", "user:secret@"])
def test_a_same_origin_location_with_userinfo_is_refused(endpoint: Server, userinfo: str) -> None:
    host_port = endpoint.url.split("://", 1)[1]
    endpoint.redirect = (302, f"http://{userinfo}{host_port}/landed")

    with pytest.raises(PrismError) as raised:
        UrllibTransport().send(_request(f"{endpoint.url}/v1/x"))

    assert raised.value.code == "redirect_refused"
    assert userinfo.rstrip("@").split(":")[-1] not in str(raised.value)
    assert len(endpoint.seen) == 1
