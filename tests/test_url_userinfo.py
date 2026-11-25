"""A url carrying userinfo is refused by the default transports before anything leaves.

urllib does not read userinfo (``user:password@`` before the host): it takes
everything between ``//`` and the first ``/``, ``?`` or ``#`` as the host,
percent-decoded. A url like ``https://user:secret@gateway.example/v1`` would be
looked up as the host name ``user:secret@gateway.example``, so the user name and
password reach the name resolver, and the resulting error repeats them. The
default transports therefore refuse such a url with ``scheme_not_allowed``
before any lookup or connection, as prism-ts does for the urls ``fetch`` cannot
send. A transport passed in still receives the url as given, so it can decide.

Every case enters where an application does (``Prism.text().using(...)``,
``Prism.embeddings()``, a guarded media fetch) over the real default
transports. Name lookups are recorded and refused except for 127.x literals,
and connections are refused except to loopback.
"""

from __future__ import annotations

import socket
import socketserver
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any

import pytest

from prism import HttpRequest, HttpResponse, Prism, PrismError
from prism.http import UrllibStreamTransport, UrllibTransport
from prism.value_objects.media_file import Image

KEY = "sk-test-userinfo-key"
USER = "fixtureuser"
PASSWORD = "fixturesecret"

ENVIRONMENT_NAMES = (
    "OPENAI_API_KEY",
    "OPENAI_URL",
    "OPENAI_ORGANIZATION",
    "OPENAI_PROJECT",
    "OPENAI_ALLOW_PRIVATE_PER_CALL_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_URL",
    "ANTHROPIC_ALLOW_PRIVATE_PER_CALL_URL",
    "MISTRAL_API_KEY",
    "MISTRAL_URL",
    "MISTRAL_ALLOW_PRIVATE_PER_CALL_URL",
)

# Each prefix goes in front of the host. Every one makes urllib read part of it
# as the host name.
USERINFO = {
    "user": f"{USER}@",
    "user and password": f"{USER}:{PASSWORD}@",
    "password only": f":{PASSWORD}@",
    "percent-encoded at sign in the user": f"{USER}%40other@",
    "percent-encoded colon": f"{USER}%3A{PASSWORD}@",
    "two at signs": f"{USER}@{PASSWORD}@",
    "empty userinfo": "@",
    "non-ASCII user": f"{USER}ü@",
}


class _LoopbackServer(ThreadingHTTPServer):
    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = str(host)
        self.server_port = int(port)


class Server:
    """A loopback server that records every request and answers 400, or redirects."""

    def __init__(self) -> None:
        self.seen: list[tuple[str, str, dict[str, str]]] = []
        self.redirect: tuple[int, str] | None = None
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _answer(self) -> None:
                outer.seen.append((self.command, self.path, dict(self.headers.items())))
                length = int(self.headers.get("Content-Length") or 0)
                if length:
                    self.rfile.read(length)
                if outer.redirect is not None and not self.path.startswith("/landed"):
                    status, location = outer.redirect
                    self.send_response(status)
                    self.send_header("Location", location)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                body = b'{"error":{"message":"fixture"}}'
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            do_GET = _answer  # noqa: N815 - the stdlib handler name
            do_POST = _answer  # noqa: N815 - the stdlib handler name

            def log_message(self, format: str, *args: object) -> None:
                pass

        self._server = _LoopbackServer(("127.0.0.1", 0), Handler)
        self.host_port = f"127.0.0.1:{self._server.server_port}"
        self.url = f"http://{self.host_port}"
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()


@pytest.fixture
def lookups(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every host name the process tried to resolve; only 127.x literals resolve."""
    seen: list[str] = []
    real_getaddrinfo = socket.getaddrinfo
    real_connect = socket.socket.connect

    def getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        seen.append(str(host))
        if isinstance(host, str) and host.startswith("127.") and host.replace(".", "").isdigit():
            return real_getaddrinfo(host, *args, **kwargs)
        raise OSError("name lookup blocked in tests")

    def connect(self: socket.socket, address: Any) -> Any:
        host = address[0] if isinstance(address, tuple) else address
        if not str(host).startswith("127."):
            raise OSError("network blocked in tests")
        return real_connect(self, address)

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(socket, "gethostbyname", lambda *_: (_ for _ in ()).throw(OSError()))
    monkeypatch.setattr(socket.socket, "connect", connect)
    for name in ENVIRONMENT_NAMES:
        monkeypatch.delenv(name, raising=False)
    return seen


@pytest.fixture
def endpoint() -> Iterator[Server]:
    server = Server()
    yield server
    server.close()


def _opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    # The loopback server is a private address, which a per-call url may name
    # only when the operator opts in, in CONFIG. These cases are about the
    # transport, not that refusal (tests/test_per_call_url.py).
    for prefix in ("OPENAI", "ANTHROPIC", "MISTRAL"):
        monkeypatch.setenv(f"{prefix}_ALLOW_PRIVATE_PER_CALL_URL", "true")


def _attempt(action: Callable[[], Any]) -> BaseException | None:
    try:
        result = action()
        if hasattr(result, "__iter__") and not isinstance(result, (str, bytes, dict)):
            list(result)
    except Exception as error:
        return error
    return None


def _assert_userinfo_refused(error: BaseException | None, lookups: list[str]) -> None:
    assert isinstance(error, PrismError), f"expected a PrismError, got {type(error).__name__}"
    assert error.code == "scheme_not_allowed"
    message = str(error)
    for fragment in (USER, PASSWORD, "gateway.example", "127.0.0.1", "other"):
        assert fragment not in message
    assert error.__cause__ is None
    assert not any("@" in host for host in lookups), "userinfo reached the name resolver"


PROVIDERS = ["openai", "anthropic", "mistral"]


@pytest.mark.parametrize("prefix", USERINFO.values(), ids=USERINFO.keys())
@pytest.mark.parametrize("provider", PROVIDERS)
def test_a_text_call_with_userinfo_in_its_per_call_url_sends_nothing(
    provider: str,
    prefix: str,
    endpoint: Server,
    lookups: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _opt_in(monkeypatch)
    url = f"http://{prefix}{endpoint.host_port}/v1"

    error = _attempt(
        lambda: Prism.text()
        .using(provider, "fixture-model", {"url": url, "api_key": KEY})
        .with_prompt("hi")
        .as_text()
    )

    _assert_userinfo_refused(error, lookups)
    assert endpoint.seen == []


@pytest.mark.parametrize("prefix", USERINFO.values(), ids=USERINFO.keys())
@pytest.mark.parametrize("provider", PROVIDERS)
def test_a_stream_with_userinfo_in_its_per_call_url_sends_nothing(
    provider: str,
    prefix: str,
    endpoint: Server,
    lookups: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _opt_in(monkeypatch)
    url = f"http://{prefix}{endpoint.host_port}/v1"

    error = _attempt(
        lambda: Prism.text()
        .using(provider, "fixture-model", {"url": url, "api_key": KEY})
        .with_prompt("hi")
        .as_stream()
    )

    _assert_userinfo_refused(error, lookups)
    assert endpoint.seen == []


@pytest.mark.parametrize("provider", ["openai", "mistral"])
def test_an_embeddings_call_with_userinfo_sends_nothing(
    provider: str, endpoint: Server, lookups: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _opt_in(monkeypatch)
    url = f"http://{USER}:{PASSWORD}@{endpoint.host_port}/v1"

    error = _attempt(
        lambda: Prism.embeddings()
        .using(provider, "fixture-model", {"url": url, "api_key": KEY})
        .from_input("hi")
        .as_embeddings()
    )

    _assert_userinfo_refused(error, lookups)
    assert endpoint.seen == []


# A public name: the per-call address check lets it through (it does not resolve
# names), so only the transport stands between the userinfo and the resolver.
PUBLIC = {
    "user": f"https://{USER}@gateway.example/v1",
    "user and password": f"https://{USER}:{PASSWORD}@gateway.example/v1",
    "password only": f"https://:{PASSWORD}@gateway.example/v1",
    "trusted host spelled as userinfo": "https://api.openai.com@gateway.example/v1",
    "percent-encoded at sign in the user": f"https://{USER}%40other@gateway.example/v1",
    "with a port": f"https://{USER}:{PASSWORD}@gateway.example:8443/v1",
}


@pytest.mark.parametrize("url", PUBLIC.values(), ids=PUBLIC.keys())
def test_a_per_call_url_with_userinfo_on_a_public_name_is_never_looked_up(
    url: str, lookups: list[str]
) -> None:
    error = _attempt(
        lambda: Prism.text()
        .using("openai", "fixture-model", {"url": url, "api_key": KEY})
        .with_prompt("hi")
        .as_text()
    )

    _assert_userinfo_refused(error, lookups)
    assert lookups == []


# Spellings the per-call check refuses on its own. Controls: they must stay
# refused before any lookup, under the per-call check's own code.
GUARDED = {
    "backslash before the at sign": f"https://gateway.example\\@{USER}.example/v1",
    "backslash after the at sign": f"https://{USER}@gateway.example\\x/v1",
    "percent-encoded at sign in the host": f"https://{USER}%40gateway.example/v1",
}


@pytest.mark.parametrize("url", GUARDED.values(), ids=GUARDED.keys())
def test_userinfo_confusion_the_per_call_check_refuses_stays_refused(
    url: str, lookups: list[str]
) -> None:
    error = _attempt(
        lambda: Prism.text()
        .using("openai", "fixture-model", {"url": url, "api_key": KEY})
        .with_prompt("hi")
        .as_text()
    )

    assert isinstance(error, PrismError)
    assert error.code == "private_address_refused"
    assert lookups == []


@pytest.mark.parametrize(
    "url",
    [
        "http://{user}:{password}@{host}/v1",
        # urllib percent-decodes the host it reads, so %40 becomes an at sign.
        "http://{user}%40{host}/v1",
        "http://{host}\\@{user}/v1",
    ],
    ids=["userinfo", "percent-encoded at sign", "backslash"],
)
def test_a_configured_url_urllib_would_misread_is_refused_too(
    url: str, endpoint: Server, lookups: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The operator's url is not held to the per-call check, but the default
    # transport cannot send this one either, so it refuses it the same way.
    monkeypatch.setenv(
        "OPENAI_URL", url.format(user=USER, password=PASSWORD, host=endpoint.host_port)
    )
    monkeypatch.setenv("OPENAI_API_KEY", KEY)

    error = _attempt(
        lambda: Prism.text().using("openai", "fixture-model").with_prompt("hi").as_text()
    )

    _assert_userinfo_refused(error, lookups)
    assert endpoint.seen == []


def test_a_url_without_userinfo_is_still_sent(
    endpoint: Server, lookups: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _opt_in(monkeypatch)

    error = _attempt(
        lambda: Prism.text()
        .using("openai", "fixture-model", {"url": f"{endpoint.url}/v1", "api_key": KEY})
        .with_prompt("hi")
        .as_text()
    )

    assert isinstance(error, PrismError)
    assert error.code != "scheme_not_allowed"
    assert len(endpoint.seen) == 1


def test_an_at_sign_in_the_path_or_query_is_still_sent(
    endpoint: Server, lookups: list[str]
) -> None:
    for url in (f"{endpoint.url}/v1/a@b", f"{endpoint.url}/v1?who=a@b", f"{endpoint.url}/v1#a@b"):
        UrllibTransport().send(HttpRequest("POST", url, body=b"{}"))
    assert len(endpoint.seen) == 3


def test_a_transport_passed_in_receives_the_url_as_given(lookups: list[str]) -> None:
    # As in prism-ts: a transport that can send userinfo decides for itself.
    url = f"https://{USER}:{PASSWORD}@gateway.example/v1"
    seen: list[str] = []

    class Recording:
        def send(self, request: HttpRequest) -> HttpResponse:
            seen.append(request.url)
            return HttpResponse(status=500, body=b'{"error":{"message":"fixture"}}')

    _attempt(
        lambda: Prism.text()
        .using("openai", "fixture-model", {"url": url, "api_key": KEY, "transport": Recording()})
        .with_prompt("hi")
        .as_text()
    )

    assert seen == [f"{url}/responses"]
    assert lookups == []


@pytest.mark.parametrize("transport", [UrllibTransport(), UrllibStreamTransport()])
def test_both_default_transports_refuse_userinfo_directly(
    transport: Any, endpoint: Server, lookups: list[str]
) -> None:
    request = HttpRequest("POST", f"http://{USER}:{PASSWORD}@{endpoint.host_port}/v1", body=b"{}")

    if isinstance(transport, UrllibTransport):
        error = _attempt(lambda: transport.send(request))
    else:
        error = _attempt(lambda: transport.stream(request))

    _assert_userinfo_refused(error, lookups)
    assert endpoint.seen == []


LOCATIONS = {
    "user": f"{USER}@",
    "user and password": f"{USER}:{PASSWORD}@",
    "password only": f":{PASSWORD}@",
    "percent-encoded at sign": f"{USER}%40other@",
    "empty userinfo": "@",
}


@pytest.mark.parametrize("prefix", LOCATIONS.values(), ids=LOCATIONS.keys())
@pytest.mark.parametrize("status", [302, 307])
def test_a_same_origin_location_with_userinfo_is_refused_on_a_provider_call(
    status: int,
    prefix: str,
    endpoint: Server,
    lookups: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _opt_in(monkeypatch)
    endpoint.redirect = (status, f"http://{prefix}{endpoint.host_port}/landed")

    error = _attempt(
        lambda: Prism.text()
        .using("openai", "fixture-model", {"url": f"{endpoint.url}/v1", "api_key": KEY})
        .with_prompt("hi")
        .as_text()
    )

    assert isinstance(error, PrismError)
    assert error.code == "redirect_refused"
    assert USER not in str(error) and PASSWORD not in str(error)
    assert len(endpoint.seen) == 1
    assert not any("@" in host for host in lookups)


@pytest.mark.parametrize(
    "location",
    ["http://{user}\\@{host}/landed", "//{user}@{host}/landed", "http://{host}%40{user}/landed"],
    ids=["backslash before the at sign", "scheme-relative", "percent-encoded at sign in the host"],
)
def test_a_location_confusing_userinfo_and_host_is_refused(
    location: str, endpoint: Server, lookups: list[str]
) -> None:
    endpoint.redirect = (302, location.format(user=USER, host=endpoint.host_port))

    error = _attempt(
        lambda: UrllibTransport().send(HttpRequest("POST", f"{endpoint.url}/v1", body=b"{}"))
    )

    assert isinstance(error, PrismError)
    assert error.code == "redirect_refused"
    assert len(endpoint.seen) == 1
    assert not any("@" in host for host in lookups)


class _PublicResolver:
    def resolve(self, host: str) -> list[str]:
        return ["93.184.216.34"]


def test_a_guarded_media_fetch_with_userinfo_never_looks_up_the_userinfo(
    lookups: list[str],
) -> None:
    # The guard checks and pins the real host. urllib would have looked up
    # "fixtureuser@media.example" instead, outside the pin.
    image = Image.from_url(f"https://{USER}@media.example/a.png")

    error = _attempt(lambda: image.fetch_public(resolver=_PublicResolver()))

    _assert_userinfo_refused(error, lookups)
    assert lookups == []
