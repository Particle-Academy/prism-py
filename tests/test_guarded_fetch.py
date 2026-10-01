"""Behavioral evidence supplements the cross-language refusal table."""

from collections.abc import Iterator, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from prism import ErrorCode, HttpRequest, HttpResponse, Image, PrismError, UrllibTransport


class Resolver:
    def __init__(self, addresses: Sequence[str] = ("93.184.216.34",)) -> None:
        self.addresses = addresses
        self.hosts: list[str] = []

    def resolve(self, host: str) -> Sequence[str]:
        self.hosts.append(host)
        return self.addresses


class FakeTransport:
    def __init__(self, responses: Sequence[HttpResponse] = ()) -> None:
        self.sent: list[HttpRequest] = []
        self.responses = iter(responses)

    def send(self, request: HttpRequest) -> HttpResponse:
        self.sent.append(request)
        assert not request.follow_redirects
        return next(self.responses, HttpResponse(200, b"image", {"Content-Type": "image/png"}))


@pytest.mark.parametrize(
    "url,code",
    [
        ("file:///etc/passwd", "scheme_not_allowed"),
        ("gopher://public.test/x", "scheme_not_allowed"),
        ("//public.test/x", "scheme_not_allowed"),
        ("http:public.test/x", "scheme_not_allowed"),
        ("https://[bad]/x", "scheme_not_allowed"),
        ("https://public.test:bad/x", "scheme_not_allowed"),
        ("https://public.test/\nx", "scheme_not_allowed"),
        ("http://169.254.169.254/x", "private_address_refused"),
        ("http://127.0.0.1/x", "private_address_refused"),
        ("http://127.1/x", "private_address_refused"),
        ("http://0x7f000001/x", "private_address_refused"),
        ("http://2130706433/x", "private_address_refused"),
        ("http://0177.0.0.1/x", "private_address_refused"),
        ("http://10.0.0.5/x", "private_address_refused"),
        ("http://172.16.0.1/x", "private_address_refused"),
        ("http://192.168.0.1/x", "private_address_refused"),
        ("http://0.1.2.3/x", "private_address_refused"),
        ("http://240.0.0.1/x", "private_address_refused"),
        ("http://[::1]/x", "private_address_refused"),
        ("http://[::]/x", "private_address_refused"),
        ("http://[fd00::1]/x", "private_address_refused"),
        ("http://[fe80::1]/x", "private_address_refused"),
        ("http://[fe80::1%25eth0]/x", "private_address_refused"),
        ("http://[::127.0.0.1]/x", "private_address_refused"),
        ("http://[::7f00:1]/x", "private_address_refused"),
        ("http://[::2]/x", "private_address_refused"),
        ("http://[::ffff:93.184.216.34]/x", "private_address_refused"),
    ],
)
def test_initial_refusal_sends_nothing(url: str, code: str) -> None:
    transport, resolver = FakeTransport(), Resolver()
    with pytest.raises(PrismError) as refused:
        Image.from_url(url).fetch_public(transport=transport, resolver=resolver)
    assert refused.value.code == code
    assert transport.sent == []
    assert resolver.hosts == []


@pytest.mark.parametrize(
    "addresses,code",
    [
        ([], "host_did_not_resolve"),
        (["10.0.0.5"], "private_address_refused"),
        (["93.184.216.34", "10.0.0.5"], "private_address_refused"),
        (["10.0.0.5", "93.184.216.34"], "private_address_refused"),
        (["2001:4860:4860::8888", "::1"], "private_address_refused"),
        (["not-an-ip"], "private_address_refused"),
        (["::7f00:1"], "private_address_refused"),
    ],
)
def test_every_dns_answer_is_checked(addresses: list[str], code: str) -> None:
    transport = FakeTransport()
    with pytest.raises(PrismError) as refused:
        Image.from_url("https://evil.test/x").fetch_public(
            transport=transport, resolver=Resolver(addresses)
        )
    assert refused.value.code == code
    assert transport.sent == []


@pytest.mark.parametrize("url", ["https://93.184.216.34/x", "https://[2001:4860:4860::8888]/x"])
def test_public_literal_fetches_without_dns(url: str) -> None:
    transport, resolver = FakeTransport(), Resolver()
    image = Image.from_url(url)
    assert image.fetch_public(transport=transport, resolver=resolver) is image
    assert resolver.hosts == []
    assert image.raw_content() == b"image"
    assert image.mime_type() == "image/png"


@pytest.mark.parametrize(
    "target", ["http://169.254.169.254/x", "//127.0.0.1/x", "file:///etc/passwd", "http://[bad]/x"]
)
def test_redirect_is_checked_before_target_request(target: str) -> None:
    transport = FakeTransport([HttpResponse(302, b"", {"Location": target})])
    with pytest.raises(PrismError) as refused:
        Image.from_url("https://public.test/x").fetch_public(
            transport=transport, resolver=Resolver()
        )
    assert refused.value.code == "redirect_refused"
    assert [request.url for request in transport.sent] == ["https://public.test/x"]


def test_dns_is_rechecked_for_same_hostname() -> None:
    answers = iter([["93.184.216.34"], ["10.0.0.5"]])

    class RebindingResolver:
        def resolve(self, host: str) -> Sequence[str]:
            return next(answers)

    transport = FakeTransport([HttpResponse(302, b"", {"location": "/next"})])
    with pytest.raises(PrismError) as refused:
        Image.from_url("https://public.test/x").fetch_public(
            transport=transport, resolver=RebindingResolver()
        )
    assert refused.value.code == "redirect_refused"
    assert len(transport.sent) == 1


@pytest.mark.parametrize("answers", [[], ["10.0.0.5"]])
def test_redirect_hostname_must_resolve_publicly(answers: list[str]) -> None:
    class RedirectResolver:
        def resolve(self, host: str) -> Sequence[str]:
            return ["93.184.216.34"] if host == "public.test" else answers

    transport = FakeTransport([HttpResponse(302, b"", {"location": "https://private.test/target"})])
    with pytest.raises(PrismError) as refused:
        Image.from_url("https://public.test/x").fetch_public(
            transport=transport, resolver=RedirectResolver()
        )
    assert refused.value.code == "redirect_refused"
    assert [request.url for request in transport.sent] == ["https://public.test/x"]


@pytest.mark.parametrize("bound", [0, 2, 5])
def test_endless_public_redirect_chain_has_exact_bound_and_code(bound: int) -> None:
    transport = FakeTransport([HttpResponse(302, b"", {"location": "/next"})] * (bound + 2))
    with pytest.raises(PrismError) as refused:
        Image.from_url("https://public.test/x").fetch_public(
            transport=transport, resolver=Resolver(), max_redirects=bound
        )
    assert refused.value.code == "too_many_redirects"
    assert len(transport.sent) == bound + 1


@pytest.mark.parametrize("bound", [0, 2, 5])
def test_exactly_allowed_hops_still_succeed(bound: int) -> None:
    # Discrimination control: refusing every redirect would satisfy the refusal
    # above. Five allowed redirects must reach a sixth, successful request.
    transport = FakeTransport(
        [HttpResponse(302, b"", {"location": f"/{i}"}) for i in range(1, bound + 1)]
    )
    image = Image.from_url("https://public.test/0")
    image.fetch_public(transport=transport, resolver=Resolver(), max_redirects=bound)
    assert len(transport.sent) == bound + 1
    assert image.raw_content() == b"image"


def test_default_hop_allowance_is_five() -> None:
    transport = FakeTransport([HttpResponse(302, b"", {"location": "/next"})] * 7)
    with pytest.raises(PrismError) as refused:
        Image.from_url("https://public.test/x").fetch_public(
            transport=transport, resolver=Resolver()
        )
    assert refused.value.code == "too_many_redirects"
    assert len(transport.sent) == 6


@pytest.mark.parametrize("bound", [-1, 1.5, True])
def test_invalid_bound_never_sends(bound: int) -> None:
    transport = FakeTransport()
    with pytest.raises(ValueError):
        Image.from_url("https://public.test/x").fetch_public(
            transport=transport, resolver=Resolver(), max_redirects=bound
        )
    assert transport.sent == []


def test_public_redirects_preserve_mime_and_refresh_cached_bytes() -> None:
    transport = FakeTransport(
        [
            HttpResponse(302, b"", {"location": "../next"}),
            HttpResponse(307, b"", {"location": "https://other.test/end"}),
            HttpResponse(200, b"new", {"content-type": "application/octet-stream"}),
        ]
    )
    image = Image.from_url("https://public.test/a/start", "image/png")
    image._base64 = "b2xk"
    image.fetch_public(transport=transport, resolver=Resolver())
    assert [request.url for request in transport.sent] == [
        "https://public.test/a/start",
        "https://public.test/next",
        "https://other.test/end",
    ]
    assert image.mime_type() == "image/png"
    assert image.raw_content() == b"new"
    assert image.base64() == "bmV3"


def test_generated_failure_redacts_credentials_and_query() -> None:
    transport = FakeTransport([HttpResponse(403, b"secret")])
    with pytest.raises(PrismError) as failure:
        Image.from_url("https://alice:password@public.test/x?token=SECRET#secret").fetch_public(
            transport=transport, resolver=Resolver()
        )
    assert failure.value.code == "unfetchable_media"
    assert str(failure.value) == "https://public.test/x?[redacted] responded 403."
    assert failure.value.body is None


def test_missing_url_and_no_implicit_or_unguarded_fetch() -> None:
    with pytest.raises(PrismError) as failure:
        Image.from_base64("aGk=").fetch_public()
    assert failure.value.code == ErrorCode.UNFETCHABLE_MEDIA
    assert Image.from_url("http://127.0.0.1/x").raw_content() is None
    assert not hasattr(Image, "fetch")


def test_unrelated_resolver_and_transport_errors_propagate() -> None:
    error = RuntimeError("unrelated failure")

    class BrokenResolver:
        def resolve(self, host: str) -> Sequence[str]:
            raise error

    class BrokenTransport:
        def send(self, request: HttpRequest) -> HttpResponse:
            raise error

    for options in (
        {"resolver": BrokenResolver()},
        {"resolver": Resolver(), "transport": BrokenTransport()},
    ):
        with pytest.raises(RuntimeError) as failure:
            Image.from_url("https://public.test/x").fetch_public(**options)
        assert failure.value is error


@pytest.fixture
def local_server() -> Iterator[tuple[str, list[str]]]:
    requests: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            requests.append(self.path)
            redirect = self.path.startswith("/redirect")
            status = int(self.path.rsplit("/", 1)[1]) if self.path.count("/") == 2 else 302
            self.send_response(status if redirect else 200)
            if redirect:
                self.send_header("Location", "/target")
            self.end_headers()
            self.wfile.write(b"bytes")

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_real_loopback_is_refused_before_request(local_server: tuple[str, list[str]]) -> None:
    url, requests = local_server
    with pytest.raises(PrismError) as refused:
        Image.from_url(url).fetch_public()
    assert refused.value.code == "private_address_refused"
    assert requests == []


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_default_transport_returns_redirect_without_following(
    local_server: tuple[str, list[str]],
    status: int,
) -> None:
    # Exercise urllib itself, through the existing Transport API, not a
    # monkeypatched urlopen. Only the direct transport is used for loopback.
    url, requests = local_server
    response = UrllibTransport().send(
        HttpRequest("GET", url + f"/redirect/{status}", follow_redirects=False)
    )
    assert response.status == status
    assert requests == [f"/redirect/{status}"]
    assert response.headers["location"] == "/target"


def test_existing_transport_default_still_follows_redirects(
    local_server: tuple[str, list[str]],
) -> None:
    url, requests = local_server
    response = UrllibTransport().send(HttpRequest("GET", url + "/redirect"))
    assert response.status == 200
    assert requests == ["/redirect", "/target"]
