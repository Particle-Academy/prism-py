"""The private-address opt-out allows private addresses, and nothing else.

``<PREFIX>_ALLOW_PRIVATE_PER_CALL_URL`` (or ``allow_private_per_call_url`` on a
custom provider or a ``configure_providers`` entry) lets a per-call url name a
loopback or private address. It used to skip the whole per-call check, so a
per-call url with another scheme, an escaped or backslashed host, or an
invisible character reached the provider, and a transport passed in received
it. The scheme and host-shape rules now hold with the opt-out on; only the
address rules are skipped.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from typing import Any

import pytest

import prism.registry
from prism import HttpRequest, HttpResponse, Prism, PrismError, register_provider
from prism.providers.openai.provider import OpenAI

NAMES = (
    "OPENAI_API_KEY",
    "OPENAI_URL",
    "OPENAI_ALLOW_PRIVATE_PER_CALL_URL",
)


def _no_network(*args: object, **kwargs: object) -> Any:
    raise AssertionError("A test tried to reach the network.")


@pytest.fixture(autouse=True)
def isolated(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(socket, "getaddrinfo", _no_network)
    monkeypatch.setattr(socket.socket, "connect", _no_network)
    for name in NAMES:
        monkeypatch.delenv(name, raising=False)
    saved = dict(prism.registry._FACTORIES)
    saved_policies = dict(prism.registry._POLICIES)
    yield
    prism.registry._FACTORIES.clear()
    prism.registry._FACTORIES.update(saved)
    prism.registry._POLICIES.clear()
    prism.registry._POLICIES.update(saved_policies)


class Recording:
    def __init__(self) -> None:
        self.urls: list[str] = []

    def send(self, request: HttpRequest) -> HttpResponse:
        self.urls.append(request.url)
        return HttpResponse(status=500, body=b'{"error":{"message":"fixture"}}')


SHAPES = {
    "file scheme": ("file:///etc/v1", "scheme_not_allowed"),
    "ftp scheme": ("ftp://127.0.0.1/v1", "scheme_not_allowed"),
    "data scheme": ("data:text/plain,x", "scheme_not_allowed"),
    "escaped host": ("http://%31%32%37.0.0.1/v1", "private_address_refused"),
    "backslash in the authority": (
        "http://127.0.0.1\\@gateway.example/v1",
        "private_address_refused",
    ),
    "zero-width joiner in the host": ("http://local‍host/v1", "private_address_refused"),
    "non-ASCII host": ("http://bücher.example/v1", "non_ascii_host_use_punycode"),
}

ALLOWED = [
    "http://127.0.0.1:8000/v1",
    "http://localhost:8000/v1",
    "http://[::1]:8000/v1",
    "http://10.0.0.5/v1",
]


def _built_in(url: str, transport: Recording) -> BaseException | None:
    try:
        Prism.text().using(
            "openai", "fixture-model", {"url": url, "api_key": "k", "transport": transport}
        ).with_prompt("hi").as_text()
    except Exception as error:
        return error
    return None


@pytest.mark.parametrize("url,code", SHAPES.values(), ids=SHAPES.keys())
def test_the_opt_out_does_not_admit_another_scheme_or_host_shape(
    url: str, code: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_ALLOW_PRIVATE_PER_CALL_URL", "true")
    transport = Recording()

    error = _built_in(url, transport)

    assert isinstance(error, PrismError), url
    assert error.code == code
    assert transport.urls == []


@pytest.mark.parametrize("url,code", SHAPES.values(), ids=SHAPES.keys())
def test_a_custom_provider_with_the_opt_out_is_held_to_the_same_shape(url: str, code: str) -> None:
    built: list[str] = []

    def factory(**config: Any) -> Any:
        built.append(config["url"])
        return OpenAI(**config)

    register_provider("fixture-custom", factory, allow_private_per_call_url=True)
    with pytest.raises(PrismError) as refused:
        Prism.text().using(
            "fixture-custom", "fixture-model", {"url": url, "api_key": "k"}
        ).with_prompt("hi").as_text()

    assert refused.value.code == code
    assert built == []


@pytest.mark.parametrize("url", ALLOWED)
def test_the_opt_out_still_admits_private_addresses(
    url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_ALLOW_PRIVATE_PER_CALL_URL", "true")
    transport = Recording()

    _built_in(url, transport)

    assert transport.urls == [f"{url}/responses"]
