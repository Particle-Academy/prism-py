"""A per-call provider url may not choose a private address.

Decision (Wish, 2026-11-24): "users shouldn't be choosing the address for a
single call." A CONFIG url -- ``<PREFIX>_URL`` in the environment, the provider
default, or a ``configure_providers`` entry's ``url`` -- may point anywhere,
including 127.0.0.1. A PER-CALL ``url`` passed to ``using(provider, model,
{"url": ...})`` is checked before the provider is built, so before any request:
a loopback, private, link-local or metadata address is refused unless the
operator turned the opt-out on in CONFIG.

Every case enters where an application does -- ``Prism.text().using(...)`` --
with sockets refused and the default transport patched to record what reached
the wire. A refusal must carry the error code, send nothing, and never echo the
url, host or address.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from typing import Any

import pytest

import prism.registry
from prism import (
    HttpRequest,
    HttpResponse,
    Prism,
    PrismError,
    configure_providers,
    register_provider,
)
from prism.http import UrllibTransport
from prism.providers.openai.provider import OpenAI

PRIVATE_MESSAGE = (
    "A per-call provider url may not point at a loopback, private, link-local or metadata"
    " address. Allow it in the provider config with allow_private_per_call_url."
)
SCHEME_MESSAGE = "A per-call provider url must be an absolute http or https url."

ENVIRONMENT_NAMES = [
    "OPENAI_API_KEY",
    "OPENAI_URL",
    "OPENAI_ORGANIZATION",
    "OPENAI_PROJECT",
    "OPENAI_API_FORMAT",
    "OPENAI_ALLOW_PRIVATE_PER_CALL_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_URL",
    "ANTHROPIC_ALLOW_PRIVATE_PER_CALL_URL",
    "MISTRAL_API_KEY",
    "MISTRAL_URL",
    "MISTRAL_ALLOW_PRIVATE_PER_CALL_URL",
]


def _no_network(*args: object, **kwargs: object) -> Any:
    raise AssertionError("A test tried to reach the network.")


@pytest.fixture(autouse=True)
def isolated(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Network blocked, environment cleared, registry restored."""
    monkeypatch.setattr(socket, "getaddrinfo", _no_network)
    monkeypatch.setattr(socket, "gethostbyname", _no_network)
    monkeypatch.setattr(socket, "create_connection", _no_network)
    monkeypatch.setattr(socket.socket, "connect", _no_network)
    for name in ENVIRONMENT_NAMES:
        monkeypatch.delenv(name, raising=False)

    saved = dict(prism.registry._FACTORIES)
    saved_policies = dict(getattr(prism.registry, "_POLICIES", {}))
    yield
    prism.registry._FACTORIES.clear()
    prism.registry._FACTORIES.update(saved)
    if hasattr(prism.registry, "_POLICIES"):
        prism.registry._POLICIES.clear()
        prism.registry._POLICIES.update(saved_policies)


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every url the default transport was asked to send to."""
    captured: list[str] = []

    def send(self: UrllibTransport, request: HttpRequest) -> HttpResponse:
        captured.append(request.url)
        return HttpResponse(status=500, body=b'{"error":{"message":"fixture"}}', headers={})

    monkeypatch.setattr(UrllibTransport, "send", send)
    return captured


def attempt(name: str, config: dict[str, Any]) -> BaseException | None:
    try:
        Prism.text().using(name, "fixture-model", config).with_prompt("hi").as_text()
    except Exception as error:  # the refusal, or the fixture's 500
        return error
    return None


def assert_refused(error: BaseException | None, sent: list[str], code: str, message: str) -> None:
    assert sent == [], "a refused per-call url must send nothing"
    assert isinstance(error, PrismError), f"expected a PrismError, got {error!r}"
    assert error.code == code
    # The exact message: it names no url, host or address.
    assert str(error) == message


def assert_dispatched(sent: list[str], prefix: str) -> None:
    assert sent, "the call must reach the transport"
    assert all(url.startswith(prefix) for url in sent), sent


REFUSED = [
    "http://127.0.0.1:8000/v1",
    "HTTP://LocalHost:8000/v1",
    "http://localhost./v1",
    "http://2130706433/v1",
    "http://0177.0.0.1/v1",
    "http://0x7f000001/v1",
    "http://0x7f.1/v1",
    "http://127.1/v1",
    "http://[::1]/v1",
    "http://[::ffff:127.0.0.1]/v1",
    "http://[fd00::1]/v1",
    "http://[fe80::1]/v1",
    "http://169.254.169.254/latest",
    "http://metadata.google.internal/computeMetadata/v1",
    "http://METADATA.GOOGLE.INTERNAL./x",
    "http://0.0.0.0/v1",
    "http://10.0.0.5/v1",
    "http://172.16.0.1/v1",
    "http://192.168.1.10/v1",
    "http://100.100.100.200/x",
    "http://evil@127.0.0.1/v1",
    "http://user:pass@10.0.0.5/v1",
    "http://[64:ff9b::a9fe:a9fe]/",
]


@pytest.mark.parametrize("url", REFUSED)
def test_a_per_call_private_url_is_refused_before_any_request(url: str, sent: list[str]) -> None:
    error = attempt("openai", {"url": url, "api_key": "sk-test"})
    assert_refused(error, sent, "private_address_refused", PRIVATE_MESSAGE)


@pytest.mark.parametrize("name", ["anthropic", "mistral"])
def test_every_built_in_refuses_a_per_call_private_url(name: str, sent: list[str]) -> None:
    error = attempt(name, {"url": "http://127.0.0.1:8000/v1", "api_key": "sk-test"})
    assert_refused(error, sent, "private_address_refused", PRIVATE_MESSAGE)


def test_a_per_call_url_that_is_not_http_is_refused(sent: list[str]) -> None:
    error = attempt("openai", {"url": "ftp://example.com/v1", "api_key": "sk-test"})
    assert_refused(error, sent, "scheme_not_allowed", SCHEME_MESSAGE)


HOST_MESSAGE = (
    "A per-call provider url host may hold only letters, digits, hyphens and dots,"
    " or an IPv6 address in brackets."
)

# A host is judged as written. Some HTTP clients percent-decode a host before
# they connect, so the guard never decodes: an escape, or any character a
# hostname or address literal may not hold, is refused, and so is a backslash
# in the authority, which URL parsers split in different places.
ESCAPED_OR_ILLEGAL = [
    "http://loc%61lhost:8080/admin",
    "http://LOC%41LHOST/v1",
    "http://localhost%2e/v1",
    "http://127.0.0.1%2e/v1",
    "http://127.0.0.1%00.evil.com/v1",
    "http://%31%32%37.0.0.1/v1",
    "http://169.254.169.254%2e/latest/meta-data/",
    "http://metadata.google.internal%2e/x",
    "https://gateway%2eexample/v1",
    "https://gate_way.example/v1",
    "http://127.0.0.1\\@gateway.example/v1",
    "http://gateway.example\\@127.0.0.1/v1",
    "http://[fe80::1%25eth0]/v1",
    "http://[::%31]/v1",
]


@pytest.mark.parametrize("url", ESCAPED_OR_ILLEGAL)
def test_a_per_call_host_with_an_escape_or_illegal_character_is_refused(
    url: str, sent: list[str]
) -> None:
    error = attempt("openai", {"url": url, "api_key": "sk-test"})
    assert_refused(error, sent, "private_address_refused", HOST_MESSAGE)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1./v1",
        "http://0x7f000001./v1",
        "http://localhost../v1",
        "http://169.254.169.254./latest",
    ],
)
def test_a_trailing_dot_is_read_as_the_host_the_client_contacts(url: str, sent: list[str]) -> None:
    error = attempt("openai", {"url": url, "api_key": "sk-test"})
    assert_refused(error, sent, "private_address_refused", PRIVATE_MESSAGE)


def test_an_empty_written_host_is_a_scheme_error(sent: list[str]) -> None:
    error = attempt("openai", {"url": "http:///127.0.0.1/v1", "api_key": "sk-test"})
    assert_refused(error, sent, "scheme_not_allowed", SCHEME_MESSAGE)


@pytest.mark.parametrize(
    ("url", "prefix"),
    [
        ("https://gateway.example./v1", "https://gateway.example./v1"),
        ("https://bücher.example/v1", "https://b"),
    ],
)
def test_a_public_host_with_a_trailing_dot_or_international_name_dispatches(
    url: str, prefix: str, sent: list[str]
) -> None:
    attempt("openai", {"url": url, "api_key": "sk-test"})
    assert_dispatched(sent, prefix)


def test_a_per_call_public_url_dispatches(sent: list[str]) -> None:
    attempt("openai", {"url": "https://gateway.example/v1", "api_key": "sk-test"})
    assert_dispatched(sent, "https://gateway.example/v1/")


def test_a_configured_private_url_with_no_per_call_url_dispatches(
    sent: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_URL", "http://127.0.0.1:1234/v1")
    attempt("openai", {"api_key": "sk-test"})
    assert_dispatched(sent, "http://127.0.0.1:1234/v1/")


@pytest.mark.parametrize("url", ["http://127.0.0.1:1234/v1", "http://127.0.0.1:1234/v1/"])
def test_a_per_call_url_equal_to_the_configured_url_dispatches(
    url: str, sent: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_URL", "http://127.0.0.1:1234/v1")
    attempt("openai", {"url": url, "api_key": "sk-test"})
    assert_dispatched(sent, url.rstrip("/"))


def test_a_per_call_private_url_other_than_the_configured_one_is_refused(
    sent: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_URL", "http://127.0.0.1:1234/v1")
    error = attempt("openai", {"url": "http://127.0.0.1:1235/v1", "api_key": "sk-test"})
    assert_refused(error, sent, "private_address_refused", PRIVATE_MESSAGE)


@pytest.mark.parametrize("value", ["true", "TRUE", "(true)"])
def test_the_config_opt_out_lets_a_per_call_private_url_dispatch(
    value: str, sent: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_ALLOW_PRIVATE_PER_CALL_URL", value)
    attempt("openai", {"url": "http://127.0.0.1:9/v1", "api_key": "sk-test"})
    assert_dispatched(sent, "http://127.0.0.1:9/v1/")


@pytest.mark.parametrize("value", ["1", "yes", "on", "false", "", "null"])
def test_an_opt_out_that_is_not_boolean_true_stays_off(
    value: str, sent: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_ALLOW_PRIVATE_PER_CALL_URL", value)
    error = attempt("openai", {"url": "http://127.0.0.1:9/v1", "api_key": "sk-test"})
    assert_refused(error, sent, "private_address_refused", PRIVATE_MESSAGE)


def test_one_providers_opt_out_does_not_open_another(
    sent: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_ALLOW_PRIVATE_PER_CALL_URL", "true")
    error = attempt("openai", {"url": "http://127.0.0.1:9/v1", "api_key": "sk-test"})
    assert_refused(error, sent, "private_address_refused", PRIVATE_MESSAGE)


def test_a_per_call_opt_out_key_is_ignored(sent: list[str]) -> None:
    error = attempt(
        "openai",
        {"url": "http://127.0.0.1:9/v1", "api_key": "sk-test", "allow_private_per_call_url": True},
    )
    assert_refused(error, sent, "private_address_refused", PRIVATE_MESSAGE)


def compatible(**fields: Any) -> dict[str, Any]:
    return {"driver": "openai-compatible", "url": "http://127.0.0.1:1234/v1", **fields}


def test_a_compatible_entry_on_a_private_address_dispatches(sent: list[str]) -> None:
    configure_providers({"local-llm": compatible()})
    attempt("local-llm", {})
    assert_dispatched(sent, "http://127.0.0.1:1234/v1/")


def test_a_compatible_entry_refuses_a_per_call_private_url(sent: list[str]) -> None:
    configure_providers({"local-llm": compatible()})
    error = attempt("local-llm", {"url": "http://10.0.0.5:8000/v1"})
    assert_refused(error, sent, "private_address_refused", PRIVATE_MESSAGE)


def test_a_compatible_entry_accepts_its_own_url_per_call(sent: list[str]) -> None:
    configure_providers({"local-llm": compatible()})
    attempt("local-llm", {"url": "http://127.0.0.1:1234/v1/"})
    assert_dispatched(sent, "http://127.0.0.1:1234/v1/")


def test_a_compatible_entry_with_the_opt_out_lets_a_per_call_private_url_dispatch(
    sent: list[str],
) -> None:
    configure_providers({"local-llm": compatible(allow_private_per_call_url=True)})
    attempt("local-llm", {"url": "http://10.0.0.5:8000/v1"})
    assert_dispatched(sent, "http://10.0.0.5:8000/v1/")


def test_a_compatible_entry_keeps_its_absolute_url_refusal(sent: list[str]) -> None:
    configure_providers({"local-llm": compatible()})
    error = attempt("local-llm", {"url": "ftp://example.com/v1"})
    assert sent == []
    assert isinstance(error, ValueError)
    assert "not a valid provider url" in str(error)


@pytest.mark.parametrize("value", ["true", 1, "yes"])
def test_a_compatible_opt_out_that_is_not_boolean_is_refused_at_configuration(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="allow_private_per_call_url"):
        configure_providers({"local-llm": compatible(allow_private_per_call_url=value)})
    assert "local-llm" not in prism.registry._FACTORIES


def _custom_factory(built: list[str]) -> Any:
    def factory(**config: Any) -> OpenAI:
        built.append("built")
        return OpenAI(url=config["url"], api_key="sk-test")

    return factory


def test_a_custom_provider_refuses_a_per_call_private_url_before_it_is_built(
    sent: list[str],
) -> None:
    built: list[str] = []
    register_provider("custom-gateway", _custom_factory(built))
    error = attempt("custom-gateway", {"url": "http://127.0.0.1:9/v1"})
    assert_refused(error, sent, "private_address_refused", PRIVATE_MESSAGE)
    assert built == []


def test_a_custom_provider_registered_with_the_opt_out_dispatches(sent: list[str]) -> None:
    built: list[str] = []
    register_provider("custom-gateway", _custom_factory(built), allow_private_per_call_url=True)
    attempt("custom-gateway", {"url": "http://127.0.0.1:9/v1"})
    assert built == ["built"]
    assert_dispatched(sent, "http://127.0.0.1:9/v1/")


def test_replacing_a_built_in_drops_its_configured_url_exemption(
    sent: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_URL", "http://127.0.0.1:1234/v1")
    built: list[str] = []
    register_provider("openai", _custom_factory(built), replace=True)
    error = attempt("openai", {"url": "http://127.0.0.1:1234/v1"})
    assert_refused(error, sent, "private_address_refused", PRIVATE_MESSAGE)
    assert built == []


# -- the helper on its own: every spelling a resolver would read as an address --

HELPER_PRIVATE = {
    "trailing dots": "http://127.0.0.1../v1",
    "hex label alone is zero": "http://0x/v1",
    "octal and hex mixed": "http://0177.0x0.0.01/v1",
    "two labels": "http://10.1/v1",
    "three labels": "http://192.168.257/v1",
    "five labels": "http://1.2.3.4.5/v1",
    "label 08 is neither octal nor decimal": "http://08.8.8.8/v1",
    "label over 255": "http://256.1.1.1/v1",
    "last label too big for its width": "http://1.16777216/v1",
    "over 32 bits": "http://4294967296/v1",
    "a very long label": "http://" + "9" * 400 + "/v1",
    "numeric last label after a name": "http://example.123/v1",
    "full-width digits": "http://" + chr(0xFF11) + chr(0xFF12) + chr(0xFF17) + ".0.0.1/v1",
    "ideographic full stop": "http://localhost" + chr(0x3002) + "/v1",
    "subdomain of localhost": "http://api.localhost/v1",
    "localhost.localdomain": "http://localhost.localdomain/v1",
    "ip6-localhost": "http://ip6-localhost/v1",
    "ip6-loopback": "http://ip6-loopback/v1",
    "bare metadata": "http://metadata/v1",
    "instance-data": "http://instance-data/v1",
    "unspecified IPv6": "http://[::]/v1",
    "scoped IPv6": "http://[fe80::1%25eth0]/v1",
    "6to4 of loopback": "http://[2002:7f00:1::]/v1",
    "an undecodable name": "http://" + chr(0x00DF) * 70 + ".example/v1",
}

HELPER_SCHEME = {
    "no scheme": "//example.com/v1",
    "no host": "http:///v1",
    "only dots": "http://.../v1",
    "long s is not s": "http" + chr(0x017F) + "://example.com/v1",
    "javascript": "javascript:alert(1)",
    "bad port": "http://example.com:99999/v1",
}

HELPER_ALLOWED = [
    "https://gateway.example/v1",
    "HTTPS://API.OPENAI.COM/v1",
    "http://8.8.8.8/v1",
    "http://0x08080808/v1",
    "http://[2606:4700:4700::1111]/v1",
    "http://[64:ff9b::808:808]/v1",
    "https://user:pass@gateway.example/v1",
    "https://localhost.example/v1",
    "https://1.2.3.4.example/v1",
]


@pytest.mark.parametrize("url", list(HELPER_PRIVATE.values()), ids=list(HELPER_PRIVATE))
def test_the_helper_refuses_every_private_spelling(url: str) -> None:
    from prism.per_call_url import assert_per_call_url

    with pytest.raises(PrismError) as caught:
        assert_per_call_url(url)
    assert caught.value.code == "private_address_refused"
    assert str(caught.value) == PRIVATE_MESSAGE


@pytest.mark.parametrize("url", list(HELPER_SCHEME.values()), ids=list(HELPER_SCHEME))
def test_the_helper_refuses_a_url_that_is_not_absolute_http(url: str) -> None:
    from prism.per_call_url import assert_per_call_url

    with pytest.raises(PrismError) as caught:
        assert_per_call_url(url)
    assert caught.value.code == "scheme_not_allowed"
    assert str(caught.value) == SCHEME_MESSAGE


@pytest.mark.parametrize("url", HELPER_ALLOWED)
def test_the_helper_allows_a_public_address_or_name(url: str) -> None:
    from prism.per_call_url import assert_per_call_url

    assert_per_call_url(url)


def test_the_built_in_defaults_the_exemption_uses_are_the_providers_own() -> None:
    from prism.providers.anthropic.provider import DEFAULT_URL as ANTHROPIC
    from prism.providers.mistral.provider import DEFAULT_URL as MISTRAL
    from prism.providers.openai.provider import DEFAULT_URL as OPENAI_DEFAULT

    for key, default in (
        ("openai", OPENAI_DEFAULT),
        ("anthropic", ANTHROPIC),
        ("mistral", MISTRAL),
    ):
        assert prism.registry._built_in_policy(key).configured_url() == default
