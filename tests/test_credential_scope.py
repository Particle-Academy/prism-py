"""A credential is scoped to the endpoint it was configured for.

Environment credentials belong to the environment's URL (or the provider
default when none is configured). A per-call config that points the provider
somewhere else must bring its own key; the environment one is never sent.

Every case enters where an application does -- ``Prism.text().using(name,
model, config)`` -- and inspects what reached the default transport: the URL,
every header and the body, because providers carry the key in different
headers (``Authorization``, ``x-api-key``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from prism import Prism, register_provider, resolve_provider
from prism.http import HttpRequest, HttpResponse, UrllibTransport
from prism.providers.anthropic.provider import Anthropic
from prism.providers.base import Provider
from prism.providers.mistral.provider import Mistral
from prism.providers.openai.provider import OpenAI


@dataclass(frozen=True)
class Spec:
    name: str
    key_env: str
    url_env: str
    default_url: str


PROVIDERS = [
    Spec("openai", "OPENAI_API_KEY", "OPENAI_URL", "https://api.openai.com/v1"),
    Spec("anthropic", "ANTHROPIC_API_KEY", "ANTHROPIC_URL", "https://api.anthropic.com/v1"),
    Spec("mistral", "MISTRAL_API_KEY", "MISTRAL_URL", "https://api.mistral.ai/v1"),
]

ENVIRONMENT_NAMES = [
    "OPENAI_API_KEY",
    "OPENAI_URL",
    "OPENAI_ORGANIZATION",
    "OPENAI_PROJECT",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_URL",
    "MISTRAL_API_KEY",
    "MISTRAL_URL",
]


@dataclass(frozen=True)
class Sent:
    url: str
    wire: str


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> list[Sent]:
    """Capture every request the default transport is asked to send."""
    for name in ENVIRONMENT_NAMES:
        monkeypatch.delenv(name, raising=False)

    captured: list[Sent] = []

    def send(self: UrllibTransport, request: HttpRequest) -> HttpResponse:
        headers = "\n".join(f"{key}: {value}" for key, value in request.headers.items())
        body = request.body.decode("utf-8", "replace") if isinstance(request.body, bytes) else ""
        captured.append(Sent(request.url, f"{request.url}\n{headers}\n{body}"))
        return HttpResponse(status=500, body=b'{"error":{"message":"fixture"}}', headers={})

    monkeypatch.setattr(UrllibTransport, "send", send)
    return captured


def call(name: str, config: dict[str, Any]) -> None:
    try:
        Prism.text().using(name, "fixture-model", config).with_prompt("hi").as_text()
    except Exception:  # the response is a fixture failure; only the request matters
        pass


def with_port(url: str, port: int) -> str:
    scheme, rest = url.split("://", 1)
    host, path = rest.split("/", 1)
    return f"{scheme}://{host}:{port}/{path}"


def trivial_variant(url: str) -> str:
    """Trailing slash added, and scheme + host upper-cased."""
    scheme, rest = url.split("://", 1)
    host, path = rest.split("/", 1)
    return f"{scheme.upper()}://{host.upper()}/{path}/"


@pytest.mark.parametrize("spec", PROVIDERS, ids=lambda spec: spec.name)
class TestCredentialScope:
    def test_does_not_send_the_environment_key_to_a_per_call_url_that_brings_no_key(
        self, spec: Spec, sent: list[Sent], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(spec.key_env, f"env-secret-{spec.name}")
        call(spec.name, {"url": "https://other-host.example/v1"})
        assert sent
        for request in sent:
            assert "other-host.example" in request.url
            assert f"env-secret-{spec.name}" not in request.wire

    def test_sends_the_key_supplied_alongside_a_per_call_url(
        self, spec: Spec, sent: list[Sent], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(spec.key_env, f"env-secret-{spec.name}")
        call(
            spec.name, {"url": "https://other-host.example/v1", "api_key": f"explicit-{spec.name}"}
        )
        assert sent
        for request in sent:
            assert f"explicit-{spec.name}" in request.wire
            assert f"env-secret-{spec.name}" not in request.wire

    def test_keeps_the_environment_key_when_no_url_is_passed(
        self, spec: Spec, sent: list[Sent], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(spec.key_env, f"env-secret-{spec.name}")
        call(spec.name, {})
        assert sent
        for request in sent:
            assert f"env-secret-{spec.name}" in request.wire

    def test_keeps_the_environment_key_when_the_per_call_url_is_the_default(
        self, spec: Spec, sent: list[Sent], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(spec.key_env, f"env-secret-{spec.name}")
        call(spec.name, {"url": spec.default_url})
        assert sent
        for request in sent:
            assert f"env-secret-{spec.name}" in request.wire

    def test_keeps_the_environment_key_when_the_per_call_url_is_the_environment_one(
        self, spec: Spec, sent: list[Sent], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(spec.key_env, f"env-secret-{spec.name}")
        monkeypatch.setenv(spec.url_env, "https://gateway.example/v1")
        call(spec.name, {"url": "https://gateway.example/v1"})
        assert sent
        for request in sent:
            assert f"env-secret-{spec.name}" in request.wire

    def test_keeps_the_environment_key_when_the_per_call_url_differs_only_trivially(
        self, spec: Spec, sent: list[Sent], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(spec.key_env, f"env-secret-{spec.name}")
        call(spec.name, {"url": trivial_variant(spec.default_url)})
        assert sent
        for request in sent:
            assert f"env-secret-{spec.name}" in request.wire

    @pytest.mark.parametrize(
        "variant",
        [
            pytest.param(lambda url: url.replace("https://", "http://", 1), id="scheme"),
            pytest.param(lambda url: with_port(url, 8443), id="port"),
            pytest.param(lambda url: url.replace("/v1", "/v2"), id="path"),
            pytest.param(lambda url: f"{url}?route=elsewhere", id="query"),
            pytest.param(lambda url: url.replace("https://", "https://user@", 1), id="userinfo"),
        ],
    )
    def test_a_different_endpoint_on_the_same_host_drops_the_environment_key(
        self, spec: Spec, sent: list[Sent], monkeypatch: pytest.MonkeyPatch, variant: Any
    ) -> None:
        monkeypatch.setenv(spec.key_env, f"env-secret-{spec.name}")
        call(spec.name, {"url": variant(spec.default_url)})
        assert sent
        for request in sent:
            assert f"env-secret-{spec.name}" not in request.wire

    def test_the_environment_key_is_bound_to_the_environment_url_not_the_default(
        self, spec: Spec, sent: list[Sent], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(spec.key_env, f"env-secret-{spec.name}")
        monkeypatch.setenv(spec.url_env, "https://gateway.example/v1")
        call(spec.name, {"url": spec.default_url})
        assert sent
        for request in sent:
            assert f"env-secret-{spec.name}" not in request.wire

    def test_does_not_send_the_environment_key_elsewhere_when_the_environment_url_is_set(
        self, spec: Spec, sent: list[Sent], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(spec.key_env, f"env-secret-{spec.name}")
        monkeypatch.setenv(spec.url_env, "https://gateway.example/v1")
        call(spec.name, {"url": "https://other-host.example/v1"})
        assert sent
        for request in sent:
            assert f"env-secret-{spec.name}" not in request.wire


def test_openai_account_identifiers_are_not_sent_from_the_environment_to_a_per_call_url(
    sent: list[Sent], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-secret-openai")
    monkeypatch.setenv("OPENAI_ORGANIZATION", "env-org-id")
    monkeypatch.setenv("OPENAI_PROJECT", "env-project-id")
    call("openai", {"url": "https://other-host.example/v1"})
    assert sent
    for request in sent:
        assert "env-org-id" not in request.wire
        assert "env-project-id" not in request.wire


def test_openai_account_identifiers_supplied_alongside_a_per_call_url_are_sent(
    sent: list[Sent], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_ORGANIZATION", "env-org-id")
    call(
        "openai",
        {"url": "https://other-host.example/v1", "api_key": "explicit", "organization": "own-org"},
    )
    assert sent
    for request in sent:
        assert "own-org" in request.wire
        assert "env-org-id" not in request.wire


class _Stub(Provider):
    pass


BUILT_INS = {"openai": OpenAI, "anthropic": Anthropic, "mistral": Mistral}


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Registration is process-wide; keep each case's registrations to itself."""
    import prism.registry as registry

    monkeypatch.setattr(registry, "_FACTORIES", dict(registry._FACTORIES))


@pytest.mark.parametrize("name", ["openai", "OpenAI", "ANTHROPIC", "mistral"])
def test_refuses_to_register_a_provider_under_a_built_in_name(name: str) -> None:
    with pytest.raises(ValueError, match="already registered"):
        register_provider(name, _Stub)

    builtin = name.lower()
    assert isinstance(resolve_provider(builtin, {"api_key": "x"}), BUILT_INS[builtin])


def test_refuses_to_silently_replace_an_already_registered_provider() -> None:
    register_provider("credential-scope-once", _Stub)
    with pytest.raises(ValueError, match="already registered"):
        register_provider("credential-scope-once", _Stub)
    with pytest.raises(ValueError, match="already registered"):
        register_provider("Credential-Scope-Once", _Stub)


def test_replaces_a_provider_when_replacement_is_explicit() -> None:
    replacement = _Stub()
    register_provider("credential-scope-twice", _Stub)
    register_provider("credential-scope-twice", lambda **_: replacement, replace=True)
    assert resolve_provider("credential-scope-twice") is replacement


def test_replacing_a_built_in_explicitly_is_allowed() -> None:
    replacement = _Stub()
    register_provider("openai", lambda **_: replacement, replace=True)
    assert resolve_provider("openai") is replacement


# A provider name is a positive ASCII allow-list,
# ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$, spelled identically in the reference and
# prism-ts. It replaces a forbidden-character test that relied on each
# language's Unicode tables. Look-alikes, padding and invisible codepoints are
# refused, never trimmed or folded. Codepoints are written with chr() so the
# test says exactly which character it sends.
UNREADABLE_NAMES = {
    "empty": "",
    "leading space": " openai",
    "trailing space": "openai ",
    "interior space": "open ai",
    "tab": "openai" + chr(0x09),
    "newline": "openai" + chr(0x0A),
    "NUL": "openai" + chr(0x00),
    "C0 control": "openai" + chr(0x1F),
    "NEL": "openai" + chr(0x85),
    "no-break space": "openai" + chr(0xA0),
    "ideographic space": "open" + chr(0x3000) + "ai",
    "zero-width space": "openai" + chr(0x200B),
    "zero-width joiner": "open" + chr(0x200D) + "ai",
    "BOM prefix": chr(0xFEFF) + "openai",
    "right-to-left override (Cf)": "openai" + chr(0x202E),
    "word joiner (Cf, outside G-36)": "openai" + chr(0x2060),
    "soft hyphen (Cf)": "open" + chr(0xAD) + "ai",
    "lone surrogate": "openai" + chr(0xD800),
    "Cyrillic o homoglyph": chr(0x043E) + "penai",
    "Cyrillic a homoglyph": "open" + chr(0x0430) + "i",
    "full-width": "".join(chr(c) for c in (0xFF4F, 0xFF50, 0xFF45, 0xFF4E, 0xFF41, 0xFF49)),
    "Latin-1 letter": "caf" + chr(0xE9),
    "leading hyphen": "-openai",
    "leading dot": ".openai",
    "leading underscore": "_openai",
    "colon": "acme:edge",
    "slash": "acme/edge",
    "65 characters": "a" * 65,
}


@pytest.mark.parametrize("name", list(UNREADABLE_NAMES.values()), ids=list(UNREADABLE_NAMES))
def test_refuses_a_provider_name_that_is_empty_or_carries_an_invisible_character(name: str) -> None:
    import prism.registry as registry

    with pytest.raises(ValueError, match="not a valid provider name"):
        register_provider(name, _Stub)
    # Refused even when replacement is explicit: replace does not license a name.
    with pytest.raises(ValueError, match="not a valid provider name"):
        register_provider(name, _Stub, replace=True)

    assert name not in registry._FACTORIES
    assert isinstance(resolve_provider("openai", {"api_key": "x"}), OpenAI)


@pytest.mark.parametrize(
    "name", ["credential-scope-plain", "my_provider.v2", "0day", "q", "b" * 64]
)
def test_still_accepts_an_ordinary_custom_provider_name(name: str) -> None:
    replacement = _Stub()
    register_provider(name, lambda **_: replacement)
    assert resolve_provider(name) is replacement


@pytest.mark.parametrize("name", sorted(BUILT_INS))
def test_the_built_in_names_pass_the_allow_list(name: str) -> None:
    # A built-in name reaches the collision check, so the name itself is valid.
    with pytest.raises(ValueError, match="already registered"):
        register_provider(name, _Stub)
    register_provider(name, _Stub, replace=True)


# A url carrying whitespace or a control character anywhere is refused before
# any request, whether it is passed per call or configured in the environment.
# The forbidden set is spelled by codepoint, identically in the reference and
# prism-ts: every Cc control (U+0000..U+001F, U+007F..U+009F), every Unicode
# White_Space codepoint, and the G-36 zero-widths U+200B..U+200D and U+FEFF.
# The error names neither the url nor the key.
DIRTY_URLS = {
    "padded both ends (cs-0016)": " https://api.openai.com/v1 ",
    "leading space": " https://api.openai.com/v1",
    "trailing space": "https://api.openai.com/v1 ",
    "interior space": "https://api.openai.com/v1 x",
    "tab": "https://api.openai.com/v1" + chr(0x09),
    "newline": "https://api.openai.com/v1" + chr(0x0A),
    "carriage return": chr(0x0D) + "https://api.openai.com/v1",
    "NUL": "https://api.openai.com" + chr(0x00) + "/v1",
    "DEL": "https://api.openai.com/v1" + chr(0x7F),
    "C1 control": "https://api.openai.com/v1" + chr(0x9B),
    "NEL": "https://api.openai.com/v1" + chr(0x85),
    "no-break space": chr(0xA0) + "https://api.openai.com/v1",
    "ideographic space": "https://api.openai.com/v1" + chr(0x3000),
    "zero-width space": "https://api.openai.com" + chr(0x200B) + "/v1",
    "BOM prefix": chr(0xFEFF) + "https://api.openai.com/v1",
}

URL_REFUSAL = "url must not contain whitespace or control characters"


def attempt(name: str, config: dict[str, Any]) -> BaseException | None:
    try:
        Prism.text().using(name, "fixture-model", config).with_prompt("hi").as_text()
    except Exception as error:  # the refusal is what is under test
        return error
    return None


@pytest.mark.parametrize("spec", PROVIDERS, ids=lambda spec: spec.name)
@pytest.mark.parametrize("url", list(DIRTY_URLS.values()), ids=list(DIRTY_URLS))
def test_refuses_a_per_call_url_with_whitespace_or_controls_before_any_request(
    spec: Spec, url: str, sent: list[Sent], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(spec.key_env, f"env-secret-{spec.name}")
    error = attempt(spec.name, {"url": url})
    assert sent == []
    assert isinstance(error, ValueError)
    assert URL_REFUSAL in str(error)
    assert "openai.com" not in str(error)
    assert f"env-secret-{spec.name}" not in str(error)


@pytest.mark.parametrize("spec", PROVIDERS, ids=lambda spec: spec.name)
def test_refuses_an_environment_url_with_a_trailing_newline_before_any_request(
    spec: Spec, sent: list[Sent], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(spec.key_env, f"env-secret-{spec.name}")
    monkeypatch.setenv(spec.url_env, "https://gateway.example/v1" + chr(0x0A))
    error = attempt(spec.name, {})
    assert sent == []
    assert isinstance(error, ValueError)
    assert URL_REFUSAL in str(error)
    assert "gateway.example" not in str(error)


def test_refuses_a_padded_url_before_a_custom_factory_runs() -> None:
    called: list[bool] = []

    def factory(**_: Any) -> Provider:
        called.append(True)
        return _Stub()

    register_provider("credential-scope-padded", factory)
    with pytest.raises(ValueError, match=URL_REFUSAL):
        resolve_provider("credential-scope-padded", {"url": " https://configured.example/v1"})
    assert called == []
