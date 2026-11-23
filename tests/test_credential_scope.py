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


# A provider name a human cannot read correctly is refused, not trimmed. The
# rule is spelled identically in the reference and prism-ts: non-empty, and no
# whitespace, control (Cc) or format (Cf) character, nor any codepoint of the
# G-36 set the human-plus confirmation guard normalises. Codepoints are written
# with chr() so the test says exactly which character it sends.
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
    "name", ["credential-scope-plain", "my_provider.v2", "acme:edge", "caf" + chr(0xE9)]
)
def test_still_accepts_an_ordinary_custom_provider_name(name: str) -> None:
    replacement = _Stub()
    register_provider(name, lambda **_: replacement)
    assert resolve_provider(name) is replacement
