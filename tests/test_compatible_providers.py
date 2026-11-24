"""Named OpenAI-compatible providers (G-321 part A).

``configure_providers`` is this library's equivalent of the reference's
``prism.providers.<name>`` config: an operator names an endpoint (vLLM on a
private address, LM Studio on localhost -- a CONFIG url may be private; a
per-call one may not, see test_per_call_url.py) and the provider built for it carries
ONLY that entry's credentials. Nothing from ``OPENAI_*`` or the ``openai``
provider ever reaches it.

Every case enters where an application does -- ``configure_providers`` and then
``Prism.text().using(name, ...)`` -- with the default transport patched and the
network blocked, and inspects everything that reached the wire.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest

import prism.registry
from prism import HttpRequest, HttpResponse, Prism, PrismError, configure_providers
from prism.http import UrllibTransport

ENV = {
    "OPENAI_API_KEY": "sk-test-env-openai",
    "OPENAI_ORGANIZATION": "org-test-env",
    "OPENAI_PROJECT": "proj-test-env",
    "OPENAI_URL": "https://env-openai.test/v1",
    "OPENAI_API_FORMAT": "responses",
}

CHAT_REPLY = (
    b'{"id":"chatcmpl-1","object":"chat.completion","model":"m","choices":[{"index":0,'
    b'"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],'
    b'"usage":{"prompt_tokens":1,"completion_tokens":1}}'
)

RESPONSES_REPLY = (
    b'{"id":"resp_1","object":"response","status":"completed","model":"m","output":[{"id":"m1",'
    b'"type":"message","status":"completed","role":"assistant","content":[{"type":"output_text",'
    b'"text":"hi"}]}],"usage":{"input_tokens":1,"output_tokens":1}}'
)


@dataclass(frozen=True)
class Sent:
    url: str
    headers: dict[str, str]
    body: str

    @property
    def wire(self) -> str:
        headers = "\n".join(f"{key}: {value}" for key, value in self.headers.items())
        return f"{self.url}\n{headers}\n{self.body}"

    def header(self, name: str) -> str | None:
        for key, value in self.headers.items():
            if key.lower() == name.lower():
                return value
        return None


class Recorder:
    def __init__(self) -> None:
        self.sent: list[Sent] = []
        self.status = 200
        self.raise_error: Exception | None = None

    def send(self, request: HttpRequest) -> HttpResponse:
        body = request.body.decode("utf-8") if isinstance(request.body, bytes) else ""
        self.sent.append(Sent(request.url, dict(request.headers), body))
        if self.raise_error is not None:
            raise self.raise_error
        if self.status >= 400:
            return HttpResponse(
                status=self.status, body=b'{"error":{"message":"fixture failure"}}', headers={}
            )
        reply = RESPONSES_REPLY if request.url.endswith("/responses") else CHAT_REPLY
        return HttpResponse(status=200, body=reply, headers={})


def _no_network(*args: object, **kwargs: object) -> Any:
    raise AssertionError("A test tried to reach the network.")


@pytest.fixture(autouse=True)
def isolated(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Network blocked, registry restored, OPENAI_* set to values that must never leak."""
    monkeypatch.setattr(socket, "getaddrinfo", _no_network)
    monkeypatch.setattr(socket, "gethostbyname", _no_network)
    monkeypatch.setattr(socket, "create_connection", _no_network)
    monkeypatch.setattr(socket.socket, "connect", _no_network)
    for name, value in ENV.items():
        monkeypatch.setenv(name, value)

    saved = dict(prism.registry._FACTORIES)
    yield
    prism.registry._FACTORIES.clear()
    prism.registry._FACTORIES.update(saved)


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    recorder = Recorder()

    def send(self: UrllibTransport, request: HttpRequest) -> HttpResponse:
        return recorder.send(request)

    monkeypatch.setattr(UrllibTransport, "send", send)
    return recorder


def entry(**fields: Any) -> dict[str, Any]:
    return {"driver": "openai-compatible", "url": "https://vllm.test/v1", **fields}


def ask(name: str, config: dict[str, Any] | None = None) -> str:
    return Prism.text().using(name, "fixture-model", config).with_prompt("hi").as_text().text


def assert_no_environment_value(sent: Sent) -> None:
    for value in ENV.values():
        if value == "responses":
            continue
        assert value not in sent.wire


# -- credentials ----------------------------------------------------------------


def test_an_entry_sends_only_its_own_key(wire: Recorder) -> None:
    configure_providers({"my-vllm": entry(api_key="sk-test-vllm")})

    assert ask("my-vllm") == "hi"

    [sent] = wire.sent
    assert sent.url == "https://vllm.test/v1/chat/completions"
    assert sent.header("Authorization") == "Bearer sk-test-vllm"
    assert sent.header("OpenAI-Organization") is None
    assert sent.header("OpenAI-Project") is None
    assert_no_environment_value(sent)


def test_an_entry_sends_its_own_organization_and_project(wire: Recorder) -> None:
    configure_providers(
        {
            "my-vllm": entry(
                api_key="sk-test-vllm", organization="org-test-entry", project="proj-test-entry"
            )
        }
    )

    ask("my-vllm")

    [sent] = wire.sent
    assert sent.header("OpenAI-Organization") == "org-test-entry"
    assert sent.header("OpenAI-Project") == "proj-test-entry"
    assert_no_environment_value(sent)


@pytest.mark.parametrize(
    "fields",
    [{}, {"api_key": None}, {"api_key": ""}],
    ids=["absent", "none", "empty"],
)
def test_no_key_means_no_authorization_header_even_with_openai_env_set(
    wire: Recorder, fields: dict[str, Any]
) -> None:
    configure_providers({"my-vllm": entry(**fields)})

    ask("my-vllm")

    [sent] = wire.sent
    assert sent.header("Authorization") is None
    assert "Bearer" not in sent.wire
    assert sent.header("OpenAI-Organization") is None
    assert sent.header("OpenAI-Project") is None
    assert_no_environment_value(sent)


def test_an_entry_at_the_openai_url_still_does_not_inherit_the_openai_key(
    wire: Recorder,
) -> None:
    configure_providers({"proxy": entry(url=ENV["OPENAI_URL"])})

    ask("proxy")

    [sent] = wire.sent
    assert sent.url == "https://env-openai.test/v1/chat/completions"
    assert sent.header("Authorization") is None
    for name in ("OPENAI_API_KEY", "OPENAI_ORGANIZATION", "OPENAI_PROJECT"):
        assert ENV[name] not in sent.wire


def test_a_broken_openai_environment_does_not_reach_a_compatible_provider(
    wire: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_URL", " https://padded.test/v1")
    monkeypatch.setenv("OPENAI_API_FORMAT", "not-a-format")
    configure_providers({"my-vllm": entry(api_key="sk-test-vllm")})

    assert ask("my-vllm") == "hi"
    assert wire.sent[0].url == "https://vllm.test/v1/chat/completions"


def test_a_private_url_is_accepted_with_only_its_own_key(wire: Recorder) -> None:
    configure_providers({"lan": entry(url="http://10.0.0.5:8000/v1", api_key="sk-test-lan")})

    ask("lan")

    [sent] = wire.sent
    assert sent.url == "http://10.0.0.5:8000/v1/chat/completions"
    assert sent.header("Authorization") == "Bearer sk-test-lan"
    assert_no_environment_value(sent)


def test_loopback_is_accepted(wire: Recorder) -> None:
    configure_providers({"lm-studio": entry(url="http://localhost:1234/v1")})

    ask("lm-studio")

    assert wire.sent[0].url == "http://localhost:1234/v1/chat/completions"


# -- per-call overrides ---------------------------------------------------------


def test_a_per_call_url_elsewhere_drops_the_entry_credentials(wire: Recorder) -> None:
    configure_providers(
        {
            "my-vllm": entry(
                api_key="sk-test-vllm", organization="org-test-entry", project="proj-test-entry"
            )
        }
    )

    ask("my-vllm", {"url": "https://elsewhere.test/v1"})

    [sent] = wire.sent
    assert sent.url == "https://elsewhere.test/v1/chat/completions"
    assert sent.header("Authorization") is None
    assert sent.header("OpenAI-Organization") is None
    assert sent.header("OpenAI-Project") is None
    assert "sk-test-vllm" not in sent.wire
    assert_no_environment_value(sent)


def test_a_per_call_url_elsewhere_uses_a_per_call_key(wire: Recorder) -> None:
    configure_providers({"my-vllm": entry(api_key="sk-test-vllm")})

    ask("my-vllm", {"url": "https://elsewhere.test/v1", "api_key": "sk-test-per-call"})

    [sent] = wire.sent
    assert sent.header("Authorization") == "Bearer sk-test-per-call"
    assert "sk-test-vllm" not in sent.wire


def test_a_per_call_url_naming_the_same_endpoint_keeps_the_entry_key(wire: Recorder) -> None:
    configure_providers({"my-vllm": entry(api_key="sk-test-vllm")})

    ask("my-vllm", {"url": "HTTPS://VLLM.test:443/v1/"})

    assert wire.sent[0].header("Authorization") == "Bearer sk-test-vllm"


def test_a_per_call_key_replaces_the_entry_key(wire: Recorder) -> None:
    configure_providers({"my-vllm": entry(api_key="sk-test-vllm")})

    ask("my-vllm", {"api_key": "sk-test-per-call"})

    [sent] = wire.sent
    assert sent.url == "https://vllm.test/v1/chat/completions"
    assert sent.header("Authorization") == "Bearer sk-test-per-call"


@pytest.mark.parametrize(
    "url", ["file:///etc/passwd", "ftp://vllm.test/v1", "/v1", " https://vllm.test/v1"]
)
def test_a_per_call_url_is_held_to_the_same_rule(wire: Recorder, url: str) -> None:
    configure_providers({"my-vllm": entry(api_key="sk-test-vllm")})

    with pytest.raises(ValueError, match="not a valid provider url") as raised:
        ask("my-vllm", {"url": url})

    assert "sk-test-vllm" not in str(raised.value)
    assert wire.sent == []


# -- names ----------------------------------------------------------------------


@pytest.mark.parametrize("name", ["openai", "OpenAI", "OPENAI", "ANTHROPIC", "Mistral"])
@pytest.mark.parametrize("replace", [False, True])
def test_a_built_in_name_is_refused(wire: Recorder, name: str, replace: bool) -> None:
    with pytest.raises(ValueError, match="already registered"):
        configure_providers({name: entry(api_key="sk-test-vllm")}, replace=replace)

    # And the built-in still answers as itself, at its own endpoint.
    Prism.text().using("openai", "m", {"transport": wire}).with_prompt("hi").as_text()
    assert wire.sent[-1].url == "https://env-openai.test/v1/responses"


@pytest.mark.parametrize(
    "name",
    [
        " my-vllm",
        "my-vllm ",
        "my vllm",
        "my-vllm\n",
        "my-vllm\u200b",
        "\u043epenai",  # Cyrillic o
        "\uff4f\uff50\uff45\uff4e\uff41\uff49",  # full-width openai
        "",
        "-vllm",
        "a" * 65,
    ],
)
def test_an_unreadable_name_is_refused(name: str) -> None:
    with pytest.raises(ValueError, match="not a valid provider name"):
        configure_providers({name: entry()})


def test_a_mixed_case_name_colliding_with_nothing_resolves_under_that_exact_name(
    wire: Recorder,
) -> None:
    configure_providers({"My-VLLM": entry(api_key="sk-test-vllm")})

    assert ask("My-VLLM") == "hi"

    with pytest.raises(PrismError) as raised:
        ask("my-vllm")
    assert raised.value.code == "unsupported_provider_action"


def test_a_name_colliding_with_an_earlier_registration_is_refused_unless_replaced(
    wire: Recorder,
) -> None:
    configure_providers({"my-vllm": entry(api_key="sk-test-first")})

    with pytest.raises(ValueError, match="already registered"):
        configure_providers({"MY-VLLM": entry(api_key="sk-test-second")})

    configure_providers(
        {"my-vllm": entry(url="https://second.test/v1", api_key="sk-test-second")}, replace=True
    )
    ask("my-vllm")

    assert wire.sent[-1].url == "https://second.test/v1/chat/completions"
    assert wire.sent[-1].header("Authorization") == "Bearer sk-test-second"


def test_two_entries_colliding_with_each_other_are_refused() -> None:
    with pytest.raises(ValueError, match="already registered"):
        configure_providers({"my-vllm": entry(), "My-Vllm": entry()}, replace=True)


def test_nothing_is_registered_when_any_entry_is_refused(wire: Recorder) -> None:
    with pytest.raises(ValueError, match="not a valid provider url"):
        configure_providers({"good": entry(), "bad": entry(url="file:///etc/passwd")})

    with pytest.raises(PrismError) as raised:
        ask("good")
    assert raised.value.code == "unsupported_provider_action"


# -- driver, url and format -----------------------------------------------------


@pytest.mark.parametrize("driver", ["openai", "OpenAI-Compatible", "anthropic", None, 1])
def test_an_unknown_driver_is_refused(driver: Any) -> None:
    with pytest.raises(ValueError, match="not a supported provider driver"):
        configure_providers({"my-vllm": {**entry(), "driver": driver}})


def test_a_missing_driver_is_refused() -> None:
    with pytest.raises(ValueError, match="not a supported provider driver"):
        configure_providers({"my-vllm": {"url": "https://vllm.test/v1"}})


@pytest.mark.parametrize(
    "url",
    [
        None,
        "",
        "file:///etc/passwd",
        "ftp://vllm.test/v1",
        "/v1",
        "vllm.test/v1",
        "http://",
        "https:///v1",
        " https://vllm.test/v1",
        "https://vllm.test/v1 ",
        "https://vllm.test/\nv1",
        "https://vllm.test/v1\u200b",
        "https://[::1/v1",
        42,
    ],
)
def test_an_unusable_url_is_refused_without_echoing_it(url: Any) -> None:
    with pytest.raises(ValueError, match="not a valid provider url") as raised:
        configure_providers({"my-vllm": entry(url=url, api_key="sk-test-vllm")})

    message = str(raised.value)
    assert "sk-test-vllm" not in message
    assert "vllm.test" not in message
    assert "passwd" not in message


def test_a_missing_url_is_refused() -> None:
    with pytest.raises(ValueError, match="not a valid provider url"):
        configure_providers({"my-vllm": {"driver": "openai-compatible"}})


@pytest.mark.parametrize("api_format", ["completions", "Chat_Completions", "", "messages", 1])
def test_an_unknown_api_format_is_refused(api_format: Any) -> None:
    with pytest.raises(ValueError, match="not a valid api format"):
        configure_providers({"my-vllm": entry(api_format=api_format)})


@pytest.mark.parametrize("fields", [{}, {"api_format": None}], ids=["absent", "none"])
def test_chat_completions_is_the_default_format(wire: Recorder, fields: dict[str, Any]) -> None:
    configure_providers({"my-vllm": entry(**fields)})

    ask("my-vllm")

    assert wire.sent[0].url.endswith("/chat/completions")


def test_responses_is_selectable(wire: Recorder) -> None:
    configure_providers({"my-vllm": entry(api_format="responses", api_key="sk-test-vllm")})

    assert ask("my-vllm") == "hi"

    [sent] = wire.sent
    assert sent.url == "https://vllm.test/v1/responses"
    assert sent.header("Authorization") == "Bearer sk-test-vllm"


@pytest.mark.parametrize("field", ["api_key", "organization", "project"])
def test_a_credential_that_is_not_text_is_refused_without_echoing_it(field: str) -> None:
    with pytest.raises(ValueError, match="must be a string") as raised:
        configure_providers({"my-vllm": entry(**{field: 12345678})})

    assert "12345678" not in str(raised.value)


@pytest.mark.parametrize("value", ["sk-test-vllm\n", "sk-test vllm", "sk-test-vllm\u200b"])
@pytest.mark.parametrize("field", ["api_key", "organization", "project"])
def test_a_credential_with_whitespace_or_controls_is_refused_without_echoing_it(
    field: str, value: str
) -> None:
    # As a header value it would be refused by the HTTP stack, whose message
    # quotes the value: the key, in a log.
    with pytest.raises(ValueError, match="without whitespace or control characters") as raised:
        configure_providers({"my-vllm": entry(**{field: value})})

    assert "sk-test" not in str(raised.value)


def test_a_per_call_credential_is_held_to_the_same_rule(wire: Recorder) -> None:
    configure_providers({"my-vllm": entry()})

    with pytest.raises(ValueError, match="without whitespace or control characters") as raised:
        ask("my-vllm", {"api_key": "sk-test-per-call\r\nX-Injected: 1"})

    assert "sk-test" not in str(raised.value)
    assert wire.sent == []


def test_an_unknown_entry_field_is_refused() -> None:
    # A misspelt key would otherwise be dropped and the provider sent no key.
    with pytest.raises(ValueError, match="not a supported provider entry field"):
        configure_providers({"my-vllm": entry(apiKey="sk-test-vllm")})


def test_an_entry_that_is_not_a_mapping_is_refused() -> None:
    with pytest.raises(ValueError, match="not a supported provider entry"):
        configure_providers({"my-vllm": "https://vllm.test/v1"})  # type: ignore[dict-item]


# -- errors and identity --------------------------------------------------------


def test_an_error_status_never_carries_the_key(wire: Recorder) -> None:
    wire.status = 500
    configure_providers({"my-vllm": entry(api_key="sk-test-vllm")})

    with pytest.raises(PrismError) as raised:
        ask("my-vllm")

    error = raised.value
    assert error.code == "provider_response_error"
    for text in (str(error), repr(error), repr(error.to_dict())):
        assert "sk-test-vllm" not in text


def test_a_transport_failure_never_carries_the_key(wire: Recorder) -> None:
    wire.raise_error = OSError("connection refused")
    configure_providers({"my-vllm": entry(api_key="sk-test-vllm")})

    with pytest.raises(OSError) as raised:
        ask("my-vllm")

    chain: BaseException | None = raised.value
    while chain is not None:
        assert "sk-test-vllm" not in str(chain)
        assert "sk-test-vllm" not in repr(chain)
        chain = chain.__cause__ or chain.__context__


def test_the_entry_name_is_the_provider_key(wire: Recorder) -> None:
    configure_providers({"my-vllm": entry()})

    pending = Prism.text().using("my-vllm", "fixture-model").with_prompt("hi")

    assert pending.provider_key == "my-vllm"
    assert pending.to_request().provider_key == "my-vllm"
