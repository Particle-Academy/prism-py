"""Resolving a provider key to a provider instance."""

from __future__ import annotations

import re
import sys
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, NamedTuple
from urllib.parse import urlsplit

from prism.errors import ErrorCode, PrismError
from prism.per_call_url import assert_per_call_url
from prism.providers.config import (
    UNSET,
    _environment_value,
    assert_clean_url,
    configured_string,
    same_endpoint,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from prism.http import StreamTransport, Transport
    from prism.providers.base import Provider

__all__ = ["configure_providers", "register_provider", "resolve_provider"]

ProviderFactory = Callable[..., "Provider"]

_FACTORIES: dict[str, ProviderFactory] = {}

_BUILT_INS = ("anthropic", "mistral", "openai")


class _UrlPolicy(NamedTuple):
    """How a per-call ``url`` is judged for one provider key.

    ``configured_url`` is the CONFIG url a per-call url may repeat without being
    checked (None: there is none). ``allow_private`` is the operator's opt-out,
    read from CONFIG only. ``validate`` runs first, so a provider's own url
    refusal keeps its error.
    """

    configured_url: Callable[[], str | None]
    allow_private: Callable[[], bool]
    validate: Callable[[object], object] | None = None


def _no_configured_url() -> str | None:
    return None


def _never() -> bool:
    return False


def _always() -> bool:
    return True


_CUSTOM_POLICY = _UrlPolicy(_no_configured_url, _never)
_CUSTOM_OPTED_OUT = _UrlPolicy(_no_configured_url, _always)

# key -> (the factory the policy was registered with, the policy). Paired with
# the factory so a policy never outlives the registration it was made for: a
# key whose factory has since changed falls back to the custom-provider policy.
_POLICIES: dict[str, tuple[ProviderFactory, _UrlPolicy]] = {}


def register_provider(
    key: str,
    factory: ProviderFactory,
    *,
    replace: bool = False,
    allow_private_per_call_url: bool = False,
) -> None:
    """Teach the builder about a provider key.

    A key that is already registered -- a built-in such as ``openai``, or one
    registered earlier, compared case-insensitively -- is refused unless
    ``replace=True`` is passed. Replacing a provider changes what every caller
    of that key gets, process-wide, so it has to be asked for.

    The key itself must be readable: see :func:`_is_valid_provider_name`. An
    invalid key is refused whether or not ``replace`` is set, and never trimmed.

    A custom provider has no configured url, so every per-call ``url`` it is
    given is checked before the factory runs: a loopback, private, link-local or
    metadata address is refused with ``private_address_refused`` unless
    ``allow_private_per_call_url=True`` (strictly ``True``) is passed here. A
    replaced built-in is a custom provider: it loses the built-in's
    configured-url exemption and its environment opt-out.

    :raises ValueError: when the key is not a valid provider name, or is taken
        and ``replace`` is not set (a programming error; it carries no Prism
        error code).
    """
    policy = _CUSTOM_OPTED_OUT if allow_private_per_call_url is True else _CUSTOM_POLICY
    _register(key, factory, policy, replace=replace)


def _register(key: str, factory: ProviderFactory, policy: _UrlPolicy, *, replace: bool) -> None:
    if not _is_valid_provider_name(key):
        raise ValueError(_INVALID_NAME)
    if not replace and _is_registered(key):
        raise ValueError(
            f"The provider '{key}' is already registered. Pass replace=True to replace it."
        )
    _FACTORIES[key] = factory
    _POLICIES[key] = (factory, policy)


_INVALID_NAME = (
    "That is not a valid provider name: a provider name must start with an"
    ' ASCII letter or digit, contain only ASCII letters, digits, ".", "_"'
    ' and "-", and be at most 64 characters long.'
)


# A provider key: ASCII, starting with a letter or digit, at most 64
# characters. A positive allow-list rather than a list of forbidden characters,
# so it needs no Unicode tables and reads the same as the reference's extend()
# and prism-ts registerProvider(). match() with \Z, not $: $ would admit a
# trailing newline. The class is spelled out, so no Unicode table is consulted.
_VALID_PROVIDER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z", re.ASCII)


def _is_valid_provider_name(key: str) -> bool:
    """Whether ``key`` matches ``^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$`` in ASCII.

    A homoglyph (Cyrillic U+043E for the first ``o``), a full-width ``openai``,
    a padded one or one carrying an invisible codepoint is a second key that
    reads as the built-in in a config file or a log, so it is refused, never
    trimmed or folded.
    """
    return isinstance(key, str) and _VALID_PROVIDER_NAME.match(key) is not None


def _ascii_lower(value: str) -> str:
    """ASCII-only case folding, as the reference's strtolower() does."""
    return value.translate(_ASCII_LOWER)


_ASCII_LOWER = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")


def _is_registered(key: str) -> bool:
    name = _ascii_lower(key)
    return name in _BUILT_INS or any(_ascii_lower(known) == name for known in _FACTORIES)


def resolve_provider(key: str, config: dict[str, Any] | None = None) -> Provider:
    """Build the provider registered under ``key``.

    :raises PrismError: with code ``unsupported_provider_action`` for a key no
        provider is registered under.
    """
    factory = _FACTORIES.get(key) or _default_factory(key)

    if factory is None:
        raise PrismError(
            ErrorCode.UNSUPPORTED_PROVIDER_ACTION,
            f"No provider is registered for '{key}'.",
        )

    config = config or {}
    # Before the factory runs, so a custom provider is held to the same rules.
    url = config.get("url")
    assert_clean_url(url)
    if isinstance(url, str):
        _check_per_call_url(url, _url_policy(key, factory))
    return factory(**config)


def _check_per_call_url(url: str, policy: _UrlPolicy) -> None:
    """Refuse a per-call url that chooses a private address.

    Exempt: the url is the provider's configured endpoint (the call is not
    choosing a new address). The opt-out in CONFIG allows a private address
    but not another scheme or a host that cannot be read. A per-call
    ``allow_private_per_call_url`` key is never read.
    """
    if policy.validate is not None:
        policy.validate(url)
    configured = policy.configured_url()
    if configured is not None and same_endpoint(url, configured):
        return
    assert_per_call_url(url, allow_private=policy.allow_private())


def _url_policy(key: str, factory: ProviderFactory) -> _UrlPolicy:
    registered = _POLICIES.get(key)
    if registered is not None and registered[0] is factory:
        return registered[1]
    if key in _BUILT_INS and factory is _built_in(key):
        return _built_in_policy(key)
    # Fail closed: no configured url and no opt-out.
    return _CUSTOM_POLICY


# key -> (environment prefix, whether an environment ``null`` url means the
# default, as Anthropic's does).
_BUILT_IN_ENVIRONMENT = {
    "anthropic": ("ANTHROPIC", True),
    "mistral": ("MISTRAL", False),
    "openai": ("OPENAI", False),
}


def _built_in_policy(key: str) -> _UrlPolicy:
    """The environment url (or the provider's DEFAULT_URL), and the env opt-out."""
    prefix, null_is_default = _BUILT_IN_ENVIRONMENT[key]

    def configured_url() -> str | None:
        provider = _built_in(key)
        if provider is None:  # pragma: no cover - key is one of _BUILT_INS
            return None
        default = str(sys.modules[provider.__module__].DEFAULT_URL)
        try:
            return configured_string(
                UNSET, f"{prefix}_URL", default, default if null_is_default else None
            )
        except TypeError:
            # An unusable environment url configures no endpoint to exempt.
            return None

    def allow_private() -> bool:
        return _environment_value(f"{prefix}_ALLOW_PRIVATE_PER_CALL_URL", False) is True

    return _UrlPolicy(configured_url, allow_private)


def _built_in(key: str) -> ProviderFactory | None:
    """A built-in provider class, imported on first use.

    Deferred rather than imported at module level so the builder and the
    providers do not have to import each other at start-up.
    """
    if key == "anthropic":
        from prism.providers.anthropic.provider import Anthropic

        return Anthropic

    if key == "mistral":
        from prism.providers.mistral.provider import Mistral

        return Mistral

    if key == "openai":
        from prism.providers.openai.provider import OpenAI

        return OpenAI

    return None


def _default_factory(key: str) -> ProviderFactory | None:
    """A built-in provider, registered under its key on first use."""
    factory = _built_in(key)
    if factory is not None:
        _FACTORIES[key] = factory
    return factory


# -- named OpenAI-compatible providers ------------------------------------------

COMPATIBLE_DRIVER = "openai-compatible"

_API_FORMATS = ("chat_completions", "responses")
_ENTRY_FIELDS = frozenset(
    {
        "driver",
        "url",
        "api_key",
        "api_format",
        "organization",
        "project",
        "allow_private_per_call_url",
    }
)

_INVALID_URL = (
    "That is not a valid provider url: a provider url must be an absolute http or https"
    " url with a host."
)

# Every Cc control, every Unicode White_Space codepoint and the G-36
# zero-widths. A credential carrying one cannot be a header value, and the HTTP
# stack's own refusal would quote the value -- the key -- in its message.
_CREDENTIAL_FORBIDDEN = frozenset(
    [
        *range(0x0000, 0x0020 + 1),
        *range(0x007F, 0x009F + 1),
        0x00A0,
        0x1680,
        *range(0x2000, 0x200D + 1),
        0x2028,
        0x2029,
        0x202F,
        0x205F,
        0x3000,
        0xFEFF,
    ]
)


class _Entry(NamedTuple):
    url: str
    api_key: str | None
    organization: str | None
    project: str | None
    api_format: str
    allow_private_per_call_url: bool


def configure_providers(entries: Mapping[str, Mapping[str, Any]], *, replace: bool = False) -> None:
    """Register named OpenAI-compatible providers (vLLM, LM Studio, a gateway).

    The library's equivalent of the reference's ``prism.providers.<name>``
    config. Each entry is ``{"driver": "openai-compatible", "url": ...}`` with
    optional ``api_key``, ``organization``, ``project``, ``api_format``
    (``chat_completions``, the default, or ``responses``) and
    ``allow_private_per_call_url``. It is then used like any provider:
    ``Prism.text().using("my-vllm", "model")``.

    An entry's provider carries ONLY that entry's credentials. Nothing is read
    from ``OPENAI_*`` or taken from the ``openai`` provider, and a missing or
    empty key sends no ``Authorization`` header at all. A per-call ``url`` that
    names a different endpoint drops the entry's key, organization and project
    unless the call supplies its own.

    Every entry is checked before ANY is registered, so a refused call changes
    nothing. Names follow :func:`register_provider`'s rules and are compared
    ASCII case-insensitively. A built-in name (``openai``, ``OpenAI``,
    ``ANTHROPIC``...) is refused even with ``replace=True``, so a configured
    endpoint can never answer to a built-in's name. With ``replace=True`` an
    earlier registration of the same name, in any case, is replaced.

    The entry's CONFIG url may be private, loopback or link-local: it is operator
    configuration, and the guarded property is that no other credential goes
    there. A PER-CALL ``url`` is different: one naming a loopback, private,
    link-local or metadata address other than the entry's own endpoint is
    refused with ``private_address_refused`` before any request, unless the
    entry sets ``allow_private_per_call_url: True`` (strictly ``True``; ``None``
    or ``False`` leave it off). No refusal names the url, a credential or a
    header value.

    :raises ValueError: for an invalid or taken name, an unsupported driver or
        field, an unusable url, an unknown api format, or a credential that is
        not a clean string.
    """
    checked: list[tuple[str, _Entry]] = []
    names: set[str] = set()

    for name, raw in entries.items():
        if not _is_valid_provider_name(name):
            raise ValueError(_INVALID_NAME)
        folded = _ascii_lower(name)
        if folded in _BUILT_INS:
            raise ValueError(
                f"The provider '{name}' is already registered as a built-in provider;"
                " an openai-compatible provider cannot take a built-in name."
            )
        if folded in names:
            raise ValueError(
                f"The provider '{name}' is already registered by another entry in the same call."
            )
        if not replace and _is_registered(name):
            raise ValueError(
                f"The provider '{name}' is already registered. Pass replace=True to replace it."
            )
        names.add(folded)
        checked.append((name, _check_entry(name, raw)))

    for name, entry in checked:
        if replace:
            for known in [key for key in _FACTORIES if _ascii_lower(key) == _ascii_lower(name)]:
                del _FACTORIES[known]
        _register(name, _compatible_factory(entry), _compatible_policy(entry), replace=replace)


def _check_entry(name: str, raw: object) -> _Entry:
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"The provider entry '{name}' is not a supported provider entry: it must be a mapping."
        )
    if any(field not in _ENTRY_FIELDS for field in raw):
        raise ValueError(
            f"The provider entry '{name}' has a field that is not a supported provider entry"
            " field. The fields are driver, url, api_key, api_format, organization, project"
            " and allow_private_per_call_url."
        )
    driver = raw.get("driver")
    if not isinstance(driver, str) or driver != COMPATIBLE_DRIVER:
        raise ValueError(
            f"That is not a supported provider driver: the provider entry '{name}' must use"
            f" the driver '{COMPATIBLE_DRIVER}'."
        )

    url = _http_url(raw.get("url"))
    api_format = raw.get("api_format")
    allow_private = raw.get("allow_private_per_call_url")
    if allow_private is not None and not isinstance(allow_private, bool):
        raise ValueError(
            f"The provider entry '{name}' field 'allow_private_per_call_url' must be"
            " True, False or None."
        )

    return _Entry(
        url=url,
        api_key=_credential(name, "api_key", raw.get("api_key")),
        organization=_credential(name, "organization", raw.get("organization")),
        project=_credential(name, "project", raw.get("project")),
        api_format="chat_completions" if api_format is None else _api_format(api_format),
        allow_private_per_call_url=allow_private is True,
    )


def _http_url(url: object) -> str:
    """An absolute ``http``/``https`` url with a host and no whitespace or controls."""
    if not isinstance(url, str):
        raise ValueError(_INVALID_URL)
    assert_clean_url(url)
    try:
        parts = urlsplit(url)
        parts.port  # noqa: B018 - parsing the port is the check
    except ValueError:
        # Not chained: the parser's own message quotes the url.
        raise ValueError(_INVALID_URL) from None
    if parts.scheme.lower() not in ("http", "https") or not parts.hostname:
        raise ValueError(_INVALID_URL)
    return url


def _api_format(value: object) -> str:
    if not isinstance(value, str) or value not in _API_FORMATS:
        raise ValueError(
            "That is not a valid api format: an openai-compatible provider speaks"
            " 'chat_completions' or 'responses'."
        )
    return value


def _credential(name: str, field: str, value: object) -> str | None:
    """None, or a string with no whitespace or control character. Never echoed."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"The provider entry '{name}' field '{field}' must be a string or None.")
    if any(ord(char) in _CREDENTIAL_FORBIDDEN for char in value):
        raise ValueError(
            f"The provider entry '{name}' field '{field}' must be a string without"
            " whitespace or control characters."
        )
    return value


def _compatible_policy(entry: _Entry) -> _UrlPolicy:
    """A per-call url may repeat the entry's own url; the opt-out is the entry's."""

    def configured_url() -> str | None:
        return entry.url

    def allow_private() -> bool:
        return entry.allow_private_per_call_url

    return _UrlPolicy(configured_url, allow_private, _http_url)


def _compatible_factory(entry: _Entry) -> ProviderFactory:
    """What ``using(name, model, config)`` calls: the entry, overridden per call."""

    def build(
        url: object = UNSET,
        api_key: object = UNSET,
        organization: object = UNSET,
        project: object = UNSET,
        api_format: object = UNSET,
        transport: Transport | None = None,
        stream_transport: StreamTransport | None = None,
        timeout: float | None = None,
    ) -> Provider:
        from prism.providers.openai.provider import OpenAI

        resolved_url = entry.url
        key, organization_id, project_id = entry.api_key, entry.organization, entry.project

        if url is not UNSET:
            resolved_url = _http_url(url)
            # The entry's credentials belong to the entry's endpoint only.
            if not same_endpoint(resolved_url, entry.url):
                key = organization_id = project_id = None

        if api_key is not UNSET:
            key = _credential("per-call", "api_key", api_key)
        if organization is not UNSET:
            organization_id = _credential("per-call", "organization", organization)
        if project is not UNSET:
            project_id = _credential("per-call", "project", project)

        return OpenAI.from_entry(
            url=resolved_url,
            api_key=key,
            organization=organization_id,
            project=project_id,
            api_format=entry.api_format
            if api_format is UNSET or api_format is None
            else _api_format(api_format),
            transport=transport,
            stream_transport=stream_transport,
            timeout=timeout,
        )

    return build
