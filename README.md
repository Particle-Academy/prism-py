# prism-ai-core

A unified API layer over LLM providers — the Python port of
[Prism](https://github.com/Particle-Academy/prism).

Zero runtime dependencies. Python 3.10+.

Anthropic, OpenAI and Mistral streams retain reported usage on failure through
immutable `StreamFailureUsage` snapshots. Read `ErrorEvent.failure_usage`, or
`PrismError.stream_failure_usage()` on an SDK-owned read exception. The full
`usage` and `reported_fields` mask preserve the difference between a reported
zero and an unknown counter. Snapshots are provisional, cover only the failed
attempt and exclude completed steps. Serialization omits `failure_usage` when
no counter was reported.

Recognized malformed stream JSON now emits one `stream_decode_error`, even
without usage. Anthropic then ends with Error; OpenAI and Mistral emit no
fabricated terminal event. A foreign transport failure with reported usage
emits one error event before rethrowing the original exception. Existing
successful accounting and exception identity/cause stay intact. Native
multi-step stream/tool continuation remains unsupported.

```python
from prism import Prism

response = (
    Prism.text()
    .using("openai", "gpt-4o")
    .with_prompt("Who are you?")
    .as_text()
)

print(response.text)
```

> **Working on this package?** Read **[`AGENTS.md`](AGENTS.md)** first — the boundary
> this package has to hold, the gates that must be green, and the traps that have
> already caught someone.
> `@link AGENTS.md`

## Scope

Stored assistant messages retain provider replay state in `additional_content`:
OpenAI Responses uses `output_items`; Anthropic uses ordered `content_blocks`,
with `thinking_blocks` as a fallback. Keep that metadata when sending history
back. Signatures, redacted thinking and encrypted reasoning remain opaque and
unchanged; avoid logging these values. The legacy thinking/signature pair still
works when full replay metadata is absent.

In 0.5.0, Anthropic EOF before `message_stop` emits non-recoverable
`incomplete_stream` and a terminal Error event. Provider errors close unfinished
text and still end with Error; later stop/usage events cannot erase the error.
Terminal usage stays null unless a final usage delta arrived. Consumers should
handle these newly observable terminal events. Native event vocabulary and
payload differences are tracked separately in the
[stream comparison contract](https://github.com/Particle-Academy/prism-parity/blob/main/docs/streaming-conformance.md).

**Every capability the reference has**, and three of its eighteen providers.
That is the shape of the remaining drift: one axis, not two.

- **Capabilities — all of them.** `text`, `structured`, `stream`, `embeddings`,
  `images`, `moderation`, `audio` (`text_to_speech` and `speech_to_text`),
  `files`, `batch`, `fim`. A user message carries images and documents, and each
  provider spells them its own way.
- **Providers — `anthropic`, `mistral`, `openai`.** The OpenAI provider speaks
  the **Responses API**; Mistral brings the chat-completions shape, which is a
  different wire format rather than a variant of it.

Not in this port: **the tool-execution loop** — a response that finishes on tool
calls is refused with `tool_loop_not_supported` rather than half-executed — and
the fifteen providers the reference has and this one does not. An unsupported
capability raises a coded error rather than a missing attribute.

Tools with `.with_provider_options({"strict": True})` send a closed root object
schema for OpenAI Responses and Anthropic, including tools without parameters.
OpenAI requires every root parameter to be required; an optional root field
raises `PrismError` with code `invalid_tool_schema` before HTTP. Make that field
required (and nullable if needed), or turn strict off. Anthropic retains declared
optional fields. Nested schemas remain exactly as declared: callers must make
those schemas valid for their provider. Mistral's mapping is unchanged.

Structured responses retain `structured`, `steps`, `raw`, and the tool carrier
tuples when serialized with `to_dict()`. Each `StructuredStep`, exported from
`prism.structured`, retains its completed `messages` and per-step parsed value.
Preserve that conversation state when continuing a conversation. Anthropic uses
native schema output by default: `text` is empty while the original JSON remains
in the assistant message. Explicit structured tool mode raises
`unsupported_provider_action` before HTTP; it does not run tools.

Not every capability reaches every provider, because not every provider has one.
`images`, `moderation`, `text_to_speech`, `files` and `batch` are OpenAI-only
here, and `fim` is Mistral-only because Mistral is the only provider that
implements it at all — in the reference as well. See the envelope's port gaps
register for which of those are work someone could do and which are facts about
the provider.

Two things this port has that the reference does not:

- **Every failure carries a stable `code`** (`prism.ErrorCode`). The reference
  identifies failures by an English sentence, which forces consumers to match on
  prose. The prose here is explicitly not part of the contract; the code is.
- **Every value object rebuilds.** `to_dict()` *and* `from_dict()`. The
  reference can write its value objects and cannot read them back, a gap that
  forced a downstream package to invent its own rehydration and ship a defect
  with it.

## Install

```
pip install prism-ai-core
```

The distribution is `prism-ai-core`; the package it installs is imported as
`prism`. To track `main` instead:

```
pip install "git+https://github.com/Particle-Academy/prism-py"
```

**Do not `pip install prism-ai`. That name on PyPI is somebody else's package**
— author `gwhite`, shipping a `prism_ai` module, unrelated to this project. It
is not a squat to route around later: it publishes **the same version number**
this port carries, so a pin of `prism-ai==0.1.0` resolves to the stranger's
distribution with nothing to tell you apart.

It is not inert either. It declares `httpx`, `requests` and `tqdm`, where this
port declares none, and installing it has been reported ROLLING HTTPX BACK
(0.28.1 → 0.25.2) and breaking an MCP client sharing the environment. pip prints
that conflict as a warning *after* the words "Successfully installed", so a CI
or otherwise non-interactive install reports success and the damage surfaces
somewhere else entirely.

Pin the git URL, and leave a comment where you pin it — this is the kind of line
a later tidy-up "corrects" into a version specifier.

Configuration comes from explicit constructor arguments, falling back to
`OPENAI_API_KEY`, `OPENAI_URL`, `OPENAI_ORGANIZATION`, `OPENAI_PROJECT` and
`OPENAI_API_FORMAT`. The
HTTP transport is injectable, so nothing has to reach the network in a test:

```python
from prism import OpenAI, Prism

pending = Prism.text().using("openai", "gpt-4o", {"transport": my_transport})
```

An omitted constructor argument reads the environment, then the provider
default. An explicit empty string stays empty. Explicit `None` follows the
PHP reference's field-specific behavior:

| Field | Explicit `None` |
| --- | --- |
| OpenAI `api_key` / `api_format` | `""` / `"responses"` |
| OpenAI `organization` / `project`; Anthropic `beta_features` | Remains `None`; environment is ignored |
| Anthropic `url` | Default Anthropic URL |
| Anthropic `api_key` / `api_version`; OpenAI `url`; Mistral `api_key` / `url` | Refused with `TypeError` |

Environment strings follow Laravel's reserved spellings: case-insensitive
`null`, `true`, `false`, `empty` and their parenthesized forms become scalars;
matching surrounding quotes are removed. Direct constructor strings remain
literal. Non-string environment scalars are refused for string-only fields.
OpenAI's `api_format` is `responses` (the default) or `chat_completions`, which
posts text, structured and stream calls to `<url>/chat/completions`. Structured
output on chat completions is JSON mode, with the schema in a system message,
as the PHP reference sends it. Any other format raises
`unsupported_provider_action` before HTTP.

### Named OpenAI-compatible providers

`configure_providers()` names an OpenAI-compatible endpoint (vLLM, LM Studio, a
gateway) and makes it a provider like any other:

```python
from prism import Prism, configure_providers

configure_providers({
    "my-vllm": {
        "driver": "openai-compatible",
        "url": "http://10.0.0.5:8000/v1",
        "api_key": vllm_key,              # optional
        "api_format": "chat_completions",  # the default; or "responses"
        "organization": None,              # optional
        "project": None,                   # optional
    },
})

Prism.text().using("my-vllm", "meta-llama/Llama-3.1-8B-Instruct").with_prompt("Hi").as_text()
```

The provider uses only its own entry's credentials. It never reads
`OPENAI_API_KEY`, `OPENAI_ORGANIZATION`, `OPENAI_PROJECT`, `OPENAI_URL` or
`OPENAI_API_FORMAT`, or takes anything from the `openai` provider. With no key it
sends no `Authorization` header. If a per-call `url` points at a different
endpoint, the entry's key, organization and project are dropped unless the call
passes its own. Its requests do not follow HTTP redirects, which would carry the
key to another host; a redirect is reported as an error.

The entry's url must be an absolute `http` or `https` url with a host. As
CONFIG it may be a private, loopback or link-local address; a per-call `url` may
not (see [Per-call urls](#per-call-urls)) unless the entry sets
`"allow_private_per_call_url": True`. Every entry is checked before any is
registered, and each of these raises `ValueError` without echoing the url or a
credential:

- a name outside `^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$` (`not a valid provider name`);
- a name equal to a built-in or an earlier registration in any ASCII case
  (`already registered`); built-in names are refused even with `replace=True`;
- a driver other than `openai-compatible` (`not a supported provider driver`);
- a bad url (`not a valid provider url`);
- an `api_format` other than `chat_completions` or `responses` (`not a valid api format`);
- an `allow_private_per_call_url` that is not `True`, `False` or `None`;
- a credential that is not a string, or contains whitespace or a control character;
- an unknown field.

The name you pass to `using()` is the provider key on the request; nothing
relabels it `openai`.

Credentials are scoped to their configured endpoint. Environment credentials
(`*_API_KEY`, and OpenAI's `OPENAI_ORGANIZATION` / `OPENAI_PROJECT`) are used
only when the effective URL is the environment's URL (`OPENAI_URL`,
`ANTHROPIC_URL`, `MISTRAL_URL`) or, when that is unset, the provider default. A
per-call `url` elsewhere must supply its own key:

```python
Prism.text().using("openai", "gpt-4o", {"url": "https://gateway.example/v1", "api_key": gateway_key})
```

A trailing slash, or a difference in the case of the scheme or host, is the same
endpoint. A different scheme, userinfo, host, port, path or query is not.

`register_provider()` refuses a key that is already registered -- a built-in
such as `openai`, or one registered earlier, compared case-insensitively --
unless `replace=True` is passed. A key must start with an ASCII letter or digit,
contain only ASCII letters, digits, `.`, `_` and `-`, and be at most 64
characters long; any other key raises `ValueError`, even with `replace=True`, so
a look-alike such as `openai` spelled with a Cyrillic `о`, a full-width
`ｏｐｅｎａｉ` or a padded `openai ` cannot be registered. The key is not trimmed.

A `url`, per call or from `OPENAI_URL` / `ANTHROPIC_URL` / `MISTRAL_URL`, that
contains whitespace or a control character anywhere, including at either end,
is refused with `ValueError` before any request is made. It is not trimmed.

### Per-call urls

A CONFIG url -- `OPENAI_URL`, `ANTHROPIC_URL`, `MISTRAL_URL`, the provider
default, or a `configure_providers` entry's `url` -- may point anywhere,
including `127.0.0.1`. A url passed for a single call may not choose a private
address: before the provider is built, a per-call `url` is refused with
`PrismError` code `private_address_refused` when its host is a loopback,
private (RFC 1918), link-local, CGNAT, unspecified, documentation, multicast or
reserved address, an IPv4-mapped, NAT64, 6to4 or ISATAP IPv6 address embedding
one of those (a dotted IPv4 tail such as `[2606:4700::8.8.8.8]` is judged like
its hex spelling), an IPv4-translated address (`::ffff:0:0:0/96`), `localhost` or
any `*.localhost` or `*.localhost.localdomain`, a loopback or local name from
the stock hosts files (`localhost.localdomain`, `localhost4`, `localhost6` and
their `.localdomainN` forms, `ip6-localhost`, `ip6-loopback`,
`ipv6-localhost`, `broadcasthost` and similar), `metadata`,
`metadata.google.internal` or `instance-data`. IPv4 literals are read in the shortened and numeric forms a
resolver accepts (`2130706433`, `0177.0.0.1`, `0x7f000001`, `127.1`), and one
that cannot be read is refused. Trailing dots are ignored, so `localhost.` is
refused like `localhost`. The host is also refused, with the same code, when as
written it holds anything other than letters, digits, `-` and `.` (or an IPv6
address in brackets): a percent escape is refused, not decoded, and so is a
backslash before the path. Invisible and formatting characters (zero-width
joiners, soft hyphens and the like) and `ß`, `ẞ` and `ς`, which IDNA converters
handle differently, are refused anywhere before the path. A url that is not
absolute `http`/`https` is refused with `scheme_not_allowed`. No message names
the url, host or address.

**A per-call host must be ASCII. Give an international name in its punycode
form**: `https://xn--bcher-kva.example/v1`, not `https://bücher.example/v1`.
IDNA converters do not agree on which name some characters spell, so the check
does not convert one. Any other non-ASCII per-call host is refused with
`non_ascii_host_use_punycode`, before any request. The punycode form is sent
unchanged and checked like any other name.

**The check does not resolve DNS.** A public hostname that resolves to a private
address (a wildcard-DNS name such as `127.0.0.1.nip.io`, for example) is not
refused. It stops mistakes and literal addresses, not someone who can choose the
hostname. If a per-call url can come from a user, allow only known hosts or
restrict outbound traffic at the network level as well.

A per-call url that is the same endpoint as the configured one is not checked:
the call is not choosing a new address. A custom provider from
`register_provider()` has no configured url, so every per-call url it is given
is checked.

To let calls choose a private address, opt in in config. It is off by default,
and a per-call `allow_private_per_call_url` key is ignored:

| Provider | Opt-out |
| --- | --- |
| Built-in | `OPENAI_ALLOW_PRIVATE_PER_CALL_URL`, `ANTHROPIC_ALLOW_PRIVATE_PER_CALL_URL`, `MISTRAL_ALLOW_PRIVATE_PER_CALL_URL` set to `true` or `(true)`, any case. Any other value leaves it off. |
| Named compatible | `"allow_private_per_call_url": True` in the `configure_providers` entry |
| Custom | `register_provider(key, factory, allow_private_per_call_url=True)` |

Anthropic reads `ANTHROPIC_API_KEY`, `ANTHROPIC_URL`, `ANTHROPIC_API_VERSION`
and canonical `ANTHROPIC_BETA`. Legacy `ANTHROPIC_BETA_FEATURES` is consumed
only when explicit `beta_features` and canonical `ANTHROPIC_BETA` are both
absent, emitting one value-free `FutureWarning` per process. Explicit
`None` or an empty canonical value prevents that fallback.

`default_thinking_budget` overrides `ANTHROPIC_DEFAULT_THINKING_BUDGET`
(default 1024). The configured scalar is cast as in PHP. An integer
per-request thinking budget wins; booleans, strings and fractions in that
per-request field use the configured fallback. No extra minimum is imposed.

Request `client_options` supports only `timeout` and `connect_timeout`, in
seconds, across JSON, stream and binary dispatch. Values must be finite,
nonnegative numbers; zero disables the corresponding limit. Other keys are
refused, so they cannot replace destinations, headers, redirect policy or body
caps. These options also reach injected transports, which own their enforcement.

The native urllib transport has a **30-second total nonstream deadline**
covering DNS, connection, headers and body. Streams use **30 seconds through
headers**, then Prism's **60-second idle limit per pending body read**, with no
overall stream lifetime deadline. A request's `timeout` overrides the total or
header budget; `connect_timeout` additionally bounds DNS/TCP/TLS connection.
There is no Python `read_timeout` option. Closing a stream closes its response,
including an iterator that was never started.

An expired DNS wait returns promptly; its isolated daemon lookup can finish at
the operating system's resolver timeout and holds no request payload or headers.
HTTPS retains the original hostname for SNI and certificate verification.
Guarded media validates and pins the same numeric addresses per redirect hop;
this native pinned path bypasses environment proxies. A custom transport must
honor `resolved_addresses`, `follow_redirects` and `max_body_bytes` to retain
those protections.

Mistral FIM now preserves response rate-limit headers through actual provider
dispatch. Existing offset serialization differences remain recorded separately.

## Parity

What this port must do is pinned by
[**prism-parity**](https://github.com/Particle-Academy/prism-parity), not by this
README. The corpus is the contract — the cases, the goldens, the per-language
skips and the discrimination probes all live there, each with its own notes
saying what it exists to catch. Restating them here would only give them a
second copy to drift from.

The corpus and its loader install as the `prism-conformance` package:

```
git clone https://github.com/Particle-Academy/prism-parity .parity
pip install ./.parity/loaders/py
```

Install it **last**: a path install can be pruned by a later install step. The
loader finds its fixtures by walking up from its own installed location, so
nothing here ever resolves a path into a sibling checkout.

Run every suite:

```
python conformance/runner.py
```

Or one suite, or under a probe:

```
python conformance/runner.py --suite openai-text-request
python conformance/runner.py --probe omit-null-keys
```

stdout is JSON and nothing else; the corpus version, digest and root go to
stderr on every run. Exit 0 when every case passed or skipped, 1 on a failure,
2 when the corpus failed to load, 3 when the runner could not start.

### Probes

`conformance/mutations.py` implements each probe the corpus declares as an
injected defect, and `tests/test_probes.py` asserts that every one fails
*exactly* the set of case ids the corpus names — not a superset, not "at least
one" — and that the faithful control fails nothing. That is what makes the
conformance table a measurement rather than decoration.

Nothing in `src/prism` knows any of this exists. A defect a port can switch on
is a defect a port can ship, so the mutants are installed from outside, over the
real library's output, and removed again.

## Development

`src/` layout, so install the package before running the tests:

```
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pip install ../prism-parity/loaders/py

python -m pytest
python -m mypy --strict
python -m ruff check .
python -m ruff format --check .
```

Set `PRISM_CORPUS_ROOT` to run the conformance tests against a parity checkout
whose loader copy has not been re-synced yet. Unset — the normal case — the
loader discovers its own fixtures.

## License

MIT. See [LICENSE](LICENSE).
