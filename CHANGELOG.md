# Changelog

## Unreleased

- `configure_providers(entries, *, replace=False)` registers named OpenAI-compatible providers (`driver: "openai-compatible"`, `url`, optional `api_key`, `organization`, `project`, `api_format`). Each one uses only its own entry's credentials. It never reads `OPENAI_*` and never takes the `openai` provider's config. With no key it sends no `Authorization` header. A per-call url for a different endpoint drops the entry's credentials unless the call passes its own. Names, drivers, urls (absolute http/https only; private addresses allowed), api formats and credentials are all checked before anything is registered. (G-321 A)
- OpenAI `api_format: "chat_completions"` is implemented for text, structured and stream, posting to `<url>/chat/completions`. Request bodies match the PHP reference byte for byte. A reply that finished on tool calls is refused with `tool_loop_not_supported` before any second request. Tool-call arguments that are not a JSON object are refused at mapping, and when read from a streamed call. A refusal from the model raises `provider_response_error`. (G-321 D)
- A named compatible provider does not follow HTTP redirects. urllib re-sends the `Authorization` header to wherever a 301, 302 or 303 points, so following one would send the entry's key to a host nobody configured. A redirect is reported as `provider_response_error`.
- On chat completions, a user message carrying a document, audio or video part is refused with `unsupported_media`. The reference drops these parts without saying so.
- A stream that ends without a finish reason reports `unknown`, not `stop`.

## 0.11.0 — 2026-11-23

BREAKING CHANGE: `ToolCall.decoded_arguments()` raises `malformed_tool_call_arguments` for a number, string, boolean, non-empty list or integer-keyed map. It used to return an empty dict for the first four. An empty string, the text `0`, `null`, `[]` and `{}` still mean no arguments. Code that reads the arguments before running a tool should catch the error and return it to the model as a tool error, or check `has_malformed_arguments()` first.
BREAKING CHANGE: Arguments the PHP reference cannot decode are refused too: `NaN` and `Infinity`, a lone surrogate, and more than 500 nested containers.
BREAKING CHANGE: A stored tool call with unusable arguments and no tool result now fails when it is sent. Answer it with a tool result first.
- `ToolCall.has_malformed_arguments()` says whether a call's arguments are unusable.
- Once a later tool result answers such a call, it is sent back with empty arguments in each provider's own shape (OpenAI, Anthropic, Mistral), including stored OpenAI output items. The caller's messages are not modified.
- A call that nothing has answered fails at mapping with `malformed_tool_call_arguments` on every provider, now including Mistral.
- Tests now check that core has no approval check and never runs a tool.

## 0.10.0 — 2026-11-23

BREAKING CHANGE: Credentials are now scoped to their configured endpoint; a per-call URL override must supply its own key. register_provider requires replace=True to replace a registered provider, and refuses any key that does not match ASCII `^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$`, including a homoglyph or full-width look-alike, even with replace=True. A provider url, per call or from the environment, that contains whitespace or a control character anywhere is refused with ValueError before any request; it was previously trimmed for the endpoint comparison.

## 0.9.0 — 2026-11-05

BREAKING CHANGE: Distinguish omitted provider arguments from explicit `None` and empty strings. Nullable fields no longer fall back to environment when passed `None`; string-only null/scalar inputs are refused. See the README migration table.
- Honor `OPENAI_API_FORMAT`, canonical `ANTHROPIC_BETA`, and configured/environment default thinking budgets. Legacy beta fallback warns once without values and only if consumed. Integer per-request thinking budgets retain precedence.
- Forward only validated `timeout` and `connect_timeout` client options to every provider dispatch. Keep zero runtime dependencies and the standard-library urllib transport.
- Enforce nonstream total 30-second deadlines, including DNS and body reads. Streams have a 30-second header budget followed by Prism's 60-second body idle limit and no lifetime deadline. Preserve original-hostname TLS verification and pin validated guarded-media addresses through redirects.
- Preserve Mistral FIM rate-limit headers. Add native-backed provider configuration/FIM corpus cases and offline network-escape controls. Existing G43 offset and G125 native stream trace divergences remain open.

- Pin conformance to merged corpus 0.1.19 (`ae4b68bb923ac74030a5d88a924fdf1f09675caa`) with 162 native provider-configuration/FIM cases.
