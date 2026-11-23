# Changelog

## Unreleased

## 0.10.0 — 2026-11-23

BREAKING CHANGE: Credentials are now scoped to their configured endpoint; a per-call URL override must supply its own key. register_provider requires replace=True to replace a registered provider, and refuses any key that does not match ASCII `^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$`, including a homoglyph or full-width look-alike, even with replace=True. A provider url, per call or from the environment, that contains whitespace or a control character anywhere is refused with ValueError before any request; it was previously trimmed for the endpoint comparison.

## 0.9.0 — 2026-11-05

BREAKING CHANGE: Distinguish omitted provider arguments from explicit `None` and empty strings. Nullable fields no longer fall back to environment when passed `None`; string-only null/scalar inputs are refused. See the README migration table.
- Honor `OPENAI_API_FORMAT`, canonical `ANTHROPIC_BETA`, and configured/environment default thinking budgets. Legacy beta fallback warns once without values and only if consumed. Integer per-request thinking budgets retain precedence.
- Forward only validated `timeout` and `connect_timeout` client options to every provider dispatch. Keep zero runtime dependencies and the standard-library urllib transport.
- Enforce nonstream total 30-second deadlines, including DNS and body reads. Streams have a 30-second header budget followed by Prism's 60-second body idle limit and no lifetime deadline. Preserve original-hostname TLS verification and pin validated guarded-media addresses through redirects.
- Preserve Mistral FIM rate-limit headers. Add native-backed provider configuration/FIM corpus cases and offline network-escape controls. Existing G43 offset and G125 native stream trace divergences remain open.

- Pin conformance to merged corpus 0.1.19 (`ae4b68bb923ac74030a5d88a924fdf1f09675caa`) with 162 native provider-configuration/FIM cases.
