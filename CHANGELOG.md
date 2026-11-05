# Changelog

## Unreleased

- **Breaking:** distinguish omitted provider arguments from explicit `None` and empty strings. Nullable fields no longer fall back to environment when passed `None`; string-only null/scalar inputs are refused. See the README migration table.
- Honor `OPENAI_API_FORMAT`, canonical `ANTHROPIC_BETA`, and configured/environment default thinking budgets. Legacy beta fallback warns once without values and only if consumed. Integer per-request thinking budgets retain precedence.
- Forward only validated `timeout` and `connect_timeout` client options to every provider dispatch. Keep zero runtime dependencies and the standard-library urllib transport.
- Enforce nonstream total 30-second deadlines, including DNS and body reads. Streams have a 30-second header budget followed by Prism's 60-second body idle limit and no lifetime deadline. Preserve original-hostname TLS verification and pin validated guarded-media addresses through redirects.
- Preserve Mistral FIM rate-limit headers. Add native-backed provider configuration/FIM corpus cases and offline network-escape controls. Existing G43 offset and G125 native stream trace divergences remain open.
