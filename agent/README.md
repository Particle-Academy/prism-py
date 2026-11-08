# prism.py — the local contributor agent

This loopback MCP server exposes source discovery, test and conformance tools for this port. Its paid `explain` and `consensus` tools currently refuse every call with typed `paid_call_not_authorised`, before a provider builder can run. A provider key, owner label, approval flag or caller-supplied run ID grants no authority. G246 tracks the future reviewed cross-process run-authority contract.

## Run

Run `python agent/server.py`. The agent uses the source package beside it. The default listener is `127.0.0.1:7412`, configurable with `PRISM_AGENT_PORT`. Provider/model reporting follows the source defaults (Anthropic/claude-sonnet-4-5) and `PRISM_AGENT_PROVIDER`/`PRISM_AGENT_MODEL`.

## Tools

| Tool | Behavior |
| --- | --- |
| status | Free; provider key presence is separate from authority. can_reason is false with paid_call_refusal. |
| describe_port | Free source/provider/capability discovery. |
| harness_probe | Free native harness probe. |
| ecosystem_probe | Free satellite security probes. |
| run_conformance | Executes the existing corpus and returns its report. |
| run_tests | Executes the local test suite with its existing bound. |
| explain | Typed paid_call_not_authorised refusal; no provider call. |
| consensus | Typed paid_call_not_authorised refusal; no provider call. |

The MCP result has `isError: true` and identical structured/text refusal data: `{ok:false,code:"paid_call_not_authorised",reason:"…"}`. Discovery and free probes remain available. No approval, GO, launch, impersonation or authentication transport is added. Published SDK runtime code is unchanged; this contributor agent directory is excluded from the package archive.

MCP `2026-07-28` over HTTP supports `server/discover`, `tools/list` and `tools/call`, with no initialize/session handshake. The Python standard library provides the transport. Keep the listener on loopback: its local test and conformance capabilities execute code.
