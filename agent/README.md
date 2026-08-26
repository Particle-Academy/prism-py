# prism.py — the Python member of the Prism agent team

An MCP server that reasons **through this port**.

That is the point rather than an implementation detail. An agent built on
prism-py is this port's most demanding consumer, and every defect it trips
over is one a user would have tripped over. An agent that reasoned through
some other SDK would test nothing.

## Running it

```bash
python agent/server.py   # listens on 127.0.0.1:7412
```

Reasoning needs `OPENAI_API_KEY`. Without it `status` reports
`can_reason: false` and `explain` says so plainly rather than calling with an
empty bearer token.

| Variable | Default |
|---|---|
| `PRISM_AGENT_PORT` | `7412` |
| `PRISM_AGENT_MODEL` | `gpt-4.1-mini` |
| `PRISM_AGENT_RUN_TIMEOUT` | `300` (seconds) |

## Tools

| Tool | Cost | Returns |
|---|---|---|
| `status` | free | language, port version, model, whether it can reason |
| `run_conformance` | cheap | the conformance **report document, unchanged** |
| `run_tests` | cheap | pass/fail and the tail of the output |
| `explain` | **billable** | model-written analysis of one named failure |

Same four tools, same names and shapes, as every other language agent. A
roster where each member answers a different vocabulary is not a team.

`run_conformance` returns the corpus report exactly as the runner emits it.
The cross-language contract is versioned and shared; reshaping it here is the
drift `prism-parity` exists to prevent.

## Protocol

MCP `2026-07-28` over HTTP — `server/discover`, `tools/list`, `tools/call`.
That revision removed `initialize` and the session, so there is no handshake.

Standard library only, like the package it lives in.

## Why loopback only

This agent runs the test suite and spends tokens. That is remote code
execution wearing a friendly name, and it has no business being reachable
from anywhere but this machine.
