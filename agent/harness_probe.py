"""An END-TO-END exercise of prism-harness-py, run against a real disk.

The point is dogfooding, not demonstration. The harness's own suite proves its
pieces; this proves the assembled package works OUTSIDE its own repo, in a
process that did not write it, reached over the wire by the Lab. Those are
different claims, and only the second is what a consumer experiences.

FREE AND DETERMINISTIC, deliberately. The model is a scripted client rather than
a provider call: what is under test is the session, the thread, the budget and
the approval gate, none of which involve a provider -- and a probe that costs
money is a probe nobody runs.

Written to a TEMPORARY directory that is removed afterwards, so polling the Lab
board does not accumulate state on disk.

Mirrors `prism-ts/agent/harness-probe.mjs` step for step.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

#: The sibling repo, two levels up from this file in `agent/`. See
#: ecosystem_probe.py -- same root, same reason, and deliberately no skip path
#: when the sibling is absent.
_PORTS_ROOT = (
    Path(os.environ["PRISM_PORTS_ROOT"]).resolve()
    if os.environ.get("PRISM_PORTS_ROOT")
    else Path(__file__).resolve().parents[2]
)

_HARNESS_SRC = _PORTS_ROOT / "prism-harness-py" / "src"


def probe_harness() -> dict[str, Any]:
    """Run the scenario and return a trace the board can render.

    Every step reports what it OBSERVED rather than a boolean, so a failure
    names which property stopped holding instead of reporting that "the harness
    broke".
    """
    if str(_HARNESS_SRC) not in sys.path:
        sys.path.insert(0, str(_HARNESS_SRC))

    try:
        from prism_harness import (
            AgentRuntime,
            FileSessionStore,
            HarnessError,
            LlmResponse,
            LlmToolCall,
            MemorySessionStore,
            ModeRegistry,
            Participant,
            PrismHarness,
            ToolRegistry,
            record_approval,
        )
    except ImportError as error:
        return {
            "ok": False,
            "reason": "prism-harness-py is not checked out beside this port",
            "detail": str(error),
            "steps": [],
        }

    directory = tempfile.mkdtemp(prefix="prism-harness-probe-")
    steps: list[dict[str, Any]] = []

    def check(name: str, observed: Any, expected: Any) -> None:
        steps.append(
            {"step": name, "observed": observed, "expected": expected, "ok": observed == expected}
        )

    try:
        modes = ModeRegistry(
            {
                "default": "chat",
                "modes": {
                    "chat": {"system_prompt": "Be brief.", "tools": ["echo"], "max_steps": 4},
                    "guarded": {
                        "system_prompt": "Careful.",
                        "tools": ["echo"],
                        "max_steps": 4,
                        "requires_approval": ["echo"],
                    },
                },
            }
        )

        runs = {"tool": 0, "turn": 0}

        class Echo:
            @property
            def name(self) -> str:
                return "echo"

            def handle(self, args: dict[str, Any]) -> str:
                runs["tool"] += 1
                return f"echoed:{args.get('value', '')}"

        tools = ToolRegistry().register(Echo())

        # The guard the package exists for. Checked FIRST, because if it does
        # not hold nothing else here means anything.
        refused = None
        try:
            PrismHarness(
                drivers={"memory": MemorySessionStore},
                stores={"ephemeral": "memory", "durable": "memory"},
            ).durable_store()
        except HarnessError as error:
            refused = error.code
        check(
            "a volatile store is refused for durable state",
            refused,
            "unsafe_state_configuration",
        )

        app = PrismHarness(
            drivers={"memory": MemorySessionStore, "files": lambda: FileSessionStore(directory)},
            stores={"ephemeral": "memory", "durable": "files"},
        )

        participant = Participant("App\\Models\\User", 7)
        session = app.for_(participant).session("probe")
        session.using_mode("guarded").using_provider("anthropic").using_model("claude-sonnet-4-5")

        # The same address PHP builds, and the same one prism-harness-ts builds.
        check(
            "the session key matches the reference format",
            session.key(),
            "session:23bd5c8949f6:7:probe",
        )

        def client(_request: Any) -> Any:
            runs["turn"] += 1
            if runs["turn"] <= 2:
                return LlmResponse(
                    text="",
                    finish_reason="tool_calls",
                    tool_calls=[LlmToolCall("call-1", "echo", {"value": "hello"})],
                )
            return LlmResponse(text="All done.", finish_reason="stop")

        runtime = AgentRuntime(client=client, modes=modes, tools=tools)

        first = runtime.send(session, "Say hello with the tool")

        check(
            "a gated tool STOPS the run instead of running",
            first.finish_reason,
            "awaiting_approval",
        )
        check("the gated tool did not execute", runs["tool"], 0)
        check(
            "the approval request is pending",
            [approval.tool for approval in first.pending_approvals],
            ["echo"],
        )

        # The decision is durable: written to the thread, which lives in the
        # file store, so a different process would read the same answer.
        record_approval(session, "call-1", True)

        resumed = runtime.send(session, "")

        check("the tool runs once the approval is recorded", runs["tool"], 1)
        check("the resumed turn completes", resumed.text, "All done.")

        # A SECOND session object over the same stores -- the "resolved, never
        # held" property. This is what a fresh worker sees.
        reopened = app.for_(participant).session("probe")

        check("a fresh session sees the same mode", reopened.mode(), "guarded")
        check("a fresh session sees the whole conversation", reopened.thread().count() > 0, True)

        run = reopened.run() or {}
        check("the run is recorded as completed", run.get("status"), "completed")
        check("the run records tool NAMES only", run.get("tool_calls"), ["echo"])
        check("no tool arguments are recorded on the run", "hello" in json.dumps(run), False)

        return {
            "ok": all(step["ok"] for step in steps),
            "language": "py",
            "package": "prism-ai-harness",
            "session_key": session.key(),
            "thread_messages": reopened.thread().count(),
            "steps": steps,
        }
    except Exception as error:  # reported, not handled
        return {"ok": False, "reason": "the probe threw", "detail": str(error), "steps": steps}
    finally:
        shutil.rmtree(directory, ignore_errors=True)
