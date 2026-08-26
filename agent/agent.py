"""prism.py - the Python member of the Prism agent team.

It reasons by calling prism-py itself. That is the point rather than an
implementation detail: an agent built on this port is the port's most
demanding consumer, and every defect it trips over is one a user would have
tripped over. An agent that reasoned through some other SDK would test
nothing.

Standard library only, like the package it lives in.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prism import Prism

LANGUAGE = "py"
MODEL = os.environ.get("PRISM_AGENT_MODEL", "gpt-4.1-mini")

# Long enough for a real suite, bounded so a hung child cannot wedge the lane.
RUN_TIMEOUT = int(os.environ.get("PRISM_AGENT_RUN_TIMEOUT", "300"))

ROOT = Path(__file__).resolve().parents[1]


def _port_version() -> str | None:
    try:
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    except OSError:
        return None

    for line in text.splitlines():
        if line.startswith("version"):
            return line.split("=", 1)[1].strip().strip('"')

    return None


def _run(argv: list[str]) -> dict[str, Any]:
    """Run a child process and hand back both streams.

    Never raises on a non-zero exit. A failing suite is the ANSWER to
    `run_tests`, not an error in asking - collapsing the two would make a red
    suite indistinguishable from a broken agent.
    """
    try:
        completed = subprocess.run(
            argv,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=RUN_TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "stdout": "", "stderr": "timed out", "timed_out": True}

    return {
        "ok": completed.returncode == 0,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "timed_out": False,
    }


def _tail(text: str, lines: int = 40) -> str:
    """Enough to diagnose, not so much that it floods a context."""
    return "\n".join(text.splitlines()[-lines:]).strip()


def status(_: dict[str, Any]) -> dict[str, Any]:
    return {
        "language": LANGUAGE,
        "agent": "prism.py",
        "port_version": _port_version(),
        "model": MODEL,
        # Named, never returned. Whether a key EXISTS is a status question;
        # what it is never is.
        "can_reason": bool(os.environ.get("OPENAI_API_KEY")),
    }


def describe_port(_: dict[str, Any]) -> dict[str, Any]:
    """What this port actually implements, read from disk rather than remembered.

    The agent was confidently wrong about a provider it had never had, because
    nothing let it check - it reasoned from the question it was asked instead
    of from the port it lives in.
    """
    providers_dir = ROOT / "src" / "prism" / "providers"
    providers = (
        sorted(
            entry.name
            for entry in providers_dir.iterdir()
            if entry.is_dir() and not entry.name.startswith("__")
        )
        if providers_dir.is_dir()
        else []
    )

    modules = (
        sorted(
            path.relative_to(ROOT / "src" / "prism").as_posix()
            for path in (ROOT / "src" / "prism").rglob("*.py")
            if "__pycache__" not in path.parts
        )
        if (ROOT / "src" / "prism").is_dir()
        else []
    )

    return {
        "language": LANGUAGE,
        "providers_implemented": providers,
        "provider_count": len(providers),
        "modules": modules,
        "note": (
            "A provider absent from providers_implemented is not implemented here at all - "
            "not merely missing a field."
        ),
    }


def run_conformance(_: dict[str, Any]) -> dict[str, Any]:
    result = _run([sys.executable, "conformance/runner.py"])

    # The runner writes the report document as JSON on stdout and nothing
    # else. Returned as it comes: the corpus contract is versioned and shared,
    # and reshaping it here is exactly the drift prism-parity exists to
    # prevent.
    line = next((ln for ln in reversed(result["stdout"].splitlines()) if ln.strip()), "")

    try:
        return {"ok": result["ok"], "report": json.loads(line)}
    except json.JSONDecodeError:
        return {
            "ok": False,
            "report": None,
            "reason": "the conformance runner did not emit a parseable report",
            "output": _tail(result["stderr"] or result["stdout"]),
        }


def run_tests(_: dict[str, Any]) -> dict[str, Any]:
    result = _run([sys.executable, "-m", "pytest", "-q", "-ra"])
    return {
        "passed": result["ok"],
        "timed_out": result["timed_out"],
        "output": _tail(result["stdout"] or result["stderr"]),
    }


SYSTEM_PROMPT = (
    "You are prism.py, the Python member of the Prism agent team. Prism is a provider-agnostic LLM "
    "library ported across PHP, TypeScript and Python; the ports must behave identically "
    "for the same "
    "input. You are given a failure in the Python port. Explain the actual cause, say "
    "whether it is a "
    "Python defect or a genuine cross-language disagreement, and propose the smallest fix. If the "
    "evidence does not support a conclusion, say what is missing instead of guessing."
)


def explain(arguments: dict[str, Any]) -> dict[str, Any]:
    subject = arguments.get("subject", "")

    if not os.environ.get("OPENAI_API_KEY"):
        # Say so rather than calling with an empty bearer token and returning
        # whatever the provider says about it.
        return {"ok": False, "reason": "no OPENAI_API_KEY set for this agent - it cannot reason"}

    parts = [f"Subject: {subject}"]
    for label in ("expected", "actual"):
        if arguments.get(label):
            parts.append(f"{label.capitalize()}: {arguments[label]}")
    if arguments.get("context"):
        parts.append(f"Context:\n{arguments['context']}")

    response = (
        Prism.text()
        .using("openai", MODEL)
        .with_system_prompt(SYSTEM_PROMPT)
        .with_prompt("\n\n".join(parts))
        .with_max_tokens(900)
        .as_text()
    )

    return {
        "ok": True,
        "language": LANGUAGE,
        "analysis": response.text,
        "model": response.meta.model,
        "tokens": {
            "prompt": response.usage.prompt_tokens,
            "completion": response.usage.completion_tokens,
        },
    }


Handler = Callable[[dict[str, Any]], dict[str, Any]]

TOOLS: dict[str, dict[str, Any]] = {
    "status": {
        "description": (
            "Report this agent's language, the port version it is running, and whether "
            "it can reason. "
            "Cheap; safe to poll."
        ),
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        "handler": status,
    },
    "describe_port": {
        "description": (
            "What this port actually implements - providers and modules. Read from the source, "
            "not remembered. Call this before reasoning about whether a feature exists here."
        ),
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        "handler": describe_port,
    },
    "run_conformance": {
        "description": (
            "Run the cross-language conformance suite for Python and return the report "
            "document unchanged."
        ),
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        "handler": run_conformance,
    },
    "run_tests": {
        "description": (
            "Run this port's own test suite. Returns pass/fail and the tail of the output."
        ),
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        "handler": run_tests,
    },
    "explain": {
        "description": (
            "Reason about a failure in this language and propose a fix. Slow and billable - "
            "call it for a "
            "specific failure, not for a whole run."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": "What failed - a case id, a test name, or a short description.",
                },
                "expected": {"type": "string"},
                "actual": {"type": "string"},
                "context": {
                    "type": "string",
                    "description": (
                        "Anything else that would help: source, corpus entry, findings."
                    ),
                },
            },
            "required": ["subject"],
            "additionalProperties": False,
        },
        "handler": explain,
    },
}
