"""Paid tools require a future reviewed authority contract, even with a key."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import pytest

from prism import Prism

AGENT_DIR = Path(__file__).resolve().parents[1] / "agent"
_spec = spec_from_file_location("paid_authority_agent", AGENT_DIR / "agent.py")
assert _spec is not None and _spec.loader is not None
agent = module_from_spec(_spec)
_spec.loader.exec_module(agent)


@pytest.mark.parametrize("name", ["explain", "consensus"])
def test_paid_tool_refuses_before_provider_builder(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-test-key")
    calls: list[str] = []

    def provider_trap() -> Any:
        calls.append("provider")
        raise RuntimeError("BILLABLE_BUILDER_REACHED")

    monkeypatch.setattr(Prism, "text", provider_trap)
    arguments = {
        "subject": "Owned failure fixture",
        "question": "Owned question fixture",
        "approved": True,
        "owner": "Wish",
        "run_id": "caller-supplied-run",
        "authority": {"approved": True, "cap": 5},
    }
    with pytest.raises(Exception) as error:
        agent.TOOLS[name]["handler"](arguments)
    assert getattr(error.value, "code", None) == "paid_call_not_authorised"
    assert calls == []


def test_status_does_not_authorise_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-test-key")
    status = agent.status({})
    assert status["can_reason"] is False
    assert status["provider_key_present"] is True
    assert status["paid_call_refusal"] == "paid_call_not_authorised"


def test_free_discovery_and_existing_tool_vocabulary_stay_available() -> None:
    described = agent.describe_port({})
    assert "anthropic" in described["providers_implemented"]
    assert "text" in described["provider_operations"]
    assert sorted(agent.TOOLS) == [
        "consensus",
        "describe_port",
        "ecosystem_probe",
        "explain",
        "harness_probe",
        "run_conformance",
        "run_tests",
        "status",
    ]
    for name in ("explain", "consensus"):
        assert "paid_call_not_authorised" in agent.TOOLS[name]["description"]
        assert "refused" in agent.TOOLS[name]["description"]
