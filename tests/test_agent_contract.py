"""The tools this port's agent exposes.

Asserted HERE rather than discovered by a consumer. prism-labs was the only
thing that ever called these servers, so a dropped or renamed tool would have
surfaced as a red banner on a Lab screen and nowhere else -- which is exactly
how the missing ``benchmark`` tool stayed invisible until someone screenshotted
a preflight failure. See the port gaps register, G-10 and G-11.

The TypeScript half of this test earned its place on the first run: the source
defined six tools and a live probe of the running agent returned five, because
the server had been started from an older build and nobody could tell. A test
over the source cannot catch a stale process, but it does establish which list
is the intended one, so the two can be compared at all. That is G-12.
"""

from __future__ import annotations

import sys
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parents[1] / "agent"

if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

import agent  # noqa: E402 - the path has to be set before the import


def test_the_agent_exposes_exactly_the_tools_the_ecosystem_expects() -> None:
    # The same six as prism-ts, asserted independently rather than read across
    # repos: this port must not need its sibling checked out to run its suite. A
    # tool on one agent and not the other is a gap the Lab shows as an empty
    # panel, so the two lists are kept identical on purpose.
    assert sorted(agent.TOOLS) == [
        "consensus",
        "describe_port",
        "explain",
        "run_conformance",
        "run_tests",
        "status",
    ]


def test_benchmark_is_not_exposed_and_that_is_a_tracked_gap() -> None:
    # G-10. Deliberately asserted rather than left absent: when a lane-execution
    # contract exists and this tool is added, THIS test fails and forces the
    # register entry to be closed in the same change.
    assert "benchmark" not in agent.TOOLS


def test_every_tool_has_a_description_and_an_input_schema() -> None:
    for name, tool in agent.TOOLS.items():
        assert tool.get("description"), f"{name} has no description"
        assert tool.get("inputSchema"), f"{name} has no input schema"
        assert callable(tool.get("handler")), f"{name} has no handler"


def test_describe_port_reports_capabilities_not_just_modules() -> None:
    # The agent was once confidently wrong about a provider it had never had,
    # because nothing let it check. A module list has the same failure mode for
    # capabilities: it invites inference from filenames.
    described = agent.describe_port({})

    assert described["capabilities_implemented"] == [
        "audio",
        "batch",
        "embeddings",
        "files",
        "fim",
        "images",
        "moderation",
        "structured",
        "text",
    ]
    assert described["providers_implemented"] == ["anthropic", "mistral", "openai"]


def test_describe_port_separates_entry_points_from_provider_operations() -> None:
    # They are different lists and only the second is what the parity manifest
    # counts: `stream` is a terminal on the text builder and the audio pair are
    # terminals on `audio`. An agent comparing eight entry points against the
    # manifest's twelve would report a gap that is not there.
    operations = agent.describe_port({})["provider_operations"]

    for name in ("stream", "text_to_speech", "speech_to_text"):
        assert name in operations, f"{name} missing from provider_operations"

    # `fim` is here now that Mistral is -- it was absent while G-14 was open,
    # and this line is what closing that gap looks like from the agent's side.
    # It is an entry point AND an operation, unlike stream and the audio pair.
    assert "fim" in operations


def test_status_reports_whether_this_process_is_running_the_code_on_disk() -> None:
    # G-12. The running server is the one thing a test over the source cannot
    # check: a server started before a tool was added keeps serving the old
    # list, and the only consumer is a Lab screen that reports what it is told.
    # This is the agent answering the question itself.
    reported = agent.status({})

    assert reported["agent_source_digest"] is not None
    assert reported["agent_stale"] is False


def test_a_changed_agent_file_is_reported_as_stale() -> None:
    # The signal has to actually fire, or it is a field that always says "fine".
    original = agent.LOADED_DIGEST
    try:
        agent.LOADED_DIGEST = "0" * 12
        assert agent.status({})["agent_stale"] is True
    finally:
        agent.LOADED_DIGEST = original
