"""The runner's own contract: JSON on stdout, humans on stderr, exit codes."""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

pytest.importorskip(
    "prism_conformance",
    reason="The corpus loader is not installed. `pip install <prism-parity>/loaders/py`.",
)

from conformance import runner

# See tests/test_probes.py: defaults to the loader's own discovery.
CORPUS_ROOT = os.environ.get("PRISM_CORPUS_ROOT") or None


def invoke(*args: str) -> list[str]:
    return [*args, *(["--root", CORPUS_ROOT] if CORPUS_ROOT else [])]


def test_version_prints_the_corpus_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert runner.main(invoke("--version")) == runner.EXIT_OK

    assert capsys.readouterr().out.strip() != ""


def test_a_single_suite_emits_a_single_document(capsys: pytest.CaptureFixture[str]) -> None:
    code = runner.main(invoke("--suite", "openai-text-request"))
    captured = capsys.readouterr()

    # stdout is JSON and NOTHING else: parsing the whole stream is the assertion.
    document = json.loads(captured.out)

    assert code == runner.EXIT_OK
    assert isinstance(document, dict)
    assert document["suite"] == "openai-text-request"
    assert document["language"] == "py"
    assert document["probe"] == "faithful"
    assert document["corpus_digest"].startswith("sha256:")


def test_stderr_carries_the_version_digest_and_root(capsys: pytest.CaptureFixture[str]) -> None:
    runner.main(invoke("--suite", "text-errors"))
    captured = capsys.readouterr()

    # A port pinned to a stale corpus otherwise stays green against a contract
    # that has moved on and nobody is told.
    assert "corpus " in captured.err
    assert "sha256:" in captured.err
    assert "root " in captured.err


def test_every_case_gets_a_row_including_skipped_ones(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner.main(invoke("--suite", "openai-text-request"))
    document = json.loads(capsys.readouterr().out)

    statuses = {row["status"] for row in document["results"]}

    assert statuses <= {"pass", "skip", "fail"}
    # A skip that vanishes from the report is a suite that quietly shrank.
    assert any(row["status"] == "skip" for row in document["results"])
    assert all(row.get("reason") for row in document["results"] if row["status"] == "skip")


def test_running_every_suite_emits_an_array(capsys: pytest.CaptureFixture[str]) -> None:
    code = runner.main(invoke())
    documents: list[dict[str, Any]] = json.loads(capsys.readouterr().out)

    assert code == runner.EXIT_OK
    assert isinstance(documents, list)
    assert len(documents) >= 5
    assert all(row["status"] in ("pass", "skip") for d in documents for row in d["results"])


def test_an_unknown_suite_is_a_corpus_failure(capsys: pytest.CaptureFixture[str]) -> None:
    code = runner.main(invoke("--suite", "nonesuch"))

    assert code == runner.EXIT_CORPUS
    assert json.loads(capsys.readouterr().out) == {"error_code": "unknown_suite"}


def test_an_unknown_probe_is_a_corpus_failure(capsys: pytest.CaptureFixture[str]) -> None:
    code = runner.main(invoke("--probe", "nonesuch"))

    assert code == runner.EXIT_CORPUS
    assert json.loads(capsys.readouterr().out) == {"error_code": "unknown_probe"}


def test_a_missing_corpus_root_is_a_corpus_failure(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Any,
) -> None:
    code = runner.main(["--root", str(tmp_path)])

    assert code == runner.EXIT_CORPUS
    assert "error_code" in json.loads(capsys.readouterr().out)


def test_a_mutant_run_exits_with_failures(capsys: pytest.CaptureFixture[str]) -> None:
    # The exit code stays as it is under a probe; the caller decides whether the
    # run was correct, and the probe test is what does the judging.
    code = runner.main(invoke("--probe", "omit-null-keys", "--suite", "openai-text-request"))
    document = json.loads(capsys.readouterr().out)

    assert code == runner.EXIT_FAILURES
    assert any(row["status"] == "fail" for row in document["results"])
    assert all(
        {"expected", "actual"} <= set(row) for row in document["results"] if row["status"] == "fail"
    )
