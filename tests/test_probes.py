"""The discrimination probes.

A conformance table every plausible implementation passes proves nothing. These
tests are what make the table a measurement: each mutant must fail the EXACT set
of case ids the corpus declares — not "at least one", not "a superset" — and the
faithful control must fail nothing at all. Without the control, the mutants
prove only that the port is broken.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

prism_conformance = pytest.importorskip(
    "prism_conformance",
    reason="The corpus loader is not installed. `pip install <prism-parity>/loaders/py`.",
)

from conformance import execution, mutations  # noqa: E402

# Defaults to the loader's own discovery, which is the contract. The override
# exists for developing against a parity checkout whose loader copy has not been
# re-synced yet, so a stale artifact is worked around explicitly and visibly
# rather than by vendoring a second copy of the corpus.
CORPUS_ROOT = os.environ.get("PRISM_CORPUS_ROOT") or None

LANGUAGE = "py"


@pytest.fixture(scope="module")
def corpus() -> Any:
    return prism_conformance.Corpus.open(CORPUS_ROOT)


def probe_ids(corpus: Any) -> list[str]:
    return [str(probe["id"]) for probe in corpus.probes()["probes"]]


def _all_probes() -> list[str]:
    return probe_ids(prism_conformance.Corpus.open(CORPUS_ROOT))


def test_the_corpus_ships_suites(corpus: Any) -> None:
    # Vacuity guard. Every assertion below quantifies over the suites, and a run
    # over zero of them passes without looking at anything.
    assert corpus.suite_ids()


def test_every_declared_probe_has_an_implementation(corpus: Any) -> None:
    # A probe nobody implemented is a probe that silently passes.
    assert set(probe_ids(corpus)) <= mutations.known_ids()


def test_the_faithful_control_fails_nothing(corpus: Any) -> None:
    assert execution.failing_ids(corpus, mutations.FAITHFUL) == {}


@pytest.mark.parametrize("probe_id", _all_probes())
def test_each_probe_fails_exactly_its_declared_set(corpus: Any, probe_id: str) -> None:
    expected = {
        suite_id: sorted(ids)
        for suite_id, ids in corpus.expected_probe_failures(probe_id, LANGUAGE).items()
        if ids
    }
    observed = {
        suite_id: sorted(ids) for suite_id, ids in execution.failing_ids(corpus, probe_id).items()
    }

    assert observed == expected, _difference_report(probe_id, expected, observed)


def _difference_report(
    probe_id: str,
    expected: dict[str, list[str]],
    observed: dict[str, list[str]],
) -> str:
    """Both directions, always. "Some rows failed" is not a measurement."""
    lines = [f"probe {probe_id} did not fail exactly its declared set:"]

    for suite_id in sorted(set(expected) | set(observed)):
        declared = set(expected.get(suite_id, []))
        actual = set(observed.get(suite_id, []))

        missing = sorted(declared - actual)
        unexpected = sorted(actual - declared)

        if missing:
            lines.append(f"  {suite_id}: declared but PASSED: {', '.join(missing)}")
        if unexpected:
            lines.append(f"  {suite_id}: failed but NOT declared: {', '.join(unexpected)}")

    return "\n".join(lines)


def test_a_skipped_row_is_never_expected_to_fail(corpus: Any) -> None:
    # The loader subtracts skips from a probe's declared set, because a skipped
    # row cannot fail and counting it would make the expectation unsatisfiable.
    # Confirmed here rather than assumed.
    for probe_id in probe_ids(corpus):
        expected = corpus.expected_probe_failures(probe_id, LANGUAGE)

        for suite_id, ids in expected.items():
            skipped = set(corpus.suite(suite_id).skipped_ids(LANGUAGE))
            assert not (set(ids) & skipped), f"{probe_id}/{suite_id} expects a skipped row to fail"


def test_omit_null_keys_expects_one_fewer_row_than_it_declares(corpus: Any) -> None:
    # The concrete instance of the subtraction above: the probe names 27 request
    # rows and trq-0025 is skipped for Python, so the effective expectation is 26.
    declared = corpus.probes()["probes"]
    probe = next(row for row in declared if row["id"] == "omit-null-keys")

    expected = corpus.expected_probe_failures("omit-null-keys", LANGUAGE)

    assert len(probe["must_fail"]["openai-text-request"]) == 27
    assert len(expected["openai-text-request"]) == 26
    assert "trq-0025" not in expected["openai-text-request"]
