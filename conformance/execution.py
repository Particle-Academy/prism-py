"""Running suites and turning cases into verdicts.

Separated from the CLI so the probe test can run the corpus in-process and
compare failure SETS, rather than shelling out and parsing its own output.
"""

from __future__ import annotations

from typing import Any

from prism_conformance import Corpus, compare

from conformance import driver, mutations
from conformance.mutations import FAITHFUL

__all__ = ["FAITHFUL", "failing_ids", "run"]

LANGUAGE = "py"


def run(corpus: Corpus, suite_ids: list[str], probe_id: str) -> list[dict[str, Any]]:
    """Run each suite under one probe and return a report document per suite."""
    mutation = mutations.for_probe(_probe_declaration(corpus, probe_id))

    return [_run_suite(corpus, suite_id, probe_id, mutation) for suite_id in suite_ids]


def failing_ids(corpus: Corpus, probe_id: str) -> dict[str, list[str]]:
    """The case ids each suite failed under one probe.

    Suites with no failures are omitted, so this lines up directly with the
    corpus's own ``expected_probe_failures``.
    """
    observed: dict[str, list[str]] = {}

    for document in run(corpus, corpus.suite_ids(), probe_id):
        failed = [row["id"] for row in document["results"] if row["status"] == "fail"]

        if failed:
            observed[document["suite"]] = failed

    return observed


def _probe_declaration(corpus: Corpus, probe_id: str) -> dict[str, Any]:
    for probe in corpus.probes()["probes"]:
        if probe["id"] == probe_id:
            return dict(probe)

    raise KeyError(f"No probe named {probe_id!r} in the corpus.")


def _run_suite(
    corpus: Corpus,
    suite_id: str,
    probe_id: str,
    mutation: mutations.Mutation,
) -> dict[str, Any]:
    suite = corpus.suite(suite_id)
    kind = str(suite.manifest["kind"])

    results: list[dict[str, Any]] = []

    # Installed once per suite rather than once per case: the rebinding is
    # global, and a case that raised mid-flight must not leave it in place.
    with mutations.installed(mutation, kind) as active:
        for case in suite.cases(LANGUAGE):
            # Skipped rows are REPORTED, not filtered. A skip that vanishes from
            # the report is a suite that quietly shrank.
            if case["skipped"]:
                results.append({"id": case["id"], "status": "skip", "reason": case["skip_reason"]})
                continue

            results.append(_run_case(kind, case, active))

    return {
        "corpus_version": corpus.version,
        "corpus_digest": corpus.digest(),
        "language": LANGUAGE,
        "suite": suite_id,
        "probe": probe_id,
        "results": results,
    }


def _run_case(kind: str, case: dict[str, Any], mutation: mutations.Mutation) -> dict[str, Any]:
    tolerance = case.get("tolerance")

    try:
        attempts = driver.run_case(kind, case, mutation)
    except Exception as error:
        # A crash is a failed case, not a dead run. A mutant is meant to break
        # things, and one that breaks them hard should still be measurable.
        return {
            "id": case["id"],
            "status": "fail",
            "expected": _expected_of(case),
            "actual": f"<raised {type(error).__name__}: {error}>",
        }

    for expected, actual in attempts:
        # The loader's compare, never a local one: byte equality on the
        # canonical JSON string, with a per-case tolerance only where the row
        # declares one. No row in the corpus declares one today.
        if not compare(expected, actual, tolerance):
            return {"id": case["id"], "status": "fail", "expected": expected, "actual": actual}

    return {"id": case["id"], "status": "pass"}


def _expected_of(case: dict[str, Any]) -> str:
    expect = case.get("expect") or {}

    for key in ("body_json", "result_json", "serialized_json", "error_code"):
        if key in expect:
            return str(expect[key])

    return str(expect)
