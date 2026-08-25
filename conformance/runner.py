"""The conformance runner.

    python conformance/runner.py [--suite <id>] [--probe <id>] [--root <path>] [--version]

stdout is JSON and nothing else. stderr carries the corpus version, digest and
root, on every run: a port pinned to a stale corpus otherwise stays green
against a contract that has moved on and nobody is told.

Exit codes: 0 every case passed or skipped, 1 at least one failed, 2 the corpus
failed to load, 3 the runner could not start. Under ``--probe`` a mutant is
EXPECTED to fail cases, so the caller decides whether the run was correct — the
exit codes stay as they are and the port's own probe test does the judging.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Runnable as `python conformance/runner.py` and as `python -m conformance.runner`.
# The first puts this file's own directory on the path rather than the repo root,
# so the package this module belongs to would not be importable without this.
if __package__ in (None, ""):  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LANGUAGE = "py"

EXIT_OK = 0
EXIT_FAILURES = 1
EXIT_CORPUS = 2
EXIT_CANNOT_START = 3


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="conformance/runner.py",
        description="Run the prism-parity conformance corpus against prism-py.",
    )
    parser.add_argument("--suite", help="Run one suite. Omitted, every suite the corpus ships.")
    parser.add_argument("--probe", help="Run under a named probe. Omitted means faithful.")
    parser.add_argument("--root", help="Load the corpus from an explicit root.")
    parser.add_argument("--version", action="store_true", help="Print the corpus version.")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    try:
        from prism_conformance import Corpus, CorpusError

        from conformance import execution
    except ImportError as error:  # pragma: no cover - exercised only by a broken install
        print(
            f"Cannot start: {error}. Install the corpus loader with "
            "`pip install ./.parity/loaders/py`.",
            file=sys.stderr,
        )
        return EXIT_CANNOT_START

    try:
        corpus = Corpus.open(args.root)

        if args.version:
            print(corpus.version)
            return EXIT_OK

        suite_ids = corpus.suite_ids()

        # Vacuity guard. A run over zero suites exits 0 and looks like success,
        # which is worse than a failure: it reports agreement it never sought.
        if not suite_ids:
            print(json.dumps({"error_code": "empty_corpus"}))
            print(f"The corpus at {corpus.root} ships no suites.", file=sys.stderr)
            return EXIT_CORPUS

        if args.suite is not None:
            if args.suite not in suite_ids:
                print(json.dumps({"error_code": "unknown_suite"}))
                print(f"No suite named {args.suite!r} in {corpus.root}.", file=sys.stderr)
                return EXIT_CORPUS

            suite_ids = [args.suite]

        probe_id = args.probe or execution.FAITHFUL

        print(
            f"corpus {corpus.version} {corpus.digest()}\nroot {corpus.root}\nprobe {probe_id}",
            file=sys.stderr,
        )

        documents = execution.run(corpus, suite_ids, probe_id)
    except CorpusError as error:
        print(json.dumps({"error_code": error.code}))
        print(f"{error.code}: {error}", file=sys.stderr)
        return EXIT_CORPUS
    except KeyError as error:
        # An unimplemented probe, or a scope this runner does not recognise.
        print(json.dumps({"error_code": "unknown_probe"}))
        print(str(error), file=sys.stderr)
        return EXIT_CORPUS

    print(json.dumps(_document(documents), ensure_ascii=False))

    return EXIT_FAILURES if _has_failures(documents) else EXIT_OK


def _document(documents: list[dict[str, Any]]) -> Any:
    """One document per suite; a bare array when more than one suite ran."""
    return documents[0] if len(documents) == 1 else documents


def _has_failures(documents: list[dict[str, Any]]) -> bool:
    return any(
        result["status"] == "fail" for document in documents for result in document["results"]
    )


if __name__ == "__main__":
    sys.exit(main())
