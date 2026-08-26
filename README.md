# prism-ai

A unified API layer over LLM providers — the Python port of
[Prism](https://github.com/Particle-Academy/prism)'s text capability.

Zero runtime dependencies. Python 3.10+.

```python
from prism import Prism

response = (
    Prism.text()
    .using("openai", "gpt-4o")
    .with_prompt("Who are you?")
    .as_text()
)

print(response.text)
```

> **Working on this package?** Read **[`AGENTS.md`](AGENTS.md)** first — the boundary
> this package has to hold, the gates that must be green, and the traps that have
> already caught someone.
> `@link AGENTS.md`

## Scope

One vertical slice, deliberately: the entry point, the pending-request builder,
the provider contract, the OpenAI provider (Responses API), and the text
capability with its message value objects.

Not in this slice: streaming, structured output, embeddings, the tool-execution
loop, and every provider except OpenAI. Unsupported capabilities raise a coded
error rather than a missing attribute, and a response that finishes on tool
calls is refused with `tool_loop_not_supported` rather than half-executed.

Two things this port has that the reference does not:

- **Every failure carries a stable `code`** (`prism.ErrorCode`). The reference
  identifies failures by an English sentence, which forces consumers to match on
  prose. The prose here is explicitly not part of the contract; the code is.
- **Every value object rebuilds.** `to_dict()` *and* `from_dict()`. The
  reference can write its value objects and cannot read them back, a gap that
  forced a downstream package to invent its own rehydration and ship a defect
  with it.

## Install

```
pip install prism-ai
```

Configuration comes from explicit constructor arguments, falling back to
`OPENAI_API_KEY`, `OPENAI_URL`, `OPENAI_ORGANIZATION` and `OPENAI_PROJECT`. The
HTTP transport is injectable, so nothing has to reach the network in a test:

```python
from prism import OpenAI, Prism

pending = Prism.text().using("openai", "gpt-4o", {"transport": my_transport})
```

## Parity

What this port must do is pinned by
[**prism-parity**](https://github.com/Particle-Academy/prism-parity), not by this
README. The corpus is the contract — the cases, the goldens, the per-language
skips and the discrimination probes all live there, each with its own notes
saying what it exists to catch. Restating them here would only give them a
second copy to drift from.

The corpus and its loader install as the `prism-conformance` package:

```
git clone https://github.com/Particle-Academy/prism-parity .parity
pip install ./.parity/loaders/py
```

Install it **last**: a path install can be pruned by a later install step. The
loader finds its fixtures by walking up from its own installed location, so
nothing here ever resolves a path into a sibling checkout.

Run every suite:

```
python conformance/runner.py
```

Or one suite, or under a probe:

```
python conformance/runner.py --suite openai-text-request
python conformance/runner.py --probe omit-null-keys
```

stdout is JSON and nothing else; the corpus version, digest and root go to
stderr on every run. Exit 0 when every case passed or skipped, 1 on a failure,
2 when the corpus failed to load, 3 when the runner could not start.

### Probes

`conformance/mutations.py` implements each probe the corpus declares as an
injected defect, and `tests/test_probes.py` asserts that every one fails
*exactly* the set of case ids the corpus names — not a superset, not "at least
one" — and that the faithful control fails nothing. That is what makes the
conformance table a measurement rather than decoration.

Nothing in `src/prism` knows any of this exists. A defect a port can switch on
is a defect a port can ship, so the mutants are installed from outside, over the
real library's output, and removed again.

## Development

`src/` layout, so install the package before running the tests:

```
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pip install ../prism-parity/loaders/py

python -m pytest
python -m mypy --strict
python -m ruff check .
python -m ruff format --check .
```

Set `PRISM_CORPUS_ROOT` to run the conformance tests against a parity checkout
whose loader copy has not been re-synced yet. Unset — the normal case — the
loader discovers its own fixtures.

## License

MIT. See [LICENSE](LICENSE).
