# Contributing

Thanks for looking. paf is a small project, so the best first step is an issue
describing what you want to change before you write a lot of code.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/landonmclain/paf.git
cd paf
uv sync
cp .env.example .env     # set OPENAI_API_KEY to run agents; tests need no key
git config core.hooksPath .githooks
```

If your shell sets a `PYTHONPATH` from another Python install, clear it for paf
commands (`PYTHONPATH= uv run ...`), or the wrong `pydantic_core` build loads.

## Checks

```bash
uv run pytest -q -n 8                                  # offline suite, about a minute
uv run ruff check src tests ui --output-format=concise
uv run ruff format src tests ui
npx --yes pyright
```

The pre-push hook runs ruff, the format check and pyright, and CI runs the
tests too. Tests marked `live` or `eval` call real models and are skipped by
default; run them with `-m live`. Test and benchmark details are in
[tests/AGENTS.md](tests/AGENTS.md).

## Where things live

- `src/paf/capabilities/`: one package per capability (tools plus the
  capability class). `build.py` assembles them; `profiles.py` decides which an
  agent may use.
- `specs/`: agent specs. Every key is documented in
  [docs/spec-reference.md](docs/spec-reference.md). Add a new spec key to that
  page and to `demo_assistant.yaml` (commented out is fine).
- `docs/`: the MkDocs site.
- `ui/`: the SSE and ops servers.

## Making a change

- Keep changes small and focused, and match the surrounding code's style.
- Add or update a test. A new capability needs an offline test; use the
  fixtures in `tests/` rather than real keys or the network.
- Adding a profile also means updating `test_profiles_are_the_complete_intentional_set`.
- Don't commit personal paths, hostnames, emails or keys. A test checks for the
  common shapes.
- Security-sensitive changes (sandbox, gating, approvals, guardrails) need a test
  that shows the bad case is refused.

## Commits and pull requests

Use short imperative subjects ("web: add a keyless search backend"), with a
sentence or two of why in the body when it isn't obvious. Open the pull request
against `main` and say what you ran. By contributing you agree to license your
work under the [Apache License 2.0](LICENSE).

## Reporting security issues

Please follow [SECURITY.md](SECURITY.md) rather than opening a public issue.
