# pydantic-agent-framework

A **thin, opinionated agent framework on pydantic-ai V2** — the M1 seed of the
flagship portfolio framework. Import name: **`paf`**.

## Principles

- **Extensibility is essential, not optional** — capabilities are the extension
  seam: an allowlisted catalog (`capabilities/registry.py`) + named profiles
  (`capabilities/profiles.py`) resolve to the `custom_capability_types` a spec
  loader registers.
- **Single-agent default** — `Agent` is the composable unit; multi-agent comes
  later via a thin `Runner`, never a rewrite.
- **Hybrid config** — the spec (`specs/*.yaml`) declares what pydantic-ai
  resolves natively (model, instructions, settings); capabilities that need real
  Python objects (tiered compaction, persistence + recall over one shared store)
  are assembled in `capabilities/build.py` and merged via `Agent.from_spec`.
- **Durable, lossless recall** — `StepPersistence` writes every run to SQLite;
  `ConversationSearch` gives the model a BM25 recall tool over the *same* store,
  so anything compaction (sliding window or summary) drops from live context is
  retrievable.
- **Structured traces, not bloated logs** — telemetry to self-hosted Langfuse via
  OTLP (no Logfire).

## Layout

```
src/paf/
  capabilities/   🔥 extension seam — registry, profiles, builders
                  (compaction.py, persistence.py, build.py)
  observability/  Langfuse OTLP instrumentation (setup/shutdown)
  runtime.py      Session — UI-agnostic run lifecycle (build + trace + stream +
                  history + flush)
  specs.py        spec loader (agent name → YAML → Agent)
  config.py       pydantic-settings config + spec/root anchoring
ui/cli.py         production REPL on Session
specs/            declarative agent specs (*.yaml)
```

## Run it

```bash
uv sync                     # install deps
cp .env.example .env        # set OPENAI_API_KEY + optional Langfuse keys
PYTHONPATH= uv run --env-file .env python ui/cli.py   # interactive REPL
```

The REPL keeps a durable conversation (`conversation_id "cli-primary"`). Run it
twice with the same id to see persistence + recall: start a session, quit,
restart, then ask for something reachable only via `search_conversation_history`.

A keyless/deterministic run path is available for debugging in
`ui/cli_demo.py` (`MODEL='test'`).

## Verify

```bash
ruff check .
PYTHONPATH= uv run pytest          # offline unit + integration (live deselected)
PYTHONPATH= uv run pytest -m live  # opt-in: real model call + Langfuse trace (needs .env)
```

## Status

Private M1 scaffold, workable and near-public. Evals deferred (not M1). Logfire
explicitly not used — observability is self-hosted Langfuse via OTLP.