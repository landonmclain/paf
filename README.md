# pydantic-agent-framework

A **thin, opinionated agent framework on pydantic-ai V2** — the M1 seed of the
flagship portfolio framework. Import name: **`paf`**.

## Principles

- **Extensibility is essential, not optional** — every subsystem is a narrow
  `Protocol` + a default impl + config-swappable backend ("extend-a-protocol").
- **Single-agent default** — `Agent` is the composable unit; multi-agent comes
  later via a thin `Runner`, never a rewrite.
- **Inspectable state** — flat-file/JSON storage (`data/state/`) that both the
  agent and a human can read and edit.
- **Structured traces, not bloated logs** — telemetry to self-hosted Langfuse.

## Layout

```
src/paf/
  agents/        one Agent per role (M1: single-agent)
  capabilities/  🔥 custom capabilities seam
  toolsets/      raw tools, shared across agents
  deps/          DI types (RunContext[Deps])
  models/        shared Pydantic schemas (typed contract layer)
  observability/ Langfuse OTLP instrumentation (no Logfire)
  storage/       🔥 Storage Protocol + flat-file backend
ui/cli.py        thin CLI
data/state/      gitignored, human-editable runtime state
```

## Run it

```bash
uv sync                     # install deps
cp .env.example .env        # configure model + optional Langfuse keys
uv run python -m ui.cli "Say hello to Maya"
```

A single agent turn runs end-to-end, emits a trace to Langfuse (if keys set),
and writes its structured output to `data/state/last_greeting.json`.

### Interactive CLI demo (cross-process memory)

```bash
uv run --env-file .env ui/cli_demo.py
```

Talks to the live `openai:gpt-5.4-nano` model and persists the conversation to
`ui/demo.db`, so a fresh process can resume the same dialog. Requires
`OPENAI_API_KEY` (added to the gitignored `.env`; uv needs the explicit
`--env-file .env` flag — it does not auto-load the project `.env` in this setup).
Set `MODEL='test'` inside the file for a keyless, deterministic run.

## Status

Private M1 scaffold. Evals deferred (not M1). Logfire explicitly not used —
observability is self-hosted Langfuse via OTLP.
