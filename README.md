# pydantic-agent-framework

A **thin, opinionated agent framework on pydantic-ai V2**. Import name: **`paf`**.

Two live agents ship out of the box: a production `assistant` and an
`demo_assistant` "everything sandbox" used to build out and tune the full stack.

## Principles

- **Extensibility is essential, not optional** — capabilities are the extension
  seam: an allowlisted catalog (`capabilities/registry.py`) + named profiles
  (`capabilities/profiles.py`) resolve to the `custom_capability_types` a spec
  loader registers.
- **Single-agent default** — `Agent` is the composable unit; multi-agent comes
  later via a thin `Runner`, never a rewrite.
- **Hybrid config** — the spec (`specs/*.yaml`) declares what pydantic-ai
  resolves natively (model, instructions, settings); capabilities that need real
  Python objects (persistence, durable memory + recall, compaction) are
  assembled in `capabilities/build.py` and merged via `Agent.from_spec`.
- **Durable, lossless recall** — `StepPersistence` writes every run to a
  flat-file store (`data/`, JSON/JSONL — greppable, no DB to administer);
  recall tools give the model access to past turns over the *same* store.
- **Structured traces, not bloated logs** — telemetry to self-hosted Langfuse via
  OTLP (no Logfire).

## Agents

Two named agents resolve from declarative YAML specs (`specs/*.yaml`) via
`Agent.from_spec`:

| Agent | Profile | Purpose |
|-------|---------|---------|
| `assistant` | `assistant` | Production agent: durability + recall + compaction. |
| `demo_assistant` | `demo` | "Everything sandbox" — tunes the full stack and exercises the demo capabilities live. |

Both default to **DeepSeek V4 Flash via OpenRouter** (requires
`OPENROUTER_API_KEY` in `.env`; bare slug, no `:nitro` suffix).

## Capabilities

Capabilities are the extension seam — an allowlisted catalog (`registry.py`)
that a named profile (`profiles.py`) resolves to concrete capability types,
assembled in `build.py` and merged into the agent by `build_agent`
(`specs.py`).

### Core (both agents)

- **StepPersistence** — every run to a flat-file store (`data/`, JSON/JSONL).
- **ConversationSearch** — paf-owned BM25 retrieval: a wide
  `search_conversation_history` over the raw transcript, plus the shared
  full-content read (`read_turn`) that expands a `run_id:turn_id` ref into the
  complete original turn.
- **Memory** — a durable, inspectable fact notebook (see below).
- **Compaction** — context management: sliding-window eviction and
  clear-tool-results scavenging to keep long conversations in bounds.

### Demo (`demo_assistant` — "everything sandbox")

The demo additionally allowlists the harness-style capabilities:

- **FileSystem** — scoped read/write/edit/search under a root.
- **Shell** — run/start/check/stop, rooted at a cwd, with allow/deny lists and
  env-masking.
- **Git** — safe inspection + checkpoint tools (explicit paths only).
- **Web** — Tavily-backed search + page extraction (`search_web_docs`,
  `search_official_docs`, `fetch_web_page`).
- **Skills** — LIST / LOAD on demand plus a deterministic POINTER trigger.
- **ToolOutputLimits** — framework-level tool-return reduction (spill/summarize/
  truncate oversized results).
- **WarnOnCacheBusts** — observer that warns when the prompt-cacheable prefix
  collapses mid-run.
- **Subagents** — `delegate_task` to named child agents in isolated runs.

Native pydantic-ai capabilities (WebSearch, WebFetch, XSearch, Thinking, MCP,
ToolSearch, ImageGeneration) resolve directly from the `demo_assistant` spec's
YAML `capabilities:` list — no Python build step needed.

## Memory

Memory is **paf-owned and flat-file-first** — an *append-only JSONL journal is
the source of truth*, and `MEMORY.md` is a pure presentation view rendered from
it (no opaque DB; inspectable, editable, diffable).

```
data/memory/
  memory-journal.jsonl   # source of truth — append-only, one JSON record per durable fact
  MEMORY.md              # rendered view — what gets injected as <memory>
```

**Write side — the consolidation owner.** A dedicated owner sub-agent extracts
durable facts from the conversation and appends them to the journal
(`append_memory(type, fact, born_on, source_ref)`) — one fact per call, six
semantic types (`identity` / `preference` / `fact` / `event` / `task` /
`current`). It never edits, moves, deletes, or dedupes; newest-wins and dedup
are resolved by the render pass.

**Read side — the assistant (read-only).** The answerer reads the rendered
`MEMORY.md`, injected into `<memory>` on every request, and recalls by the
ladder *in-context → `search_conversation_history` → `read_turn`* — it has no
memory-write tool (recording is the owner's job).

**As-of date / traceability.** Each record carries `born_on` (ISO date) and
`source_ref` (`run_id:turn_id`), so every fact is date-grounded and
`read_turn`-resolvable back to the turn it came from.

## UIs

All UIs drive the same `Session`/`Runtime` underneath.

- `ui/cli.py` — plain REPL.
- `ui/cli2.py` / `ui/cli.py` — Rich live render with streaming thinking/tool
  status.
- `ui/cli_textual.py` — Textual full-screen.
- `ui/cli_demo.py` — keyless/deterministic debug path (`MODEL='test'`).
- `ui/web.py` — **Gradio web UI.** Serves a single-lane conversation for the
  chosen agent; defaults to `demo_assistant`.

```bash
# Gradio web UI (default agent demo_assistant on 0.0.0.0:8080)
# option 1: root venv
set -a && . ./.env && set +a && .venv/bin/python ui/web.py
# option 2: uv
set -a && . ./.env && set +a && uv run --env-file .env python ui/web.py
# pass an agent name / port explicitly
python ui/web.py demo_assistant 8080
```

## Benchmarking: LongMemEval

The graded proof gate is **LongMemEval** (`tests/run_lmem.py`) — the official
binary judge for answer correctness plus **ID-membership recall@k** over
surfaced `session=` tags (no token-overlap F1) for retrieval. Oracle split lives
in `data/benchmarks/lme_oracle.json`; graded records in `tests/results/`.

```bash
cd /home/alice/projects/Personal/pydantic-agent-framework
set -a && . ./.env && set +a && \
  LME_MAX_QUESTIONS=4 PYTHONPATH= .venv/bin/python -u -m tests.run_lmem
```

Benchmark data, seeding, and knobs are documented in
[`tests/AGENTS.md`](tests/AGENTS.md).

## Layout

```
src/paf/
  capabilities/   extension seam — registry, profiles, builders
    search/       paf-owned BM25 ConversationSearch (search_conversation_history)
                  + shared full-content read (read_turn)
    memory/       paf-owned Memory — journal + render + consolidation owner
    fs/ shell/ git/ web/ skills/   demo-profile capabilities ("everything sandbox")
  observability/  Langfuse OTLP instrumentation (setup/shutdown)
  runtime.py      Session — UI-agnostic run lifecycle (build + trace + stream +
                  history + flush)
  specs.py        spec loader (agent name → YAML → Agent)
  config.py       pydantic-settings config + spec/root anchoring
specs/            declarative agent specs (*.yaml)
ui/               cli.py (plain), cli.py (Rich), cli_textual.py (Textual),
                  cli_demo.py (keyless), web.py (Gradio)
tests/            unit tests + benchmark harness (run_lmem, ...)
```

## Run it

Dependencies resolve from a sibling `agent-foundation` package
(`[tool.uv.sources]` path) — the real workflow uses the local venv + env:

```bash
# one-time
cp .env.example .env        # OPENAI_API_KEY + optional Langfuse keys
PYTHONPATH= .venv/bin/pip install -e .   # or: uv sync

# interactive REPL — plain assistant
PYTHONPATH="" .venv/bin/python -m ui.cli
# or the richer live UIs (same Session/Runtime underneath)
PYTHONPATH="" .venv/bin/python -m ui.cli2            # Rich live render
PYTHONPATH="" .venv/bin/python -m ui.cli_textual     # Textual full-screen
# the "everything sandbox" demo profile (fs/shell/git/web/skills/subagents)
uv run --env-file .env python ui/cli2.py demo_assistant
```

Prepend `set -a && . ./.env && set +a;` to export secrets for the runner. The
REPL keeps a durable conversation (`conversation_id "cli-primary"` for
`ui.cli`, per-agent lanes for `cli2`); run it twice with the same id to see
persistence + recall — start, quit, restart, then ask for something reachable
only via `search_conversation_history`.

A keyless/deterministic run path is available in `ui/cli_demo.py`
(`MODEL='test'`).

## Verify

```bash
PYTHONPATH="" .venv/bin/ruff check . --fix --output-format=concise
PYTHONPATH="" .venv/bin/python -m pytest -q          # offline unit + integration
PYTHONPATH="" .venv/bin/python -m pytest -m live     # opt-in: real model + Langfuse (needs .env)
```

> **PYTHONPATH footgun.** Hermes injects a leaked `PYTHONPATH` (3.11
> site-packages) per terminal call that breaks the project venv's
> `pydantic_core`. Prefix commands with `PYTHONPATH=""` (or use the compiled
> binary directly) to bypass it. The same applies to the benchmark commands.

A root `conftest.py` wires `.env` into `os.environ` at import time so bare
`pytest` and `pytest -m live` both run green.
