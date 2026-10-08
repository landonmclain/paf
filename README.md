# pydantic-agent-framework

A **thin, opinionated agent framework on pydantic-ai V2**. Import name: **`paf`**.

The primary agent is `demo_assistant` — an "everything sandbox" that exercises
the full stack (persistence, recall, durable memory, compaction) plus the demo
capabilities live. A leaner `assistant` profile ships the production core
without the demo surface.

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

Agents resolve from declarative YAML specs (`specs/*.yaml`) via `Agent.from_spec`.
The primary agent is `demo_assistant` — the "everything sandbox" that exercises
the full stack (persistence, recall, durable memory, compaction) plus the demo
capabilities live. A leaner `assistant` profile ships the production core
(durability + recall + compaction) without the demo surface.

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
- **CodebaseMap** — navigation-first codebase browser (overview/lookup_file/
  read_block/search_codebase + re-index).
- **Vault** — navigation-first note ladder over an Obsidian-style markdown vault.
- **Web** — Tavily-backed search + page extraction (`search_web_docs`,
  `search_official_docs`, `fetch_web_page`).
- **Browser** — mobile-first headless Chromium automation (navigate/snapshot/
  click/type/screenshot + set_viewport), with SSRF defense.
- **Visual** — vision-model description of captured screenshots.
- **Skills** — LIST / LOAD on demand plus a deterministic POINTER trigger.
- **ToolOutputLimits** — framework-level tool-return reduction (spill/summarize/
  truncate oversized results).
- **WarnOnCacheBusts** — observer that warns when the prompt-cacheable prefix
  collapses mid-run.
- **Subagents** — `delegate_task` to named child agents in isolated runs.
- **Planning** — model-owned task plan (write_plan/read_plan/add_task/...).
- **ThreadSummarizer** — per-conversation episodic background summarizer.
  Phase 1: records each eviction block (per-message `{run_id}:{turn_id}` refs) to
  `data/summaries/demo_assistant/evictions.jsonl`. OFF for the production agent.
- **Guardrails** — structural prompt-injection defense.
- **ToolGating** — per-call execution-time tool gate for the highest-risk tools.
- **Secrets** — secret-handle resolution + result redaction (real values never
  enter the window).

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

All UIs drive the same `Session` underneath.

- `paf chat` (`src/paf/cli/`) — inline terminal chat: streams thinking, tool
  calls, and a markdown-rendered answer into normal scrollback, with a bottom
  toolbar for background processes and plan progress. Type while a turn runs to
  steer it; `/image <path>` sends an image; `/help` lists the rest. Runs
  in-process (no server needed). From a source checkout, `python ui/chat.py` is
  the same command.
- `ui/sse.py` — FastAPI SSE server for a web frontend: turn streams, the
  background-process feed, steer/cancel, conversations, projects, and uploads.

## Benchmarking: LongMemEval

The graded proof gate is **LongMemEval** (`tests/run_lmem.py`) — the official
binary judge for answer correctness plus **ID-membership recall@k** over
surfaced `session=` tags (no token-overlap F1) for retrieval. Oracle split lives
in `data/benchmarks/lme_oracle.json`; graded records in `tests/results/`.

```bash
set -a && . ./.env && set +a && \
  LME_MAX_QUESTIONS=4 PYTHONPATH= python -u -m tests.run_lmem
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
ui/               chat.py (inline terminal chat), sse.py (background-process feed),
                  web.py (Gradio)
tests/            unit tests + benchmark harness (run_lmem, ...)
```

## Run it

The real workflow uses the local venv + env:

```bash
# one-time
cp .env.example .env        # OPENROUTER_API_KEY + optional Langfuse keys
PYTHONPATH= .venv/bin/pip install -e .   # or: uv sync

# interactive CLI — defaults to the demo_assistant "everything sandbox"
paf chat
# or the leaner production assistant, on a fresh conversation
paf chat assistant --new
```

Prepend `set -a && . ./.env && set +a;` to export secrets for the runner. The
REPL keeps a durable conversation; run it twice with the same conversation id
to see persistence + recall — start, quit, restart, then ask for something
reachable only via `search_conversation_history`.

## Verify

```bash
.venv/bin/ruff check . --fix --output-format=concise
python -m pytest -q          # offline unit + integration
python -m pytest -m live     # opt-in: real model + Langfuse (needs .env)
```

A root `conftest.py` wires `.env` into `os.environ` at import time so bare
`pytest` and `pytest -m live` both run green.
