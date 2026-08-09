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
- **Durable, lossless recall** — `StepPersistence` writes every run to a
  flat-file store (`data/`, JSON/JSONL — greppable, no DB to administer);
  `ConversationSearch` gives the model a BM25 recall tool over the *same* store,
  so anything compaction (sliding window or summary) drops from live context is
  retrievable.
- **Structured traces, not bloated logs** — telemetry to self-hosted Langfuse via
  OTLP (no Logfire).

## M2 — the episodic turn index (feature 1)

M2's first feature built as an installable pydantic-ai capability:
`paf.capabilities.turn_index`.

**The problem.** As a conversation grows, keeping every turn verbatim in context
degrades. Compaction drops detail into a lossy summary; you want durable, lossless
*recall* without paying for it per-turn. The answer here is **recognition beats
recall**: keep a one-line-per-turn *index* in context (cheap, bounded), and pull a
given turn's full content out of persistence on demand.

**Two concerns, deliberately decoupled:**

1. **Write path — index each completed turn (synchronous today, background-ready).**
   `after_model_request` fires the moment the model answers each turn. It slices
   *this* turn straight from the live `ctx.messages`, summarizes it via a real
   pydantic-ai structured-output call, and appends one line to an append-only
   JSONL `index_store`, keyed by a **stable `turn_id`** (the ordinal of the turn's
   user anchor — never rots as the run grows). A pure per-turn write: never touches
   context, independent of compaction, moves to a background worker unchanged.

2. **Read path — inject + recall.** `before_model_request` injects a bounded tail
   of the index into the current user request (marker-scoped so it strips its own
   prior injection and never accumulates), and a `read_turn` tool resolves a ref
   back into pydantic-ai persistence through the `HistorySource` seam. A
   `search_index` tool BM25-matches the one-liners *and* their key terms.

**The summarizer (`_summarizer.py`) — a structured record, not a sentence:**

- `TurnSummary(BaseModel)`: `sentence` (the in-context one-liner, hard char-capped)
  + `key_terms` (the **search vocabulary** — what a future query actually matches,
  so "the database" turns are still found by "postgres") + `outcome`
  (`resolved/blocked/error/decided/awaiting`).
- `TurnSummarizer(Agent)`: a real pydantic-ai agent, `output_type=TurnSummary`,
  bounded `retries=` for schema validation. Defaults to a cheap secondary model
  (`openrouter:deepseek/deepseek-v4-flash-0731:nitro`), decoupled from the chat
  model, so the write path stays cheap.
- **Char-budget retry (the axis pydantic-ai's builtin `retries=` can't cover).**
  Builtin retries resend the *same* prompt, which can't shorten output. So an
  over-budget `sentence` triggers our own loop: re-run with an explicit *"you wrote
  N chars, cap is M, cut to essentials"* corrective prompt, bounded, with a hard
  floor so nothing over-budget ever persists, plus a `key_terms` clamp.
- **Stateless / order-independent** — each turn summarizes from its own messages
  only. Composable, reorderable, background-idempotent. Cross-turn carry-forward is
  a possible *later* layer.

**Dual persistence, intentionally.** `index_store` is the capability's *own*
append-only JSONL of derived one-liners (never full turns — only the
`(conversation_id, run_id, turn_id)` keys). Full turns live in pydantic-ai
persistence (e.g. the same `FileStepStore` a `StepPersistence` capability writes),
resolved through the `HistorySource` protocol seam — so it works with **any**
pydantic-ai backend (InMemory / File / Sqlite / Mongo). Shared store, two
surfaces.

Declarative `from_spec` reconstructs the whole capability from a single directory:

```yaml
capabilities:
  - TurnIndex: { index_dir: data/paf }
```

Verified live (2026-08-08): a real-model call at the 160-char budget returned a
156-char `sentence` with `key_terms` `['pg_stat_activity','pgbouncer',
'auth_sessions','statement_timeout',…]` and `outcome: resolved`; at a tight
70-char budget the shorten retry fired and returned a 57-char sentence. 53 tests
green (6 summarizer).

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

M1 shipped: thin, opinionated pydantic-ai V2 scaffold — capabilities/registry,
profiles, tiered compaction, durable flat-file `StepPersistence` + BM25
`ConversationSearch` over one shared store, Langfuse OTLP (no Logfire).

M2 in progress on `feature/m2`. Feature 1 (the **episodic turn index**) is
committed: per-turn synchronous indexing via `after_model_request`, a real
structured-output LLM summarizer with char-budget retry, one-line index injected
into context (recognition beats recall), `read_turn` back into persistence, and
`search_index` over summaries + key terms — built as a clean, installable
capability.

## Roadmap

- **Background the summarizer (M2 next).** The per-turn write is already a pure,
  self-contained call; move it off the request path onto a background worker.
  Because it's stateless and keyed by stable `turn_id`, this is a wiring change,
  not a redesign.
- **In-context read path (separate slice).** Decide how many index lines to
  inject, eviction/ordering among competing summaries, and how the injected index
  composes with `Memory` and compaction.
- **Installable package split.** The turn-index subtree is `paf`-free in spirit;
  extract it into a standalone installable package later.
- **M2 memory arm.** The `Memory` capability (durable-fact extraction) is charted
  as the second M2 feature, sharing the same flat-file seam.
- **Evals** (deferred, not M1/M2 core): long-conversation recall fidelity,
  injection-in-context effectiveness, summarizer budget adherence.