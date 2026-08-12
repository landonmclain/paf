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

## Capabilities (all paf-owned)

The full recall/memory stack — `turn_index`, `search`, and `memory` — is now
**implemented inside `paf` itself**, not split across the harness.

```
src/paf/capabilities/
  turn_index/   episodic index: one-line-per-turn + key_terms, read_turn + search_index
  search/       paf-owned BM25 ConversationSearch (search_conversation_history tool)
  memory/       paf-owned Memory capability, markdown backend (readable .md + CAS sidecar)
  registry.py   allowlisted catalog;  profiles.py named profiles;  build.py assembles
  compaction.py tiered compaction (sliding window + summary);  persistence.py StepPersistence
```

`agent-foundation` remains a *dependency* — but only for the **turn-index
store** (the derived/index shape). The memory arm's atomic helpers
(`write_atomic` / `read_json` / `write_json_atomic`) are inlined in
`paf/capabilities/memory/_atomic.py`, so **memory no longer needs
`agent-foundation`** — the memory `.md` book is the authoritative shape; the
`StorageService` seam idea for it was thrown out (a mis-fit; the seam is the
*derived* shape).

## The episodic turn index (`paf.capabilities.turn_index`)

**The problem.** As a conversation grows, keeping every turn verbatim in context
degrades. Compaction drops detail into a lossy summary; you want durable, lossless
*recall* without paying for it per-turn. The answer here is **recognition beats
recall**: keep a one-line-per-turn *index* in context (cheap, bounded), and pull a
given turn's full content out of persistence on demand.

**Write path — index completed turns.** `after_model_request` fires when the
model answers each turn. It is **gated on terminal responses**: turns whose
`response.parts` contain a `ToolCallPart` are *skipped* — only final model
answers are indexed, so intermediate tool-call chatter never contaminates the
index (fix `981c533`). Each indexed turn is sliced straight from the live
`ctx.messages`, summarized via a real pydantic-ai structured-output call, and
appended as one line to an append-only JSONL `index_store`, keyed by a stable
`turn_id` (the ordinal of the turn's user anchor — never rots as the run grows).

**Read path — inject + recall.** `before_model_request` injects a bounded tail
of the index into the current user request (marker-scoped so it strips its own
prior injection and never accumulates); `read_turn` resolves a ref back into
pydantic-ai persistence through the `HistorySource` seam; `search_index`
BM25-matches the one-liners **and** their key terms, with a **conservative
recency nudge** (`recency_weight=0.1`, anchored to the newest result) that
breaks ties toward newer facts without overriding significant relevance.

**The summarizer (`_summarizer.py`) — a structured record, not a sentence:**

- `TurnSummary(BaseModel)`: `sentence` (the in-context one-liner, hard
  char-capped) + `key_terms` (the **search vocabulary** — so "the database"
  turns are still found by "postgres") + `outcome`
  (`resolved/blocked/error/decided/awaiting`).
- `TurnSummarizer(Agent)`: a real pydantic-ai agent, `output_type=TurnSummary`,
  bounded `retries=` for schema validation. Defaults to a cheap secondary model
  (`openrouter:deepseek/deepseek-v4-flash-0731:nitro`), decoupled from the chat
  model, so the write path stays cheap.
- **Char-budget retry** (the axis pydantic-ai's builtin `retries=` can't cover).
  An over-budget `sentence` triggers our own loop: re-run with an explicit
  *"you wrote N chars, cap is M, cut to essentials"* corrective prompt, bounded,
  with a hard floor so nothing over-budget ever persists, plus a `key_terms`
  clamp.
- **Stateless / order-independent** — each turn summarizes from its own messages
  only. Composable, reorderable, background-idempotent.

**Dual persistence, intentionally.** `index_store` is the capability's *own*
append-only JSONL of derived one-liners (never full turns — only the
`(conversation_id, run_id, turn_id)` keys). Full turns live in pydantic-ai
persistence (e.g. the same `FileStepStore` a `StepPersistence` capability
writes), resolved through the `HistorySource` protocol seam — so it works with
**any** pydantic-ai backend (InMemory / File / Sqlite / Mongo). Shared store, two
surfaces.

Declarative `from_spec` reconstructs the whole capability from a single directory,
with an optional flat `summarizer:` block:

```yaml
capabilities:
  - TurnIndex:
      index_dir: data/paf
      summarizer:                 # omit for the deterministic (keyless) stub
        budget_chars: 160         # one-liner char cap
        max_key_terms: 6          # search vocabulary cap
        # model: ollama:llama3.1  # cheap/local, independent of the chat model
```

**Every recall tool tags its output with the source `session_id`**
(`search_index`, `read_turn`, `search_conversation_history`). This is what makes
**ID-membership** benchmark scoring possible — see LongMemEval below.

## Memory (`paf.capabilities.memory`)

The memory arm is **paf-owned and live**, un-gated as a capability. It runs with
`backend="markdown"`, writing **readable `.md` bodies** plus a flat
`.memory-state.json` CAS sidecar under `data/memory/` (root-anchored) — so saved
knowledge is inspectable, editable, and diffable (flat-file-first; no opaque DB).

Conflict resolution is **"most recently user-stated value wins"** — the Memory
capability's guidance treats the most recent user-stated fact as truth over
older conflicting entries (commit `48f8c14`).

## Benchmarking: LongMemEval (the grading gate) + LoCoMo

How a change to the recall stack "helps or hurts" is answered **by a graded
benchmark, not vibes** — and the authoritative gate is **LongMemEval** (with
LoCoMo for the eviction/compaction axis), not the previous Beam-Lite framing.

LongMemEval is graded by the **official binary judge** + aggregation metrics,
scored on two axes — **answer accuracy** (autoeval vs gold answer) and
**retrieval recall** — implemented as **ID-membership recall** faithful to the
spec (are the *session IDs* the tools returned a superset of the gold
`answer_session_ids`?), not text-overlap F1.

- **Durable seed store.** Question seeds (summaries) persist to
  `data/benchmarks/generated/<question_id>.jsonl`. A replay is **zero-cost and
  model-agnostic** for the same eviction knobs: all questions load as
  "durable, 0 summarizer" — the summarizer never re-runs and seed tokens aren't
  consumed. **Footgun:** changing an eviction knob
  (`LOCOMO_TARGET_TOKENS` / `MAX_LINES` / `LOCOMO_SW_*`) *invalidates* the
  durable seed cache and forces a live re-summarize of all affected questions.
  `PAF_FRESH_SEED=1` forces it regardless. `tests/AGENTS.md` is the record.
- **ID-membership recall rule.** The three tools
  (`search_index`, `read_turn`, `search_conversation_history`) each emit
  `session=<id>`; recall scores tool-only (in-context answers score 0).
  Abstention is distinct from a miss (`UNANSWERABLE` rule in the subject prompt).
- **CLI flags.** `run_lmem.py` accepts **`--subject-model`** and
  `--judge-model`, taking precedence over the env vars
  `BEAM_LITE_SUBJECT_MODEL` / `LME_JUDGE_MODEL` (mirrors `test_beam_lite`'s
  `--arm`). **Swapping the subject model does NOT invalidate the durable seed
  store** — answers are re-scored against the persisted seeds, which is the
  intended benchmark design.
- **Eviction knobs** (LoCoMo): `LOCOMO_TARGET_TOKENS` / `MAX_LINES` /
  `LOCOMO_SW_MAX_MESSAGES` / `LOCOMO_SW_KEEP`.
- **Beam-Lite is demoted to a smoke test** in the harness (`test_beam_lite.py`,
  `--arm` selects `without` / `turn_index` / `memory` for parallel runs) — a
  quick regression surface, not the gate.

Benchmark harness scripts live in `tests/`:

```
tests/
  run_lmem.py          LongMemEval graded run (official binary judge + aggregation)
  tiny_lmem.py         fast lone-question gate for the real TurnSummarizer
  run_lmem_memoff.py   long-mem-off ablation
  run_locomo.py        LoCoMo eviction benchmark   locomo_scoring.py  F1+recall scorer
  resume_replay.py     pristine-backend replay harness
  recall_corpus.py / recall_test.py
```

## Layout

```
src/paf/
  capabilities/   extension seam — registry, profiles, builders
    turn_index/   episodic index (write gate + inject/recall tools + summarizer)
    search/       paf-owned BM25 ConversationSearch (search_conversation_history)
    memory/       paf-owned Memory, markdown backend (atomics inlined)
  observability/  Langfuse OTLP instrumentation (setup/shutdown)
  runtime.py      Session — UI-agnostic run lifecycle (build + trace + stream +
                  history + flush)
  specs.py        spec loader (agent name → YAML → Agent)
  config.py       pydantic-settings config + spec/root anchoring
specs/            declarative agent specs (*.yaml)
tests/            benchmark harness (run_lmem, run_locomo, tiny_lmem, ...)
```

## Run it

Dependencies resolve from a sibling `agent-foundation` package
(`[tool.uv.sources]` path) — the real workflow uses the local venv + env:

```bash
# one-time
cp .env.example .env        # OPENAI_API_KEY + optional Langfuse keys
PYTHONPATH= .venv/bin/pip install -e .   # or: uv sync

# interactive REPL
PYTHONPATH="" .venv/bin/python -m ui.cli
```

Prepend `set -a && . ./.env && set +a;` to export secrets for the runner. The
REPL keeps a durable conversation (`conversation_id "cli-primary"`); run it
twice with the same id to see persistence + recall — start, quit, restart, then
ask for something reachable only via `search_conversation_history`.

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

## Status

M1 shipped: thin, opinionated pydantic-ai V2 scaffold — capabilities/registry,
profiles, tiered compaction, durable flat-file `StepPersistence` + BM25
`ConversationSearch` over one shared store, Langfuse OTLP (no Logfire).

M2 (`feature/m2`), feature 1 — the **episodic turn index** — is shipped and
mature: per-turn synchronous indexing gated to terminal responses, a real
structured-output LLM summarizer with char-budget retry, one-line index
injected into context (recognition beats recall), `read_turn` back into
persistence, and `search_index` (BM25 over summaries + key terms, with a
conservative recency nudge).

The recall/memory stack is now **entirely paf-owned**:
- `ConversationSearch` is paf-local BM25 (`paf.capabilities.search`,
  `search_conversation_history` tool).
- `Memory` is re-homed to `paf.capabilities.memory` as a **paf-owned
  capability**, markdown backend, `agent-foundation` dropped for memory.
- All three recall tools emit `session=<id>`, enabling **ID-membership recall**
  scoring.

Benchmarking has pivoted **from Beam-Lite to a LongMemEval+LoCoMo gate** (the
north-star): official-binary-judge grading, ID-membership retrieval recall,
durable model-agnostic seed summaries, and CLI-first model selection
(`--subject-model` / `--judge-model`). Beam-Lite remains as a parallel smoke
arm.

## Roadmap

M2's design specs (the *living* plan, iterating as we build) live in the vault:
`History/M2-Builtins-Queues-Codebase-Map.md` and
`Designs/{Foundation-Package,Background-Queue-Storage,Codebase-Search-Call-Graph-Storage}-Spec.md`.

**The unifying rule (M2 §B):** *anything that **forms or transforms** knowledge
is an LLM **write**; anything that **reads** it is deterministic.* Every write
that isn't on the current answer's critical path goes **background** — that's
*why* the queue exists, not just an optimization. The turn-index summarizer is
already a pure, stateless, `turn_id`-keyed write: it sits **synchronously**
today only for testing, and moves to the background worker unchanged once the
queue lands.

1. **Background the writes (M2 §B).** Route the LLM-write consumers through a
   background queue, off the answer's critical path: the turn-index one-liner
   generator (already spec'd as a `TurnIndexJob`), compaction's
   `MdSummaryService`, and the **memory reconciler**. The enqueue site is *one*,
   owned by the hook; results are **eventually** consistent.
2. **Memory reconciler on top of the paf-owned Memory.** Layer SteadierPath's
   **LangMem** extractor + a background reconciler (the §B "off the critical
   path" write). mem0 rejected (paywall on good extraction); Graphiti not-a-fit
   for flat-file-first.
3. **Codebase map (M2 §C) — 4 phases (spec'd).** P0 `StorageService` seam →
   P1 tree-sitter symbol upgrade → P2 **call graph + `callers`** (the headline
   add) → P3 broadened `code_search` + aider-style `map_tokens` distilled map →
   P4 (opt) MCP tool-source. Rides the graph for free: dead-code detection,
   docstring enrichment (both ideas, not decisions).
4. **BEAM-style three-context ablation as a smoke (not the gate).** `bench/beam/`
   driver (judge ≠ answerer, stable judge): baseline (`message_history=`),
   working-memory (+background scratchpad), retrieval
   (`ConversationSearch`/`read_turn`) — kept in `test_beam_lite.py` as a fast
   regression surface. The authoritative correctness gate is LongMemEval (+LoCoMo
   for eviction).

**Secondary / deferred:** tuning the in-context read path (index-line injection
budget, eviction/ordering, composition with `Memory`); extracting turn-index into
a standalone installable package; M3 multi-agent (`Runner` over the same `Agent`
units); M4 security toolkit.