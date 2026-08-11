# AGENTS.md — paf benchmark harness (`tests/`)

Concise operating guide for any agent (Claude Code, Codex, Cursor, Hermes, ...)
working on the graded mini-BEAM benchmark in this directory. This is a **live
LLM benchmark** — it costs real OpenRouter tokens and writes real Langfuse
traces. Read before running.

## What this is

`tests/test_beam_lite.py` is a graded retrieval benchmark over the C1–C4
challenge ladder. It ablates storage mechanisms (arms) against a fixed recall
corpus:

- `without` — baseline, no storage.
- `turn_index` — TurnIndex / ConversationSearch indexing layer.
- `memory` — Memory + StepPersistence (+ `inject_memory`).

Each arm runs all of C1–C4 (4 cells). Scores: `atom_mean` (recall quality),
`tool_gate_rate` (did the right mechanism fire), `capture_rate` (did the model
persist the fact). Grounding tag (`tool` vs `guess`) shows whether the judge saw
a tool invocation behind the answer.

## Run commands (the important part)

Always run from the repo root. The venv is polluted by Hermes' leaked
`PYTHONPATH`, so **prefix every python invocation with `PYTHONPATH=`**.

```bash
cd /home/alice/projects/Personal/pydantic-agent-framework

# Full run, all 3 arms (12 cells): ~7 min
set -a && . ./.env 2>/dev/null && set +a && \
  PYTHONPATH= .venv/bin/python -u -m tests.test_beam_lite

# FASTER: split by arm and run the 3 shards in PARALLEL (~2-3 min wall-clock).
# Each shard runs only its 4 cells and writes beam-lite-<arm>-<stamp>.json.
for A in without turn_index memory; do
  set -a && . ./.env && set +a && PYTHONPATH= \
    .venv/bin/python -u -m tests.test_beam_lite --arm "$A" &
done
wait
```

`.env` supplies `OPENROUTER_API_KEY` (subject + judge models) and the Langfuse
keys/endpoint (optional tracing). `setup_observability()` returns `False` (and
prints a warning) if tracing isn't configured — JSON still writes.

## CLI

- `--arm {without,turn_index,memory}` — repeatable; restrict to these arm(s).
  Default = all arms. **Single-arm shards omit `delta_vs_baseline`** (meaningless
  without the `without` baseline in the same run); merge shards first.

## Merge shards → single aggregate

`main()` writes per-arm JSON for a split run; there is **no built-in merge in
the module**. A shard's `arms.<arm>` block is already the correct aggregate; to
reconstruct the full picture, collect each arm's block + `rows` into one payload
and compute `delta_vs_baseline` against `without` (Δ atom / Δ gate / Δ tokens)
manually. Merged artifact lives at `tests/results/beam-lite-merged-<stamp>.json`.

## Lint / sanity (no tokens)

```bash
PYTHONPATH= .venv/bin/ruff check tests/test_beam_lite.py --output-format=concise
PYTHONPATH= .venv/bin/python -m py_compile tests/test_beam_lite.py
```

`tests/results/_archive/` holds invalid historical JSONs — do not trust them.

## Analytical framing (verified 2026-08-10)

- `turn_index` recalls 4/4 with capture=0 — the index retrieves on demand, so it
  never needs the model to persist anything. It is the "wall."
- `memory` caps at 0.75 because the model abstains on C4 (seed says *"don't make
  a note of it"*) — captured=3/4, then nothing to recall. **The memory arm's
  ceiling is capture discipline, not retrieval quality.** Improving it = tuning
  capture-judgment on abstain cases, NOT tuning the indexer.
- Full reasoning trail: vault note
  `Documents/notes/Personal/Portfolio-GitHub/Turn-Index-Testing.md`.
