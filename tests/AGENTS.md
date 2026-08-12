# AGENTS.md — paf benchmark harness (`tests/`)

Concise operating guide for any agent (Claude Code, Codex, Cursor, Hermes, ...)
working on the graded memory/retrieval benchmarks in this directory. These are
**live LLM benchmarks** — they cost real OpenRouter tokens and write real Langfuse
traces. Read before running.

## What this is (2026-08-11 — grading pivot)

The **graded gate** for M2 is now **LongMemEval** and **LoCoMo**, not Beam-Lite.
Beam-Lite is demoted to a fast smoke/ablation harness. See the vault note
`.../Portfolio-GitHub/M2.md` → "Grading pivot (2026-08-11)".

- **Graded benchmarks (the proof gate):**
  - `tests/run_lmem.py` — LongMemEval, official binary judge (a port of
    `get_anscheck_prompt` → `tests/locomo_scoring.py` `_lme_judge`). Answer
    correctness per-direction + aggregation metrics. `LME_MAX_QUESTIONS` caps size.
  - `tests/run_locomo.py` — LoCoMo, token-level F1 vs gold + separate retrieval
    recall scored from tool outputs. Uses `tests/locomo_scoring.py`.
  - Benchmark data in `data/benchmarks/` (`lme_oracle.json`, `locomo10.json`,
    `lme_oracle` subset). Graded records in `tests/results/`. Oracle split ≠ full
    LongMemEval set — see vault `LongMemEval-Testing.md`.
- **Smoke/ablation (no longer the gate):** `tests/test_beam_lite.py` — a fast
  graded-lite ablation over the C1–C4 ladder comparing storage arms:
  - `without` — baseline, no storage.
  - `turn_index` — TurnIndex / ConversationSearch indexing layer.
  - `memory` — Memory + StepPersistence (+ `inject_memory`).
  Each arm runs C1–C4 (4 cells). Scores: `atom_mean`, `tool_gate_rate`,
  `capture_rate`; `tool` vs `guess` shows whether the judge saw a tool invocation.

## Run the GRADED benchmarks (LongMemEval, LoCoMo)

Always run from the repo root. The venv is polluted by Hermes' leaked
`PYTHONPATH`, so **prefix every python invocation with `PYTHONPATH=`**.

```bash
cd /home/alice/projects/Personal/pydantic-agent-framework

# LongMemEval — 4-question run (default subject = deepseek-v4-flash:nitro;
# judge = gpt-4o-mini; TurnSummarizer wired always-on). ~30+ min with summaries.
set -a && . ./.env 2>/dev/null && set +a && \
  LME_MAX_QUESTIONS=4 PYTHONPATH= .venv/bin/python -u -m tests.run_lmem

# LoCoMo — eviction knobs in env: LOCOMO_TARGET_TOKENS / MAX_LINES.
set -a && . ./.env 2>/dev/null && set +a && \
  PYTHONPATH= .venv/bin/python -u -m tests.run_locomo
```

`.env` supplies `OPENROUTER_API_KEY` (subject + judge models) and the Langfuse
keys/endpoint (tracing). `setup_observability()` returns `False` and prints a
warning if tracing isn't configured — graded JSON still writes.

## Run the SMOKE harness (Beam-Lite)

```bash
cd /home/alice/projects/Personal/pydantic-agent-framework

# Full run, all 3 arms (12 cells): ~7 min
set -a && . ./.env 2>/dev/null && set +a && \
  PYTHONPATH= .venv/bin/python -u -m tests.test_beam_lite

# FASTER: split by arm and run the 3 shards in PARALLEL (~2-3 min wall-clock).
for A in without turn_index memory; do
  set -a && . ./.env && set +a && PYTHONPATH= \
    .venv/bin/python -u -m tests.test_beam_lite --arm "$A" &
done
wait
```

### Beam-Lite CLI

- `--arm {without,turn_index,memory}` — repeatable; restrict to these arm(s).
  Default = all arms. **Single-arm shards omit `delta_vs_baseline`** (meaningless
  without the `without` baseline in the same run); merge shards first.

### Merge Beam-Lite shards → single aggregate

`main()` writes per-arm JSON for a split run; there is **no built-in merge in
the module**. A shard's `arms.<arm>` block is already the correct aggregate; to
reconstruct the full picture, collect each arm's block + `rows` into one payload
and compute `delta_vs_baseline` against `without` (Δ atom / Δ gate / Δ tokens)
manually. Merged artifact lives at `tests/results/beam-lite-merged-<stamp>.json`.

## Lint / sanity (no tokens)

```bash
PYTHONPATH= .venv/bin/ruff check tests/ --output-format=concise
PYTHONPATH= .venv/bin/python -m py_compile tests/run_lmem.py tests/run_locomo.py tests/test_beam_lite.py
```

`tests/results/_archive/` holds invalid historical JSONs — do not trust them.

## Analytical framing (verified 2026-08-10, Beam-Lite)

- `turn_index` recalls 4/4 with capture=0 — the index retrieves on demand, so it
  never needs the model to persist anything. It is the "wall."
- `memory` caps at 0.75 because the model abstains on C4 (seed says *"don't make
  a note of it"*) — captured=3/4, then nothing to recall. **The memory arm's
  ceiling is capture discipline, not retrieval quality.** Improving it = tuning
  capture-judgment on abstain cases, NOT tuning the indexer. This capture-vs-
  retrieval distinction is exactly why the graded gate moved to LongMemEval +
  LoCoMo (they score capture fidelity + reasoning, which Beam-Lite's token-budget
  ablation can't see).
- Full reasoning trail: vault note
  `Documents/notes/Personal/Portfolio-GitHub/Turn-Index-Testing.md`.