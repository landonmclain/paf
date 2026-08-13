# AGENTS.md — paf benchmark harness (`tests/`)

Concise operating guide for any agent (Claude Code, Codex, Cursor, Hermes, ...)
working on the graded memory/retrieval benchmarks in this directory. These are
**live LLM benchmarks** — they cost real OpenRouter tokens and write real Langfuse
traces. Read before running.

## What this is (2026-08 — grading pivot)

The **graded gate** for M2 is **LongMemEval** only. LoCoMo and Beam-Lite are
**retired** (2026-08): they were cut because the real bottleneck is capture
fidelity + reasoning, which their token-budget/evidence ablations couldn't see.
See the vault note `.../Portfolio-GitHub/M2.md` → "Grading pivot".

- **The graded benchmark (the proof gate):**
  - `tests/run_lmem.py` — LongMemEval, official binary judge. Answer correctness
    via the VERBATIM `get_anscheck_prompt` port in `tests/lme_scoring.py` +
    retrieval recall via **ID-membership recall@k** over surfaced `session=`
    tags (no token-overlap F1). `LME_MAX_QUESTIONS` caps size.
  - Benchmark data in `data/benchmarks/lme_oracle.json` (+ durable per-question
    seeds in `data/benchmarks/generated/`, and `data/benchmarks/memory/`).
    Graded records in `tests/results/`. Oracle split ≠ full LongMemEval set —
    see vault `LongMemEval-Testing.md`.

### Shared harness builder — `tests/framework.py`

`tests/framework.py` is the **one source of truth** for the agent-assembly +
turn-runner plumbing (`_build_agent`, `_run_turn`) and the env knobs
(`ABLATE_TURN_INDEX`, `MEMORY`, `SUBJECT_MODEL`, `MAX_LINES`, the `LOCOMO_*`
eviction dials). It was carved out of the retired `tests/run_locomo.py` when
LoCoMo was cut. All runners (`run_lmem`, `tiny_lmem`) and the builder tests
(`test_build_agent`, `test_turn_index_ablation`) import from it — **edit it in
one place**, never maintain a parallel builder. The LoCoMo-only `main`/scoring
lived in `run_locomo.py`/`locomo_scoring.py`, both deleted.

`tests/lme_scoring.py` holds ONLY the official LongMemEval judge surface:
`LME_TYPES`, `get_lme_anscheck_prompt`, `lme_is_abstention`, `lme_parse_label`.

## Run the GRADED benchmark (LongMemEval)

Always run from the repo root. The venv is polluted by Hermes' leaked
`PYTHONPATH`, so **prefix every python invocation with `PYTHONPATH=`**.

```bash
cd /home/alice/projects/Personal/pydantic-agent-framework

# LongMemEval — 4-question run (default subject = deepseek-v4-flash:nitro;
# judge = gpt-4o-mini; TurnSummarizer wired always-on). ~30+ min with summaries.
# MEMORY defaults to ON (1) now — the memory arm injects durable MEMORY.md and
# reuses per-question snapshots (0 owner tokens). Explicitly disable for a
# no-memory ablation: add `MEMORY=0` (hard multi-session questions then fall to
# cold BM25 retrieval and can spin to the request cap — keep MEMORY on unless a
# no-memory baseline is the actual point).
set -a && . ./.env 2>/dev/null && set +a && \
  LME_MAX_QUESTIONS=4 PYTHONPATH= .venv/bin/python -u -m tests.run_lmem

# Swap the answering model (and/or judge) via CLI flag (overrides env):
set -a && . ./.env 2>/dev/null && set +a && \
  PYTHONPATH= .venv/bin/python -u -m tests.run_lmem \
  --subject-model openrouter:openai/gpt-5.6-luna-pro --judge-model openrouter:openai/gpt-4o-mini
```

`.env` supplies `OPENROUTER_API_KEY` (subject + judge models) and the Langfuse
keys/endpoint (tracing). `setup_observability()` returns `False` and prints a
warning if tracing isn't configured — graded JSON still writes.

**Run ONE question (iterate on a specific qid):** `LME_QIDS="<qid>"` overrides
sampling. MUST source `.env` first — run_lmem does NOT load it itself (bare
`OPENROUTER_API_KEY` → owner/agent `UserError`). Use `python -m tests.run_lmem`
(module form puts repo root on `sys.path`; the script form `python
tests/run_lmem.py` fails with `ModuleNotFoundError: tests`).

```bash
set -a && . ./.env 2>/dev/null && set +a && \
  LME_QIDS="51a45a95" PYTHONPATH= .venv/bin/python -u -m tests.run_lmem
```

**Memory re-use on a single qid:** a per-question durable memory snapshot is
auto-restored → `REUSED ... (0 owner tokens)` and the owner is skipped. To force
re-consolidation (e.g. after editing the consolidation prompt) add
`PAF_FRESH_MEMORY=1` — but NOT to isolate an *answer-guidance* change: the
recall/answer phase reads the same durable memory and a fresh consolidation is
unneeded overhead (and an extra fragile live-LLM step). Only force fresh memory
when the consolidation output itself is the thing under test.

**`MEMORY` is ON by default now** — no need to pass `MEMORY=1`. Every harness
that shares the builder defaults to the memory arm, so a bare invocation is the
fast baseline. Pass `MEMORY=0` for a no-memory ablation arm. If a multi-session
question ever spins in `search_index` (repeating the same two queries), that is
the no-memory cold-retrieval signature — re-check that `MEMORY=0` didn't leak
into the run — not a memory-arm bug.

**Bender guard (2026-08-13) — `RECALL_TOOL_CALLS_LIMIT` (default 25):**
`_run_turn` caps tool calls per recall turn. On a runaway retrieval loop that
would trip `request_limit=50` (the crash that killed earlier full-50 runs at
q16), instead of dying the run catches `UsageLimitExceeded`, repairs the
dangling tool calls (`_repair_dangling_tool_calls`), and hands the accumulated
context to a **zero-tool fallback agent** for a graceful best-effort answer.
This is how the q16 model-kits multi-session count and `aae3761f` (driving
hours) both previously crashed runs — now each caps at 24 tools and the run
completes all 50. Override with `RECALL_TOOL_CALLS_LIMIT=0`/`''` to disable, or
a lower number (e.g. 8) to test the cut rapidly. A `bailed` flag on the `_run_turn`
return marks the fallback path; note the fallback sets `usage.requests` but not
`total_tokens` (capped rows may report `tok=0`).

### Retrieval scoring (no token-overlap F1)

The retrieval/recall column is **ID-membership recall@k** only: does the surfaced
`session=<id>` set (from `source_session_id` provenance) contain the gold
`answer_session_ids`? Independent of the final answer. `_abs` questions are
excluded. There is **no text-presence fallback and no `f1_score`** — the old
LoCoMo-era F1 probe was removed with LoCoMo. A run whose records carry no
`answer_session_ids` simply scores recall 0.

### Answer-direction guidance (`_INSTRUCTIONS` in `turn_index/_capability.py`)

The recall agent's system guidance has three rules: CONFLICT (most-recent
user-stated value wins), UNANSWERABLE (say "not in our history" EXPLICITLY
FIRST when the fact isn't on record — **don't fabricate**), and INFERENCE
(2026-08-13) which SOFTENS UNANSWERABLE for the strongly-implied case: if
retrieved context never states the fact but single-mindedly implies one answer
(e.g. the whole conversation is about Target/the Cartwheel app, no other
retailer appears), give the hedged inference — *"Most likely Target, but you
never told me the exact location"* — rather than a bare refusal.

Verified: `51a45a95` flipped judge WRONG→CORRECT with this rule (gold=Target is
an *inference*, never stated; the model's earlier honest "not on record" answer
was marked wrong by the judge). The three rules collide on exactly these
inference-answer questions — UNANSWERABLE's honesty and the eval's reward for
asserting the implied answer can't both fully win; the INFERENCE RULE is the
compromise (keeps the honesty floor via "most likely", but leads with the useful
answer so the judge sees it). `_INSTRUCTIONS` is a raw-string block; keep it
that way (parens-tuple fragment editing trips on backslash escapes).

### Durable seed store (replay = near-zero seed tokens) — READ BEFORE CHANGING KNOBS

Each question's seed history is summarized ONCE and auto-persisted by
`run_lmem._persist_durable_summary` to
`data/benchmarks/generated/<question_id>.jsonl` (git-TRACKED). On the next run
`_load_durable_seed_summaries` reads it → **"durable, 0 summarizer"** → no
TurnSummarizer tokens for that question. A fresh clone re-runs zero-live-
summarizer because the store is checked in.

> **FOOTGUN — changing `LOCOMO_TARGET_TOKENS` / `MAX_LINES` (or any eviction /
> index knob, `LOCOMO_SW_*`, etc.) INVALIDATES the durable cache.** The durable
> summaries are keyed by `question_id` + the turn structure as it was summarized.
> Re-run with different eviction thresholds → the index shape differs → the stored
> summaries no longer match → `_persist_seed_direct` finds turns with no durable
> summary and **falls back to the live summarizer for ALL affected questions**
> (a real token + ~30min/time cost, one-time, then re-persisted).
>
> To change a knob and NOT burn tokens: kill `data/benchmarks/generated/*.jsonl`
> FIRST and accept a full live re-summarize, OR keep the same knob values and the
> replay stays zero-summarizer. Never tweak a knob and assume the replay is still
> cheap — it silently isn't. Replay is only zero-cost for the SAME config.

## Lint / sanity (no tokens)

```bash
PYTHONPATH= .venv/bin/ruff check tests/ --output-format=concise
PYTHONPATH= .venv/bin/python -m py_compile tests/run_lmem.py tests/framework.py
```

`tests/results/_archive/` holds invalid historical JSONs — do not trust them.

## Run the UNIT suite (pytest) — the PYTHONPATH footgun lives here too

```bash
PYTHONPATH= .venv/bin/python -m pytest -q        # the correct invocation (or `PYTHONPATH= pytest -q`)
```

**Why you can't just rely on `conftest.py` for this.** There IS a root `conftest.py`
(loads `.env` at import time), and it fixes the `.env` leakage — but it **cannot**
fix the PYTHONPATH leakage, and it's not a bug we can patch. Mechanical reason:

- Bare `pytest` (the binary on `$PATH`) executes Hermes' interpreter
  `/home/alice/.hermes/hermes-agent/venv/bin/python3` — **Python 3.11** — not the
  project's `./.venv` (**Python 3.12**). This repo is uv-managed (`uv` 0.11.26,
  `pyvenv.cfg` home = `/usr/bin`, py 3.12.3); the world-PATH `pytest` is the wrong
  interpreter entirely, never mind the path.
- The leaked `PYTHONPATH=/home/alice/.hermes/hermes-agent/venv/lib/python3.11/site-packages`
  (and `/home/alice/.hermes/hermes-agent`) is injected by Hermes into **every**
  terminal-catalog call. That 3.11 site-packages lands **ahead of** the project's
  `src` on `sys.path`.
- So importing the 3.11-built `pydantic_core` (`_pydantic_core.cpython-311-….so`)
  into the 3.12 process raises `ModuleNotFoundError: pydantic_core._pydantic_core`
  — a C-extension **ABI mismatch**, not a `sys.path` ordering a runtime tweak survives.
- **conftest can't reach early enough:** conftest is a pytest *plugin*. pytest imports
  `pydantic`/`pydantic_core` (pydantic-graph, fixtures) while bootstrapping the plugin
  env — **before any conftest body runs**. The import that crashes is the very one
  pytest needs to load conftest. And conftest can't switch the interpreter or re-import
  a wrong-ABI wheel mid-process anyway. The fix must act on the *interpreter/`.so`*,
  which is exactly what `PYTHONPATH=` + the project venv does.

**Also note:** `uv run pytest` ALONE is NOT enough — verified that uv runs the correct
3.12 `./.venv/bin/python3` but still inherits the leaked `PYTHONPATH` (uv doesn't strip
it), so `uv run pytest` reproduces the same 26 collection errors. You must clear it:
`PYTHONPATH= uv run pytest` works (uv then also skips its own sync when the venv is
fresh). The minimal, always-correct form remains `PYTHONPATH= .venv/bin/python -m pytest`.

**Pitfall when resuming an interrupted run:** don't lose the prefix on the LME
harness commands above — they're all under the same leaked-PYTHONPATH rule. A bare
`python -m tests.run_lmem` from Hermes will hit the same `_pydantic_core` import error.

**Checking the leak is actually the cause** (if collection errors ever seem unrelated):
`head -1 "$(which pytest)"` prints the shebang — if it's the Hermes 3.11 path, that's
it; `PYTHONPATH= .venv/bin/python -m pytest` then proves it by passing (145 passed, 4
deselected as of 2026-08-13).

> **Still trips every time** — the gotcha is that **bare `pytest` / bare
> `python -m tests.run_lmem` (no `PYTHONPATH=` prefix) is ALWAYS the wrong
> interpreter**, even when the doc says "145 passed". The harness/CI/agents keep
> invoking `pytest` without the prefix and hit the ABI error. The RULE to remember:
> **in this repo, a python/pytest invocation without `PYTHONPATH=` is invalid by
> default.** And note the prefix form matters: `PYTHONPATH=` (empty) and
> `PYTHONPATH=""` (empty string) are equivalent in shell — `PYTHONPATH= .venv/bin/python -m pytest`
> is the canonical form. Also: even `.venv/bin/python -c "import pydantic_ai"`
> (no prefix) fails on the leaked 3.11 site-packages — the prefix is required for
> ANY venv python in this repo, not just pytest.

## Analytical framing (verified 2026-08-10, via the retired Beam-Lite cargo)

Kept as historical reasoning:

- `turn_index` recalls 4/4 with capture=0 — the index retrieves on demand, so it
  never needs the model to persist anything. It is the "wall."
- `memory` caps at 0.75 because the model abstains on C4 (seed says *"don't make
  a note of it"*) — captured=3/4, then nothing to recall. **The memory arm's
  ceiling is capture discipline, not retrieval quality.** Improving it = tuning
  capture-judgment on abstain cases, NOT tuning the indexer. This capture-vs-
  retrieval distinction is exactly why the graded gate moved to LongMemEval
  (it scores capture fidelity + reasoning, which Beam-Lite's token-budget
  ablation can't see).
- Full reasoning trail: vault note
  `Documents/notes/Personal/Portfolio-GitHub/Turn-Index-Testing.md`.
