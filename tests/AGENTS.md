# AGENTS.md — paf benchmark harness (`tests/`)

Concise operating guide for any agent (Claude Code, Codex, Cursor, Hermes, ...)
working on the graded memory/retrieval benchmarks in this directory. These are
**live LLM benchmarks** — they cost real OpenRouter tokens and write real Langfuse
traces. Read before running.

> **When unsure, lean on the traces, not the logs.** A run's stdout/log is the
> harness's *narrative*; the Langfuse ClickHouse data is what actually happened
> (which tool calls fired, in what order, how many LLM generations, what the model
> actually saw in its input, error/retry deltas). When a log line and the traces
> disagree — or the log is ambiguous (e.g. a wrapped/backgrounded process, an
> interrupted tail, a `bash &` that detached the real child) — the **traces win**:
> query `langfuse-clickhouse-1` for the run's `memory-owner run` / `lmem.recall.1` /
> `lmem.judge.1` traces and read the `observations` for generation counts and tool
> calls before trusting any log. Hermes agents can load the
> `hermes-local-observability` skill (`references/clickhouse-trace-query.md`) for the
> exact query patterns; non-Hermes agents should clickhouse-query
> `langfuse-clickhouse-1` directly (`docker exec langfuse-clickhouse-1 clickhouse-client`)
> filtering `traces` by `name`/`session_id`/timestamp window. A trace showing a
> normal 1–3 generation-call owner is a *sane* consolidation; a run that spiked to
> ~50 identical writes is the old chunk-loop — the traces distinguish them
> instantly where the flat log cannot.

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

> **TWO-PASS CONVENTION (the operating rule, verified via traces 2026-08-25):**
> **build once, then recall read-only.** On a restored durable snapshot the
> `memory-owner` does **NOT** fire — the recall trace is inject-only (`memory.inject`
> → `JournalViewStore`, `outcome=ok`, `main_truncated=false`, zero `memory-owner run`
> observations), which is the cheap/desirable path. Fresh-built qids each carry one
> `memory-owner run` (the build cost) — so the memory-owner firing on a recall turn
> is *always* a sign that qid had no durable snapshot yet (a build pass), never a
> bug. To batch-run questions with minimal owner cost, let the first run build their
> durables, then a second run on the same questions is owner-free recall. Only add
> `PAF_FRESH_MEMORY=1` when the consolidation output itself is under test — a fresh
> consolidation on a restored qid is a redundant live-LLM cost.

**Chunked seed consolidation (2026-08-16, DEFAULTS ON):** the owner now
consolidates the seed in **token-bounded, message-boundary-aligned chunks**
(`consolidate_conversation_chunked`, `src/paf/capabilities/memory/_consolidate.py`)
instead of one giant pass, so single-mention thematic clusters (e.g. a "video
editing / Premiere" preference dropped when a dense "photography" theme
dominates) survive the owner's single-shot output budget. Each chunk pass runs
against the SAME store with `inject_memory=True`, so later chunks merge into
accumulated memory (append / newest-wins) rather than re-derive. Cut on whole
message `User:`/`Assistant:` boundaries only — never mid-message — so multi-turn
threads aren't bisected. Knobs (env):
- `LME_CONSOL_CHUNK=0` → restore the original one-shot `consolidate_conversation`.
- `LME_CONSOL_CHUNK_TOKENS` → chars/4 token budget per chunk (default `4000`).
The live/eviction `consolidate()` path is UNCHANGED (still `inject_memory=False`,
tool-read owner).

**`MEMORY` is ON by default now** — no need to pass `MEMORY=1`. Every harness
that shares the builder defaults to the memory arm, so a bare invocation is the
fast baseline. Pass `MEMORY=0` for a no-memory ablation arm. If a multi-session
question ever spins in `search_index` (repeating the same two queries), that is
the no-memory cold-retrieval signature — re-check that `MEMORY=0` didn't leak
into the run — not a memory-arm bug.

**Owner no-progress read guard (2026-08-16) — `_guard_repeat` in `_toolset.py`:**
the memory-owner consolidates by WRITING, so `read_memory`/`search_memory` are
only preparation. If the owner issues **3 consecutive read-only calls with no
intervening write**, the tool returns a redirect nudge ("it is time to write");
at **10 consecutive reads** it raises `ModelRetry`, which pydantic-ai escalates
(after `_OWNER_TOOL_MAX_RETRIES=5`) to `UnexpectedModelBehavior` and the chunk
loop in `_consolidate.py` aborts that chunk gracefully. A write
(`write_memory`/`delete_memory`) resets the window; a different sig does NOT
reset it (alternating `read`/`search` is itself a stall). This keys on *progress*,
not identical args, so the 2026-08-16 chunk-8 50-read runaway can't recur. Scope:
owner agent only — recall/live agents are untouched.

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

Verified: `51a45a95` flipped judge WRONG→CORRECT with this rule (gold=Target is an
*inference*, never stated; the model's earlier honest "not on record" answer
was marked wrong by the judge). The three rules collide on exactly these
inference-answer questions — UNANSWERABLE's honesty and the eval's reward for
asserting the implied answer can't both fully win; the INFERENCE RULE is the
compromise (keeps the honesty floor via "most likely", but leads with the useful
answer so the judge sees it). `_INSTRUCTIONS` is a raw-string block; keep it
that way (parens-tuple fragment editing trips on backslash escapes).

> **MEASURED DOWNSIDE of INFERENCE+ORDER (accepted 2026-08-25):** on the
> *unanswerable* ordering class (two events in separate sessions with no shared
> timeline, gold=`None`, e.g. `gpt4_70e84552` "fixing fence or trimming hooves
> first"), the INFERENCE rule ("prefer hedged inference over bare refusal") +
> ORDER rule ("state the dates being compared before giving the order") push the
> model to **over-commit and fabricate an ordering it can't justify** even when it
> had the right instinct ("I want to be honest about the second part"). Result:
> it answers instead of declaring unanswerable → judge marks it wrong.
> **Decision (Landon, definitive):** leave the prompt as-is; accept
> temporal-reasoning/unanswerable as a **known-miss-by-design** class — not a
> TODO, not a regression. The tier-render work targets factual/preference/
> knowledge-update classes, not this ordering class.

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

## Cross-run variance: same-config runs differ by ~4.5pt (findings, 2026-08-15)

Two full 472-Q gate runs, **identical params** (same 472 qids, subject
`openrouter:deepseek/deepseek-v4-flash-0731:nitro`, judge `gpt-4o-mini`,
`ABLATE_TURN_INDEX=1`, eviction 5000/6/3, max_lines=10, same oracle data file),
recorded **0.9043** (`lmem-20260814-230559.json`) and **0.9468**
(`lmem-20260815-121016.json`). Net +0.045 is answer-side judgment, not
retrieval: recall held (0.9964→0.9923), and 27 qids flipped wrong→right / 7
right→wrong, gains spread across types (multi-session +9, single-session-
assistant +7) with bilateral regression elsewhere. Honest band: **~91–95% on
this setup; a single run carries ±2pt variance.**

**Root cause = provider routing, not code.** Subject is pinned
`temperature: 0, max_tokens: 12288` (`tests/framework.py:330`, confirmed in
ClickHouse `model_parameters={"temperature":0}`), but **no `top_p`/`top_k` is
sent** — so the selected provider's own sampler defaults apply and can differ
per provider. `:nitro` = `provider.sort:"throughput"` (OpenRouter routing
shortcut), so routing can bounce between upstream providers (Together /
Fireworks / DeepInfra / ...) across runs/requests. OpenRouter explicitly
allows providers to serve the **same model slug at different quantization**
(`int4`/`fp8`/`fp16`/`bf16`/`fp32`) — FP4 vs FP16 of identical weights is
enough to change argmax outputs at temperature 0, especially on a small
"flash"-class model. This is documented as "the hidden quality variable" by
OpenRouter and is the likely driver of the 27-flip scramble. Reference (the
exact scenario): LessWrong, "Not Pinning Your OpenRouter Provider Might
Invalidate Your Research."

**Traces do NOT capture the provider.** Confirmed in ClickHouse
(`langfuse-clickhouse-1`): `internal_model_id` is **NULL on all ~14,849 deepseek
GENERATION spans** (only the gpt-4o-mini judge path records it). No column
records which provider/precision served each deepseek call; the `:nitro`
suffix is stripped from `provided_model_name`. So routing/provider bounce is
**invisible retroactively** in Langfuse — provider attribution would require
capturing the OpenRouter response `provider`/`host` fields into traced metadata
(a harness change) going forward.

### Fix (decided 2026-08-15) — pin provider + quantization, add sampler knobs

To collapse the same-config variance to near-zero for future runs, the agreed
change (pending application to `tests/framework.py` model settings):

```python
provider = {
    "order": ["DeepInfra", "Parasail", "CoreWeave"],
    "quantizations": ["fp8"],
    "allow_fallbacks": True,
}
# + sampler determinism: top_p=1.0, fixed seed (v4-flash-0731 also supports top_k/min_p/repetition_penalty)
```

- **fp8 is the recommended precision** (no fp16 provider for this model; bf16 = Morph only = niche
  host). DeepInfra primary (cheap + solid), Parasail + CoreWeave as precision-identical fp8
  fallbacks (Landon vouches for both). **Novita explicitly excluded** (unsure of it; no price edge).
- `quantizations: ["fp8"]` is the safety net — even with fallbacks, OpenRouter can never route to
  an fp4 box. Also pin the `gpt-4o-mini` judge to a single provider so the *judge* doesn't flap
  while measuring the *subject*.
- Full provider × quantization map (28 providers) + rationale in the vault note
  `.../Portfolio-GitHub/OpenRouter-Provider-Quantization-Nondeterminism.md`.

**Caveat:** pinning a single provider trades uptime for determinism (a pinned host down = run fails,
no graceful route). For a reproducibility benchmark that is the correct tradeoff.

## Memory render path (current architecture — 2026-08, tier-sorted)

The persisted memory a recall turn sees is a **deterministic, pure-journal render** —
no LLM, no index in the path. This supersedes the earlier organizer/index stage (see
the deprecation note under "Memory organizer live test" below).

- **Append-only journal** (`memory-journal.jsonl`) is the **sole source of truth**.
  The memory-owner consolidates new seed turns ONCE into journal rows that each carry
  a **required `tier`** (`MEMORY_TIERS = {t1, t2, t3}` in `_journal.py` — no default,
  no backward-compat), plus intent/importance rubric in `_consolidate.py`:
  - **t1 = stable/person-level** (supersede-only): identity, preferences, constraints.
  - **t2 = goals/current-work** (superseded on shift): objectives, active context.
  - **t3 = ephemeral/recent** (uncertainty never drops, it goes here).
  - Completeness rule = canonical self-contained facts; uncertainty → t3.
- **`write_render(journal, path)`** (`_window.py`) renders `MEMORY.md` deterministically:
  groups rows by type, sorts by `born_on`, emits `## t1 / ## t2 / ## t3` sections,
  **newest-first (`born_on` desc) within tier**, line shape
  `- <fact> — <born_on> — <source_ref>`. **Decouples tier position from recency** — a
  newer t3 line appears AFTER all t1/t2 lines; true recency is the `born_on` date.
- **The unhook (commit `10addf6`):** the organizer and the owner/consolidator second
  pass no longer run in `MemoryConsolidationWindow.compact()`. `compact()` calls the
  owner-only consolidate → append-only journal, then `write_render` → `MEMORY.md`.
  No `IndexStore`, no `run_organizer`, no `write_index_render`. Rationale: the owner is
  context-blind by construction (extracts with full conversation, reconciles from
  strings+index only) → index churn + reliability failures (lost good facts). Any future
  dedup/supersede/tier-sort must be a **deterministic pure-code derivation**, never an
  LLM pass.
- **`JournalViewStore` (`_view.py`)** reads the root `MEMORY.md` verbatim for injection —
  producer-agnostic; a recall agent always sees the last rendered file. `render_journal_tail`
  reads only `type/fact/born_on/source_ref`. The benchmark `SUBJECT_SYSTEM_PROMPT`
  (in `tests/framework.py`, the IN CONTEXT+MEMORY rule) names this tier ladder and the
  `t1→t2→t3` ordering + target_tokens budget.
- **Living-branch fact:** this is all on `feature/m2-prod-sync` (live branch); the owner
  tier rubric + `_journal.tier` landed in `d340b75`, the deterministic render+unhook in
  `10addf6`/`64d5f32`. Benchmarks on a **fresh durable stem** (`LME_DATA_FILE=...` →
  `data/benchmarks/memory-<stem>/`) exercise the tier render end-to-end.

## Memory organizer live test — `tests/run_organizer_live.py`

> **DEPRECATED / SUPERSEDED (2026-08-25):** the organizer and `memory-index.json` were
> **unhooked** from the memory path (`10addf6`) — compaction no longer runs
> `run_organizer`/`write_index_render`; the journal + `write_render` replace it. This
> section is retained only as historical reference for running the (now-orphaned)
> organizer classify brain if it's ever resurrected as a deterministic derivation. It
> does NOT reflect the current benchmark memory path.

The memory **organizer** (`src/paf/capabilities/memory/_organize.py`) is the
sibling sub-agent that maintains the curated `memory-index.json` from the
append-only journal tail (NEW / CONFIRM / SUPERSEDE, prune, watermark advanced
atomically at `finalize_index`). Its mechanical rules are pinned by the
deterministic `tests/test_memory_index.py`; the live LLM *classify* brain is
exercised here. Cost: real OpenRouter tokens — NOT part of the plain pytest run.

```bash
# Disposable fixture (2 batches, ~17 records) exercising all classifications:
PYTHONPATH= .venv/bin/python data/benchmarks/memory/organizer-live/generate_fixture.py
set -a && . ./.env 2>/dev/null && set +a && \
  PYTHONPATH= .venv/bin/python -m tests.run_organizer_live --fixture

# First index over the REAL data/memory journal (adds memory-index.json there):
set -a && . ./.env 2>/dev/null && set +a && \
  PYTHONPATH= .venv/bin/python -m tests.run_organizer_live --live
```

The organizer shares the owner's model by default (`ORGANIZE_MODEL` unset →
`consol_model`); set `ORGANIZE_MODEL=openrouter:<higher-level>` to override.
Run-end auto-trigger threshold: `LME_ORGANIZE_BATCH` (default 25 new lines).
