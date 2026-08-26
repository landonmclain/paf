# organizer — ARCHIVED (unhooked from the live memory path, kept importable)

**Status: archived 2026-08-26.** This package preserves the **memory-organizer**,
the **curated index** (`IndexStore` / `IndexEntry`), the **consolidation second
pass** (`run_consolidation` / `memory-consolidator`), and the **index renderer**
(`render_index` / `write_index_render`) — the entire "LLM-maintained curated
index" idea. It was **unhooked from `MemoryConsolidationWindow.compact()` on
2026-08-25** and the index was removed from the memory pipeline. It is kept here,
whole and importable, so the idea can be revisited later without rewriting it.

## Why it's archived (decision history)

The live memory path is now the **append-only journal + deterministic
`write_render`**:

```
owner (consolidate)  →  memory-journal.jsonl  →  write_render()  →  MEMORY.md
(append-only facts)         (source of truth)      (pure, no LLM)   (injected)
```

The archived pipeline was the three-layer / two-agent design:

```
owner (consolidate)  →  journal →  organizer (run_organizer)  →  curated index
                                                                    ↓
                                                   write_index_render() → MEMORY.md
```

It was removed because the **organizer was context-blind by construction**: the
owner extracts with the full conversation in context, but the organizer
reconciled from *strings + the index only* (no original conversation), which
drove **index churn** and **reliability failures** (most-notably the
"most recent wins" dedup losing good facts). Given the owner already scores an
importance tier (t1–t3) at write time, the extra index-maintenance pass added
LLM passes and nondeterminism without improving recall — and the **full 472-Q
gate benchmark** (0.9468, matching the old index ceiling) confirmed the journal+
render path is behaviorally equivalent or better with strictly less machinery.

The decision is documented in-repo in `tests/AGENTS.md` and in the vault note
`Benchmark/Tier-Sorted-Render-Benchmark.md`. If dedup/supersede/tier-sorting
ever returns, it should be a **deterministic, pure-code derivation** (the
`_pick_winner` logic is a good starting point), never another LLM maintenance
pass.

## What happened to "1st vs 2nd pass"

The naming confusion that prompted this archive:
- **`consolidate()`** in `src/paf/capabilities/memory/_consolidate.py` — the
  **owner**, first pass, appends facts to the journal. **STILL LIVE.**
- **`run_organizer()`** / **`run_consolidation()`** in this archive — the index
  maintenance **second pass**. **ARCHIVED.** (`run_consolidation` was a
  re-run of the organizer under a distinct `memory-consolidator` agent name.)

## Contents

- `_organize.py` — the full organizer: `IndexStore`, `IndexEntry`, decision
  types (`NewDecision` / `ConfirmDecision` / `SupersedeDecision`), `Stage`,
  `reconcile()`, `run_organizer()`, `run_consolidation()`, `render_index_text`,
  `_pick_winner`, `_build_index_toolset`, the near-match/consolidation
  candidate finders, and `render_journal_tail`.
- `render_index.py` — the curated-index renderer (`render_index`,
  `write_index_render`), split out of the live `_render.py` when it was
  unhooked so the live module keeps only the journal renderer.
- `tests/` — the full organizer test suite (47 tests, all pass) plus the live
  LLM drivers:
  - `test_memory_index.py`, `test_memory_index_render.py`,
    `test_memory_organizer_driver.py` — unit/deterministic tests.
  - `run_organizer_live.py` — LIVE LLM runner (spends real OpenRouter tokens;
    NOT part of pytest).
  - `_live_guard_test.py` — drives `_build_index_toolset` against the REAL
    live `data/memory` store; manual, NOT part of pytest.

## Kept importable (why / how)

Unlike the older `archive/turn_index/`, this code is **fully importable and
re-runnable**, not reference-only. Relative imports were converted to absolute
`paf.capabilities.memory.*` / `archive.organizer.*` so it resolves against the
live package. To use it:

```bash
# from the repo root
PYTHONPATH=.:archive .venv/bin/python -c "import archive.organizer"
# run the archived unit tests
PYTHONPATH=.:archive .venv/bin/python -m pytest archive/organizer/tests/test_memory_index.py
# run the archived live-LLM driver (real tokens; source .env first)
set -a && . ./.env && set +a
PYTHONPATH=.:archive .venv/bin/python -m archive.organizer.tests.run_organizer_live --fixture
```

## Notes / caveats

- `archive/` is **not** part of the installed package (`pyproject.toml` packages
  only `src/paf`), so nothing in the runtime imports this — it must be added to
  `sys.path` explicitly as above. That is intentional: it cannot be pulled into
  the live import graph by accident.
- `_organize.py` imports live siblings (`paf.capabilities.memory._journal`,
  `._atomic`, `._toolset`, `._consolidate.consolidate_model` for the model
  default), so it requires `paf` installed. If those live names drift in the
  future, this archive may need light touch-ups to re-import — the unit tests
  here are the check.
- `run_organizer_live.py` docstring mentions a `tests/run_organizer_live.py`
  path from before the move; the canonical location is now
  `archive/organizer/tests/run_organizer_live.py`.

## History / provenance

- Unhooked from the compaction hook: `10addf6` ("memory: unhook organizer +
  consolidation second pass; render MEMORY.md from journal").
- Moved into this archive: current commit on `feature/m2-prod-sync`.
- Original intro of the organizer: `43d74c0` ("memory: wire organizer + index
  render into compaction hook"); evolved through `eb824b9` (dedup + 2nd pass),
  `b567c84` (rate limiting), `d340b75` (owner tier rubric augments the design).
