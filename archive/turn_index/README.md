# turn_index — ARCHIVED (failed experiment)

**Status: archived 2026-08-21.** The turn-index capability (one-line-per-turn
index injected into context + a `read_turn` resolver) was a failed experiment.
It had "good stuff and ideas" but did not help — the M2 memory pipeline
(journal → curated index → injected MEMORY.md) is the superior recall path and
the versions here are dead in production.

## Why it's here
Kept as an archive (not deleted) to preserve the genuinely useful ideas before
they are lost. Moved out of `src/paf/capabilities/` so it cannot be imported by
the live tree (it is no longer registered in `registry.py` / `build.py`).

## Salvaged ideas (now live elsewhere)
- **`user_anchor_indices`** — turn-aware pointer addressing (ordinal of user-turn
  anchors in a message list). Reparented into `src/paf/capabilities/_turns.py`
  (shared); used by `memory/_consolidate.py` and `search/_toolset.py`.
- **Marker-strip convention** — `_markers.py` was already extracted to
  `src/paf/capabilities/_markers.py` before archiving; shared by `search/` and
  `memory/`.
- **`read_turn` full-content read primitive** — lives on in
  `src/paf/capabilities/search/_reader.py` (owned by `search`, which is the
  permanent retrieval layer).
- **Good ideas captured but NOT ported** (candidate patterns for the future
  codebase-map capability): the **BlockRef byte-pointer / "query-the-pointer"
  deferred-IO model** (symbol = cheap pointer; full content read only on demand)
  and the **StorageService seam** (append-only JSONL + per-row atomic append).
  See the codebase History vault note for the full write-up.

## Dependency warning
`_store.py` imports `agent_foundation.storage` (FlatFileStorage/StorageService).
That package is **no longer a pyproject dependency** of paf (it was dropped when
this capability was archived; `memory/_atomic.py` carries the inlined port). The
archived `_store.py` will not import without `agent-foundation` reinstalled — it
is kept only for reference.

## Files
- `_capability.py` — the capability (this was registered as `TurnIndex`).
- `_store.py` — append-only turn index store (the `agent-foundation` seam consumer).
- `_summarizer.py` / `_generate.py` / `_extract.py` — per-turn summarization.
- `_toolset.py` — `TurnIndexToolset`.
- `_builders.py` — `build_turn_index`.
- `specs/turn-index.yaml` — the original spec.
