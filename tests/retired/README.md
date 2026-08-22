# retired/

Benchmark runners/tests that were once useful but are no longer part of the
active pipeline. Kept for reference/archival per Landon's "archive, don't
destroy" preference — they are NOT collected by pytest (this dir has no
`test_*.py` files and is not imported by the active runners).

> Naming rule: any file here that is `test_*.py`-prefixed WILL be collected by
> pytest and break (its `from paf.capabilities.turn_index import ...` no longer
> resolves). So retired pytest modules use a NON-`test_` name (e.g.
> `recall_live_turn_index.py`, `recall_matrix_turn_index.py`) — the pytest
> collection pattern is `test_*.py`, and these deliberately don't match.

## Contents

- **tiny_lmem.py** — LoCoMo-style mini-runner that wired the real
  `TurnSummarizer` into `TurnIndexCapability` (the turn-index summary layer).
- **recall_matrix_turn_index.py** — the original deterministic Tier-1 challenge
  matrix for the turn-index retrieval tools (`search_index`, index ladder,
  escalation geometry C1–C4). Retired with the turn-index experiment; the two
  lasting tests (tool exposure + `read_turn`) were kept in
  `tests/test_recall_matrix.py`, rewritten without turn_index.
- **recall_live_turn_index.py** — the Tier-2 live recall test over the
  turn-index escalation ladder (needs a real model; `-m live`). Retired with
  turn index.
- **resume_replay.py** — benchmark runner wiring `TurnIndexCapability` +
  `TurnSummarizer` into the resume/replay path. Retired with turn index (turn
  index is no longer in the benchmark pipeline).

All four reference the retired turn-index machinery and exist for reference
should any of its mechanics need revisiting. See also `archive/turn_index/` in
the repo root (the archived capability source).

> Remove a retired file only when it has genuinely outlived its reference
> value, not merely because it's unused.
