# retired/

Benchmark runners/tests that were once useful but are no longer part of the
active pipeline. Kept for reference/archival per Landon's "archive, don't
destroy" preference — they are NOT collected by pytest (this dir has no
`test_*.py` files and is not imported by the active runners).

## Contents

- **tiny_lmem.py** — LoCoMo-style mini-runner that wired the real
  `TurnSummarizer` into `TurnIndexCapability` (the turn-index summary layer).
  Retired when the turn index was removed from the benchmark pipeline
  (2026-08-17): the benchmark retrieval ladder is now purely
  `search_conversation_history` / `read_turn` (ConversationSearch) + durable
  Memory. Not deleted — archived here should the turn-index mechanics ever
  need reference.

> Remove a retired file only when it has genuinely outlived its reference
> value, not merely because it's unused.
