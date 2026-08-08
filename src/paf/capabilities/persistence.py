"""Persistence + recall + turn-index assembly — one shared store for write, search, and read_turn.

Like :mod:`paf.capabilities.compaction`, this owns capabilities that can't cleanly
resolve from YAML because they need real Python objects as arguments (see
``build.py`` for the full rationale on the ``capabilities=`` seam). For M1 that is
`ConversationSearch` and `TurnSummaryIndex`: their ``source: HistorySource`` needs
a real ``SnapshotHistorySource(store)`` object, and no harness recall class overrides
``from_spec`` (and `TurnSummaryIndex`, though it DOES override `from_spec`, is wired
here to share the live store) — so they are assembled here, in Python.

`StepPersistence` *could* resolve from YAML (it overrides ``from_spec``), but it is
built here too so it shares ONE ``FileStepStore`` with `ConversationSearch` and
`TurnSummaryIndex`. Sharing the identical store instance guarantees `ConversationSearch`
searches — and `read_turn` resolves — exactly the history `StepPersistence` writes:
no divergence between the write path and the recall-corpus read path.

This module owns that assembly. ``build_persistence()`` builds the single shared store
and returns the persistence + recall + turn-index capabilities wired over it;
``build_runtime_capabilities()`` (in ``build.py``) merges them with ``build_compaction()``
into the agent.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_ai_harness.conversation_search import (
    ConversationSearch,
    SnapshotHistorySource,
)
from pydantic_ai_harness.step_persistence import FileStepStore, StepPersistence

from paf.capabilities.turn_summary import TurnIndexStore, TurnSummaryIndex
from paf.config import PROJECT_ROOT, settings

# Canonical store root for M1 persistence + recall. A directory of flat files
# (JSONL events, JSON snapshots per run) — greppable/inspectable, no DB to
# administer. One instance, handed to both StepPersistence (writer) and
# ConversationSearch (reader), so the search corpus is exactly the persisted
# history. Anchored to the repo root (not CWD).
_STORE_DIR = PROJECT_ROOT / Path(settings.persistence_dir)

# The append-only one-line turn index lives next to the step store, not inside
# it (they are two different write surfaces). Anchored to the repo root.
_INDEX_DIR = PROJECT_ROOT / ".turn-index"


def _build_store() -> FileStepStore:
    """Build the single shared persistence store."""
    _STORE_DIR.mkdir(parents=True, exist_ok=True)
    return FileStepStore(directory=_STORE_DIR)


def build_persistence() -> list:
    """Assemble the persistence + recall capabilities over one shared store.

    Returns `StepPersistence` (the writer), `ConversationSearch` (the lossless
    recall reader), and `TurnSummaryIndex` (the one-line-per-turn index + gated
    `read_turn`) all wired to the SAME ``FileStepStore``, so the model can
    retrieve anything compaction (sliding window or summary) dropped from the
    live context, plus anything from past runs — and both the search corpus and
    the `read_turn` resolution target provably match what gets persisted.
    """
    store = _build_store()
    return [
        StepPersistence(store=store, agent_name="assistant"),
        # Lossless recall over the SAME store StepPersistence writes to.
        ConversationSearch(source=SnapshotHistorySource(store), scope="conversation"),
        # One-line-per-turn index in context. source reads the SAME store (so
        # read_turn resolves the exact history StepPersistence writes); the
        # append-only index lives in its own JSONL store.
        TurnSummaryIndex(
            source=SnapshotHistorySource(store),
            index_store=TurnIndexStore(_INDEX_DIR),
        ),
    ]
