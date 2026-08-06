"""Persistence + recall assembly — one shared store for write and search.

Like :mod:`paf.capabilities.compaction`, this owns capabilities that can't cleanly
resolve from YAML because they need real Python objects as arguments (see
``build.py`` for the full rationale on the ``capabilities=`` seam). For M1 that is
`ConversationSearch`: its ``source: HistorySource`` needs a real
``SnapshotHistorySource(store)`` object, and no harness recall class overrides
``from_spec`` — so it must be assembled here, in Python.

`StepPersistence` *could* resolve from YAML (it overrides ``from_spec``), but it is
built here too so it shares ONE ``SqliteStepStore`` with `ConversationSearch`. Sharing
the identical store instance guarantees `ConversationSearch` searches exactly the
history `StepPersistence` writes — no divergence between the write path and the
recall-corpus read path.

This module owns that assembly. ``build_persistence()`` builds the single shared store
and returns the persistence + recall capabilities wired over it;
``build_runtime_capabilities()`` (in ``build.py``) merges them with ``build_compaction()``
into the agent.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_ai_harness.conversation_search import (
    ConversationSearch,
    SnapshotHistorySource,
)
from pydantic_ai_harness.step_persistence import SqliteStepStore, StepPersistence

from paf.config import PROJECT_ROOT, settings

# Canonical store for M1 persistence + recall. One instance, handed to both
# StepPersistence (writer) and ConversationSearch (reader), so the search corpus
# is exactly the persisted history. Anchored to the repo root (not CWD).
_DB_PATH = PROJECT_ROOT / Path(settings.db_path)


def _build_store() -> SqliteStepStore:
    """Build the single shared persistence store."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return SqliteStepStore(database=_DB_PATH)


def build_persistence() -> list:
    """Assemble the persistence + recall capabilities over one shared store.

    Returns `StepPersistence` (the writer) and `ConversationSearch` (the lossless
    recall reader) wired to the SAME ``SqliteStepStore``, so the model can retrieve
    anything compaction (sliding window or summary) dropped from the live context,
    plus anything from past runs — and the search corpus provably matches what gets
    persisted.
    """
    store = _build_store()
    return [
        StepPersistence(store=store, agent_name="assistant"),
        # Lossless recall over the SAME store StepPersistence writes to.
        ConversationSearch(source=SnapshotHistorySource(store), scope="conversation"),
    ]
