"""Persistence + recall + turn-index assembly — one shared store for write, search, and read_turn.

Like :mod:`paf.capabilities.compaction`, this owns capabilities that can't cleanly
resolve from YAML because they need real Python objects as arguments (see
``build.py`` for the full rationale on the ``capabilities=`` seam). For M1 that is
`ConversationSearch` and `TurnIndexCapability`: their ``source: HistorySource`` needs
a real ``SnapshotHistorySource(store)`` object, and no harness recall class overrides
``from_spec`` (and `TurnIndexCapability`, though it DOES override `from_spec`, is wired
here to share the live store) — so they are assembled here, in Python.

`StepPersistence` *could* resolve from YAML (it overrides ``from_spec``), but it is
built here too so it shares ONE ``FileStepStore`` with `ConversationSearch` and
`TurnIndexCapability`. Sharing the identical store instance guarantees `ConversationSearch`
searches — and `read_turn` resolves — exactly the history `StepPersistence` writes:
no divergence between the write path and the recall-corpus read path.

This module owns that assembly. ``build_persistence()`` builds the single shared store
and returns the persistence + recall + turn-index capabilities wired over it;
``build_runtime_capabilities(profile)`` (in ``build.py``) selects the subset the profile
allowlists and hands it to the agent via ``Agent.from_spec(capabilities=)``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from pydantic_ai_harness.conversation_search import (
    ConversationSearch,
    SnapshotHistorySource,
)
from pydantic_ai_harness.step_persistence import FileStepStore, StepPersistence

from paf.capabilities.turn_index import TurnIndexCapability, TurnIndexStore
from paf.config import PROJECT_ROOT, settings

# Canonical store root for M1 persistence + recall. A directory of flat files
# (JSONL events, JSON snapshots per run) — greppable/inspectable, no DB to
# administer. One instance, handed to both StepPersistence (writer) and
# ConversationSearch (reader), so the search corpus is exactly the persisted
# history. Anchored to the repo root (not CWD).
_STORE_DIR = PROJECT_ROOT / Path(settings.persistence_dir)

# The append-only one-line turn index lives next to the step store, not inside
# it (they are two different write surfaces). Anchored to the repo root; sibling
# of the persistence store under data/ (default data/.turn-index).
_INDEX_DIR = PROJECT_ROOT / Path(settings.turn_index_dir)


def _build_store() -> FileStepStore:
    """Build the single shared persistence store."""
    _STORE_DIR.mkdir(parents=True, exist_ok=True)
    return FileStepStore(directory=_STORE_DIR)


def _build_step_persistence(store: FileStepStore) -> StepPersistence:
    """The write side: appends every step of a run to the shared store."""
    return StepPersistence(store=store, agent_name="assistant")


def _build_conversation_search(store: FileStepStore) -> ConversationSearch:
    """Lossless recall over the SAME store StepPersistence writes to."""
    return ConversationSearch(
        source=SnapshotHistorySource(store), scope="conversation"
    )


def _build_turn_index(store: FileStepStore) -> TurnIndexCapability:
    """One-line-per-turn index in context; appends via the agent-foundation
    `StorageService` seam (its `TurnIndexStore` wraps a `FlatFileStorage` at
    `_INDEX_DIR`), but reads the shared store so `read_turn` resolves what
    `StepPersistence` writes."""
    return TurnIndexCapability(
        source=SnapshotHistorySource(store),
        index_store=TurnIndexStore(_INDEX_DIR),
        store=store,
    )


# Name -> builder, keyed by the SAME names a profile allowlists (registry.py).
# The profile is the gatekeeper: `build_runtime_capabilities(profile)` selects
# builders by profile.capability_names, so *only* what the profile allows is
# constructed — one shared store handed to whichever are selected. There is no
# separate hardcoded list; this IS the resolution of the allowlist names.
PERSISTENCE_BUILDERS: dict[str, Callable[[FileStepStore], Any]] = {
    "StepPersistence": _build_step_persistence,
    "ConversationSearch": _build_conversation_search,
    "TurnIndex": _build_turn_index,
}


def build_persistence(names: Iterable[str] | None = None) -> list:
    """Assemble the persistence + recall capabilities over one shared store.

    Defaults to every builder in :data:`PERSISTENCE_BUILDERS`, i.e. the full
    persistence + recall + turn-index set, all wired to the SAME ``FileStepStore``.
    Pass an explicit ``names`` iterable to build only those (used by tests and by
    ``build_runtime_capabilities(profile)`` to honor a profile's allowlist).
    """
    store = _build_store()
    return [
        builder(store)
        for name, builder in PERSISTENCE_BUILDERS.items()
        if names is None or name in names
    ]
