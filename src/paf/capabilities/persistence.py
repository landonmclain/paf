"""Persistence + recall assembly — one shared store for write and search.

Like :mod:`paf.capabilities.compaction`, this owns capabilities that can't cleanly
resolve from YAML because they need real Python objects as arguments (see
``build.py`` for the full rationale on the ``capabilities=`` seam). For M1 that is
`ConversationSearch` and `TurnIndexCapability`: their ``source: HistorySource`` needs
a real ``SnapshotHistorySource(store)`` object, and no harness recall class overrides
``from_spec`` (and `TurnIndexCapability`, though it DOES override `from_spec`, is wired
here to share the live store) — so they are assembled here, in Python.

``TurnIndexCapability`` is NOT built here: it owns its builder in
``paf.capabilities.turn_index`` (``build_turn_index``) so its summarizer/content
config flows from the capability's own spec. This module is scoped to pure
store construction for StepPersistence (write) and ConversationSearch (recall).

`StepPersistence` *could* resolve from YAML (it overrides ``from_spec``), but it is
built here too so it shares ONE ``FileStepStore`` with `ConversationSearch`.
Sharing the identical store instance guarantees `ConversationSearch`
searches — exactly the history `StepPersistence` writes:
no divergence between the write path and the recall-corpus read path.

This module owns that assembly. ``build_persistence()`` builds the single shared store
and returns the persistence + recall capabilities wired over it;
``build_runtime_capabilities(profile)`` (in ``build.py``) selects the subset the profile
allowlists (merging in the TurnIndex builder) and hands it to the agent via
``Agent.from_spec(capabilities=)``.
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

from paf.config import PROJECT_ROOT, settings

# Canonical store root for M1 persistence + recall. A directory of flat files
# (JSONL events, JSON snapshots per run) — greppable/inspectable, no DB to
# administer. One instance, handed to both StepPersistence (writer) and
# ConversationSearch (reader), so the search corpus is exactly the persisted
# history. Anchored to the repo root (not CWD).
_STORE_DIR = PROJECT_ROOT / Path(settings.persistence_dir)


def _build_store() -> FileStepStore:
    """Build the single shared persistence store."""
    _STORE_DIR.mkdir(parents=True, exist_ok=True)
    return FileStepStore(directory=_STORE_DIR)


def _build_step_persistence(store: FileStepStore) -> StepPersistence:
    """The write side: appends every step of a run to the shared store.

    IMPORTANT: no ``agent_name``. StepPersistence derives its run_id as
    ``{agent_name}-{uuid}`` when agent_name is set (the harness's ``_derive_run_id``),
    which would make the STORE key runs by ``assistant-<hex>`` while the TurnIndex
    capability writes pydantic-ai's ``ctx.run_id`` (``019fe…``) -- a run-id namespace
    mismatch that silently broke ``read_turn`` (refs never resolve). With agent_name
    unset, StepPersistence falls through to ``ctx.run_id``, so store and index agree
    on the same id and every ref round-trips.
    """
    return StepPersistence(store=store)


def _build_conversation_search(store: FileStepStore) -> ConversationSearch:
    """Lossless recall over the SAME store StepPersistence writes to."""
    return ConversationSearch(
        source=SnapshotHistorySource(store), scope="conversation"
    )


# Name -> builder, keyed by the SAME names a profile allowlists (registry.py).
# The profile is the gatekeeper: `build_runtime_capabilities(profile)` selects
# builders by profile.capability_names, so *only* what the profile allows is
# constructed — one shared store handed to whichever are selected. There is no
# separate hardcoded list; this IS the resolution of the allowlist names.
#
# NOTE: `TurnIndex` is deliberately NOT here — it builds its own capability in
# `paf.capabilities.turn_index` (`build_turn_index`) so its content config
# (summarizer) lives with the capability, not in persistence. build.py merges
# that builder with this dict so the shared store still flows to it.
PERSISTENCE_BUILDERS: dict[str, Callable[[FileStepStore], Any]] = {
    "StepPersistence": _build_step_persistence,
    "ConversationSearch": _build_conversation_search,
}


def build_persistence(names: Iterable[str] | None = None) -> list:
    """Assemble the persistence + recall capabilities over one shared store.

    Defaults to every builder in :data:`PERSISTENCE_BUILDERS`, i.e. the full
    persistence + recall set, all wired to the SAME ``FileStepStore``.
    Pass an explicit ``names`` iterable to build only those (used by tests and by
    ``build_runtime_capabilities(profile)`` to honor a profile's allowlist).
    """
    store = _build_store()
    return [
        builder(store)
        for name, builder in PERSISTENCE_BUILDERS.items()
        if names is None or name in names
    ]
