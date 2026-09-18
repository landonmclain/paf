"""Persistence + recall assembly — one shared store for write and search.

Like :mod:`paf.capabilities.compaction`, this owns capabilities that can't cleanly
resolve from YAML because they need real Python objects as arguments (see
``build.py`` for the full rationale on the ``capabilities=`` seam). For M1 that is
`ConversationSearch`: its ``source: HistorySource`` needs
a real ``SnapshotHistorySource(store)`` object, and no recall class overrides
``from_spec`` — so it is assembled here, in Python.

``ConversationSearch`` is the paf-owned search capability
(``paf.capabilities.search``), a drop-in for the harness's of the same name — so the
recall surface shares the marker-strip that keeps injected ``<memory>``
blocks out of the corpus.

(The original ``TurnIndex`` recall capability — one-line-per-turn index + ``read_turn``
resolver — was retired as a failed experiment and archived; see the codebase History
notes. It was built in ``paf.capabilities.turn_index`` and is no longer part of this
module.)

`StepPersistence` *could* resolve from YAML (it overrides ``from_spec``), but it is
built here too so it shares ONE ``FileStepStore`` with `ConversationSearch`.
Sharing the identical store instance guarantees `ConversationSearch`
searches — exactly the history `StepPersistence` writes:
no divergence between the write path and the recall-corpus read path.

This module owns that assembly. ``build_persistence()`` builds the single shared store
and returns the persistence + recall capabilities wired over it;
``build_runtime_capabilities(profile)`` (in ``build.py``) selects the subset the profile
allowlists and hands it to the agent via
``Agent.from_spec(capabilities=)``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic_ai_harness.step_persistence import FileStepStore, StepPersistence

from paf.capabilities._hashes import _content_hash
from paf.capabilities.search import ConversationSearch, SnapshotHistorySource
from paf.capabilities.search._render import is_summary_artifact
from paf.config import PROJECT_ROOT, settings

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage

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


class StableSnapshotHistorySource(SnapshotHistorySource):
    """``SnapshotHistorySource`` with content-stable within-run overlap removal.

    The harness's ``run_history`` reconciles the accumulated history's suffix
    against each snapshot's prefix using the harness ``message_hash``, which
    hashes raw serialized bytes INCLUDING pydantic's per-instantiation
    ``timestamp``. Every snapshot is re-instantiated on read, so byte-identical
    messages across snapshots hash differently, overlap computes 0, and the
    run's full cumulative history is appended once per snapshot — the
    within-run duplication that blows up a conversation's span (up to
    ``#snapshots`` copies of every turn). We reimplement the same union with
    :func:`_content_hash` (content-stable across re-instantiation), so a run's
    ``run_history`` returns each turn exactly once.
    """

    async def run_history(self, *, run_id: str) -> list[ModelMessage]:
        history: list[ModelMessage] = []
        history_keys: set[str] = set()
        for snapshot in await self._store.list_snapshots(run_id=run_id):
            messages = [
                message
                for message in snapshot.messages
                if not is_summary_artifact(message)
            ]
            uniq: list[ModelMessage] = []
            for message in messages:
                key = _content_hash(message)
                if key in history_keys:
                    continue  # already carried by an earlier (cumulative) snapshot
                history_keys.add(key)
                uniq.append(message)
            history.extend(uniq)
        return history


def build_history_source(store: FileStepStore) -> StableSnapshotHistorySource:
    """Build the recall history source with content-stable overlap dedup.

    Every consumer that reads persisted history to resolve an unseen span (the
    memory owner and the thread summarizer) should use this instead of a raw
    harness ``SnapshotHistorySource`` so neither re-collects duplicated turns
    from cumulative snapshots / runs.
    """
    return StableSnapshotHistorySource(store)


def _build_step_persistence(store: FileStepStore) -> StepPersistence:
    """The write side: appends every step of a run to the shared store.

    IMPORTANT: no ``agent_name``. StepPersistence derives its run_id as
    ``{agent_name}-{uuid}`` when agent_name is set (the harness's ``_derive_run_id``),
    which would make the STORE key runs by ``assistant-<hex>`` while the recall
    capability reads pydantic-ai's ``ctx.run_id`` (``019fe…``) -- a run-id namespace
    mismatch that silently broke ``read_turn`` (refs never resolve). With agent_name
    unset, StepPersistence falls through to ``ctx.run_id``, so store and recall agree
    on the same id and every ref round-trips.
    """
    return StepPersistence(store=store)


def _build_conversation_search(store: FileStepStore) -> ConversationSearch:
    """Lossless recall over the SAME store StepPersistence writes to."""
    return ConversationSearch(source=SnapshotHistorySource(store), scope="conversation")


# Name -> builder, keyed by the SAME names a profile allowlists (registry.py).
# The profile is the gatekeeper: `build_runtime_capabilities(profile)` selects
# builders by profile.capability_names, so *only* what the profile allows is
# constructed — one shared store handed to whichever are selected. There is no
# separate hardcoded list; this IS the resolution of the allowlist names.
#
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
