"""Persistence wrapper — thin, doc-backed seam over pydantic-ai-harness's ``StepPersistence``.

``StepPersistence`` (with the ``SqliteStepStore`` backend) is the durable,
cross-process message-history store for M1. This module wraps it so the rest of
the framework never touches harness internals directly: it renders the agent
from its spec (registering ``StepPersistence`` as a custom capability), supplies
the ``PROJECT_ROOT``-anchored DB path, and exposes the two operations the CLI
loop needs — ``resume`` (load prior history for a conversation) and ``append``
(tack this run's new messages onto it).

Why not hand-roll a store? pydantic-ai-harness already ships ``StepPersistence``
and ``SqliteStepStore`` (proven cross-process in ``ui/cli_demo.py``). Reusing a
capability that already exists — documented, tested, spec-declarable — is M1's
"embrace the harness" decision. This wrapper is the seam we own; the harness is
the implementation. See `pydantic-ai-harness` skill →
step_persistence/README.md.

Search is deliberately a separate layer, not built here (see the framework
notes). Storage only persists; it does not answer "find me X".
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai_harness.step_persistence import (
    SqliteStepStore,
    StepPersistence,
)

from paf.config import PROJECT_ROOT

# Default DB location, anchored to the repo root so it's stable regardless of
# where the CLI is launched from. The spec may override via a `database` field,
# in which case relative paths resolve against PROJECT_ROOT too.
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "paf.db"


def _resolve_database(spec_database: str | None) -> str:
    """Anchor a relative spec ``database`` to ``PROJECT_ROOT``.

    ``StepPersistence.from_spec`` treats a relative path as CWD-relative, which
    would scatter the DB across launch directories. We fix the base so
    persistence lands in one canonical place.
    """
    if not spec_database:
        return str(DEFAULT_DB_PATH)
    path = Path(spec_database)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


class Persistence:
    """Own a ``SqliteStepStore`` and the resume/append operations over it.

    The agent is built from its spec with ``StepPersistence`` registered as a
    custom capability, so the same YAML that describes the agent also
    configures its storage (``backend`` / ``database`` / ``max_snapshots_per_run``).
    """

    def __init__(self, spec_path: Path | str, *, database: str | None = None) -> None:
        self.spec_path = Path(spec_path)

        # Build the agent from the spec, telling the loader to register the
        # harness capability (not in pydantic-ai's default registry).
        self.agent = Agent.from_file(
            self.spec_path,
            custom_capability_types=[StepPersistence],
        )
        self.step_persistence = self._find_step_persistence()
        self.store: SqliteStepStore = self._resolve_store(database)

    def _find_step_persistence(self) -> StepPersistence[Any]:
        """Locate the spec-built ``StepPersistence`` inside the agent's capability tree.

        The spec loader composes capabilities into ``agent._root_capability``
        (a ``CombinedCapability``); we walk it to hand the caller the same
        instance the agent will use.
        """
        root = getattr(self.agent, "_root_capability", None)
        found: StepPersistence[Any] | None = None

        def walk(obj: Any) -> None:
            nonlocal found
            if found is not None or obj is None:
                return
            if isinstance(obj, StepPersistence):
                found = obj
                return
            if isinstance(obj, (list, tuple)):
                for item in obj:
                    walk(item)
            elif isinstance(obj, dict):
                for item in obj.values():
                    walk(item)
            elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                for field in dataclasses.fields(obj):
                    walk(getattr(obj, field.name, None))

        walk(root)
        if found is None:
            raise RuntimeError(
                "StepPersistence capability not found after loading agent spec "
                f"{self.spec_path}. Ensure the spec lists it under `capabilities`."
            )
        return found

    def _resolve_store(self, database: str | None) -> SqliteStepStore:
        """Use a caller-supplied database, else the capability's store, else default."""
        if database is not None:
            return SqliteStepStore(database=_resolve_database(database))
        store = getattr(self.step_persistence, "store", None)
        if isinstance(store, SqliteStepStore):
            return store
        # Spec backend was memory/file or store unavailable — default to sqlite.
        return SqliteStepStore(database=str(DEFAULT_DB_PATH))

    async def resume(self, conversation_id: str) -> list[ModelMessage]:
        """Return the prior message history for ``conversation_id`` (empty if new).

        Finds the newest complete snapshot across the conversation's runs and
        loads its messages as ``message_history`` for the next ``Agent.run``.
        """
        runs = await self.store.list_runs(conversation_id=conversation_id)
        if not runs:
            return []
        run_id = runs[-1].run_id  # list_runs is ascending; newest last
        snapshot = await self.store.latest_snapshot(run_id=run_id)
        if snapshot is None:
            return []
        return snapshot.messages

    async def append(
        self, history: list[ModelMessage], new_messages: list[ModelMessage]
    ) -> list[ModelMessage]:
        """Return the full history for the next run: prior history + new messages."""
        return [*history, *new_messages]


def build_persistence(
    spec_path: Path | str, *, database: str | None = None
) -> Persistence:
    """Convenience factory so callers don't import ``Persistence`` directly."""
    return Persistence(spec_path, database=database)
