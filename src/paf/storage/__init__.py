"""Storage — persistence over pydantic-ai-harness's ``StepPersistence``.

M1 embraces the harness for durable, cross-process message history
(``SqliteStepStore``) rather than a custom protocol. ``Persistence`` is the
thin wrapper the rest of the framework uses: it builds the agent from its spec
(registering the harness capability), anchors the DB path to ``PROJECT_ROOT``,
and exposes ``resume`` / ``append`` for the CLI loop.

Search is a separate layer and deliberately lives elsewhere — storage persists;
it does not answer queries.
"""

from paf.storage.persistence import Persistence, build_persistence

__all__ = ["Persistence", "build_persistence"]
