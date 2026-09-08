"""Memory directory helpers (moved out of the shared eviction window).

Historical note (2026-09-07): this module used to also hold the assistant's
eviction tier ``TurnBoundSlidingWindow``, which has moved to the shared flat
module :mod:`paf.capabilities._window` (it is compaction infrastructure, not
memory functionality — drives no memory work, is built iff the profile
allowlists ``"Compaction"``). What remains here is memory-owned-only: the
canonical per-agent durable-memory directory resolution used by the owner,
organizer, reviewer, and the runtime's queue/watermark defaults.
"""

from __future__ import annotations

from pathlib import Path

from paf.config import PROJECT_ROOT, settings

# Same fixed store the memory-owner uses (build.py `_build_memory`). The owner
# writes these plain `.md` bodies + CAS sidecar so the subject reads the exact
# same durable facts it would have written itself. Anchored to PROJECT_ROOT
# through the shared Settings object (MEMORY_DIR / .env), so both always target
# one canonical directory no matter where the process is launched from.
DEFAULT_MEMORY_DIR = str(PROJECT_ROOT / Path(settings.memory_dir))


def memory_dir_for(agent_name: str) -> str:
    """The per-agent memory directory: ``<MEMORY_DIR>/<agent_name>/``.

    Agent-isolated durable memory (one journal/MEMORY.md set per agent, keyed by
    the spec base name — ``demo_assistant`` → ``data/memory/demo_assistant/``).
    The directory is created on first use; no corpus is copied in. The shared
    Session watermark/queue default and the injected ``JournalViewStore`` (in
    ``build.py._build_memory``) both derive from this so they can never point at
    different scopes.
    """
    path = PROJECT_ROOT / Path(settings.memory_dir) / agent_name
    path.mkdir(parents=True, exist_ok=True)
    return str(path)
