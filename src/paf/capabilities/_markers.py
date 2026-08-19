"""Injected-prompt markers for capabilities that strip transient artifacts.

Capabilities that inject into the live prompt as a ``UserPromptPart`` tag the
injected ``TextContent`` with a ``metadata`` marker. That marker is invisible to
the model but lets any downstream reader (read_turn render, a search corpus,
StepPersistence) *recognize and strip* the injected artifact so it never leaks
into durable turns or recalled history.

Lives in the shared `capabilities` package (not under `turn_index`) because BOTH
the turn index (injection) and the `search` read primitive (rendering/stripping a
recalled turn) depend on it -- `search` must not depend on `turn_index`.

Mirrors the pattern `pydantic_ai_harness.memory` established (`_MEMORY_PART_METADATA`
on its injected `TextContent`). When we render a persisted User message back to
the model, we drop any part carrying these markers -- the injected block is a
transient per-request artifact, not part of the durable turn.
"""

from __future__ import annotations

# Our injected turn-index block marker. Matches `_capability.py:_MARKER`.
TURN_INDEX_MARKER = 'turn-index.v1'

# The harness Memory capability's scope-qualified marker base. We strip this
# family too so a recalled turn never carries a stale memory block pasted into
# a User message that StepPersistence snapshotted.
MEMORY_MARKER_BASE = 'pydantic-ai-harness.memory.v1'


def injected_artifact_markers() -> tuple[str, ...]:
    """Return the set of markers we strip from rendered User messages.

    Each is matched as a prefix against a ``TextContent.metadata`` value:
    our marker is exact, the harness memory marker is scope-qualified
    (``pydantic-ai-harness.memory.v1:<scope_hash>``).
    """
    # Exact match for ours, prefix match for the scope-qualified memory marker.
    return (TURN_INDEX_MARKER, MEMORY_MARKER_BASE)
