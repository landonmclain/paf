"""Capability catalog — the master registry of capability types an agent can enable.

A profile is a *named allowlist* over this catalog (see ``profiles.py``). M1 ships
all harness capabilities we actually attach to an agent; nothing is scaffolded
ahead of need (YAGNI). Core pydantic-ai capabilities (``WebSearch``, ``Thinking``,
etc.) do NOT need a catalog entry — they are native to ``Agent.from_spec`` and the
spec loader resolves them without ``custom_capability_types``. This catalog exists
only for capabilities the loader cannot resolve by default (harness capabilities
like ``StepPersistence``), keyed by the CamelCase name used in the spec YAML.

Compaction is built in Python (see ``compaction.py``): ``TieredCompaction``'s nested
``tiers`` must be real strategy objects, and no harness compaction class overrides
``from_spec``, so a YAML ``tiers:`` dict list would crash at run time. The tier types
are still cataloged so a profile can reference them and the schema stays complete.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai_harness.compaction import (
    ClampOversizedMessages,
    ClearToolResults,
    SlidingWindowCompaction,
    SummarizingCompaction,
    TieredCompaction,
)
from pydantic_ai_harness.conversation_search import ConversationSearch
from pydantic_ai_harness.step_persistence import StepPersistence

from paf.capabilities.turn_summary import TurnSummaryIndex

# name -> capability type. The name is the serialization form used in a spec's
# `capabilities:` entry and the key a profile allowlists against.
CAPABILITY_TYPES: dict[str, type[Any]] = {
    "StepPersistence": StepPersistence,
    # Compaction types. TieredCompaction is assembled in Python (compaction.py);
    # the tier types are cataloged so the profile allowlist + schema stay complete.
    "TieredCompaction": TieredCompaction,
    "ClampOversizedMessages": ClampOversizedMessages,
    "ClearToolResults": ClearToolResults,
    "SlidingWindowCompaction": SlidingWindowCompaction,
    "SummarizingCompaction": SummarizingCompaction,
    # Lossless recall over the persistence store. Built in Python (build.py) because
    # its `source` needs a real SnapshotHistorySource object; cataloged so the
    # profile allowlist + schema stay complete.
    "ConversationSearch": ConversationSearch,
    # One-line-per-turn index in context + gated `read_turn`. Built in Python (build.py)
    # to share the single persistence store; has `from_spec`, so it can ALSO resolve
    # declaratively from a spec (`- TurnSummaryIndex: {index_dir: ...}`).
    "TurnSummaryIndex": TurnSummaryIndex,
}
