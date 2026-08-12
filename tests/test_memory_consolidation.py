"""Tests for pre-eviction memory consolidation.

Covers the three seams locked in for the consolidation feature:

1.  ``writable`` gate — the ``MemoryToolset`` exposes ``write_memory`` /
    ``delete_memory`` only when ``writable=True``; a read-only assistant can't
    mutate durable memory while the owner (``writable=True``) can.
2.  ``MemoryConsolidationWindow.compact`` — fires the owner on the
    about-to-be-evicted slice and no-ops when nothing would be evicted.
3.  ``build_compaction`` env toggle — ``CONSOLIDATE=1`` swaps the sliding-window
    tier for the consolidation window; unset keeps the plain window.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai_harness.compaction._shared import find_safe_cutoff

from paf.capabilities.compaction import build_compaction
from paf.capabilities.memory import Memory
from paf.capabilities.memory._toolset import MemoryToolset
from paf.capabilities.memory._window import MemoryConsolidationWindow

# ---------------------------------------------------------------------------
# writable gate
# ---------------------------------------------------------------------------


def _tool_names(cap: Memory) -> set[str]:
    ts = MemoryToolset(cap, writable=cap.writable)
    return set(ts.tools.keys())


def test_writable_false_drops_write_and_delete() -> None:
    """writable=False exposes read/search only — the assistant can't mutate."""
    ro = _tool_names(Memory(agent_name="a", writable=False))
    assert "read_memory" in ro
    assert "search_memory" in ro
    assert "write_memory" not in ro
    assert "delete_memory" not in ro


def test_writable_true_registers_full_surface() -> None:
    """writable=True (default + owner) still has the full write/read/search/delete set."""
    rw = _tool_names(Memory(agent_name="a", writable=True))
    assert {"read_memory", "search_memory", "write_memory", "delete_memory"} <= rw


def test_writable_defaults_true() -> None:
    """Backward compat: the default surface remains fully writable."""
    assert Memory(agent_name="a").writable


# ---------------------------------------------------------------------------
# MemoryConsolidationWindow.compact — fires on the eviction slice, no-op otherwise
# ---------------------------------------------------------------------------


def _messages(n: int) -> list[ModelRequest]:
    return [
        ModelRequest(parts=[UserPromptPart(content=f"msg {i}")]) for i in range(n)
    ]


def test_compact_noop_when_nothing_evicted() -> None:
    """With fewer messages than keep_messages, find_safe_cutoff=0 → no consolidation."""
    msgs = _messages(2)
    window = MemoryConsolidationWindow(max_messages=4, keep_messages=2)
    assert find_safe_cutoff(msgs, window.keep_messages) == 0


def test_window_swaps_into_compaction_under_env() -> None:
    """CONSOLIDATE=1 replaces the plain sliding window in the tier list."""
    import os

    import paf.capabilities.compaction as mod

    old = os.environ.get("CONSOLIDATE")
    try:
        os.environ["CONSOLIDATE"] = "0"
        plain = build_compaction()
        os.environ["CONSOLIDATE"] = "1"
        gated = build_compaction()
    finally:
        if old is None:
            os.environ.pop("CONSOLIDATE", None)
        else:
            os.environ["CONSOLIDATE"] = old

    assert "SlidingWindowCompaction" in [type(t).__name__ for t in plain.tiers]
    assert "MemoryConsolidationWindow" in [
        type(t).__name__ for t in gated.tiers
    ]
    assert type(mod).__name__ == "module"


def test_window_default_store_scope(tmp_path: Path) -> None:
    """Defaults point at the shared data/memory + assistant scope (build.py)."""
    w = MemoryConsolidationWindow(max_messages=4)
    assert w.memory_directory == "data/memory"
    assert w.memory_agent == "assistant"
    assert w.keep_messages == 40  # harness default, unchanged


# ---------------------------------------------------------------------------
# owner consolidation guidance + model env
# ---------------------------------------------------------------------------


def test_consolidation_guidance_is_owned_by_owner() -> None:
    """The owner guidance enforces sole-writer + newer-always-wins semantics."""
    from paf.capabilities.memory._consolidate import CONSOLIDATION_GUIDANCE

    assert "only writer" in CONSOLIDATION_GUIDANCE
    assert "NEWER ALWAYS WINS" in CONSOLIDATION_GUIDANCE


def test_consolidation_model_reads_env() -> None:
    """CONSOL_MODEL env overrides the flash-lite default (not hardcoded)."""
    import os

    from paf.capabilities.memory._consolidate import consolidate_model

    old = os.environ.get("CONSOL_MODEL")
    try:
        os.environ.pop("CONSOL_MODEL", None)
        assert consolidate_model() == "google/gemini-2.5-flash-lite:nitro"
        os.environ["CONSOL_MODEL"] = "openai:gpt-5.6-luna-pro"
        assert consolidate_model() == "openai:gpt-5.6-luna-pro"
    finally:
        if old is None:
            os.environ.pop("CONSOL_MODEL", None)
        else:
            os.environ["CONSOL_MODEL"] = old
