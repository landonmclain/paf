"""Tests for pre-eviction memory consolidation.

Covers the three seams locked in for the consolidation feature:

1.  ``writable`` gate — the ``MemoryToolset`` exposes ``write_memory`` /
    ``delete_memory`` only when ``writable=True``; a read-only assistant can't
    mutate durable memory while the owner (``writable=True``) can.
2.  ``MemoryConsolidationWindow.compact`` — fires the owner on the
    about-to-be-evicted slice and no-ops when nothing would be evicted.
3.  ``build_compaction`` config toggle — ``consolidate`` on swaps the sliding-window
    tier for the consolidation window; off keeps the plain window.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai_harness.compaction._shared import find_safe_cutoff

from paf.capabilities.compaction import build_compaction
from paf.capabilities.memory import Memory
from paf.capabilities.memory._consolidate import (
    CONSOLIDATION_GUIDANCE,
    consolidate_model,
)
from paf.capabilities.memory._format import MEMORY_FORMAT, extract_sections
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


def test_window_swaps_into_compaction_under_config(monkeypatch) -> None:
    """consolidate config on swaps the plain sliding window for the consolidation window."""
    from pydantic_ai_harness.compaction import SlidingWindowCompaction

    from paf.config import settings

    monkeypatch.setattr(settings, "consolidate", False)
    plain = build_compaction()
    monkeypatch.setattr(settings, "consolidate", True)
    gated = build_compaction()

    assert isinstance(plain.tiers[-1], SlidingWindowCompaction)
    assert isinstance(gated.tiers[-1], MemoryConsolidationWindow)


def test_window_default_store_scope(monkeypatch) -> None:
    """Defaults point at the anchored shared data/memory + assistant scope (build.py)."""
    from pydantic_ai_harness.compaction import SlidingWindowCompaction

    w = MemoryConsolidationWindow(max_messages=4)
    # Anchored to PROJECT_ROOT through Settings.memory_dir, matching build.py.
    from paf.config import PROJECT_ROOT

    assert w.memory_directory == str(PROJECT_ROOT / Path("data/memory"))
    assert w.memory_agent == "assistant"
    # Parent's defaults are inherited unchanged.
    assert isinstance(w, SlidingWindowCompaction)
    assert w.keep_messages == 40  # harness default, unchanged


# ---------------------------------------------------------------------------
# owner consolidation guidance + model env
# ---------------------------------------------------------------------------


def test_consolidation_guidance_is_owned_by_owner() -> None:
    """The owner guidance enforces sole-writer + newer-always-wins semantics."""
    assert "only writer" in CONSOLIDATION_GUIDANCE
    assert "NEWER ALWAYS WINS" in CONSOLIDATION_GUIDANCE
    assert "{format_spec}" in CONSOLIDATION_GUIDANCE  # format embedded at build time


def test_memory_format_pinned_sections_and_order() -> None:
    """The canonical MEMORY.md format declares exactly four sections, in order.

    Both the subject's default guidance and the owner's consolidation manual
    must embed the exact same MEMORY_FORMAT so the writer and reader agree on
    the notebook shape by construction — if either stops carrying the format,
    this test fails.
    """
    from paf.capabilities.memory._capability import _DEFAULT_GUIDANCE

    # The raw format pins the four sections, in the canonical order.
    assert extract_sections(MEMORY_FORMAT) == [
        "Facts",
        "Preferences",
        "Pending",
        "Notes",
    ]
    # No stray sections beyond the four.
    assert len(extract_sections(MEMORY_FORMAT)) == 4
    # Write bias is explicit ("write too much") in the format.
    assert "WHEN IN DOUBT, RECORD IT" in MEMORY_FORMAT
    assert "## Pending" in MEMORY_FORMAT and "- [ ]" in MEMORY_FORMAT
    # Both guidance strings carry the exact section block (single source of truth).
    from paf.capabilities.memory._consolidate import _build_owner_memory

    assert extract_sections(
        _DEFAULT_GUIDANCE.format(format_spec=MEMORY_FORMAT)
    ) == extract_sections(MEMORY_FORMAT)
    built_owner = _build_owner_memory(None, agent_name="assistant", namespace="")
    assert built_owner.guidance is not None
    assert extract_sections(built_owner.guidance) == extract_sections(MEMORY_FORMAT)


def test_consolidation_model_reads_config(monkeypatch) -> None:
    """Settings.consol_model (CONSOL_MODEL / .env) overrides the flash-lite default."""
    from paf.config import settings

    monkeypatch.setattr(settings, "consol_model", "")
    assert consolidate_model() == "google/gemini-2.5-flash-lite:nitro"
    monkeypatch.setattr(settings, "consol_model", "openai:gpt-5.6-luna-pro")
    assert consolidate_model() == "openai:gpt-5.6-luna-pro"


# ---------------------------------------------------------------------------
# owner construction + the seed-time full-conversation variant
# ---------------------------------------------------------------------------


def test_build_owner_is_single_writer_and_read_via_tools() -> None:
    """The owner's Memory capability is writable=True, inject_memory=False, guided.

    The hard-won ownership invariants (sole writer, no injected context, fixed
    operating manual) live in :func:`_build_owner_memory` and apply to BOTH
    consolidation paths — the eviction slice and the seed-time full-conversation
    variant. Tested on the capability directly (no Agent construction, so no
    model provider / API key needed).
    """
    from paf.capabilities.memory import Memory
    from paf.capabilities.memory._consolidate import _build_owner_memory

    cap = _build_owner_memory(None, agent_name="assistant", namespace="")
    assert isinstance(cap, Memory)
    assert cap.writable is True
    assert cap.inject_memory is False
    assert cap.guidance == CONSOLIDATION_GUIDANCE.format(format_spec=MEMORY_FORMAT)


def test_resident_consolidate_text_none_when_empty() -> None:
    """No messages → resident_consolidate_text returns None (owner never fires)."""
    import asyncio

    from paf.capabilities.memory._consolidate import resident_consolidate_text

    out = asyncio.run(resident_consolidate_text([]))
    assert out is None


def test_resident_consolidate_text_renders_every_segment() -> None:
    """Every message segment is folded into the owner prompt (no eviction cut)."""
    import asyncio

    from pydantic_ai.messages import ModelRequest, UserPromptPart

    from paf.capabilities.memory._consolidate import resident_consolidate_text

    msgs = [
        ModelRequest(parts=[UserPromptPart(content=f"turn {i}")]) for i in range(4)
    ]
    out = asyncio.run(resident_consolidate_text(msgs))
    assert out is not None
    for i in range(4):
        assert f"turn {i}" in out
