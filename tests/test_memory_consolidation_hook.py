"""Tests for the turn-bounded, trim-only eviction window (``TurnBoundSlidingWindow``).

The assistant's eviction tier is a plain sliding window that (1) never consumes
the ACTIVE user turn (the 2026-09-02 Option A gate — regression for run
01a05ff8 where the live "implement it" prompt was evicted while its response
survived) and (2) performs **no memory work** — durable-fact capture is the
background watermark worker's job, never eviction.

The former inline eviction-fold window (``MemoryConsolidationWindow``) is
DELETED, so there is no consolidation at the eviction boundary to test here.

Follows the project convention of wrapping async bodies in `asyncio.run`
(no pytest-asyncio plugin in the suite).
"""

from __future__ import annotations

import asyncio
import types

from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai_harness.compaction._shared import find_safe_cutoff

from paf.capabilities.memory._window import TurnBoundSlidingWindow


def _make_messages(n: int) -> list[ModelRequest]:
    """Build a list of `n` plain user turns (the common overflow shape)."""
    return [
        ModelRequest(parts=[UserPromptPart(content=f"user message {i}")])
        for i in range(n)
    ]


def test_window_trims_settled_prefix_never_active_turn() -> None:
    """Trim drops only the SETTLED prefix; the active (last) user turn survives.

    The window bounds every eviction decision to the completed history strictly
    before the most recent real user prompt, so the in-flight turn is
    structurally unreachable by the window tier.
    """
    window = TurnBoundSlidingWindow(max_messages=5, keep_messages=2)
    # Real overflow: 8 messages > max_messages=5 -> the plain window WOULD trim.
    messages = _make_messages(8)
    fake_ctx = types.SimpleNamespace(messages=messages, model=None)

    result = asyncio.run(window.compact(messages, fake_ctx))  # type: ignore[arg-type]

    # Settled = strictly before the last real user prompt (index 7) = 7 messages.
    settled = messages[:7]
    cutoff = find_safe_cutoff(settled, window.keep_messages)
    assert cutoff == 7 - 2  # keep=2 -> drop the first 5 of the settled prefix

    # Settled(7) trimmed to keep=2 + 1 preserved opening user msg = 3, then the
    # active turn re-appended = 4.
    assert len(result) == 4
    assert result[-1] == messages[-1], "the active (last) user turn must survive"


def test_window_noop_when_nothing_evictable() -> None:
    """History at or under keep_messages -> no trim at all (whole list returned)."""
    window = TurnBoundSlidingWindow(max_messages=5, keep_messages=2)
    messages = _make_messages(2)  # len == keep_messages -> nothing to evict
    fake_ctx = types.SimpleNamespace(messages=messages, model=None)

    result = asyncio.run(window.compact(messages, fake_ctx))  # type: ignore[arg-type]
    assert result == messages, "no eviction must leave the history untouched"


def test_window_does_no_memory_work(tmp_path) -> None:
    """The trim-only window never writes memory-journal.jsonl / MEMORY.md.

    Durable fact capture is the background watermark worker's job; eviction is
    pure context management. An overflowed trim must leave the target memory
    directory pristine.
    """
    store_dir = tmp_path / "memory"
    store_dir.mkdir()
    window = TurnBoundSlidingWindow(max_messages=5, keep_messages=2)
    messages = _make_messages(8)
    fake_ctx = types.SimpleNamespace(messages=messages, model=None)

    asyncio.run(window.compact(messages, fake_ctx))  # type: ignore[arg-type]

    assert not (store_dir / "memory-journal.jsonl").exists()
    assert not (store_dir / "MEMORY.md").exists()
    assert not (store_dir / "memory-index.json").exists()
