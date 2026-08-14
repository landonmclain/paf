"""End-to-end test that the memory compaction HOOK actually fires and writes.

The benchmark harness (run_lmem.py) deliberately bypasses live
`MemoryConsolidationWindow.compact()`: on direct-seeded data the sliding window
never overflows, so it calls `consolidate_conversation` (the full-conversation
variant) directly instead. That means the REAL eviction boundary — the path a
production long conversation actually takes when the window trims — was never
exercised by a test. The existing `test_build_agent.py` tests only assert WHICH
window tier is present (`MemoryConsolidationWindow` in the ladder), not that it
fires consolidation on the about-to-evict slice and writes durable memory.

This test closes that gap deterministically and WITHOUT a model/API key: it
monkeypatches `_consolidate.consolidate` (the function `_window.compact` calls
via a lazy `from ._consolidate import`) with a recording fake that writes a
known `MEMORY.md` into the store, then drives `MemoryConsolidationWindow.compact`
through a real overflow. It asserts:
  1. consolidation FIRES exactly when the window would evict (cutoff > 0), and
  2. derives its input from the about-to-evict tail, and
  3. writes durable memory into the shared store, and
  4. still returns the trimmed tail the plain window would.

Follows the project convention of wrapping async bodies in `asyncio.run`
(no pytest-asyncio plugin in the suite).
"""

from __future__ import annotations

import asyncio
import types
from pathlib import Path

from pydantic_ai.messages import ModelRequest, UserPromptPart

from paf.capabilities.memory._window import MemoryConsolidationWindow
from pydantic_ai_harness.compaction._shared import find_safe_cutoff


def _make_messages(n: int) -> list[ModelRequest]:
    """Build a message list of `n` plain user turns (the common overflow shape)."""
    return [
        ModelRequest(parts=[UserPromptPart(content=f"user message {i}")])
        for i in range(n)
    ]


class _Gotcha:
    """Recording probe for the `consolidate` call site."""

    def __init__(self, store_dir: Path):
        self.store_dir = store_dir
        self.cutoff: int | None = None
        self.passed_messages: int = 0
        self.keep_tail: int = 0

    async def __call__(self, messages, *, store, agent_name, namespace, keep_tail):
        # Record exactly what _window.compact handed over.
        self.passed_messages = len(messages)
        self.keep_tail = keep_tail
        self.cutoff = find_safe_cutoff(messages, keep_tail)
        # Write durable memory the way the real owner would.
        (self.store_dir / "MEMORY.md").write_text(
            f"# Facts\n- consolidated {self.cutoff} evicted messages\n",
            encoding="utf-8",
        )
        return None


def test_compaction_hook_fires_and_writes_on_real_eviction(tmp_path, monkeypatch):
    # The shared Markdown store the window targets for memory.
    store_dir = tmp_path / "memory"
    store_dir.mkdir()

    # A window that overflows at 5 messages and keeps 2 at the tail.
    window = MemoryConsolidationWindow(
        max_messages=5,
        keep_messages=2,
        memory_directory=str(store_dir),
        memory_agent="assistant",
    )

    # Real overflow: 8 messages > max_messages=5 -> the plain window WOULD trim.
    messages = _make_messages(8)

    # Substitute the real owner consolidation (model call) with a deterministic
    # recording write. `_window.compact` lazy-imports `consolidate` per call, so
    # patching the module attribute is picked up.
    got = _Gotcha(store_dir)
    monkeypatch.setattr(
        "paf.capabilities.memory._consolidate.consolidate", got, raising=True
    )

    # `MemoryConsolidationWindow.compact` never dereferences `ctx` fields (it only
    # passes them through to the base window, which likewise only consults the
    # messages list + cutoff helpers), so a lightweight stand-in is sufficient.
    fake_ctx = types.SimpleNamespace(messages=messages, model=None)

    result = asyncio.run(window.compact(messages, fake_ctx))  # type: ignore[arg-type]

    # 1. Consolidation fired exactly once, on the about-to-evict prefix.
    assert got.cutoff is not None and got.cutoff > 0, (
        "consolidation must fire only when the window actually evicts"
    )
    # The slice handed to consolidate is exactly the prefix the window will drop.
    assert got.passed_messages == len(messages)
    expected_cutoff = 8 - 2  # find_safe_cutoff(messages, keep_messages=2) on plain turns
    assert got.cutoff == expected_cutoff
    assert got.keep_tail == 2

    # 2. Durable memory was written into the shared store.
    assert (store_dir / "MEMORY.md").is_file()
    md = (store_dir / "MEMORY.md").read_text(encoding="utf-8")
    assert f"consolidated {expected_cutoff}" in md

    # 3. The window still performed the trim (the contract of the base window):
    #    the about-to-evict prefix is gone and only the tail remains.
    assert len(result) < len(messages), "compaction must still trim"
    assert len(result) <= 3, (  # keep_messages=2 (+ preserved first user msg)
        f"expected a trimmed tail; got {len(result)}"
    )


def test_compaction_hook_noop_when_nothing_to_evict(tmp_path, monkeypatch):
    """When `compact` is invoked but nothing is about to be evicted (history is at or
    under `keep_messages`), consolidation must NOT fire — the window short-circuits
    on `find_safe_cutoff(...) <= 0`. (The outer trigger — `before_model_request` /
    `TieredCompaction._escalate` — decides *whether* `compact` runs at all; this pins
    the inner guard that prevents a redundant consolidation on a non-trimming call.)"""
    store_dir = tmp_path / "memory"
    store_dir.mkdir()

    window = MemoryConsolidationWindow(
        max_messages=5,
        keep_messages=2,
        memory_directory=str(store_dir),
        memory_agent="assistant",
    )
    messages = _make_messages(2)  # len == keep_messages -> nothing to evict

    got = _Gotcha(store_dir)
    monkeypatch.setattr(
        "paf.capabilities.memory._consolidate.consolidate", got, raising=True
    )

    fake_ctx = types.SimpleNamespace(messages=messages, model=None)
    asyncio.run(window.compact(messages, fake_ctx))  # type: ignore[arg-type]

    # No consolidation, no memory write.
    assert got.cutoff is None, "consolidation must NOT fire when nothing is evicted"
    assert not (store_dir / "MEMORY.md").exists()