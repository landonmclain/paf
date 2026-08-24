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
from pydantic_ai_harness.compaction._shared import find_safe_cutoff

from paf.capabilities.memory._window import MemoryConsolidationWindow


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
        self.call_count: int = 0

    async def __call__(self, messages, *, store, agent_name, namespace, keep_tail, journal=None):
        # Record exactly what _window.compact handed over.
        self.call_count += 1
        self.passed_messages = len(messages)
        self.keep_tail = keep_tail
        self.cutoff = find_safe_cutoff(messages, keep_tail)
        # Durable memory is written through the M2 journal (append_memory), which
        # the window then renders to root MEMORY.md — NOT the legacy store.
        assert journal is not None, "window must pass the journal through to consolidate"
        journal.append(
            type="fact",
            fact=f"consolidated {self.cutoff} evicted messages",
            born_on="2023-08-20",
            source_ref="test:0",
        )
        return None


async def _stub_organizer(journal, *, index_store, agent_name="memory-organizer"):
    """Deterministic stand-in for the real (LLM) organizer.

    Mirrors what an organizer pass does at the seam we need to pin: it reads the
    journal records the owner just appended and curates them into the index —
    here, trivially one ``IndexEntry`` per journal record (no dedup/supersede,
    which is the LLM's job and out of scope for a hermetic test). __window.compact
    lazy-imports ``run_organizer`` per call, so patching the module attribute is
    picked up — same pattern as the ``consolidate`` patch.
    """
    from paf.capabilities.memory._journal import _normalize_born_on
    from paf.capabilities.memory._organize import IndexEntry

    entries = []
    nid = 1
    for _, rec in journal.read_all():
        entries.append(
            IndexEntry(
                id=f"mem-{nid}",
                type=rec.type,
                subject="consolidation",
                fact=rec.fact,
                born_on=_normalize_born_on(rec.born_on),
                updated_on=_normalize_born_on(rec.born_on),
                source_ref=rec.source_ref,
                source=[nid],
            )
        )
        nid += 1
    index_store._save(entries, watermark=len(entries), next_id=nid)
    return f"organized {len(entries)} entries"


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
    # recording write, and the real organizer (also a model call) with a stub
    # that curates the journal into the index. `_window.compact` lazy-imports
    # both per call, so patching the module attributes is picked up.
    got = _Gotcha(store_dir)
    monkeypatch.setattr(
        "paf.capabilities.memory._consolidate.consolidate", got, raising=True
    )
    monkeypatch.setattr(
        "paf.capabilities.memory._organize.run_organizer", _stub_organizer
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

    # 2. Durable memory was written into the shared store, through the new
    #    owner → organizer → index-render path: MEMORY.md comes from the curated
    #    index (the distilled fact), not the raw journal.
    assert (store_dir / "MEMORY.md").is_file()
    md = (store_dir / "MEMORY.md").read_text(encoding="utf-8")
    assert f"consolidated {expected_cutoff} evicted messages" in md
    # The index file itself was produced by the organizer step.
    assert (store_dir / "memory-index.json").is_file()

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


def test_compaction_hook_rate_limits_within_one_turn(tmp_path, monkeypatch):
    """Consolidation is rate-limited to once per eviction generation.

    The harness drives `compact` once per model request, so a long tool-heavy turn
    can cross the token trigger repeatedly and call `compact` several times over
    roughly the same history (witnessed: 5 consolidations in one turn). The window
    must fold the about-to-evict slice into memory only when it extends BEYOND what
    was last folded — a repeat call over the same/smaller cutoff is a no-op (the
    plain window still trims, but memory is not re-written).
    """
    store_dir = tmp_path / "memory"
    store_dir.mkdir()

    window = MemoryConsolidationWindow(
        max_messages=5,
        keep_messages=2,
        memory_directory=str(store_dir),
        memory_agent="assistant",
    )

    got = _Gotcha(store_dir)
    monkeypatch.setattr(
        "paf.capabilities.memory._consolidate.consolidate", got, raising=True
    )
    monkeypatch.setattr(
        "paf.capabilities.memory._organize.run_organizer", _stub_organizer
    )

    fake_ctx = types.SimpleNamespace(messages=[], model=None)

    # 1st fire: 8 messages -> cutoff 6, consolidates (fresh material).
    m8 = _make_messages(8)
    fake_ctx.messages = m8
    asyncio.run(window.compact(m8, fake_ctx))  # type: ignore[arg-type]
    assert got.cutoff == 6
    assert got.call_count == 1

    # 2nd fire, same turn: history re-inflated to the same size (a tool result
    # pushed the estimate back over the token trigger). cutoff 6 is NOT beyond
    # the last folded cutoff -> memory must NOT be re-consolidated.
    m8b = _make_messages(8)
    fake_ctx.messages = m8b
    asyncio.run(window.compact(m8b, fake_ctx))  # type: ignore[arg-type]
    assert got.call_count == 1, "repeat consolidation over same cutoff must be skipped"

    # 3rd fire: genuinely new material accumulated (12 messages). cutoff 10
    # exceeds the last folded 6 -> consolidates the newly-evictable prefix.
    m12 = _make_messages(12)
    fake_ctx.messages = m12
    asyncio.run(window.compact(m12, fake_ctx))  # type: ignore[arg-type]
    assert got.cutoff == 10
    assert got.call_count == 2
