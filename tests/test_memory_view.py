"""Tests for the isolated journal-view reader (`JournalViewStore`).

The M2 recall seam: the answerer's injected `<memory>` comes from the
journal-rendered ``MEMORY.md`` file at the TOP LEVEL of the memory dir -- NOT
from ``MarkdownStorage``'s agent-scoped pathing. ``JournalViewStore`` is that
read-only view. These tests pin the reachable read contract: read the rendered
md, ignore the scoped path, no subfiles, read-only.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from paf.capabilities.memory._view import JournalViewStore

RENDERED = (
    "## preference\n"
    "- user prefers flat-file storage — 2026-08-17T10:30:00+00:00 — seed-a1b2:12\n"
)


@pytest.fixture
def mem_dir(tmp_path: Path) -> Path:
    (tmp_path / "MEMORY.md").write_text(RENDERED, encoding="utf-8")
    return tmp_path


def test_read_returns_top_level_rendered_md(mem_dir: Path) -> None:
    store = JournalViewStore(mem_dir)

    async def _t():
        return await store.read("assistant/MEMORY.md", max_chars=100_000)

    main = asyncio.run(_t())
    assert main is not None
    assert main.content == RENDERED


def test_read_ignores_scoped_path(mem_dir: Path) -> None:
    # Injection asks for `assistant/MEMORY.md`; we answer with the root render.
    store = JournalViewStore(mem_dir)

    async def _t():
        return [
            await store.read(p, max_chars=100_000)
            for p in ("assistant/MEMORY.md", "MEMORY.md", "wyvern/other.md")
        ]

    for main in asyncio.run(_t()):
        assert main is not None
        assert main.content == RENDERED


def test_read_respects_max_chars(mem_dir: Path) -> None:
    store = JournalViewStore(mem_dir)

    async def _t():
        return await store.read("assistant/MEMORY.md", max_chars=12)

    main = asyncio.run(_t())
    assert main is not None
    assert main.content == RENDERED[:12]
    assert main.truncated is True


def test_read_missing_file_returns_none(tmp_path: Path) -> None:
    store = JournalViewStore(tmp_path)  # no MEMORY.md present

    async def _t():
        return await store.read("assistant/MEMORY.md", max_chars=100_000)

    assert asyncio.run(_t()) is None


def test_no_subfiles(mem_dir: Path) -> None:
    store = JournalViewStore(mem_dir)

    async def _t():
        return await store.list_paths("assistant/", limit=10)

    assert asyncio.run(_t()) == []


def test_read_only_blocks_writes(mem_dir: Path) -> None:
    store = JournalViewStore(mem_dir)

    async def _t():
        with pytest.raises(NotImplementedError):
            await store.write("assistant/MEMORY.md", "x", expected_version=None)
        with pytest.raises(NotImplementedError):
            await store.delete("assistant/MEMORY.md", expected_version=None)

    asyncio.run(_t())


def test_expose_tools_false_registers_no_memory_tools(mem_dir: Path) -> None:
    """The recall/assistant carries NO memory tools (inject-only journal view).

    The M2 recall ladder is search_conversation_history/read_turn (Conversation-
    Search), never memory tools. `expose_tools=False` must suppress the whole
    read/search/write/delete surface -- NOT just the write side.
    """
    from paf.capabilities.memory import Memory
    from paf.capabilities.memory._view import JournalViewStore as _JVS

    cap = Memory(
        store=_JVS(mem_dir),
        agent_name="assistant",
        inject_memory=True,
        writable=False,
        expose_tools=False,
        guidance="",
    )
    assert cap.get_toolset() is None


def test_expose_tools_true_nonjournal_registers_no_toolset(mem_dir: Path) -> None:
    """A non-journal Memory exposes NO toolset (legacy 4-tool surface removed).

    With ``MemoryToolset`` gone, `expose_tools=True` alone does not produce a
    toolset: the only remaining tool surface is the journal-append owner, which
    requires ``journal``. Anything else is injection-only and carries no tools.
    """
    from paf.capabilities.memory import Memory
    from paf.capabilities.memory._view import JournalViewStore as _JVS

    cap = Memory(
        store=_JVS(mem_dir),
        agent_name="assistant",
        inject_memory=True,
        writable=False,
        expose_tools=True,
        guidance="",
    )
    assert cap.get_toolset() is None
