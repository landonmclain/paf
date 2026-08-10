"""Tests for the paf-owned `ConversationSearch` / `search_conversation_history`.

Covers (a) the injected-artifact strip so marker-tagged `<turn-index>` /
`<memory>` blocks never leak into the BM25 corpus -- the gap the harness version
had -- and (b) the tool's BM25 mechanics (ranking, conversation scoping,
run scoping, empty/N.C. fallbacks) that it inherited from the harness core.

"Own the whole context search" means the corpus is built with the SAME strip
convention as `read_turn` / `search_index`, so a search never surfaces an
injected block. These tests pin that.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextContent,
    TextPart,
    UserPromptPart,
)
from pydantic_ai_harness.conversation_search import SnapshotHistorySource
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    FileStepStore,
    RunRecord,
)

from paf.capabilities.search import ConversationSearchToolset


class _FakeCtx:
    def __init__(self, conversation_id):
        self.conversation_id = conversation_id


def _mk_store(tmp_path: Path, conversations: dict[str, list[object]]) -> FileStepStore:
    """Write a FileStepStore with one run per conversation carrying the given messages.

    Each conversation becomes a single run with a deterministic run_id, so the
    search tool has real persisted snapshots (the corpus) to search.
    """
    store = FileStepStore(directory=tmp_path)
    for i, (cid, msgs) in enumerate(conversations.items()):
        run_id = f"run-{cid}"
        # StepPersistence writes a run record on run start (register_run) and
        # snapshots each boundary (save_snapshot). SnapshotHistorySource.list_runs
        # only sees runs with a run.json, so we register each run too.
        asyncio.run(
            store.register_run(
                RunRecord(
                    run_id=run_id,
                    conversation_id=cid,
                    started_at=datetime(2026, 1, 1 + i, tzinfo=UTC),
                )
            )
        )
        # One snapshot holding the full message list -- matches how StepPersistence
        # snapshots each boundary.
        asyncio.run(
            store.save_snapshot(
                ContinuableSnapshot(
                    run_id=run_id,
                    step_index=0,
                    conversation_id=cid,
                    messages=msgs,
                )
            )
        )
    return store


def _msg(*, user: str, assistant: str = "", injected: list[str] | None = None) -> list[object]:
    """Build a ModelRequest (UserPromptPart) + optional assistant reply.

    `injected` allows carrying marker-tagged TextContent blocks alongside the real
    user text (the shape StepPersistence snapshots after a capability injects).
    """
    content: list[object] = [user]
    for tag in injected or []:
        content.append(TextContent(f"<turn-index>{tag}</turn-index>", metadata="turn-index.v1"))
    req = ModelRequest(parts=[UserPromptPart(content=content)])
    if assistant:
        return [req, ModelResponse(parts=[TextPart(assistant)])]
    return [req]


# ---------------------------------------------------------------------------
# Marker-strip -- the reason paf owns the search
# ---------------------------------------------------------------------------


def test_injected_blocks_never_leak_into_corpus(tmp_path: Path) -> None:
    """A persisted User message carrying an injected `<turn-index>` block must search
    only its REAL durable text, not the injected artifact.

    This is the gap the harness `ConversationSearch` had: it rendered marker-tagged
    `UserPromptPart` blocks verbatim, so a search for text that only exists in the
    injected block would match. Our corpus strips it.
    """
    store = _mk_store(
        tmp_path,
        {
            "c1": _msg(
                user="tell me about the database schema",
                assistant="the orders table is here",
                injected=["[abc:1] user asked earlier about ports"],
            ),
        },
    )
    ts = ConversationSearchToolset(
        SnapshotHistorySource(store),
        tool_id="conversation-search",
        max_matches=5,
        context_lines=2,
        bm25_k1=1.5,
        bm25_b=0.75,
        scope="conversation",
    )

    async def go():
        # Real durable content is findable.
        out = await ts.search_conversation_history(_FakeCtx("c1"), "orders table")
        assert "orders table is here" in out, out
        # Text that ONLY exists inside the injected block must NOT match.
        out = await ts.search_conversation_history(_FakeCtx("c1"), "ports")
        assert "No matches" in out, out

    asyncio.run(go())


def test_injected_memory_block_leak_is_stripped(tmp_path: Path) -> None:
    """The harness `Memory` scope-qualified marker (`pydantic-ai-harness.memory.v1:...`)
    is also stripped -- a recalled User message doesn't surface a stale memory block."""
    store = _mk_store(
        tmp_path,
        {
            "c1": [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content=[
                                "what timers are set",
                                TextContent(
                                    "<memory>set a reminder for 4pm</memory>",
                                    metadata="pydantic-ai-harness.memory.v1:abc123",
                                ),
                            ]
                        )
                    ]
                )
            ],
        },
    )
    ts = ConversationSearchToolset(
        SnapshotHistorySource(store),
        tool_id="conversation-search",
        max_matches=5,
        context_lines=2,
        bm25_k1=1.5,
        bm25_b=0.75,
        scope="conversation",
    )

    async def go():
        out = await ts.search_conversation_history(_FakeCtx("c1"), "timers")
        assert "timers" in out, out
        out = await ts.search_conversation_history(_FakeCtx("c1"), "4pm")
        assert "No matches" in out, out

    asyncio.run(go())


# ---------------------------------------------------------------------------
# Mechanics (inherited from the harness BM25 core)
# ---------------------------------------------------------------------------


def test_ranks_rare_terms_higher(tmp_path: Path) -> None:
    """BM25: the doc containing the rare query term ranks above filler."""
    store = _mk_store(
        tmp_path,
        {
            "c1": [
                *_msg(user="the avalanche beacon is in my pack", assistant="noted"),
                *_msg(user="the weather looks clear today", assistant="ok"),
            ],
        },
    )
    ts = ConversationSearchToolset(
        SnapshotHistorySource(store),
        tool_id="conversation-search",
        max_matches=5,
        context_lines=1,
        bm25_k1=1.5,
        bm25_b=0.75,
        scope="conversation",
    )

    async def go():
        out = await ts.search_conversation_history(_FakeCtx("c1"), "avalanche beacon")
        assert "avalanche beacon is in my pack" in out, out
        # The weather doc shouldn't rank for the rare term "avalanche".
        assert "weather looks clear" not in out, out

    asyncio.run(go())


def test_conversation_scoping_excludes_other_conversations(tmp_path: Path) -> None:
    """Under `scope='conversation'`, a run in c2's history must not surface for c1."""
    store = _mk_store(
        tmp_path,
        {
            "c1": [*_msg(user="the vault code is zip-tie", assistant="ok")],
            "c2": [*_msg(user="the vault code is 7729", assistant="ok")],
        },
    )
    ts = ConversationSearchToolset(
        SnapshotHistorySource(store),
        tool_id="conversation-search",
        max_matches=5,
        context_lines=1,
        bm25_k1=1.5,
        bm25_b=0.75,
        scope="conversation",
    )

    async def go():
        out = await ts.search_conversation_history(_FakeCtx("c1"), "zip-tie")
        assert "zip-tie" in out, out
        out = await ts.search_conversation_history(_FakeCtx("c1"), "7729")
        assert "No matches" in out, out

    asyncio.run(go())


def test_fail_closed_without_conversation_id(tmp_path: Path) -> None:
    """Under `scope='conversation'`, a run with no conversation_id searches nothing."""
    store = _mk_store(tmp_path, {"c1": [*_msg(user="hello", assistant="hi")]})
    ts = ConversationSearchToolset(
        SnapshotHistorySource(store),
        tool_id="conversation-search",
        max_matches=5,
        context_lines=1,
        bm25_k1=1.5,
        bm25_b=0.75,
        scope="conversation",
    )

    async def go():
        out = await ts.search_conversation_history(_FakeCtx(None), "hello")
        assert "needs a conversation id" in out, out

    asyncio.run(go())


def test_run_scoped_search(tmp_path: Path) -> None:
    """`run_id=...` restricts the corpus to one run."""
    store = _mk_store(
        tmp_path,
        {
            "c1": [
                *_msg(user="port 8080 in this run", assistant="ok"),
                *_msg(user="port 9090 in the other run", assistant="ok"),
            ],
        },
    )
    ts = ConversationSearchToolset(
        SnapshotHistorySource(store),
        tool_id="conversation-search",
        max_matches=5,
        context_lines=1,
        bm25_k1=1.5,
        bm25_b=0.75,
        scope="all",  # not scoping to conversation, so run_id is the only filter
    )
    # Both messages are in the same run here (single run per cid), so this is a
    # light check: run-scoping still resolves. A malformed run_id yields nothing.

    async def go():
        out = await ts.search_conversation_history(_FakeCtx("c1"), "8080")
        assert "8080" in out, out
        out = await ts.search_conversation_history(
            _FakeCtx("c1"), "8080", run_id="run-c1"
        )
        assert "8080" in out, out
        out = await ts.search_conversation_history(
            _FakeCtx("c1"), "8080", run_id="nonexistent-run"
        )
        assert "No persisted history for run" in out, out

    asyncio.run(go())


def test_run_scoped_unknown_run_message(tmp_path: Path) -> None:
    """A run_id that isn't in the store yields the 'No persisted history for run'
    answer (same as the harness), not a bare no-match."""
    store = _mk_store(tmp_path, {"c1": [*_msg(user="hello", assistant="hi")]})
    ts = ConversationSearchToolset(
        SnapshotHistorySource(store),
        tool_id="conversation-search",
        max_matches=5,
        context_lines=1,
        bm25_k1=1.5,
        bm25_b=0.75,
        scope="all",
    )

    async def go():
        out = await ts.search_conversation_history(
            _FakeCtx("c1"), "hello", run_id="ghost"
        )
        assert "No persisted history" in out, out

    asyncio.run(go())
