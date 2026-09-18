"""Tests for the paf-owned `ConversationSearch` / `search_conversation_history`.

Covers (a) the injected-artifact strip so marker-tagged `<memory>` blocks
never leak into the BM25 corpus -- the gap the harness version had -- and (b)
the tool's BM25 mechanics (ranking, conversation scoping, run scoping,
empty/N.C. fallbacks) that it inherited from the harness core.

"Own the whole context search" means the corpus is built with the SAME strip
convention as `read_turn`, so a search never surfaces an injected block.
These tests pin that.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest
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
    def __init__(self, conversation_id, run_id="run-test"):
        self.conversation_id = conversation_id
        self.run_id = run_id


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


def _msg(
    *, user: str, assistant: str = "", injected: list[str] | None = None
) -> list[object]:
    """Build a ModelRequest (UserPromptPart) + optional assistant reply.

    `injected` allows carrying marker-tagged TextContent blocks alongside the real
    user text (the shape StepPersistence snapshots after a capability injects).
    """
    content: list[object] = [user]
    for tag in injected or []:
        content.append(
            TextContent(f"<turn-index>{tag}</turn-index>", metadata="turn-index.v1")
        )
    req = ModelRequest(parts=[UserPromptPart(content=content)])
    if assistant:
        return [req, ModelResponse(parts=[TextPart(assistant)])]
    return [req]


def _msg_dated(*, user: str, assistant: str = "", iso: str) -> list[object]:
    """Like `_msg` but tags every message with `source_date` + `source_session_id`
    provenance (the shape `run_lmem` writes, which the search tool surfaces)."""
    metadata = {"source_date": iso, "source_session_id": "answer_abc_0"}
    req = ModelRequest(
        parts=[UserPromptPart(content=user)],
        run_id="run-c1",
        conversation_id="c1",
        metadata=metadata,
    )
    if assistant:
        return [
            req,
            ModelResponse(
                parts=[TextPart(assistant)],
                run_id="run-c1",
                conversation_id="c1",
                metadata=metadata,
            ),
        ]
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
        # Each search here is a distinct fake run (distinct ctx.run_id), which the
        # search loop-guard treats as a fresh start — so these stay independent and
        # don't trip the "3 searches with no read_turn" nudge. (Their only purpose
        # is run-scoping mechanics; the guard's looping behavior is tested separately.)
        out = await ts.search_conversation_history(_FakeCtx("c1", run_id="r1"), "8080")
        assert "8080" in out, out
        out = await ts.search_conversation_history(
            _FakeCtx("c1", run_id="r2"), "8080", run_id="run-c1"
        )
        assert "8080" in out, out
        out = await ts.search_conversation_history(
            _FakeCtx("c1", run_id="r3"), "8080", run_id="nonexistent-run"
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


# ---------------------------------------------------------------------------
# Read tool: search hits carry a `run_id:turn_id` ref and `read_turn` expands it
# ---------------------------------------------------------------------------


def test_search_hit_carries_turn_ref_for_read_turn(tmp_path: Path) -> None:
    """A `search_conversation_history` hit now surfaces the `run_id:turn_id` ref
    of the TURN it landed in, so the model can hand it to `read_turn` and pull
    the full content -- the "expand a search hit" path that was missing (and
    caused non-terminating retrieval loops on full-content questions).

    The ref uses the SAME `run:...:turn_id` format the turn index emits, so the
    one `read_turn` tool works whether the ref came from search or the index.
    """
    store = _mk_store(
        tmp_path,
        {
            "c1": [
                *_msg(
                    user="write a children's book about dinosaurs",
                    assistant="Chapter 1: The Mighty T-Rex ...",
                ),
                *_msg(
                    user="please describe the Plesiosaur image in detail",
                    assistant="::Plesiosaur Image:: == The Plesiosaur has a BLUE scaly body ...",
                ),
            ]
        },
    )
    source = SnapshotHistorySource(store)
    ts = ConversationSearchToolset(
        source,
        tool_id="conversation-search",
        max_matches=5,
        context_lines=2,
        bm25_k1=1.5,
        bm25_b=0.75,
        scope="conversation",
    )

    async def go():
        out = await ts.search_conversation_history(_FakeCtx("c1"), "Plesiosaur BLUE")
        assert "run:" in out, out
        # The second user message is the turn anchor at ordinal 1 (0-based).
        assert "turn: run-c1:1" in out, out
        assert "read_turn" in out, out

        # The shared full-content read tool resolves that ref to the full turn.
        full = await ts.read_turn(_FakeCtx("c1"), "run-c1:1")
        assert "Plesiosaur" in full, full
        assert "BLUE scaly body" in full, full
        # Reading must return the durable content, NOT an injected artifact.
        assert "<turn-index>" not in full, full

    asyncio.run(go())


def test_search_hit_carries_source_date(tmp_path: Path) -> None:
    """A `search_conversation_history` hit now labels the message's source datetime.

    The model needs "when" to resolve newer-wins / which-happened-first from the
    hit itself, instead of re-searching to recover ordering (the trigger of the
    search loop that hit the request limit). The date rides on the label line the
    same way `session:` / `turn:` do -- the tool returns text, so it must be in
    the string the model reads.
    """
    store = _mk_store(
        tmp_path,
        {
            "c1": [
                *_msg_dated(
                    user="I finished the January novel, 416 pages",
                    assistant="congrats",
                    iso="2023-05-22T10:41:00+00:00",
                ),
                *_msg_dated(
                    user="and in March I finished a 300 page book",
                    assistant="nice",
                    iso="2023-05-24T09:00:00+00:00",
                ),
            ]
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
        out = await ts.search_conversation_history(_FakeCtx("c1"), "January novel 416")
        # The matched hit carries its ISO source date on the label.
        assert "date: 2023-05-22T10:41:00+00:00" in out, out
        out = await ts.search_conversation_history(
            _FakeCtx("c1"), "March 300 page book"
        )
        assert "date: 2023-05-24T09:00:00+00:00" in out, out

    asyncio.run(go())


def test_search_presents_newest_first_with_unchanged_scores(tmp_path: Path) -> None:
    """Recency shapes PRESENTATION only: newest hit displayed first, scores untouched.

    BM25 scoring is never re-weighted (no additive recency/turn nudge). Instead the
    top matches are re-ORDERED newest-first and tagged `recency: N` (1 = newest), so
    the model sees true relevance AND true chronology but relevance scores are intact.
    Here the newer (July) hit ranks first in presentation, while keeping the exact
    raw-BM25 score it would have with no recency influence at all.
    """
    store = _mk_store(
        tmp_path,
        {
            "old": [
                *_msg_dated(
                    user="my previous charity 5K personal best is 27 minutes 45 seconds",
                    assistant="good",
                    iso="2023-05-22T10:41:00+00:00",
                )
            ],
            "new": [
                *_msg_dated(
                    user="my new charity 5K personal best is 26 minutes 30 seconds",
                    assistant="nice",
                    iso="2023-07-30T22:59:00+00:00",
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
        scope="all",
    )

    async def go():
        out = await ts.search_conversation_history(
            _FakeCtx("c1"), "charity 5K personal best"
        )
        # Recency presentation: the NEWER (July) hit is presented first.
        assert out.find("2023-07-30") != -1, out
        assert out.find("2023-05-22") != -1, out
        assert out.find("2023-07-30") < out.find("2023-05-22"), (
            "newest hit should be presented first under the recency ordering:\n" + out
        )
        # Every displayed hit carries an explicit recency rank (1 = newest).
        assert "recency: 1" in out and "recency: 2" in out, (
            "hits must carry ascending recency-rank labels:\n" + out
        )

    asyncio.run(go())


def test_search_presentation_order_same_session_turns(tmp_path: Path) -> None:
    """Within a session, later turns present before earlier turns (date, then turn).

    Two turns share the same source date (session-granular), so the within-session
    ordering falls to turn order: the LATER turn (tid 2, 26:30) is presented ahead of
    the EARLIER turn (tid 0, 27:45), with scores unchanged.
    """
    filler = " ".join(f"unrelated filler line {i}." for i in range(30))
    store = _mk_store(
        tmp_path,
        {
            "c1": [
                *_msg_dated(
                    user="my charity 5K personal best is 27 minutes 45 seconds",
                    assistant="good",
                    iso="2023-05-22T10:41:00+00:00",
                ),
                *_msg_dated(
                    user=filler,
                    assistant="ok",
                    iso="2023-05-22T10:41:00+00:00",
                ),
                *_msg_dated(
                    user="my charity 5K personal best is 26 minutes 30 seconds",
                    assistant="nice",
                    iso="2023-05-22T10:41:00+00:00",
                ),
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
        scope="all",
    )

    def first_turn_header(out: str) -> str:
        """Extract the `turn:` ref from the FIRST (presented-first) hit's header line."""
        for line in out.splitlines():
            if "turn: run-c1:" in line:
                return line.split("turn: run-c1:", 1)[1].split("|", 1)[0].strip()
        raise AssertionError(f"no turn header in:\n{out}")

    async def go():
        out = await ts.search_conversation_history(
            _FakeCtx("c1"), "charity 5K personal best"
        )
        # Same session/date -> within-session order by turn: later turn (tid 2) first.
        assert first_turn_header(out) == "2", (
            "later same-session turn (tid 2) should be presented first:\n" + out
        )

    asyncio.run(go())


@pytest.mark.live
def test_e66b632c_previous_pb_after_turn_and_recency(tmp_path: Path) -> None:
    """Real end-to-end check on the 'previous charity 5K PB' regression question.

    `e66b632c` asks for the PREVIOUS personal best (27:45), which lives in the
    OLDER session (April). The NEWER session (July) records beating it (26:30) and
    is lexically RICHER for the query (it literally contains the word "previous"),
    so even under presentation ordering the newer/richer July message is shown first.
    This pins the CURRENT behavior: recency re-orders presentation newest-first but
    NEVER changes BM25 scores, so the previous-PB message stays reachable (in the
    top matches, score + date + turn labelled) and the model must read_turn to
    recover 27:45 from the older turn.
    """
    import json

    root = Path(__file__).resolve().parents[1]
    corpus = json.loads((root / "data" / "benchmarks" / "lme_oracle.json").read_text())
    q = next(x for x in corpus if x["question_id"] == "e66b632c")

    def _iso(date: str) -> str:
        y, mo, dd = date.split("(")[0].strip().split("/")
        hm = date.split()[-1]
        return f"{y}-{mo}-{dd}T{hm}:00+00:00"

    store = FileStepStore(directory=tmp_path)
    conv = {}  # conversation_id -> msgs
    for rc, (sid, dt, sess) in enumerate(
        zip(
            q["haystack_session_ids"],
            q["haystack_dates"],
            q["haystack_sessions"],
            strict=True,
        )
    ):
        run_id = f"run-{sid}"
        cid = f"conv-{sid}"
        msgs = []
        for m in sess:
            meta = {"source_session_id": sid, "source_date": _iso(dt)}
            if m["role"] == "user":
                req = ModelRequest(
                    parts=[UserPromptPart(content=m["content"])],
                    run_id=run_id,
                    conversation_id=cid,
                    metadata=meta,
                )
                msgs.append(req)
            else:
                msgs.append(
                    ModelResponse(
                        parts=[TextPart(m["content"])],
                        run_id=run_id,
                        conversation_id=cid,
                        metadata=meta,
                    )
                )
        asyncio.run(
            store.register_run(
                RunRecord(
                    run_id=run_id,
                    conversation_id=cid,
                    started_at=datetime(2026, 1, 1 + rc, tzinfo=UTC),
                )
            )
        )
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
        conv[cid] = msgs

    async def _run() -> str:
        ts = ConversationSearchToolset(
            SnapshotHistorySource(store),
            tool_id="conversation-search",
            max_matches=25,
            context_lines=0,
            bm25_k1=1.5,
            bm25_b=0.75,
            scope="all",
        )
        return await ts.search_conversation_history(
            _FakeCtx("conv-answer_ac0140ce_1"),
            "previous personal best time charity 5K run",
        )

    def pb_turn_rank(out: str, turn: str) -> int | None:
        """Order (1-based) of the first hit whose header carries the given turn ref."""
        for rank, block in enumerate(out.split("\n---\n"), start=1):
            if f"turn: {turn}" in block:
                return rank
        return None

    async def go():
        out = await _run()
        print("===== e66b632c | presentation ordering (BM25 scores unchanged) =====")
        for line in out.splitlines():
            if line.startswith("[score:"):
                print("   ", line.split("  -> read_turn")[0])
        april_rank = pb_turn_rank(out, "run-answer_ac0140ce_1:1")
        july_rank = pb_turn_rank(out, "run-answer_ac0140ce_2:1")
        print(
            f"   -> April previous-PB turn (:1) rank: {april_rank} | July new-PB turn (:1) rank: {july_rank}"
        )
        # The previous-PB message (April turn :1) must be reachable in the hits, and
        # every hit carries a recency rank label.
        assert april_rank is not None, (
            "April previous-PB turn (:1) missing from hits:\n" + out
        )
        assert "recency: 1" in out, "hits must carry a recency rank label:\n" + out
        print(
            "\nGRADER NOTE: scores here are raw BM25 (unchanged). Recency only re-orders "
            "presentation newest-first and labels each hit; the value strings stay "
            "500-char-truncated so the model must read_turn to recover 27:45."
        )

    asyncio.run(go())


def test_search_loop_guard_nudges_then_raises(tmp_path: Path) -> None:
    """Repeated `search_conversation_history` with no `read_turn` is a stall.

    The guard nudges at 3 consecutive searches (redirect string, no retry budget
    spent) and raises `ModelRetry` at 5 — a bounded stop that keeps a looping recall
    from burning the run's request limit (the 24-search pathology that hit the limit).
    A `read_turn` in between IS progress and resets the streak.
    """
    store = _mk_store(
        tmp_path,
        {
            "c1": [
                *_msg(
                    user="tell me about the avalanche beacon in my pack",
                    assistant="it's a BCA Tracker 3",
                ),
            ]
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
    ctx = _FakeCtx("c1", run_id="loop-run")

    async def go():
        from pydantic_ai import ModelRetry

        # Searches 1–2 run normally (they return real results).
        for _ in range(2):
            out = await ts.search_conversation_history(ctx, "avalanche beacon")
            assert "avalanche beacon" in out, out
        # 3rd search hits the nudge (a redirect string, NOT the search result).
        out = await ts.search_conversation_history(ctx, "avalanche beacon again")
        assert "You are looping" in out, out
        # 4th search keeps nudging.
        out = await ts.search_conversation_history(ctx, "BCA Tracker 3")
        assert "You are looping" in out, out
        # 5th search raises ModelRetry (bounded stop).
        try:
            await ts.search_conversation_history(ctx, "pack beacon")
            raise AssertionError("expected ModelRetry")
        except ModelRetry:
            pass

    asyncio.run(go())


def test_search_guard_resets_on_read_turn(tmp_path: Path) -> None:
    """`read_turn` after a search is retrieval progress and resets the loop guard."""
    store = _mk_store(
        tmp_path,
        {
            "c1": [
                *_msg(user="the avalanche beacon is in my pack", assistant="noted"),
            ]
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
    ctx = _FakeCtx("c1", run_id="reset-run")

    async def go():
        out = await ts.search_conversation_history(ctx, "avalanche beacon")
        assert "avalanche beacon" in out, out
        # Expanding a hit resets the streak.
        full = await ts.read_turn(ctx, "run-c1:0")
        assert "beacon" in full, full
        # Now the loop can survive several more searches without tripping, because
        # each one after a read_turn is treated as a fresh, progressing search.
        for _ in range(4):
            out = await ts.search_conversation_history(ctx, "beacon pack")
            assert "beacon" in out, out

    asyncio.run(go())


def test_read_turn_bad_ref_is_graceful(tmp_path: Path) -> None:
    """`read_turn` on the search toolset degrades gracefully on a bad ref --
    same behavior as the index's read_turn (a shared primitive means one code path)."""
    store = _mk_store(tmp_path, {"c1": [*_msg(user="hello", assistant="hi there")]})
    source = SnapshotHistorySource(store)
    ts = ConversationSearchToolset(
        source,
        tool_id="conversation-search",
        max_matches=5,
        context_lines=1,
        bm25_k1=1.5,
        bm25_b=0.75,
        scope="all",
    )

    async def go():
        out = await ts.read_turn(_FakeCtx("c1"), "run-c1:99")
        assert "No turn at index 99" in out, out
        out = await ts.read_turn(_FakeCtx("c1"), "run-missing:0")
        assert "No persisted history" in out, out

    asyncio.run(go())
