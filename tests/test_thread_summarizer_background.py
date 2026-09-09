"""Tests for the conversation-scoped thread summarizer worker contract.

These exercise the deterministic mechanics of :class:`ThreadSummarizerQueue`
against the conversation-span + per-conversation ``MemoryWatermark`` model:

  1. A pre-resolved turn span is summarized (curated) and DUrable beats appended,
     then the per-conversation watermark advances to the span's target.
  2. An empty span is a successful pass: watermark still advances (nothing new).
  3. A thrown summarize op advances NOTHING so the span is retried next cadence.
  4. Retry after a partial failure is idempotent (anchor-based append).
  5. Session/job conversation mismatch is rejected.
  6. Queue status/drain single-flight shape.

The summarize seam is injected so no LLM is needed (mirrors the memory worker's
journal-fenced tests).
"""

from __future__ import annotations

import asyncio

import pytest

from paf.background import QueueState
from paf.capabilities.memory._watermark import MemoryWatermark
from paf.capabilities.thread_summarizer._background import (
    ThreadSummarizeJob,
    ThreadSummarizerQueue,
)
from paf.capabilities.thread_summarizer._store import (
    THREAD_WATERMARK_FILENAME,
    ThreadSummaryRecord,
    ThreadSummaryStore,
)
from paf.capabilities.thread_summarizer._summarize import Beat, BeatsRecord

CID = "profile:cid"


def _target(run: str = "run-a", turn: int = 1) -> MemoryWatermark:
    from datetime import UTC, datetime

    return MemoryWatermark(run_id=run, turn_id=turn, consumed_ts=datetime.now(UTC))


def _wm(store: ThreadSummaryStore) -> MemoryWatermark:
    return MemoryWatermark.load(store.directory, filename=THREAD_WATERMARK_FILENAME)


def _beats(run: str = "run-a", turn: int = 0) -> BeatsRecord:
    return BeatsRecord(
        turns_covered=[f"{run}:{turn}"],
        beats=[Beat(at=f"{run}:{turn}", what=f"did {run}:{turn}")],
        summary=f"started {run}:{turn} and progressed",
    )


async def _async_beats(*_args, **__kwargs) -> BeatsRecord:
    return _beats()


def test_queue_summarizes_span_and_advances_watermark(tmp_path) -> None:
    root = tmp_path / "assistant"
    seen: list[MemoryWatermark] = []

    async def summarize(span, conversation_id):
        seen.append(conversation_id)
        return BeatsRecord(
            turns_covered=["run-a:0", "run-a:1"],
            beats=[Beat(at="run-a:0", what="hello")],
            summary="started and progressed",
        )

    queue = ThreadSummarizerQueue(capture_root=str(root), summarize_span_fn=summarize)
    target = _target(turn=1)

    async def scenario():
        queue.enqueue(
            CID,
            ThreadSummarizeJob(
                conversation_id=CID,
                messages=_fake_span(2),
                watermark=target,
            ),
        )
        await queue.drain(CID)

    asyncio.run(scenario())

    store = ThreadSummaryStore(root, CID)
    assert seen == [CID]
    assert _wm(store).run_id == "run-a"
    assert _wm(store).turn_id == 1  # advanced to the target on success
    recs = store.existing_summaries()
    assert len(recs) == 1
    # record anchor === the watermark advance position (the span's end), not
    # the first beat ref
    assert recs[0].run_id == "run-a" and recs[0].turn_id == 1


def test_queue_advances_on_empty_span_without_writing(tmp_path) -> None:
    root = tmp_path / "assistant"
    queue = ThreadSummarizerQueue(capture_root=str(root), summarize_span_fn=_beats)

    asyncio.run(
        queue.process_job(
            CID, ThreadSummarizeJob(conversation_id=CID, messages=[], watermark=_target(turn=0)), QueueState()
        )
    )

    store = ThreadSummaryStore(root, CID)
    assert _wm(store).turn_id == 0  # successful empty pass still advances
    assert not store.existing_summaries()


def test_queue_failure_does_not_advance_watermark(tmp_path) -> None:
    root = tmp_path / "assistant"

    async def fail(_span, _cid):
        raise RuntimeError("model unavailable")

    queue = ThreadSummarizerQueue(capture_root=str(root), summarize_span_fn=fail)
    with pytest.raises(RuntimeError, match="model unavailable"):
        asyncio.run(
            queue.process_job(
                CID,
                ThreadSummarizeJob(
                    conversation_id=CID, messages=_fake_span(2), watermark=_target(turn=1)
                ),
                QueueState(),
            )
        )

    store = ThreadSummaryStore(root, CID)
    # watermark untouched -> span retried next cadence, never skipped
    wm = _wm(store)
    assert wm.run_id == "" and wm.turn_id is None


def test_queue_retry_is_idempotent_after_partial_failure(tmp_path) -> None:
    root = tmp_path / "assistant"
    calls = 0

    async def flaky(span, cid):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("first pass failed")
        return _beats(tmp_path / "not-used" if False else "run-a", 0)

    queue = ThreadSummarizerQueue(capture_root=str(root), summarize_span_fn=flaky)
    job = ThreadSummarizeJob(
        conversation_id=CID, messages=_fake_span(2), watermark=_target(turn=1)
    )
    with pytest.raises(RuntimeError):
        asyncio.run(queue.process_job(CID, job, QueueState()))

    asyncio.run(queue.process_job(CID, job, QueueState()))

    store = ThreadSummaryStore(root, CID)
    assert _wm(store).turn_id == 1
    # idempotent: a retried pass over the same anchor writes exactly one record
    assert len(store.existing_summaries()) == 1


def test_queue_rejects_mismatched_session_and_job(tmp_path) -> None:
    queue = ThreadSummarizerQueue(capture_root=str(tmp_path))
    with pytest.raises(ValueError, match="session_id"):
        asyncio.run(
            queue.process_job(
                "profile:other", ThreadSummarizeJob(conversation_id=CID), QueueState()
            )
        )


def test_queue_status_and_drain_single_worker(tmp_path) -> None:
    root = tmp_path / "assistant"
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow(span, cid):
        started.set()
        await release.wait()
        return _beats()

    async def scenario():
        queue = ThreadSummarizerQueue(capture_root=str(root), summarize_span_fn=slow)
        queue.enqueue(CID, ThreadSummarizeJob(conversation_id=CID, messages=_fake_span(2), watermark=_target()))
        await started.wait()
        assert queue.status(CID).running
        release.set()
        assert await queue.drain_with_timeout(CID, 1.0)
        assert queue.status(CID).queued == 0

    asyncio.run(scenario())


def test_queue_stamps_block_time_range(tmp_path) -> None:
    """Each summarized block records the span's wall-clock range.

    ``started_at`` = first distinct turn's user-prompt timestamp;
    ``ended_at`` = last distinct turn's user-prompt timestamp. Distinct from
    ``summarized_at`` (when the worker ran). Empty strings when no timestamp
    resolves (e.g. legacy/injected-only messages).
    """
    from datetime import UTC, datetime, timedelta

    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        TextPart,
        UserPromptPart,
    )

    root = tmp_path / "assistant"
    t0 = datetime(2026, 9, 9, 9, 0, 0, tzinfo=UTC)
    t1 = t0 + timedelta(minutes=30)
    t2 = t0 + timedelta(hours=3)

    # three distinct real turns with explicit, well-separated user-prompt times
    span = []
    for i, ts in enumerate((t0, t1, t2)):
        req = ModelRequest(parts=[UserPromptPart(content=f"u{i}", timestamp=ts)])
        req.run_id = "run-a"
        resp = ModelResponse(parts=[TextPart(f"a{i}")])
        resp.run_id = "run-a"
        span.append(req)
        span.append(resp)

    queue = ThreadSummarizerQueue(
        capture_root=str(root),
        summarize_span_fn=lambda span, cid: _async_beats(),
    )
    target = _target(turn=2)

    async def scenario():
        queue.enqueue(
            CID,
            ThreadSummarizeJob(conversation_id=CID, messages=span, watermark=target),
        )
        await queue.drain(CID)

    asyncio.run(scenario())

    store = ThreadSummaryStore(root, CID)
    recs = store.existing_summaries()
    assert len(recs) == 1
    # range = first .. last distinct user-prompt timestamps (NOT summarized_at)
    assert recs[0].started_at == t0.isoformat()
    assert recs[0].ended_at == t2.isoformat()
    assert recs[0].started_at != recs[0].summarized_at
    assert recs[0].ended_at != recs[0].summarized_at


def test_queue_time_range_defaults_empty_without_timestamps(tmp_path) -> None:
    """A block whose turns carry no resolvable timestamp stores empty range."""

    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        TextPart,
        UserPromptPart,
    )

    root = tmp_path / "assistant"
    span = []
    for i in range(5):
        # UserPromptPart timestamp is default-instantiated; simulate unset by
        # building a request with no UserPromptPart timestamp reachable.
        req = ModelRequest(parts=[UserPromptPart(content=f"u{i}")])
        req.run_id = "run-a"
        resp = ModelResponse(parts=[TextPart(f"a{i}")])
        resp.run_id = "run-a"
        span.append(req)
        span.append(resp)

    # Delete the timestamp field on every user part to emulate a legacy/no-ts
    # message path (field is required by the dataclass, so set to None).
    for m in span:
        for p in getattr(m, "parts", []):
            if isinstance(p, UserPromptPart):
                object.__setattr__(p, "timestamp", None)

    queue = ThreadSummarizerQueue(
        capture_root=str(root), summarize_span_fn=_async_beats
    )
    asyncio.run(
        queue.process_job(
            CID,
            ThreadSummarizeJob(
                conversation_id=CID, messages=span, watermark=_target(turn=4)
            ),
            QueueState(),
        )
    )

    store = ThreadSummaryStore(root, CID)
    recs = store.existing_summaries()
    assert len(recs) == 1
    assert recs[0].started_at == ""
    assert recs[0].ended_at == ""


def test_queue_catches_up_over_stale_watermark_without_crashing(tmp_path) -> None:
    """A watermark behind the already-persisted summary file converges, not crashes.

    Regression for the live deadlock: a stale watermark (e.g. re-loaded after a
    process restart, or shared across two workers) resolves a span that a
    divergent lineage already summarized and persisted. The worker must NOT
    hard-fail on "anchor must increase" forever — it should skip the
    already-covered block(s), advance the watermark past them, and summarize
    only the genuinely-new tail.
    """
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        TextPart,
        UserPromptPart,
    )

    root = tmp_path / "assistant"

    # Pre-seed the store: a summary already persisted at (run-b, turn=30) —
    # i.e. the file is "ahead" of the watermark (run-a, turn=2) a stale reader
    # would load. This is the divergent-lineage state from the bug.
    store = ThreadSummaryStore(root, CID)
    store.append_summary(
        ThreadSummaryRecord(
            conversation_id=CID,
            run_id="run-b",
            turn_id=30,
            summarized_at="2026-09-09T10:00:00+00:00",
            summary="already summarized the later turns",
            refs=["run-b:30"],
        )
    )

    summarize_calls: list[list] = []

    async def tracking_summarize(span, cid):
        summarize_calls.append(span)
        return _beats("run-c", 0)

    queue = ThreadSummarizerQueue(capture_root=str(root), summarize_span_fn=tracking_summarize)

    # The resolved span is OLD (runs 'run-a'..'run-c') relative to what's
    # persisted (run-b turn 30) — exactly the stale-watermark re-coverage case.
    # The message run_ids are staggered so blocks sort older than the file.
    span = []
    for i in range(25):  # 25 turns -> covers old runs
        req = ModelRequest(parts=[UserPromptPart(content=f"u{i}")])
        req.run_id = "run-a" if i < 20 else "run-c"
        resp = ModelResponse(parts=[TextPart(f"a{i}")])
        resp.run_id = "run-a" if i < 20 else "run-c"
        span.append(req)
        span.append(resp)

    # The job's watermark target is the old tip (run-a turn 24), NOT the
    # persisted run-b:30 — a stale reader that never saw run-b's writes.
    target = MemoryWatermark(run_id="run-a", turn_id=24, consumed_ts=None)

    # process_job must NOT raise "anchor must increase" — the covered blocks are
    # skipped and the watermark advances; nothing is written for covered turns.
    asyncio.run(
        queue.process_job(
            CID,
            ThreadSummarizeJob(conversation_id=CID, messages=span, watermark=target),
            QueueState(),
        )
    )

    # Only the truly-new tail (run-c) is summarized, not the covered old turns.
    assert len(summarize_calls) >= 1
    # The watermark advanced to the span's target (run-a:24), past the covered
    # range, so the next cadence won't re-cover it.
    wm = _wm(store)
    assert wm.turn_id == 24
    # No crash, and covered old turns did not append duplicate blocks.
    assert len(store.existing_summaries()) >= 1  # pre-seed + any new tail


def _fake_span(n: int, run_id: str = "run-a") -> list:
    """A minimal persisted span with ``n`` complete turns (user + assistant).

    Messages carry the ``run_id`` stamp the store persists, so refs resolve.
    """
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        TextPart,
        UserPromptPart,
    )

    msgs: list = []
    for i in range(n):
        req = ModelRequest(parts=[UserPromptPart(content=f"u{i}")])
        resp = ModelResponse(parts=[TextPart(f"a{i}")])
        req.run_id = run_id
        resp.run_id = run_id
        msgs.append(req)
        msgs.append(resp)
    return msgs


def test_queue_chunks_span_into_10_turn_blocks(tmp_path) -> None:
    """A >10-turn span is folded as consecutive 10-turn blocks, each its own record.

    The watermark advances only past successfully-summarized blocks, so a 20-turn
    span yields exactly two 10-turn records and watermarks at the last turn covered.
    """
    import asyncio

    root = tmp_path / "assistant"
    seen_turns: list[int] = []

    async def summarize(span, conversation_id):

        seen_turns.append(len(_refs(span)))
        refs = _refs(span)
        return BeatsRecord(
            turns_covered=refs,
            beats=[Beat(at=refs[0], what="block beat") for _ in refs if refs],
            summary="a ten-turn block",
        )

    def _refs(msgs):
        from paf.capabilities.thread_summarizer._summarize import _ref_for

        return [r for r in (_ref_for(msgs, i) for i in range(len(msgs))) if r][::2]

    queue = ThreadSummarizerQueue(capture_root=str(root), summarize_span_fn=summarize)
    target = _target(turn=19)

    async def scenario():
        queue.enqueue(
            CID,
            ThreadSummarizeJob(
                conversation_id=CID,
                messages=_fake_span(20, run_id="run-a"),
                watermark=target,
            ),
        )
        await queue.drain(CID)

    asyncio.run(scenario())

    store = ThreadSummaryStore(root, CID)
    recs = store.existing_summaries()
    assert len(recs) == 2  # 20 turns -> two 10-turn blocks
    assert seen_turns == [10, 10]  # the summarize fn saw two 10-turn slices
    assert {r.run_id for r in recs} == {"run-a"}
    assert sorted(r.turn_id for r in recs) == [9, 19]  # anchored at each block's last turn
    assert _wm(store).turn_id == 19  # advanced past the final block


def test_distinct_turns_collapse_repersisted_and_injected(tmp_path) -> None:
    """Blocks chunk on *distinct real* turns, never raw addresses.

    Guards the live-branch bug: a span whose real user prompts appear at many
    re-persisted anchor addresses (and interleaved with injected <memory>
    blocks) was chunked by raw addresses — 28 addresses of ~10 distinct prompts
    produced 3 tool/noise-heavy blocks instead of 1 clean 10-prompt block. The
    gate and worker must share the same "distinct real turn" yardstick.
    """
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        TextContent,
        TextPart,
        UserPromptPart,
    )

    from paf.capabilities._markers import injected_artifact_markers
    from paf.capabilities.thread_summarizer._background import _distinct_complete_turns

    marker = injected_artifact_markers()[1] + ":scope_hash"

    # The same real prompt, re-persisted at 3 different anchors (clean, with an
    # injected block, and again) — ALL are one distinct turn.
    def _user() -> ModelRequest:
        return ModelRequest(parts=[UserPromptPart(content="real question")], run_id="run-a")

    def _injected() -> ModelRequest:
        return ModelRequest(
            parts=[
                UserPromptPart(
                    content=[TextContent(content="\n\n<memory>\n### MEMORY.md\n...", metadata=marker)]
                )
            ],
            run_id="run-a",
        )

    def _resp(txt: str) -> ModelResponse:
        r = ModelResponse(parts=[TextPart(txt)])
        return r

    # A span with one real prompt at 3 addresses + an injected-only request in
    # between. Real answer differs per copy so the slice still carries its turn.
    span = [
        _user(), _resp("answer0"),          # distinct turn 1 (kept, first addr)
        _injected(),                        # not a user turn
        _user(), _resp("answer-b"),         # re-persisted copy -> dropped
        _user(), _resp("answer-c"),         # re-persisted copy -> dropped
    ]
    turns, anchors = _distinct_complete_turns(span)
    assert len(turns) == 1, turns          # ONE distinct real turn, not 3
    assert len(anchors) == 1
    assert anchors == [0]                  # kept the FIRST address

    # Now 12 distinct real prompts across 24 addresses (each duplicated) must
    # collapse to 12 distinct turns (not 24), so chunking yields blocks of real
    # prompts — and the worker gate/block count agree.
    big: list = []
    for i in range(12):
        req = ModelRequest(parts=[UserPromptPart(content=f"prompt{i}")], run_id="run-a")
        resp = ModelResponse(parts=[TextPart(f"answer{i}")])
        big.append(req)
        big.append(resp)
        # duplicate the prompt address again
        req2 = ModelRequest(parts=[UserPromptPart(content=f"prompt{i}")], run_id="run-a")
        big.append(req2)
    turns2, anchors2 = _distinct_complete_turns(big)
    assert len(turns2) == 12, len(turns2)   # dedup the 12 duplicate addresses
    assert len(anchors2) == 12
    # each distinct turn keeps a real user prompt (never an injected block)
    for t in turns2:
        assert any(
            isinstance(p.content, str) and p.content.startswith("prompt")
            for p in (pp for m in t for pp in m.parts)
            if getattr(p, "content", None) is not None
        )
