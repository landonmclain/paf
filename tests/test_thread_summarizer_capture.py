"""Phase 1 tests: eviction capture for ThreadSummarizer.

Covers the capture record + atomic JSONL WAL (`_capture.py`), the eviction-hook
computation in `TurnBoundSlidingWindow` (correct `{run_id}:{turn_id}` refs,
OFF by default), and the wiring (capability attaches to the window only when the
profile allowlists BOTH ThreadSummarizer and Compaction).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from paf.capabilities._window import TurnBoundSlidingWindow
from paf.capabilities.thread_summarizer import ThreadSummarizer, capture_directory_for
from paf.capabilities.thread_summarizer._capture import (
    CapturedMessage,
    EvictionBlock,
    append_eviction_block,
    iter_blocks,
    next_sequence,
)


def _turn(label: str, run_id: str | None = "run-z") -> list[ModelMessage]:
    """One (user, assistant) turn with ``run_id`` stamped on both messages."""
    req = ModelRequest(parts=[UserPromptPart(content=label)])
    resp = ModelResponse(parts=[TextPart(f"reply:{label}")])
    if run_id is not None:
        req.run_id = run_id
        resp.run_id = run_id
    return [req, resp]


def _ctx_like(conversation_id: str | None):
    """A minimal stand-in exposing ``conversation_id`` (RunContext has it)."""
    return type("Ctx", (), {"conversation_id": conversation_id})


# ---------------------------------------------------------------------------
# Task 1.1 — capture record + atomic JSONL WAL
# ---------------------------------------------------------------------------


def test_capture_directory_for_is_per_agent(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    # relative to CWD (mirrors memory's DEFAULT_MEMORY_DIR data/memory/<agent>)
    assert capture_directory_for("assistant") == Path("data") / "summaries" / "assistant"
    assert capture_directory_for(None) == Path("data") / "summaries" / "agent"


def test_eviction_block_round_trips():
    block = EvictionBlock(
        seq=0,
        conversation_id="prof:c1",
        evicted_at="2026-09-07T00:00:00+00:00",
        keep_messages=20,
        messages=[
            CapturedMessage(ref="run-a:0", role="ModelRequest", text="hello"),
            CapturedMessage(ref=None, role="ModelResponse", text=None),
        ],
    )
    row = block.to_row()
    assert row["conversation_id"] == "prof:c1"
    assert row["messages"][0]["ref"] == "run-a:0"
    assert EvictionBlock.from_row(row) == block


def test_append_and_iter_blocks_are_one_jsonl_line(tmp_path):
    d = tmp_path / "s"
    append_eviction_block(d, EvictionBlock(seq=0, conversation_id="c", evicted_at="t", keep_messages=5, messages=[]))
    append_eviction_block(d, EvictionBlock(seq=1, conversation_id="c", evicted_at="t2", keep_messages=5, messages=[]))
    blocks = iter_blocks(d)
    assert [b.seq for b in blocks] == [0, 1]
    # every line is complete, parseable JSON (no torn line)
    raw = (d / "evictions.jsonl").read_text()
    for ln in raw.splitlines():
        json.loads(ln)


def test_append_is_atomic_no_staging_left(tmp_path):
    d = tmp_path / "s"
    append_eviction_block(d, EvictionBlock(seq=0, conversation_id="c", evicted_at="t", keep_messages=5, messages=[]))
    # no staging file remains after a successful append
    assert not list(d.glob(".*.staging"))


def test_seq_mismatch_raises(tmp_path):
    d = tmp_path / "s"
    append_eviction_block(d, EvictionBlock(seq=0, conversation_id="c", evicted_at="t", keep_messages=5, messages=[]))
    with pytest.raises(AssertionError):
        append_eviction_block(d, EvictionBlock(seq=3, conversation_id="c", evicted_at="t", keep_messages=5, messages=[]))


def test_next_sequence_counts_lines(tmp_path):
    d = tmp_path / "s"
    assert next_sequence(d) == 0
    append_eviction_block(d, EvictionBlock(seq=0, conversation_id="c", evicted_at="t", keep_messages=5, messages=[]))
    append_eviction_block(d, EvictionBlock(seq=1, conversation_id="c", evicted_at="t", keep_messages=5, messages=[]))
    assert next_sequence(d) == 2


def test_iter_blocks_skips_corrupt_line(tmp_path):
    d = tmp_path / "s"
    d.mkdir(parents=True)
    (d / "evictions.jsonl").write_text('{"seq": 0, "broken}\n')
    assert iter_blocks(d) == []


# ---------------------------------------------------------------------------
# Task 1.2 — window eviction hook (refs, keep_messages, OFF default)
# ---------------------------------------------------------------------------


def test_window_fires_hook_with_correct_refs():
    # history long enough that keep_messages=2 evicts the oldest turns
    msgs: list[ModelMessage] = []
    for i in range(5):
        msgs.extend(_turn(f"q{i}", run_id="run-a" if i < 3 else "run-b"))
    captured: list[tuple[list[dict], int]] = []
    window = TurnBoundSlidingWindow(max_messages=100, keep_messages=2, capture_hook=lambda rows, keep: captured.append((rows, keep)))

    asyncio.run(window.compact(msgs, _ctx_like("prof:cid")))

    assert len(captured) == 1
    rows, keep = captured[0]
    assert keep == 2
    # every evicted message's ref is {its run_id}:{its turn ordinal}: refs are
    # monotonic non-decreasing, and each turn's (request,response) pair shares it
    refs = [r["ref"] for r in rows]
    assert refs == sorted(refs)
    # the oldest turns (run-a: q0,q1,q2) and their responses are evicted;
    # the later run-b turns are kept by keep_messages=2
    assert set(refs) == {"run-a:0", "run-a:1", "run-a:2"}
    assert refs[0] == refs[1]           # first turn's req+resp share run-a:0
    # every row also carries role + text provenance
    assert all("role" in r and "text" in r for r in rows)


def test_window_hook_off_by_default_no_capture():
    window = TurnBoundSlidingWindow(max_messages=100, keep_messages=2)
    msgs: list[ModelMessage] = []
    for i in range(5):
        msgs.extend(_turn(f"q{i}"))
    out = asyncio.run(window.compact(msgs, _ctx_like("prof:cid")))
    # default: no hook, window still trims normally
    assert len(out) < len(msgs)


def test_window_no_eviction_no_hook():
    captured: list = []
    window = TurnBoundSlidingWindow(max_messages=100, keep_messages=2, capture_hook=lambda rows, keep: captured.append((rows, keep)))
    msgs = _turn("only-one-turn")
    asyncio.run(window.compact(msgs, _ctx_like("prof:cid")))
    assert captured == []  # nothing evictable -> hook not called


def test_window_never_evicts_active_turn():
    """The gate still holds with a hook attached: active turn survives."""
    captured: list = []
    window = TurnBoundSlidingWindow(max_messages=100, keep_messages=2, capture_hook=lambda rows, keep: captured.append((rows, keep)))
    msgs: list[ModelMessage] = []
    for i in range(3):
        msgs.extend(_turn(f"q{i}"))
    # active turn only, less than the window -> nothing evicted, active intact
    out = asyncio.run(window.compact(msgs, _ctx_like("prof:cid")))
    assert out[-2].parts[0].content == "q2"
    assert out[-1].parts[0].content == "reply:q2"


# ---------------------------------------------------------------------------
# Task 1.2 — capability capture → WAL, and build wiring
# ---------------------------------------------------------------------------


def test_capability_captures_conversation_and_writes_wal(tmp_path):
    cap = ThreadSummarizer(agent_name="assistant", capture_directory=tmp_path / "summ")
    # before_model_request learns the thread
    asyncio.run(cap.before_model_request(_ctx_like("prof:cid"), object()))
    cap._capture([{"ref": "run-a:0", "role": "ModelRequest", "text": "hi"}], keep_messages=3)

    blocks = iter_blocks(tmp_path / "summ")
    assert len(blocks) == 1
    assert blocks[0].conversation_id == "prof:cid"
    assert blocks[0].keep_messages == 3
    assert blocks[0].messages[0].ref == "run-a:0"
    assert blocks[0].messages[0].text == "hi"


def test_capability_attach_to_window_wires_hook(tmp_path):
    cap = ThreadSummarizer(agent_name="assistant", capture_directory=tmp_path / "summ")
    window = TurnBoundSlidingWindow(max_messages=100, keep_messages=2)
    cap.attach_to_window(window)
    assert window.capture_hook is not None

    msgs: list[ModelMessage] = []
    for i in range(5):
        msgs.extend(_turn(f"q{i}"))
    # real flow: before_model_request learns the thread at the start of the turn
    asyncio.run(cap.before_model_request(_ctx_like("prof:cid"), object()))
    asyncio.run(window.compact(msgs, _ctx_like("prof:cid")))
    blocks = iter_blocks(tmp_path / "summ")
    assert len(blocks) == 1
    assert blocks[0].conversation_id == "prof:cid"
    assert blocks[0].keep_messages == 2


def test_attach_ignores_non_window(tmp_path):
    cap = ThreadSummarizer(agent_name="a", capture_directory=tmp_path / "summ")
    cap.attach_to_window(object())  # not a TurnBoundSlidingWindow — no crash
    assert True


# ---------------------------------------------------------------------------
# Task 1.2b — build wiring: hook attaches iff profile allowlists BOTH
# ---------------------------------------------------------------------------


def _runtime_capabilities_for(names: tuple[str, ...]):
    from paf.capabilities.build import build_runtime_capabilities
    from paf.capabilities.profiles import AgentProfile

    return build_runtime_capabilities(AgentProfile(name="tmp", capability_names=names))


def _window_in(caps) -> TurnBoundSlidingWindow | None:
    # find the TurnBoundSlidingWindow tier inside whatever compaction was built
    for c in caps:
        tiers = getattr(c, "tiers", None)
        if tiers:
            for t in tiers:
                if isinstance(t, TurnBoundSlidingWindow):
                    return t
    return None


def test_build_wires_hook_when_both_allowlisted():
    """ThreadSummarizer + Compaction -> the eviction window's hook is set."""
    caps = _runtime_capabilities_for(("Compaction", "ThreadSummarizer"))
    window = _window_in(caps)
    assert window is not None
    assert window.capture_hook is not None


def test_build_does_not_wire_hook_when_thread_summarizer_absent():
    """Compaction alone -> no ThreadSummarizer capability, no hook."""
    caps = _runtime_capabilities_for(("Compaction",))
    assert not any(isinstance(c, ThreadSummarizer) for c in caps)
    window = _window_in(caps)
    assert window is not None
    assert window.capture_hook is None


def test_build_does_not_wire_hook_when_compaction_absent():
    """ThreadSummarizer alone -> no window to attach to, no crash, no hook."""
    caps = _runtime_capabilities_for(("ThreadSummarizer",))
    assert any(isinstance(c, ThreadSummarizer) for c in caps)
    assert _window_in(caps) is None
