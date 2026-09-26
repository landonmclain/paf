"""Tests for the M2 journal-append owner surface + labeled owner input.

Covers the two seams landed to make ``source_ref`` / ``born_on`` meaningful:

1. **``JournalAppendToolset``** — the owner's ONLY tool. It exposes a single
   ``append_memory`` that appends an immutable record to a :class:`JournalStore`.
   No read/search/write/delete surface, no in-place edit. It validates the
   semantic ``type`` and refuses bad types with ``ModelRetry``.

2. **Labeled owner input** — ``resident_consolidate_text`` now emits each message
   with its ``{run_id}:{turn_id}`` turn address (the SAME convention
   ``search_conversation_history`` labels and ``read_turn`` resolves) and its
   ISO ``[<source_date>]`` when the message metadata carries one. These are the
   exact strings the owner copies into ``append_memory(type, fact, born_on,
   source_ref)``.

3. **Owner capability swap** — ``_build_owner_memory`` with a ``journal`` returns
   a ``Memory`` whose toolset is the append-only ``JournalAppendToolset`` (not the
   legacy 4-tool ``MemoryToolset``); without a ``journal`` the legacy owner
   surface is unchanged.

No model, no network — the tool is driven through the null-tracer fake context
pattern already used by ``test_memory.py``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from paf.capabilities.memory import (
    JournalAppendToolset,
    JournalStore,
    Memory,
    MemoryRecord,
)
from paf.capabilities.memory._consolidate import (
    _strip_inline_date,
    resident_consolidate_text,
)
from paf.capabilities.memory._view import JournalViewStore

# --- null tracer/context (mirrors test_memory.py) -------------------------


class _NullSpan:
    def is_recording(self) -> bool:
        return False

    def set_attributes(self, attributes: dict) -> None:
        pass

    def __enter__(self) -> _NullSpan:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class _NullTracer:
    def start_as_current_span(self, name: str, **kwargs: object) -> _NullSpan:
        return _NullSpan()


class _FakeCtx:
    def __init__(self) -> None:
        self.run_id = "x"
        self.tool_call_id = "t1"
        self.agent = None
        self.tracer = _NullTracer()  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# JournalAppendToolset
# ---------------------------------------------------------------------------


def test_append_toolset_exposes_only_append_memory(tmp_path: Path) -> None:
    """The owner's toolset has exactly one tool: append_memory."""
    ts = JournalAppendToolset(JournalStore(tmp_path))
    assert set(ts.tools.keys()) == {"append_memory"}


def test_append_toolset_appends_to_journal(tmp_path: Path) -> None:
    """append_memory writes one immutable record to the journal."""
    journal = JournalStore(tmp_path)
    ts = JournalAppendToolset(journal)
    result = asyncio.run(
        ts.append_memory(
            _FakeCtx(),
            type="preference",
            fact="user prefers flat-file storage",
            tier="t1",
            born_on="2026-08-17T10:00:00+00:00",
            source_ref="seed-lmem-3f9a:12",
        )
    )
    assert result["status"] == "appended"
    assert result["memory"] == "user prefers flat-file storage"
    assert journal.count() == 1
    recs = journal.read_all()
    assert recs[0][1] == MemoryRecord(
        type="preference",
        fact="user prefers flat-file storage",
        tier="t1",
        born_on="2026-08-17T10:00:00+00:00",
        source_ref="seed-lmem-3f9a:12",
    )


def test_append_toolset_is_append_only(tmp_path: Path) -> None:
    """Two appends of the same fact make two records (dump, no dedup)."""
    journal = JournalStore(tmp_path)
    ts = JournalAppendToolset(journal)
    for _ in range(2):
        asyncio.run(
            ts.append_memory(
                _FakeCtx(),
                type="fact",
                fact="same fact twice",
                tier="t1",
                born_on="2026-08-17T10:00:00+00:00",
                source_ref="seed-lmem-3f9a:12",
            )
        )
    assert journal.count() == 2


def test_append_toolset_rejects_bad_type(tmp_path: Path) -> None:
    """An unknown semantic type raises ModelRetry (the owner is re-prompted)."""
    from pydantic_ai import ModelRetry

    ts = JournalAppendToolset(JournalStore(tmp_path))
    with pytest.raises(ModelRetry):
        asyncio.run(
            ts.append_memory(
                _FakeCtx(),
                type="not-a-type",
                fact="anything",
                tier="t1",
                born_on="2026-08-17T10:00:00+00:00",
                source_ref="seed-lmem-3f9a:12",
            )
        )


def test_capability_journal_swaps_toolset(tmp_path: Path) -> None:
    """Memory with a journal exposes append_memory only; non-journal has no toolset."""
    cap = Memory(
        store=JournalViewStore(tmp_path),
        agent_name="assistant",
        journal=JournalStore(tmp_path),
    )
    ts = cap.get_toolset()
    assert isinstance(ts, JournalAppendToolset)
    assert set(ts.tools.keys()) == {"append_memory"}
    # A non-journal Memory exposes NO toolset (the legacy 4-tool surface is gone).
    assert (
        Memory(store=JournalViewStore(tmp_path), agent_name="a").get_toolset() is None
    )


# ---------------------------------------------------------------------------
# Labeled owner input (resident_consolidate_text)
# ---------------------------------------------------------------------------


def _msg(body: str, role: str, sid: str, dt: str | None):
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        TextPart,
        UserPromptPart,
    )

    run_id = "seed-lmem-abcd1234"
    cid = "lmem-abcd1234"
    metadata = (
        {"source_date": dt, "source_session_id": sid}
        if dt
        else {"source_session_id": sid}
    )
    if role == "u":
        return ModelRequest(
            parts=[UserPromptPart(content=body)],
            run_id=run_id,
            conversation_id=cid,
            metadata=metadata,
        )
    return ModelResponse(
        parts=[TextPart(content=body)],
        run_id=run_id,
        conversation_id=cid,
        metadata=metadata,
    )


def test_resident_labels_turn_ref_and_date() -> None:
    """Each message carries its {run_id}:{turn_id} ref and ISO source date."""
    msgs = [
        _msg("first Q", "u", "sess_1", "2023-05-20T02:21:00+00:00"),
        _msg("first A", "a", "sess_1", "2023-05-20T02:21:00+00:00"),
        _msg("second Q", "u", "sess_1", "2023-05-20T02:21:00+00:00"),
        _msg("new session Q", "u", "sess_2", "2023-05-20T02:57:00+00:00"),
    ]
    out = asyncio.run(resident_consolidate_text(msgs))
    assert out is not None
    lines = out.split("\n")
    assert lines[0].startswith(
        "User seed-lmem-abcd1234:0 [2023-05-20T02:21:00+00:00]: first Q"
    )
    # Assistant inherits its containing user turn (turn 0).
    assert lines[1].startswith(
        "Assistant seed-lmem-abcd1234:0 [2023-05-20T02:21:00+00:00]: first A"
    )
    # Second user message advances to turn 1.
    assert lines[2].startswith(
        "User seed-lmem-abcd1234:1 [2023-05-20T02:21:00+00:00]: second Q"
    )
    # New session gets its own date, turn 2.
    assert lines[3].startswith(
        "User seed-lmem-abcd1234:2 [2023-05-20T02:57:00+00:00]: new session Q"
    )


def test_resident_strips_inline_date_from_user_content() -> None:
    """The owner view drops the first-turn inline [date:] prefix (ISO label covers it)."""
    msgs = [
        _msg(
            "[date: 2023/05/20 (Sat) 02:21] first Q",
            "u",
            "sess_1",
            "2023-05-20T02:21:00+00:00",
        ),
    ]
    out = asyncio.run(resident_consolidate_text(msgs))
    assert out is not None
    assert "[date: 2023/05/20 (Sat) 02:21]" not in out
    assert ": first Q" in out


def test_strip_inline_date() -> None:
    assert _strip_inline_date("[date: 2023/05/20 (Sat) 02:21] hello") == "hello"
    assert _strip_inline_date("plain text") == "plain text"
    assert _strip_inline_date("[date: 2023/05/20]  spaced") == "spaced"


def test_missing_date_survives() -> None:
    """A message with no source_date still labels its turn ref (ref only)."""
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    msgs = [
        ModelRequest(
            parts=[UserPromptPart(content="no date here")],
            run_id="seed-r",
            conversation_id="c",
            metadata={"source_session_id": "sess_1"},
        )
    ]
    out = asyncio.run(resident_consolidate_text(msgs))
    assert out is not None
    assert out.startswith("User seed-r:0:")


def test_iso_date_keeps_time() -> None:
    """LME dates with a weekday keep their hour/minute in ISO form."""
    from tests.run_lmem import _iso_date

    assert _iso_date("2023/05/20 (Sat) 02:21") == "2023-05-20T02:21:00+00:00"
    assert _iso_date("2023/04/10 (Mon) 23:07") == "2023-04-10T23:07:00+00:00"
    assert _iso_date("2023/05/21 (Sun) 10:00") == "2023-05-21T10:00:00+00:00"
    # No date -> None (callers drop the source_date metadata key).
    assert _iso_date("") is None
    assert _iso_date(None) is None
    # Date-only (no time) -> midnight.
    assert _iso_date("2023/05/20") == "2023-05-20T00:00:00+00:00"
