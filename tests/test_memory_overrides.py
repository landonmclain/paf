"""Override sidecar (``_overrides.py``) + its render hook (``_render.py``).

Covers the append-only sidecar, the deterministic fold to per-target override
state, validation, and the one narrow render-hook pass: applied archives drop
the fact, applied tier-changes move it under a new tier, and pending assistant
archives render with a ``*(pending review)*`` marker. All deterministic, no
model.
"""

from __future__ import annotations

from paf.capabilities.memory._index import build_index
from paf.capabilities.memory._journal import JournalStore, MemoryRecord
from paf.capabilities.memory._overrides import (
    OverrideStore,
    fold_overrides,
    load_overrides,
)
from paf.capabilities.memory._render import render_index_memory, write_render

_SEP = " — "


def _mk_record(fact, *, tier="t1", born="2026-08-17T00:00:00Z", src="run:0", typ="fact"):
    return MemoryRecord(type=typ, fact=fact, tier=tier, born_on=born, source_ref=src)


def _seed_store(tmp_path, records) -> JournalStore:
    store = JournalStore(tmp_path)
    for r in records:
        store.append_record(r)
    return store


class _NullSpan:
    def is_recording(self) -> bool:
        return False

    def set_attributes(self, attrs: dict) -> None:  # pragma: no cover
        pass

    def __enter__(self) -> _NullSpan:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class _NullTracer:
    def start_as_current_span(self, name: str, **kwargs: object) -> _NullSpan:
        return _NullSpan()


class _FakeCtx:
    """Minimal RunContext stand-in: the override tool only uses ``ctx.tracer``."""

    def __init__(self) -> None:
        self.run_id = "x"
        self.tool_call_id = "t1"
        self.agent = None
        self.tracer = _NullTracer()  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# OverrideStore.append validation + append-only file
# ---------------------------------------------------------------------------


def test_assistant_cannot_tier_change(tmp_path):
    s = OverrideStore(tmp_path)
    try:
        s.append(op="tier_change", target_line=5, by="assistant", tier="t3")
        raise AssertionError("assistant tier_change should be rejected")
    except ValueError:
        pass


def test_bad_writer_rejected(tmp_path):
    s = OverrideStore(tmp_path)
    try:
        s.append(op="archive", target_line=5, by="hacker")
        raise AssertionError()
    except ValueError:
        pass


def test_tier_change_requires_tier(tmp_path):
    s = OverrideStore(tmp_path)
    try:
        s.append(op="tier_change", target_line=5, by="reviewer")
        raise AssertionError()
    except ValueError:
        pass


def test_target_line_must_be_positive(tmp_path):
    s = OverrideStore(tmp_path)
    try:
        s.append(op="archive", target_line=0, by="assistant")
        raise AssertionError()
    except ValueError:
        pass


def test_append_is_append_only(tmp_path):
    s = OverrideStore(tmp_path)
    s.append(op="archive", target_line=1, by="assistant")
    s.append(op="archive", target_line=2, by="assistant")
    recs = s.read_all()
    assert len(recs) == 2
    assert recs[0]["target_line"] == 1
    assert recs[1]["target_line"] == 2
    assert all(r["by"] == "assistant" for r in recs)


# ---------------------------------------------------------------------------
# fold: raw log -> derived per-target override state
# ---------------------------------------------------------------------------


def test_fold_assistant_archive_is_pending(tmp_path):
    s = OverrideStore(tmp_path)
    s.append(op="archive", target_line=113, by="assistant", reason="rm-gate disproven")
    state = fold_overrides(s.read_all())
    assert state[113].is_pending
    assert state[113].by == "assistant"
    assert state[113].reason == "rm-gate disproven"


def test_fold_reviewer_archive_applies(tmp_path):
    s = OverrideStore(tmp_path)
    s.append(op="archive", target_line=113, by="assistant")  # pending first
    s.append(op="archive", target_line=113, by="reviewer")   # reviewer resolves -> applied
    state = fold_overrides(s.read_all())
    assert state[113].is_applied_archive
    assert not state[113].is_pending


def test_fold_reviewer_tier_change_applies(tmp_path):
    s = OverrideStore(tmp_path)
    s.append(op="tier_change", target_line=121, by="reviewer", tier="t3")
    state = fold_overrides(s.read_all())
    assert state[121].is_applied_tier
    assert state[121].tier == "t3"


def test_fold_reviewer_reject_defeats_pending(tmp_path):
    s = OverrideStore(tmp_path)
    s.append(op="archive", target_line=121, by="assistant")  # pending
    s.append(op="reject", target_line=121, by="reviewer")    # reviewer vetoes
    state = fold_overrides(s.read_all())
    assert state[121].status == "rejected"
    assert not state[121].is_pending
    assert not state[121].is_applied_archive


# ---------------------------------------------------------------------------
# render hook — applied archive drops the fact
# ---------------------------------------------------------------------------

def test_render_applied_archive_drops_fact(tmp_path):
    store = _seed_store(tmp_path, [
        _mk_record("User prefers flat-file storage", src="run:0"),
        _mk_record("User's site has four arcs", src="run:1"),
    ])
    build_index(tmp_path)
    from paf.capabilities.memory._index import MemoryIndex

    idx = MemoryIndex.load(tmp_path)
    # reviewer archives line 2 ("four arcs")
    s = OverrideStore(tmp_path)
    s.append(op="archive", target_line=2, by="reviewer")
    ov = load_overrides(tmp_path)
    md = render_index_memory(store, idx, overrides=ov)
    lines = [ln for ln in md.splitlines() if ln.startswith("- ")]
    assert len(lines) == 1
    assert "flat-file" in lines[0]
    assert "four arcs" not in "".join(lines)


# ---------------------------------------------------------------------------
# render hook — pending marker
# ---------------------------------------------------------------------------


def test_render_pending_archive_marks_line(tmp_path):
    store = _seed_store(tmp_path, [
        _mk_record("User prefers flat-file storage", src="run:0"),
    ])
    build_index(tmp_path)
    from paf.capabilities.memory._index import MemoryIndex
    idx = MemoryIndex.load(tmp_path)
    s = OverrideStore(tmp_path)
    s.append(op="archive", target_line=1, by="assistant", reason="rm-gate disproven")
    ov = load_overrides(tmp_path)
    md = render_index_memory(store, idx, overrides=ov)
    lines = [ln for ln in md.splitlines() if ln.startswith("- ")]
    assert len(lines) == 1
    assert "*(pending review)*" in lines[0]
    # the fact is NOT dropped — just flagged
    assert "flat-file storage" in lines[0]


# ---------------------------------------------------------------------------
# render hook — reject clears the marker
# ---------------------------------------------------------------------------


def test_render_reject_clears_pending_marker(tmp_path):
    store = _seed_store(tmp_path, [
        _mk_record("User prefers flat-file storage", src="run:0"),
    ])
    build_index(tmp_path)
    from paf.capabilities.memory._index import MemoryIndex

    idx = MemoryIndex.load(tmp_path)
    s = OverrideStore(tmp_path)
    s.append(op="archive", target_line=1, by="assistant")  # pending
    s.append(op="reject", target_line=1, by="reviewer")    # veto
    ov = load_overrides(tmp_path)
    md = render_index_memory(store, idx, overrides=ov)
    lines = [ln for ln in md.splitlines() if ln.startswith("- ")]
    assert len(lines) == 1
    assert "*(pending review)*" not in "".join(lines)  # marker cleared
    assert "flat-file storage" in lines[0]  # fact stays


# ---------------------------------------------------------------------------
# render hook — tier_change moves sort position
# ---------------------------------------------------------------------------


def test_render_tier_change_moves_fact_to_lower_section(tmp_path):
    store = _seed_store(tmp_path, [
        _mk_record("Stable identity fact", tier="t1", born="2026-01-01T00:00:00Z", src="run:0"),
        _mk_record("Ephemeral goal detail", tier="t3", born="2026-02-01T00:00:00Z", src="run:1"),
    ])
    build_index(tmp_path)
    from paf.capabilities.memory._index import MemoryIndex

    idx = MemoryIndex.load(tmp_path)
    s = OverrideStore(tmp_path)
    # demote the t1 fact to t3 (it "aged out" of identity)
    s.append(op="tier_change", target_line=1, by="reviewer", tier="t3")
    ov = load_overrides(tmp_path)
    md = render_index_memory(store, idx, overrides=ov)
    # both facts now under t3: the demoted one joined it
    assert "## t1" not in md  # t1 section is gone (its only fact was demoted)
    t3_section = md.split("## t3", 1)[1]
    assert "Stable identity fact" in t3_section
    assert "Ephemeral goal detail" in t3_section


# ---------------------------------------------------------------------------
# end-to-end: write_render honors a sidecar beside MEMORY.md
# ---------------------------------------------------------------------------


def test_write_render_end_to_end_applied_archive(tmp_path):
    store = _seed_store(tmp_path, [
        _mk_record("User prefers flat-file storage", src="run:0"),
        _mk_record("User's site has four arcs", src="run:1"),
    ])
    build_index(tmp_path)
    s = OverrideStore(tmp_path)
    s.append(op="archive", target_line=2, by="reviewer")
    md_path = tmp_path / "MEMORY.md"
    write_render(store, str(md_path))
    text = md_path.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln.startswith("- ")]
    assert len(lines) == 1
    assert "flat-file" in lines[0]
    assert "four arcs" not in "".join(lines)


def test_write_render_end_to_end_pending_marker(tmp_path):
    store = _seed_store(tmp_path, [
        _mk_record("User prefers flat-file storage", src="run:0"),
    ])
    build_index(tmp_path)
    s = OverrideStore(tmp_path)
    s.append(op="archive", target_line=1, by="assistant")
    md_path = tmp_path / "MEMORY.md"
    write_render(store, str(md_path))
    text = md_path.read_text(encoding="utf-8")
    assert "*(pending review)*" in text
    assert "flat-file storage" in text


# ---------------------------------------------------------------------------
# the assistant tool: memory_archive_suggestion
# ---------------------------------------------------------------------------


def test_override_toolset_exposes_one_tool(tmp_path):
    from paf.capabilities.memory._toolset import OverrideToolset

    ts = OverrideToolset(OverrideStore(tmp_path))
    assert set(ts.tools.keys()) == {"memory_archive_suggestion"}


def test_override_tool_writes_pending_archive(tmp_path):
    import asyncio

    from paf.capabilities.memory._toolset import OverrideToolset

    overrides = OverrideStore(tmp_path)
    ts = OverrideToolset(overrides)

    result = asyncio.run(
        ts.memory_archive_suggestion(_FakeCtx(), journal_id=42, reason="rm disproven")
    )
    assert result["status"] == "pending_review"
    assert result["by"] == "assistant"
    assert result["journal_id"] == 42
    # the sidecar now holds one pending archive intent
    st = fold_overrides(overrides.read_all())[42]
    assert st.is_pending
    assert st.reason == "rm disproven"
    assert st.by == "assistant"


def test_memory_capability_wires_override_toolset(tmp_path):
    """The assistant's Memory carries the override toolset when `overrides` set."""
    from paf.capabilities.memory import Memory, OverrideStore

    m = Memory(
        store=object(),  # type: ignore[arg-type]  (injection tests elsewhere)
        agent_name="x",
        overrides=OverrideStore(tmp_path),
    )
    ts = m.get_toolset()
    assert ts is not None
    assert "memory_archive_suggestion" in set(ts.tools.keys())
    assert "append_memory" not in set(ts.tools.keys())  # never the owner's tool

