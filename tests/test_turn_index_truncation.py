"""Unit tests for the turn-index truncation notice.

When the injected one-line index exceeds `max_lines`, `_render_index` trims the
tail to the most recent `max_lines` and appends an explicit notice telling the
model that older turn indexes exist and are reachable via `search_index`. This
prevents the model from assuming the injected list is the complete history.

Runs ungated (no model / no API key — pure store + render).
"""

from __future__ import annotations

from pathlib import Path

from paf.capabilities.turn_index import TurnIndexCapability

CONV = "conv-trunc"


def _seed(cap: TurnIndexCapability, n: int) -> list[str]:
    """Append `n` durable index entries; return their refs (oldest-first)."""
    refs = []
    for i in range(n):
        refs.append(cap.index_store.append(
            conversation_id=CONV,
            run_id="run-test",
            turn_id=i,
            summary=f"turn {i} summary",
        ).ref)
    return refs


def test_render_index_omits_notice_when_within_max_lines(tmp_path: Path) -> None:
    cap = TurnIndexCapability.from_spec(tmp_path, max_lines=10)
    _seed(cap, 5)
    rendered = cap._render_index(CONV)
    assert "not shown" not in rendered
    assert "search_index" not in rendered
    # all 5 lines present
    assert rendered.count("- [run-test:") == 5


def test_render_index_adds_notice_when_truncated(tmp_path: Path) -> None:
    cap = TurnIndexCapability.from_spec(tmp_path, max_lines=10)
    refs = _seed(cap, 24)
    rendered = cap._render_index(CONV)
    # Only the tail slide is injected, with the older-turn notice appended.
    assert "not shown" in rendered
    assert "search_index" in rendered
    # 10 shown lines + 1 notice line
    assert rendered.count("- [run-test:") == 10
    # Explicit count of hidden older turns (24 total - 10 shown = 14)
    assert "14 older turns not shown" in rendered
    # The most recent turns are the ones injected (tail), not the oldest.
    assert refs[-1] in rendered
    assert refs[0] not in rendered


def test_render_index_notice_pluralization(tmp_path: Path) -> None:
    cap = TurnIndexCapability.from_spec(tmp_path, max_lines=3)
    _seed(cap, 4)  # exactly one hidden turn
    rendered = cap._render_index(CONV)
    assert "1 older turn not shown" in rendered


def test_render_index_unbounded_never_notices(tmp_path: Path) -> None:
    cap = TurnIndexCapability.from_spec(tmp_path, max_lines=0)  # 0 => no bound
    _seed(cap, 30)
    rendered = cap._render_index(CONV)
    assert "not shown" not in rendered
    assert rendered.count("- [run-test:") == 30
