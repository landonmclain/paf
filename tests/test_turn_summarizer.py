"""Tests for the real-LLM turn summarizer (`_summarizer.py`).

The summarizer is the piece that turns the one-line index from a commodity into
something searchable: a REAL pydantic-ai structured-output call producing a
`TurnSummary` record (sentence + key_terms + outcome), with a CHAR-BUDGET retry
loop (the axis pydantic-ai's builtin `retries=` can't cover -- it resends the
*same* prompt, which can't shorten output).

These tests pin the LOGIC without a live model: the `_summarizer.Agent.run`
seam is monkeypatched so we verify (a) the structured contract, (b) the
char-budget shorten loop drives the sentence under budget, (c) `key_terms` are
clamped, and (d) the full WRITE path (capability hook -> summarizer ->
`summarize_turn_append` -> store persists structured fields -> `search_index`
matches key_terms).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from paf.capabilities.turn_index._generate import Turn, summarize_turn_append
from paf.capabilities.turn_index._store import TurnIndexStore
from paf.capabilities.turn_index._summarizer import TurnSummarizer, TurnSummary


def _turn() -> Turn:
    return Turn(
        run_id="r1",
        conversation_id="c1",
        turn_id=0,
        messages=[
            ModelRequest(parts=[UserPromptPart("migrate the user auth store")]),
            ModelResponse(parts=[TextPart("Switched to Postgres with row-level security.")]),
        ],
    )


def _fake_run(sequence: list[TurnSummary]):
    """Return a monkeypatch-able `run` that yields `sequence` summaries in order.

    Each call pops the next summary, wrapping it in an AgentRunResult-shaped
    object exposing `.output`. The `_shorten` path ALSO calls `run`, so a
    sequence like [long, short] exercises exactly one shorten iteration.
    """

    async def run(self, *args, **kwargs) -> SimpleNamespace:
        item = sequence.pop(0)
        return SimpleNamespace(output=item)

    return run


def test_structured_schema_fields_exist() -> None:
    """The `TurnSummary` record carries sentence + key_terms + outcome."""
    s = TurnSummary(
        sentence="Migrated user auth to Postgres with RLS.",
        key_terms=["postgres", "row-level security", "auth"],
        outcome="resolved",
    )
    assert s.sentence
    assert s.key_terms == ["postgres", "row-level security", "auth"]
    assert s.outcome == "resolved"


def test_summarize_runs_and_builds_turn_summary(monkeypatch) -> None:
    """A single in-budget model result round-trips through `summarize`."""
    done = TurnSummary(
        sentence="Migrated user auth to Postgres with RLS.",
        key_terms=["postgres", "RLS", "auth"],
        outcome="resolved",
    )
    s = TurnSummarizer("openrouter:deepseek/deepseek-v4-flash-0731:nitro")
    monkeypatch.setattr(s._agent, "run", _fake_run([done]))
    out = asyncio.run(s.summarize("context text"))
    assert out.sentence == done.sentence
    assert out.key_terms == done.key_terms
    assert out.outcome == done.outcome


def test_summarize_shortens_over_budget_sentence(monkeypatch) -> None:
    """Over-budget `sentence` triggers the shorten loop until under the cap.

    `_shorten` builds a fresh `Agent` and calls its `.run`, so we monkeypatch
    the class-level Agent factory seam by patching the instance used in the base
    `_agent.run` AND the `_shorten` path. Simpler: drive it through the real
    sequence via the instance `_agent.run` (base call) and reach `_shorten` by
    making the first result over-budget; `_shorten` uses a NEW agent we also
    must stub. We monkeypatch `Agent` construction is too deep -- instead we
    stub both the base and the shorten result by patching the helper.
    """
    over = TurnSummary(
        sentence="x" * 300,  # massively over the 160 default budget
        key_terms=["postgres"],
        outcome="resolved",
    )
    under = TurnSummary(
        sentence="Migrated auth to Postgres.",
        key_terms=["postgres"],
        outcome="resolved",
    )
    s = TurnSummarizer("openrouter:deepseek/deepseek-v4-flash-0731:nitro", budget_chars=160)
    # Base `_agent.run` returns the over-budget one; `_shorten` creates a fresh
    # agent -- we stub the module-level `_extract_output`+agent path by patching
    # `s._agent.run` for BOTH calls is not possible (shorten's agent is new).
    # Instead: replace `s._shorten` with a trivial coroutine returning `under`.
    async def fake_shorten(previous: TurnSummary) -> TurnSummary:
        assert len(previous.sentence) > 160  # shorten only called when over cap
        return under

    monkeypatch.setattr(s._agent, "run", _fake_run([over]))
    monkeypatch.setattr(s, "_shorten", fake_shorten)
    out = asyncio.run(s.summarize("context"))
    assert out.sentence == under.sentence
    assert len(out.sentence) <= 160


def test_key_terms_clamped_to_max(monkeypatch) -> None:
    """`key_terms` beyond the cap are truncated so the record stays cheap."""
    many = TurnSummary(
        sentence="short sentence",
        key_terms=[f"term{i}" for i in range(30)],
        outcome="decided",
    )
    s = TurnSummarizer("openrouter:deepseek/deepseek-v4-flash-0731:nitro")
    monkeypatch.setattr(s._agent, "run", _fake_run([many]))
    out = asyncio.run(s.summarize("ctx"))
    assert len(out.key_terms) <= s.max_key_terms


def test_write_path_persists_structured_fields(tmp_path: Path) -> None:
    """Capability seam: a `TurnSummary` written through `summarize_turn_append`
    persists `key_terms` + `outcome` into the store, and `search_index` matches
    a term that is ONLY in `key_terms` (not the sentence)."""
    index = TurnIndexStore(tmp_path / "index")
    turn = _turn()
    summary = TurnSummary(
        sentence="Migrated the database.",
        key_terms=["postgres", "row-level security"],
        outcome="resolved",
    )
    appended, _ = asyncio.run(summarize_turn_append(turn=turn, index_store=index, summary=summary))
    assert appended is True
    entries = index.read_all("c1")
    assert len(entries) == 1
    e = entries[0]
    assert e.summary == "Migrated the database."
    assert e.key_terms == ["postgres", "row-level security"]
    assert e.outcome == "resolved"
    # The sentence says "the database"; the searchable vocab says "postgres".
    assert "postgres" not in e.summary.lower()


def test_search_index_matches_key_terms(tmp_path: Path) -> None:
    """`search_index` finds a turn via a term that appears ONLY in `key_terms`,
    not in the one-liner -- the whole point of splitting the record."""
    from paf.capabilities.turn_index._toolset import TurnIndexToolset

    index = TurnIndexStore(tmp_path / "index")
    asyncio.run(
        summarize_turn_append(
            turn=_turn(),
            index_store=index,
            summary=TurnSummary(
                sentence="Migrated the database.",
                key_terms=["postgres", "row-level security"],
                outcome="resolved",
            ),
        )
    )
    ts = TurnIndexToolset(index_store=index, source=None)  # search_index doesn't touch source

    class _Ctx:
        conversation_id = "c1"

    hits = asyncio.run(ts.search_index(_Ctx(), "postgres"))
    assert "r1:0" in hits
    hits_neg = asyncio.run(ts.search_index(_Ctx(), "sqlite"))
    assert "No index entries" in hits_neg