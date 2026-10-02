"""Scoped memory render (memory/_scoped.py) and memory-id provenance."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from paf import conversations
from paf.capabilities._storage import FlatFileStorage
from paf.capabilities.memory import _scoped
from paf.capabilities.memory._journal import JournalStore
from paf.capabilities.memory._overrides import OverrideStore
from paf.capabilities.memory._provenance import memory_turn_refs
from paf.capabilities.memory._scoped import RenderBudget, render_scoped

AGENT = "scoped-test-agent"  # no spec: no profile prefix on conversation ids


class _Run:
    def __init__(self, run_id, cid):
        self.run_id, self.conversation_id = run_id, cid
        self.started_at = datetime.now(UTC)


class _Steps:
    def __init__(self, runs):
        self.runs = runs

    async def list_runs(self):
        return self.runs


@pytest.fixture()
def world(tmp_path: Path, monkeypatch):
    """Two tagged conversations (paf, site), one untagged; a registry."""
    meta = conversations.ConversationMetaStore(
        FlatFileStorage(tmp_path / "conv"), AGENT
    )
    meta.set_project("conv-paf", "paf")
    meta.set_project("conv-site", "site")
    projects = conversations.ProjectStore(FlatFileStorage(tmp_path / "reg"))
    projects.create("paf")
    projects.create("site")
    projects.create("trading")
    monkeypatch.setattr(
        conversations, "default_conversation_meta_store", lambda a: meta
    )
    monkeypatch.setattr(conversations, "default_project_store", lambda: projects)
    monkeypatch.setattr(_scoped, "_RUN_CONV", {})
    monkeypatch.setattr(_scoped, "_RUN_CONV_LOADED", 0.0)
    steps = _Steps(
        [_Run("r-paf", "conv-paf"), _Run("r-site", "conv-site"), _Run("r-x", "conv-x")]
    )
    return tmp_path / "mem", steps


def _add(
    j: JournalStore,
    fact,
    *,
    type="context",
    scope=None,
    project=None,
    tier="t2",
    ref="r-paf:0",
    days_ago=1,
):
    born = (datetime.now(UTC) - timedelta(days=days_ago)).isoformat()
    if scope is None:  # a legacy v1 row
        j.append(type=type, fact=fact, tier=tier, born_on=born, source_ref=ref)
    else:
        j.append(
            type=type,
            fact=fact,
            tier=tier,
            born_on=born,
            source_ref=ref,
            scope=scope,
            project=project,
            source="stated",
        )


def _render(mem: Path, steps, cid: str | None, query: str = "", **budget) -> str:
    return asyncio.run(
        render_scoped(
            str(mem),
            agent_name=AGENT,
            conversation_id=cid,
            query=query,
            budget=RenderBudget(**budget),
            step_store=steps,
        )
    )


def test_sections_and_render_time_filing(world) -> None:
    mem, steps = world
    j = JournalStore(mem)
    _add(
        j, "The user lives in Bentonville.", type="identity", scope="global", tier="t1"
    )  # 1
    _add(j, "paf memory is scoped per project.", scope="project", project="paf")  # 2
    _add(
        j,
        "The site is built with Astro.",
        scope="project",
        project="site",
        ref="r-site:0",
    )  # 3
    _add(
        j, "Legacy paf fact about the extractor.", ref="r-paf:1"
    )  # 4 v1 t2 -> paf by tag
    _add(
        j, "Legacy stable preference for short sentences.", tier="t1", ref="r-x:0"
    )  # 5 v1 t1 -> global
    _add(
        j,
        "Unfiled fact from an untagged chat.",
        scope="project",
        project=None,
        ref="r-x:1",
    )  # 6
    _add(j, "The user ran a 5K race.", type="event", scope="global", tier="t3")  # 7
    _add(
        j,
        "Old event outside the window.",
        type="event",
        scope="global",
        tier="t3",
        days_ago=90,
    )  # 8

    out = _render(mem, steps, "conv-paf", "how is the extractor going")
    about, rest = out.split("## paf", 1)
    assert "Bentonville. — #1" in about and "short sentences. — #5" in about
    assert "scoped per project. — #2" in rest and "extractor. — #4" in rest
    assert "Astro" not in out  # other project: not injected...
    assert "Other projects: site (1 facts)" in out  # ...but named
    assert "Unfiled" not in out
    assert "## Recent history" in out and "5K race. — #7" in out
    assert "Old event" not in out


def test_mentioned_project_and_untagged_conversation(world) -> None:
    mem, steps = world
    j = JournalStore(mem)
    _add(j, "paf memory is scoped per project.", scope="project", project="paf")
    _add(
        j,
        "The site is built with Astro.",
        scope="project",
        project="site",
        ref="r-site:0",
    )
    out = _render(mem, steps, "conv-paf", "does the site still use Astro?")
    assert "## site (mentioned)" in out and "Astro. — #2" in out
    # An untagged conversation gets no project section.
    out = _render(mem, steps, "conv-x", "anything")
    assert "## paf" not in out and "## site" not in out


def test_supersede_renders_as_a_replaced_history_line(world) -> None:
    mem, steps = world
    j = JournalStore(mem)
    _add(j, "The user lives in Tulsa.", type="identity", scope="global", tier="t1")  # 1
    _add(
        j, "The user lives in Bentonville.", type="identity", scope="global", tier="t1"
    )  # 2
    OverrideStore(mem).append(
        op="supersede", target_line=1, by="reviewer", superseded_by=2
    )
    out = _render(mem, steps, "conv-paf")
    about = out.split("## Recent history")[0]
    assert "Tulsa" not in about and "Bentonville. — #2" in about
    assert "replaced: The user lives in Tulsa. — #1 → #2" in out


def test_budget_overflow_leaves_a_pointer(world) -> None:
    mem, steps = world
    j = JournalStore(mem)
    facts = [
        "The paf extractor stamps born_on from the turn date.",
        "Reviewer judges run on DeepSeek with medium reasoning.",
        "Render budgets live in the agent spec memory block.",
        "Journal rows are never rewritten after append.",
        "Override sidecar records archive and supersede verdicts.",
        "Index folds exact repeats into one winner per group.",
        "Watermarks advance only after a completed fold.",
        "History shows events within a thirty day window.",
        "Provenance resolves memory ids to source turns.",
        "Registry slugs are permanent once created.",
    ]
    for f in facts:  # distinct facts (near-duplicates are skipped, not counted)
        _add(j, f, scope="project", project="paf")
    out = _render(mem, steps, "conv-paf", "reviewer", project_tokens=40)
    assert "more paf facts; search to recall" in out


def test_budget_from_spec_config() -> None:
    b = RenderBudget.from_config(
        {"mode": "scoped", "global_tokens": 500, "redundancy": 0.9, "history_days": "x"}
    )
    assert (b.global_tokens, b.redundancy, b.history_days) == (500, 0.9, 30)


def test_memory_turn_refs_follows_merges_and_supersedes(tmp_path: Path) -> None:
    j = JournalStore(tmp_path)
    j.append(type="fact", fact="a", tier="t2", born_on="", source_ref="r1:0")  # 1
    j.append(type="fact", fact="b", tier="t2", born_on="", source_ref="r2:3")  # 2
    j.append(
        type="fact", fact="a+b", tier="t2", born_on="", source_ref="merge(#1,#2)"
    )  # 3
    j.append(type="fact", fact="c", tier="t2", born_on="", source_ref="r3:1")  # 4
    OverrideStore(tmp_path).append(
        op="supersede", target_line=4, by="reviewer", superseded_by=1
    )
    assert [r for _, r in memory_turn_refs(str(tmp_path), 3)] == ["r1:0", "r2:3"]
    assert [r for _, r in memory_turn_refs(str(tmp_path), 4)] == ["r3:1", "r1:0"]
    assert [r for _, r in memory_turn_refs(str(tmp_path), 1)] == ["r1:0", "r3:1"]
    with pytest.raises(KeyError):
        memory_turn_refs(str(tmp_path), 99)


def test_read_turn_memory_id_without_memory_retries() -> None:
    from pydantic_ai import ModelRetry

    from paf.capabilities.search._toolset import ConversationSearchToolset

    ts = ConversationSearchToolset(
        _Steps([]),
        tool_id="t",
        max_matches=5,
        context_lines=2,
        bm25_k1=1.5,
        bm25_b=0.75,
    )

    class Ctx:
        conversation_id = None
        run_id = "x"

    with pytest.raises(ModelRetry):
        asyncio.run(ts.read_turn(Ctx(), "#12"))
