"""Legacy re-filing: the ``reclassify`` override and the backfill pass."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from paf import conversations
from paf.capabilities._storage import FlatFileStorage
from paf.capabilities.memory import _backfill, _scoped
from paf.capabilities.memory._backfill import FactLabel, backfill
from paf.capabilities.memory._journal import JournalStore
from paf.capabilities.memory._overrides import OverrideStore, load_overrides
from paf.capabilities.memory._pairwise import current_facts


def test_reclassify_override_refiles_without_changing_status(tmp_path: Path) -> None:
    j = JournalStore(tmp_path)
    j.append(type="preference", fact="In paf, use the codebase tools.", tier="t1",
             born_on="2026-09-01", source_ref="r:0")  # fmt: skip
    j.append(type="fact", fact="Old status snapshot.", tier="t2",
             born_on="2026-09-01", source_ref="r:1")  # fmt: skip
    store = OverrideStore(tmp_path)
    fields = {"type": "rule", "scope": "project", "project": "paf"}
    store.append(op="reclassify", target_line=1, by="reviewer", fields=fields)
    store.append(op="reclassify", target_line=2, by="reviewer",
                 fields={"type": "context", "scope": "global", "project": None})  # fmt: skip
    store.append(op="archive", target_line=2, by="reviewer")

    st = load_overrides(tmp_path)
    assert st[1].status == "none" and st[1].fields == fields
    assert st[2].is_applied_archive and st[2].fields["scope"] == "global"

    facts = current_facts(str(tmp_path))
    assert list(facts) == [1]
    rec = facts[1]
    assert (rec.type, rec.scope, rec.project, rec.tier) == (
        "rule",
        "project",
        "paf",
        "t2",
    )
    assert rec.fact == "In paf, use the codebase tools." and rec.born_on

    with pytest.raises(ValueError):
        store.append(op="reclassify", target_line=1, by="reviewer")
    with pytest.raises(ValueError):
        store.append(op="archive", target_line=1, by="reviewer", fields=fields)


@pytest.fixture()
def world(tmp_path: Path, monkeypatch):
    agent = "backfill-test-agent"
    meta = conversations.ConversationMetaStore(FlatFileStorage(tmp_path / "c"), agent)
    meta.set_project("conv-paf", "paf")
    projects = conversations.ProjectStore(FlatFileStorage(tmp_path / "reg"))
    projects.create("paf")
    projects.create("site")
    monkeypatch.setattr(
        conversations, "default_conversation_meta_store", lambda a: meta
    )
    monkeypatch.setattr(conversations, "default_project_store", lambda: projects)
    monkeypatch.setattr(_scoped, "_RUN_CONV", {"r-paf": "conv-paf"})
    import time

    monkeypatch.setattr(_scoped, "_RUN_CONV_LOADED", time.monotonic())
    mem = tmp_path / "mem"
    j = JournalStore(mem)
    for i, fact in enumerate(
        [
            "The user lives in Bentonville.",
            "Committed the fix as abc123.",
            "The site is built with Astro.",
            "In paf, defaults live on the capability.",
        ]
    ):
        j.append(type="fact", fact=fact, tier="t1", born_on="2026-09-01",
                 source_ref=f"r-paf:{i}")  # fmt: skip
    j.append(type="identity", fact="Already v2.", tier="t1", born_on="2026-09-01",
             source_ref="r-paf:9", scope="global", source="stated")  # fmt: skip
    return mem, agent


def _labels(batch, projects):
    seen.append(([f.line for f in batch], [f.hint for f in batch], projects))
    out = {
        1: FactLabel(
            id=1, reason="identity", keep=True, type="identity", scope="global"
        ),
        2: FactLabel(id=2, reason="work log", keep=False),
        3: FactLabel(
            id=3,
            reason="site",
            keep=True,
            type="context",
            scope="project",
            project="site",
        ),
        4: FactLabel(
            id=4,
            reason="paf rule",
            keep=True,
            type="rule",
            scope="project",
            project="nope",
        ),
    }
    return [out[f.line] for f in batch if f.line in out]


seen: list = []


def test_backfill_dry_run_then_apply(world) -> None:
    mem, agent = world
    seen.clear()

    async def labeler(batch, projects):
        return _labels(batch, projects)

    report = asyncio.run(
        backfill(str(mem), agent_name=agent, labeler=labeler, batch_size=3)
    )
    assert report.dry_run and [d.fact.line for d in report.decisions] == [1, 2, 3, 4]
    assert seen[0][0] == [1, 2, 3] and seen[0][1] == ["paf"] * 3  # v2 row #5 skipped
    assert {p["slug"] for p in seen[0][2]} == {"paf", "site"}
    assert load_overrides(mem) == {}  # dry run wrote nothing

    report = asyncio.run(
        backfill(str(mem), agent_name=agent, labeler=labeler, dry_run=False)
    )
    assert report.remaining == 0
    facts = current_facts(str(mem))
    assert 2 not in facts
    assert (facts[1].scope, facts[1].tier) == ("global", "t1")
    assert (facts[3].scope, facts[3].project) == ("project", "site")
    assert (facts[4].type, facts[4].project) == ("rule", None)  # unknown slug: unfiled
    # Resumable: nothing legacy left.
    assert asyncio.run(_backfill.legacy_facts(str(mem), agent_name=agent)) == []
