"""Pairwise memory reviewer (memory/_pairwise.py) with a scripted judge."""

from __future__ import annotations

import asyncio
from pathlib import Path

from paf.capabilities.memory._journal import JournalStore
from paf.capabilities.memory._overrides import OverrideStore, load_overrides
from paf.capabilities.memory._pairwise import (
    PairJudgement,
    SuggestionVerdict,
    current_facts,
    load_watermark,
    review_new_facts,
    save_watermark,
)


def _journal(tmp_path: Path, facts: list[tuple[str, str | None]]) -> str:
    """Append v2 facts ``(text, project)``; returns the memory dir."""
    j = JournalStore(tmp_path)
    for n, (text, project) in enumerate(facts, start=1):
        j.append(
            type="context",
            fact=text,
            tier="t2" if project else "t1",
            born_on=f"2026-09-{n:02d}T12:00:00+00:00",
            source_ref=f"r:{n}",
            scope="project" if project else "global",
            project=project,
            source="stated",
        )
    return str(tmp_path)


class ScriptedJudge:
    """Labels pairs from a {(new_line, old_line): (label, keep)} script; anything
    unscripted is `different`. Records what it was shown."""

    def __init__(self, script: dict[tuple[int, int], tuple[str, str | None]]):
        self.script = script
        self.seen: dict[int, list[int]] = {}

    async def __call__(self, new, older):
        self.seen[new.line] = [f.line for f in older]
        out = []
        for i, f in enumerate(older, start=1):
            label, keep = self.script.get((new.line, f.line), ("different", None))
            out.append(
                PairJudgement(index=i, reason="scripted", label=label, keep=keep)
            )
        return out


FACTS = [
    ("The user lives in Tulsa.", None),  # 1
    ("The user prefers dark roast coffee.", None),  # 2
    ("The paf test suite takes about a minute.", "paf"),  # 3
    ("The site test suite takes about a minute.", "site"),  # 4
    ("The user lives in Bentonville.", None),  # 5  update of 1
    ("The user prefers dark roast coffee, black.", None),  # 6  covers 2
    ("The paf test suite runs in GitHub Actions for every push.", "paf"),  # 7
]


def _run(d: str, judge, **kw):
    return asyncio.run(review_new_facts(d, judge=judge, judge_name="scripted", **kw))


def test_update_supersedes_and_same_archives_the_covered_fact(tmp_path: Path) -> None:
    d = _journal(tmp_path, FACTS)
    save_watermark(d, 4)
    judge = ScriptedJudge({(5, 1): ("update", None), (6, 2): ("same", "new")})
    report = _run(d, judge)

    st = load_overrides(d)
    assert st[1].is_superseded and st[1].superseded_by == 5
    assert st[2].is_applied_archive
    assert set(current_facts(d)) == {3, 4, 5, 6, 7}
    assert load_watermark(d) == 7
    assert {(a.op, a.target, a.other) for a in report.actions} == {
        ("supersede", 1, 5),
        ("archive", 2, 6),
    }
    # Every action carries its audit detail.
    rec = [r for r in OverrideStore(d).read_all() if r["target_line"] == 1][-1]
    assert rec["detail"] == {
        "reviewer": "pairwise",
        "label": "update",
        "other": 5,
        "judge": "scripted",
    }


def test_same_keep_old_archives_the_new_fact(tmp_path: Path) -> None:
    d = _journal(tmp_path, FACTS)
    save_watermark(d, 5)
    _run(d, ScriptedJudge({(6, 2): ("same", "old")}))
    st = load_overrides(d)
    assert st[6].is_applied_archive and 2 not in st


def test_candidates_stay_within_project_plus_global(tmp_path: Path) -> None:
    d = _journal(tmp_path, FACTS)
    save_watermark(d, 6)
    judge = ScriptedJudge({})
    _run(d, judge)
    shown = judge.seen[7]
    assert 3 in shown  # same project
    assert 4 not in shown  # the site's test suite is never a paf candidate


def test_dry_run_proposes_but_writes_nothing(tmp_path: Path) -> None:
    d = _journal(tmp_path, FACTS)
    report = _run(
        d, ScriptedJudge({(5, 1): ("update", None)}), dry_run=True, start_after=4
    )
    assert [(a.op, a.target) for a in report.actions] == [("supersede", 1)]
    assert OverrideStore(d).read_all() == []
    assert load_watermark(d) is None


def test_first_run_starts_from_the_tail(tmp_path: Path) -> None:
    """No watermark yet: old facts are a separate backfill, so the reviewer only
    records the tail and judges nothing."""
    d = _journal(tmp_path, FACTS)
    judge = ScriptedJudge({(5, 1): ("update", None)})
    report = _run(d, judge)
    assert judge.seen == {} and report.actions == []
    assert load_watermark(d) == len(FACTS)


def test_a_failing_judge_call_skips_only_that_fact(tmp_path: Path) -> None:
    d = _journal(tmp_path, FACTS)
    save_watermark(d, 4)
    inner = ScriptedJudge({(6, 2): ("same", "new")})

    async def flaky(new, older):
        if new.line == 5:
            raise RuntimeError("provider down")
        return await inner(new, older)

    _run(d, flaky)
    st = load_overrides(d)
    assert 1 not in st  # #5 was skipped
    assert st[2].is_applied_archive  # #6 still reviewed
    assert load_watermark(d) == 7


def test_pending_suggestions_are_approved_or_rejected(tmp_path: Path) -> None:
    d = _journal(tmp_path, FACTS)
    save_watermark(d, len(FACTS))
    store = OverrideStore(d)
    store.append(op="archive", target_line=1, by="assistant", reason="moved")
    store.append(op="archive", target_line=2, by="assistant", reason="old")

    async def sugg(fact, reason, context):
        if fact.line == 1:
            return SuggestionVerdict(
                reason="newer fact says Bentonville", verdict="approve"
            )
        return SuggestionVerdict(reason="still true", verdict="reject")

    report = _run(d, ScriptedJudge({}), suggestion_judge=sugg)
    st = load_overrides(d)
    assert st[1].is_applied_archive
    assert st[2].status == "rejected"
    assert {(a.op, a.target) for a in report.actions} == {("approve", 1), ("reject", 2)}


def test_worker_passes_project_and_reviews_what_it_folded(
    tmp_path: Path, monkeypatch
) -> None:
    """Production path: the coordinator's 3-arg fold reads the job's project off
    the queue, and with pairwise_review on, the facts that fold wrote are
    reviewed — even on the first-ever run (no reviewer watermark yet)."""
    from paf.capabilities.memory import _pairwise
    from paf.capabilities.memory._background import (
        ConsolidationJob,
        MemoryConsolidationQueue,
    )

    d = _journal(tmp_path, FACTS[:4])
    seen_projects: list[str | None] = []
    ref: dict = {}

    async def fold(messages, directory, watermark) -> int:
        seen_projects.append(ref["q"].current_project)
        JournalStore(directory).append(
            type="identity",
            fact="The user lives in Bentonville.",
            tier="t1",
            born_on="2026-09-30T12:00:00+00:00",
            source_ref="r:9",
            scope="global",
            source="stated",
        )
        return 1

    monkeypatch.setattr(
        _pairwise, "deepseek_judge", ScriptedJudge({(5, 1): ("update", None)})
    )
    q = MemoryConsolidationQueue(
        memory_directory=d,
        agent_name="pairwise-test",
        consolidate_span=fold,
        pairwise_review=True,
    )
    ref["q"] = q

    async def scenario():  # enqueue starts a worker task: needs a running loop
        q.enqueue(
            "pairwise-test",
            ConsolidationJob(run_id="r", turn_id=0, messages=[], project="paf"),
        )
        await q.drain("pairwise-test")

    asyncio.run(scenario())

    assert seen_projects == ["paf"]
    st = load_overrides(d)
    assert st[1].is_superseded and st[1].superseded_by == 5
    assert load_watermark(d) == 5


# --- status reporting: the job's review phase and outcome ----------------------


def _fold_bentonville(directory, *_args) -> int:
    JournalStore(directory).append(
        type="identity",
        fact="The user lives in Bentonville.",
        tier="t1",
        born_on="2026-09-30T12:00:00+00:00",
        source_ref="r:9",
        scope="global",
        source="stated",
    )
    return 1


def _run_job(d: str, **hooks) -> list:
    """Run one fold job (writing fact #5) with the pairwise reviewer on."""
    from paf.capabilities.memory._background import (
        ConsolidationJob,
        MemoryConsolidationQueue,
    )

    async def fold(messages, directory, watermark) -> int:
        return _fold_bentonville(directory)

    q = MemoryConsolidationQueue(
        memory_directory=d,
        agent_name="pairwise-test",
        consolidate_span=fold,
        pairwise_review=True,
        **hooks,
    )

    async def scenario():
        q.enqueue("pairwise-test", ConsolidationJob(run_id="r", turn_id=0, messages=[]))
        await q.drain("pairwise-test")

    asyncio.run(scenario())


def test_job_reports_review_phase_then_outcome(tmp_path: Path, monkeypatch) -> None:
    """The row must not finish when the fold returns: the reviewer is still to
    run. The finish hook fires once, after the review, with its result."""
    from paf.capabilities.memory import _pairwise
    from paf.capabilities.memory._pairwise import describe_actions

    d = _journal(tmp_path, FACTS[:4])
    monkeypatch.setattr(
        _pairwise, "deepseek_judge", ScriptedJudge({(5, 1): ("update", None)})
    )
    calls: list = []

    async def on_reviewing() -> None:
        calls.append("reviewing")

    async def on_finished(outcome) -> None:
        calls.append(outcome)

    _run_job(d, on_reviewing=on_reviewing, on_finished=on_finished)

    assert calls[0] == "reviewing" and len(calls) == 2
    outcome = calls[1]
    assert outcome.records == 1 and outcome.error is None
    assert outcome.review.outcome() == "reviewed 1: 1 superseded"
    assert describe_actions(outcome.review, d) == [
        'superseded "The user lives in Tulsa." by "The user lives in Bentonville."'
        " — scripted"
    ]


def test_job_finishes_row_when_review_fails(tmp_path: Path, monkeypatch) -> None:
    from paf.capabilities.memory import _pairwise

    async def broken(*_a, **_k):
        raise RuntimeError("judge offline")

    monkeypatch.setattr(_pairwise, "review_new_facts", broken)
    finished: list = []

    async def on_finished(outcome) -> None:
        finished.append(outcome)

    _run_job(_journal(tmp_path, FACTS[:4]), on_finished=on_finished)

    assert [o.review_error for o in finished] == ["judge offline"]
    assert finished[0].records == 1 and finished[0].review is None


def test_job_finishes_row_when_post_fold_step_fails(
    tmp_path: Path, monkeypatch
) -> None:
    """A failure between the fold and the review must still close the row."""
    from paf.capabilities.memory import _render

    def broken(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(_render, "write_render", broken)
    finished: list = []

    async def on_finished(outcome) -> None:
        finished.append(outcome)

    _run_job(_journal(tmp_path, FACTS[:4]), on_finished=on_finished)

    assert [o.error for o in finished] == ["disk full"]


def test_describe_actions_quotes_facts_and_reasons(tmp_path: Path) -> None:
    from paf.capabilities.memory._pairwise import (
        Action,
        ReviewReport,
        describe_actions,
    )

    d = _journal(tmp_path, FACTS)
    report = ReviewReport(
        reviewed=[6],
        actions=[
            Action("archive", 2, 6, "same", "covered by #6: new adds 'black'"),
            Action("approve", 3, None, "approve", "stale"),
        ],
    )
    assert report.outcome() == "reviewed 1: 1 archived, 1 suggestion(s) approved"
    assert describe_actions(report, d) == [
        'archived "The user prefers dark roast coffee." (same as '
        "\"The user prefers dark roast coffee, black.\") — new adds 'black'",
        'archived "The paf test suite takes about a minute." (approved suggestion)'
        " — stale",
    ]
    assert ReviewReport(reviewed=[1, 2]).outcome() == "reviewed 2: no changes"
