"""Backfill — re-file legacy (v1) memory facts under the v2 schema.

v1 rows have no ``scope``; the scoped render files them by rules (``t1`` →
About you, else the source conversation's project), which puts project rules in
About you and keeps noise the v2 extractor would have skipped. This pass asks a
labeler (spec ``memory_backfill``) about each CURRENT legacy fact, in batches:

- ``keep=false`` → ``archive`` (by reviewer, ``detail.reviewer = "backfill"``)
- ``keep=true``  → ``reclassify`` with ``type`` / ``scope`` / ``project``

Both are override-sidecar entries: the journal is never rewritten and every
entry is reversible (``reject`` an archive; a later ``reclassify`` wins).
Resumable: facts already reclassified or archived are not current-legacy any
more, so a re-run picks up where the last one stopped.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from ._journal import JournalStore, MemoryRecord
from ._overrides import OverrideStore, load_overrides

logger = logging.getLogger(__name__)

LABELER_SPEC = "memory_backfill"
DEFAULT_BATCH = 20
DEFAULT_CONCURRENCY = 6


class FactLabel(BaseModel):
    """The labeler's verdict on one legacy fact."""

    id: int = Field(description="The fact's #id.")
    reason: str = Field(description="Under 15 words, written before the label.")
    keep: bool
    type: (
        Literal["identity", "rule", "preference", "decision", "context", "event"] | None
    ) = None
    scope: Literal["global", "project"] | None = None
    project: str | None = Field(
        default=None, description="Project slug, for project scope only."
    )


class BatchLabels(BaseModel):
    labels: list[FactLabel]


@dataclass(frozen=True)
class LegacyFact:
    line: int
    record: MemoryRecord
    hint: str | None  # the source conversation's project tag

    def render(self) -> str:
        date = (self.record.born_on or "")[:10] or "undated"
        return f"#{self.line} ({date}; from: {self.hint or 'none'}) {self.record.fact}"


Labeler = Callable[[list[LegacyFact], list[dict]], Awaitable[list[FactLabel]]]


@dataclass
class Decision:
    fact: LegacyFact
    label: FactLabel
    op: Literal["archive", "reclassify"]
    fields: dict | None = None


@dataclass
class BackfillReport:
    decisions: list[Decision] = field(default_factory=list)
    unlabeled: list[int] = field(default_factory=list)
    remaining: int = 0
    dry_run: bool = False

    def summary(self) -> str:
        arch = sum(d.op == "archive" for d in self.decisions)
        moved: dict[str, int] = {}
        for d in self.decisions:
            if d.fields:
                key = (
                    d.fields["scope"]
                    if d.fields["scope"] == "global"
                    else (d.fields.get("project") or "unfiled")
                )
                moved[key] = moved.get(key, 0) + 1
        mode = " (dry run)" if self.dry_run else ""
        where = ", ".join(
            f"{k} {n}" for k, n in sorted(moved.items(), key=lambda x: -x[1])
        )
        return (
            f"labeled {len(self.decisions)} legacy fact(s){mode}: {arch} archive, "
            f"{len(self.decisions) - arch} reclassify ({where or '-'}); "
            f"{len(self.unlabeled)} unlabeled; {self.remaining} legacy left after this"
        )


# --- the default (DeepSeek, spec-driven) labeler ------------------------------------


def batch_prompt(facts: list[LegacyFact], projects: list[dict]) -> str:
    lines = ["PROJECTS:"]
    for p in projects:
        parts = [p["slug"], p.get("name") or p["slug"], p.get("description") or ""]
        lines.append("- " + " — ".join(x for x in parts if x))
    lines += ["", "FACTS:"]
    lines += [f.render() for f in facts]
    return "\n".join(lines)


async def deepseek_labeler(
    facts: list[LegacyFact], projects: list[dict]
) -> list[FactLabel]:
    from pydantic_ai import Agent

    from paf.degeneration_guard import run_guarded
    from paf.specs import resolve_background_spec

    spec, _raw = resolve_background_spec(LABELER_SPEC)
    if not spec.model or not spec.instructions:
        raise RuntimeError(f"{LABELER_SPEC} spec needs a model and instructions")
    agent = Agent(
        str(spec.model),
        name="memory-backfill",
        instructions=str(spec.instructions),
        output_type=BatchLabels,
        model_settings=spec.model_settings,  # type: ignore[arg-type]
        retries=2,
    )
    result = await run_guarded(agent, batch_prompt(facts, projects))
    return result.output.labels


# --- the pass ------------------------------------------------------------------------


async def legacy_facts(directory: str, *, agent_name: str) -> list[LegacyFact]:
    """Current (index-winning, not archived/superseded) facts with no scope —
    v1 rows not yet re-filed — oldest first, each with its project hint."""
    from ._index import MemoryIndex, update_index
    from ._render import _apply_overrides, _select_index_winners
    from ._scoped import project_resolver

    update_index(directory)
    journal = JournalStore(directory)
    winners = _select_index_winners(journal, MemoryIndex.load(directory))
    current, _pending = _apply_overrides(winners, load_overrides(directory))
    resolve = project_resolver(dict(journal.read_all()), agent_name=agent_name)
    return [
        LegacyFact(line, rec, await resolve(rec.source_ref))
        for line, rec in sorted(current, key=lambda x: x[0])
        if rec.scope is None
    ]


def _decide(fact: LegacyFact, label: FactLabel, slugs: set[str]) -> Decision:
    if not label.keep or label.type is None or label.scope is None:
        return Decision(fact, label, "archive")
    project = None
    if label.scope == "project":
        # Unknown/null slug -> unfiled; the render still files it by its
        # source conversation's tag.
        project = label.project if label.project in slugs else None
    return Decision(
        fact,
        label,
        "reclassify",
        {"type": label.type, "scope": label.scope, "project": project},
    )


async def backfill(
    directory: str,
    *,
    agent_name: str,
    labeler: Labeler | None = None,
    limit: int | None = None,
    only: set[int] | None = None,
    batch_size: int = DEFAULT_BATCH,
    concurrency: int = DEFAULT_CONCURRENCY,
    dry_run: bool = True,
    labeler_name: str = "",
) -> BackfillReport:
    """Label up to ``limit`` legacy facts (or just the ``only`` journal lines);
    apply, or with ``dry_run`` only report."""
    from paf.conversations import default_project_store

    labeler = labeler or deepseek_labeler
    if not labeler_name:
        labeler_name = LABELER_SPEC if labeler is deepseek_labeler else "custom"
    projects = [p for p in default_project_store().list() if not p.get("archived")]
    slugs = {p["slug"] for p in projects}

    todo = await legacy_facts(directory, agent_name=agent_name)
    total = len(todo)
    if only is not None:
        todo = [f for f in todo if f.line in only]
    if limit is not None:
        todo = todo[:limit]
    batches = [todo[i : i + batch_size] for i in range(0, len(todo), batch_size)]
    report = BackfillReport(dry_run=dry_run)
    store = OverrideStore(directory)
    sem = asyncio.Semaphore(max(1, concurrency))

    async def run(batch: list[LegacyFact]) -> None:
        async with sem:
            try:
                labels = await labeler(batch, projects)
            except Exception:  # noqa: BLE001 - one bad batch never stops the pass
                logger.exception("backfill: labeler failed on #%s..", batch[0].line)
                report.unlabeled += [f.line for f in batch]
                return
        by_id = {lb.id: lb for lb in labels}
        for f in batch:
            lb = by_id.get(f.line)
            if lb is None:
                report.unlabeled.append(f.line)
                continue
            d = _decide(f, lb, slugs)
            report.decisions.append(d)
            if dry_run:
                continue
            detail = {"reviewer": "backfill", "judge": labeler_name}
            if d.op == "archive":
                store.append(
                    op="archive",
                    target_line=f.line,
                    by="reviewer",
                    reason=lb.reason,
                    detail=detail,
                )
            else:
                store.append(
                    op="reclassify",
                    target_line=f.line,
                    by="reviewer",
                    reason=lb.reason,
                    fields=d.fields,
                    detail=detail,
                )

    await asyncio.gather(*(run(b) for b in batches))
    report.decisions.sort(key=lambda d: d.fact.line)
    report.remaining = total - (0 if dry_run else len(report.decisions))
    if not dry_run and report.decisions:
        from ._index import update_index
        from ._render import write_render

        update_index(directory)
        write_render(
            JournalStore(directory),
            os.path.join(directory, "MEMORY.md"),
            agent_name=agent_name,
        )
    logger.info("memory backfill %s: %s", directory, report.summary())
    return report
