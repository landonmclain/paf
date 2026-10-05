"""Pairwise memory reviewer — keep every current fact unique and up to date.

Replaces the old whole-memory reviewer (removed), which merged and
over-archived to fit the render's count caps. This one is narrow by design
(vault Designs/Memory-Pipeline-v2-Plan §2):

- **Incremental.** Each pass reviews only facts appended since its watermark
  (``memory-review-watermark.json``), each against OLDER current facts.
- **Code picks the comparisons.** For a new fact B: BM25 top-k among older
  current facts in B's project plus global/legacy facts.
- **The model only labels.** One judge call per new fact: for each candidate A,
  ``same`` / ``update`` / ``different`` (+ which to keep for ``same``).
- **Code applies the actions**, into the override sidecar (append-only,
  reversible with ``reject``):
    same      -> ``archive`` whichever fact adds nothing
    update    -> ``supersede`` A by B (A leaves the current view; history)
    different -> nothing — the default when unsure
- **Assistant suggestions** (``memory_archive_suggestion``, pending archives)
  are resolved by a second narrow prompt: approve or reject.

Never merges, never rewrites a fact, never changes a tier. Every action records
the pair, the label, the judge's reason, and the judge model in ``detail``.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from paf.capabilities.search._ranking import Bm25Params, bm25_score, tokenize

from ._journal import JournalStore, MemoryRecord
from ._overrides import OverrideStore, load_overrides

logger = logging.getLogger(__name__)

WATERMARK_FILENAME = "memory-review-watermark.json"
DEFAULT_TOP_K = 12
JUDGE_SPEC = "memory_pair_reviewer"

Label = Literal["same", "update", "different"]


# --- judge I/O ------------------------------------------------------------------


class PairJudgement(BaseModel):
    """The judge's verdict on one (older fact, new fact) pair."""

    index: int = Field(description="The older fact's number in the list.")
    reason: str = Field(description="Under 15 words, written before the label.")
    label: Label
    keep: Literal["new", "old"] | None = Field(
        default=None,
        description="For `same` only: which fact to keep (the one that says more).",
    )


class PairReview(BaseModel):
    judgements: list[PairJudgement]


class SuggestionVerdict(BaseModel):
    reason: str = Field(description="Under 15 words, written before the verdict.")
    verdict: Literal["approve", "reject"]


@dataclass(frozen=True)
class Fact:
    line: int
    record: MemoryRecord

    def render(self) -> str:
        date = (self.record.born_on or "")[:10] or "undated"
        return f"(#{self.line}, {date}) {self.record.fact}"


Judge = Callable[[Fact, list[Fact]], Awaitable[list[PairJudgement]]]
SuggestionJudge = Callable[[Fact, str, list[Fact]], Awaitable[SuggestionVerdict]]


@dataclass
class Action:
    """One proposed/applied override, for the report and the audit trail."""

    op: Literal["archive", "supersede", "approve", "reject"]
    target: int
    other: int | None
    label: str
    reason: str


@dataclass
class ReviewReport:
    reviewed: list[int] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)
    watermark: int = 0
    dry_run: bool = False

    def summary(self) -> str:
        counts: dict[str, int] = {}
        for a in self.actions:
            counts[a.op] = counts.get(a.op, 0) + 1
        mode = " (dry run)" if self.dry_run else ""
        return f"reviewed {len(self.reviewed)} new fact(s){mode}: " + (
            ", ".join(f"{n} {op}" for op, n in sorted(counts.items())) or "no actions"
        )

    def outcome(self) -> str:
        """Short past-tense tally for a status line: ``reviewed 5: 2 archived``."""
        counts: dict[str, int] = {}
        for a in self.actions:
            counts[a.op] = counts.get(a.op, 0) + 1
        tally = ", ".join(
            f"{counts[op]} {label}" for op, label in _OUTCOME.items() if op in counts
        )
        return f"reviewed {len(self.reviewed)}: {tally or 'no changes'}"


_OUTCOME = {
    "archive": "archived",
    "supersede": "superseded",
    "approve": "suggestion(s) approved",
    "reject": "suggestion(s) rejected",
}


def describe_actions(
    report: ReviewReport, directory: str, *, width: int = 100
) -> list[str]:
    """One readable line per action, naming facts by their text, not line number.

    For a status panel: the reader wants to see *what* was archived or replaced
    and why, without opening the journal. The judge's reason follows the dash
    (its ``covered by #N:`` / ``superseded by #N:`` prefix is dropped, since
    the other fact is already quoted).
    """
    if not report.actions:
        return []
    records = dict(JournalStore(directory).read_all())

    def quote(line: int | None) -> str:
        record = records.get(line) if line is not None else None
        text = record.fact if record is not None else f"#{line}"
        return f'"{text[: width - 1]}…"' if len(text) > width else f'"{text}"'

    lines = []
    for a in report.actions:
        reason = a.reason
        if reason.startswith(("covered by #", "superseded by #")) and ": " in reason:
            reason = reason.split(": ", 1)[1]
        if a.op == "archive":
            line = f"archived {quote(a.target)} (same as {quote(a.other)})"
        elif a.op == "supersede":
            line = f"superseded {quote(a.target)} by {quote(a.other)}"
        elif a.op == "approve":
            line = f"archived {quote(a.target)} (approved suggestion)"
        else:
            line = f"kept {quote(a.target)} (rejected suggestion)"
        lines.append(f"{line} — {reason}" if reason else line)
    return lines


# --- current facts + candidates ------------------------------------------------


def current_facts(directory: str) -> dict[int, MemoryRecord]:
    """Line -> record for every CURRENT fact: index winners, minus facts the
    overrides archived or superseded, minus legacy ``merge(...)`` rows."""
    from ._index import MemoryIndex, update_index
    from ._render import _apply_overrides, _select_index_winners

    update_index(directory)
    winners = _select_index_winners(
        JournalStore(directory), MemoryIndex.load(directory)
    )
    kept, _pending = _apply_overrides(winners, load_overrides(directory))
    return {line: rec for line, rec in kept if not rec.source_ref.startswith("merge(")}


def _same_scope(new: MemoryRecord, old: MemoryRecord) -> bool:
    """Older facts B may be compared with: its project's, plus global/legacy
    (``project`` None). Different projects never duplicate each other."""
    return old.project is None or old.project == new.project


class _Bm25:
    def __init__(self, facts: dict[int, MemoryRecord]) -> None:
        import math

        self.lines = list(facts)
        self.docs = {ln: tokenize(facts[ln].fact) for ln in self.lines}
        n = max(1, len(self.lines))
        self.avgdl = sum(map(len, self.docs.values())) / n
        df: dict[str, int] = {}
        for toks in self.docs.values():
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log((n - c + 0.5) / (c + 0.5) + 1) for t, c in df.items()}
        self.params = Bm25Params()

    def top(self, text: str, eligible: list[int], k: int) -> list[int]:
        q = list(dict.fromkeys(tokenize(text)))
        scored = [
            (bm25_score(q, self.docs[ln], self.idf, self.avgdl, self.params), ln)
            for ln in eligible
        ]
        scored = [(s, ln) for s, ln in scored if s > 0]
        scored.sort(reverse=True)
        return [ln for _, ln in scored[:k]]


# --- watermark -----------------------------------------------------------------


def load_watermark(directory: str) -> int | None:
    try:
        with open(os.path.join(directory, WATERMARK_FILENAME), encoding="utf-8") as fh:
            return int(json.load(fh)["line"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save_watermark(directory: str, line: int) -> None:
    from paf.capabilities._storage import write_json_atomic

    write_json_atomic(os.path.join(directory, WATERMARK_FILENAME), {"line": line})


# --- the default (DeepSeek, spec-driven) judges ----------------------------------


def _judge_spec() -> tuple[str, object, str, str]:
    from paf.specs import resolve_background_spec

    spec, raw = resolve_background_spec(JUDGE_SPEC)
    if not spec.model or not spec.instructions:
        raise RuntimeError(f"{JUDGE_SPEC} spec needs a model and instructions")
    return (
        str(spec.model),
        spec.model_settings,
        str(spec.instructions),
        str(raw.get("suggestion_instructions") or ""),
    )


def _agent(output_type, instructions: str):
    from pydantic_ai import Agent

    model, settings, _pair, _sugg = _judge_spec()
    return Agent(
        model,
        name="memory-pair-reviewer",
        instructions=instructions,
        output_type=output_type,
        model_settings=settings,  # type: ignore[arg-type]
        retries=2,
    )


def pair_prompt(new: Fact, older: list[Fact]) -> str:
    lines = [f"NEW {new.render()}", "", "OLDER:"]
    lines += [f"{i}. {f.render()}" for i, f in enumerate(older, start=1)]
    return "\n".join(lines)


async def deepseek_judge(new: Fact, older: list[Fact]) -> list[PairJudgement]:
    from paf.degeneration_guard import run_guarded

    _m, _s, instructions, _sugg = _judge_spec()
    result = await run_guarded(
        _agent(PairReview, instructions), pair_prompt(new, older)
    )
    return result.output.judgements


async def deepseek_suggestion_judge(
    fact: Fact, reason: str, context: list[Fact]
) -> SuggestionVerdict:
    from paf.degeneration_guard import run_guarded

    _m, _s, _pair, instructions = _judge_spec()
    prompt = "\n".join(
        [
            f"FACT {fact.render()}",
            f"REASON GIVEN: {reason or '(none)'}",
            "",
            "NEWER FACTS:",
        ]
        + [f"- {f.render()}" for f in context]
        + ([] if context else ["(none)"])
    )
    result = await run_guarded(_agent(SuggestionVerdict, instructions), prompt)
    return result.output


# --- the pass ------------------------------------------------------------------


async def review_new_facts(
    directory: str,
    *,
    judge: Judge | None = None,
    suggestion_judge: SuggestionJudge | None = None,
    top_k: int = DEFAULT_TOP_K,
    dry_run: bool = False,
    start_after: int | None = None,
    max_facts: int | None = None,
    judge_name: str = "",
    agent_name: str | None = None,
) -> ReviewReport:
    """Review facts appended after the watermark; apply (or, dry run, propose).

    ``start_after`` overrides the watermark (a journal line to start after —
    e.g. ``journal_count - 30`` for a dry run over the last 30 facts). With no
    watermark yet and no ``start_after``, the watermark is initialized to the
    journal tail: the reviewer starts with facts appended from now on (old
    facts are a separate, explicit backfill).
    """
    if judge is None:
        from ._cascade import default_judge

        judge, default_name = default_judge()
        judge_name = judge_name or default_name
    suggestion_judge = suggestion_judge or deepseek_suggestion_judge
    if not judge_name:
        judge_name = JUDGE_SPEC if judge is deepseek_judge else "custom"

    journal = JournalStore(directory)
    tail = journal.count()
    after = start_after if start_after is not None else load_watermark(directory)
    report = ReviewReport(dry_run=dry_run)
    if after is None:
        report.watermark = tail
        if not dry_run:
            save_watermark(directory, tail)
        return report

    facts = current_facts(directory)
    retired: set[int] = set()  # retired during THIS pass (archived/superseded)
    store = OverrideStore(directory)

    def apply(action: Action, *, op: str, superseded_by: int | None = None) -> None:
        report.actions.append(action)
        retired.add(action.target)
        if dry_run:
            return
        store.append(
            op=op,
            target_line=action.target,
            by="reviewer",
            reason=action.reason,
            superseded_by=superseded_by,
            detail={
                "reviewer": "pairwise",
                "label": action.label,
                "other": action.other,
                "judge": judge_name,
            },
        )

    # 1) Pending assistant suggestions -> approve (archive) / reject.
    overrides = load_overrides(directory)
    records = dict(journal.read_all())
    for line, st in sorted(overrides.items()):
        if not st.is_pending or line not in records:
            continue
        rec = records[line]
        newer = [
            Fact(ln, facts[ln])
            for ln in _Bm25(facts).top(
                rec.fact,
                [ln for ln in facts if ln > line and _same_scope(rec, facts[ln])],
                5,
            )
        ]
        verdict = await suggestion_judge(Fact(line, rec), st.reason, newer)
        op = "archive" if verdict.verdict == "approve" else "reject"
        apply(
            Action(
                op="approve" if op == "archive" else "reject",
                target=line,
                other=None,
                label=verdict.verdict,
                reason=verdict.reason,
            ),
            op=op,
        )

    # 2) New facts since the watermark, each vs older current facts.
    new_lines = sorted(ln for ln in facts if ln > after)
    if max_facts is not None:
        new_lines = new_lines[:max_facts]
    bm = _Bm25(facts)
    for b in new_lines:
        if b in retired:
            continue
        B = facts[b]
        eligible = [
            ln
            for ln in facts
            if ln < b and ln not in retired and _same_scope(B, facts[ln])
        ]
        cands = bm.top(B.fact, eligible, top_k)
        report.reviewed.append(b)
        if not cands:
            continue
        older = [Fact(ln, facts[ln]) for ln in cands]
        try:
            judgements = await judge(Fact(b, B), older)
        except Exception:  # noqa: BLE001 - one bad call never stops the pass
            logger.exception("pairwise review: judge failed on #%s; skipped", b)
            continue
        for j in judgements:
            if not 1 <= j.index <= len(older) or b in retired:
                continue
            a = older[j.index - 1].line
            if a in retired:
                continue
            if j.label == "same":
                if j.keep == "old":
                    apply(
                        Action("archive", b, a, "same", f"covered by #{a}: {j.reason}"),
                        op="archive",
                    )
                else:
                    apply(
                        Action("archive", a, b, "same", f"covered by #{b}: {j.reason}"),
                        op="archive",
                    )
            elif j.label == "update":
                apply(
                    Action(
                        "supersede", a, b, "update", f"superseded by #{b}: {j.reason}"
                    ),
                    op="supersede",
                    superseded_by=b,
                )

    report.watermark = max([after, *report.reviewed]) if report.reviewed else after
    if not dry_run:
        save_watermark(directory, report.watermark)
        if report.actions:
            from ._index import update_index
            from ._render import write_render

            update_index(directory)
            write_render(
                journal, os.path.join(directory, "MEMORY.md"), agent_name=agent_name
            )
    logger.info("pairwise review %s: %s", directory, report.summary())
    return report
