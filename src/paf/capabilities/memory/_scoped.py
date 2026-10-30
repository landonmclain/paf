"""Scoped memory render — what one conversation sees, computed per turn.

Replaces the tier-capped ``MEMORY.md`` injection when the spec opts in
(``memory: render:``; vault Designs/Memory-Pipeline-v2-Plan §3).
``MEMORY.md`` stays the static full view (``/memory``, legacy tooling); this
render depends on the CONVERSATION (its project) and the CURRENT MESSAGE, so
it is built at injection time.

Sections, each with a token budget (spec-overridable, see :class:`RenderBudget`):

- **About you** — global facts, everywhere. Ordered by type (identity, rules,
  preferences, decisions, context), newest first within a type.
- **<this project>** — the conversation's project, selected by BM25 against the
  latest user turns, then recency; near-duplicates skipped; "+N more" pointer.
- **<mentioned project>** — another registered project named in the current
  message: its top few relevant facts only.
- **Recent history** — dated, newest first: events plus "replaced" lines from
  ``supersede`` overrides, within a day window; global + this/mentioned project.
- **Other projects** — one line naming the rest, so the model knows they exist.

**Render-time filing.** Facts without a project (all v1 rows; v2 facts from
untagged conversations) are filed by their source turn: ``source_ref`` → run →
conversation → that conversation's CURRENT project tag. Tagging a conversation
later therefore files its facts on the next render. v1 rows: ``t1`` → global,
``event`` → history, others → the source conversation's project (else unfiled,
not injected — still searchable).

Every line ends with its journal id ``#N`` (``read_turn("#N")`` reads the source).
"""

from __future__ import annotations

import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, fields
from datetime import datetime, timedelta
from typing import Any

from paf.capabilities.search._ranking import Bm25Params, bm25_score, tokenize
from paf.clock import now_local, to_local

from ._index import MemoryIndex, _tokens, update_index
from ._journal import JournalStore, MemoryRecord
from ._overrides import load_overrides

_TYPE_RANK = {
    "identity": 0,
    "rule": 1,
    "preference": 2,
    "decision": 3,
    "context": 4,
    "fact": 5,
    "current": 6,
    "task": 7,
    "event": 8,
}
_MERGE_REF = re.compile(r"#(\d+)")


@dataclass(frozen=True)
class RenderBudget:
    """Token budgets + windows for the scoped render (spec ``memory.render``)."""

    global_tokens: int = 2000
    project_tokens: int = 2500
    history_tokens: int = 1000
    mentioned_tokens: int = 300
    history_days: int = 30
    mentioned_max: int = 5
    redundancy: float = 0.8
    """Skip a candidate whose content tokens are this covered by one already shown."""

    @classmethod
    def from_config(cls, cfg: dict | None) -> RenderBudget:
        """Spec values override defaults; unknown/invalid keys are ignored."""
        kw: dict[str, Any] = {}
        for f in fields(cls):
            v = (cfg or {}).get(f.name)
            if isinstance(v, bool) or v is None:
                continue
            if f.name == "redundancy" and isinstance(v, int | float) and 0 < v <= 1:
                kw[f.name] = float(v)
            elif isinstance(v, int) and v >= 0:
                kw[f.name] = v
        return cls(**kw)


def _tok(text: str) -> int:
    return max(1, len(text) // 4)


# --- run -> conversation (cached; first load unconditional, then refreshed on a
#     miss at most once a minute) -----------------------------------------------

_RUN_CONV: dict[str, str] = {}
_RUN_CONV_LOADED = 0.0


async def _conversation_of_run(run_id: str, step_store=None) -> str | None:
    global _RUN_CONV_LOADED
    # `not _RUN_CONV` is the cold-start guard: time.monotonic() starts near 0 in a
    # fresh process, so the 60s staleness test alone would never fire at startup.
    if run_id not in _RUN_CONV and (not _RUN_CONV or time.monotonic() - _RUN_CONV_LOADED > 60):
        if step_store is None:
            from paf.capabilities.persistence import _build_store

            step_store = _build_store()
        for run in await step_store.list_runs():
            if run.conversation_id:
                _RUN_CONV[run.run_id] = run.conversation_id
        _RUN_CONV_LOADED = time.monotonic()
    return _RUN_CONV.get(run_id)


# --- filing ---------------------------------------------------------------------


@dataclass
class _Filed:
    globals: list[tuple[int, MemoryRecord]] = field(default_factory=list)
    projects: dict[str, list[tuple[int, MemoryRecord]]] = field(default_factory=dict)
    events: list[tuple[int, MemoryRecord, str | None]] = field(default_factory=list)
    section: Callable[[MemoryRecord], Awaitable[str | None]] | None = None
    """The filing rule, for rows outside ``current`` (history's replaced facts)."""


GLOBAL = "global"


def section_resolver(
    project_of_ref: Callable[[str], Awaitable[str | None]],
) -> Callable[[MemoryRecord], Awaitable[str | None]]:
    """The ONE filing rule: a fact's section is ``GLOBAL``, a project slug, or
    ``None`` (unfiled — never injected, still searchable). Global: scope
    ``global``, or a legacy ``t1`` non-event. Otherwise the fact's project, else
    its source conversation's current tag."""

    async def section(rec: MemoryRecord) -> str | None:
        if rec.scope == "global":
            return GLOBAL
        if rec.scope is None and rec.tier == "t1" and rec.type != "event":
            return GLOBAL
        return rec.project or await project_of_ref(rec.source_ref)

    return section


def project_resolver(
    records: dict[int, MemoryRecord], *, agent_name: str, step_store=None
) -> Callable[[str], Awaitable[str | None]]:
    """``source_ref`` -> the source conversation's CURRENT project tag (or
    ``None``). Legacy ``merge(...)`` refs resolve through their sources."""
    from paf.conversations import default_conversation_meta_store, lane_id
    from paf.specs import resolve_profile_name

    meta = default_conversation_meta_store(agent_name).load()
    profile = resolve_profile_name(agent_name)

    async def project_of_ref(ref: str, depth: int = 0) -> str | None:
        if ref.startswith("merge("):  # legacy merge row: file by its sources
            if depth > 5:
                return None
            for m in _MERGE_REF.findall(ref):
                src = records.get(int(m))
                if src is not None:
                    p = await project_of_ref(src.source_ref, depth + 1)
                    if p:
                        return p
            return None
        cid = await _conversation_of_run(ref.split(":", 1)[0], step_store)
        if not cid:
            return None
        entry = meta.get(lane_id(cid, profile)) or {}
        return entry.get("project") or None

    return project_of_ref


async def _file(
    current: list[tuple[int, MemoryRecord]],
    records: dict[int, MemoryRecord],
    *,
    agent_name: str,
    step_store=None,
) -> _Filed:
    section = section_resolver(
        project_resolver(records, agent_name=agent_name, step_store=step_store)
    )
    filed = _Filed(section=section)
    for line, rec in current:
        sec = await section(rec)
        if sec is None:
            continue  # unfiled — not injected, still searchable
        if rec.type == "event":
            filed.events.append((line, rec, None if sec == GLOBAL else sec))
        elif sec == GLOBAL:
            filed.globals.append((line, rec))
        else:
            filed.projects.setdefault(sec, []).append((line, rec))
    return filed


async def filed_memory(
    directory: str, *, agent_name: str, step_store=None
) -> tuple[_Filed, dict[int, MemoryRecord], dict]:
    """Current facts filed into global / per-project / events — the one view
    the render, ``list_memory`` and ``search_memory`` all share. Also returns
    every journal record and the folded overrides (for history lines)."""
    from ._render import _apply_overrides, _select_index_winners

    update_index(directory)
    journal = JournalStore(directory)
    records = dict(journal.read_all())
    overrides = load_overrides(directory)
    winners = _select_index_winners(journal, MemoryIndex.load(directory))
    current, _pending = _apply_overrides(winners, overrides)
    filed = await _file(current, records, agent_name=agent_name, step_store=step_store)
    return filed, records, overrides


def _pool(filed: _Filed, project: str) -> list[tuple[int, MemoryRecord]]:
    """Facts in one section: ``"global"`` (About you) or a project slug —
    including that section's events."""
    if project == "global":
        facts = list(filed.globals)
    else:
        facts = list(filed.projects.get(project, []))
    key = None if project == "global" else project
    facts += [(ln, r) for ln, r, p in filed.events if p == key]
    return facts


def _line(ln: int, r: MemoryRecord, label: str = "") -> str:
    date = f"{_date(r.born_on)}  " if r.type == "event" else ""
    return f"- {label}{date}{r.fact} — #{ln}"


LIST_PAGE = 25
SEARCH_TOP = 10


async def list_memory(
    directory: str,
    *,
    agent_name: str,
    project: str,
    type: str | None = None,
    offset: int = 0,
    page_size: int = LIST_PAGE,
    step_store=None,
) -> str:
    """One page of a section's facts, in render order (type, newest first)."""
    filed, _records, _ov = await filed_memory(
        directory, agent_name=agent_name, step_store=step_store
    )
    facts = _pool(filed, project)
    if type:
        facts = [(ln, r) for ln, r in facts if r.type == type]
    facts.sort(key=lambda x: (_TYPE_RANK.get(x[1].type, 9), _neg(x[1].born_on)))
    total = len(facts)
    where = "About you" if project == "global" else project
    what = f"{where}" + (f", type={type}" if type else "")
    if not total:
        return f"No memory facts in {what}."
    start = min(max(0, offset), total)
    end = min(total, start + page_size)
    out = [f"{what}: facts {start + 1}-{end} of {total}"]
    out += [_line(ln, r) for ln, r in facts[start:end]]
    if end < total:
        args = f'project="{project}"' + (f', type="{type}"' if type else "")
        out.append(f"(more: list_memory({args}, offset={end}))")
    return "\n".join(out)


async def search_memory(
    directory: str,
    *,
    agent_name: str,
    query: str,
    sections: list[str],
    top: int = SEARCH_TOP,
    step_store=None,
) -> str:
    """BM25 over the facts in ``sections`` ("global" and/or project slugs);
    only facts that match at least one query word."""
    filed, _records, _ov = await filed_memory(
        directory, agent_name=agent_name, step_store=step_store
    )
    labelled: dict[int, tuple[MemoryRecord, str]] = {}
    for sec in sections:
        tag = "[about you] " if sec == "global" else f"[{sec}] "
        for ln, r in _pool(filed, sec):
            labelled[ln] = (r, tag)
    ranked = _rank_by_query([(ln, r) for ln, (r, _t) in labelled.items()], query)
    q = set(tokenize(query))
    hits = [(ln, r) for ln, r in ranked if q & set(tokenize(r.fact))][:top]
    if not hits:
        return f"No memory facts match {query!r} in {', '.join(sections)}."
    return "\n".join(_line(ln, r, labelled[ln][1]) for ln, r in hits)


# --- selection -------------------------------------------------------------------


class _Picker:
    """Greedy fill to a token budget, skipping near-duplicates of shown lines."""

    def __init__(self, budget: int, redundancy: float) -> None:
        self.budget = budget
        self.redundancy = redundancy
        self.used = 0
        self.lines: list[str] = []
        self.shown: list[set[str]] = []
        self.skipped = 0

    def offer(self, text: str, line: str) -> bool:
        toks = _tokens(text)
        if toks and any(
            len(toks & s) / len(toks) >= self.redundancy for s in self.shown
        ):
            self.skipped += 1
            return True  # covered: counts as handled, costs nothing
        cost = _tok(line)
        if self.used + cost > self.budget:
            return False
        self.used += cost
        self.lines.append(line)
        self.shown.append(toks)
        return True


def _rank_by_query(
    facts: list[tuple[int, MemoryRecord]], query: str
) -> list[tuple[int, MemoryRecord]]:
    """BM25 relevance to the query, then recency for the rest."""
    import math

    q = list(dict.fromkeys(tokenize(query)))
    docs = {ln: tokenize(r.fact) for ln, r in facts}
    n = max(1, len(facts))
    avgdl = sum(map(len, docs.values())) / n if docs else 1.0
    df: dict[str, int] = {}
    for toks in docs.values():
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    idf = {t: math.log((n - c + 0.5) / (c + 0.5) + 1) for t, c in df.items()}
    params = Bm25Params()
    scored = [
        (bm25_score(q, docs[ln], idf, avgdl, params) if q else 0.0, r.born_on, ln, r)
        for ln, r in facts
    ]
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [(ln, r) for _s, _b, ln, r in scored]


def _mentioned(query: str, projects: list[dict], exclude: str | None) -> list[str]:
    text = query.lower()
    hits = []
    for p in projects:
        slug = p["slug"]
        if slug == exclude:
            continue
        names = {slug, slug.replace("-", " "), str(p.get("name") or "").lower()}
        names |= {str(a).lower() for a in p.get("aliases") or []}
        if any(
            n and re.search(rf"(?<![\w-]){re.escape(n)}(?![\w-])", text) for n in names
        ):
            hits.append(slug)
    return hits


def _date(born_on: str) -> str:
    dt = to_local(born_on)
    return dt.date().isoformat() if dt else "undated"


# --- the render ------------------------------------------------------------------


async def render_scoped(
    directory: str,
    *,
    agent_name: str,
    conversation_id: str | None,
    query: str,
    budget: RenderBudget | None = None,
    project: str | None = None,
    now: datetime | None = None,
    step_store=None,
) -> str:
    """The ``<memory>`` body for one conversation turn (empty string if nothing).

    ``project`` and ``now`` are opt-in overrides for the benchmark/isolated
    harness: ``project`` forces the conversation's project section instead of
    resolving it from the conversation registry (``project_for_conversation``),
    and ``now`` anchors the recent-history window to a fixed time instead of
    ``now_local()`` (LongMemEval seeds are dated 2023; with no anchor every
    2023 event falls outside the 30-day window and is dropped). Both default to
    ``None`` = the live behavior (registry resolution / ``now_local()``).
    """
    from paf.conversations import default_project_store, project_for_conversation

    budget = budget or RenderBudget()
    filed, records, overrides = await filed_memory(
        directory, agent_name=agent_name, step_store=step_store
    )

    if project is None:
        project = project_for_conversation(agent_name, conversation_id)
    registry = default_project_store().list()
    mentioned = _mentioned(query, registry, project)[:2]
    names = {p["slug"]: p.get("name") or p["slug"] for p in registry}
    out: list[str] = []

    # About you — global, by type then recency.
    g = _Picker(budget.global_tokens, budget.redundancy)
    glob = sorted(
        filed.globals,
        key=lambda x: (_TYPE_RANK.get(x[1].type, 9), _neg(x[1].born_on)),
    )
    dropped = sum(1 for ln, r in glob if not g.offer(r.fact, f"- {r.fact} — #{ln}"))
    if g.lines:
        out += ["## About you", *g.lines]
        if dropped:
            out.append(f'(+{dropped} more about you; list_memory(project="global"))')

    # This conversation's project — relevance, then recency.
    if project:
        facts = _rank_by_query(filed.projects.get(project, []), query)
        p = _Picker(budget.project_tokens, budget.redundancy)
        more = sum(1 for ln, r in facts if not p.offer(r.fact, f"- {r.fact} — #{ln}"))
        if p.lines:
            out += ["", f"## {names.get(project, project)}", *p.lines]
            if more:
                out.append(
                    f"(+{more} more {names.get(project, project)} facts; "
                    f'list_memory(project="{project}"))'
                )

    # Projects the message names — top few relevant facts only.
    for slug in mentioned:
        facts = [
            (ln, r) for ln, r in _rank_by_query(filed.projects.get(slug, []), query)
        ][: budget.mentioned_max]
        m = _Picker(budget.mentioned_tokens, budget.redundancy)
        for ln, r in facts:
            m.offer(r.fact, f"- {r.fact} — #{ln}")
        if m.lines:
            out += ["", f"## {names.get(slug, slug)} (mentioned)", *m.lines]

    # Recent history — events + replaced values, dated, newest first.
    # Undated rows (pre-born_on legacy) never count as recent: "undated" would
    # otherwise sort above every ISO date in the window check.
    since = (
        ((now or now_local()) - timedelta(days=budget.history_days)).date().isoformat()
    )
    visible = {None, project, *mentioned}
    items: list[tuple[str, str, str]] = []  # (date, kind, line)
    for ln, r, proj in filed.events:
        d = _date(r.born_on)
        if proj in visible and d != "undated" and d >= since:
            items.append((d, "event", f"- {d}  {r.fact} — #{ln}"))

    def refiled(line: int) -> MemoryRecord | None:
        rec, st = records.get(line), overrides.get(line)
        if rec is not None and st is not None and st.fields:
            rec = rec.with_fields(**st.fields)
        return rec

    for target, st in overrides.items():
        if not st.is_superseded or target not in records or st.superseded_by is None:
            continue
        old, new = refiled(target), refiled(st.superseded_by)
        if old is None or new is None:
            continue
        d = _date(new.born_on)
        if d == "undated" or d < since:
            continue
        assert filed.section is not None
        sec = await filed.section(new) or await filed.section(old)
        if sec is None:
            continue  # unfiled: shown nowhere
        proj = None if sec == GLOBAL else sec
        if proj in visible:
            items.append(
                (
                    d,
                    "change",
                    f"- {d}  replaced: {old.fact} — #{target} → #{st.superseded_by}",
                )
            )
    items.sort(key=lambda x: x[0], reverse=True)
    h = _Picker(budget.history_tokens, budget.redundancy)
    for _d, _k, line in items:
        if not h.offer(line, line):
            break
    if h.lines:
        out += ["", "## Recent history", *h.lines]

    # Other projects — one line so the model knows they exist.
    others = [
        f"{s} ({len(filed.projects[s])} facts)"
        for s in sorted(filed.projects, key=lambda s: -len(filed.projects[s]))
        if s in names and s != project and s not in mentioned
    ]
    if others:
        out += [
            "",
            f"Other projects: {', '.join(others)} — mention one to bring it in, "
            'or list_memory(project="<slug>").',
        ]
    return "\n".join(out).strip()


def _neg(born_on: str) -> str:
    """Sort key making newer ISO dates sort first in an ascending sort."""
    return "".join(chr(0x10FFFF - ord(c)) for c in (born_on or ""))
