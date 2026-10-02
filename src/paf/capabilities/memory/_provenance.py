"""Memory id (``#N``) -> the conversation turns a fact came from.

Backs ``read_turn("#N")``: every injected memory line ends with its journal
line id, and the journal row is the lookup table id -> ``source_ref``
(``run_id:turn_id``). Follows the relationships a line carries:

- a legacy ``merge(#a,#b,…)`` row -> its source facts' turns (nested merges
  unwound), so one id can return several turns;
- a ``supersede`` override -> the replacing fact's turn too (old AND new
  statement), and a replacing fact -> the facts it replaced.

**Read-time repair of legacy refs** (:func:`repair_ref`). Before 2026-10-01,
consolidation numbered a turn by its position in the consolidation SLICE
instead of in its own run (see ``_consolidate._turn_refs``), so ~41% of old
refs point at the wrong turn. Their run id is still right — it is the run
that created the turn — so a ref is corrected to one of that run's OWN turns:
the only one, or the one whose text best matches the fact (BM25). Refs written
since the fix always point at an own turn, so the repair never changes them.
"""

from __future__ import annotations

import re

from ._journal import JournalStore
from ._overrides import load_overrides

_REF = re.compile(r"#(\d+)")
MAX_TURNS = 5


def memory_turn_refs(directory: str, line: int) -> list[tuple[str, str, str]]:
    """``[(label, run_id:turn_id, fact), …]`` for memory line ``line`` (max 5).

    Raises ``KeyError`` when the line does not exist.
    """
    records = dict(JournalStore(directory).read_all())
    if line not in records:
        raise KeyError(line)
    overrides = load_overrides(directory)
    out: list[tuple[str, str, str]] = []
    seen: set[int] = set()

    def add(n: int, label: str, depth: int = 0) -> None:
        if n in seen or n not in records or depth > 6 or len(out) >= MAX_TURNS:
            return
        seen.add(n)
        ref = records[n].source_ref
        if ref.startswith("merge("):
            for m in _REF.findall(ref):
                add(int(m), f"{label}, merged source #{m}", depth + 1)
        else:
            out.append((label, ref, records[n].fact))

    add(line, f"#{line}")
    st = overrides.get(line)
    if st is not None and st.is_superseded and st.superseded_by:
        add(st.superseded_by, f"#{st.superseded_by} (replaced #{line})")
    for target, s in overrides.items():
        if s.is_superseded and s.superseded_by == line:
            add(target, f"#{target} (replaced by #{line})")
    return out


def _turn_text(history, start: int, end: int) -> str:
    from pydantic_ai.messages import ModelResponse, TextPart

    from paf.capabilities._turns import _real_user_prompt_text

    parts: list[str] = []
    for msg in history[start:end]:
        text = _real_user_prompt_text(msg)
        if text:
            parts.append(text)
        if isinstance(msg, ModelResponse):
            parts += [p.content for p in msg.parts if isinstance(p, TextPart)]
    return "\n".join(parts)


async def repair_ref(ref: str, fact: str, history_source) -> tuple[str, bool]:
    """``(ref, repaired?)``: a legacy ref moved onto its run's own turn.

    Keeps the ref when it already lands on a turn its run created (every ref
    written since the numbering fix). Otherwise picks the run's own turn — the
    only one, or the best BM25 match for ``fact``. Unreadable runs, or runs
    with no own turn, keep the ref unchanged.
    """
    from paf.capabilities._turns import user_anchor_indices
    from paf.capabilities.search._ranking import bm25_rank

    run_id, _, k = ref.rpartition(":")
    try:
        k_int = int(k)
        history = await history_source.run_history(run_id=run_id)
    except Exception:  # noqa: BLE001 - leave the ref as stored
        return ref, False
    anchors = user_anchor_indices(history)

    def own(i: int) -> bool:
        return getattr(history[anchors[i]], "run_id", None) == run_id

    if 0 <= k_int < len(anchors) and own(k_int):
        return ref, False
    mine = [i for i in range(len(anchors)) if own(i)]
    if not mine:
        return ref, False
    if len(mine) == 1:
        return f"{run_id}:{mine[0]}", True
    ends = [*anchors[1:], len(history)]
    texts = [_turn_text(history, anchors[i], ends[i]) for i in mine]
    ranked = bm25_rank(fact, texts)
    best = mine[ranked[0][0]] if ranked else mine[-1]
    return f"{run_id}:{best}", True
