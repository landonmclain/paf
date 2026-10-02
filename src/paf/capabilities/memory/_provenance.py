"""Memory id (``#N``) -> the conversation turns a fact came from.

Backs ``read_turn("#N")``: every injected memory line ends with its journal
line id, and the journal row is the lookup table id -> ``source_ref``
(``run_id:turn_id``). Follows the relationships a line carries:

- a legacy ``merge(#a,#b,…)`` row -> its source facts' turns (nested merges
  unwound), so one id can return several turns;
- a ``supersede`` override -> the replacing fact's turn too (old AND new
  statement), and a replacing fact -> the facts it replaced.
"""

from __future__ import annotations

import re

from ._journal import JournalStore
from ._overrides import load_overrides

_REF = re.compile(r"#(\d+)")
MAX_TURNS = 5


def memory_turn_refs(directory: str, line: int) -> list[tuple[str, str]]:
    """``[(label, run_id:turn_id), …]`` for memory line ``line`` (max 5).

    Raises ``KeyError`` when the line does not exist.
    """
    records = dict(JournalStore(directory).read_all())
    if line not in records:
        raise KeyError(line)
    overrides = load_overrides(directory)
    out: list[tuple[str, str]] = []
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
            out.append((label, ref))

    add(line, f"#{line}")
    st = overrides.get(line)
    if st is not None and st.is_superseded and st.superseded_by:
        add(st.superseded_by, f"#{st.superseded_by} (replaced #{line})")
    for target, s in overrides.items():
        if s.is_superseded and s.superseded_by == line:
            add(target, f"#{target} (replaced by #{line})")
    return out
