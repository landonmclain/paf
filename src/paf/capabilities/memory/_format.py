"""The canonical memory file format, shared by writer and reader alike.

Single source of truth for how the durable-fact notebook (``MEMORY.md``) is
structured. Both the subject's capture guidance (``_DEFAULT_GUIDANCE`` in
``_capability.py``) and the consolidation owner's operating manual
(``CONSOLIDATION_GUIDANCE`` in ``_consolidate.py``) embed ``MEMORY_FORMAT``
verbatim, so the agent that writes memory and the agent that reads it agree on
the exact shape by construction.

Design contract (Landon, 2026-08-12):
- ONE canonical file, ``MEMORY.md``. As long as it stays small enough to inject
  in full, do not split into topic files — an unread subfile is worse than a
  slightly fuller notebook (a pending-action fact orphaned in ``actions.md``
  was the trigger for this).
- Four fixed sections, in this order. ``Facts`` / ``Preferences`` / ``Pending``
  are one-fact-per-line bullets. ``Notes`` is the escape hatch for genuine
  multi-sentence context.
- Write bias: when in doubt, record it — persist every durable fact and every
  user-stated somatic (preferences, names, plans, decisions, pending actions).
  Skip only truly incidental small talk. It is cheaper to prune a kept entry
  than to have missed it at compaction.
- ``Pending`` is checkbox-format; countable items keep their count inline.
- Update in place, newest value wins, delete when done/false. Never append a
  duplicate of a fact already recorded.
"""

MEMORY_FORMAT = """MEMORY.md format — follow it exactly:
- ONE canonical file: MEMORY.md. Keep everything durable there. Do NOT create
  separate topic/action files unless a single entry genuinely needs prose;
  prefer folding it into ## Notes. An unread subfile is worse than a slightly
  fuller notebook.
- Fixed sections, in this order:
    ## Facts
    one durable fact per line — stable, countable facts (incl. conventions)
    ## Preferences
    one user preference per line — style, tastes, how they like things done
    ## Pending
    checkbox lines for open actions: `- [ ] action (with count if countable)`
    finish an item by checking it `- [x]` and deleting it when done
    ## Notes
    multi-sentence context only when a single line cannot capture it — keep rare
- One fact = one line. No sub-bullets, no prose, under Facts/Preferences/Pending.
- Keep countable things countable inline: "Pick up 2 items (navy blazer, Zara
  boots)" not a vague "errands".
- Write bias — WHEN IN DOUBT, RECORD IT. Persist every durable fact and every
  user-stated somatic (preferences, names, plans, decisions, pending actions).
  Skip only truly incidental small talk (greetings, thanks, one-off niceties).
  It costs less to prune a kept entry than to have missed it at compaction.
- Update in place, never append a duplicate: if the fact already exists, edit it
  to the newest value (NEWER WINS). Check off and delete finished `[x]` lines.
- Never claim something was remembered unless you actually wrote it this turn."""


def extract_sections(format_spec: str = MEMORY_FORMAT) -> list[str]:
    """Return the section headers, in declared order, from a format spec.

    Used by tests to pin that the written format keeps the canonical section
    order and shape regardless of surrounding guidance prose.
    """
    return [
        line.strip()[3:].rstrip(":")
        for line in format_spec.splitlines()
        if line.lstrip().startswith("## ")
    ]
