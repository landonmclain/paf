"""paf-owned Skills capability — one capability for all skill functionality.

The skill *load* mechanism (list/load/read-resources): indexes a skill-library
directory at construction, re-reads SKILL.md bodies on demand, and exposes skill
resources skill-scoped.

- `_corpus`   — the tree-aware skill corpus (BM25 over name + description + tags;
  adapted from skill-note-hint, MIT).
- `_toolset`  — the `SkillToolset` (list_skills / load_skill /
  list_skill_files / read_skill_file / create_skill / edit_skill).
- `_capability` — the `Skills` capability (wraps the toolset).

Wired through the capability-toolset seam (registry + profile + builder). There is
no per-turn skill-hint injection here; that is planned as part of the keyword
curator.
"""

from __future__ import annotations

from ._capability import Skills
from ._toolset import SkillToolset

__all__ = ["Skills", "SkillToolset"]
