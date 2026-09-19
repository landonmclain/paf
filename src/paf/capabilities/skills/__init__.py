"""paf-owned Skills capability — one capability for all skill functionality.

Unifies the skill *load* mechanism (list/load/read-resources) with the skill-hint
*pointer engine* (deterministic BM25 `point_skills` pre-arming which skill a
message needs). The pointer is an **internal primitive** for the auto per-turn
hint layer — it is NOT an agent-facing tool. Indexes a skill-library directory at
construction; re-reads SKILL.md bodies on demand; exposes skill resources
skill-scoped.

- `_corpus`   — the tree-aware hint corpus (adapted from skill-note-hint, MIT).
- `_point`    — the standalone `point()` pointer entrypoint.
- `_toolset`  — the `SkillToolset` (list_skills / load_skill /
  list_skill_files / read_skill_file; internal point_skills).
- `_capability` — the `Skills` capability (wraps the toolset).

Wired through the capability-toolset seam (registry + profile + builder); the
standalone skill-note-hint repo stays independent for the Hermes-plugin path.
"""

from __future__ import annotations

from ._capability import Skills
from ._toolset import SkillToolset

__all__ = ["Skills", "SkillToolset"]
