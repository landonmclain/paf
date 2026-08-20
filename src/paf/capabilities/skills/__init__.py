"""paf-owned Skills capability — one capability for all skill functionality.

Unifies the skill *load* mechanism (list/load, the idea from the harness
`Skills` / pydantic-ai deferred-capability catalog) with the skill-note-hint
*trigger* (deterministic BM25 `point_to_skill` pre-arming which skill a message
needs). Indexes a skill-library directory at construction; re-reads SKILL.md
bodies on demand.

- `_corpus`   — the hint corpus (adapted from skill-note-hint, MIT).
- `_point`    — the standalone `point()` pointer entrypoint.
- `_toolset`  — the `SkillToolset` (list_skills / load_skill / point_to_skill).
- `_capability` — the `Skills` capability (wraps the toolset).

Re-homed so paf owns the surface and wires it through the capability-toolset
seam (registry + profile + builder); the standalone skill-note-hint repo stays
independent for the Hermes-plugin path.
"""

from __future__ import annotations

from ._capability import Skills
from ._toolset import SkillToolset

__all__ = ["Skills", "SkillToolset"]
