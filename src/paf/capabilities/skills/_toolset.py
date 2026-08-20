"""The paf-owned Skills toolset — list, load, and point-to-skill.

One toolset for all skill functionality (Landon 2026-08-19): fold the
pydantic-ai load-on-demand idea (``load_skill`` pulls a SKILL.md body when
relevant, so the prompt stays thin) together with the skill-note-hint pointer
engine (``point_to_skill`` deterministically pre-arms *which* skill a message
needs — the trigger neither the harness nor a plain list tool provides).

Tools:
  - ``list_skills``  — names + descriptions of every indexed skill (a menu).
  - ``load_skill``   — full SKILL.md body for one skill, on demand.
  - ``point_to_skill`` — BM25 pointer: `consider: <skill>` for a user message.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from pydantic_ai import ModelRetry
from pydantic_ai.toolsets import FunctionToolset

from ._corpus import HintCorpus, SkillDoc

_MAX_DESCRIPTION_CHARS = 240  # keep list_skills output tight


class SkillToolset(FunctionToolset[object]):
    """List / load skills and point the model at the relevant one.

    Indexes a skill library directory once at construction (name + description +
    tags vocabulary) and exposes the three tools above. ``load_skill`` re-reads
    the SKILL.md body from disk on demand so the model only pays for a skill's
    instructions when it actually needs them.
    """

    def __init__(self, libraries: list[str] | str, *, top_k: int = 3) -> None:
        if isinstance(libraries, str):
            libraries = [libraries]
        super().__init__()
        self.libraries = list(libraries)
        self.top_k = top_k
        self._corpus = HintCorpus(roots=self.libraries)
        self._by_name: dict[str, SkillDoc] = {d.name: d for d in self._corpus.docs}

        # Register the method-tools explicitly (FunctionToolset pops `.tools`
        # from `add_function`, not from plain subclass methods — same pattern as
        # the FileSystem toolset).
        self.add_function(self.list_skills, name="list_skills")
        self.add_function(self.load_skill, name="load_skill")
        self.add_function(self.point_to_skill, name="point_to_skill")

    # -- helpers -----------------------------------------------------------

    def _find(self, skill_id: str) -> SkillDoc | None:
        # exact name first, then normalized alias (case/space-insensitive)
        if skill_id in self._by_name:
            return self._by_name[skill_id]
        norm = skill_id.strip().lower().replace("_", "-")
        for doc in self._corpus.docs:
            if doc.name == norm or doc.name.replace("_", "-") == norm:
                return doc
        return None

    # -- tools -------------------------------------------------------------

    def list_skills(self) -> str:
        """List the skills available to load, each as `name: <one-line description>`.

        Use this to see what skills exist before deciding which to load. The
        library is indexed at construction; a skill dropped in later appears
        only after a rebuild.
        """
        if not self._corpus.docs:
            return "No skills or notes indexed in the configured library."
        lines = []
        for doc in self._corpus.docs:
            desc = doc.description.split("\n")[0].strip()
            if len(desc) > _MAX_DESCRIPTION_CHARS:
                desc = desc[:_MAX_DESCRIPTION_CHARS - 1] + "…"
            lines.append(f"- {doc.name}: {desc}")
        return "\n".join(lines)

    def load_skill(self, skill_id: str) -> str:
        """Load the full instructions/body of one skill by name.

        Returns the SKILL.md body (title + content) for the named skill so its
        procedure or reference material is available in context. Only load a
        skill you actually intend to use — loading pulls its full body into
        context. Raises a model retry if the skill is unknown.
        """
        doc = self._find(skill_id)
        if doc is None:
            known = ", ".join(sorted(self._by_name)) or "(none)"
            raise ModelRetry(
                f"Unknown skill {skill_id!r}. Available skills: {known}. "
                "Call list_skills first."
            )
        if doc.kind == "note":
            raise ModelRetry(
                f"{doc.name!r} is a note, not a skill. Use point_to_skill for "
                "notes; load_skill loads SKILL.md procedures."
            )
        path = Path(doc.ref) if doc.ref else None
        if path is None or not path.is_file():
            raise ModelRetry(f"Skill {doc.name!r} has no readable SKILL.md on disk ({doc.ref!r}).")
        body = path.read_text(encoding="utf-8", errors="ignore")
        return textwrap.dedent(body).strip()

    def point_to_skill(self, query: str, top_k: int | None = None) -> str:
        """Given a user message or task, return pointer hints to the most relevant skill.

        Deterministic BM25 over the skill library's declared vocabulary
        (name/description/tags) — zero LLM cost. Returns one line per hit like
        `consider: <skill>` (or `note: <name>`), most relevant first. A pointer
        names *where to look*, not the answer; follow it by calling load_skill
        to pull the full instructions. Fire this when a task matches a known
        procedure but the model is not already holding that skill.
        """
        k = top_k if top_k is not None else self.top_k
        hits = self._corpus.rank(query, k=k)
        if not hits:
            return (
                f"No skills in the library overlap {query!r}. "
                "Consider adding a SKILL.md with matching tags/description."
            )
        lines = [f"{doc.hint}  # {score:.2f}" for doc, score in hits]
        return "\n".join(lines)
