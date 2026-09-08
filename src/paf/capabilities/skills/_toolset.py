"""The paf-owned Skills toolset (v2) — list, load, and read skill resources.

v2 rework (Landon 2026-09-08) makes the Skills capability handle
**Hermes/Claude-style skill trees**: skills grouped by category, with
skill-scoped resources (`references/`, `scripts/`, `templates/`, `assets/`)
that belong to their skill rather than polluting a flat note corpus. See the
vault `Skills-Capability-Spec.md`.

Tools (agent-facing):
  - ``list_skills``    — the skill tree: category-grouped names + descriptions.
  - ``load_skill``     — full SKILL.md body (plus a resource manifest) on demand.
  - ``list_skill_files`` — enumerate one skill's support resources.
  - ``read_skill_file`` — read one resource, confined to its skill's directory.

Internal (NOT agent tools):
  - ``point_skills``   — BM25 pointer engine (`consider: <skill_id>`), called by
    the auto per-turn hint layer, not offered to the model.
"""

from __future__ import annotations

import textwrap

from pydantic_ai import ModelRetry
from pydantic_ai.toolsets import FunctionToolset

from ._corpus import HintCorpus, SkillDoc

_MAX_DESCRIPTION_CHARS = 240  # keep list_skills output tight
_READ_MAX_CHARS = 16_000  # read_skill_file content bound (spill beyond this)


class SkillToolset(FunctionToolset[object]):
    """List / load skills (tree-aware) and read skill resources.

    Indexes one or more skill-library directories once at construction as a
    **tree** (category -> skills, each with attached resources). ``load_skill``
    re-reads the SKILL.md body from disk on demand so the model only pays for a
    skill's instructions when it actually needs them; ``read_skill_file`` pulls a
    resource confined to its skill's directory.
    """

    def __init__(self, libraries: list[str] | str, *, top_k: int = 3) -> None:
        if isinstance(libraries, str):
            libraries = [libraries]
        super().__init__()
        self.libraries = list(libraries)
        self.top_k = top_k
        self._corpus = HintCorpus(roots=self.libraries)
        self._by_id: dict[str, SkillDoc] = self._corpus.by_id
        self._by_short_name: dict[str, list[SkillDoc]] = self._corpus.by_short_name

        # Register the method-tools explicitly (FunctionToolset pops `.tools`
        # from `add_function`, not from plain subclass methods — same pattern as
        # the FileSystem toolset). Note: `point_skills` (the BM25 pointer engine)
        # is deliberately NOT registered — it is an internal capability
        # primitive for the auto per-turn hint layer, not an agent-facing tool
        # (see vault Skills-Capability-Spec §6).
        self.add_function(self.list_skills, name="list_skills")
        self.add_function(self.load_skill, name="load_skill")
        self.add_function(self.list_skill_files, name="list_skill_files")
        self.add_function(self.read_skill_file, name="read_skill_file")

    # -- helpers -----------------------------------------------------------

    def _find(self, skill_id: str) -> SkillDoc | None:
        """Resolve a skill by canonical id first, then an unambiguous short name.

        Canonical: `category/name` (or bare `name` at root). Falls back to a
        case/space-insensitive short-name match ONLY when it names exactly one
        skill (a duplicate short name across categories is ambiguous → None).
        """
        if skill_id in self._by_id:
            return self._by_id[skill_id]
        norm = skill_id.strip().lower().replace("_", "-")
        # Try canonical-id form with normalization (category/name).
        for doc in self._corpus.docs:
            if doc.skill_id.replace("_", "-").lower() == norm:
                return doc
        # Short-name fallback only when unambiguous.
        matching = [
            doc for doc in self._corpus.docs if doc.name.replace("_", "-").lower() == norm
        ]
        if len(matching) == 1:
            return matching[0]
        return None

    @staticmethod
    def _resource_manifest(doc: SkillDoc) -> str:
        """Compact resource listing for a skill's support files."""
        if not doc.resources:
            return ""
        lines = ["Available resources:"]
        for r in doc.resources:
            lines.append(f"- {r.relative_path}  # {r.kind}")
        return "\n".join(lines)

    # -- tools -------------------------------------------------------------

    def list_skills(self) -> str:
        """List the available skills as a category tree.

        Groups every indexed skill under its category (path-derived; skills at
        the library root appear under `(root)`). Each line is
        `  - <name>: <one-line description>`. Use this to see what skills exist
        and where they live before deciding which to load. The library is
        indexed at construction; a skill dropped in later appears only after a
        rebuild.
        """
        if not self._corpus.docs:
            return "No skills indexed in the configured library."
        # Group by category, preserving first-seen order for stable output.
        by_cat: dict[str, list[SkillDoc]] = {}
        for doc in self._corpus.docs:
            cat = doc.category or "(root)"
            by_cat.setdefault(cat, []).append(doc)

        lines = []
        for cat in sorted(by_cat):
            lines.append(f"{cat}/")
            for doc in sorted(by_cat[cat], key=lambda d: d.name):
                desc = doc.description.split("\n")[0].strip()
                if len(desc) > _MAX_DESCRIPTION_CHARS:
                    desc = desc[:_MAX_DESCRIPTION_CHARS - 1] + "…"
                lines.append(f"  - {doc.name}: {desc}")
        return "\n".join(lines)

    def load_skill(self, skill_id: str) -> str:
        """Load the full instructions/body of one skill by id.

        Returns the SKILL.md body (title + content) for the named skill so its
        procedure or reference material is available in context. A compact
        **resource manifest** (its `references/`, `scripts/`, `templates/`,
        `assets/` files) is appended so you can read them on demand with
        `read_skill_file`. Only load a skill you actually intend to use —
        loading pulls its full body into context. Raises a model retry if the
        skill is unknown.
        """
        doc = self._find(skill_id)
        if doc is None:
            known = ", ".join(sorted(self._by_id)) or "(none)"
            raise ModelRetry(
                f"Unknown skill {skill_id!r}. Available skills: {known}. "
                "Call list_skills first. For skills in a category, use the "
                "full id like `category/name`."
            )
        if not doc.skill_file.is_file():
            raise ModelRetry(
                f"Skill {doc.skill_id!r} has no readable SKILL.md on disk ({doc.skill_file!r})."
            )
        body = doc.skill_file.read_text(encoding="utf-8", errors="ignore")
        manifest = self._resource_manifest(doc)
        if manifest:
            return f"{textwrap.dedent(body).strip()}\n\n{manifest}"
        return textwrap.dedent(body).strip()

    def list_skill_files(self, skill_id: str) -> str:
        """List the support resources attached to one skill (references/scripts/etc.).

        Returns one relative path per line under the skill's `references/`,
        `scripts/`, `templates/`, `assets/` subdirs. Use this to see what a
        skill's support files are before reading one with `read_skill_file`.
        """
        doc = self._find(skill_id)
        if doc is None:
            known = ", ".join(sorted(self._by_id)) or "(none)"
            raise ModelRetry(
                f"Unknown skill {skill_id!r}. Available skills: {known}. "
                "Use the full id like `category/name`."
            )
        if not doc.resources:
            return f"Skill {doc.skill_id!r} has no support resources."
        return "\n".join(r.relative_path for r in doc.resources)

    def read_skill_file(self, skill_id: str, relative_path: str) -> str:
        """Read one support file of a skill, confined to that skill's directory.

        Returns the file's content bounded to a size cap (oversized files are
        truncated with a pointer). Reads treat scripts as readable text — they
        are never executed here. Rejects any path that escapes the skill's
        directory (`../`, absolute paths, symlink escapes).
        """
        doc = self._find(skill_id)
        if doc is None:
            known = ", ".join(sorted(self._by_id)) or "(none)"
            raise ModelRetry(
                f"Unknown skill {skill_id!r}. Available skills: {known}. "
                "Use the full id like `category/name`."
            )
        # Resolve within the skill root only; rejects traversal + abs paths.
        skill_root = doc.skill_root.resolve()
        try:
            target = (skill_root / relative_path).resolve()
        except (OSError, RuntimeError) as err:
            raise ModelRetry(
                f"Cannot resolve {relative_path!r} for skill {doc.skill_id!r}."
            ) from err
        if not target.is_relative_to(skill_root):
            raise ModelRetry(
                f"Path {relative_path!r} escapes skill {doc.skill_id!r}'s directory."
            )
        # Only files actually attached as resources are readable.
        allowed = {r.relative_path for r in doc.resources}
        rel = target.relative_to(skill_root)
        if str(rel) not in allowed:
            raise ModelRetry(
                f"{rel} is not a resource of {doc.skill_id!r}. "
                "Use list_skill_files to see available resources."
            )
        if not target.is_file():
            raise ModelRetry(f"{rel} is not a readable file.")
        content = target.read_text(encoding="utf-8", errors="ignore")
        if len(content) > _READ_MAX_CHARS:
            content = content[:_READ_MAX_CHARS] + f"\n… [truncated at {_READ_MAX_CHARS} chars]"
        return f"{rel}\n\n{content}"

    def point_skills(self, query: str, top_k: int | None = None) -> str:
        """INTERNAL primitive (NOT an agent tool): BM25 pointer hints for the hint layer.

        Deterministic BM25 over the skill library's declared vocabulary
        (name/description/tags) — zero LLM cost. Returns one line per hit like
        `consider: <skill_id>` (canonical, category-qualified), most relevant
        first. A pointer names *where to look*, not the answer; the auto per-turn
        hint layer (Skills-Capability-Spec §6) calls this and injects
        `consider:` lines into turn context; the agent then follows up with
        load_skill / read_skill_file. Deliberately NOT in `.tools` — it is not a
        tool the model should call on its own volition.
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
