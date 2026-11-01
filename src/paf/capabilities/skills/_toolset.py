"""The paf-owned Skills toolset — list, load, and read skill resources.

The Skills capability handles **category-grouped skill trees**: skills
grouped by category, with skill-scoped resources (`references/`, `scripts/`,
`templates/`, `assets/`) that belong to their skill rather than polluting a flat
note corpus.

Tools (agent-facing):
  - ``list_skills``    — the skill tree: category-grouped names + descriptions.
  - ``load_skill``     — full SKILL.md body (plus a resource manifest) on demand.
  - ``list_skill_files`` — enumerate one skill's support resources.
  - ``read_skill_file`` — read one resource, confined to its skill's directory.
  - ``create_skill``   — create a new skill (SKILL.md) with valid frontmatter
    in the configured library (mirrors vault ``note_create``).
  - ``edit_skill``     — CAS conflict-safe edit of a skill's SKILL.md (mirrors
    vault ``note_edit``; paired with ``create_skill``).
"""

from __future__ import annotations

import hashlib
import textwrap
from pathlib import Path

from pydantic_ai import ModelRetry
from pydantic_ai.toolsets import FunctionToolset

from paf.capabilities._edit import apply_edits
from paf.capabilities._storage._atomic import write_atomic as _write_atomic

from ._corpus import HintCorpus, SkillDoc
from ._store import FlatFileSkillsStorage

# Recovery-prone write/edit tools get real retry budget (not pydantic-ai's
# default of 1). Mirrors fs `_FS_TOOL_RETRIES` and vault `_VAULT_TOOL_RETRIES`.
_SKILL_TOOL_RETRIES = 5

_MAX_DESCRIPTION_CHARS = 240  # keep list_skills output tight
_READ_MAX_CHARS = 16_000  # read_skill_file content bound (spill beyond this)


def _content_hash(content: str) -> str:
    """Short content hash — the CAS version token for skill writes.

    Mirrors ``fs._content_hash`` / vault ``_content_hash`` (sha256 truncated to
    12 hex chars): opaque, cheap, stable for a given SKILL.md body. ``load_skill``
    reports it so the model can pass it back as ``expected_version`` to
    ``skill_edit`` for conflict-safe (optimistic-concurrency) writes.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]


class SkillToolset(FunctionToolset[object]):
    """List / load skills (tree-aware) and read skill resources.

    Indexes one or more skill-library directories once at construction as a
    **tree** (category -> skills, each with attached resources). ``load_skill``
    re-reads the SKILL.md body from disk on demand so the model only pays for a
    skill's instructions when it actually needs them; ``read_skill_file`` pulls a
    resource confined to its skill's directory.
    """

    def __init__(
        self,
        libraries: list[str] | str,
        *,
        storage: FlatFileSkillsStorage | None = None,
    ) -> None:
        if isinstance(libraries, str):
            libraries = [libraries]
        super().__init__()
        self.libraries = list(libraries)
        self._storage = storage if storage is not None else FlatFileSkillsStorage()
        self._corpus = HintCorpus(roots=self.libraries, storage=self._storage)
        self._by_id: dict[str, SkillDoc] = self._corpus.by_id
        self._by_short_name: dict[str, list[SkillDoc]] = self._corpus.by_short_name

        # Register the method-tools explicitly (FunctionToolset pops `.tools`
        # from `add_function`, not from plain subclass methods — same pattern as
        # the FileSystem toolset).
        self.add_function(self.list_skills, name="list_skills")
        self.add_function(self.load_skill, name="load_skill")
        self.add_function(self.list_skill_files, name="list_skill_files")
        self.add_function(self.read_skill_file, name="read_skill_file")
        self.add_function(
            self.create_skill, name="create_skill", retries=_SKILL_TOOL_RETRIES
        )
        self.add_function(
            self.edit_skill, name="edit_skill", retries=_SKILL_TOOL_RETRIES
        )

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
            doc
            for doc in self._corpus.docs
            if doc.name.replace("_", "-").lower() == norm
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
                    desc = desc[: _MAX_DESCRIPTION_CHARS - 1] + "…"
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
        version = _content_hash(body)
        manifest = self._resource_manifest(doc)
        head = f"[skill: {doc.skill_id}  version: {version}]"
        if manifest:
            return f"{head}\n\n{textwrap.dedent(body).strip()}\n\n{manifest}"
        return f"{head}\n\n{textwrap.dedent(body).strip()}"

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
            content = (
                content[:_READ_MAX_CHARS]
                + f"\n… [truncated at {_READ_MAX_CHARS} chars]"
            )
        return f"{rel}\n\n{content}"

    def create_skill(
        self,
        category: str,
        name: str,
        *,
        description: str,
        tags: str = "",
        content: str = "",
    ) -> str:
        """Create a new skill (a SKILL.md) in the configured skill library.

        Mirrors the vault capability's `note_create`: the capability owns the
        *domain conventions* of a skill (valid frontmatter schema, the
        `<category>/<name>/SKILL.md` layout, canonical name) instead of making
        the model re-derive them with raw filesystem writes. Writes live **only**
        inside the configured library (never outside a library root). Refuses to
        overwrite an existing skill.

        Args:
            category: Skill category dir (may be ``""``/empty for a root-level
                skill). Creates the category if it does not exist.
            name: Skill directory name — becomes the canonical short name and
                the ``name`` frontmatter field.
            description: One-line ``description`` frontmatter (what the skill
                does). Should not contain a newline.
            tags: Optional space- or comma-separated tags (``description.regex``
                style); rendered into the frontmatter ``tags`` list. May be empty.
            content: Optional body text below the frontmatter. When empty a
                minimal ``# Skill: <name>`` stub is written.

        Returns a confirmation with the canonical id and the on-disk path. The
        new skill is immediately listable/loadable.
        """
        if not name or not name.strip():
            raise ModelRetry("create_skill requires a non-empty skill `name`.")
        if not description or not description.strip():
            raise ModelRetry("create_skill requires a `description` (one line).")
        for part in (name, category):
            if part and ("/" in part or "\\" in part or part in {".", ".."}):
                raise ModelRetry(
                    f"Skill/category {part!r} must be a single directory name "
                    "(no separators or traversal)."
                )
        if not self.libraries:
            raise ModelRetry("No skill library is configured — nothing to create into.")

        # Write into the FIRST configured library root (the default authoring target).
        library_root = Path(self.libraries[0]).expanduser().resolve()
        dest = (
            library_root / category / name if category else library_root / name
        ).resolve()
        if not dest.is_relative_to(library_root):
            raise ModelRetry(
                f"Category/name {category!r}/{name!r} escapes the library root."
            )
        target = dest / "SKILL.md"
        if target.exists():
            raise ModelRetry(
                f"Skill {name!r} already exists at {target}. Use a different name, "
                "or edit it with the filesystem tools."
            )

        tags_clean = " ".join(tags.replace(",", " ").split()) if tags else ""
        tags_line = f"tags: [{tags_clean}]" if tags_clean else "tags: []"
        description_line = description.strip().replace("\n", " ")
        body = content.strip()
        if not body:
            body = f"# Skill: {name}\n\n(*stub — add the skill body here.*)"
        skill_md = (
            "---\n"
            f"name: {name}\n"
            f"description: {description_line}\n"
            f"{tags_line}\n"
            "---\n"
            f"\n{body}\n"
        )

        dest.mkdir(parents=True, exist_ok=True)
        target.write_text(skill_md, encoding="utf-8")
        self._corpus.refresh()
        skill_id = f"{category}/{name}" if category else name
        return f"Created skill {skill_id!r} at {target} (category={category or '(root)'!r})."

    def edit_skill(
        self,
        skill_id: str,
        edits: list[dict],
        *,
        expected_version: str | None = None,
    ) -> str:
        """Edit a skill's SKILL.md by exact string replacement, with conflict detection.

        Mirrors the vault capability's `note_edit` (paired with our
        `create_skill`): ``edits`` is a list of ``{"old_text", "new_text"}``
        pairs, applied in one atomic write. Each ``old_text`` must appear
        exactly once — include enough surrounding context to make it unique
        (zero or multiple matches is an error, so you never edit blindly). Pass
        multiple pairs to batch independent changes to the same skill in one call.

        ``expected_version`` is the skill's content version reported by
        ``load_skill`` (the ``version:`` in its header). If provided, the edit is
        REJECTED when the skill's current version has changed — someone (e.g. a
        human) edited it since you read it. Re-load the skill and retry with the
        fresh version. Returns a confirmation with the new version and a per-pair
        diff. Writes atomically and refreshes the corpus so the change is
        immediately reflected in list/load.

        Args:
            skill_id: Canonical skill id (`category/name`) or unique short name.
            edits: ``[{old_text, new_text}, ...]`` pairs to apply, in order.
            expected_version: Optional version token from `load_skill` for CAS.

        Returns a confirmation with the new version + per-pair diffs; raises
        ModelRetry on any failure (never reports success for an unpersisted edit).
        """
        doc = self._find(skill_id)
        if doc is None:
            known = ", ".join(sorted(self._by_id)) or "(none)"
            raise ModelRetry(
                f"Unknown skill {skill_id!r}. Available skills: {known}. "
                "Use the full id like `category/name`."
            )
        if not doc.skill_file.is_file():
            raise ModelRetry(
                f"Skill {doc.skill_id!r} has no readable SKILL.md on disk ({doc.skill_file!r})."
            )

        result = apply_edits(
            doc.skill_file,
            edits,
            expected_version=expected_version,
            read=lambda p: p.read_text(encoding="utf-8", errors="replace"),
            write=_write_atomic,
        )
        self._corpus.refresh()
        return result.summary(doc.skill_id)
