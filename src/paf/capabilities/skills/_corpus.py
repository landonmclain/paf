"""Corpus ingestion for the paf Skills capability (v2 — tree-aware).

Adapted from `skill_note_hint.corpus` (MIT) so the pointer engine lives inside
paf as a first-class capability. v2 reworks the corpus to model **skills as a
tree** (the Hermes/Claude shape) instead of a flat skill/note dump (Landon
2026-09-08; see vault `Skills-Capability-Spec.md`):

- A **skill** is a directory containing a `SKILL.md`, optionally under a
  *category* directory: `<root>/<category>/<skill>/SKILL.md` (or flat
  `<root>/<skill>/SKILL.md`, category ``""``).
- A skill's **resources** — files under `references/`, `scripts/`, `templates/`,
  `assets/` — are attached to that skill, NOT indexed as standalone notes.
- The corpus indexes **skills only**. ``rank()`` BM25s over skill vocabulary
  (name + description + tags), so the pointer engine (`point_skills`) never
  scores against a polluted note dump (the v1 failure mode: pointed at a real
  tree it indexed 480 docs — 130 skills + 350 junk notes).

Ranking reuses paf's own dependency-free BM25 core
(`paf.capabilities.search._ranking`) — paf owns exactly one BM25.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from paf.capabilities.search._ranking import bm25_rank, tokenize

__all__ = [
    "SkillDoc",
    "SkillResource",
    "HintCorpus",
    "parse_frontmatter",
    "tokenize",
]

# Recognized resource subdirectories under a skill root, mapped to a kind.
_RESOURCE_DIRS: dict[str, Literal["reference", "script", "template", "asset"]] = {
    "references": "reference",
    "scripts": "script",
    "templates": "template",
    "assets": "asset",
}


@dataclass
class SkillResource:
    """One support file belonging to a skill (references/, scripts/, ...)."""

    relative_path: str
    """Path relative to the skill root, e.g. ``references/x.md``."""
    kind: Literal["reference", "script", "template", "asset", "other"]
    """Resource class, inferred from the enclosing subdir."""


@dataclass
class SkillDoc:
    """One indexable skill in the tree.

    ``text`` is what BM25 scores (name + description + tags, lowercased);
    ``hint`` is the one-line pointer injected (``consider: <skill_id>``);
    ``skill_file`` is the on-disk SKILL.md so `load_skill` can read the body;
    ``skill_root`` is the skill's directory; ``resources`` are its support files.
    """

    skill_id: str
    """Canonical id: ``<category>/<name>`` (or ``<name>`` when flat at root)."""
    name: str
    """Short name — the skill directory's basename."""
    category: str
    """Category path component (``""`` for a flat, root-level skill)."""
    skill_root: Path
    """Directory containing SKILL.md."""
    skill_file: Path
    """The SKILL.md itself."""
    description: str = ""
    text: str = ""
    kind: str = "skill"
    hint: str = ""
    resources: tuple[SkillResource, ...] = field(default_factory=tuple)

    def __repr__(self) -> str:
        return f"SkillDoc(skill_id={self.skill_id!r}, category={self.category!r})"


def parse_frontmatter(path: Path) -> dict:
    """Parse the ``--- ... ---`` YAML frontmatter of a file into a flat dict.

    Handles simple scalar values, quoted strings, inline `[a, b]` lists, and
    block-style `- item` lists (flattened to space-joined strings where it
    matters). Nested dicts are collapsed: ``a.b`` keys. Returns {} if there is
    no frontmatter block.
    """
    text = path.read_text(encoding="utf-8", errors="ignore")
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    block = text[4:end]

    fm: dict = {}
    current_list_key: str | None = None
    list_items: list[str] = []

    def _flush_list() -> None:
        nonlocal current_list_key, list_items
        if current_list_key is not None and list_items:
            fm[current_list_key] = " ".join(list_items)
        current_list_key = None
        list_items = []

    for raw in block.splitlines():
        line = raw.strip()
        if not line:
            _flush_list()
            continue
        if line.startswith("- "):
            if current_list_key is not None:
                list_items.append(line[2:].strip().strip('"\' '))
            continue
        if ":" not in line:
            _flush_list()
            continue
        _flush_list()
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if not val:  # block-style list starts here
            current_list_key = key
            continue
        # inline list [a, b, c]
        if val.startswith("[") and val.endswith("]"):
            inner = val[1:-1]
            items = [t.strip().strip('"\' ') for t in inner.split(",") if t.strip()]
            fm[key] = " ".join(items)
            continue
        fm[key] = val.strip('"\' ')

    _flush_list()
    return fm


def _collect_resources(skill_root: Path) -> tuple[SkillResource, ...]:
    """Gather a skill's support files under references//scripts//templates//assets/.

    Only recognized resource subdirs are included; files directly in the skill
    root (e.g. the SKILL.md itself) are not resources. Returns an empty tuple
    when the skill has no resource subdirs.
    """
    resources: list[SkillResource] = []
    for subdir, kind in _RESOURCE_DIRS.items():
        base = skill_root / subdir
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file():
                resources.append(
                    SkillResource(
                        relative_path=f"{subdir}/{p.relative_to(base)}",
                        kind=kind,
                    )
                )
    return tuple(resources)


class HintCorpus:
    """Tiny, swappable, tree-aware corpus of skills. Scanned once on init, cached.

    Indexes **skills only** (SKILL.md files), each with its category, canonical
    id, and attached resources. Plain .md files are NOT indexed — the
    standalone-note pointer is a vault-capability concern, not Skills (see the
    vault spec).
    """

    def __init__(self, roots: list[str] | str) -> None:
        if isinstance(roots, str):
            roots = [roots]
        self.roots = [Path(r) for r in roots]
        self.docs: list[SkillDoc] = []
        # Canonical id -> doc (unique) and short name -> docs (may be several).
        self.by_id: dict[str, SkillDoc] = {}
        self.by_short_name: dict[str, list[SkillDoc]] = {}
        self._load()

    def _load(self) -> None:
        self.docs.clear()
        self.by_id.clear()
        self.by_short_name.clear()
        # Resolve each library root once so flat-vs-nested category detection is
        # robust regardless of how the spec spelled the path.
        resolved_roots = {r.resolve() for r in self.roots if r.exists()}
        for root in self.roots:
            if not root.exists():
                continue
            for p in root.rglob("SKILL.md"):
                doc = self._ingest(root, p, resolved_roots)
                if doc is None:
                    continue
                self.docs.append(doc)
                self.by_id[doc.skill_id] = doc
                self.by_short_name.setdefault(doc.name, []).append(doc)

    def _ingest(self, root: Path, skill_file: Path, resolved_roots: set[Path]) -> SkillDoc | None:
        skill_root = skill_file.parent
        name = skill_root.name
        # Category: the parent dir of the skill root, when that parent is not a
        # library root itself (i.e. the skill is nested, not flat at root).
        parent = skill_root.parent
        category = "" if parent.resolve() in resolved_roots else parent.name
        skill_id = f"{category}/{name}" if category else name
        fm = parse_frontmatter(skill_file)
        description = fm.get("description", "")
        tags = fm.get("tags", "")
        if not tags and fm.get("metadata.hermes.tags"):
            tags = fm["metadata.hermes.tags"]
        text = f"{name} {description} {tags}".strip().lower()
        if not text:
            return None
        resources = _collect_resources(skill_root)
        return SkillDoc(
            skill_id=skill_id,
            name=name,
            category=category,
            skill_root=skill_root,
            skill_file=skill_file,
            description=description,
            text=text,
            hint=f"consider: {skill_id}",
            resources=resources,
        )

    def refresh(self) -> None:
        """Re-scan library roots from disk, rebuilding the in-memory indexes.

        Called after `create_skill` so a newly written skill is immediately
        indexable (list/load/point) without reconstructing the toolset.
        """
        self._load()

    def rank(self, query: str, k: int) -> list[tuple[SkillDoc, float]]:
        texts = [d.text for d in self.docs]
        if not texts:
            return []
        hits = bm25_rank(query, texts)
        return [(self.docs[idx], score) for idx, score in hits[:k]]
