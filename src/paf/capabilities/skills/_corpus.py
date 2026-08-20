"""Corpus ingestion for the paf Skills capability.

Adapted from `skill_note_hint.corpus` (MIT, https://github.com/...skill-note-hint)
so the pointer engine lives inside paf as a first-class capability instead of a
foreign standalone package. This is the "copy skill-note-hint functionality into
a custom skills capability" direction Landon signed off on (2026-08-19).

A tiny, flat, swappable corpus of hint docs. It scans a set of root directories
for knowledge files (`SKILL.md` skills and plain `.md` notes), parses their YAML
frontmatter into a small indexable text surface, and exposes a `HintCorpus` that
can be ranked against a user query. Only stdlib.

Ranking reuses paf's own dependency-free BM25 core
(`paf.capabilities.search._ranking`) rather than vendoring a second copy — paf
owns exactly one BM25.
"""

from __future__ import annotations

from pathlib import Path

from paf.capabilities.search._ranking import bm25_rank, tokenize

__all__ = ["SkillDoc", "HintCorpus", "parse_frontmatter", "tokenize"]


class SkillDoc:
    """One indexable hint source.

    ``text`` is what BM25 scores (name + description + tags, lowercased);
    ``hint`` is the one-line pointer injected (e.g. ``consider: skill-x``);
    ``ref`` is the on-disk path so ``load_skill`` can pull the full body.
    """

    def __init__(
        self,
        name: str,
        kind: str = "skill",
        text: str = "",
        hint: str | None = None,
        ref: str | None = None,
        description: str = "",
    ) -> None:
        self.name = name
        self.kind = kind  # 'skill' | 'note'
        self.text = text
        self.hint = hint or f"{'consider' if kind == 'skill' else 'note'}: {name}"
        self.ref = ref
        self.description = description

    def __repr__(self) -> str:
        return f"SkillDoc(name={self.name!r}, kind={self.kind!r}, hint={self.hint!r})"


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


class HintCorpus:
    """Tiny, flat, swappable corpus of hint docs. Scanned once on init, cached."""

    def __init__(self, roots: list[str] | str) -> None:
        if isinstance(roots, str):
            roots = [roots]
        self.roots = [Path(r) for r in roots]
        self.docs: list[SkillDoc] = []
        self._load()

    def _load(self) -> None:
        self.docs.clear()
        for root in self.roots:
            if not root.exists():
                continue
            # skills: any SKILL.md under the root (handles nested category dirs)
            for p in root.rglob("SKILL.md"):
                self._ingest(p, kind="skill")
            # notes: plain .md files that AREN'T SKILL.md
            for p in root.rglob("*.md"):
                if p.name == "SKILL.md":
                    continue
                self._ingest(p, kind="note")

    def _ingest(self, path: Path, kind: str) -> None:
        fm = parse_frontmatter(path)
        name = fm.get("name") or path.stem
        description = fm.get("description", "")
        tags = fm.get("tags", "")
        # fold nested metadata.hermes.tags if top-level tags absent
        if not tags and fm.get("metadata.hermes.tags"):
            tags = fm["metadata.hermes.tags"]
        text = f"{name} {description} {tags}".strip().lower()
        if not text:
            return
        ref = str(path)
        hint = f"consider: {name}" if kind == "skill" else f"note: {path.stem}"
        self.docs.append(
            SkillDoc(
                name=name,
                kind=kind,
                text=text,
                hint=hint,
                ref=ref,
                description=description,
            )
        )

    def rank(self, query: str, k: int) -> list[tuple[SkillDoc, float]]:
        texts = [d.text for d in self.docs]
        hits = bm25_rank(query, texts)
        return [(self.docs[idx], score) for idx, score in hits[:k]]
