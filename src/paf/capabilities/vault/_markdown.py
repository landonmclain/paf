"""Markdown helpers for the vault capability — frontmatter, wikilinks, sections.

Small, stdlib-only, deterministic. We only need *enough* markdown awareness to
build the note index and ladder — NOT a full parser:

- ``parse_frontmatter(text)``  -> (meta: dict, body: str)  — split + YAML-parse the
  leading ``---`` block; malformed frontmatter degrades to (``{}``, whole text).
- ``extract_wikilinks(text)``  -> list[str]  — every ``[[Target]]`` / ``[[Target|alias]]``
  link, Target normalized (alias stripped, whitespace kept — Obsidian matches on
  the note *name*, case-insensitive).
- ``resolve_note``             -> the note whose basename (case-insensitive) matches a
  target, or None. Solves ``[[Note Name]]`` -> path in a vault, so the model never
  has to guess a vault path.
- ``section_after``            -> the body text of a note's section under a heading
  (for block-level targeting).

No wikilink *interpretation* (Obsidian rendering), no AST, no embeddings. This is
exactly enough to index and navigate a vault of notes.
"""

from __future__ import annotations

import re
from pathlib import Path

try:
    import yaml
except Exception:  # pragma: no cover - yaml is a hard dep of paf
    yaml = None  # type: ignore[assignment]

_FRONTMATTER_RE = re.compile(r"\A\s*---\s*\n(.*?)\n\s*---\s*\n?", re.DOTALL)
_WIKILINK_RE = re.compile(r"\[\[([^\[\]|]+)((?:\|[^\]]+)?)\]\]")


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split a markdown doc into (frontmatter meta dict, body).

    Recognizes a leading ``---`` YAML block. Returns ``({}, text)`` when there is
    no frontmatter or the block fails to parse — a note without frontmatter is not
    an error.
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    raw = m.group(1)
    body = text[m.end() :]
    meta: dict = {}
    if yaml is not None:
        try:
            parsed = yaml.safe_load(raw)
            if isinstance(parsed, dict):
                meta = {k: v for k, v in parsed.items() if isinstance(k, str)}
        except yaml.YAMLError:
            meta = {}
    return meta, body


def extract_wikilinks(text: str) -> list[str]:
    """Every distinct ``[[Target]]`` link target in a note body, in order.

    Normalizes by stripping the ``|alias`` and trimming whitespace; Obsidian
    resolves links by note *name*, so that is what we index. Drops empty targets.
    """
    out: list[str] = []
    for m in _WIKILINK_RE.finditer(text):
        target = m.group(1).strip()
        if target:
            out.append(target)
    return out


def _index_key(name: str) -> str:
    """Normalize a note name for matching: lowercased basename without ``.md``."""
    return Path(name).stem.lower()


def rewrite_wikilink_target(text: str, old_target: str, new_target: str) -> str:
    """Rewrite every ``[[old_target]]``/``[[old_target|alias]]`` link to ``new_target``.

    Matches the link target case-insensitively against ``old_target`` (Obsidian
    resolves links by note name, case-insensitive). ``[[Old|alias]]`` keeps its
    ``|alias`` (only the target half changes). Used by ``note_rename`` to update
    inbound wikilinks when a note is moved. The plain ``[[old_target]]`` (no
    alias) becomes ``[[new_target]]``.
    """
    if not old_target or old_target == new_target:
        return text

    def _sub(m: re.Match) -> str:
        target = m.group(1)
        alias = m.group(2)  # may be None
        if target.lower() != old_target.lower():
            return m.group(0)
        return f"[[{new_target}{alias or ''}]]"

    return _WIKILINK_RE.sub(_sub, text)


def rename_note_title(text: str, old_stem: str, new_stem: str) -> str:
    """Update a note's frontmatter ``title:`` to its new stem after a rename.

    Only rewrites when the frontmatter ``title:`` equals the old stem (i.e. it
    was auto-derived from the filename, not a hand-set display name). A *custom*
    ``title:`` that differs from the old stem is left untouched — the user set
    it deliberately and a rename shouldn't clobber it.
    """
    if old_stem == new_stem:
        return text
    meta, body = parse_frontmatter(text)
    if meta.get("title") != old_stem:
        return text  # no frontmatter title, or a custom one — leave it
    # Rebuild frontmatter: strip the leading --- block, swap the title line.
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return text
    raw = m.group(1)
    # Replace only the top-level `title: <old>` line (not nested, not quoted
    # as a list). Keep formatting of the rest of the YAML.
    new_raw = re.sub(
        r"^(title\s*:\s*).*$",
        lambda mm: f"{mm.group(1)}{new_stem}",
        raw,
        count=1,
        flags=re.MULTILINE,
    )
    return text.replace(raw, new_raw, 1)


def resolve_note(target: str, notes: list[dict]) -> str | None:
    """Resolve a wikilink ``Target`` (or bare title) to a note's repo-relative path.

    ``notes`` is a list of index rows (each with ``path``), matching case-insensitive
    on the note basename (Obsidian semantics). Returns the first match or ``None``.
    """
    want = _index_key(target)
    if not want:
        return None
    for row in notes:
        path = row.get("path")
        if isinstance(path, str) and _index_key(path) == want:
            return path
    return None


def section_after(body: str, heading: str) -> str | None:
    """Return the body text under a markdown ``## heading`` (or heading + '##').

    Looks for the first heading whose text contains ``heading`` (case-insensitive)
    and returns everything up to the next same-or-lower-level heading. ``None`` when
    the heading isn't found. Used for block-level targeting.
    """
    lines = body.splitlines()
    start: int | None = None
    level = 0
    for i, line in enumerate(lines):
        m = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
        if not m:
            continue
        lvl, text = len(m.group(1)), m.group(2).strip()
        if start is None:
            if heading.lower() in text.lower():
                start, level = i, lvl
        elif lvl <= level:
            return "\n".join(lines[start:i])
    if start is not None:
        return "\n".join(lines[start:])
    return None
