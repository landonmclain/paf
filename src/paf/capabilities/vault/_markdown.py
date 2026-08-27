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
  (for future vault_patch block targeting).

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
_WIKILINK_RE = re.compile(r"\[\[([^\[\]|]+)(?:\|[^\]]+)?\]\]")


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
    the heading isn't found. Used by vault_patch for block-level targeting.
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
