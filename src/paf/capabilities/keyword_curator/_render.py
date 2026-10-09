"""Render hints as the injected ``<relevance-hints>`` block.

The block is **pointers, never content**: it names where to look and how to open
it with an existing tool, so a wrong hint costs a line the model skims past and
can never feed a wrong passage in as evidence.

Everything shown is sanitized. The matched terms are curator-written aliases and
keywords derived from file contents; a hostile note could try to plant
instructions in one. They are flattened to one short plain-text line (no angle
brackets, no newlines, no markup) before they reach the prompt, and the ref/label
come from file paths, not from a model.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ._match import Hint

HINTS_PREFIX = "\n\n<relevance-hints>\n"
HINTS_SUFFIX = "\n</relevance-hints>"
HEADER = "Possibly relevant to this message (pointers only — open one if it helps):"

DEFAULT_MAX_CHARS = 1200
_MAX_TERM = 40
_MAX_LABEL = 160
_UNSAFE = re.compile(r"[<>`\x00-\x1f\x7f]")


def _clean(text: str, limit: int) -> str:
    flat = _UNSAFE.sub(" ", str(text))
    flat = re.sub(r"\s+", " ", flat).strip()
    return flat[:limit].rstrip()


def _pointer(hint: Hint) -> str:
    label = _clean(hint.label, _MAX_LABEL)
    if hint.kind == "skill":
        return f'skill {label} → load_skill("{label}")'
    if hint.kind == "note":
        return f'note {label} → read_note("{label}")'
    return f"{_clean(hint.kind, 20)} {label}"


def render_hints(hints: Sequence[Hint], *, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """The block (with its open/close tags), or ``""`` when there is nothing to say.

    Lines are dropped from the tail until the block fits ``max_chars`` — the
    strongest hint is always kept first.
    """
    if not hints:
        return ""
    lines: list[str] = []
    for hint in hints:
        terms = [t for t in (_clean(m, _MAX_TERM) for m in hint.matched) if t]
        tail = f"   (matched: {', '.join(terms)})" if terms else ""
        lines.append(f"- {_pointer(hint)}{tail}")
    while len(lines) > 1 and len(HEADER) + sum(len(x) + 1 for x in lines) > max_chars:
        lines.pop()
    body = HEADER + "\n" + "\n".join(lines)
    return HINTS_PREFIX + body + HINTS_SUFFIX
