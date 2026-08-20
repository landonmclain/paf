"""Standalone `point()` pointer entrypoint for the paf Skills capability.

Adapted from `skill_note_hint.point` (MIT) so the pointer engine is re-homed
inside paf. Given a freeform user message, it scans the known vocabulary of a
set of source directories (skills/notes) and returns POINTERS — one-line hints
like ``consider: skill-x`` / ``note: <name>`` — never content. Framework-free,
zero LLM on the call path (deterministic BM25 over frontmatter vocab).
"""

from __future__ import annotations

from ._corpus import HintCorpus

__all__ = ["point"]


def point(query_text: str, roots: list[str] | str, top_k: int = 3) -> list[str]:
    """Rank ``roots`` (dirs of SKILL.md skills and/or vault .md notes) against
    ``query_text`` and return up to ``top_k`` one-line pointer hints.

    A pointer names *where to look* (the skill/note), not the answer content,
    so a wrong match is a line you skim past, never injected evidence.
    """
    corpus = HintCorpus(roots=roots)
    hits = corpus.rank(query_text, k=top_k)
    return [doc.hint for doc, _score in hits]
