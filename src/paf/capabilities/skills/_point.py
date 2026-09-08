"""Standalone `point()` pointer entrypoint for the paf Skills capability.

Adapted from `skill_note_hint.point` (MIT) so the pointer engine is re-homed
inside paf. Given a freeform user message, it scans a skills tree and returns
**skill pointers** — one-line hints like ``consider: category/skill`` — never
content. Framework-free, zero LLM on the call path (deterministic BM25 over
frontmatter vocab). Standalone notes are owned by the Vault capability, not
this skills-only pointer.
"""

from __future__ import annotations

from ._corpus import HintCorpus

__all__ = ["point"]


def point(query_text: str, roots: list[str] | str, top_k: int = 3) -> list[str]:
    """Rank ``roots`` (directories containing SKILL.md trees) against
    ``query_text`` and return up to ``top_k`` canonical skill pointers.

    A pointer names *where to look* (the skill), not answer content, so a
    wrong match is a line you skim past, never injected evidence. Notes are
    intentionally outside this skills-only entrypoint.
    """
    corpus = HintCorpus(roots=roots)
    hits = corpus.rank(query_text, k=top_k)
    return [doc.hint for doc, _score in hits]
