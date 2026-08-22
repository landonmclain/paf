"""Regex fallback for unparseable/niche files (STUB — Phase 0 layout only).

Planned (per Codebase-Map-Capability-Spec § `_rules.py`): a lightweight regex fallback
for files tree-sitter can't parse (config, markdown, unusual DSLs, generated code).
Runs when `_analyze.py` yields nothing/near-nothing for a file, so every likely-indexable
file still contributes *some* symbol/name signal to search.

Not needed in Phase 0 (which is storage + schema + eval shell only) — stub for layout.
"""

from __future__ import annotations

from ._schema import FileInfo


def fallback_analyze(path: str) -> FileInfo:
    """STUB — regex-based fallback extraction for unparseable files (Phase 1/4)."""
    raise NotImplementedError("fallback_analyze is not built until a later phase")
