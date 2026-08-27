"""Vault capability: navigation-first note ladder over an Obsidian-style vault.

**Status: Phase 0 + Tier A (read ladder) shipped 2026-08-26.** The five-tool
read-model (`vault_overview`, `lookup_note`, `read_note`, `note_find`,
`note_search`), the flat-file index (`FlatFileVaultStorage`), and vault-root
confinement with `allow`/`deny` subtree patterns are all landed, plus the
spec-driven `vault:` config block. Tier B (note graph: backlinks/rename/create)
and Tier C (writes: vault_write/vault_patch) are next — not built yet (YAGNI).

Public surface: the markdown helpers (`parse_frontmatter`, `extract_wikilinks`,
`resolve_note`, `section_after`), the index store (`ScanScope`,
`FlatFileVaultStorage`, `build_index`, `load_index_notes`, `index_dir_for_root`),
and the capability (`Vault`, `VaultToolset`).
"""

from __future__ import annotations

from ._capability import Vault
from ._markdown import extract_wikilinks, parse_frontmatter, resolve_note, section_after
from ._store import (
    FlatFileVaultStorage,
    ScanScope,
    build_index,
    index_dir_for_root,
    load_index_notes,
)
from ._toolset import VaultToolset

__all__ = [
    # capability
    "Vault",
    "VaultToolset",
    # markdown helpers
    "parse_frontmatter",
    "extract_wikilinks",
    "resolve_note",
    "section_after",
    # index store
    "ScanScope",
    "FlatFileVaultStorage",
    "build_index",
    "load_index_notes",
    "index_dir_for_root",
]
