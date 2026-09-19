"""Vault capability: navigation-first note ladder over an Obsidian-style vault.

Ships the five-tool read-model (`vault_overview`, `lookup_note`, `read_note`,
`note_find`, `note_search`), the note-graph tools (`note_links`,
`note_backlinks`, `note_create`, `note_rename` with inbound-rewrite), the edit
tools (`note_edit`, `note_append`), the flat-file index
(`FlatFileVaultStorage`), vault-root confinement with `allow`/`deny`, and the
spec-driven `vault:` config block.

Public surface: the markdown helpers (`parse_frontmatter`, `extract_wikilinks`,
`resolve_note`, `rewrite_wikilink_target`, `section_after`), the index store
(`ScanScope`, `FlatFileVaultStorage`, `build_index`, `load_index_notes`,
`index_dir_for_root`), and the capability (`Vault`, `VaultToolset`).
"""

from __future__ import annotations

from ._capability import Vault
from ._markdown import (
    extract_wikilinks,
    parse_frontmatter,
    rename_note_title,
    resolve_note,
    rewrite_wikilink_target,
    section_after,
)
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
    "rename_note_title",
    "resolve_note",
    "rewrite_wikilink_target",
    "section_after",
    # index store
    "ScanScope",
    "FlatFileVaultStorage",
    "build_index",
    "load_index_notes",
    "index_dir_for_root",
]
