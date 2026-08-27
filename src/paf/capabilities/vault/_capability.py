"""Vault capability — navigation-first note ladder over an Obsidian-style markdown vault.

Wraps the vault note index behind the navigation-first read-model (mirrors the
codebase-map ladder LOCKED 2026-08-23, applied to *notes* instead of *symbols*):

    vault_overview() → lookup_note(path, detail) → read_note(path)
                              ↕
                    note_find(title/tag/folder) / note_search(text, ...)

- ``vault_root``  : the vault directory (required). All reads confined to it.
- ``allow``/``deny`` : optional glob subtree patterns (allow = only these when
  non-empty; deny = always excluded, wins over allow) — expresses Landon's
  workspace-vs-personal boundary inside a root.
- ``storage``     : the flat-file seam (``FlatFileVaultStorage`` default).

Search is the *secondary* escape hatch; navigation is primary. Tier B (the
note graph: ``note_links``/``note_backlinks``/``note_create``/``note_rename``)
is built. Tier C (raw content writes) and shared machinery with codebase_map
are later phases — not built here (YAGNI).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pydantic_ai.capabilities import AbstractCapability

from ._store import FlatFileVaultStorage
from ._toolset import VaultToolset

_VAULT_GUIDANCE = (
    "Note / vault questions. When the user is asking about notes and documents "
    "in a markdown vault (Obsidian-style) -- which note covers a topic, what a "
    "note says, what links to what -- start with the vault tools, not the raw "
    "file tools: use `vault_overview` to orient, `lookup_note` / `read_note` to "
    "read, `note_find` / `note_search` to locate, and `note_links` / "
    "`note_backlinks` to follow the note graph. To add or rename a note, use "
    "`note_create` / `note_rename` (these keep inbound wikilinks consistent) "
    "rather than raw file writes. These are scoped to the configured vault and "
    "already respect its folder boundaries, so they return far less noise than "
    "a raw `search_files` over the whole tree. Keep the raw FileSystem tools "
    "for editing files, non-note documents, and anything outside the vault."
)


@dataclass
class Vault(AbstractCapability):
    """Navigation-first vault: overview → lookup → read + find/search + note graph.

    Exposes the nine-tool ladder (`vault_overview`, `lookup_note`, `read_note`,
    `note_find`, `note_search`, `note_links`, `note_backlinks`, `note_create`,
    `note_rename`). Reads go through the flat-file index and are confined to
    ``vault_root``; writes (create/rename) are scope-guarded and atomic.
    """

    vault_root: str | Path
    """Vault directory to navigate. All reads confined to it."""

    storage: FlatFileVaultStorage = field(default_factory=FlatFileVaultStorage)
    """Flat-file seam (atomic JSON index; a SQLite backend can swap in later)."""

    allow: list[str] = field(default_factory=list)
    """If non-empty, only subtrees matching these globs are in scope."""

    deny: list[str] = field(default_factory=list)
    """Subtrees matching these globs are always out of scope (wins over allow)."""

    def get_toolset(self) -> VaultToolset:
        """Build and return the five-tool vault ladder."""
        return VaultToolset(
            vault_root=self.vault_root,
            storage=self.storage,
            allow=self.allow,
            deny=self.deny,
        )

    def get_instructions(self) -> list[str] | None:
        """Provide a tool-choice steer: for note questions, use the ladder first.

        A static, additive nudge — do not touch memory recall. Prefers the
        scoped, token-bounded vault tools over raw file tools for note work
        without removing the FileSystem tools for editing/etc.
        """
        return [_VAULT_GUIDANCE]
