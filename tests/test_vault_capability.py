"""VaultToolset — the navigation-first note ladder (Tier A read tools).

Tests the agent-facing tools against a small synthetic vault seeded with
frontmatter + wikilinks:

- ``vault_overview`` → nested folder/note tree with counts
- ``lookup_note``    → note head (frontmatter, headings, links-out); unknown → None
- ``read_note``      → full body; resolves [[wikilink]]/title → path
- ``note_find``      → Obsidian-style lookup by title/tag/folder
- ``note_search``    → content search with tag/folder filter
- root confinement   → paths outside the vault root are rejected
- allow/deny         → subtree allowlist/denylist honored by the index

All tool methods return bounded, readable ``str`` (None only for unknown notes),
matching the codebase_map / git / fs toolset convention.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from paf.capabilities.vault import (
    FlatFileVaultStorage,
    Vault,
    VaultToolset,
    build_index,
)
from paf.capabilities.vault._store import ScanScope


def _seed(root: Path, rel: str, content: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


_IDX = """---
title: Index
tags: [index, vault]
---
# Index
Links: [[Grounding]], [[Memory Design]]
"""
_GROUNDING = """---
title: Grounding
tags: [grounding, curated]
---
# Grounding
See [[Index]].
"""
_MEMORY = """---
title: Memory Design
tags: [memory]
aliases: [mem]
---
# Memory Design
Owner -> organizer -> index. Links to [[Grounding]].
"""
_NESTED = """---
title: Nested Note
tags: [nested]
---
# Nested
A [[Grounding]] wikilink.
"""


@pytest.fixture()
def vault(tmp_path: Path) -> tuple[Path, VaultToolset]:
    root = tmp_path / "vault"
    root.mkdir()
    _seed(root, "Index.md", _IDX)
    _seed(root, "Grounding.md", _GROUNDING)
    _seed(root, "Concepts/Memory Design.md", _MEMORY)
    _seed(root, "Concepts/Nested Note.md", _NESTED)
    store = FlatFileVaultStorage(index_root=tmp_path / "idx")
    build_index(root, ScanScope(root), store)
    toolset = VaultToolset(vault_root=root, storage=store)
    return root, toolset


# ------------------------------------------------------------------ vault_overview


def test_vault_overview_shows_nested_tree(vault: tuple[Path, VaultToolset]) -> None:
    _, toolset = vault
    out = toolset.vault_overview()
    assert "vault  (4 notes)" in out
    assert "Concepts/  (2 notes)" in out
    assert "Memory Design.md" in out
    assert "Nested Note.md" in out
    assert "Index.md" in out
    assert "Grounding.md" in out


def test_vault_overview_empty(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    root.mkdir()
    toolset = VaultToolset(vault_root=root, storage=FlatFileVaultStorage(index_root=tmp_path / "eidx"))
    out = toolset.vault_overview()
    assert "empty" in out


# ------------------------------------------------------------------ lookup_note


def test_lookup_note_head_and_detail(vault: tuple[Path, VaultToolset]) -> None:
    _, toolset = vault
    head = toolset.lookup_note("Index")
    assert head is not None
    assert "Note: Index.md" in head
    assert "tags: index, vault" in head
    assert "links to" in head
    assert "Grounding" in head


def test_lookup_note_unknown(vault: tuple[Path, VaultToolset]) -> None:
    _, toolset = vault
    assert toolset.lookup_note("Does Not Exist") is None


# ------------------------------------------------------------------ read_note


def test_read_note_by_title_and_wikilink(vault: tuple[Path, VaultToolset]) -> None:
    _, toolset = vault
    by_title = toolset.read_note("Memory Design")
    assert by_title is not None and "Owner -> organizer" in by_title
    by_link = toolset.read_note("[[Grounding]]")
    assert by_link is not None and "See [[Index]]" in by_link


def test_read_note_outside_root_rejected(vault: tuple[Path, VaultToolset]) -> None:
    _, toolset = vault
    # traversal attempt must not read a file outside the vault root
    assert toolset.read_note("../Index.md") is None


# ------------------------------------------------------------------ note_find


def test_note_find_by_tag(vault: tuple[Path, VaultToolset]) -> None:
    _, toolset = vault
    out = toolset.note_find(tag="memory")
    assert "Concepts/Memory Design.md" in out


def test_note_find_by_title_and_folder(vault: tuple[Path, VaultToolset]) -> None:
    _, toolset = vault
    out = toolset.note_find(title="Grounding")
    assert "Grounding.md" in out and "Memory" not in out
    out2 = toolset.note_find(folder="Concepts")
    assert "Concepts/Memory Design.md" in out2
    assert "Index.md" not in out2


def test_note_find_no_match(vault: tuple[Path, VaultToolset]) -> None:
    _, toolset = vault
    assert "No notes match" in toolset.note_find(tag="nope")


# ------------------------------------------------------------------ note_search


def test_note_search_finds_body(vault: tuple[Path, VaultToolset]) -> None:
    _, toolset = vault
    out = toolset.note_search("organizer")
    assert "Concepts/Memory Design.md" in out


def test_note_search_folder_filter(vault: tuple[Path, VaultToolset]) -> None:
    _, toolset = vault
    # 'Grounding' appears in both Index.md and Grounding.md; restrict to Concepts
    out = toolset.note_search("Grounding", folder="Concepts")
    assert "Grounding" in out  # Concepts/Nested Note.md links to Grounding


# ------------------------------------------------------------------ confinement


def test_root_confinement(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    _seed(root, "Index.md", _IDX)
    outside = tmp_path / "secret.md"
    outside.write_text("secret", encoding="utf-8")
    store = FlatFileVaultStorage(index_root=tmp_path / "cx")
    build_index(root, ScanScope(root), store)
    toolset = VaultToolset(vault_root=root, storage=store)
    # 'secret.md' is not in the index at all, and a path outside the root is denied
    assert toolset.read_note("secret.md") is None
    # symlink escaping the root is refused
    link = root / "link.md"
    link.symlink_to(outside)
    assert toolset.read_note("link.md") is None


# ------------------------------------------------------------------ allow / deny


def test_allow_restricts_scope(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    _seed(root, "Index.md", _IDX)
    _seed(root, "Concepts/Memory Design.md", _MEMORY)
    store = FlatFileVaultStorage(index_root=tmp_path / "ai")
    scoped = ScanScope(root=root, allow=["Concepts/**"])
    build_index(root, scoped, store)
    toolset = VaultToolset(
        vault_root=root,
        storage=store,
        allow=["Concepts/**"],
    )
    out = toolset.note_find()  # list all in scope
    assert "Concepts/Memory Design.md" in out
    assert "Index.md" not in out


def test_deny_excludes_subtree(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    _seed(root, "Index.md", _IDX)
    _seed(root, "Concepts/Memory Design.md", _MEMORY)
    store = FlatFileVaultStorage(index_root=tmp_path / "di")
    scoped = ScanScope(root=root, deny=["Concepts/**"])
    build_index(root, scoped, store)
    toolset = VaultToolset(
        vault_root=root,
        storage=store,
        deny=["Concepts/**"],
    )
    out = toolset.note_find()
    assert "Index.md" in out
    assert "Concepts" not in out


# ------------------------------------------------------------------ markdown helpers


def test_parse_frontmatter() -> None:
    from paf.capabilities.vault import extract_wikilinks, parse_frontmatter
    meta, body = parse_frontmatter(_MEMORY)
    assert meta.get("title") == "Memory Design"
    assert "memory" in meta.get("tags", [])
    assert "# Memory Design" in body
    assert extract_wikilinks(body) == ["Grounding"]


# ------------------------------------------------------------------ capability surface


def test_vault_capability_get_toolset_and_instructions() -> None:
    cap = Vault(vault_root=".", deny=["**/.trash/**"])
    ts = cap.get_toolset()
    assert isinstance(ts, VaultToolset)
    instr = cap.get_instructions()
    assert instr and any("vault_overview" in i for i in instr)


# ------------------------------------------------------------------ Tier B: note_links / note_backlinks


def test_note_links_resolves_outgoing(vault: tuple[Path, VaultToolset]) -> None:
    _, toolset = vault
    out = toolset.note_links("Index")
    assert out is not None
    assert "Grounding.md" in out
    assert "Concepts/Memory Design.md" in out


def test_note_links_marks_dangling(vault: tuple[Path, VaultToolset]) -> None:
    _, toolset = vault
    _seed(vault[0], "Dangling.md", "---\ntitle: Dangling\n---\nPoints to [[Nowhere]].\n")
    out = toolset.note_links("Dangling")
    assert out is not None
    assert "Nowhere (unresolved)" in out


def test_note_backlinks_reverse_edges(vault: tuple[Path, VaultToolset]) -> None:
    _, toolset = vault
    # Grounding.md is linked from Index.md, Concepts/Memory Design.md, and
    # Concepts/Nested Note.md
    out = toolset.note_backlinks("Grounding")
    assert "Index.md" in out
    assert "Concepts/Memory Design.md" in out
    assert "Concepts/Nested Note.md" in out
    assert "Grounding.md" not in out.splitlines()  # no self-link


def test_note_backlinks_none(vault: tuple[Path, VaultToolset]) -> None:
    _, toolset = vault
    # nothing in the seed links to Nested Note
    assert "no backlinks" in toolset.note_backlinks("Concepts/Nested Note")

    # ------------------------------------------------------------------ Tier B: note_create


def test_note_create_writes_note_with_frontmatter(vault: tuple[Path, VaultToolset]) -> None:
    _, toolset = vault
    out = toolset.note_create("Concepts/New Idea", body="Fresh idea.", tags=["draft"])
    assert "Created Concepts/New Idea.md" in out
    text = (vault[0] / "Concepts/New Idea.md").read_text(encoding="utf-8")
    assert "title: New Idea" in text
    assert "Fresh idea." in text
    # the new note is now visible to the read ladder
    assert "New Idea.md" in toolset.vault_overview()


def test_note_create_refuses_overwrite(vault: tuple[Path, VaultToolset]) -> None:
    _, toolset = vault
    out = toolset.note_create("Index")
    assert "already exists" in out


def test_note_create_refuses_outside_root(vault: tuple[Path, VaultToolset]) -> None:
    _, toolset = vault
    out = toolset.note_create("../evil.md")
    assert "outside the vault root" in out
    assert not (vault[0].parent / "evil.md").exists()


def test_note_create_refuses_denied_scope(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    _seed(root, "Index.md", _IDX)
    store = FlatFileVaultStorage(index_root=tmp_path / "ci")
    build_index(root, ScanScope(root, deny=["Private/**"]), store)
    toolset = VaultToolset(vault_root=root, storage=store, deny=["Private/**"])
    out = toolset.note_create("Private/hidden")
    assert "excluded by scope" in out
    assert not (root / "Private/hidden.md").exists()


# ------------------------------------------------------------------ Tier B: note_rename


def test_note_rename_rewrites_inbound_links(vault: tuple[Path, VaultToolset]) -> None:
    _, toolset = vault
    # 'Grounding' is pointed at by Index, Memory Design, and Nested Note.
    out = toolset.note_rename("Grounding", "Foundations")
    assert "Renamed Grounding.md → Foundations.md" in out
    assert "Rewrote inbound links in 3 note(s)" in out
    # the file itself moved
    assert not (vault[0] / "Grounding.md").exists()
    assert (vault[0] / "Foundations.md").exists()
    # one of the inbounds (Memory Design) now points at Foundations
    md = (vault[0] / "Concepts/Memory Design.md").read_text(encoding="utf-8")
    assert "[[Foundations]]" in md
    # the index reflects the move (foundations reachable by its new title)
    assert toolset.lookup_note("Foundations") is not None


def test_note_rename_updates_auto_title_keeps_custom(vault: tuple[Path, VaultToolset]) -> None:
    _, toolset = vault
    toolset.note_create("Auto", body="x")
    toolset.note_rename("Auto", "AutoRenamed")
    assert "title: AutoRenamed" in (vault[0] / "AutoRenamed.md").read_text(encoding="utf-8")
    # a note with a hand-set display title keeps it
    _seed(vault[0], "Concept.md", "---\ntitle: My Display Name\n---\nbody")
    toolset.note_rename("Concept", "ConceptRenamed")
    assert "title: My Display Name" in (vault[0] / "ConceptRenamed.md").read_text(encoding="utf-8")


def test_note_rename_refuses_collision_and_escape(vault: tuple[Path, VaultToolset]) -> None:
    _, toolset = vault
    out = toolset.note_rename("Index", "Grounding")  # collides with existing Grounding.md
    assert "already exists" in out
    assert (vault[0] / "Index.md").exists()  # untouched
    out2 = toolset.note_rename("Index", "../escape.md")
    assert "outside the vault root" in out2
    assert not (vault[0].parent / "escape.md").exists()


# ------------------------------------------------------------------ Tier B: markdown rewrite helpers


def test_rewrite_wikilink_target_handles_alias() -> None:
    from paf.capabilities.vault import rewrite_wikilink_target

    out = rewrite_wikilink_target(
        "see [[Memory Design]] and [[Memory Design|MD]] and [[Index]]",
        "Memory Design",
        "Memory Architecture",
    )
    assert "[[Memory Architecture]]" in out
    assert "[[Memory Architecture|MD]]" in out
    assert "[[Index]]" in out  # untouched


def test_rename_note_title_preserves_custom() -> None:
    from paf.capabilities.vault import rename_note_title

    auto = "---\ntitle: Old\n---\nbody"
    assert "title: New" in rename_note_title(auto, "Old", "New")
    custom = "---\ntitle: Hand Tuned\n---\nbody"
    assert "Hand Tuned" in rename_note_title(custom, "Old", "New")
