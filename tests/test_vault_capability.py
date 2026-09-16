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
from pydantic_ai.exceptions import ModelRetry

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


def test_resolve_pathlike_miss_is_hard_never_basename_match(tmp_path: Path) -> None:
    """A path-like ref that isn't an exact index path is a clean miss that
    regrounds — it returns a hint, and NEVER silently reads a *different* note
    sharing the same basename (e.g. 28 notes titled ``Index``). A bare title
    still uses the Obsidian basename matcher."""
    root = tmp_path / "vault"
    root.mkdir()
    _seed(root, "Index.md", _IDX)                       # first in scan order
    _seed(root, "Investigations/Foo/Index.md", _IDX)    # also basename "Index"
    _seed(root, "Portfolio-GitHub/Index.md", _IDX)      # the intended target
    store = FlatFileVaultStorage(index_root=tmp_path / "idx")
    build_index(root, ScanScope(root), store)
    toolset = VaultToolset(vault_root=root, storage=store)

    # Exact path-like target → resolves to that note.
    assert toolset.read_note("Portfolio-GitHub/Index.md") is not None

    # Path-like ref may drop the ".md" extension.
    assert toolset.read_note("Portfolio-GitHub/Index") is not None

    # Over-prefixed / wrong-folder path-like ref → regrounding hint (a string),
    # NOT the first-note-with-that-basename (would have been Investigations/Foo).
    hint = toolset.read_note("Projects/Portfolio-GitHub/Index.md")
    assert hint is not None
    assert "do NOT prepend the root folder name" in hint
    assert "Investigations" not in hint
    assert "No note found" in hint

    # Bare basename still uses the Obsidian title matcher (no hard miss).
    assert toolset.read_note("Index") is not None

    # A different over-prefix also regrounds — same recovery hint, never a leak.
    hint2 = toolset.read_note("Personal/Portfolio-GitHub/Index.md")
    assert hint2 is not None
    assert "do NOT prepend the root folder name" in hint2
    assert "Investigations" not in hint2


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


# ------------------------------------------------------------------ index location (repo-owned, keyed by vault root)


def test_index_root_override_wins(tmp_path: Path) -> None:
    """An explicit FlatFileVaultStorage(index_root=...) controls where the index lives."""
    root = tmp_path / "vault"
    root.mkdir()
    _seed(root, "Index.md", _IDX)
    idx = tmp_path / "custom-idx"
    store = FlatFileVaultStorage(index_root=idx)
    vs = VaultToolset(vault_root=root, storage=store)
    vs.vault_overview()  # forces index build
    assert (idx / "index.json").exists()
    assert (idx / "snapshot.json").exists()
    # nothing co-located in the vault tree
    assert not (root / ".paf-vault").exists()
    assert not (root / "data").exists()


def test_default_index_is_repo_owned_not_in_vault(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With default storage, the index is keyed under the capability's default
    index dir (data/.paf-vault-index by default), NOT co-located inside the vault tree.

    The default index dir is monkeypatched to a tmp dir so the test writes to a
    pytest-cleaned location: with a real tmp_path vault root the default hash-keyed
    dir would accumulate one `data/.paf-vault-index/<sha256(tmp root)>/` folder PER
    pytest run (pytest's tmp counter is unique each run) and never get cleaned up.
    """
    from paf.capabilities.vault import _store
    from paf.capabilities.vault._store import FlatFileVaultStorage as FFS
    from paf.capabilities.vault._store import index_dir_for_root
    from paf.config import PROJECT_ROOT

    root = tmp_path / "vault"
    root.mkdir()
    _seed(root, "Index.md", _IDX)
    # Point the default index dir at pytest's tmp area so the "repo-owned,
    # keyed by vault-root hash" behavior is proven WITHOUT leaking one unique
    # folder into the real data/.paf-vault-index per test run.
    monkeypatch.setattr(_store, "_DEFAULT_VAULT_INDEX_DIR", str(tmp_path / "repo-index"))
    store = FFS()  # default storage -> capability default (monkeypatched) index dir
    build_index(root, ScanScope(root), store)
    expected = PROJECT_ROOT / Path(_store._DEFAULT_VAULT_INDEX_DIR) / index_dir_for_root(root).name
    assert expected == index_dir_for_root(root)
    # index built at the configured location, keyed by the vault root's hash
    assert (index_dir_for_root(root) / "index.json").exists()
    # no index artifacts leak into the vault tree
    assert not (root / ".paf-vault").exists()
    assert not (root / "data").exists()


def test_default_index_keyed_per_vault_root(tmp_path: Path) -> None:
    """Distinct vault roots get distinct hash-keyed index dirs under the repo area."""
    from paf.capabilities.vault._store import index_dir_for_root

    a = tmp_path / "vault-a"
    b = tmp_path / "vault-b"
    a.mkdir()
    b.mkdir()
    _seed(a, "Index.md", _IDX)
    _seed(b, "Index.md", _IDX)
    da = index_dir_for_root(a)
    db = index_dir_for_root(b)
    assert da != db
    # both are keyed under the same repo-owned base
    assert da.parent == db.parent


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
    with pytest.raises(ModelRetry, match="already exists"):
        toolset.note_create("Index")


def test_note_create_refuses_outside_root(vault: tuple[Path, VaultToolset]) -> None:
    _, toolset = vault
    with pytest.raises(ModelRetry, match="outside the vault root"):
        toolset.note_create("../evil.md")
    assert not (vault[0].parent / "evil.md").exists()


def test_note_create_refuses_denied_scope(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    _seed(root, "Index.md", _IDX)
    store = FlatFileVaultStorage(index_root=tmp_path / "ci")
    build_index(root, ScanScope(root, deny=["Private/**"]), store)
    toolset = VaultToolset(vault_root=root, storage=store, deny=["Private/**"])
    with pytest.raises(ModelRetry, match="excluded by scope"):
        toolset.note_create("Private/hidden")
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
    with pytest.raises(ModelRetry, match="already exists"):
        toolset.note_rename("Index", "Grounding")  # collides with existing Grounding.md
    assert (vault[0] / "Index.md").exists()  # untouched
    with pytest.raises(ModelRetry, match="outside the vault root"):
        toolset.note_rename("Index", "../escape.md")
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


# ------------------------------------------------------------------ Tier C: read_note version header


def test_read_note_reports_content_version_header(vault: tuple[Path, VaultToolset]) -> None:
    _, toolset = vault
    out = toolset.read_note("Grounding")
    assert out is not None
    lines = out.splitlines()
    # first line is the version header; the body follows intact
    assert lines[0].startswith("[path: Grounding.md | version:")
    assert "version:" in lines[0]
    # body still fully present (substring survives the added header)
    assert "See [[Index]]." in out


def test_read_note_version_matches_content_hash(vault: tuple[Path, VaultToolset]) -> None:
    _, toolset = vault
    import hashlib

    out = toolset.read_note("Grounding")
    assert out is not None
    version = out.splitlines()[0].rsplit("version: ", 1)[1].rstrip("]")
    raw = (vault[0] / "Grounding.md").read_text(encoding="utf-8")
    assert version == hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


# ------------------------------------------------------------------ Tier C: note_edit


def test_note_edit_replaces_exact_once_and_returns_new_version(vault: tuple[Path, VaultToolset]) -> None:
    _, toolset = vault
    out = toolset.note_edit("Grounding", [{"old_text": "See [[Index]].", "new_text": "See [[Foundations]]."}])
    assert "Edited Grounding.md" in out
    assert "[version:" in out
    text = (vault[0] / "Grounding.md").read_text(encoding="utf-8")
    assert "See [[Foundations]]." in text
    assert "See [[Index]]." not in text
    # the index reflects the new link so the read ladder stays consistent
    assert "Foundations" in toolset.lookup_note("Grounding")


def test_note_edit_version_chaining_read_edit_edit(vault: tuple[Path, VaultToolset]) -> None:
    _, toolset = vault
    read = toolset.read_note("Grounding")
    assert read is not None
    v1 = read.splitlines()[0].rsplit("version: ", 1)[1].rstrip("]")
    r1 = toolset.note_edit(
        "Grounding", [{"old_text": "See [[Index]].", "new_text": "First pass."}], expected_version=v1
    )
    v2 = r1.splitlines()[0].rsplit("version:", 1)[1].strip("]")
    # second edit with the freshly returned version succeeds
    r2 = toolset.note_edit(
        "Grounding", [{"old_text": "First pass.", "new_text": "Second pass."}], expected_version=v2
    )
    assert "Edited Grounding.md" in r2
    assert "Second pass." in (vault[0] / "Grounding.md").read_text(encoding="utf-8")


def test_note_edit_batch_applies_all_pairs_in_one_write(vault: tuple[Path, VaultToolset]) -> None:
    _, toolset = vault
    # two independent changes, one atomic call, and a per-pair diff each
    out = toolset.note_edit(
        "Grounding",
        [
            {"old_text": "See [[Index]].", "new_text": "See [[Foundations]]."},
            {"old_text": "tags: [grounding, curated]", "new_text": "tags: [grounding, curated, edited]"},
        ],
    )
    assert "Edited Grounding.md" in out
    text = (vault[0] / "Grounding.md").read_text(encoding="utf-8")
    assert "See [[Foundations]]." in text
    assert "tags: [grounding, curated, edited]" in text


def test_note_edit_conflict_on_stale_version(vault: tuple[Path, VaultToolset]) -> None:
    _, toolset = vault
    with pytest.raises(ModelRetry, match="Conflict"):
        toolset.note_edit(
            "Grounding", [{"old_text": "See [[Index]].", "new_text": "Changed."}], expected_version="stale-deadbeef"
        )
    # the note was NOT modified
    text = (vault[0] / "Grounding.md").read_text(encoding="utf-8")
    assert "See [[Index]]." in text
    assert "Changed." not in text


def test_note_edit_no_expected_version_allowed(vault: tuple[Path, VaultToolset]) -> None:
    _, toolset = vault
    # expected_version is optional when the caller doesn't need CAS
    out = toolset.note_edit("Grounding", [{"old_text": "See [[Index]].", "new_text": "See [[Index]] fine."}])
    assert "Edited Grounding.md" in out


def test_note_edit_old_text_not_found(vault: tuple[Path, VaultToolset]) -> None:
    _, toolset = vault
    with pytest.raises(ModelRetry, match="old_text not found"):
        toolset.note_edit("Grounding", [{"old_text": "no such text anywhere", "new_text": "x"}])
    assert "See [[Index]]." in (vault[0] / "Grounding.md").read_text(encoding="utf-8")


def test_note_edit_old_text_not_unique(vault: tuple[Path, VaultToolset]) -> None:
    _, toolset = vault
    # 'Index' appears in both the frontmatter title and the heading; ask to
    # replace it → must refuse because the match isn't unique.
    with pytest.raises(ModelRetry, match="found 2 times"):
        toolset.note_edit("Index", [{"old_text": "Index", "new_text": "IDX"}])
    assert "IDX" not in (vault[0] / "Index.md").read_text(encoding="utf-8")


def test_note_edit_batch_rejects_all_when_any_pair_fails(vault: tuple[Path, VaultToolset]) -> None:
    _, toolset = vault
    # one valid + one garbage pair → the WHOLE batch is rejected, nothing lands
    with pytest.raises(ModelRetry, match="Refusing to edit"):
        toolset.note_edit(
            "Grounding",
            [
                {"old_text": "See [[Index]].", "new_text": "See [[Foundations]]."},
                {"old_text": "no such text anywhere", "new_text": "x"},
            ],
        )
    text = (vault[0] / "Grounding.md").read_text(encoding="utf-8")
    assert "See [[Foundations]]." not in text
    assert "See [[Index]]." in text


def test_note_edit_unknown_or_outside_root(vault: tuple[Path, VaultToolset]) -> None:
    _, toolset = vault
    # out-of-root traversal is rejected without touching anything
    out = toolset.note_edit("../Index.md", [{"old_text": "a", "new_text": "b"}])
    assert "(unknown note" in out or "vault" in out
    out2 = toolset.note_edit("Does Not Exist", [{"old_text": "a", "new_text": "b"}])
    assert "unknown" in out2.lower() or "not found" in out2.lower()
    # unknown path-like ref regrounds rather than guessing
    out3 = toolset.note_edit("Projects/Index.md", [{"old_text": "a", "new_text": "b"}])
    assert "do NOT prepend" in out3


def test_write_atomic_is_thread_safe() -> None:
    """Regression: concurrent writes to the SAME target must not race.

    A fixed ``.name.staging`` filename was shared across writers — one thread's
    ``os.replace`` could consume the staging file another thread was about to move,
    raising ``FileNotFoundError:[Errno 2] ... .index.json.staging -> ... index.json``
    (seen live in the web server where Gradio runs agent turns in parallel).
    Each write must use a process-unique temp name and leave no ``*.tmp`` debris.
    """
    import tempfile
    import threading

    from paf.capabilities.vault._store import _write_atomic

    d = Path(tempfile.mkdtemp())
    target = d / "index.json"
    errors: list[tuple[int, str]] = []
    def _worker(n: int) -> None:
        try:
            for _ in range(3000):
                _write_atomic(target, f"payload {n} x" * 40)
        except Exception as exc:  # pragma: no cover -- assertion below reports
            errors.append((n, repr(exc)))

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [], f"concurrent write raced: {errors[:3]}"
    assert target.exists()
    # no temp debris left behind
    leftover = [p.name for p in d.glob("*.tmp") or d.glob(".*.tmp")]
    assert leftover == [], f"leftover temp files: {leftover}"
