"""VaultToolset — the navigation-first note ladder (Tier A read tools).

The model navigates a vault the way a human does — climb down from the overview,
rather than brute-forcing keyword scans:

    vault_overview() → lookup_note(path, detail) → read_note(path)
                              ↕
                    note_find(title/tag/folder) / note_search(text, ...)

- ``vault_overview``  — orient to the top: folder tree + Index.md heads.
- ``lookup_note``     — the bridge: note head (frontmatter, headings, links-out).
- ``read_note``       — the whole note body (resolves [[wikilink]] → path).
- ``note_find``       — Obsidian-style lookup by title/tag/frontmatter.
- ``note_search``     — content search with optional tag/folder filter.
- ``note_links``      — a note's outgoing wikilinks, resolved to paths.
- ``note_backlinks``  — what links TO a note (reverse edges of the note graph).
- ``note_create``     — add a note (frontmatter + body), scope-guarded.
- ``note_rename``     — rename a note and rewrite inbound wikilinks atomically.

Tier C (raw content writes: vault_write/vault_patch with CAS) is a later phase
— deliberately NOT registered here yet (YAGNI). Tools are agent-facing: every
method returns a bounded, readable ``str``, never raw dicts.

Index freshness is lazy: before a read, we ``sync_changed`` against the snapshot
and rebuild the index rows for any changed/added notes (stat gate, hash confirm),
so the ladder always reflects the on-disk vault without a full pre-scan.
"""

from __future__ import annotations

import os
import posixpath
from pathlib import Path

from pydantic_ai.toolsets import FunctionToolset

from ._markdown import rename_note_title, resolve_note, rewrite_wikilink_target
from ._store import (
    FlatFileVaultStorage,
    ScanScope,
    build_index,
    load_index_notes,
)

_MAX_HEAD_LINES = 15   # lines of note body shown in lookup_note at default
_MAX_SEARCH_HITS = 20  # note_search cap

_INSERT = "  "  # tree indent


def _yaml_list(values: list[str]) -> str:
    """Render a list as a YAML flow sequence for frontmatter (safe for one word)."""
    import json

    return json.dumps(values, ensure_ascii=False)


class VaultToolset(FunctionToolset):
    """Navigation-first vault ladder rooted at a vault root.

    All reads go through the flat-file index (default ``FlatFileVaultStorage``)
    and are confined to ``vault_root`` (allow/deny honored by ``ScanScope``).
    """

    def __init__(
        self,
        *,
        vault_root: str | Path,
        storage: FlatFileVaultStorage | None = None,
        allow: list[str] | None = None,
        deny: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._root = Path(vault_root).resolve()
        self._storage = storage or FlatFileVaultStorage()
        self._scope = ScanScope(root=self._root, allow=allow or [], deny=deny or [])

        self.add_function(self.vault_overview, name="vault_overview")
        self.add_function(self.lookup_note, name="lookup_note")
        self.add_function(self.read_note, name="read_note")
        self.add_function(self.note_find, name="note_find")
        self.add_function(self.note_search, name="note_search")
        self.add_function(self.note_links, name="note_links")
        self.add_function(self.note_backlinks, name="note_backlinks")
        self.add_function(self.note_create, name="note_create")
        self.add_function(self.note_rename, name="note_rename")

    # -- index freshness -----------------------------------------------------

    def _notes(self) -> list[dict]:
        """Ensure the index is fresh (lazy sync) and return the per-note rows."""
        changed = self._storage.sync_changed(self._root, self._scope)
        if changed:
            build_index(self._root, self._scope, self._storage)
        index = self._storage.read_index(self._root)
        return load_index_notes(index)

    def _ensure_index(self) -> None:
        """Build the index from scratch if it doesn't exist yet (first call)."""
        if self._storage.read_index(self._root) is None:
            build_index(self._root, self._scope, self._storage)

    def _resolve(self, ref: str) -> str | None:
        """Resolve a note reference: a path, a title, or a [[wikilink]] target.

        Prefers an exact path; falls back to a case-insensitive title/wikilink
        match against the index. Returns a repo-relative path or ``None``.
        Refuses path-traversal and absolute refs outright.
        """
        ref = ref.strip()
        if ref.startswith("[[") and ref.endswith("]]"):
            ref = ref[2:-2].split("|")[0].strip()
        # Hard traversal guard: no absolute paths, no parent-dir segments, no
        # window-style drive traversal. A bare title/wikilink is fine.
        if ref.startswith("/") or "\\" in ref:
            return None
        parts = Path(ref).parts
        if ".." in parts or (parts and parts[0].endswith(":")):
            return None
        notes = self._notes()
        for row in notes:
            if row.get("path") == ref:
                return ref
        return resolve_note(ref, notes)

    def _abs(self, rel: str) -> Path | None:
        """Absolute path for a vault-relative ref, confined to the root."""
        full = (self._root / rel).resolve()
        try:
            full.relative_to(self._root)
        except ValueError:
            return None
        return full

    # -- tools ---------------------------------------------------------------

    def vault_overview(self, max_depth: int | None = None) -> str:
        """Orient to the top of the vault: return the folder/note tree.

        Use this FIRST to see how the vault is laid out (folders → notes), like
        reading a table of contents. This is the starting point — a tree you climb
        down, not a search. ``max_depth`` caps how deep the tree opens (default:
        open fully; pass e.g. ``max_depth=2`` on a big vault to see the top
        folders, then drill with ``lookup_note``/``read_note``).
        """
        self._ensure_index()
        notes = self._notes()
        if not notes:
            return f"Vault at {self._root} is empty (no in-scope .md notes)."
        # Build the nested folder tree from note paths.
        lines: list[str] = [f"{self._root.name}  ({len(notes)} notes)"]

        def _emit(dir_rel: str, prefix: str) -> None:
            # direct .md files in this exact dir
            direct_files = sorted(
                n["path"] for n in notes if posixpath.dirname(n["path"]) == dir_rel
            )
            # child subdirs: the next segment after dir_rel of notes deeper than
            # one level down (a real folder, not a sibling file).
            real_prefix = dir_rel + "/" if dir_rel else ""
            subdirs: set[str] = set()
            for n in notes:
                rel = n["path"]
                if rel == dir_rel or not rel.startswith(real_prefix):
                    continue
                rest = rel[len(real_prefix) :]
                if "/" in rest:
                    subdirs.add(rest.split("/")[0])
            for name in sorted(subdirs):
                child = f"{dir_rel}/{name}" if dir_rel else name
                count = sum(1 for n in notes if posixpath.dirname(n["path"]) == child or posixpath.dirname(n["path"]).startswith(child + "/"))
                lines.append(f"{prefix}{name}/  ({count} notes)")
                _emit(child, prefix + _INSERT)
            for name in direct_files:
                lines.append(f"{prefix}{name.rsplit('/', 1)[-1]}")
        _emit("", "")
        return "\n".join(lines)

    def lookup_note(self, path: str, detail: bool = False) -> str | None:
        """Look up a note's head (frontmatter, headings, links-out).

        ``path`` is a vault-relative path, a note title, or a ``[[wikilink]]``
        target. Returns the note's head: frontmatter meta (title/tags/aliases),
        its heading list, the first lines of the body, and the notes it links to.
        Use ``read_note`` when you need the whole body. Returns ``None`` for an
        unknown note.
        """
        rel = self._resolve(path)
        if rel is None:
            return None
        for n in self._notes():
            if n["path"] == rel:
                return self._format_head(n, detail=detail)
        return None

    def read_note(self, path: str) -> str | None:
        """Read the full body of a note.

        ``path`` is a vault-relative path, a note title, or a ``[[wikilink]]``
        target. Returns the entire note contents (frontmatter included). Returns
        ``None`` for an unknown note or one outside the vault root.
        """
        rel = self._resolve(path)
        if rel is None:
            return None
        full = self._abs(rel)
        if full is None or not full.is_file():
            return None
        return full.read_text(encoding="utf-8", errors="replace")

    def note_find(
        self, title: str | None = None, tag: str | None = None, folder: str | None = None
    ) -> str:
        """Find notes by title, tag, and/or folder (Obsidian-style lookup).

        At least one of ``title``, ``tag``, or ``folder`` should be provided.
        ``title`` matches the note title/basename (case-insensitive substring);
        ``tag`` matches a frontmatter tag; ``folder`` matches the leading vault
        folder path. Returns a short list of matching note paths.
        """
        notes = self._notes()
        if not notes:
            return "(vault empty)"
        hits = []
        for n in notes:
            if title and title.lower() not in n.get("title", "").lower():
                continue
            if tag and tag not in n.get("tags", []):
                continue
            if folder and not n["path"].startswith(folder.rstrip("/") + "/"):
                continue
            hits.append(n["path"])
        if not hits:
            return "No notes match."
        return "\n".join(hits)

    def note_search(self, text: str, tag: str | None = None, folder: str | None = None) -> str:
        """Content search over notes (escape hatch when you know a word).

        Returns up to ``_MAX_SEARCH_HITS`` notes whose *body* contains ``text``
        (case-insensitive), each with a one-line context snippet and its path.
        ``tag``/``folder`` filter the searched set. Prefer ``vault_overview`` +
        ``lookup_note`` / ``read_note`` for navigation; use this when you know a
        term but not where it lives.
        """
        notes = self._notes()
        if not notes:
            return "(vault empty)"
        needle = text.lower()
        hits: list[tuple[str, str]] = []
        for n in notes:
            if folder and not n["path"].startswith(folder.rstrip("/") + "/"):
                continue
            if tag and tag not in n.get("tags", []):
                continue
            full = self._abs(n["path"])
            if full is None or not full.is_file():
                continue
            body = full.read_text(encoding="utf-8", errors="replace")
            if needle in body.lower():
                idx = body.lower().find(needle)
                start = max(0, idx - 60)
                snippet = body[start : idx + len(text) + 60].replace("\n", " ")
                hits.append((n["path"], snippet.strip()))
                if len(hits) >= _MAX_SEARCH_HITS:
                    break
        if not hits:
            return f"No notes contain {text!r}."
        out = [f"{len(hits)} note(s) containing {text!r}:"]
        for path, snip in hits:
            out.append(f"- {path}\n    …{snip}…")
        return "\n".join(out)

    # -- Tier B: note graph ----------------------------------------------------
    # These are the graph-aware operations: they read or maintain the links
    # between notes, not just raw bytes. note_create/note_rename are scope-
    # guarded (confined to the root and to allow/deny) and written atomically.

    def _in_scope_path(self, rel: str) -> Path | None:
        """Absolute path for a prospective write ref, or None if out of scope.

        Rejects traversal/absolute refs, paths escaping the root, and paths the
        ``ScanScope`` would not include (allow/deny). Returns the confined
        absolute path (the note need not exist yet for create).
        """
        rel = rel.strip()
        if rel.startswith("/") or "\\" in rel:
            return None
        parts = Path(rel).parts
        if not parts or ".." in parts or parts[0].endswith(":"):
            return None
        full = self._abs(rel)
        if full is None:
            return None
        if not self._scope._allowed_rel(rel):
            return None
        return full

    def _write_note_atomic(self, full: Path, content: str) -> None:
        """Write a note's contents to disk atomically (staging + rename)."""
        full.parent.mkdir(parents=True, exist_ok=True)
        staging = full.with_name(f".{full.name}.staging")
        staging.write_text(content, encoding="utf-8")
        os.replace(staging, full)

    def note_links(self, path: str) -> str | None:
        """List a note's outgoing wikilinks, resolved to vault-relative paths.

        ``path`` is a vault-relative path, a title, or a ``[[wikilink]]``.
        Returns each link target the note points at, resolved to its note path
        (dangling links are shown with their raw target marked ``(unresolved)``).
        Returns ``None`` for an unknown note.
        """
        rel = self._resolve(path)
        if rel is None:
            return None
        notes = self._notes()
        row = next((n for n in notes if n["path"] == rel), None)
        if row is None:
            return None
        raw = row.get("links") or []
        if not raw:
            return f"{rel} links to nothing."
        resolved: list[str] = []
        for target in raw:
            hit = resolve_note(target, notes)
            resolved.append(hit if hit else f"{target} (unresolved)")
        return "\n".join(resolved)

    def note_backlinks(self, path: str) -> str:
        """What links TO a note — the reverse edges of the note graph.

        ``path`` is a vault-relative path, a title, or a ``[[wikilink]]``.
        Returns every other note whose wikilinks resolve to this note (its
        inbound references). Returns ``(no backlinks)`` when nothing points in.
        """
        rel = self._resolve(path)
        if rel is None:
            return f"(unknown note: {path})"
        notes = self._notes()
        inbound: list[str] = []
        for n in notes:
            if n["path"] == rel:
                continue
            for link in n.get("links") or []:
                if resolve_note(link, notes) == rel:
                    inbound.append(n["path"])
                    break
        if not inbound:
            return f"{rel} has no backlinks."
        return "\n".join(inbound)

    def note_create(
        self,
        path: str,
        body: str | None = None,
        tags: list[str] | None = None,
        alias: str | None = None,
    ) -> str:
        """Create a new note in the vault (scope-guarded).

        ``path`` is a vault-relative path ending in ``.md`` (nested folders are
        created as needed). Writes an optional YAML frontmatter block (title
        derived from the basename; ``tags``/``alias`` are optional) followed by
        ``body``. Refuses to overwrite an existing note, to write outside the
        vault root, or into a folder excluded by the vault's allow/deny scope.
        Returns a confirmation naming the created path.
        """
        # Normalize: require a .md extension.
        rel = path.strip()
        if not rel.endswith(".md"):
            rel = rel + ".md"
        full = self._in_scope_path(rel)
        if full is None:
            return f"Refusing to create {path!r}: outside the vault root or excluded by scope."
        if full.exists():
            return f"Refusing to create {rel!r}: a note already exists there."
        title = Path(rel).stem
        parts: list[str] = ["---", f"title: {title}"]
        if tags:
            parts.append(f"tags: {_yaml_list(tags)}")
        if alias:
            parts.append(f"aliases: {_yaml_list([alias])}")
        parts.append("---")
        if body:
            content = "\n".join(parts) + "\n\n" + body.rstrip() + "\n"
        else:
            content = "\n".join(parts) + "\n"
        self._write_note_atomic(full, content)
        return f"Created {rel} ({len(content)} bytes)."

    def note_rename(self, path: str, new_path: str) -> str:
        """Rename a note and rewrite every inbound wikilink to match.

        ``path`` is the existing note (vault-relative path, title, or
        ``[[wikilink]]``); ``new_path`` is its new vault-relative path. Moves the
        file, then updates each other note that links to it so ``[[Old Name]]``
        (and ``[[Old Name|alias]]``) becomes ``[[New Name]]`` — keeping the note
        graph consistent. Atomic on both the move and each rewrite. Refuses a
        rename that escapes scope or collides with an existing note.
        """
        old_rel = self._resolve(path)
        if old_rel is None:
            return f"(unknown note: {path})"
        new_rel = new_path.strip()
        if not new_rel.endswith(".md"):
            new_rel = new_rel + ".md"
        if new_rel.lower() == old_rel.lower():
            return "New path is the same as the old path; nothing to do."
        new_full = self._in_scope_path(new_rel)
        if new_full is None:
            return f"Refusing to rename to {new_path!r}: outside the vault root or excluded by scope."
        if new_full.exists():
            return f"Refusing to rename to {new_rel!r}: a note already exists there."

        old_full = self._abs(old_rel)
        if old_full is None or not old_full.is_file():
            return f"(note not found on disk: {old_rel})"

        old_target = Path(old_rel).stem
        new_target = Path(new_rel).stem

        # 1. Rewrite inbound links in every other note BEFORE the move, so we
        # fail safely (no partial rename) if a rewrite errors out.
        notes = self._notes()
        rewritten: list[str] = []
        for n in notes:
            if n["path"] == old_rel:
                continue
            if not any(resolve_note(link, notes) == old_rel for link in (n.get("links") or [])):
                continue
            full = self._abs(n["path"])
            if full is None or not full.is_file():
                continue
            text = full.read_text(encoding="utf-8", errors="replace")
            updated = rewrite_wikilink_target(text, old_target, new_target)
            if updated != text:
                self._write_note_atomic(full, updated)
                rewritten.append(n["path"])

        # 2. Move the note file itself, then align its frontmatter title with
        # the new stem (only when the title was auto-derived, i.e. == old stem).
        new_full.parent.mkdir(parents=True, exist_ok=True)
        os.replace(old_full, new_full)
        try:
            moved_text = new_full.read_text(encoding="utf-8", errors="replace")
            updated_text = rename_note_title(moved_text, old_target, new_target)
            if updated_text != moved_text:
                self._write_note_atomic(new_full, updated_text)
        except OSError:
            pass  # the rename already succeeded; a title-fix failure is non-fatal

        head = f"Renamed {old_rel} → {new_rel}."
        if rewritten:
            head += f" Rewrote inbound links in {len(rewritten)} note(s): " + ", ".join(rewritten) + "."
        else:
            head += " No inbound links to rewrite."
        return head

    # -- formatting ----------------------------------------------------------

    def _format_head(self, n: dict, detail: bool = False) -> str:
        lines = [f"Note: {n['path']}"]
        if n.get("tags"):
            lines.append(f"tags: {', '.join(n['tags'])}")
        if n.get("aliases"):
            lines.append(f"aliases: {', '.join(n['aliases'])}")
        if n.get("headings"):
            shown = n["headings"] if detail else n["headings"][:_MAX_HEAD_LINES]
            lines.append(f"headings ({len(n['headings'])}): " + "; ".join(shown))
        if n.get("links"):
            lines.append(f"links to ({len(n['links'])}): " + ", ".join(n["links"][:15]))
        # first lines of the body
        full = self._abs(n["path"])
        if full is not None and full.is_file():
            body = full.read_text(encoding="utf-8", errors="replace")
            body_lines = [ln for ln in body.splitlines() if ln.strip()][:3]
            if body_lines and not detail:
                lines.append("preview: " + " | ".join(ln.strip()[:80] for ln in body_lines))
        return "\n".join(lines)
