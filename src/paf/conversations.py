"""Conversation metadata + the project registry — small user-owned stores.

Two flat JSON maps behind the shared storage seam
(``paf.capabilities._storage``), deliberately OUTSIDE ``data/memory/``: they
are user/conversation concerns, not agent memory, so they must not be coupled
to the memory capability. Both the SSE transport (``ui/sse.py``) and the CLI
(``ui/chat.py``) read and write them; memory consolidation reads them (a
conversation's project scopes what the extractor files facts under).

Layout::

    data/
      projects/projects.json                         # global project registry
      conversations/<agent>/conversation-titles.json # per-agent conversation meta

**Project registry** — global, not per-agent: projects belong to the user.
Keyed by a permanent ``slug`` derived from the name at creation. Memory records
store the slug, so ``name``/``description`` can change freely without touching
the journal. Projects are archived, never deleted (journal records point at
slugs); an archived project is hidden from listings but its slug stays taken.

**Conversation meta** — per-agent map ``conversation_id -> {title?, project?,
updated_at}``. The filename stays ``conversation-titles.json`` for backward
compatibility with existing data. Keyed by the bare lane id (the per-agent
file already scopes it). Clearing a title removes only the ``title`` key; an
entry is dropped only when it carries neither a title nor a project.
"""

from __future__ import annotations

import re
import time
from datetime import UTC, datetime
from typing import Any

from paf.capabilities._storage import FlatFileStorage

_META_KIND = "conversation-titles"
_PROJECTS_KIND = "projects"
_PROJECTS_KEY = "projects"

# Reserved: the memory extractor uses `none` for user-wide (no project) facts.
_RESERVED_SLUGS = frozenset({"none"})


class ProjectExistsError(ValueError):
    """A project with this slug already exists (archived ones included)."""


class ProjectNotFoundError(KeyError):
    """No project with this slug."""


def slugify(name: str) -> str:
    """Lowercase, runs of non-alphanumerics -> ``-``, trimmed."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class ProjectStore:
    """The global project registry: ``slug -> {name, description, archived, created_at}``."""

    def __init__(self, storage: FlatFileStorage) -> None:
        self._storage = storage

    def _load(self) -> dict[str, dict[str, Any]]:
        data = self._storage.read(_PROJECTS_KIND, _PROJECTS_KEY)
        return data if isinstance(data, dict) else {}

    def _save(self, projects: dict[str, dict[str, Any]]) -> None:
        self._storage.write(_PROJECTS_KIND, _PROJECTS_KEY, projects)

    @staticmethod
    def _row(slug: str, entry: dict[str, Any]) -> dict[str, Any]:
        return {"slug": slug, **entry}

    def list(self, *, include_archived: bool = False) -> list[dict[str, Any]]:
        """Projects sorted by name; archived ones only when asked."""
        rows = [
            self._row(slug, entry)
            for slug, entry in self._load().items()
            if include_archived or not entry.get("archived")
        ]
        rows.sort(key=lambda r: str(r.get("name", r["slug"])).lower())
        return rows

    def get(self, slug: str) -> dict[str, Any] | None:
        entry = self._load().get(slug)
        return self._row(slug, entry) if entry is not None else None

    def create(self, name: str, description: str = "") -> dict[str, Any]:
        """Register a project; the slug is derived from ``name`` and permanent."""
        name = name.strip()
        slug = slugify(name)
        if not slug:
            raise ValueError("project name must contain a letter or digit")
        if slug in _RESERVED_SLUGS:
            raise ValueError(f"{slug!r} is reserved")
        projects = self._load()
        if slug in projects:
            raise ProjectExistsError(slug)
        projects[slug] = {
            "name": name,
            "description": description.strip(),
            "archived": False,
            "created_at": _utc_now(),
        }
        self._save(projects)
        return self._row(slug, projects[slug])

    def update(
        self,
        slug: str,
        *,
        name: str | None = None,
        description: str | None = None,
        archived: bool | None = None,
    ) -> dict[str, Any]:
        """Edit display fields or (un)archive. The slug never changes."""
        projects = self._load()
        entry = projects.get(slug)
        if entry is None:
            raise ProjectNotFoundError(slug)
        if name is not None:
            if not name.strip():
                raise ValueError("project name must be non-empty")
            entry["name"] = name.strip()
        if description is not None:
            entry["description"] = description.strip()
        if archived is not None:
            entry["archived"] = bool(archived)
        self._save(projects)
        return self._row(slug, entry)


class ConversationMetaStore:
    """One agent's conversation meta: ``conversation_id -> {title?, project?, updated_at}``."""

    def __init__(self, storage: FlatFileStorage, agent: str) -> None:
        self._storage = storage
        self._agent = agent

    def load(self) -> dict[str, dict[str, Any]]:
        """The whole map (missing/corrupt -> ``{}``); never raises."""
        data = self._storage.read(_META_KIND, self._agent)
        return data if isinstance(data, dict) else {}

    def get(self, conversation_id: str) -> dict[str, Any]:
        entry = self.load().get(conversation_id)
        return entry if isinstance(entry, dict) else {}

    def _put(self, conversation_id: str, **fields: str | None) -> dict[str, Any]:
        """Set (value) or clear (None) fields; drop the entry once it is empty."""
        meta = self.load()
        entry = dict(meta.get(conversation_id) or {})
        for key, value in fields.items():
            if value:
                entry[key] = value
            else:
                entry.pop(key, None)
        if entry.get("title") or entry.get("project"):
            entry["updated_at"] = time.time()
            meta[conversation_id] = entry
        else:
            meta.pop(conversation_id, None)
            entry = {}
        self._storage.write(_META_KIND, self._agent, meta)
        return entry

    def set_title(self, conversation_id: str, title: str | None) -> dict[str, Any]:
        """Set a title, or clear it (None/empty) without touching the project."""
        return self._put(conversation_id, title=(title or "").strip() or None)

    def set_project(self, conversation_id: str, project: str | None) -> dict[str, Any]:
        """Assign a project slug, or unassign (None). Callers validate the slug."""
        return self._put(conversation_id, project=project or None)

    def project_for(self, conversation_id: str) -> str | None:
        project = self.get(conversation_id).get("project")
        return project if isinstance(project, str) and project else None


def default_project_store() -> ProjectStore:
    """The registry at ``<project_root>/data/projects/projects.json``."""
    from paf.config import PROJECT_ROOT

    return ProjectStore(FlatFileStorage(PROJECT_ROOT / "data"))


def default_conversation_meta_store(agent: str) -> ConversationMetaStore:
    """``<project_root>/data/conversations/<agent>/conversation-titles.json``."""
    from paf.config import PROJECT_ROOT

    return ConversationMetaStore(
        FlatFileStorage(PROJECT_ROOT / "data" / "conversations"), agent
    )
