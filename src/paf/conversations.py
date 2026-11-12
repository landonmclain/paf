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
slugs). An optional ``root`` selects a local working directory. An archived
project is hidden from listings but its slug stays taken.

**Conversation meta** — per-agent map ``conversation_id -> {title?, project?,
hidden?, updated_at}``. The filename stays ``conversation-titles.json`` for backward
compatibility with existing data. Keyed by the bare lane id (the per-agent
file already scopes it). Clearing a title removes only the ``title`` key; an
entry is dropped only when it carries no title, project, or hidden flag.
``hidden`` only hides a lane from the default sidebar view; nothing is deleted.
"""

from __future__ import annotations

import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from paf.capabilities._storage import FlatFileStorage

_META_KIND = "conversation-titles"
_META_FIELDS = ("title", "project", "hidden")
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


def _validated_root(root: str) -> str:
    """Validate a local directory, retaining a portable ``~`` in storage."""
    if not isinstance(root, str):
        raise ValueError("project root must be a string")
    root = root.strip()
    try:
        path = Path(root).expanduser()
        valid = bool(root) and path.is_absolute() and path.is_dir()
    except (OSError, RuntimeError, ValueError):
        valid = False
    if not valid:
        raise ValueError(
            "project root must be an absolute path to an existing directory"
        )
    return root


class ProjectStore:
    """The global project registry: ``slug -> {name, description, root?, archived, created_at}``."""

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

    def create(
        self, name: str, description: str = "", root: str | None = None
    ) -> dict[str, Any]:
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
        if root is not None:
            projects[slug]["root"] = _validated_root(root)
        self._save(projects)
        return self._row(slug, projects[slug])

    def update(
        self,
        slug: str,
        *,
        name: str | None = None,
        description: str | None = None,
        archived: bool | None = None,
        root: str | None = None,
    ) -> dict[str, Any]:
        """Edit display fields, root, or (un)archive; the slug never changes.

        Empty root clears the setting; None preserves it.
        """
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
        if root is not None:
            if root == "":
                entry.pop("root", None)
            else:
                entry["root"] = _validated_root(root)
        self._save(projects)
        return self._row(slug, entry)


class ConversationMetaStore:
    """One agent's conversation meta: ``conversation_id -> {title?, project?, hidden?, updated_at}``."""

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

    def _put(self, conversation_id: str, **fields: str | bool | None) -> dict[str, Any]:
        """Set (truthy) or clear (falsy) fields; drop the entry once it is empty."""
        meta = self.load()
        entry = dict(meta.get(conversation_id) or {})
        for key, value in fields.items():
            if value:
                entry[key] = value
            else:
                entry.pop(key, None)
        if any(entry.get(k) for k in _META_FIELDS):
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

    def set_hidden(self, conversation_id: str, hidden: bool) -> dict[str, Any]:
        """Hide a lane from the default sidebar view, or show it again."""
        return self._put(conversation_id, hidden=bool(hidden))

    def project_for(self, conversation_id: str) -> str | None:
        project = self.get(conversation_id).get("project")
        return project if isinstance(project, str) and project else None


def lane_id(conversation_id: str, profile: str | None) -> str:
    """Strip a ``{profile}:`` scope prefix back to the bare lane id.

    ``Session.scoped_id`` stores conversations as ``{profile}:{cid}``
    (``demo:paf-e4b3ff13d560``), while conversation meta is keyed by the bare
    lane id. Only the known profile prefix is stripped — lane ids are arbitrary
    (``chat.py --id``) and may themselves contain a colon. An id without that
    prefix is returned unchanged, so bare ids pass straight through.
    """
    prefix = f"{profile}:" if profile else ""
    if prefix and conversation_id.startswith(prefix):
        return conversation_id[len(prefix) :]
    return conversation_id


def project_for_conversation(
    agent: str,
    conversation_id: str | None,
    *,
    profile: str | None = None,
    store: ConversationMetaStore | None = None,
) -> str | None:
    """The project slug assigned to a conversation, or ``None``.

    The single lookup every consumer (memory extraction, injection, backfill)
    uses, so none of them handles the scope prefix itself. ``conversation_id``
    may be profile-scoped (what ``ctx.conversation_id``, the background
    triggers, and persisted runs carry) or bare. ``profile`` is the session's
    profile name; when omitted it is resolved from the agent's spec (its
    ``profile:`` key) — pass it explicitly when the session runs under an
    override (``chat.py --profile``).
    """
    if not conversation_id:
        return None
    if profile is None:
        from paf.specs import resolve_profile_name

        profile = resolve_profile_name(agent)
    meta = store if store is not None else default_conversation_meta_store(agent)
    return meta.project_for(lane_id(conversation_id, profile))


async def persisted_lanes(profile: str | None, step_store=None) -> dict[str, float]:
    """Lane id -> newest run start (epoch seconds) for one profile's lanes.

    Reads the persisted step store, keeping only runs scoped ``{profile}:`` so
    CLI/smoke/benchmark/UUID lanes of other profiles in the shared store never
    surface. ``step_store`` defaults to the canonical store
    (``paf.capabilities.persistence._build_store``).
    """
    if step_store is None:
        from paf.capabilities.persistence import _build_store

        step_store = _build_store()
    prefix = f"{profile}:" if profile else ""
    lanes: dict[str, float] = {}
    for run in await step_store.list_runs():
        cid = run.conversation_id
        if not cid or (prefix and not cid.startswith(prefix)):
            continue
        lane = cid[len(prefix) :]
        started = run.started_at.timestamp() if run.started_at else 0.0
        lanes[lane] = max(started, lanes.get(lane, 0.0))
    return lanes


def overlay_meta(lanes: list[dict[str, Any]], meta: dict[str, dict[str, Any]]) -> None:
    """Fill ``title`` (when user-set), ``project``, ``hidden`` on each lane in place."""
    for lane in lanes:
        saved = meta.get(lane["conversation_id"])
        saved = saved if isinstance(saved, dict) else {}
        if saved.get("title"):
            lane["title"] = saved["title"]
        lane["project"] = saved.get("project") or None
        lane["hidden"] = bool(saved.get("hidden"))


async def list_conversations(
    agent: str,
    *,
    profile: str | None = None,
    step_store=None,
    meta_store: ConversationMetaStore | None = None,
) -> list[dict[str, Any]]:
    """An agent's persisted conversations, newest first, with their meta.

    Each row: ``{conversation_id, title, project, hidden, updated_at}`` where
    ``title`` is the user-set name or ``None``. Persisted lanes only — a lane
    appears once it has at least one run. ``profile`` defaults to the agent
    spec's ``profile:`` key, matching how ``Session`` scopes its lanes.
    """
    if profile is None:
        from paf.specs import resolve_profile_name

        profile = resolve_profile_name(agent)
    lanes = [
        {"conversation_id": lane, "title": None, "updated_at": started}
        for lane, started in (await persisted_lanes(profile, step_store)).items()
    ]
    meta = (
        meta_store if meta_store is not None else default_conversation_meta_store(agent)
    )
    overlay_meta(lanes, meta.load())
    lanes.sort(key=lambda r: r["updated_at"] or 0.0, reverse=True)
    return lanes


def default_project_store() -> ProjectStore:
    """The registry at ``<project_root>/data/projects/projects.json``."""
    from paf.config import DATA_DIR

    return ProjectStore(FlatFileStorage(DATA_DIR))


def default_conversation_meta_store(agent: str) -> ConversationMetaStore:
    """``<project_root>/data/conversations/<agent>/conversation-titles.json``."""
    from paf.config import DATA_DIR

    return ConversationMetaStore(FlatFileStorage(DATA_DIR / "conversations"), agent)
