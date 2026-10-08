"""Keyword store: the append-only journal and the entity map derived from it.

Per-agent flat files in one directory (``data/keywords/<agent>/``), the same
journal → derived map → watermark spine as the memory capability:

- ``keyword-journal.jsonl`` — append-only and lossless, one line per source
  item. Re-processing a source appends a NEW line with the same ``source_ref``;
  the latest line wins when the map is derived. Keywords are siblings of their
  source by ``source_ref`` and are never written onto the source itself.
- ``entity-map.json`` — the derived inverted index: entity → canonical name,
  aliases, and the sources that mention it. Rebuilt whole from the journal.

Nothing here calls a model. The curator agent (a later phase) produces the
records; this module only stores them and derives the map.
"""

from __future__ import annotations

import json
import os
import re
import threading
from collections.abc import Container, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from paf.capabilities._storage._atomic import read_json, write_atomic, write_json_atomic

JOURNAL_FILENAME = "keyword-journal.jsonl"
ENTITY_MAP_FILENAME = "entity-map.json"

_SEPARATORS = re.compile(r"[\s_\-]+")


def normalize_term(text: str) -> str:
    """The comparison form of an entity name or alias.

    Case-folded, with runs of whitespace, hyphens and underscores collapsed to a
    single space — so ``Memory-Extractor``, ``memory_extractor`` and
    ``memory extractor`` are the same term. Used to merge entities and dedupe
    aliases here, and by the hot-path matcher in a later phase.
    """
    return _SEPARATORS.sub(" ", str(text).casefold()).strip()


def _clean_terms(values: Iterable[Any]) -> list[str]:
    """Strip, drop empties, dedupe by normalized form (first spelling wins)."""
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value).strip()
        key = normalize_term(text)
        if key and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def _clean_entities(entities: Iterable[Any]) -> list[dict[str, Any]]:
    """Normalize raw entities to ``{"canonical": str, "aliases": [str]}``."""
    out: list[dict[str, Any]] = []
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        canonical = str(ent.get("canonical") or "").strip()
        if not normalize_term(canonical):
            continue
        aliases = [
            a
            for a in _clean_terms(ent.get("aliases") or [])
            if normalize_term(a) != normalize_term(canonical)
        ]
        out.append({"canonical": canonical, "aliases": aliases})
    return out


def make_record(
    source_ref: str,
    source_type: str,
    keywords: Iterable[str] = (),
    entities: Iterable[dict[str, Any]] = (),
    *,
    at: str | None = None,
) -> dict[str, Any]:
    """Build a journal record, normalizing keywords and entities.

    ``source_ref`` ties the record to its source (``skill:<id>``,
    ``note:<vault-path>``, ``memory:<line>``, ...). ``at`` defaults to now (UTC).
    """
    ref = str(source_ref).strip()
    if not ref:
        raise ValueError("source_ref must be non-empty")
    return {
        "source_ref": ref,
        "source_type": str(source_type).strip(),
        "at": at or datetime.now(UTC).isoformat(timespec="seconds"),
        "keywords": _clean_terms(keywords),
        "entities": _clean_entities(entities),
    }


def _content(record: dict[str, Any]) -> str:
    """The record's identity for idempotency: everything except ``at``."""
    return json.dumps(
        {
            "source_type": record.get("source_type"),
            "keywords": record.get("keywords") or [],
            "entities": record.get("entities") or [],
        },
        sort_keys=True,
        ensure_ascii=False,
    )


class KeywordJournalStore:
    """Append-only keyword journal (one JSON record per line).

    ``directory`` is the per-agent keyword dir; the journal is
    ``<directory>/keyword-journal.jsonl``. Appends rewrite the file atomically
    (staging + ``os.replace``), so a crash never leaves a torn line, and
    :meth:`read_all` skips any corrupt line it does find.
    """

    def __init__(self, directory: str | os.PathLike) -> None:
        self._dir = Path(directory)
        self._path = self._dir / JOURNAL_FILENAME
        self._lock = threading.RLock()

    @property
    def path(self) -> Path:
        return self._path

    def read_all(self) -> list[dict[str, Any]]:
        """Every well-formed record in append order (torn/invalid lines skipped)."""
        with self._lock:
            if not self._path.exists():
                return []
            out: list[dict[str, Any]] = []
            with self._path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(rec, dict) and str(rec.get("source_ref") or ""):
                        out.append(rec)
            return out

    def latest(self) -> dict[str, dict[str, Any]]:
        """The current record per ``source_ref`` (latest line wins)."""
        return {rec["source_ref"]: rec for rec in self.read_all()}

    def keyed_refs(self) -> set[str]:
        """Every ``source_ref`` that has at least one journal line."""
        return set(self.latest())

    def append(self, record: dict[str, Any]) -> bool:
        """Append ``record``; ``False`` if it is a no-op.

        Idempotent by ``source_ref``: a record whose content (everything but
        ``at``) equals the source's latest line is not written. New content for
        the same ref appends a new line — the old one stays (lossless).
        """
        ref = str(record.get("source_ref") or "")
        if not ref:
            raise ValueError("record needs a non-empty source_ref")
        with self._lock:
            current = self.latest().get(ref)
            if current is not None and _content(current) == _content(record):
                return False
            existing = ""
            if self._path.exists():
                existing = self._path.read_text(encoding="utf-8")
                if existing and not existing.endswith("\n"):
                    existing += "\n"  # a torn trailing line must not glue on
            line = json.dumps(record, ensure_ascii=False) + "\n"
            write_atomic(self._path, existing + line)
            return True


def derive_entity_map(
    records: Iterable[dict[str, Any]],
    live_refs: Container[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Derive the entity map from journal records. Pure.

    - Latest record per ``source_ref`` wins.
    - ``live_refs`` (when given) drops sources that no longer exist (a deleted
      skill, a moved note); the journal itself is never pruned.
    - Entities merge by :func:`normalize_term` of the canonical name. The
      canonical spelling shown is the first seen in ``source_ref`` order, so
      the result does not depend on journal order. Aliases are unioned,
      deduped, and never repeat the canonical.

    Returns ``{canonical: {"canonical", "aliases", "sources"}}`` sorted by
    normalized name; ``aliases`` and ``sources`` are sorted too.
    """
    latest: dict[str, dict[str, Any]] = {}
    for rec in records:
        ref = str(rec.get("source_ref") or "")
        if ref:
            latest[ref] = rec

    merged: dict[str, dict[str, Any]] = {}
    for ref in sorted(latest):
        if live_refs is not None and ref not in live_refs:
            continue
        for ent in _clean_entities(latest[ref].get("entities") or []):
            key = normalize_term(ent["canonical"])
            slot = merged.setdefault(
                key,
                {"canonical": ent["canonical"], "aliases": {}, "sources": set()},
            )
            slot["sources"].add(ref)
            for alias in ent["aliases"]:
                akey = normalize_term(alias)
                if akey != key:
                    slot["aliases"].setdefault(akey, alias)

    out: dict[str, dict[str, Any]] = {}
    for key in sorted(merged):
        slot = merged[key]
        out[slot["canonical"]] = {
            "canonical": slot["canonical"],
            "aliases": [slot["aliases"][k] for k in sorted(slot["aliases"])],
            "sources": sorted(slot["sources"]),
        }
    return out


def write_entity_map(
    directory: str | os.PathLike, entity_map: dict[str, dict[str, Any]]
) -> None:
    """Persist the entity map atomically as ``<directory>/entity-map.json``."""
    write_json_atomic(Path(directory) / ENTITY_MAP_FILENAME, entity_map)


def read_entity_map(directory: str | os.PathLike) -> dict[str, dict[str, Any]]:
    """Load the entity map (``{}`` if absent or corrupt)."""
    data = read_json(Path(directory) / ENTITY_MAP_FILENAME, default={})
    return data if isinstance(data, dict) else {}
