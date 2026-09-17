"""Tests for the skills index store — persistence + lazy freshness.

Covers the flat-file skills index (``data/.skills-index/<sha256(root)>/``): the
per-skill label rows in ``index.json``, the freshness snapshot, and the
corpus's lazy rebuild when the snapshot detects a change. Uses the checked-in
``tests/fixtures/skills/`` corpus as the deterministic test library; tests that
mutate a library copy it to a pytest tmp dir first so the fixture stays intact.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from paf.capabilities.skills._corpus import HintCorpus
from paf.capabilities.skills._store import (
    FlatFileSkillsStorage,
    build_index,
    index_dir_for_root,
    sync_changed,
)

FIXTURE_SKILLS = str(Path(__file__).resolve().parent / "fixtures" / "skills")

# The fixture library's canonical skill ids (5 skills across 3 categories).
FIXTURE_IDS = {
    "code-review",
    "mlops/lean-backtest",
    "mlops/tools",
    "research/tools",
    "vectorized-backtest",
}


def _storage(tmp_path: Path) -> FlatFileSkillsStorage:
    """A storage whose index root is a pytest tmp dir (hermetic, no repo writes)."""
    return FlatFileSkillsStorage(index_root=tmp_path / "idx")


def _copy_library(tmp_path: Path) -> Path:
    lib = tmp_path / "lib"
    shutil.copytree(FIXTURE_SKILLS, lib)
    return lib


# --- index_dir_for_root -----------------------------------------------------


def test_index_dir_keyed_by_root_hash(tmp_path: Path) -> None:
    a = tmp_path / "lib-a"
    b = tmp_path / "lib-b"
    a.mkdir()
    b.mkdir()
    da = index_dir_for_root(a)
    db = index_dir_for_root(b)
    assert da != db  # different library roots -> different index dirs
    assert da.parent == db.parent  # same repo-owned base
    assert da.parent.name == ".skills-index"


def test_index_dir_is_repo_owned_not_in_library(tmp_path: Path) -> None:
    lib = tmp_path / "lib"
    lib.mkdir()
    d = index_dir_for_root(lib)
    # The index lives under the paf repo's data dir, NOT co-located in the library.
    assert not str(d).startswith(str(lib))


# --- build_index ------------------------------------------------------------


def test_build_index_writes_label_rows_and_snapshot(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    corpus = HintCorpus(roots=FIXTURE_SKILLS)
    payload = build_index(corpus.docs, storage, FIXTURE_SKILLS)

    idx = storage._dir(FIXTURE_SKILLS)
    assert (idx / "index.json").exists()
    assert (idx / "snapshot.json").exists()

    assert payload["version"] == 1
    assert payload["skill_count"] == 5
    assert {s["skill_id"] for s in payload["skills"]} == FIXTURE_IDS

    row = next(s for s in payload["skills"] if s["skill_id"] == "code-review")
    assert row["name"] == "code-review"
    assert row["category"] == ""
    assert row["description"].startswith("Review a code change")
    assert row["tags"] == ["review", "code", "quality", "pr", "conventions"]
    assert row["hermes_tags"] == []
    assert row["keywords"] == []
    assert row["hint"] == "consider: code-review"
    assert isinstance(row["resources"], list)


def test_build_index_skips_skills_outside_root(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    corpus = HintCorpus(roots=[FIXTURE_SKILLS, str(tmp_path / "other")])
    # Only the skills under FIXTURE_SKILLS are indexed for that root.
    payload = build_index(corpus.docs, storage, FIXTURE_SKILLS)
    assert payload["skill_count"] == 5


# --- sync_changed -----------------------------------------------------------


def test_sync_changed_reports_nothing_when_fresh(tmp_path: Path) -> None:
    lib = _copy_library(tmp_path)
    storage = _storage(tmp_path)
    HintCorpus(roots=str(lib), storage=storage)
    assert sync_changed(lib, storage) == set()


def test_sync_changed_detects_modified_skill(tmp_path: Path) -> None:
    lib = _copy_library(tmp_path)
    storage = _storage(tmp_path)
    HintCorpus(roots=str(lib), storage=storage)

    target = lib / "code-review" / "SKILL.md"
    target.write_text(target.read_text() + "\n# extra\n")
    changed = sync_changed(lib, storage)
    assert "code-review/SKILL.md" in changed


def test_sync_changed_detects_new_skill(tmp_path: Path) -> None:
    lib = _copy_library(tmp_path)
    storage = _storage(tmp_path)
    HintCorpus(roots=str(lib), storage=storage)

    new_skill = lib / "new-skill" / "SKILL.md"
    new_skill.parent.mkdir(parents=True, exist_ok=True)
    new_skill.write_text("---\ndescription: brand new skill\n---\n# New\n")
    changed = sync_changed(lib, storage)
    assert "new-skill/SKILL.md" in changed


# --- corpus persistence + lazy freshness ------------------------------------


def test_corpus_persists_index_on_construction(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    HintCorpus(roots=FIXTURE_SKILLS, storage=storage)
    assert (tmp_path / "idx" / "index.json").exists()
    assert (tmp_path / "idx" / "snapshot.json").exists()


def test_index_not_co_located_in_library(tmp_path: Path) -> None:
    lib = _copy_library(tmp_path)
    storage = _storage(tmp_path)
    HintCorpus(roots=str(lib), storage=storage)
    assert not (lib / ".skills-index").exists()
    assert not (lib / "data").exists()


def test_corpus_rebuilds_when_snapshot_detects_change(tmp_path: Path) -> None:
    lib = _copy_library(tmp_path)
    storage = _storage(tmp_path)
    corpus = HintCorpus(roots=str(lib), storage=storage)
    assert len(corpus.docs) == 5

    new_skill = lib / "new-skill" / "SKILL.md"
    new_skill.parent.mkdir(parents=True, exist_ok=True)
    new_skill.write_text("---\ndescription: brand new skill\n---\n# New\n")

    # Accessing docs triggers the lazy freshness check -> rebuild picks it up.
    assert any(d.skill_id == "new-skill" for d in corpus.docs)


def test_index_root_override_wins(tmp_path: Path) -> None:
    storage = FlatFileSkillsStorage(index_root=tmp_path / "custom")
    HintCorpus(roots=FIXTURE_SKILLS, storage=storage)
    assert (tmp_path / "custom" / "index.json").exists()
    assert (tmp_path / "custom" / "snapshot.json").exists()
