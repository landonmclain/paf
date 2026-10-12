"""paf.clock (store UTC, show local) and the Workspace context block."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from paf import clock
from paf.capabilities.memory._journal import _normalize_born_on
from paf.capabilities.workspace import Workspace, build_workspace


@pytest.fixture()
def chicago(monkeypatch):
    monkeypatch.setattr(clock.settings, "timezone", "America/Chicago")
    clock.local_zone.cache_clear()
    yield
    clock.local_zone.cache_clear()


def test_local_zone_from_setting(chicago) -> None:
    assert clock.zone_name() == "America/Chicago"
    assert clock.now_local().tzinfo == ZoneInfo("America/Chicago")


def test_bad_zone_falls_back(monkeypatch) -> None:
    monkeypatch.setattr(clock.settings, "timezone", "Not/AZone")
    clock.local_zone.cache_clear()
    try:
        assert clock.local_zone() is not None  # system zone or UTC, never raises
    finally:
        clock.local_zone.cache_clear()


def test_local_iso_converts_utc_and_naive(chicago) -> None:
    assert clock.local_iso("2026-10-02T02:30:00+00:00") == "2026-10-01T21:30:00-05:00"
    assert clock.local_iso("2026-10-02T02:30:00") == "2026-10-01T21:30:00-05:00"
    assert clock.local_iso("2026-01-15T12:00:00Z") == "2026-01-15T06:00:00-06:00"  # CST
    assert clock.local_iso("not a date") == "not a date"
    assert clock.local_iso(None) == ""


def test_born_on_is_stored_as_utc_so_lexical_order_is_chronological() -> None:
    evening_central = _normalize_born_on("2026-10-01T21:30:00-05:00")
    next_morning_utc = _normalize_born_on("2026-10-02T01:00:00+00:00")
    assert evening_central == "2026-10-02T02:30:00+00:00"
    # 01:00Z is EARLIER than 21:30 CDT (= 02:30Z): lexical order must agree.
    assert next_morning_utc < evening_central
    assert (
        _normalize_born_on(
            datetime(2026, 10, 1, 21, 30, tzinfo=ZoneInfo("America/Chicago"))
        )
        == evening_central
    )
    assert (
        _normalize_born_on(datetime(2026, 10, 1, 9, 0)) == "2026-10-01T09:00:00+00:00"
    )
    assert _normalize_born_on("") == ""


class _Cap:
    def __init__(self, **attrs):
        self.__dict__.update(attrs)


def _named(name: str, **attrs):
    return type(name, (_Cap,), {})(**attrs)


def test_workspace_lists_only_built_roots_and_example(chicago, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    caps = [
        _named("FileSystem", root_dir=str(tmp_path)),
        _named("CodebaseMap", scan_root=str(repo)),
        _named("Shell", cwd=str(tmp_path)),
    ]
    text = build_workspace(caps, ["Projects live under ~/projects."]).render()
    assert text.startswith("<workspace>") and text.endswith("</workspace>")
    assert "File tools (" in text and f"paths relative to {tmp_path.resolve()}" in text
    assert f"repo {repo.resolve()}" in text
    assert "Git" not in text and "Vault" not in text  # not built -> not listed
    assert "Absolute paths work with every tool above" in text
    assert f"→ {repo.resolve() / 'src' / 'app.py'} for file tools" in text
    assert "America/Chicago" in text and ("CDT" in text or "CST" in text)
    assert "Projects live under ~/projects." in text


def test_workspace_without_roots_still_names_the_zone(chicago) -> None:
    text = Workspace().render()
    assert "America/Chicago" in text
    assert "Tools resolve" not in text


def test_workspace_hiding_paths_names_no_host_path(chicago, tmp_path: Path) -> None:
    caps = [
        _named("FileSystem", root_dir=str(tmp_path)),
        _named("Python", cwd=str(tmp_path)),
    ]
    text = build_workspace(caps, expose_paths=False).render()
    assert str(tmp_path) not in text and str(Path.home()) not in text
    assert "Home:" not in text and "Absolute paths work" not in text
    assert "relative to the workspace root" in text
    assert "programs run in the workspace root" in text
    assert "America/Chicago" in text  # timestamps still need their zone


def test_spec_workspace_expose_paths(tmp_path: Path) -> None:
    from paf.specs import _spec_workspace_config

    hidden = tmp_path / "a.yaml"
    hidden.write_text("workspace:\n  expose_paths: false\n")
    shown = tmp_path / "b.yaml"
    shown.write_text("workspace:\n  expose_paths: true\n  notes: [hi]\n")
    assert _spec_workspace_config(hidden) == {"expose_paths": False}
    assert _spec_workspace_config(shown) == {"notes": ["hi"]}  # default stays true
