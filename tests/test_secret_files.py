"""Credential files stay out of reach on BOTH sides of the sandbox boundary.

The Docker sandbox blanks them (empty overlay mounted over the real file) and
the host-side file tools refuse to touch them; both are derived from the single
list in ``_secret_files.py`` and must agree.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import pytest

from paf.capabilities._sandbox import DockerExecutor
from paf.capabilities._sandbox._docker import docker_available, find_excluded
from paf.capabilities._secret_files import (
    SECRET_DIRS,
    SECRET_FILE_GLOBS,
    fs_denied_patterns,
    sandbox_excludes,
)
from paf.capabilities.build import DEFAULT_EXCLUDE, _build_filesystem
from paf.capabilities.fs import FileSystem
from paf.specs import _spec_filesystem_config


def _run(coro):
    return asyncio.run(coro)


def _tree(root: Path) -> Path:
    """A workspace with secrets at several depths, plus look-alike source files."""
    files = {
        ".env": "TOP=secret-top",
        "sub/.env": "NESTED=secret-nested",
        "a/b/c/.env.production": "DEEP=secret-deep",
        "sub/server.pem": "-----BEGIN KEY-----",
        "deploy.key": "keymaterial",
        ".ssh/id_ed25519": "sshsecret",
        "home/.aws/credentials": "aws_secret=abc",
        ".git/HEAD": "ref: refs/heads/main",
        "config/secrets.yaml": "token: abc",
        "code.py": "print('hi')",
        "sub/secrets_manager.py": "def load(): ...",  # source: must stay visible
        "docs/keyboard.md": "not a key file",
        "node_modules/pkg/.env": "PRUNED=1",
    }
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


# --- one list, two enforcement points ---------------------------------------------
def test_both_sides_derive_from_the_one_list():
    assert tuple(DEFAULT_EXCLUDE) == sandbox_excludes()
    assert set(SECRET_FILE_GLOBS) <= set(sandbox_excludes())
    assert set(SECRET_DIRS) <= set(sandbox_excludes())


def _sample(glob: str) -> str:
    return glob.replace("*", "x")


@pytest.mark.parametrize("depth", ["", "sub/", "a/b/c/"])
@pytest.mark.parametrize("glob", SECRET_FILE_GLOBS)
def test_every_secret_glob_is_caught_by_both_mechanisms(tmp_path, glob, depth):
    """No name may be protected on one side only."""
    rel = f"{depth}{_sample(glob)}"
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("secret")
    assert path in find_excluded(tmp_path, sandbox_excludes()), f"sandbox misses {rel}"
    fs = FileSystem(
        root_dir=tmp_path, denied_patterns=fs_denied_patterns()
    ).get_toolset()
    assert not fs._is_accessible(rel), f"file tools allow {rel}"  # noqa: SLF001


@pytest.mark.parametrize("depth", ["", "sub/"])
@pytest.mark.parametrize("directory", SECRET_DIRS)
def test_secret_directories_are_caught_by_both_mechanisms(tmp_path, directory, depth):
    inner = tmp_path / depth / directory / "nested" / "thing"
    inner.parent.mkdir(parents=True)
    inner.write_text("secret")
    assert tmp_path / depth / directory in find_excluded(tmp_path, sandbox_excludes())
    fs = FileSystem(
        root_dir=tmp_path, denied_patterns=fs_denied_patterns()
    ).get_toolset()
    assert not fs._is_accessible(f"{depth}{directory}/nested/thing")  # noqa: SLF001


# --- the sandbox walk ---------------------------------------------------------------
def test_find_excluded_matches_names_at_any_depth(tmp_path):
    root = _tree(tmp_path)
    found = {
        p.relative_to(root).as_posix() for p in find_excluded(root, DEFAULT_EXCLUDE)
    }
    assert {
        ".env",
        "sub/.env",
        "a/b/c/.env.production",
        "sub/server.pem",
        "deploy.key",
        ".ssh",
        "home/.aws",
        ".git",
        "config/secrets.yaml",
    } <= found


def test_find_excluded_leaves_source_code_alone(tmp_path):
    root = _tree(tmp_path)
    found = {
        p.relative_to(root).as_posix() for p in find_excluded(root, DEFAULT_EXCLUDE)
    }
    assert "code.py" not in found
    assert "sub/secrets_manager.py" not in found
    assert "docs/keyboard.md" not in found


def test_a_matched_directory_is_returned_whole_not_descended(tmp_path):
    (tmp_path / ".git" / "modules" / "x").mkdir(parents=True)
    (tmp_path / ".git" / "modules" / "x" / ".env").write_text("s")
    found = find_excluded(tmp_path, DEFAULT_EXCLUDE)
    assert found == [tmp_path / ".git"]


def test_vendored_trees_are_pruned(tmp_path):
    """Documented trade-off: node_modules/virtualenvs are not walked (cost, and
    legitimate CA bundles), so a .env inside one is NOT blanked."""
    root = _tree(tmp_path)
    found = {
        p.relative_to(root).as_posix() for p in find_excluded(root, DEFAULT_EXCLUDE)
    }
    assert "node_modules/pkg/.env" not in found


def test_symlinked_alias_inside_the_workspace_is_followed_to_the_target(tmp_path):
    (tmp_path / "real_secret.txt").write_text("s")
    (tmp_path / ".env").symlink_to("real_secret.txt")
    found = find_excluded(tmp_path, DEFAULT_EXCLUDE)
    assert found == [tmp_path / "real_secret.txt"]  # the content is what gets blanked


def test_symlink_pointing_outside_the_workspace_is_skipped(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("s")
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / ".env").symlink_to(outside)
    assert find_excluded(ws, DEFAULT_EXCLUDE) == []


def test_path_patterns_keep_their_original_glob_semantics(tmp_path):
    (tmp_path / "keep").mkdir()
    (tmp_path / "keep" / "token.txt").write_text("s")
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "token.txt").write_text("s")
    found = find_excluded(tmp_path, ["keep/token.txt"])
    assert found == [tmp_path / "keep" / "token.txt"]


# --- the host-side file tools --------------------------------------------------------
def _fs(root: Path):
    return _build_filesystem(
        None, {"root_dir": str(root), "deny_secret_files": True}
    ).get_toolset()


def _call(coro):
    try:
        return str(_run(coro))
    except Exception as exc:  # noqa: BLE001 - the message is what matters
        return f"ERROR {exc}"


@pytest.mark.parametrize(
    "rel",
    [
        ".env",
        "sub/.env",
        "a/b/c/.env.production",
        "sub/server.pem",
        "deploy.key",
        ".ssh/id_ed25519",
        "home/.aws/credentials",
        ".git/HEAD",
        "config/secrets.yaml",
    ],
)
def test_file_tools_refuse_to_read_or_write_secrets(tmp_path, rel):
    root = _tree(tmp_path)
    fs = _fs(root)
    original = (root / rel).read_text()
    read = _call(fs.read_file(rel))
    assert read.startswith("ERROR") and "denied" in read and original not in read
    wrote = _call(fs.write_file(rel, "overwritten"))
    assert wrote.startswith("ERROR")
    assert (root / rel).read_text() == original


def test_file_tools_still_serve_source_code(tmp_path):
    root = _tree(tmp_path)
    fs = _fs(root)
    assert "print('hi')" in _call(fs.read_file("code.py"))
    assert "def load" in _call(fs.read_file("sub/secrets_manager.py"))
    assert "not a key file" in _call(fs.read_file("docs/keyboard.md"))


def test_secrets_are_invisible_to_listing_search_and_find(tmp_path):
    root = _tree(tmp_path)
    fs = _fs(root)
    listing = _call(fs.list_directory("."))
    assert ".env" not in listing and ".ssh" not in listing and ".git" not in listing
    assert "code.py" in listing
    for needle in (
        "secret-top",
        "secret-nested",
        "secret-deep",
        "sshsecret",
        "aws_secret",
    ):
        assert needle not in _call(fs.search_files(needle)), needle
    found = _call(fs.find_files("**/*"))
    for name in (".env", "server.pem", "id_ed25519", "credentials", "secrets.yaml"):
        assert name not in found, name
    assert "code.py" in found


def test_denial_is_opt_in_so_other_agents_are_unchanged(tmp_path):
    root = _tree(tmp_path)
    plain = _build_filesystem(None, {"root_dir": str(root)}).get_toolset()
    assert "TOP=secret-top" in _call(plain.read_file(".env"))  # readable, as before


def test_extra_denied_patterns_compose_with_the_secret_set(tmp_path):
    (tmp_path / "data.sqlite").write_text("db")
    (tmp_path / ".env").write_text("s")
    fs = _build_filesystem(
        None,
        {
            "root_dir": str(tmp_path),
            "deny_secret_files": True,
            "denied_patterns": ["**/*.sqlite"],
        },
    ).get_toolset()
    assert _call(fs.read_file("data.sqlite")).startswith("ERROR")
    assert _call(fs.read_file(".env")).startswith("ERROR")


def test_spec_reader_and_the_shipped_secure_spec(tmp_path):
    spec = tmp_path / "a.yaml"
    spec.write_text(
        "filesystem:\n  root_dir: ~/w\n  deny_secret_files: true\n"
        "  denied_patterns: ['**/*.db', '']\n"
    )
    assert _spec_filesystem_config(spec) == {
        "root_dir": "~/w",
        "deny_secret_files": True,
        "denied_patterns": ["**/*.db"],
    }
    plain = tmp_path / "b.yaml"
    plain.write_text("filesystem:\n  root_dir: ~/w\n")
    assert _spec_filesystem_config(plain) == {"root_dir": "~/w"}
    secure = Path(__file__).resolve().parent.parent / "specs" / "secure_assistant.yaml"
    assert _spec_filesystem_config(secure)["deny_secret_files"] is True


# --- Docker -------------------------------------------------------------------------
needs_docker = pytest.mark.skipif(not docker_available(), reason="Docker not available")


def _overlay_dirs() -> set[Path]:
    return set(Path(tempfile.gettempdir()).glob("paf-overlay-*"))


@pytest.fixture
def private_tmpdir(tmp_path, monkeypatch) -> Path:
    """Point this process's temp dir at a fresh folder, so overlay counts are ours.

    Overlays are created with ``tempfile.mkdtemp``; in the shared system temp dir,
    other pytest-xdist workers' overlays come and go while we count.
    """
    private = tmp_path / "tmp"
    private.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(private))
    return private


@needs_docker
def test_docker_blanks_nested_secrets_and_aliases_but_not_source(tmp_path):
    root = _tree(tmp_path)
    (root / "alias").symlink_to("sub/server.pem")

    async def scenario() -> str:
        ex = DockerExecutor(mode="run", exclude=DEFAULT_EXCLUDE)
        res = await ex.run(
            "for f in .env sub/.env a/b/c/.env.production sub/server.pem deploy.key "
            "alias config/secrets.yaml code.py sub/secrets_manager.py; do "
            'printf "%s=[" "$f"; cat "$f"; echo "]"; done; '
            "ls -A .ssh home/.aws .git",
            cwd=root,
            env=None,
            timeout=30,
        )
        return res.stdout

    out = _run(scenario())
    for blanked in (
        ".env",
        "sub/.env",
        "a/b/c/.env.production",
        "sub/server.pem",
        "alias",
    ):
        assert f"{blanked}=[]" in out, out
    assert "deploy.key=[]" in out and "config/secrets.yaml=[]" in out
    for value in (
        "secret-top",
        "secret-nested",
        "secret-deep",
        "keymaterial",
        "BEGIN KEY",
    ):
        assert value not in out, value
    assert "code.py=[print('hi')]" in out
    assert "sub/secrets_manager.py=[def load(): ...]" in out
    # The host files are untouched.
    assert (root / "sub" / ".env").read_text() == "NESTED=secret-nested"


@needs_docker
def test_docker_overlay_temp_dirs_are_cleaned_up_after_each_call(
    tmp_path, private_tmpdir
):
    root = _tree(tmp_path)
    before = _overlay_dirs()

    sock = root / "dockersock"
    sock.symlink_to("/var/run/docker.sock")

    async def scenario() -> None:
        ex = DockerExecutor(mode="run", exclude=DEFAULT_EXCLUDE)
        sock.unlink()
        for _ in range(3):
            await ex.run("true", cwd=root, env=None, timeout=30)
        # A failing build (the socket guard) must clean up after itself too.
        sock.symlink_to("/var/run/docker.sock")
        guarded = DockerExecutor(mode="run", exclude=["dockersock"])
        with pytest.raises(RuntimeError, match="Docker socket"):
            await guarded.run("true", cwd=root, env=None, timeout=30)

    _run(scenario())
    assert _overlay_dirs() - before == set()


@needs_docker
def test_docker_background_overlay_is_removed_on_stop(tmp_path, private_tmpdir):
    root = _tree(tmp_path)
    before = _overlay_dirs()

    async def scenario() -> set[Path]:
        ex = DockerExecutor(mode="run", exclude=DEFAULT_EXCLUDE)
        handle = await ex.start("sleep 30", cwd=root, env=None)
        during = _overlay_dirs() - before
        await ex.stop(handle)
        return during

    during = _run(scenario())
    assert len(during) == 1  # mounted while it runs
    assert _overlay_dirs() - before == set()


def test_overlays_are_not_left_behind_when_nothing_is_excluded(tmp_path):
    ex = DockerExecutor(mode="run", exclude=())
    args, root = ex._workspace_args(tmp_path)  # noqa: SLF001
    assert root is None and os.fspath(tmp_path) in " ".join(args)
