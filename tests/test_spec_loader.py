"""Tests for the spec loader: name→path resolution and spec parsing."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic_ai import AgentSpec

from paf.config import SPECS_DIR
from paf.specs import (
    AgentSpecNotFoundError,
    _spec_browser_config,
    get_spec_path,
    load_agent_spec,
)


def test_get_spec_path_prefers_yaml_over_yml(tmp_path: Path) -> None:
    """When both `{name}.yaml` and `{name}.yml` exist, `.yaml` wins."""
    (tmp_path / "assistant.yaml").write_text("name: assistant\nmodel: test\n")
    (tmp_path / "assistant.yml").write_text("name: assistant\nyaml\n")
    assert get_spec_path("assistant", tmp_path) == tmp_path / "assistant.yaml"


def test_get_spec_path_falls_back_to_yml(tmp_path: Path) -> None:
    """A lone `{name}.yml` is still found (yaml → yml → json order)."""
    (tmp_path / "agent.yml").write_text("name: agent\nmodel: test\n")
    assert get_spec_path("agent", tmp_path) == tmp_path / "agent.yml"


def test_get_spec_path_raises_for_unknown_agent(tmp_path: Path) -> None:
    """An unknown agent name raises AgentSpecNotFoundError, not a generic error."""
    with pytest.raises(AgentSpecNotFoundError):
        get_spec_path("does-not-exist", tmp_path)


def test_load_agent_spec_returns_parsed_spec(tmp_path: Path) -> None:
    """load_agent_spec parses a real YAML into a validated AgentSpec."""
    (tmp_path / "assistant.yaml").write_text(
        "name: assistant\nmodel: openai:gpt-5.4-nano\n"
    )
    spec = load_agent_spec("assistant", tmp_path)
    assert isinstance(spec, AgentSpec)
    assert spec.name == "assistant"


@pytest.mark.parametrize("name", sorted(p.stem for p in SPECS_DIR.glob("*.yaml")))
def test_every_shipped_spec_parses(name: str) -> None:
    """Each spec in specs/ loads. The values are deployment config, not under test."""
    spec = load_agent_spec(name, SPECS_DIR)
    assert isinstance(spec, AgentSpec)
    assert spec.name == name


def test_spec_browser_config_absent_block_returns_none(tmp_path: Path) -> None:
    """A spec with no `browser:` block yields None (fall back to env default)."""
    (tmp_path / "a.yaml").write_text("name: a\nmodel: test\n")
    assert _spec_browser_config(tmp_path / "a.yaml") is None


def test_spec_browser_config_empty_list_returns_none(tmp_path: Path) -> None:
    """An empty `allowed_domains` yields None (fall back to env default)."""
    (tmp_path / "b.yaml").write_text(
        "name: b\nmodel: test\nbrowser:\n  allowed_domains: []\n"
    )
    assert _spec_browser_config(tmp_path / "b.yaml") is None


def test_spec_browser_config_unknown_keys_ignored(tmp_path: Path) -> None:
    """Unknown keys inside `browser:` are ignored; allowed_domains still read."""
    (tmp_path / "c.yaml").write_text(
        "name: c\nmodel: test\nbrowser:\n"
        "  allowed_domains:\n"
        "    - example.com\n"
        "    - api.example.com\n"
        "  future_field: true\n"
    )
    assert _spec_browser_config(tmp_path / "c.yaml") == {
        "allowed_domains": ["example.com", "api.example.com"]
    }


def test_spec_browser_config_private_override_false(tmp_path: Path) -> None:
    """A spec declaring `block_private_addresses: false` threads it through."""
    (tmp_path / "d.yaml").write_text(
        "name: d\nmodel: test\nbrowser:\n"
        "  allowed_domains:\n"
        "    - localhost\n"
        "  block_private_addresses: false\n"
    )
    assert _spec_browser_config(tmp_path / "d.yaml") == {
        "allowed_domains": ["localhost"],
        "block_private_addresses": False,
    }


def test_spec_browser_config_private_override_true(tmp_path: Path) -> None:
    """`block_private_addresses: true` is explicit and never dissolved."""
    (tmp_path / "e.yaml").write_text(
        "name: e\nmodel: test\nbrowser:\n  block_private_addresses: true\n"
    )
    assert _spec_browser_config(tmp_path / "e.yaml") == {
        "block_private_addresses": True
    }


def test_a_spec_with_invalid_yaml_is_reported_with_its_line(tmp_path: Path) -> None:
    """A syntax error must not masquerade as "no profile" (which hid the real cause)."""
    from paf.specs import InvalidSpecError, resolve_profile_name

    (tmp_path / "broken.yaml").write_text(
        "name: broken\nprofile: assistant\nmodel_settings:\n  # a comment\n"
        "  an uncommented note (oops)\n  temperature: 0\n"
    )
    with pytest.raises(
        InvalidSpecError, match=r"broken\.yaml is not valid YAML at line 6"
    ):
        resolve_profile_name("broken", tmp_path)


def test_a_missing_spec_still_resolves_to_no_profile(tmp_path: Path) -> None:
    from paf.specs import resolve_profile_name

    assert resolve_profile_name("absent", tmp_path) is None


def test_building_an_unknown_agent_names_the_known_ones(tmp_path: Path) -> None:
    from paf.specs import build_agent

    (tmp_path / "real_one.yaml").write_text("name: real_one\nprofile: assistant\n")
    with pytest.raises(AgentSpecNotFoundError, match=r"Unknown agent 'typo'.*real_one"):
        build_agent("typo", spec_directory=tmp_path)


def test_building_a_spec_without_a_profile_says_so(tmp_path: Path) -> None:
    from paf.specs import MissingProfileError, build_agent

    (tmp_path / "noprof.yaml").write_text("name: noprof\nmodel: test\n")
    with pytest.raises(MissingProfileError, match="No profile for agent 'noprof'"):
        build_agent("noprof", spec_directory=tmp_path)


# --- the shipped specs name no machine --------------------------------------------------
# The specs that ship (the demo and secure agents, the sub-agent and background-agent
# leaves, and the example MCP config) must work on any checkout, so they carry no paths
# into a particular person's home folder, notes or tooling. A personal spec such as
# `assistant.yaml` is gitignored and not covered here.
_MACHINE_SPECIFIC = re.compile(
    r"/home/|/Users/|Obsidian|Documents/|~/\.hermes", re.IGNORECASE
)


def _shipped_spec_files() -> list[Path]:
    files = [SPECS_DIR / "demo_assistant.yaml", SPECS_DIR / "secure_assistant.yaml"]
    for sub in ("sub_agents", "background_agents"):
        files += sorted((SPECS_DIR / sub).glob("*.yaml"))
    files += sorted((SPECS_DIR / "mcp").glob("*.json"))
    return files


@pytest.mark.parametrize("path", _shipped_spec_files(), ids=lambda p: p.name)
def test_shipped_specs_name_no_machine_specific_paths(path: Path) -> None:
    hits = [
        f"line {n}: {line.strip()[:90]}"
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if _MACHINE_SPECIFIC.search(line)
    ]
    assert not hits, f"{path.name} names a specific machine:\n" + "\n".join(hits)


def test_tracked_files_name_no_personal_details():
    """No author email, home directory, tailnet host or client domain in the tree."""
    import re
    import subprocess

    # Generic shapes only: a home directory other than the placeholder, a
    # personal-mail address, a tailnet host, a Hermes skills path.
    pattern = re.compile(
        r"/home/(?!alice\b|runner\b|user\b|me\b)\w+|[\w.+-]+@gmail\.com|[\w-]+\.ts\.net"
        r"|~/\.hermes"
    )
    root = Path(__file__).resolve().parent.parent
    files = subprocess.run(
        ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    skip = {"uv.lock", "tests/test_spec_loader.py", ".gitignore"}
    hits = []
    for name in files:
        if name in skip or name.endswith(
            (".png", ".ico", ".lock", "package-lock.json")
        ):
            continue
        try:
            text = (root / name).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        hits += [f"{name}: {m.group(0)}" for m in pattern.finditer(text)]
    assert not hits, "\n".join(hits)
