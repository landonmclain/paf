"""Tests for the spec loader: name→path resolution and spec parsing."""

from __future__ import annotations

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
