"""Tests for the spec loader: name→path resolution and spec parsing."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic_ai import AgentSpec

from paf.config import SPECS_DIR
from paf.specs import (
    AgentSpecNotFoundError,
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


def test_real_assistant_spec_exists_and_parses() -> None:
    """The shipped spec must load — CI/production depends on it resolving."""
    assert (SPECS_DIR / "assistant.yaml").is_file(), "specs/assistant.yaml missing"
    spec = load_agent_spec("assistant", SPECS_DIR)
    assert isinstance(spec, AgentSpec)
    assert spec.name == "assistant"
