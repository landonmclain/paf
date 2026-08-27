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


def test_real_assistant_spec_exists_and_parses() -> None:
    """The shipped spec must load — CI/production depends on it resolving."""
    assert (SPECS_DIR / "assistant.yaml").is_file(), "specs/assistant.yaml missing"
    spec = load_agent_spec("assistant", SPECS_DIR)
    assert isinstance(spec, AgentSpec)
    assert spec.name == "assistant"


def test_real_demo_spec_browser_block_parses() -> None:
    """The demo spec's shipped `browser:` block resolves as declared.

    ⚠️ TEMPORARY CONTRACT (2026-08-27): the demo spec currently declares
    loopback + the private-address guard disabled (`block_private_addresses:
    false`) so the demo agent can reach local/loopback targets while building
    the landonmclain-site Astro repo. This is a LOCAL-TROUBLESHOOTING release
    of the SSRF guard — it MUST be reverted to the secure public contract
    (drop localhost/127.0.0.1 + `block_private_addresses: false`, keep only
    public-reach domains) BEFORE the repo goes public.

    Tracks: Portfolio-GitHub/Publish.md SCRUB pre-flight; see vault
    Tier-Sorted-Render-Benchmark / publish notes for the revert gate.

    Secure contract it returns to (the intent this test used to enforce):
    only public staging/reach domains, `block_private_addresses` absent so the
    secure default (true) holds, no loopback hosts in the allowlist.
    """
    cfg = _spec_browser_config(SPECS_DIR / "demo_assistant.yaml")
    assert cfg is not None
    assert cfg == {
        "allowed_domains": [
            "staging.example.com",
            "example.com",
            "localhost",
            "127.0.0.1",
        ],
        "block_private_addresses": False,
    }


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
        "name: e\nmodel: test\nbrowser:\n"
        "  block_private_addresses: true\n"
    )
    assert _spec_browser_config(tmp_path / "e.yaml") == {
        "block_private_addresses": True
    }
