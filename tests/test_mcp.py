"""Tests for the MCP capability wiring (Phase 2, Option A).

MCP servers come from a Claude-Desktop-style ``mcpServers`` JSON (path declared by
the spec's ``mcp:`` block), loaded by the builtin ``load_mcp_toolsets`` (recursive
``${VAR}`` / ``${VAR:-default}`` env expansion built in) and wrapped one-per-server
in the builtin ``MCP`` capability. The profile allowlist is the gate: only a
profile that names ``"MCP"`` gets any MCP capability, and it gets exactly the
servers declared in the JSON. Connection is lazy — these tests only *build*, never
connect, so no server subprocess or socket is ever spawned.
"""

from __future__ import annotations

import json

import pytest
from pydantic_ai.capabilities import MCP

from paf.capabilities.build import _build_mcp, build_runtime_capabilities
from paf.capabilities.profiles import AgentProfile
from paf.specs import _spec_mcp_config


def _write_mcp_json(tmp_path, servers: dict):
    p = tmp_path / "mcp_servers.json"
    p.write_text(json.dumps({"mcpServers": servers}))
    return p


def test_build_mcp_wraps_one_capability_per_server(tmp_path):
    p = _write_mcp_json(
        tmp_path,
        {
            "alpha": {"command": "true", "args": []},
            "beta": {"command": "true", "args": []},
        },
    )
    caps = _build_mcp(None, str(p))
    assert len(caps) == 2
    assert all(isinstance(c, MCP) for c in caps)
    for cap in caps:
        assert cap.get_toolset() is not None


def test_build_mcp_none_config_yields_none():
    assert _build_mcp(None, None) == []


def test_build_mcp_expands_env_in_headers(tmp_path, monkeypatch):
    monkeypatch.setenv("DEMO_MCP_AUTH", "Basic abc123")
    p = _write_mcp_json(
        tmp_path,
        {
            "demo": {
                "url": "https://example.com/mcp",
                "headers": {"Authorization": "${DEMO_MCP_AUTH}"},
            }
        },
    )
    # Load succeeds (expansion ran); a missing var would raise at load time.
    caps = _build_mcp(None, str(p))
    assert len(caps) == 1
    assert isinstance(caps[0], MCP)


def test_build_mcp_missing_env_var_raises_at_load(tmp_path):
    p = _write_mcp_json(
        tmp_path,
        {
            "demo": {
                "url": "https://example.com/mcp",
                "headers": {"Authorization": "${UNDEFINED_MCP_AUTH}"},
            }
        },
    )
    with pytest.raises(ValueError):
        _build_mcp(None, str(p))


def test_profile_is_the_gatekeeper_for_mcp(tmp_path):
    p = _write_mcp_json(tmp_path, {"demo": {"command": "true", "args": []}})

    on = AgentProfile(name="mcp-on", capability_names=("MCP",))
    got = build_runtime_capabilities(on, mcp_config=str(p))
    assert any(isinstance(c, MCP) for c in got)

    off = AgentProfile(name="mcp-off", capability_names=())
    got_off = build_runtime_capabilities(off, mcp_config=str(p))
    assert not any(isinstance(c, MCP) for c in got_off)


def test_spec_mcp_config_reads_path(tmp_path):
    spec = tmp_path / "agent.yaml"
    spec.write_text("mcp:\n  config: data/mcp_servers.json\nmodel: openai:gpt-4o\n")
    assert _spec_mcp_config(spec) == "data/mcp_servers.json"


def test_spec_mcp_config_absent(tmp_path):
    spec = tmp_path / "agent.yaml"
    spec.write_text("model: openai:gpt-4o\n")
    assert _spec_mcp_config(spec) is None


# --- the shipped demo config ----------------------------------------------------------
# The demo agent ships with two keyless example servers so MCP works on a fresh clone.


def test_shipped_demo_config_is_valid_keyless_and_lazy():
    """It parses, builds one lazy capability per server, and holds no credentials."""
    from paf.config import PROJECT_ROOT

    path = PROJECT_ROOT / "specs" / "mcp" / "demo_servers.json"
    raw = path.read_text(encoding="utf-8")
    servers = json.loads(raw)["mcpServers"]
    assert set(servers) == {"time", "deepwiki"}
    # A stdio server runs third-party code, so it is pinned to the tested version.
    assert servers["time"]["args"] == ["mcp-server-time@2026.10.19"]
    for name, cfg in servers.items():
        assert ("command" in cfg) ^ ("url" in cfg), f"{name}: stdio or http, not both"
        assert not {"env", "headers", "authorization_token"} & set(cfg), (
            f"{name}: the tracked config must stay keyless"
        )
    assert "${" not in raw  # no env placeholders either: nothing to set up
    caps = _build_mcp(None, str(path))  # building connects to nothing
    assert len(caps) == 2 and all(isinstance(c, MCP) for c in caps)


def test_the_demo_spec_and_profile_use_the_shipped_config():
    from paf.capabilities.profiles import PROFILES
    from paf.config import PROJECT_ROOT, SPECS_DIR

    configured = _spec_mcp_config(SPECS_DIR / "demo_assistant.yaml")
    assert configured == "specs/mcp/demo_servers.json"
    assert (PROJECT_ROOT / configured).is_file()
    assert "MCP" in PROFILES["demo"].capability_names
