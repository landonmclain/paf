"""An MCP server that cannot start must not end the run."""

from __future__ import annotations

import json
import logging

from pydantic_ai import Agent

from paf.capabilities.build import _build_mcp


def _config(tmp_path, servers: dict) -> str:
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"mcpServers": servers}))
    return str(path)


def test_unstartable_server_logs_and_the_run_completes(tmp_path, caplog):
    cfg = _config(tmp_path, {"broken": {"command": "/nonexistent/paf-test-server"}})
    caps = _build_mcp(None, cfg)
    assert len(caps) == 1
    agent = Agent("test", capabilities=caps)
    with caplog.at_level(logging.WARNING, logger="paf.capabilities._mcp"):
        result = agent.run_sync("hello")
    assert result.output
    assert any("unavailable" in r.getMessage() for r in caplog.records)


def test_each_server_degrades_on_its_own(tmp_path, caplog):
    cfg = _config(
        tmp_path,
        {
            "one": {"command": "/nonexistent/one"},
            "two": {"command": "/nonexistent/two"},
        },
    )
    caps = _build_mcp(None, cfg)
    assert len(caps) == 2
    with caplog.at_level(logging.WARNING, logger="paf.capabilities._mcp"):
        Agent("test", capabilities=caps).run_sync("hello")
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "one" in messages and "two" in messages
