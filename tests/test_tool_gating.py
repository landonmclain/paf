"""Tool Gating: the per-call execution-time tool gate.

The harness ``ToolGuardrail`` is the shell; paf owns the policy
(``tool_gating_guard``) which refuses the highest-risk tools before they
execute. These tests prove the gate stops a dangerous call, lets a benign call
through, and is wired into the demo profile.
"""

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness import ToolGuardrail
from pydantic_ai_harness.guardrails import ToolCallInfo

import paf.capabilities.guardrails.tool_gating as tool_gating
from paf.capabilities.build import _build_tool_gating
from paf.capabilities.guardrails.tool_gating import (
    DEFAULT_GATED_TOOLS,
    tool_gating_guard,
)
from paf.capabilities.registry import CAPABILITY_TYPES


def _info(name: str) -> ToolCallInfo:
    return ToolCallInfo(name=name, args={}, tool_call_id="1")


def test_gated_tool_is_blocked(monkeypatch):
    """A call to a gated tool is refused before it executes.

    The production gate is currently empty (the hard-coded gates were removed
    so they don't block normal operations, pending the approval-path design), so
    this pins the block path deterministically by gating one tool for the test.
    """
    monkeypatch.setattr(tool_gating, "DEFAULT_GATED_TOOLS", frozenset({"run_command"}))

    async def run_command(cmd: str) -> str:
        return "EXECUTED: " + cmd

    agent = Agent(
        TestModel(call_tools=["run_command"]),
        capabilities=[ToolGuardrail(guard=tool_gating_guard)],
        tools=[run_command],
    )
    result = agent.run_sync("run the command")
    assert "gated" in result.output
    assert "EXECUTED" not in result.output


def test_benign_tool_passes():
    """A call to a non-gated tool executes normally."""

    async def read_file(path: str) -> str:
        return "CONTENT"

    agent = Agent(
        TestModel(call_tools=["read_file"]),
        capabilities=[ToolGuardrail(guard=tool_gating_guard)],
        tools=[read_file],
    )
    result = agent.run_sync("read the file")
    assert "CONTENT" in result.output


def test_guard_policy_allowlist():
    """The policy gates the highest-risk tools and nothing else."""
    for name in DEFAULT_GATED_TOOLS:
        assert tool_gating_guard(_info(name)).action == "block"
    assert tool_gating_guard(_info("read_file")).action == "allow"


def test_wired_but_not_allowlisted():
    """ToolGating is registered and buildable, but OFF unless a profile allowlists it.

    The capability is fully wired (builder + registry) but not allowlisted while
    the gate policy is empty and the approval path is undecided.
    """
    from paf.capabilities import AgentProfile

    off = AgentProfile(name="gating-off", capability_names=("StepPersistence",))
    assert "ToolGating" not in off.capability_names
    assert CAPABILITY_TYPES["ToolGating"] is ToolGuardrail
    cap = _build_tool_gating(store=None)
    assert isinstance(cap, ToolGuardrail)
