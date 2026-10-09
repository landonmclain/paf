"""Phase 1 MCP smoke test — prove the builtin pydantic-ai MCP capability works.

Runs a tiny **in-process** FastMCP server (no subprocess, no network), attaches it
to an Agent through the builtin ``pydantic_ai.capabilities.MCP`` capability, and
verifies tool discovery + invocation end-to-end. Zero changes to paf's
build/profile/spec machinery — this is the "simplest way to see how it functions."

Usage:
    uv run --env-file .env python scripts/smoke_mcp.py      # live model
    uv run python scripts/smoke_mcp.py --testmodel          # offline, TestModel
"""

from __future__ import annotations

import os
import sys

# A tiny in-process MCP server: one trivial tool we can ask the model to call.
from mcp.server.fastmcp import FastMCP


def make_server() -> FastMCP:
    server = FastMCP("smoke")

    @server.tool()
    def add(a: int, b: int) -> int:
        """Add two integers together."""
        return a + b

    return server


def make_agent(model, server: FastMCP):
    from pydantic_ai import Agent
    from pydantic_ai.capabilities import MCP

    return Agent(model=model, capabilities=[MCP(local=server)])


def tool_names(agent) -> list[str]:
    """Best-effort list of the tool names the agent currently exposes."""
    names: list[str] = []
    for toolset in agent.toolsets or []:
        tools = getattr(toolset, "tools", None) or []
        for t in tools:
            name = getattr(t, "name", None) or getattr(t, "tool_def", None)
            if name is not None:
                if isinstance(name, str):
                    names.append(name)
                else:
                    names.append(getattr(name, "name", "<unknown>"))
    # The `tool` decorator / function tools registered directly on the agent.
    for name in getattr(agent, "_function_tools", {}) or {}:
        names.append(name)
    return sorted(set(names))


def dump_steps(result) -> None:
    """Print every tool-call / tool-return part across the run's messages."""
    from pydantic_ai.messages import ToolCallPart, ToolReturnPart

    for msg in result.all_messages():
        for part in getattr(msg, "parts", []):
            if isinstance(part, ToolCallPart):
                print(f"  🛠️ call  -> {part.tool_name}({part.args})")
            elif isinstance(part, ToolReturnPart):
                print(f"  ✅ return<- {part.tool_name} = {part.content!r}")


def main() -> int:
    server = make_server()

    if "--testmodel" in sys.argv:
        from pydantic_ai.models.test import TestModel

        agent = make_agent(TestModel(call_tools="all"), server)
        print(f"== discovered tools: {tool_names(agent)}")
        print("== running a turn with TestModel (offline, deterministic)")
        result = agent.run_sync("Use the add tool to compute 2 + 3.")
        dump_steps(result)
        print(f"== final output: {result.output!r}")
        print("== PASS: MCP tool was discovered and invoked (TestModel).")
        return 0

    # Live path — real model from the repo .env.
    if not os.environ.get("OPENROUTER_API_KEY"):
        print(
            "OPENROUTER_API_KEY not set; run with `--env-file .env` or pass --testmodel."
        )
        return 2

    from pydantic_ai import Agent
    from pydantic_ai.capabilities import MCP

    model = "openrouter:deepseek/deepseek-v4-flash-0731"
    agent = Agent(model=model, capabilities=[MCP(local=server)])
    print(f"== model: {model}")
    print(f"== discovered tools: {tool_names(agent)}")

    print("== running a live turn: 'Use the add tool to compute 2 + 3.'")
    result = agent.run_sync("Use the add tool to compute 2 + 3.")
    dump_steps(result)
    print(f"== final output: {result.output!r}")
    print("== PASS: MCP tool was discovered and invoked (live).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
