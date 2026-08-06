"""Minimal one-turn entrypoint — build agent from spec, run once, print."""

from __future__ import annotations

from pydantic_ai import Agent

from paf.observability.instrumentation import (
    setup_observability,
    shutdown_observability,
)
from paf.specs import build_agent


def build_assistant() -> Agent:
    """Build the assistant agent from specs/assistant.yaml."""
    return build_agent("assistant")


def main() -> None:
    """Build agent, send one prompt, print response."""
    setup_observability()
    agent = build_assistant()
    print(f"Agent: {agent.name}", flush=True)
    result = agent.run_sync("Say hello in one sentence.")
    print(f"\n{result.output}")
    # Flush buffered OTel spans before the short-lived process exits, or the
    # BatchSpanProcessor drops them and traces never reach Langfuse.
    shutdown_observability()


if __name__ == "__main__":
    main()
