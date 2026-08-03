"""Example single agent for M1.

A minimal pydantic-ai V2 Agent wired with one typed tool + structured output.
The same ``Agent.run()`` unit serves as the framework's composable primitive
(single-agent now, multi-agent later via a Runner).
"""

from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.capabilities import Instrumentation

from paf.config import settings
from paf.models.schemas import Greeting
from paf.toolsets.example_tools import add, current_datetime


def build_agent(model_name: str | None = None) -> Agent[None, Greeting]:
    """Build the example agent.

    ``model_name`` overrides ``settings.model_name`` (used for smoke tests with
    pydantic-ai's keyless ``test`` model, or to point at a specific provider).

    Observability is a first-class V2 capability: ``Instrumentation`` emits OTel
    spans (agent run / model request / tool execution) to the global tracer
    provider that ``setup_observability()`` configures (Langfuse via OTLP).
    """
    return Agent[None, Greeting](
        model_name or settings.model_name,
        system_prompt="You are a helpful, concise assistant. You respond politely.",
        output_type=Greeting,
        tools=[add, current_datetime],
        capabilities=[Instrumentation()],
    )
