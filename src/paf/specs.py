"""Spec loader — resolve agent names to YAML specs and build agents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic_ai import Agent, AgentSpec

from paf.capabilities import ASSISTANT_PROFILE, AgentProfile
from paf.capabilities.runtime import build_runtime_capabilities
from paf.config import SPECS_DIR

DEFAULT_SPEC_DIRECTORY = SPECS_DIR


class AgentSpecNotFoundError(FileNotFoundError):
    """Raised when an agent specification cannot be found."""


def get_spec_path(
    agent_name: str,
    spec_directory: Path = DEFAULT_SPEC_DIRECTORY,
) -> Path:
    """Resolve an agent name to a YAML/JSON specification path.

    Tries ``{name}.yaml``, ``{name}.yml``, then ``{name}.json`` in order.
    """
    for suffix in (".yaml", ".yml", ".json"):
        path = spec_directory / f"{agent_name}{suffix}"
        if path.is_file():
            return path
    raise AgentSpecNotFoundError(
        f"No agent specification found for {agent_name!r} in {spec_directory}"
    )


def load_agent_spec(
    agent_name: str,
    spec_directory: Path = DEFAULT_SPEC_DIRECTORY,
) -> AgentSpec:
    """Load and validate an agent specification from disk."""
    path = get_spec_path(agent_name, spec_directory)
    return AgentSpec.from_file(path)


def build_agent(
    agent_name: str,
    spec_directory: Path = DEFAULT_SPEC_DIRECTORY,
    profile: AgentProfile | None = None,
    **overrides: Any,
) -> Agent[Any, Any]:
    """Construct an agent from a named spec, with optional overrides.

    ```agent_name`` is the spec file base name (``assistant`` → ``assistant.yaml``).

    ``profile`` is the capability allowlist for this agent kind — resolved to
    ``custom_capability_types`` so the spec's ``capabilities:`` entries can only
    come from the allowlisted catalog. Defaults to ``ASSISTANT_PROFILE``, M1's
    one real agent kind.

    ``build_runtime_capabilities()`` assembles capabilities that cannot resolve
    from YAML (the tiered ``TieredCompaction``, whose nested strategy objects have
    no ``from_spec``) and passes them via ``Agent.from_spec(..., capabilities=)``,
    which merges them with any spec-declared capabilities (e.g. ``StepPersistence``).

    Overrides are passed as kwargs to ``Agent.from_spec``. Supported:
    ``model``, ``instructions``, ``model_settings``, ``name``, ``output_type``.
    Tools and spec-declarable capabilities must be declared in the spec YAML.
    """
    profile = profile or ASSISTANT_PROFILE
    spec = load_agent_spec(agent_name, spec_directory)
    return Agent.from_spec(
        spec,
        # Resolve the agent kind's allowlist to the capability TYPES the loader
        # registers. Harness capabilities (StepPersistence) aren't in pydantic-ai's
        # default registry; allowlisting them here is what lets the YAML
        # `capabilities:` entry resolve.
        custom_capability_types=profile.capability_types(),
        # Capabilities that only Python can construct (tiered compaction) merge
        # here with the spec-declared ones via Agent.from_spec.
        capabilities=build_runtime_capabilities(),
        **overrides,
    )
