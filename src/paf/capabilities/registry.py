"""Capability catalog — the master registry of capability types an agent can enable.

A profile is a *named allowlist* over this catalog (see ``profiles.py``). M1 ships
all harness capabilities we actually attach to an agent; nothing is scaffolded
ahead of need (YAGNI). Core pydantic-ai capabilities (``WebSearch``, ``Thinking``,
etc.) do NOT need a catalog entry — they are native to ``Agent.from_spec`` and the
spec loader resolves them without ``custom_capability_types``. This catalog exists
only for capabilities the loader cannot resolve by default (harness capabilities
like ``StepPersistence``), keyed by the CamelCase name used in the spec YAML.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai_harness.step_persistence import StepPersistence

# name -> capability type. The name is the serialization form used in a spec's
# `capabilities:` entry and the key a profile allowlists against.
CAPABILITY_TYPES: dict[str, type[Any]] = {
    "StepPersistence": StepPersistence,
}
