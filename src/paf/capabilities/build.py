"""Runtime capability builders — the ``capabilities=`` seam for Agent.from_spec.

Spec files (`specs/*.yaml`) go through pydantic-ai's ``AgentSpec``/``Agent.from_spec``,
which resolves each top-level ``capabilities:`` entry via that class's ``from_spec()``
and — crucially — only ONE level deep (see ``_capabilities_from_spec`` in pydantic-ai's
``agent/__init__.py``). Capabilities that can't be expressed that way — because a class
doesn't override ``from_spec`` or needs real objects as arguments — are assembled here and
injected via the ``capabilities=`` kwarg, which ``Agent.from_spec`` MERGES with any
spec-declared capabilities.

`build_runtime_capabilities()` returns that Python-assembled list. For M1 it is the
union of:

- the tiered context compaction, assembled in :mod:`paf.capabilities.compaction`
  (``build_compaction()``);
- persistence + recall — `StepPersistence` and `ConversationSearch` over one shared
  ``SqliteStepStore`` — assembled in :mod:`paf.capabilities.persistence`
  (``build_persistence()``).

This module is just the collection point: it calls the two builders and hands the
results to ``build_agent``, which passes them via the ``capabilities=`` kwarg to
``Agent.from_spec``. See each builder module for why its pieces can't be YAML-declared.

Config lives in code (not YAML) for now, mirroring ``ui/cli_demo.py``. If we later
want the values in YAML, the loader would strip a ``compaction:``/``persistence:`` block
before handing the rest to ``AgentSpec``; that's a documented follow-up, not scaffolded
yet (YAGNI).
"""

from __future__ import annotations

from paf.capabilities.compaction import build_compaction
from paf.capabilities.persistence import build_persistence


def build_runtime_capabilities() -> list:
    """Return the capabilities that can't resolve from a spec, to merge at Agent.from_spec.

    Mirrors the hybrid pattern: YAML owns what pydantic-ai resolves declaratively; this
    owns what only Python can construct. Delegates to the per-domain builders so each
    capability family stays self-contained.
    """
    return [build_compaction(), *build_persistence()]
