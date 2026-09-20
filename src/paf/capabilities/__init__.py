"""capabilities — the extend-a-protocol seam for custom capabilities.

A profile is a named allowlist over the ``CAPABILITY_TYPES`` catalog, resolved to
``custom_capability_types`` when the spec loader builds an agent. See
``registry.py`` (the master catalog) and ``profiles.py`` (``AgentProfile`` + the
built-in profiles).

Custom capability *implementations* (memory, guardrails, cost-tracking, approvals)
also live here as composable bundles of tools + hooks + instructions. Prefer
pydantic-ai built-ins / pydantic-ai-harness unless it's genuinely custom.

Context compaction is built in Python (``compaction.py``) — ``TieredCompaction``'s nested
``tiers`` must be real strategy objects and none of the pydantic-ai-harness compaction
classes override ``from_spec``, so it can't resolve from a YAML ``tiers:`` dict list. The
compaction types are cataloged (``registry.py``) so profiles can allowlist them.
"""

from __future__ import annotations

from paf.capabilities.profiles import ASSISTANT_PROFILE, PROFILES, AgentProfile
from paf.capabilities.registry import CAPABILITY_TYPES

__all__ = [
    "ASSISTANT_PROFILE",
    "PROFILES",
    "AgentProfile",
    "CAPABILITY_TYPES",
]
