"""capabilities — the extend-a-protocol seam for custom capabilities.

M1 ships the *profile* concept here: a profile is a named allowlist over the
``CAPABILITY_TYPES`` catalog, resolved to ``custom_capability_types`` when the
spec loader builds an agent. See ``registry.py`` (the master catalog) and
``profiles.py`` (``AgentProfile`` + the one real M1 profile, ``ASSISTANT_PROFILE``).

Custom capability *implementations* (memory, guardrails, cost-tracking, approvals)
also live here as composable bundles of tools + hooks + instructions. Prefer
pydantic-ai built-ins / Harness unless it's genuinely custom.
"""

from __future__ import annotations

from .profiles import ASSISTANT_PROFILE, PROFILES, AgentProfile
from .registry import CAPABILITY_TYPES

__all__ = [
    "ASSISTANT_PROFILE",
    "PROFILES",
    "AgentProfile",
    "CAPABILITY_TYPES",
]
