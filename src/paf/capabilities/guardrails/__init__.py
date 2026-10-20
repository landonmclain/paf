"""paf-owned guard policies.

The harness ``ToolGuardrail`` is the shell for the call-time guards;
paf owns the policy as callables here, per the settled don't-pin stance. Each
module in this package is one guard policy, wired into a capability by the
builders in ``build.py``.
"""

from paf.capabilities.guardrails.gating_policy import (
    GatingPolicy,
    GatingRule,
    make_gating_guard,
    parse_policy,
)
from paf.capabilities.guardrails.tool_gating import (
    DEFAULT_GATED_TOOLS,
    tool_gating_guard,
)

__all__ = [
    "DEFAULT_GATED_TOOLS",
    "GatingPolicy",
    "GatingRule",
    "make_gating_guard",
    "parse_policy",
    "tool_gating_guard",
]
