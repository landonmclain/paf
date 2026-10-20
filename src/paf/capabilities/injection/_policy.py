"""The injection policy: what to do with each tool's results, from the spec.

    guardrails:
      injection:
        scorer: stackone        # the only scorer so far
        semantic: true          # stackone's ML tier (false = patterns only)
        threshold: null         # null = the scorer's calibrated default
        default_action: block   # off | observe | block   (flag | redact: later)
        tools:                  # glob -> action, first match wins
          list_directory: observe
          "file_*": block

With no block at all the policy is the old behaviour: every tool ``block`` at the
scorer's default threshold. A malformed block raises at build time rather than
loading half a policy.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

ACTIONS: tuple[str, ...] = ("off", "observe", "flag", "redact", "block")
IMPLEMENTED_ACTIONS: frozenset[str] = frozenset({"off", "observe", "block"})
SCORERS: tuple[str, ...] = ("stackone",)
_KEYS = {"scorer", "semantic", "threshold", "default_action", "tools"}
# Reserved for the next phases; named here so a spec that asks for them fails
# loudly instead of being ignored.
_NOT_YET = {"mark_untrusted": "phase 2", "on_flag": "phase 2"}


@dataclass(frozen=True)
class InjectionPolicy:
    scorer: str = "stackone"
    semantic: bool = True
    threshold: float | None = None
    default_action: str = "block"
    tools: tuple[tuple[str, str], ...] = ()
    """Ordered ``(glob, action)`` pairs; the first match wins."""

    def action_for(self, tool_name: str) -> str:
        for pattern, action in self.tools:
            if fnmatch.fnmatchcase(tool_name, pattern):
                return action
        return self.default_action


def _action(value: Any, where: str) -> str:
    if value not in ACTIONS:
        raise ValueError(f"{where} must be one of {ACTIONS}, got {value!r}")
    if value not in IMPLEMENTED_ACTIONS:
        raise ValueError(
            f"{where}: action {value!r} is not implemented yet; "
            f"available: {sorted(IMPLEMENTED_ACTIONS)}"
        )
    return value


def parse_injection_policy(config: Mapping[str, Any] | None) -> InjectionPolicy:
    """Validate a spec ``guardrails.injection`` block into a policy."""
    if not config:
        return InjectionPolicy()
    if not isinstance(config, Mapping):
        raise ValueError("guardrails.injection must be a mapping")
    for key, phase in _NOT_YET.items():
        if key in config:
            raise ValueError(
                f"guardrails.injection.{key} is not implemented yet ({phase})"
            )
    unknown = set(config) - _KEYS
    if unknown:
        raise ValueError(f"guardrails.injection: unknown keys {sorted(unknown)}")
    scorer = config.get("scorer", "stackone")
    if scorer not in SCORERS:
        raise ValueError(
            f"guardrails.injection.scorer must be one of {SCORERS}, got {scorer!r}"
        )
    semantic = config.get("semantic", True)
    if not isinstance(semantic, bool):
        raise ValueError("guardrails.injection.semantic must be true or false")
    threshold = config.get("threshold")
    if threshold is not None:
        if isinstance(threshold, bool) or not isinstance(threshold, int | float):
            raise ValueError("guardrails.injection.threshold must be a number or null")
        if not 0.0 < float(threshold) < 1.0:
            raise ValueError("guardrails.injection.threshold must be between 0 and 1")
        if not semantic:
            raise ValueError("guardrails.injection.threshold needs semantic: true")
        threshold = float(threshold)
    default_action = _action(
        config.get("default_action", "block"), "guardrails.injection.default_action"
    )
    raw_tools = config.get("tools") or {}
    if not isinstance(raw_tools, Mapping):
        raise ValueError("guardrails.injection.tools must map tool globs to actions")
    tools = tuple(
        (str(pattern), _action(action, f"guardrails.injection.tools[{pattern!r}]"))
        for pattern, action in raw_tools.items()
    )
    return InjectionPolicy(
        scorer=scorer,
        semantic=semantic,
        threshold=threshold,
        default_action=default_action,
        tools=tools,
    )
