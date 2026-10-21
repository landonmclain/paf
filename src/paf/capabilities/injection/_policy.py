"""The injection policy: what to do with each tool's results, from the spec.

    guardrails:
      injection:
        scorer: stackone          # the only scorer so far
        semantic: true            # stackone's ML tier (false = patterns only)
        threshold: null           # null = the scorer's calibrated default
        default_action: block     # off | observe | flag | block   (redact: later)
        tools:                    # glob -> action, first match wins
          list_directory: observe
          read_file: flag
        mark_untrusted: flagged   # never | flagged | always
        on_flag:                  # once a hit under flag/block lands in a run,
          require_approval: [write_file, edit_file]   # these need approval
          scope: run              # for the rest of that run (conversation: later)

With no block at all the policy is the old behaviour: every tool ``block`` at the
scorer's default threshold, nothing marked, no tightening. A malformed block
raises at build time rather than loading half a policy, and a reserved feature
that is not built yet (``redact``, ``scope: conversation``) refuses to load
instead of being silently ignored.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

ACTIONS: tuple[str, ...] = ("off", "observe", "flag", "redact", "block")
IMPLEMENTED_ACTIONS: frozenset[str] = frozenset({"off", "observe", "flag", "block"})
TAINTING_ACTIONS: frozenset[str] = frozenset({"flag", "redact", "block"})
"""A hit under one of these tightens gating for the rest of the run (``on_flag``).
``observe`` is for measuring, so it never changes behaviour."""
MARK_MODES: tuple[str, ...] = ("never", "flagged", "always")
SCOPES: tuple[str, ...] = ("run",)
SCORERS: tuple[str, ...] = ("stackone",)
_KEYS = {
    "scorer",
    "semantic",
    "threshold",
    "default_action",
    "tools",
    "mark_untrusted",
    "on_flag",
}
_ON_FLAG_KEYS = {"require_approval", "scope"}


@dataclass(frozen=True)
class InjectionPolicy:
    scorer: str = "stackone"
    semantic: bool = True
    threshold: float | None = None
    default_action: str = "block"
    tools: tuple[tuple[str, str], ...] = ()
    """Ordered ``(glob, action)`` pairs; the first match wins."""
    mark_untrusted: str = "flagged"
    """Wrap results as untrusted data: ``never``; only ``flagged`` hits (under the
    ``flag`` action); or ``always`` (every scored result that reaches the model)."""
    require_approval: tuple[str, ...] = ()
    """Tool globs that need approval once the run has been tainted by a hit."""
    scope: str = "run"

    def action_for(self, tool_name: str) -> str:
        for pattern, action in self.tools:
            if fnmatch.fnmatchcase(tool_name, pattern):
                return action
        return self.default_action

    def needs_approval_when_tainted(self, tool_name: str) -> bool:
        return any(fnmatch.fnmatchcase(tool_name, p) for p in self.require_approval)

    @property
    def can_mark(self) -> bool:
        """Whether this policy can ever wrap a result (so the model needs the rule)."""
        if self.mark_untrusted == "never":
            return False
        actions = {self.default_action, *(a for _, a in self.tools)}
        if self.mark_untrusted == "always":
            return bool(actions - {"off"})
        return "flag" in actions


def _action(value: Any, where: str) -> str:
    if value not in ACTIONS:
        raise ValueError(f"{where} must be one of {ACTIONS}, got {value!r}")
    if value not in IMPLEMENTED_ACTIONS:
        raise ValueError(
            f"{where}: action {value!r} is not implemented yet; "
            f"available: {sorted(IMPLEMENTED_ACTIONS)}"
        )
    return value


def _err(msg: str) -> ValueError:
    return ValueError(f"guardrails.injection.{msg}")


def parse_injection_policy(config: Mapping[str, Any] | None) -> InjectionPolicy:
    """Validate a spec ``guardrails.injection`` block into a policy."""
    if not config:
        return InjectionPolicy()
    if not isinstance(config, Mapping):
        raise ValueError("guardrails.injection must be a mapping")
    unknown = set(config) - _KEYS
    if unknown:
        raise ValueError(f"guardrails.injection: unknown keys {sorted(unknown)}")

    scorer = config.get("scorer", "stackone")
    if scorer not in SCORERS:
        raise _err(f"scorer must be one of {SCORERS}, got {scorer!r}")
    semantic = config.get("semantic", True)
    if not isinstance(semantic, bool):
        raise _err("semantic must be true or false")
    threshold = config.get("threshold")
    if threshold is not None:
        if isinstance(threshold, bool) or not isinstance(threshold, int | float):
            raise _err("threshold must be a number or null")
        if not 0.0 < float(threshold) < 1.0:
            raise _err("threshold must be between 0 and 1")
        if not semantic:
            raise _err("threshold needs semantic: true")
        threshold = float(threshold)

    default_action = _action(
        config.get("default_action", "block"), "guardrails.injection.default_action"
    )
    raw_tools = config.get("tools") or {}
    if not isinstance(raw_tools, Mapping):
        raise _err("tools must map tool globs to actions")
    tools = tuple(
        (str(pattern), _action(action, f"guardrails.injection.tools[{pattern!r}]"))
        for pattern, action in raw_tools.items()
    )

    mark = config.get("mark_untrusted", "flagged")
    if mark not in MARK_MODES:
        raise _err(f"mark_untrusted must be one of {MARK_MODES}, got {mark!r}")

    on_flag = config.get("on_flag") or {}
    if not isinstance(on_flag, Mapping):
        raise _err("on_flag must be a mapping")
    unknown = set(on_flag) - _ON_FLAG_KEYS
    if unknown:
        raise _err(f"on_flag: unknown keys {sorted(unknown)}")
    require = on_flag.get("require_approval") or []
    if not (
        isinstance(require, list) and all(isinstance(t, str) and t for t in require)
    ):
        raise _err("on_flag.require_approval must be a list of tool globs")
    scope = on_flag.get("scope", "run")
    if scope == "conversation":
        raise _err("on_flag.scope 'conversation' is not implemented yet")
    if scope not in SCOPES:
        raise _err(f"on_flag.scope must be one of {SCOPES}, got {scope!r}")

    return InjectionPolicy(
        scorer=scorer,
        semantic=semantic,
        threshold=threshold,
        default_action=default_action,
        tools=tools,
        mark_untrusted=mark,
        require_approval=tuple(require),
        scope=scope,
    )
