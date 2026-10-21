"""paf-owned declarative tool-gating policy (spec ``gating:`` block).

``tool_gating.py`` holds the original hard-coded gate. This module is the
spec-driven successor: an ordered rule list that decides, per tool call,
whether the runtime lets it run (``allow``), refuses it (``block``), or defers
it to a human (``approve``). Rules can look at one argument — a workspace path —
so the policy can say "writes are fine under ``out/``, anything else needs a
human", which the profile allowlist (coarse, per capability) cannot express.

First matching rule wins; no match falls to ``default``. Anything the policy
cannot interpret (a non-string path, an absolute path, ``..``) counts as
OUTSIDE every allowed directory, so ambiguity fails closed.

``approve`` needs a host that can show the request to a person and resume the
run. Until a transport does, the builder passes ``approval_available=False`` and
``approve`` is enforced as a ``block`` with an explanatory message — never as a
deferred call nothing will answer, which would end the run in a state no one
handles.
"""

from __future__ import annotations

import posixpath
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic_ai_harness.guardrails import GuardrailResult, ToolCallInfo

Action = Literal["allow", "block", "approve"]
_ACTIONS: tuple[str, ...] = ("allow", "block", "approve")


@dataclass(frozen=True)
class GatingRule:
    """One ordered rule: which tools, under what path condition, what outcome."""

    tools: frozenset[str]
    action: Action
    path_arg: str = "path"
    """The tool argument holding a workspace-relative path."""
    inside: tuple[str, ...] | None = None
    """When set, the rule matches only if the path is inside one of these
    workspace-relative directories."""
    outside: tuple[str, ...] | None = None
    """When set, the rule matches only if the path is NOT inside any of these
    workspace-relative directories. With neither ``inside`` nor ``outside`` the
    rule matches every call to its tools."""
    message: str | None = None


@dataclass(frozen=True)
class GatingPolicy:
    rules: tuple[GatingRule, ...] = ()
    default: Action = "allow"


def _normalize_dir(directory: str) -> str:
    return posixpath.normpath(directory.strip().strip("/"))


def _inside(path: Any, directories: Sequence[str]) -> bool:
    """Whether ``path`` is a relative path that stays inside one of ``directories``."""
    if not isinstance(path, str) or not path.strip():
        return False
    if path.startswith(("/", "~")) or "\\" in path:
        return False
    normal = posixpath.normpath(path)
    if normal == ".." or normal.startswith("../") or normal == ".":
        return False
    return any(
        normal == d or normal.startswith(d + "/")
        for d in map(_normalize_dir, directories)
    )


def _matches(rule: GatingRule, name: str, args: Mapping[str, Any]) -> bool:
    if name not in rule.tools:
        return False
    path = args.get(rule.path_arg)
    if rule.inside is not None and not _inside(path, rule.inside):
        return False
    if rule.outside is not None and _inside(path, rule.outside):
        return False
    return True


def parse_policy(config: Mapping[str, Any] | None) -> GatingPolicy | None:
    """Build a policy from a spec ``gating:`` block, or ``None`` when absent.

    Raises ``ValueError`` on a malformed block: a security policy that is
    silently half-applied is worse than one that refuses to load.
    """
    if not config:
        return None
    default = config.get("default", "allow")
    if default not in _ACTIONS:
        raise ValueError(f"gating.default must be one of {_ACTIONS}, got {default!r}")
    rules: list[GatingRule] = []
    for i, raw in enumerate(config.get("rules") or []):
        if not isinstance(raw, Mapping):
            raise ValueError(f"gating.rules[{i}] must be a mapping")
        tools = raw.get("tools")
        if not (
            isinstance(tools, list) and tools and all(isinstance(t, str) for t in tools)
        ):
            raise ValueError(f"gating.rules[{i}].tools must be a non-empty string list")
        action = raw.get("action")
        if action not in _ACTIONS:
            raise ValueError(
                f"gating.rules[{i}].action must be one of {_ACTIONS}, got {action!r}"
            )
        scopes: dict[str, tuple[str, ...] | None] = {}
        for key in ("inside", "outside"):
            dirs = raw.get(key)
            if dirs is not None and not (
                isinstance(dirs, list) and all(isinstance(d, str) and d for d in dirs)
            ):
                raise ValueError(
                    f"gating.rules[{i}].{key} must be a list of directories"
                )
            scopes[key] = tuple(dirs) if dirs is not None else None
        rules.append(
            GatingRule(
                tools=frozenset(tools),
                action=action,
                path_arg=str(raw.get("path_arg", "path")),
                inside=scopes["inside"],
                outside=scopes["outside"],
                message=raw.get("message"),
            )
        )
    return GatingPolicy(rules=tuple(rules), default=default)


def _block_message(name: str, rule: GatingRule | None, action: Action) -> str:
    if rule is not None and rule.message:
        return rule.message
    if action == "approve":
        return (
            f"Tool '{name}' needs human approval for this call, and approval is "
            "not available in this environment, so it was refused. Do not retry "
            "it; tell the user what you needed to do and why."
        )
    return (
        f"Tool '{name}' is blocked by this agent's gating policy. The runtime "
        "refused this call before it executed."
    )


def _approval_reason(name: str, rule: GatingRule | None) -> str:
    """Why this call needs a person, in words a person can act on."""
    if rule is None:
        return f"this agent's gating policy requires approval for every '{name}' call"
    where = ""
    if rule.inside is not None:
        where = f" under {', '.join(rule.inside)}"
    elif rule.outside is not None:
        where = f" outside {', '.join(rule.outside)}"
    target = f" ({rule.path_arg}{where})" if where else ""
    return f"this agent's gating policy requires approval for '{name}'{target}"


def make_gating_guard(
    policy: GatingPolicy,
    *,
    approval_available: bool | Callable[[], bool] = False,
) -> Callable[[ToolCallInfo], GuardrailResult]:
    """Compile a policy into a ``ToolGuardrail`` ``guard=`` callable.

    ``approval_available`` may be a callable checked at call time (the builder
    passes ``paf.approvals.approvals_available``: is a person attached to THIS
    turn?). When it is, an ``approve`` outcome asks that person, with the reason
    attached for the approval request; when it is not, ``approve`` is refused
    exactly as before approvals existed.
    """

    def available() -> bool:
        return (
            approval_available() if callable(approval_available) else approval_available
        )

    def guard(info: ToolCallInfo) -> GuardrailResult:
        rule = next(
            (r for r in policy.rules if _matches(r, info.name, info.args)), None
        )
        action = rule.action if rule is not None else policy.default
        if action == "allow":
            return GuardrailResult.allow()
        if action == "approve" and available():
            from paf.approvals import current_channel

            channel = current_channel()
            if channel is not None:
                channel.annotate(
                    info.tool_call_id, reason=_approval_reason(info.name, rule)
                )
            return GuardrailResult.approve()
        return GuardrailResult.block(_block_message(info.name, rule, action))

    return guard
