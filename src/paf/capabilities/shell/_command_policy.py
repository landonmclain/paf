"""Shell command policy: allow, ask a person, or refuse, per command.

Declared in the spec's ``shell.command_policy`` block:

    shell:
      command_policy:
        default: allow            # simple commands no rule matches
        unparseable: approve      # parse errors, eval, computed command names
        unknown_scripts: allow    # ./clean.sh, bash deploy.sh: contents unseen
        rules:                    # per simple command; first match wins
          - effects_any: [delete]
            targets_any: ["/", "/*", "~", "~/", "$HOME", "/home", "/etc", "/usr"]
            action: block
            why: deletes the root, your home, or a system directory
          - effects_any: [delete, discard, force_push, disk]
            action: approve
          - commands: [chmod, chown]
            flags_any: [-R, --recursive]
            action: approve

A rule's conditions are ANDed; any omitted condition matches everything:
``commands`` (name globs), ``subcommands`` (``git push``), ``flags_any``,
``effects_any`` (from ``_analyze``), ``targets_any`` (path patterns, checked
both as written and resolved against the shell's directory). A compound
command takes the strictest decision of its parts: block > approve > allow.

When a spec declares a command policy it REPLACES the shell's first-word deny
list, so a destructive command reaches the person instead of being refused (a
refusal is what taught the model to work around it).
"""

from __future__ import annotations

import fnmatch
import os
import posixpath
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from ._analyze import CommandAnalysis, SimpleCommand

ACTIONS = ("allow", "approve", "block")
_RANK = {"allow": 0, "approve": 1, "block": 2}
_RULE_KEYS = {
    "commands",
    "subcommands",
    "flags_any",
    "effects_any",
    "targets_any",
    "action",
    "why",
}
_KEYS = {"default", "unparseable", "unknown_scripts", "rules"}
EFFECTS = (
    "delete",
    "discard",
    "force_push",
    "disk",
    "permissions",
    "privileged",
    "script",
)
_DESCRIBE = {
    "delete": "deletes files",
    "discard": "throws away uncommitted or local git changes",
    "force_push": "rewrites history on a remote",
    "disk": "writes to a disk device",
    "permissions": "changes permissions or ownership",
    "privileged": "runs with elevated privileges",
    "script": "runs a script whose contents are not visible",
}


@dataclass(frozen=True)
class CommandRule:
    action: str
    commands: tuple[str, ...] | None = None
    subcommands: tuple[str, ...] | None = None
    flags_any: tuple[str, ...] | None = None
    effects_any: tuple[str, ...] | None = None
    targets_any: tuple[str, ...] | None = None
    why: str | None = None


@dataclass(frozen=True)
class CommandPolicy:
    rules: tuple[CommandRule, ...] = ()
    default: str = "allow"
    unparseable: str = "approve"
    unknown_scripts: str = "allow"


@dataclass(frozen=True)
class CommandDecision:
    action: str
    reasons: tuple[str, ...] = ()
    deciding: tuple[SimpleCommand, ...] = field(default_factory=tuple)
    """The commands that produced the decision (for the preview)."""

    @property
    def reason(self) -> str:
        return "; ".join(self.reasons) or "the shell command policy"


# --- parsing ----------------------------------------------------------------------------
def _err(msg: str) -> ValueError:
    return ValueError(f"shell.command_policy.{msg}")


def _action(value: Any, where: str) -> str:
    if value not in ACTIONS:
        raise _err(f"{where} must be one of {ACTIONS}, got {value!r}")
    return value


def _strings(value: Any, where: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not (
        isinstance(value, list)
        and value
        and all(isinstance(v, str) and v for v in value)
    ):
        raise _err(f"{where} must be a non-empty list of strings")
    return tuple(value)


def parse_command_policy(config: Mapping[str, Any] | None) -> CommandPolicy | None:
    """Validate a ``shell.command_policy`` block; ``None`` when absent."""
    if not config:
        return None
    if not isinstance(config, Mapping):
        raise ValueError("shell.command_policy must be a mapping")
    unknown = set(config) - _KEYS
    if unknown:
        raise _err(f"unknown keys {sorted(unknown)}")
    rules = []
    for i, raw in enumerate(config.get("rules") or []):
        if not isinstance(raw, Mapping):
            raise _err(f"rules[{i}] must be a mapping")
        bad = set(raw) - _RULE_KEYS
        if bad:
            raise _err(f"rules[{i}]: unknown keys {sorted(bad)}")
        effects = _strings(raw.get("effects_any"), f"rules[{i}].effects_any")
        if effects and set(effects) - set(EFFECTS):
            raise _err(
                f"rules[{i}].effects_any: unknown effects {sorted(set(effects) - set(EFFECTS))}; known: {EFFECTS}"
            )
        rules.append(
            CommandRule(
                action=_action(raw.get("action"), f"rules[{i}].action"),
                commands=_strings(raw.get("commands"), f"rules[{i}].commands"),
                subcommands=_strings(raw.get("subcommands"), f"rules[{i}].subcommands"),
                flags_any=_strings(raw.get("flags_any"), f"rules[{i}].flags_any"),
                effects_any=effects,
                targets_any=_strings(raw.get("targets_any"), f"rules[{i}].targets_any"),
                why=raw.get("why") if isinstance(raw.get("why"), str) else None,
            )
        )
    return CommandPolicy(
        rules=tuple(rules),
        default=_action(config.get("default", "allow"), "default"),
        unparseable=_action(config.get("unparseable", "approve"), "unparseable"),
        unknown_scripts=_action(
            config.get("unknown_scripts", "allow"), "unknown_scripts"
        ),
    )


# --- evaluation -------------------------------------------------------------------------
def _norm(path: str) -> str:
    return path if path == "/" else posixpath.normpath(path).rstrip("/") or "/"


def _target_forms(target: str, cwd: Path) -> set[str]:
    home = str(Path.home())
    forms = {target, target.rstrip("/") or "/"}
    expanded = target.replace("$HOME", home).replace("${HOME}", home)
    expanded = os.path.expanduser(expanded)
    forms.add(_norm(expanded))
    if not expanded.startswith("/"):
        forms.add(_norm(str(cwd / expanded)))
    return forms


def _pattern_forms(pattern: str) -> set[str]:
    home = str(Path.home())
    forms = {pattern}
    expanded = os.path.expanduser(
        pattern.replace("$HOME", home).replace("${HOME}", home)
    )
    forms.add(_norm(expanded) if not any(c in expanded for c in "*?[") else expanded)
    return forms


def _target_hit(targets: tuple[str, ...], patterns: tuple[str, ...], cwd: Path) -> bool:
    """Does any target hit any pattern?

    A plain pattern (``/``, ``~``, ``$HOME``, ``/etc``) is an exact path, compared
    with the target both as written and resolved against ``cwd``. A glob pattern
    is matched path-aware, one component per ``*`` (``/*`` is a top-level
    directory, never ``/tmp/project/build``), and also literally as written, so
    ``rm -rf /*`` itself is caught.
    """
    for target in targets:
        forms = _target_forms(target, cwd)
        for pattern in patterns:
            is_glob = any(c in pattern for c in "*?[")
            for pf in _pattern_forms(pattern):
                for form in forms:
                    if form == pf:
                        return True
                    if (
                        is_glob
                        and form.startswith("/")
                        and pf.startswith("/")
                        and PurePosixPath(form).match(pf)
                    ):
                        return True
    return False


def _matches(rule: CommandRule, cmd: SimpleCommand, cwd: Path) -> bool:
    if rule.commands is not None and not any(
        fnmatch.fnmatchcase(cmd.name, p) for p in rule.commands
    ):
        return False
    if rule.subcommands is not None and cmd.subcommand not in rule.subcommands:
        return False
    if rule.flags_any is not None and not set(rule.flags_any) & cmd.flags:
        return False
    if rule.effects_any is not None and not set(rule.effects_any) & cmd.effects:
        return False
    if rule.targets_any is not None and not _target_hit(
        cmd.targets, rule.targets_any, cwd
    ):
        return False
    return True


def _describe(cmd: SimpleCommand, rule: CommandRule | None) -> str:
    shown = " ".join(cmd.argv)
    if len(shown) > 80:
        shown = shown[:77] + "..."
    if rule is not None and rule.why:
        why = rule.why
    else:
        effects = [e for e in EFFECTS if e in cmd.effects]
        why = (
            ", ".join(_DESCRIBE[e] for e in effects)
            or "matches the shell command policy"
        )
    via = f" (via {' → '.join(cmd.via)})" if cmd.via else ""
    return f"`{shown}`{via}: {why}"


def evaluate(
    policy: CommandPolicy, analysis: CommandAnalysis, cwd: Path
) -> CommandDecision:
    """The strictest decision over every simple command in ``analysis``."""
    best = "allow"
    reasons: list[str] = []
    deciding: list[SimpleCommand] = []

    def consider(action: str, reason: str | None, cmd: SimpleCommand | None) -> None:
        nonlocal best
        if _RANK[action] > _RANK[best]:
            best = action
            reasons.clear()
            deciding.clear()
        if action == best and action != "allow":
            if reason:
                reasons.append(reason)
            if cmd is not None:
                deciding.append(cmd)

    for cmd in analysis.commands:
        rule = next((r for r in policy.rules if _matches(r, cmd, cwd)), None)
        if rule is not None:
            consider(rule.action, _describe(cmd, rule), cmd)
        elif "script" in cmd.effects:
            consider(policy.unknown_scripts, _describe(cmd, None), cmd)
        else:
            consider(policy.default, _describe(cmd, None), cmd)
    if analysis.problems:
        consider(policy.unparseable, "; ".join(analysis.problems), None)
    return CommandDecision(best, tuple(dict.fromkeys(reasons)), tuple(deciding))
