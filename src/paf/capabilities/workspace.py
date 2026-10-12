"""Workspace: tell the model where each tool family's paths start, and the time zone.

Root-bearing capabilities resolve relative paths against DIFFERENT roots (file
tools at ``~``, git/codebase-map at a repo, the vault at a notes folder), and
nothing told the model which. It would read ``ui/chat.py`` from the codebase
map and hand that to ``read_file`` rooted at ``~``. This capability renders one
static ``<workspace>`` block into the instructions, built from the roots the
OTHER capabilities were actually constructed with — so it can never drift from
the spec — plus the user's time zone (message ``[date: …]`` stamps are local;
see ``paf.clock``) and any spec ``workspace.notes``.

Static per agent build (no per-turn content), so it never breaks prompt
caching. The zone line names the current abbreviation/offset (CDT/CST), which
changes only at DST transitions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic_ai.capabilities import AbstractCapability

from paf.clock import now_local, zone_name

# (capability class name, root attribute, label, tools, how paths resolve)
_ROOTED: tuple[tuple[str, str, str, str, str], ...] = (
    (
        "FileSystem",
        "root_dir",
        "File tools",
        "read_file, write_file, edit_file, list_directory, search_files, find_files",
        "paths relative to {root}",
    ),
    ("Shell", "cwd", "Shell", "run_command, start_command", "commands start in {root}"),
    (
        "Python",
        "cwd",
        "Python",
        "run_python",
        "programs run in {root}; fresh interpreter per call",
    ),
    ("Git", "root", "Git", "git_*", "repo {root}; paths relative to it"),
    (
        "CodebaseMap",
        "scan_root",
        "Codebase map",
        "get_overview, lookup_file, read_block, search_codebase",
        "repo {root}; paths relative to it",
    ),
    (
        "Vault",
        "vault_root",
        "Vault",
        "vault_overview, lookup_note, read_note, note_*",
        "notes under {root}; note paths relative to it",
    ),
)


def _resolved(root: Any) -> Path:
    return Path(os.path.expanduser(str(root))).resolve()


@dataclass
class Workspace(AbstractCapability):
    """A static ``<workspace>`` block: tool roots, path rule, time zone, notes."""

    roots: list[tuple[str, str, str, Path]] = field(default_factory=list)
    """``(label, tools, how, root)`` per root-bearing capability that was built."""

    notes: list[str] = field(default_factory=list)
    """Extra always-present lines from the spec's ``workspace.notes``."""

    expose_paths: bool = True
    """Whether the block names host filesystem paths (home dir, tool roots).

    ``False`` for locked-down agents: every root is described as "the workspace
    root" and absolute paths are not mentioned. Besides not telling the model
    where the user's home is, this keeps the text true when a tool runs in a
    sandbox that mounts the workspace at a different path than the host's.
    """

    def render(self) -> str:
        now = now_local()
        lines = [] if not self.expose_paths else [f"Home: {Path.home()} (~)."]
        lines += [
            f"Message timestamps ([date: …]) are local time in {zone_name()}"
            f" — currently {now.tzname()}, UTC{now.strftime('%z')[:3]}:"
            f"{now.strftime('%z')[3:]}.",
        ]
        if self.roots and not self.expose_paths:
            lines.append("Tools resolve paths relative to the workspace root:")
            lines += [
                f"- {label} ({tools}): {how.format(root='the workspace root')}"
                for label, tools, how, _ in self.roots
            ]
            lines.append("Use relative paths; absolute paths are not supported.")
        elif self.roots:
            lines.append("Tools resolve relative paths against different roots:")
            lines += [
                f"- {label} ({tools}): {how.format(root=root)}"
                for label, tools, how, root in self.roots
            ]
            lines.append(
                "Absolute paths work with every tool above — use them when "
                "passing a path from one tool to another."
            )
            example = self._example()
            if example:
                lines.append(example)
        lines += self.notes
        return "<workspace>\n" + "\n".join(lines) + "\n</workspace>"

    def _example(self) -> str | None:
        """A repo-relative → file-tools conversion, when those roots differ."""
        by_label = {label: root for label, _, _, root in self.roots}
        files = by_label.get("File tools")
        repo = by_label.get("Codebase map") or by_label.get("Git")
        if files is None or repo is None or repo == files:
            return None
        return (
            f"Example: repo path src/app.py → {repo / 'src' / 'app.py'} for file tools."
        )

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Not spec-declared: built from the other capabilities' live roots."""
        return None

    def get_instructions(self) -> list[str] | None:
        return [self.render()]


def build_workspace(
    capabilities: list[Any],
    notes: list[str] | None = None,
    expose_paths: bool = True,
) -> Workspace:
    """Collect the roots of the built capabilities into a :class:`Workspace`.

    Matched by class name + root attribute so this module needs no import of
    the capability classes (and no import cycle through ``build``).
    """
    roots: list[tuple[str, str, str, Path]] = []
    for cls_name, attr, label, tools, how in _ROOTED:
        for cap in capabilities:
            if type(cap).__name__ == cls_name and getattr(cap, attr, None):
                roots.append((label, tools, how, _resolved(getattr(cap, attr))))
                break
    return Workspace(roots=roots, notes=list(notes or []), expose_paths=expose_paths)
