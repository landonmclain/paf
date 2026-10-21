"""Understand a shell command well enough to decide whether a person should see it.

The old check looked only at the first word, so ``ls; rm -rf x``, ``sudo rm``,
``bash -c 'rm ...'``, ``xargs rm`` and ``find -delete`` all passed while a plain
``rm`` was refused, which taught the model the workaround. This parses the
command with the bash grammar and reports every simple command that would run:

- across ``;`` ``&&`` ``||`` pipes, subshells, ``$( )`` and backticks;
- through wrappers (``sudo env nohup time timeout nice xargs exec command``),
  re-parsing ``bash -c`` / ``sh -c`` strings and ``find -exec`` commands;
- with the *effects* that matter for approval: ``delete`` (rm and friends,
  ``find -delete``, ``git rm`` / ``git clean``, interpreter one-liners that
  delete), ``discard`` (git changes thrown away), ``force_push``, ``disk``,
  ``permissions``, ``privileged`` (sudo), and ``script`` (a local script whose
  contents are not visible).

This reads the command TEXT. It cannot see what a script does, and variables,
``eval`` and aliases defeat it; those are reported as ``problems`` so policy can
treat them conservatively. It is a seatbelt for mistakes, not a sandbox.
"""

from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache

_WRAPPERS = {
    "sudo",
    "doas",
    "env",
    "nohup",
    "time",
    "timeout",
    "nice",
    "ionice",
    "stdbuf",
    "exec",
    "command",
    "builtin",
    "xargs",
    "chronic",
}
# Wrapper options that consume the following word.
_WRAPPER_OPTS_WITH_VALUE = {
    "sudo": {"-u", "-g", "-C", "-h", "-p", "-U", "-r", "-t", "-D"},
    "doas": {"-u", "-C"},
    "env": {"-u", "-C", "-S"},
    "timeout": {"-s", "-k", "--signal", "--kill-after"},
    "nice": {"-n", "--adjustment"},
    "ionice": {"-c", "-n", "-p"},
    "stdbuf": {"-i", "-o", "-e"},
    "xargs": {
        "-a",
        "-d",
        "-E",
        "-I",
        "-L",
        "-n",
        "-P",
        "-s",
        "--arg-file",
        "--delimiter",
        "--max-args",
        "--max-procs",
    },
}
_SHELLS = {"bash", "sh", "zsh", "dash", "ksh", "fish"}
_EVALS = {"eval", "source", "."}
_DELETE_COMMANDS = {"rm", "rmdir", "unlink", "shred", "srm", "trash", "trash-put"}
_DISK_COMMANDS = {"dd", "fdisk", "sfdisk", "parted", "wipefs", "mkswap"}
_PERM_COMMANDS = {"chmod", "chown", "chgrp"}
_INTERPRETERS = re.compile(r"^(python[0-9.]*|perl|ruby|node|php)$")
_DELETE_API = re.compile(
    r"os\.(remove|unlink|rmdir|removedirs)|shutil\.rmtree|\.unlink\(|\.rmdir\(|"
    r"fs\.(rm|rmSync|unlink|unlinkSync|rmdir|rmdirSync)\b|File\.delete|FileUtils\.rm|"
    r"\bunlink\s*\(|rm_rf|rmtree"
)
_SYSTEM_PREFIXES = ("/usr/", "/bin/", "/sbin/", "/opt/homebrew/", "/usr/local/")


@dataclass(frozen=True)
class SimpleCommand:
    """One command that would execute, after unwrapping."""

    argv: tuple[str, ...]
    raw: str
    via: tuple[str, ...] = ()
    """How it was reached: ``sudo``, ``bash -c``, ``xargs``, ``find -exec``..."""
    dynamic: bool = False
    """Some argument is computed at run time ($VAR, $(...)): targets may differ."""
    effects: frozenset[str] = field(default_factory=frozenset)
    targets: tuple[str, ...] = ()
    """Path-like arguments of a destructive command (as written)."""

    @property
    def name(self) -> str:
        return os.path.basename(self.argv[0]) if self.argv else ""

    @property
    def flags(self) -> frozenset[str]:
        out: set[str] = set()
        for arg in self.argv[1:]:
            if arg == "--":
                break
            if arg.startswith("--"):
                out.add(arg.split("=", 1)[0])
            elif arg.startswith("-") and len(arg) > 1 and not arg[1].isdigit():
                out.add(arg)
                out.update(f"-{c}" for c in arg[1:])
        return frozenset(out)

    @property
    def subcommand(self) -> str:
        return _subcommand(self.argv)


@dataclass(frozen=True)
class CommandAnalysis:
    command: str
    commands: tuple[SimpleCommand, ...]
    problems: tuple[str, ...] = ()
    """Why the analysis may be incomplete: parse error, eval, dynamic command name."""

    @property
    def effects(self) -> frozenset[str]:
        return (
            frozenset().union(*(c.effects for c in self.commands))
            if self.commands
            else frozenset()
        )


# --- parsing ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _parser():
    from tree_sitter_language_pack import get_parser

    return get_parser("bash")


_ESCAPE = re.compile(r"\\(.)", re.DOTALL)


def _text(node, src: bytes) -> str:
    return src[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _word(node, src: bytes) -> tuple[str, bool]:
    """(literal value, dynamic?) of one argument node, quotes removed."""
    kind = node.type
    if kind in (
        "simple_expansion",
        "expansion",
        "command_substitution",
        "process_substitution",
        "arithmetic_expansion",
    ):
        return _text(node, src), True
    if kind == "raw_string":
        return _text(node, src)[1:-1], False
    if kind == "ansi_c_string":
        return _text(node, src)[2:-1], False
    if kind == "string":
        dynamic = any(
            c.type
            in (
                "simple_expansion",
                "expansion",
                "command_substitution",
                "arithmetic_expansion",
            )
            for c in node.children
        )
        return _text(node, src)[1:-1], dynamic
    if kind == "concatenation":
        parts = [_word(c, src) for c in node.children]
        return "".join(p for p, _ in parts), any(d for _, d in parts)
    if kind in ("word", "number"):
        return _ESCAPE.sub(r"\1", _text(node, src)), False
    return _text(node, src), True


def _argv(node, src: bytes) -> tuple[list[str], bool, bool]:
    """(argv, any_dynamic, command_name_dynamic) for a ``command`` node."""
    argv: list[str] = []
    dynamic = False
    name_dynamic = False
    for child in node.children:
        if child.type in (
            "variable_assignment",
            "file_redirect",
            "heredoc_redirect",
            "herestring_redirect",
        ):
            continue
        if child.type == "command_name":
            inner = child.children[0] if child.children else child
            value, dyn = _word(inner, src)
            name_dynamic = dyn
        else:
            if not child.is_named:
                continue
            value, dyn = _word(child, src)
        argv.append(value)
        dynamic = dynamic or dyn
    return argv, dynamic, name_dynamic


def analyze(command: str) -> CommandAnalysis:
    """Every simple command ``command`` would run, with its effects."""
    problems: list[str] = []
    out: list[SimpleCommand] = []
    _analyze_into(command, (), out, problems, depth=0)
    return CommandAnalysis(
        command=command, commands=tuple(out), problems=tuple(dict.fromkeys(problems))
    )


def _analyze_into(
    command: str, via: tuple[str, ...], out: list, problems: list, depth: int
) -> None:
    if depth > 4:
        problems.append("nested too deeply")
        return
    src = command.encode("utf-8")
    tree = _parser().parse(src)
    if tree.root_node.has_error:
        problems.append("could not parse the command")

    def walk(node) -> None:
        if node.type == "command":
            argv, dynamic, name_dynamic = _argv(node, src)
            if name_dynamic:
                problems.append("the command name is computed at run time")
            if argv:
                _expand(argv, _text(node, src), via, dynamic, out, problems, depth)
        for child in node.children:
            walk(child)

    walk(tree.root_node)


def _strip_wrapper(name: str, args: list[str]) -> list[str]:
    """Drop a wrapper's own options (and env's VAR=value words)."""
    opts = _WRAPPER_OPTS_WITH_VALUE.get(name, set())
    i = 0
    while i < len(args):
        arg = args[i]
        if name == "env" and "=" in arg and not arg.startswith("-"):
            i += 1
            continue
        if arg == "--":
            return args[i + 1 :]
        if arg.startswith("-") and len(arg) > 1:
            i += 2 if arg in opts else 1
            continue
        if name == "timeout":  # the duration
            return args[i + 1 :]
        if name in ("nice", "ionice") and arg.lstrip("-").isdigit():
            i += 1
            continue
        return args[i:]
    return []


def _expand(argv, raw, via, dynamic, out, problems, depth) -> None:
    name = os.path.basename(argv[0])
    rest = list(argv[1:])
    if name in _WRAPPERS:
        inner = _strip_wrapper(name, rest)
        if not inner:
            return  # e.g. bare `time` or `env` (prints the environment)
        extra = ("privileged",) if name in ("sudo", "doas") else ()
        before = len(out)
        _expand(inner, raw, (*via, name), dynamic, out, problems, depth)
        if extra:
            for i in range(before, len(out)):
                c = out[i]
                out[i] = SimpleCommand(
                    c.argv, c.raw, c.via, c.dynamic, c.effects | set(extra), c.targets
                )
        return
    if name in _EVALS:
        problems.append(f"`{name}` runs text this analysis cannot see")
        out.append(SimpleCommand(tuple(argv), raw, via, True, frozenset({"script"})))
        return
    if name in _SHELLS:
        if "-c" in rest:
            idx = rest.index("-c")
            if idx + 1 < len(rest):
                if dynamic:
                    problems.append(f"`{name} -c` runs a string computed at run time")
                _analyze_into(
                    rest[idx + 1], (*via, f"{name} -c"), out, problems, depth + 1
                )
                return
        files = [a for a in rest if not a.startswith("-")]
        if files:  # `bash script.sh`
            out.append(
                SimpleCommand(tuple(argv), raw, via, dynamic, frozenset({"script"}))
            )
            return
    if name == "find":
        _find(argv, raw, via, dynamic, out, problems, depth)
        return
    effects, targets = _effects(argv)
    out.append(
        SimpleCommand(
            tuple(argv), raw, via, dynamic, frozenset(effects), tuple(targets)
        )
    )


def _find(argv, raw, via, dynamic, out, problems, depth) -> None:
    roots: list[str] = []
    i = 1
    while i < len(argv) and not (argv[i].startswith("-") or argv[i] in ("(", "!", ")")):
        roots.append(argv[i])
        i += 1
    effects: set[str] = set()
    if "-delete" in argv[i:]:
        effects.add("delete")
    out.append(
        SimpleCommand(
            tuple(argv), raw, via, dynamic, frozenset(effects), tuple(roots or ["."])
        )
    )
    # -exec / -execdir / -ok / -okdir CMD ... ; or +
    j = i
    while j < len(argv):
        if argv[j] in ("-exec", "-execdir", "-ok", "-okdir"):
            k = j + 1
            inner: list[str] = []
            while k < len(argv) and argv[k] not in (";", "+", "\\;"):
                inner.append(argv[k])
                k += 1
            if inner:
                inner = [
                    a if a != "{}" else (roots[0] if roots else ".") for a in inner
                ]
                _expand(
                    inner, raw, (*via, f"find {argv[j]}"), True, out, problems, depth
                )
            j = k
        j += 1


def _positional(args: list[str]) -> list[str]:
    out, after_dashdash = [], False
    for arg in args:
        if after_dashdash:
            out.append(arg)
        elif arg == "--":
            after_dashdash = True
        elif not (arg.startswith("-") and len(arg) > 1):
            out.append(arg)
    return out


_GIT_GLOBAL_WITH_VALUE = {"-C", "-c", "--git-dir", "--work-tree", "--namespace"}


def _subcommand(argv) -> str:
    if not argv:
        return ""
    if os.path.basename(argv[0]) != "git":
        rest = [a for a in argv[1:] if not a.startswith("-")]
        return rest[0] if rest else ""
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg in _GIT_GLOBAL_WITH_VALUE:
            i += 2
            continue
        if arg.startswith("-"):
            i += 1
            continue
        return arg
    return ""


def _effects(argv: list[str]) -> tuple[set[str], list[str]]:
    name = os.path.basename(argv[0])
    args = list(argv[1:])
    effects: set[str] = set()
    targets: list[str] = []
    if name in _DELETE_COMMANDS:
        effects.add("delete")
        targets = _positional(args)
    elif name in _DISK_COMMANDS or name.startswith("mkfs"):
        effects.add("disk")
    elif name in _PERM_COMMANDS:
        effects.add("permissions")
        targets = _positional(args)[1:]
    elif _INTERPRETERS.match(name):
        for flag in ("-c", "-e", "-E"):
            if flag in args:
                idx = args.index(flag)
                code = args[idx + 1] if idx + 1 < len(args) else ""
                if _DELETE_API.search(code):
                    effects.add("delete")
                break
        else:
            scripts = [a for a in args if not a.startswith("-")]
            if scripts and not scripts[0].startswith(("-",)) and scripts[0] != "-m":
                if "-m" not in args:
                    effects.add("script")
    elif name == "git":
        effects |= _git_effects(argv)
        if "delete" in effects:
            targets = (
                _positional(argv[argv.index(_subcommand(argv)) + 1 :])
                if _subcommand(argv)
                else []
            )
    if "/" in argv[0] and not argv[0].startswith(_SYSTEM_PREFIXES):
        effects.add("script")
    return effects, targets


def _git_effects(argv: list[str]) -> set[str]:
    sub = _subcommand(argv)
    if not sub:
        return set()
    rest = argv[argv.index(sub) + 1 :]
    flags = set()
    for a in rest:
        if a.startswith("--"):
            flags.add(a.split("=", 1)[0])
        elif a.startswith("-") and len(a) > 1:
            flags.add(a)
            flags.update(f"-{c}" for c in a[1:])
    if sub == "rm" and "--cached" not in flags:
        return {"delete"}
    if sub == "clean" and not ({"-n", "--dry-run"} & flags):
        return {"delete"}
    if sub == "reset" and "--hard" in flags:
        return {"discard"}
    if sub == "checkout" and ("--" in rest or "." in rest or {"-f", "--force"} & flags):
        return {"discard"}
    if sub == "restore" and not (
        {"--staged", "-S"} & flags and not {"--worktree", "-W"} & flags
    ):
        return {"discard"}
    if sub == "branch" and (
        {"-D"} & flags or ({"-d", "--delete"} & flags and {"-f", "--force"} & flags)
    ):
        return {"discard"}
    if sub == "stash" and rest and rest[0] in ("drop", "clear"):
        return {"discard"}
    if sub == "push" and (
        {"-f", "--force", "--force-with-lease", "--mirror", "--delete", "-d"} & flags
        or any(a.startswith(("+", ":")) for a in rest if not a.startswith("-"))
    ):
        return {"force_push"}
    return set()


def matches_any(value: str, patterns) -> bool:
    return any(fnmatch.fnmatchcase(value, p) for p in patterns)
