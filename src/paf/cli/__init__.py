"""paf's command line: ``paf chat [agent] [options]``."""

from __future__ import annotations

import sys

_USAGE = """\
usage: paf <command> [options]

commands:
  chat    chat with an agent in the terminal (paf chat --help)
"""


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args or args[0] in ("-h", "--help"):
        print(_USAGE, end="")
        return 0 if args else 2
    command, rest = args[0], args[1:]
    if command == "chat":
        from paf.cli.chat import main as chat_main

        return chat_main(rest, prog="paf chat")
    print(f"paf: unknown command {command!r}\n\n{_USAGE}", end="", file=sys.stderr)
    return 2
