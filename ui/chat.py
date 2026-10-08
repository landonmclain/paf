"""Run the chat CLI from a source checkout; the same as ``paf chat``.

python ui/chat.py [agent] [--profile P] [--id CID | --new | --resume N|NAME]
"""

from paf.cli.chat import main

if __name__ == "__main__":
    raise SystemExit(main())
