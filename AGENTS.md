# AGENTS.md — paf operating guide

Guide for any agent (Claude Code, Codex, Cursor, ...) working in this repo. The web
front end is the SSE server (`ui/sse.py`) plus the separate paf-ui Next.js app. The old
Gradio UI (`ui/web.py`) is retired; don't use it. Test and benchmark details are in
[`tests/AGENTS.md`](tests/AGENTS.md); orientation is in `README.md`.

> To load `.env` into a shell use `set -a && . ./.env && set +a`; plain `source` does not
> export the variables. If a `PYTHONPATH` is inherited from another tool and imports
> fail with a `pydantic_core` error, clear it (`env -u PYTHONPATH ...`).

## Everyday commands

```bash
.venv/bin/python -m pytest -q -n 8                    # unit suite, ~50s
.venv/bin/ruff check src tests ui --output-format=concise
npx --yes pyright                                                  # typecheck
```

`.githooks/pre-push` runs ruff, the format check and pyright over `src tests ui`.

## Servers are systemd user services

| Service | What | Port |
| --- | --- | --- |
| `paf-sse` | `ui/sse.py` SSE + REST transport over `Session` | 8090 |
| `paf-ops` | `ui/ops.py` systemd control for paf-sse | 8091 |
| `paf-ui` | paf-ui Next.js production server (put it behind your own HTTPS proxy if you need one; list its origin in `PAF_EXTRA_ORIGINS`) | 3001 |

Don't start these by hand: a second copy collides on the port and leaves orphans.
After changing Python under `src/` or `ui/`, restart `paf-sse`:

```bash
systemctl --user restart paf-sse && sleep 4 && systemctl --user is-active paf-sse
journalctl --user -u paf-sse -f          # live logs; check here first for chat errors
systemctl --user daemon-reload           # after editing a unit
```

Units live in `~/.config/systemd/user/paf-{sse,ops,ui}.service`.
`scripts/paf-sse-start.sh [port]` is the wrapper they call and the manual run path.

## CLI

`paf chat [agent] [--new | --id CID | --resume N|NAME] [--profile P]` (code in
`src/paf/cli/`; `ui/chat.py` is the same command from a checkout). The default agent is
`demo_assistant`.

- Runs turns in-process, so it does not need `paf-sse`.
- With no conversation flag, an interactive run reopens the last conversation; piped runs
  use lane `chat-<agent>`. `/list`, `/resume`, `/name` manage conversations; `/help`
  lists the rest.
- Interactive mode needs a tty. To drive it from an agent, pipe the prompt in: the whole
  of stdin is one message, then it exits.
- Ctrl-C cancels a turn. Background shell and subagent completions stream a follow-up
  turn at the prompt.

## Specs and profiles

- A spec (`specs/<agent>.yaml`) configures an agent; its `profile:` names a capability
  allowlist in `src/paf/capabilities/profiles.py`. Every spec key is documented in
  `docs/spec-reference.md`.
- Shipped: `demo_assistant` (default), `secure_assistant`, the sub-agent leaves under
  `specs/sub_agents/`, and the benchmark specs. `specs/assistant.yaml` is a personal,
  git-ignored spec.
- The `assistant` profile name is kept on purpose: conversations are stored under
  `{profile}:{cid}`, so renaming it would orphan saved history.

## Lint

Clean `ruff` before and after edits. Code under `archive/` is untracked and not linted.
