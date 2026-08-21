# AGENTS.md — paf web UI operating guide

Concise operating guide for any agent (Claude Code, Codex, Cursor, Hermes, ...)
working on the **web chat UI (`ui/web.py`)** in this repo. This file is
web-UI-centric; the graded benchmark harness has its own guide — **see
[`tests/AGENTS.md`](tests/AGENTS.md)** before running or editing anything under
`tests/`. The repo README (`README.md`) is the broader orientation.

> **The single most important rule in this repo:** any `python`/`pytest`
> invocation MUST be prefixed with `PYTHONPATH=` (empty) or it runs Hermes'
> leaked **Python 3.11** site-packages against this repo's **Python 3.12** venv
> and dies with a `pydantic_core` ABI error. See `tests/AGENTS.md` for the full
> mechanical explanation. `.env` must be sourced with `set -a && . ./.env && set +a`
> (plain `source` does not export `OPENROUTER_API_KEY`).

## The web server is a systemd user service — do not run it ad hoc

`ui/web.py` serves the paf Gradio chat on `0.0.0.0:8080` (Tailscale
`http://100.107.176.116:8080`), default agent `demo_assistant`. It is owned by a
**systemd user service**, not a chat background process — a chat-spawned server
collides on the port and leaves orphans.

```bash
systemctl --user daemon-reload                                   # after editing the unit
systemctl --user enable --now paf-web.service                    # start + enable at login
systemctl --user start|stop|restart|status paf-web               # daily driver
journalctl --user -u paf-web -f                                  # live logs
```

- Unit: `~/.config/systemd/user/paf-web.service` — `Restart=on-failure`,
  `RestartSec=2` (crash auto-restart verified), `WantedBy=default.target`.
- Wrapper: `scripts/paf-web-start.sh` — `cd`s to root, sources `.env` (exports
  keys), `exec`s `.venv/bin/python ui/web.py`. Doubles as the manual run path.

## Run / debug manually (one-off)

```bash
cd /home/alice/projects/Personal/pydantic-agent-framework
PYTHONPATH= .venv/bin/python -m py_compile ui/web.py && \
PYTHONPATH= .venv/bin/ruff check ui/web.py --output-format=concise   # CLEAN before/after edits
# server (ONE-OFF; normally systemd runs it):
set -a && . ./.env && set +a && .venv/bin/python ui/web.py
sleep 6 && curl -s -o /dev/null -w "HTTP %{http_code}\n" http://127.0.0.1:8080/   # → 200
```

**Gotchas:**
- `python ui/web.py 8080` makes `8080` the **agent name** →
  `AgentSpecNotFoundError for '8080'`. Use no args, or `demo_assistant 8080`.
- Gradio 6.25: no `type="messages"`; bot fn must be `functools.partial(...)` of
  an async generator (a lambda breaks streaming); `.load()` needs a
  `with gr.Blocks()` context; `css`/`fill_width` go to `launch()`.
- `Session.close()` is SYNC (returns None) — do not `asyncio.run` it.

## History rendering model (ui/web.py)

- **Live** (`event_messages`): rebuild the full transcript every frame from a
  fixed base + `_assistant_messages(out)` — never accumulate via slice.
- **Restore** (`_render_session_history`): folds flat pydantic-ai messages into
  `(user, TurnStream)` turns; consecutive `UserPromptPart`s collapse to one
  user bubble; empty/interrupted turns are skipped.
- Restored history loads on **page refresh** via `demo.load` → `_load_history`
  → `session.refresh()` (re-reads the store past the one-shot `_resumed` guard).

A Hermes skill (`paf-web-ui`, category `autonomous-ai-agents`) has the full
Gradio gotchas, the fold model, and browser-DOM verification selectors.

## Other entry points

- **CLIs:** `ui/cli2.py demo_assistant` = demo profile (fs/shell/git/web);
  bare `ui/cli2.py` = assistant profile (search_conversation_history + read_turn
  only, by design). `ui/cli3.py` is the base for the streaming StepStream/
  TurnStream lives.
- **Benchmarks:** `tests/run_lmem.py` (LongMemEval) — **read `tests/AGENTS.md`
  first**; live LLM, costs real tokens.
- **Tests:** `PYTHONPATH= .venv/bin/python -m pytest -q` (not bare `pytest`).
- **Lint:** `PYTHONPATH= .venv/bin/ruff check <path> --output-format=concise`.
