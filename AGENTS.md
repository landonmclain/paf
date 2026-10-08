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

> **Full unit test suite:** from the repo root, `PYTHONPATH= .venv/bin/python -m pytest`
> takes ~50s (~769 tests).

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

- **CLI:** `paf chat [agent] [--new | --id CID | --resume N|NAME] [--profile P]`
  (code in `src/paf/cli/`; `ui/chat.py` is the same command from a checkout)
  — inline chat that streams into normal scrollback (default agent
  `demo_assistant`). With no conversation flag an interactive run reopens the
  conversation last open in the CLI (piped runs use lane `chat-<agent>`);
  `/list`, `/resume`, `/name` manage conversations. Runs turns in-process via `Session.start_turn` — it does NOT
  need `paf-sse` running. Ctrl-C cancels a turn; background shell/subagent
  completions stream their follow-up turn at the prompt; `/help` lists slash
  commands. Piped stdin = one-shot (whole stdin is one message, then exit). It
  needs a real tty for interactive mode — to drive it from an agent, pipe the
  prompt in. Also the runner for the **contextbench** profile. The old Rich
  `Live` CLI is archived at `archive/cli-rich/cli.py`.
- **Benchmarks:** `tests/run_lmem.py` (LongMemEval) — **read `tests/AGENTS.md`
  first**; live LLM, costs real tokens. Run as a module from the repo root
  (`PYTHONPATH= .venv/bin/python -m tests.run_lmem`) so `tests.framework`
  imports — bare `python tests/run_lmem.py` fails with `ModuleNotFoundError: tests`.
- **ContextBench / codebase-map eval** — see the dedicated section below.
- **Tests:** `PYTHONPATH= .venv/bin/python -m pytest -q` (not bare `pytest`).
- **Lint:** `PYTHONPATH= .venv/bin/ruff check <path> --output-format=concise`.

## ContextBench / codebase-map eval (manual map-navigation arm)

The benchmark that tests **paf's codebase-map as NAVIGATION over search**. Gold
contexts are exact line-ranges required to produce a passing fix. Current phase:
a **hand-driven live run** on one easy instance (`flask-5014`) via chat.py, the map
vs cold-start-shell A/B. Not yet the standalone `tests/run_lmem.py`-style harness
— that comes later.

**Profile / spec:** the `contextbench` profile (`src/paf/capabilities/profiles.py`)
is the capability allowlist, and `specs/contextbench.yaml` roots it at a snapshot.
Per the experiment, they hold BOTH arms — branch by *instruction*, not separate
profiles/specs. **For the map arm, Shell/FileSystem/Git are COMMENTED OUT**, so
the model MUST navigate purely by the codebase-map ladder.

**How to run the map arm (with approval / user at the keyboard):**

```bash
cd /home/alice/projects/Personal/pydantic-agent-framework
# 1. Run on a fresh conversation lane (--new = no resume of prior turns):
set -a && . ./.env && set +a && PYTHONPATH= .venv/bin/python ui/chat.py contextbench --new
# 2. Send the flask-5014 task as the user message (NOT baked into the spec)
#    — or one-shot: cat task.md | PYTHONPATH= .venv/bin/python ui/chat.py contextbench --new
# 3. Check the trace in Langfuse/ClickHouse (project cmsi71sj00003mp07im3de6ke)
```

The profile is `contextbench`; `resolve_profile_name()` reads the spec's
`profile:` key so a one-arg `chat.py contextbench` picks the RIGHT (Memory-less)
profile — no hardcoded fallback needed. `--new` mints a fresh lane id
(`chat-contextbench-<hex>`, printed at startup) per run/arm — it replaces the old
cli.py `CONV_EPOCH` bump; `--id <cid>` reopens a specific run's lane.

**Why this tests NAVIGATION, and what the ladder expects:**

- The locked read-model is a **sequential ladder**: `get_overview` (orient) →
  `lookup_file` (head → detail) → `read_block(path:L1-L2)` (tight region only).
  `search_codebase` is an **escape hatch AFTER orienting**, never a first move.
  `_CODEMAP_GUIDANCE` (codebase_map/_capability.py) encodes this — keep it that
  way; it's what stopped the model free-firing get_overview+search in parallel.
- **`read_block` returns LINE NUMBERS** (tool-layer, e.g. `190: if "." in name:`)
  for `path:L1-L2` reads, so the model can report the gold line range without the
  raw FileSystem `read_file`. The storage seam stays raw (swappable backend).
- **No Memory in this profile** — verified in traces (no `memory.inject` span).
- Traces: Langfuse self-hosted (`localhost:3000`), or read ClickHouse directly
  (skill: `hermes-local-observability` → `references/clickhouse-trace-query.md`);
  the `paf-framework` project id is `cmsi71sj00003mp07im3de6ke`. Look for a clean
  sequential tool ordering and NO `memory.inject` / memory tools.

See the experiment journal in the vault
(`Hermes/Projects/Personal/Portfolio-GitHub/Benchmark/ContextBench-Codebase-Map-Experiment-Journal.md`)
for the run log, before/after token/cost numbers, and open questions.
