# AGENTS.md — paf operating guide

Concise operating guide for any agent (Claude Code, Codex, Cursor, Hermes, ...)
working in this repo. The web front end is the **SSE server (`ui/sse.py`)** plus
the separate paf-ui Next.js app; the old Gradio web UI (`ui/web.py`,
`paf-web.service`) is **retired and archived — do not use or restart it**. The
graded benchmark harness has its own guide — **see
[`tests/AGENTS.md`](tests/AGENTS.md)** before running or editing anything under
`tests/`. The repo README (`README.md`) is the broader orientation.

> **The single most important rule in this repo:** any `python`/`pytest`
> invocation MUST be prefixed with `PYTHONPATH=` (empty) or it runs Hermes'
> leaked **Python 3.11** site-packages against this repo's **Python 3.12** venv
> and dies with a `pydantic_core` ABI error. See `tests/AGENTS.md` for the full
> mechanical explanation. `.env` must be sourced with `set -a && . ./.env && set +a`
> (plain `source` does not export `OPENROUTER_API_KEY`).

> **Full unit test suite:** from the repo root, `PYTHONPATH= .venv/bin/python -m pytest`
> takes ~50s.

## Servers are systemd user services — do not run them ad hoc

| Service | What | Port |
|---|---|---|
| `paf-sse` | `ui/sse.py` FastAPI SSE + REST transport over `Session` | 8090 |
| `paf-ops` | `ui/ops.py` ops transport (systemd control for paf-sse) | 8091 |
| `paf-ui` | paf-ui Next.js production server (rebuilds on start; behind `tailscale serve` HTTPS) | 3001 |

A chat-spawned server collides on the port and leaves orphans. **After changing
Python code under `src/` or `ui/`, restart `paf-sse`** to pick it up:

```bash
systemctl --user restart paf-sse && sleep 4 && systemctl --user is-active paf-sse
journalctl --user -u paf-sse -f                                  # live logs
systemctl --user daemon-reload                                   # after editing a unit
```

- Units: `~/.config/systemd/user/paf-{sse,ops,ui}.service` (`Restart=on-failure`).
- Wrapper: `scripts/paf-sse-start.sh [port]` — sources `.env`, `exec`s uvicorn
  `ui.sse:app` with a bounded graceful shutdown. Doubles as the manual run path.
- Errors on a chat turn (e.g. a capability failing at run time) surface from
  paf-sse — check its journal first.

## Lint / compile

```bash
PYTHONPATH= .venv/bin/ruff check <path> --output-format=concise   # CLEAN before/after edits
```

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
