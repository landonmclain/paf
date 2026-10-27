# AGENTS.md — tests

Guide for agents working in `tests/`. The repo-level guide is [`../AGENTS.md`](../AGENTS.md).

## Unit suite

```bash
PYTHONPATH= .venv/bin/python -m pytest -q -n 8     # ~50s, no network, no tokens
```

- `-n 8` runs it in parallel (pytest-xdist). Tests must not depend on machine-global
  state (all containers, `/tmp` paths, ports); scope such checks to the test's own
  resources, or they flake under xdist.
- Async tests use `asyncio.run(...)` inside a plain `def test_...`.
- `live` and `eval` tests are deselected by default (`addopts` in `pyproject.toml`).
  They call real models or services and need `.env`. Run them by marker:
  `pytest -m live tests/test_web.py`.
- Always use `PYTHONPATH=` (empty) before `python`/`pytest`: a `PYTHONPATH` inherited
  from another Python install puts the wrong `pydantic_core` build ahead of the venv's.
- Adding a profile to `PROFILES` (`src/paf/capabilities/profiles.py`) means updating
  `test_profiles_are_the_complete_intentional_set` in `tests/test_capabilities.py`.
- `tests/fixtures/sample_repo/` is the committed codebase the codebase-map tests analyze.

## Lint

```bash
PYTHONPATH= .venv/bin/ruff check tests/ --output-format=concise
```

## Defender eval (free)

Calibrates the prompt-injection guard against labeled tool outputs. Local classifier,
deterministic, ~1s. See [`defender_eval/README.md`](defender_eval/README.md).

```bash
PYTHONPATH= .venv/bin/python -m tests.defender_eval.eval
```

## LongMemEval benchmark (live, costs tokens)

`tests/run_lmem.py` is the graded memory benchmark. It uses real models (subject and
judge via OpenRouter) and writes Langfuse traces, so it costs money. Don't run it
unless asked.

- Scoring: the verbatim LongMemEval judge prompts (`tests/lme_scoring.py`) for answer
  correctness, plus recall@k over the surfaced `session=` tags.
- Shared agent assembly: `tests/framework.py` (`_build_agent`, `_run_turn`). Edit it
  there; runners must not keep a parallel builder. It builds the same scoped memory
  render as the live agent, with its knobs read from `specs/longmemeval.yaml`.
- Data: `data/benchmarks/` (the question file, per-question seeds and durable memory
  snapshots). Graded records go to `tests/results/`.

```bash
set -a && . ./.env && set +a      # plain `source` does not export; run_lmem does not load .env
LME_MAX_QUESTIONS=4 PYTHONPATH= .venv/bin/python -u -m tests.run_lmem
LME_QIDS="51a45a95" PYTHONPATH= .venv/bin/python -u -m tests.run_lmem     # one question
```

Run it as a module from the repo root; `python tests/run_lmem.py` fails with
`ModuleNotFoundError: tests`.

Env knobs:

| Variable | Default | Effect |
| --- | --- | --- |
| `LME_MAX_QUESTIONS` | `15` | Questions sampled. |
| `LME_QIDS` | unset | Comma-separated question ids; overrides sampling. |
| `BEAM_LITE_SUBJECT_MODEL` | `openrouter:deepseek/deepseek-v4-flash-0731` | Answering model. |
| `MEMORY` | `1` | `0` disables the memory arm (a no-memory baseline). |
| `PAF_FRESH_MEMORY` | unset | Rebuild each question's durable memory instead of reusing the snapshot. |
| `LME_NO_REVIEWER` | unset | Skip the pairwise reviewer when building memory. |
| `LME_RESUME_PARTIAL` | `0` | Resume an interrupted run. |
| `RECALL_REQUEST_LIMIT` | `50` | Model requests per recall turn. |
| `RECALL_TOOL_CALLS_LIMIT` | `25` | Tool calls per recall turn; on overrun the turn falls back to a zero-tool answer instead of crashing. `0` disables. |

Things that cost time or tokens without telling you:

- **Build once, then recall.** A question with a saved memory snapshot skips the
  memory-building step (`REUSED ... (0 owner tokens)`). Use `PAF_FRESH_MEMORY=1` only
  when memory construction itself is under test.
- **Changing eviction or index settings invalidates the saved seeds** in
  `data/benchmarks/generated/`, which are keyed to the turn structure they were built
  from. The next run re-summarizes live for every affected question. Keep the settings
  fixed, or accept the one-time cost.
- **Same-config runs vary by a few points** (about ±2pt on a full run). Routing between
  OpenRouter providers can change quantization; the traces don't record which provider
  served a call. Compare runs on the same pinned settings, and don't read a small
  difference as signal.
- When a log line and the trace disagree, the trace is right. Read the run's
  observations in Langfuse for generation counts and tool calls.
