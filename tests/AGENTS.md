# AGENTS.md — tests

Guide for agents working in `tests/`. The repo-level guide is [`../AGENTS.md`](../AGENTS.md).

## Unit suite

```bash
.venv/bin/python -m pytest -q -n 8     # ~55s, no network, no tokens
```

- `-n 8` runs it in parallel (pytest-xdist). Tests must not depend on machine-global
  state (all containers, `/tmp` paths, ports); scope such checks to the test's own
  resources, or they flake under xdist.
- There is no pytest-asyncio. Async code runs through `asyncio.run(...)` inside a
  plain `def test_...`; a few files wrap that with a local `_sync` decorator.
- Files whose name starts with `_` (`_lmem_consolidate.py`, `_lmem_summarize.py`,
  `_judge_probe.py`) are runner helpers, not tests.
- `live` and `eval` tests are deselected by default (`addopts` in `pyproject.toml`).
  They call real models or services and need `.env`. Run them by marker:
  `pytest -m live tests/test_web.py`.
- A green run is not always a full run. Some suites skip themselves when their
  dependency is missing: `tests/test_sandbox_executor.py` and
  `tests/test_sandbox_security.py` need a Docker daemon, the Docker tests in
  `tests/test_python_tool.py` carry a `needs_docker` skipif, and
  `tests/test_build_agent.py` requires `OPENAI_API_KEY`. Check the skip count, not
  just the pass count.
- Adding a profile to `PROFILES` (`src/paf/capabilities/profiles.py`) means updating
  `test_profiles_are_the_complete_intentional_set` in `tests/test_capabilities.py`.
- `tests/fixtures/sample_repo/` is the committed codebase the codebase-map tests analyze.

## Lint, format, typecheck

```bash
.venv/bin/ruff check tests/ --output-format=concise
.venv/bin/ruff format --check tests/
npx --yes pyright
```

CI (`.github/workflows/ci.yml`) runs these over `src tests ui`, then the offline
suite and `pytest -m "eval and not live"`. `.githooks/pre-push` runs the same three
checks locally before a push; it only fires if hooks are enabled:

```bash
git config core.hooksPath .githooks
```

## Defender eval (free)

Calibrates the prompt-injection guard against labeled tool outputs. Local classifier,
deterministic, ~1s. See [`defender_eval/README.md`](defender_eval/README.md).

```bash
.venv/bin/python -m tests.defender_eval.eval
.venv/bin/python -m pytest tests/test_defender_eval.py     # the regression gate
```

## LongMemEval benchmark (live, costs tokens)

`tests/run_lmem.py` is the graded memory benchmark. It uses real models (subject and
judge via OpenRouter) and writes Langfuse traces, so it costs money. Don't run it
unless asked. The judge is `openrouter:openai/gpt-4o-mini` by default; override with
`LME_JUDGE_MODEL`.

- Scoring: the verbatim LongMemEval judge prompts (`tests/lme_scoring.py`) for answer
  correctness, plus recall@k over the surfaced `session=` tags.
- Shared agent assembly: `tests/framework.py` (`_build_agent`, `_run_turn`). Edit it
  there; runners must not keep a parallel builder. It builds the same scoped memory
  render as the live agent, with its knobs read from `specs/benchmarks/longmemeval.yaml`.
- **Data is not in the repo.** The harness reads `data/benchmarks/memory-v3-lme_oracle.json`
  (`LME_DATA_FILE` in `run_lmem.py`), which is untracked. Fetch it with
  `uv run scripts/download_benchmarks.py` — that script pulls the official LongMemEval
  oracle split (github.com/xiaowu0162/LongMemEval), pins it to a SHA-256, and validates
  the schema before writing it. The v3 name is deliberate: it forces a distinct
  durable-memory directory so v3 runs never clobber the older v2 snapshots.
- Data: `data/benchmarks/` (the question file, per-question seeds and durable memory
  snapshots). Graded records go to `tests/results/`. Both directories are gitignored.

```bash
set -a && . ./.env && set +a      # plain `source` does not export; run_lmem does not load .env
LME_MAX_QUESTIONS=4 .venv/bin/python -u -m tests.run_lmem
LME_QIDS="51a45a95" .venv/bin/python -u -m tests.run_lmem     # one question
```

Run it as a module from the repo root; `python tests/run_lmem.py` fails with
`ModuleNotFoundError: tests`.

Env knobs:

| Variable | Default | Effect |
| --- | --- | --- |
| `LME_MAX_QUESTIONS` | `15` | Questions sampled. |
| `LME_QIDS` | unset | Comma-separated question ids; overrides sampling. |
| `LME_SAMPLER_SEED` | `20260815` | Seed for the question sample. |
| `LME_SUBJECT_MODEL` | `openrouter:deepseek/deepseek-v4-flash-0731` | Answering model. |
| `MEMORY` | `1` | `0` disables the memory arm (a no-memory baseline). |
| `THREAD_SUMMARY` | `1` | `0` drops the thread-summarizer navigation layer from the recall ladder. Requires a `thread_summary_instructions:` block in the spec when on. |
| `NO_READ_TURN` | `0` | `1` exposes search without `read_turn` expansion — a pure-BM25 floor arm. Only meaningful with `MEMORY=0`; requires a `search_only_no_readturn_instructions:` block. |
| `PAF_FRESH_MEMORY` | unset | Rebuild each question's durable memory instead of reusing the snapshot. |
| `LME_NO_REVIEWER` | unset | Skip the pairwise reviewer when building memory. |
| `LME_RESUME_PARTIAL` | `0` | Resume an interrupted run. |
| `RECALL_REQUEST_LIMIT` | `50` | Model requests per recall turn. |
| `RECALL_TOOL_CALLS_LIMIT` | `25` | Tool calls per recall turn; on overrun the turn falls back to a zero-tool answer instead of crashing. `0` disables. |
| `RECALL_CUTOFF_MODEL` | the subject model | Model for that zero-tool fallback agent. |
| `LME_RETRY_MAX_ATTEMPTS` | `6` | Attempts per model call. |
| `LME_RETRY_BASE_BACKOFF` | `30.0` | Seconds between attempts. |
| `LME_RETRY_BACKOFF_EXPONENTIAL` | `0` | `1` grows the backoff, capped at `LME_RETRY_MAX_BACKOFF` (`60.0`). |
| `LME_RETRYABLE_STATUSES` | `408,429,500,502,503,504` | HTTP statuses that trigger a retry. |
| `LME_JUDGE_PROVIDER_ORDER` | `openai` | Judge provider preference order. |
| `LME_JUDGE_PROVIDER_QUANTIZATIONS` | unset | Allowed quantizations for the judge. |

Things that cost time or tokens without telling you:

- **Build once, then recall.** A question with a saved memory snapshot skips the
  memory-building step (`REUSED ... (0 owner tokens)`). Use `PAF_FRESH_MEMORY=1` only
  when memory construction itself is under test.
- **Changing eviction or index settings invalidates the saved seeds** in
  `data/benchmarks/generated/`, which are keyed to the turn structure they were built
  from. The next run re-summarizes live for every affected question. Keep the settings
  fixed, or accept the one-time cost.
- **Runs are not reproducible to the point.** Provider routing can change
  quantization between runs, so compare only runs on the same pinned settings; a
  small delta is noise, not signal.
