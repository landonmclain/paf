# M1 — Spec Loader + One-Turn Agent

**Goal:** One YAML spec file, a reusable spec loader, an agent built from it, and one live-model turn that prints a response. Nothing else. No persistence, no compaction, no search, no streaming, no CLI loop.

**Scope boundary:** "spec file first, spec wrapper, build agent, very simple one turn test. that's all." — step 4 is the verification; after that, commit and stop.

---

## Current state

- `specs/assistant.yaml` exists but has a `capabilities:` block with `StepPersistence` (from the abandoned persistence approach). Needs trimming to minimal core fields.
- `src/paf/main.py` currently wires persistence (uncommitted, going to be rewritten).
- `src/paf/config.py` has `PROJECT_ROOT`, `SPECS_DIR`, `ASSISTANT_SPEC` constants (good, keep).
- Uncommitted files: `src/paf/storage/persistence.py`, `data/paf.db`, `.step-persistence.db` — leftovers from the old approach.
- `src/paf/agents/` folder exists but agent.py is not yet created.

---

## Task 1: Trim spec to core fields

**Objective:** `specs/assistant.yaml` declares only `name`, `model`, `instructions`, `model_settings` — no capabilities.

**Files:**
- Modify: `specs/assistant.yaml`

**Action:** Strip the `capabilities:` block entirely. The spec is just:

```yaml
name: assistant
model: openai:gpt-5.4-nano
instructions: |
  You are a helpful assistant. Be concise and direct.
model_settings:
  temperature: 0.2
```

---

## Task 2: Create `src/paf/specs.py` — spec loader

**Objective:** Reusable spec loader: resolve spec path by name, load as `AgentSpec`, build agent with override support.

**Files:**
- Create: `src/paf/specs.py`

**Code:**

```python
"""Spec loader — resolve agent names to YAML specs and build agents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic_ai import Agent, AgentSpec

from paf.config import PROJECT_ROOT

DEFAULT_SPEC_DIRECTORY = PROJECT_ROOT / "specs"


class AgentSpecNotFoundError(FileNotFoundError):
    """Raised when an agent specification cannot be found."""


def get_spec_path(
    agent_name: str,
    spec_directory: Path = DEFAULT_SPEC_DIRECTORY,
) -> Path:
    """Resolve an agent name to a YAML/JSON specification path."""
    for suffix in (".yaml", ".yml", ".json"):
        path = spec_directory / f"{agent_name}{suffix}"
        if path.is_file():
            return path
    raise AgentSpecNotFoundError(
        f"No agent specification found for {agent_name!r} in {spec_directory}"
    )


def load_agent_spec(
    agent_name: str,
    spec_directory: Path = DEFAULT_SPEC_DIRECTORY,
) -> AgentSpec:
    """Load and validate an agent specification from disk."""
    path = get_spec_path(agent_name, spec_directory)
    return AgentSpec.from_file(path)


def build_agent(
    agent_name: str,
    spec_directory: Path = DEFAULT_SPEC_DIRECTORY,
    **overrides: Any,
) -> Agent[Any, Any]:
    """Construct an agent from a named spec, with optional overrides.

    Overrides are passed as kwargs to ``Agent.from_spec``. Supported:
    ``model``, ``instructions``, ``model_settings``, ``name``, ``output_type``.
    Capabilities and tools must be declared in the spec YAML.
    """
    spec = load_agent_spec(agent_name, spec_directory)
    return Agent.from_spec(spec, **overrides)
```

---

## Task 3: Create `src/paf/main_simple.py` — one-turn agent

**Objective:** A separate, minimal entrypoint file (do NOT touch the existing `main.py`) that builds the agent from the spec and runs one prompt, printing the response.

**Files:**
- Create: `src/paf/main_simple.py`

**Code:**

```python
"""minimal one-turn entrypoint — build agent from spec, run once, print."""

from __future__ import annotations

from pydantic_ai import Agent

from paf.observability.instrumentation import setup_observability
from paf.specs import build_agent


def build_assistant() -> Agent:
    """Build the assistant agent from specs/assistant.yaml."""
    return build_agent("assistant")


def main() -> None:
    """Build agent, send one prompt, print response."""
    setup_observability()
    agent = build_assistant()
    print(f"Agent: {agent.name}", flush=True)
    result = agent.run_sync("Say hello in one sentence.")
    print(f"\n{result.output}")


if __name__ == "__main__":
    main()
```

> Note: `main.py` is left untouched — it still wires the (abandoned) persistence
> bits. `main_simple.py` is the new minimal entrypoint for M1.

---

## Task 4: Verify — one-turn live run

**Objective:** Run the agent once with a live model, confirm output prints. The package is installed editable (verified: `import paf` resolves to `src/paf` via `uv run`), so the repo itself needs no PYTHONPATH tweak.

> **Gotcha:** the Hermes desktop exports `PYTHONPATH=/home/alice/.hermes/hermes-agent/venv/lib/python3.11/site-packages` into the shell. That py3.11 venv shadows the project's py3.12 site-packages, so a bare `uv run …` fails with `No module named 'pydantic_core._pydantic_core'`. Prefix the command with `PYTHONPATH=` (inline, no persistent export) to clear it.

**Command:**

```bash
cd /home/alice/projects/Personal/pydantic-agent-framework
PYTHONPATH= uv run --env-file .env python -m paf.main_simple
```

Expected: agent prints its name, model responds to "Say hello in one sentence."
(Verified live: `Agent: assistant` → `Hello!`.)

**Also verify:**
```bash
PYTHONPATH= uv run ruff check src/
PYTHONPATH= uv run ruff format --check src/
```

---

## Cleanup (optional)

Files from the old approach that are no longer needed:

- `src/paf/storage/persistence.py` (uncommitted — let it be overwritten or delete)
- `data/paf.db`, `.step-persistence.db` (uncommitted — can delete)

These are not part of the core scope — purely cleanup.

---

## Commit

```bash
git add -A
git commit -m "m1: spec loader + one-turn agent (minimal)"
```