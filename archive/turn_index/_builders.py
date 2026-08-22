"""TurnIndex builder — owned by the capability, not by persistence.

TurnIndex needs a live ``FileStepStore`` to share with ``StepPersistence`` and
``ConversationSearch`` (so ``read_turn`` resolves exactly the history the write
path produces). That store is created once by ``build_runtime_capabilities`` and
passed in; what this module owns is the *construction* of ``TurnIndexCapability``
— including choosing the summarizer. It reads the capability's own declarative
spec (``specs/turn-index.yaml``) so the runtime honors the same ``summarizer:``
block a ``from_spec`` consumer would, instead of a hardcoded model string buried
in ``persistence.py``.

This keeps the *content* config (summarizer model/budget/retries) with the
capability that uses it, and leaves ``persistence.py`` scoped to pure
write/recall storage.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic_ai_harness.conversation_search import SnapshotHistorySource
from pydantic_ai_harness.step_persistence import FileStepStore

from paf.config import PROJECT_ROOT, settings

from ._capability import TurnIndexCapability
from ._store import TurnIndexStore
from ._summarizer import TurnSummarizer

# Where this capability's declarative spec lives (sibling of this module).
# The `summarizer:` block inside it is the source of truth for the LLM config.
_SPEC_FILE = Path(__file__).resolve().parent / "specs" / "turn-index.yaml"

# The append-only turn index lives beside the step store as its own JSONL
# surface (the agent-foundation StorageService seam). Anchored to the repo root.
_INDEX_DIR = PROJECT_ROOT / Path(settings.turn_index_dir)


def _summarizer_from_spec(spec: dict) -> TurnSummarizer | None:
    """Build the summarizer from the capability's ``specs/turn-index.yaml``.

    The top-level ``capabilities[].TurnIndex.summarizer`` entry is a FLAT dict of
    ``TurnSummarizer`` constructor kwargs (TurnSummarizer is a leaf helper, NOT a
    registered capability). Absent ``summarizer`` -> deterministic stub (the
    capability's no-model default); a present block -> real LLM summarizer.
    """
    for cap in spec.get("capabilities", []):
        if isinstance(cap, dict) and "TurnIndex" in cap:
            summarizer_kwargs = cap["TurnIndex"].get("summarizer")
            if summarizer_kwargs:
                return TurnSummarizer(**summarizer_kwargs)
            return None
    return None


def build_turn_index(store: FileStepStore) -> TurnIndexCapability:
    """Build ``TurnIndexCapability`` over the SHARED ``store``.

    ``store`` is the single ``FileStepStore`` built by ``build_runtime_capabilities``
    and handed to ``StepPersistence`` / ``ConversationSearch`` too, so
    ``read_turn`` resolves exactly the history the writer persists.

    The summarizer is sourced from ``specs/turn-index.yaml`` (real LLM when the
    ``summarizer:`` block is present, deterministic stub otherwise) — the same
    declarative surface a future ``from_spec`` / YAML consumer would use.
    """
    spec = yaml.safe_load(_SPEC_FILE.read_text())
    return TurnIndexCapability(
        source=SnapshotHistorySource(store),
        index_store=TurnIndexStore(_INDEX_DIR),
        store=store,
        summarizer=_summarizer_from_spec(spec),
    )
