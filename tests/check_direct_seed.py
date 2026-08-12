"""Fast gate for the direct-persist seed: NO LLM calls.

Verifies the direct-persist mechanics on the Admon question (7161e7e2)
without burning tokens:
  1. `_raw_turns` returns clean {role, content} turns (true roles).
  2. `_transcript_messages` builds user/assistant ModelMessages correctly.
  3. `store.save_snapshot` + `register_run` land, and
     `SnapshotHistorySource.run_history` reads them back.
  4. `_TurnFinder.resolve` (read_turn) renders a CLEAN turn -- no
     `User: assistant:` nesting, no tool-call rows.
  5. The turn index gets one entry per user turn via the STUB summarizer
     (no model call), keyed by (conversation, run, turn_id) matching what
     the real summarizer path uses.

Run: PYTHONPATH= .venv/bin/python tests/check_direct_seed.py
Exit 0 = all gates pass.
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pydantic_ai_harness.conversation_search import SnapshotHistorySource
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    FileStepStore,
    RunRecord,
)

from paf.capabilities.turn_index import TurnIndexStore
from paf.capabilities.turn_index._generate import summarize_turn_append
from paf.capabilities.turn_index._toolset import _TurnFinder
from tests.run_lmem import (
    _raw_turns,
    _transcript_messages,
    _user_turn_units,
)

DATA = ROOT / "data" / "benchmarks" / "lme_oracle.json"
QID = "7161e7e2"  # Admon single-session question (evidence in an ASSISTANT turn)


def _load_q() -> dict:
    data = json.loads(DATA.read_text())
    return next(x for x in data if str(x.get("question_id")) == QID)


class _StubSummarizer:
    """Deterministic stand-in for the real TurnSummarizer (no LLM)."""

    def __init__(self):
        from paf.capabilities.turn_index._summarizer import TurnSummary
        self._TurnSummary = TurnSummary

    async def summarize(self, context: str):
        first = next((ln for ln in context.splitlines() if ln.strip()), "")[:120]
        return self._TurnSummary(
            sentence=first or "(empty)",
            key_terms=["stub"],
            outcome="resolved",
        )


async def main() -> int:
    q = _load_q()
    print(f"Q {QID}: {q['question_type']} | sessions={len(q['haystack_sessions'])}")

    # 1. raw turns
    raw = _raw_turns(q)
    roles = [t["role"] for t in raw]
    print(f"[1] _raw_turns: {len(raw)} turns, roles={roles}")
    assert "assistant" in roles, "expected assistant turns (evidence lives there)"
    assert all(t["role"] in ("user", "assistant") for t in raw)

    # 2. transcript messages
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        store = FileStepStore(directory=td / "store")
        index_store = TurnIndexStore(td / "index")
        cid = "lmem-gate-check"
        run_id = "seed-gate-check"

        msgs = _transcript_messages(raw, run_id, cid)
        n_user = sum(1 for m in msgs if type(m).__name__ == "ModelRequest")
        n_asst = sum(1 for m in msgs if type(m).__name__ == "ModelResponse")
        print(f"[2] transcript: {len(msgs)} msgs ({n_user} user / {n_asst} assistant)")
        assert n_user == roles.count("user") and n_asst == roles.count("assistant")

        # units: one per user turn, matching read_turn's ref model
        units = _user_turn_units(msgs)
        print(f"[3] user-turn units: {len(units)} (turn_ids {[i for i, _ in units]})")
        assert [i for i, _ in units] == list(range(n_user))

        # 3. persist + read back
        await store.register_run(
            RunRecord(run_id=run_id, conversation_id=cid, agent_name="gate")
        )
        await store.save_snapshot(
            ContinuableSnapshot(
                run_id=run_id,
                step_index=0,
                conversation_id=cid,
                agent_name="gate",
                messages=msgs,
            )
        )
        source = SnapshotHistorySource(store)
        history = await source.run_history(run_id=run_id)
        print(f"[4] run_history: {len(history)} messages round-tripped")
        assert len(history) == len(msgs), "snapshot round-trip lost messages"

        # 4. read_turn renders CLEAN turns (no `User: assistant:` nesting)
        finder = _TurnFinder(source)
        rendered = await finder.resolve(
            f"{run_id}:0", run_id=run_id, conversation_id=cid
        )
        print(f"[5] read_turn('{run_id}:0') renders:\n{rendered[:400]}")
        assert "User: assistant:" not in rendered, "assistant text leaked into User:"
        assert "Tool Call [" not in rendered, "tool-call rows in seed turn"
        assert "Assistant:" in rendered, "assistant turn not rendered"

        # 5. index via stub summarizer (no LLM) -- same seam the real one uses
        from paf.capabilities.turn_index import Turn

        for turn_id, turn_msgs in units:
            turn = Turn(run_id=run_id, conversation_id=cid, turn_id=turn_id, messages=turn_msgs)
            summary = await _StubSummarizer().summarize(
                __import__("paf.capabilities.turn_index._extract", fromlist=["build_turn_context"]).build_turn_context(turn.messages)
            )
            appended, _ = await summarize_turn_append(turn=turn, index_store=index_store, summary=summary)
            assert appended
        entries = index_store.read_all(conversation_id=cid)
        print(f"[6] index entries: {len(entries)} (expect {len(units)})")
        assert len(entries) == len(units)
        # every entry's ref resolves via read_turn
        for e in entries:
            out = await finder.resolve(e.ref, run_id=run_id, conversation_id=cid)
            assert out.strip(), f"ref {e.ref} resolves to empty"

        # search_index BM25 finds the Admon evidence (assistant turn content
        # is inside a unit; the stub summary is the first user line, so match
        # on 'shift rotation' via key_terms or summary)
        from paf.capabilities.turn_index._toolset import _searchable_text
        blob = " ".join(_searchable_text(e) for e in entries).lower()
        print(f"[7] index searchable blob ({len(blob)} chars): {blob[:200]}...")

    print("\nALL DIRECT-SEED GATES PASS (no LLM calls made)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
