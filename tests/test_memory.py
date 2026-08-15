"""Tests for the paf-owned `Memory` capability (near-copy of harness memory).

This package re-homes `pydantic_ai_harness.memory` so the durable-facts notebook
sits under paf's single marker-strip convention. These tests pin two things:

1. **Marker-strip regression (the search-side guarantee).** A persisted User
   message may carry a marker-tagged `<memory>` block (what a harness-`Memory`
   injection writes next to real user text, and what StepPersistence would
   snapshot). The paf search corpus (`search/_render`) must drop that injected
   block — search should match the REAL durable text, never text that only
   exists in the injected memory block. This is the memory analogue of the
   `<turn-index>` strip test in `test_search.py`.

2. **Backend parity.** `FileStore` and `InMemoryStore` both satisfy the
   `MemoryStore` protocol: versioned read/write/delete with CAS, bounded list,
   and lexical search. And the `MemoryToolset` (write/read/search/delete) works
   end-to-end against both backends.

Kept memory (as decided): only `FileStore` + `InMemoryStore`; SQLite/Postgres
stores are dropped. The injected block keeps the harness marker
(`pydantic-ai-harness.memory.v1:<scope_hash>`), so the existing strip sites
already recognize it — no strip churn.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic_ai.messages import ModelRequest, TextContent, UserPromptPart
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters

from paf.capabilities.memory import (
    MAIN_FILENAME,
    FileStore,
    InMemoryStore,
    Memory,
    MemoryOperation,
    MemoryToolset,
    MemoryWriteResult,
)
from paf.capabilities.search._render import format_message

# --- null tracer/span so the toolset's tracing calls work without OTEL ---


class _NullSpan:
    def is_recording(self) -> bool:
        return False

    def set_attributes(self, attributes: dict) -> None:
        pass

    def __enter__(self) -> _NullSpan:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class _NullTracer:
    def start_as_current_span(self, name: str, **kwargs: object) -> _NullSpan:
        return _NullSpan()


class _FakeCtx:
    def __init__(self, run_id: str, tool_call_id: str) -> None:
        self.run_id = run_id
        self.tool_call_id = tool_call_id
        self.tracer = _NullTracer()  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Marker-strip -- a harness-<memory> injection never leaks into the corpus
# ---------------------------------------------------------------------------


def test_memory_injection_never_leaks_into_search_corpus() -> None:
    """A User message carrying a marked `<memory>` block searches only its real text.

    The near-copy keeps the harness marker base, so the paf corpus strip (which
    already recognizes `pydantic-ai-harness.memory.v1`) drops the injected block.
    Text that exists ONLY in the injection must not be findable.
    """
    # Real user text + a harness-Memory injection tagged TextContent block.
    req = ModelRequest(
        parts=[
            UserPromptPart(
                content=[
                    "what's the deploy story for the scheduler?",
                    TextContent(
                        "<memory>\n- the scheduler was moved to its own box\n</memory>",
                        metadata="pydantic-ai-harness.memory.v1:deadbeefcafe",
                    ),
                ]
            )
        ]
    )

    rendered = format_message(req, truncate=False)
    # Real durable text is present.
    assert "deploy story for the scheduler" in rendered
    # The injected memory block is gone.
    assert "scheduler was moved to its own box" not in rendered
    assert "<memory>" not in rendered


# ---------------------------------------------------------------------------
# Empty-scope injection -- the memory surface is present even before any memory
# ---------------------------------------------------------------------------


async def _run_before_model_request(cap: Memory) -> ModelRequest:
    """Call `before_model_request` on a fresh user turn, return the mutated request."""
    ctx = _FakeCtx(run_id="run-empty", tool_call_id="unused")
    req = ModelRequest(parts=[UserPromptPart("first turn")])
    mrc = ModelRequestContext(
        model=object(),  # type: ignore[arg-type] - only threaded through, never used
        messages=[req],
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )
    out = await cap.before_model_request(ctx, mrc)  # type: ignore[arg-type] - test stub ctx is duck-typed
    return out.messages[-1]  # type: ignore[return-value] - always the user ModelRequest we passed


def _injected_text(req: ModelRequest) -> str:
    """Concatenate the TextContent parts (the injected memory block) of a request."""
    return "\n".join(
        c.content
        for p in req.parts
        if hasattr(p, "content") and isinstance(p.content, list)
        for c in p.content
        if isinstance(c, TextContent)
    )


def test_empty_scope_injects_placeholder_memory_block() -> None:
    """A brand-new scope still shows the `<memory>` surface with an empty note.

    Guards the cold-start deadlock: with no `MEMORY.md` and no topic files the
    old code injected nothing, so the model never saw the durable-notebook
    surface and had no incentive to make the first `write_memory` call. The
    placeholder keeps the block present from turn one.
    """

    async def _t() -> None:
        cap = Memory(store=InMemoryStore(), agent_name="locomo-subject")
        req = await _run_before_model_request(cap)
        text = _injected_text(req)
        assert "<memory>" in text
        assert "No memories recorded/written yet." in text

    asyncio.run(_t())


def test_populated_scope_renders_normally_not_placeholder() -> None:
    """Once memory exists, the placeholder is replaced by the real notebook."""

    async def _t() -> None:
        store = InMemoryStore()
        await store.write(
            "locomo-subject/MEMORY.md",
            "- the blazer is ready for pickup\n",
            expected_version=None,
            operation=MemoryOperation(id="op-seed", fingerprint="f"),
        )
        cap = Memory(store=store, agent_name="locomo-subject")
        req = await _run_before_model_request(cap)
        text = _injected_text(req)
        assert "the blazer is ready for pickup" in text
        assert "No memories recorded/written yet." not in text

    asyncio.run(_t())


# ---------------------------------------------------------------------------
# Backend parity: FileStore and InMemoryStore against the MemoryStore protocol
# ---------------------------------------------------------------------------


async def _roundtrip(store) -> None:
    """Exercise write/read/search/delete with CAS on one store."""
    op1 = MemoryOperation(id="op-1", fingerprint="f1")
    # write creates
    m1 = await store.write("scope/mem.md", "first fact\n", expected_version=None, operation=op1)
    assert m1.existed is False and m1.version is not None
    f = await store.read("scope/mem.md", max_chars=10_000)
    assert f is not None and f.content == "first fact\n"
    assert f.version == m1.version

    # CAS: stale expected_version must conflict
    try:
        await store.write(
            "scope/mem.md",
            "second\n",
            expected_version="stale",
            operation=MemoryOperation(id="op-2", fingerprint="f2"),
        )
        raise AssertionError("expected MemoryConflictError")
    except Exception as exc:  # noqa: BLE001 - want the store's conflict type
        assert type(exc).__name__ == "MemoryConflictError", exc

    # idempotent replay of the same operation id returns the stored mutation
    m1_replayed = await store.write("scope/mem.md", "first fact\n", expected_version=None, operation=op1)
    assert m1_replayed.replayed is True

    # list + search
    await store.write(
        "scope/postgres.md",
        "the scheduler uses Postgres\n",
        expected_version=None,
        operation=MemoryOperation(id="op-3", fingerprint="f3"),
    )
    paths = await store.list_paths("scope/", limit=10)
    assert "scope/mem.md" in paths and "scope/postgres.md" in paths
    res = await store.search("scope/", "scheduler", limit=5, max_files=5, max_chars=10_000, max_file_chars=10_000)
    assert res.scanned >= 1
    assert any("scheduler" in m.snippet for m in res.matches)

    # delete
    cur = await store.read("scope/mem.md", max_chars=10_000)
    d = await store.delete(
        "scope/mem.md",
        expected_version=cur.version if cur else None,
        operation=MemoryOperation(id="op-4", fingerprint="f4"),
    )
    assert d.existed is True
    assert await store.read("scope/mem.md", max_chars=10_000) is None


def test_innemstore_roundtrip() -> None:
    asyncio.run(_roundtrip(InMemoryStore()))


def test_filestore_roundtrip(tmp_path: Path) -> None:
    asyncio.run(_roundtrip(FileStore(tmp_path)))
    # FileStore wrote real .md files under the root.
    assert (tmp_path / "scope" / "postgres.md").is_file()


# ---------------------------------------------------------------------------
# MemoryToolset end-to-end against both backends
# ---------------------------------------------------------------------------


async def _toolset_roundtrip(capability: Memory) -> None:
    ts = MemoryToolset(capability)

    def ctx_for(i: int) -> _FakeCtx:
        # Each distinct model tool call gets its own tool_call_id; the operation
        # id/idempotency key derives from (scope, run_id, tool_call_id).
        return _FakeCtx(run_id="run-123", tool_call_id=f"call-{i}")

    # write (create) then append -- distinct tool calls, distinct operation ids
    r: MemoryWriteResult = await ts.write_memory(ctx_for(1), content="- first durable fact\n")
    assert r["status"] == "created" and r["file"] == MAIN_FILENAME
    r2: MemoryWriteResult = await ts.write_memory(ctx_for(2), content="- second durable fact\n")
    assert r2["status"] == "appended"

    # read back
    out = await ts.read_memory(ctx_for(3), file=MAIN_FILENAME)
    assert "- first durable fact" in out and "- second durable fact" in out

    # search finds it
    sr = await ts.search_memory(ctx_for(4), query="durable fact")
    assert len(sr["matches"]) >= 1

    # delete a topic file (MEMORY.md cannot be deleted)
    await ts.write_memory(ctx_for(5), content="topic content\n", file="topic.md")
    del_r = await ts.delete_memory(ctx_for(6), file="topic.md")
    assert del_r["status"] == "deleted"
    try:
        await ts.delete_memory(ctx_for(7), file=MAIN_FILENAME)
        raise AssertionError("expected ModelRetry on deleting MEMORY.md")
    except Exception as exc:  # noqa: BLE001
        assert type(exc).__name__ == "ModelRetry", exc


def test_toolset_roundtrip_innem() -> None:
    cap = Memory(store=InMemoryStore(), agent_name="test-agent")
    asyncio.run(_toolset_roundtrip(cap))


async def _empty_scope_search(capability: Memory) -> None:
    """A fresh scope with no memory files returns an actionable string, not a bare empty match list.

    A bare `{'matches': [], 'scanned': 0}` sent earlier memory-owner runs into a
    search-the-emptiness loop (search_memory was called repeatedly with nothing
    to find). Returning a plain string tells the model the notebook is empty and
    to create memory instead of searching.
    """
    ts = MemoryToolset(capability)
    out = await ts.search_memory(_FakeCtx(run_id="run-0", tool_call_id="call-0"), query="anything")
    assert isinstance(out, str), f"expected a string on empty scope, got {type(out)!r}: {out!r}"
    assert "no memory files" in out.lower()
    assert MAIN_FILENAME in out


def test_search_empty_scope_innem() -> None:
    asyncio.run(_empty_scope_search(Memory(store=InMemoryStore(), agent_name="test-agent")))


def test_search_empty_scope_file(tmp_path: Path) -> None:
    asyncio.run(_empty_scope_search(Memory(store=FileStore(tmp_path), agent_name="test-agent")))


def test_toolset_roundtrip_file(tmp_path: Path) -> None:
    cap = Memory(store=FileStore(tmp_path), agent_name="test-agent")
    asyncio.run(_toolset_roundtrip(cap))


# ---------------------------------------------------------------------------
# MarkdownStorage: the storage-seam backend (real .md bodies + flat sidecar)
# ---------------------------------------------------------------------------


def test_markdown_roundtrip(tmp_path: Path) -> None:
    """MarkdownStorage matches the MemoryStore protocol end-to-end.

    Flat-file backend: bodies are plain inspectable `.md` files with a flat
    ``.memory-state.json`` CAS sidecar per scope — no hidden journal, no seam
    fabrication. (Memory is authoritative/document shape; the ``StorageService``
    protocol models the derived/index shape and is deliberately not forced on it.)
    """
    from paf.capabilities.memory._markdown import MarkdownStorage

    store = MarkdownStorage(tmp_path)
    asyncio.run(_roundtrip(store))

    # Real readable .md bodies (flat-file principle), not JSON blobs.
    assert (tmp_path / "scope" / "postgres.md").is_file()
    assert "the scheduler uses Postgres" in (tmp_path / "scope" / "postgres.md").read_text()
    # Flat, diffable version sidecar — no sqlite journal.
    state = tmp_path / "scope" / ".memory-state.json"
    assert state.is_file()
    assert "postgres.md" in state.read_text()


def test_markdown_from_spec(tmp_path: Path) -> None:
    """from_spec(backend='markdown') dispatches to MarkdownStorage, not FileStore."""
    from paf.capabilities.memory._markdown import MarkdownStorage

    cap = Memory.from_spec(backend="markdown", directory=str(tmp_path), agent_name="test-agent")
    assert isinstance(cap.store, MarkdownStorage)
    assert cap.store._root == tmp_path  # type: ignore[attr-defined]
    # The old `file` backend is untouched.
    from paf.capabilities.memory._store import FileStore

    cap_file = Memory.from_spec(backend="file", directory=str(tmp_path), agent_name="test-agent")
    assert isinstance(cap_file.store, FileStore)
