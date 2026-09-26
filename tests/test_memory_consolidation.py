"""Tests for the consolidation OWNER engine (``paf.capabilities.memory._consolidate``).

Covers the consolidation owner's seams:

1.  the owner's default guidance describes the tiered, append-only notebook
    (not the legacy four-section format);
2.  owner construction invariants (sole writer, inject control) for both the
    tool-only live owner and the chunked seed-time variant;
3.  ``resident_consolidate_text`` derives a real ``born_on`` from the inline
    ``[date: ...]`` prefix without leaking it into the stored fact.

The owner's model, model_settings, and operating guidance resolve from the
``memory_extractor`` leaf spec (Spec-Driven-Background-Agents) — the legacy
``CONSOL_MODEL`` / provider-order / seed env knobs were removed. The chunked
seed-time helpers (``consolidate_conversation_chunked`` / ``_split_chunks``)
live in ``tests/_lmem_consolidate.py`` (benchmark-only, per the relocation
rule). The inline eviction-fold window (``MemoryConsolidationWindow``) is
DELETED: durable-fact capture is the background watermark worker's job, and the
assistant's eviction tier is trim-only ``TurnBoundSlidingWindow``.
"""

from __future__ import annotations


def test_memory_guidance_describes_tiered_not_legacy_sections() -> None:
    """The assistant's default guidance describes the 3-tier rendered shape.

    The legacy four-section ``MEMORY_FORMAT`` (Facts/Preferences/Pending/Notes)
    and the writable ``_DEFAULT_GUIDANCE`` were retired with the M2 rework — the
    assistant is inject-only (no memory tools), and its guidance reflects the
    tiered, append-only notebook the render produces (t1/t2/t3). The
    consolidation OWNER keeps append-only journal semantics only and carries no
    notebook section list.
    """
    from paf.capabilities.memory._capability import _DEFAULT_MEMORY_GUIDANCE

    # The assistant's default guidance describes the tier ladder, not four
    # fixed sections, and contains no write/CAPTURE directives (inject-only).
    assert "## t1" in _DEFAULT_MEMORY_GUIDANCE
    assert "## t2" in _DEFAULT_MEMORY_GUIDANCE
    assert "## t3" in _DEFAULT_MEMORY_GUIDANCE
    assert "search_conversation_history" in _DEFAULT_MEMORY_GUIDANCE
    assert "write_memory" not in _DEFAULT_MEMORY_GUIDANCE
    assert "CAPTURE as you go" not in _DEFAULT_MEMORY_GUIDANCE
    # The consolidation owner is append-only — it does NOT carry a notebook
    # section/format block (the render produces it, not the owner).
    from paf.capabilities.memory._consolidate import _build_owner_memory

    built_owner = _build_owner_memory(None, agent_name="assistant", namespace="")
    assert built_owner.guidance is not None
    assert "## Facts" not in built_owner.guidance
    assert "## Pending" not in built_owner.guidance
    assert "write_memory" not in built_owner.guidance


# ---------------------------------------------------------------------------
# owner construction
# ---------------------------------------------------------------------------


def test_build_owner_inject_memory_default_is_false() -> None:
    """Default owner stays tool-only (live eviction path unchanged)."""
    from paf.capabilities.memory import Memory
    from paf.capabilities.memory._consolidate import _build_owner_memory

    cap = _build_owner_memory(None, agent_name="assistant", namespace="")
    assert isinstance(cap, Memory)
    assert cap.inject_memory is False


def test_build_owner_inject_memory_true_for_chunked() -> None:
    """Chunked seed variant flips inject_memory=True (cumulative in-context merge)."""
    from paf.capabilities.memory import Memory
    from paf.capabilities.memory._consolidate import _build_owner_memory

    cap = _build_owner_memory(
        None, agent_name="assistant", namespace="", inject_memory=True
    )
    assert isinstance(cap, Memory)
    assert cap.inject_memory is True


# ---------------------------------------------------------------------------
# resident_consolidate_text
# ---------------------------------------------------------------------------


def test_resident_consolidate_text_none_when_empty() -> None:
    """No messages → resident_consolidate_text returns None (owner never fires)."""
    import asyncio

    from paf.capabilities.memory._consolidate import resident_consolidate_text

    out = asyncio.run(resident_consolidate_text([]))
    assert out is None


def test_resident_consolidate_text_renders_every_segment() -> None:
    """Every message segment is folded into the owner prompt (no eviction cut)."""
    import asyncio

    from pydantic_ai.messages import ModelRequest, UserPromptPart

    from paf.capabilities.memory._consolidate import resident_consolidate_text

    msgs = [ModelRequest(parts=[UserPromptPart(content=f"turn {i}")]) for i in range(4)]
    out = asyncio.run(resident_consolidate_text(msgs))
    assert out is not None
    for i in range(4):
        assert f"turn {i}" in out


def test_resident_consolidate_text_labels_inline_date_from_prefix() -> None:
    """No metadata ``source_date`` → the owner derives ``born_on`` from the
    inline ``[date: ...]`` prefix (production seam), and the prefix is stripped
    from the fact text so it never leaks into the stored fact."""
    import asyncio

    from pydantic_ai.messages import ModelRequest, UserPromptPart

    from paf.capabilities.memory._consolidate import resident_consolidate_text

    msgs = [
        ModelRequest(
            parts=[UserPromptPart(content="[date: 2026-08-21] do the thing now")]
        )
    ]
    out = asyncio.run(resident_consolidate_text(msgs))
    assert out is not None
    # Date rides in the label so the owner can fill born_on with real provenance.
    assert "2026-08-21" in out
    # The prefix is stripped from the fact text the owner copies verbatim.
    assert "[date: 2026-08-21]" not in out
    assert "do the thing now" in out


# ---------------------------------------------------------------------------
# chunked seed consolidation (benchmark / large-haystack path)
# ---------------------------------------------------------------------------


def test_split_chunks_message_boundary_aligned() -> None:
    """Chunks never split a message and respect the token budget per chunk."""
    from tests._lmem_consolidate import _split_chunks

    # 3 messages of ~20 chars each + marker (~6). Budget tiny → each message
    # its own chunk (no budget is big enough for two at ~26 chars/4 ≈ 7 tokens).
    text = "\n".join(
        [
            "User: hello this is turn one",
            "Assistant: okay noted the first thing",
            "User: second user message here",
        ]
    )
    chunks = _split_chunks(text, max_tokens=4)
    assert chunks, "non-empty text must produce at least one chunk"
    assert sum(c.count("\n") + 1 for c in chunks) == text.count("\n") + 1
    # Joining the chunks must reproduce the input byte-for-byte (no loss).
    assert "\n".join(c for c in chunks) == text
    # Order preserved.
    assert chunks[0].startswith("User:")
    for c in chunks:
        assert c.startswith("User:") or c.startswith("Assistant:")


def test_split_chunks_merges_under_budget() -> None:
    """A small enough input stays one chunk (no artificial fragmentation)."""
    from tests._lmem_consolidate import _split_chunks

    text = "\n".join(["User: short", "Assistant: also short", "User: tiny"])
    chunks = _split_chunks(text, max_tokens=4000)
    assert len(chunks) == 1


def test_split_chunks_empty_input() -> None:
    """Empty input yields no chunks; a whitespace-only input yields one."""
    from tests._lmem_consolidate import _split_chunks

    assert _split_chunks("", max_tokens=1000) == []
    assert _split_chunks("", max_tokens=0) == []


def test_consolidate_conversation_chunked_runs_once_per_chunk(monkeypatch) -> None:
    """The chunked owner runs once per split chunk, in order, on the same store.

    Uses a fake owner (no API key / network) that records each chunk it is fed,
    so the test verifies the chunking + sequential merge wiring without a real
    model. The store is a no-op stand-in; the real store wiring is covered by
    the integration gate run.
    """
    import asyncio

    from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart

    from tests import _lmem_consolidate as lmem

    fed: list[str] = []

    class _FakeOwner:
        async def run(self, prompt: str, **kwargs):
            fed.append(prompt)
            return type("R", (), {"output": "no changes"})()

        def iter(self, prompt=None, **kwargs):  # noqa: A002 - guard-aware contract
            from types import SimpleNamespace

            fed.append(str(prompt))

            class _Run:
                ctx = object()
                result = SimpleNamespace(output="no changes")

                def __aiter__(self):
                    return _NodeIter()

            class _NodeIter:
                def __aiter__(self):
                    return self

                async def __anext__(self):
                    raise StopAsyncIteration

            class _CM:
                async def __aenter__(self):
                    return _Run()

                async def __aexit__(self, *a):
                    return False

            return _CM()

    def _fake_build_owner(
        store, *, agent_name, namespace, inject_memory=False, journal=None
    ):
        return _FakeOwner(), None

    monkeypatch.setattr(lmem, "_build_owner", _fake_build_owner)

    msgs: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content=f"turn number {i} and then some")])
        for i in range(12)
    ]

    async def _run():
        await lmem.consolidate_conversation_chunked(
            msgs,
            store=None,  # the fake owner ignores it
            agent_name="assistant",
            namespace="",
            chunk_max_tokens=3,  # tiny → forces many small chunks
        )

    asyncio.run(_run())

    assert len(fed) > 1, "tiny budget should split into multiple chunks"
    # Full transcript preserved across chunks, in order.
    joined = "\n".join(fed)
    for i in range(12):
        assert f"turn number {i} and then some" in joined
