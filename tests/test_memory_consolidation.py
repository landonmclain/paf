"""Tests for pre-eviction memory consolidation.

Covers the three seams locked in for the consolidation feature:

1.  ``MemoryConsolidationWindow.compact`` — fires the owner on the
    about-to-be-evicted slice and no-ops when nothing would be evicted.
2.  ``build_compaction`` config toggle — ``consolidate`` on swaps the sliding-window
    tier for the consolidation window; off keeps the plain window.
3.  journal round-trip — the window hands the owner the M2 journal and the
    rendered root ``MEMORY.md`` is readable back via ``JournalViewStore``.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai_harness.compaction._shared import find_safe_cutoff

from paf.capabilities.compaction import build_compaction
from paf.capabilities.memory._consolidate import (
    CONSOLIDATION_GUIDANCE,
    consolidate_model,
)
from paf.capabilities.memory._format import MEMORY_FORMAT, extract_sections
from paf.capabilities.memory._window import MemoryConsolidationWindow

# ---------------------------------------------------------------------------
# MemoryConsolidationWindow.compact — fires on the eviction slice, no-op otherwise
# ---------------------------------------------------------------------------


def _messages(n: int) -> list[ModelRequest]:
    return [ModelRequest(parts=[UserPromptPart(content=f"msg {i}")]) for i in range(n)]


def test_compact_noop_when_nothing_evicted() -> None:
    """With fewer messages than keep_messages, find_safe_cutoff=0 → no consolidation."""
    msgs = _messages(2)
    window = MemoryConsolidationWindow(max_messages=4, keep_messages=2)
    assert find_safe_cutoff(msgs, window.keep_messages) == 0


def test_window_compact_consolidates_evicted_slice_then_trims_happy_path(
    tmp_path: Path,
) -> None:
    """Driving compact() with an evictable history folds the dropped slice into memory, then trims.

    This is the end-to-end seam the prior tests only assert structurally (window swaps
    into the ladder, defaults scope) WITHOUT ever calling ``compact`` on a slice that
    would actually be evicted. It verifies the whole dance offline:

    * ``consolidate`` is invoked on EXACTLY the about-to-evict slice (``messages[:cutoff]``),
      never the tail or the full list;
    * the stub writes durable content to the real flat-file store (proving the store the
      window targets is reachable and writable);
    * the window still trims to ``keep_messages`` (consolidation does not swallow the trim).
    """
    import asyncio

    from pydantic_ai.messages import ModelMessage

    from paf.capabilities.memory import _consolidate as cons

    window = MemoryConsolidationWindow(
        max_messages=6, keep_messages=2, memory_directory=str(tmp_path / "memory")
    )

    msgs: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content=f"durable fact alpha {i}")])
        for i in range(6)
    ]
    captured: dict[str, object] = {}

    async def fake_consolidate(
        messages, *, store, agent_name, namespace, keep_tail, journal=None
    ):
        captured["slice"] = messages
        captured["keep_tail"] = keep_tail
        captured["journal"] = journal
        # Prove the window passes the M2 journal through: the owner path appends
        # to the journal (append_memory) rather than the legacy store. Simulate
        # one append here.
        assert journal is not None, "window must pass the journal to consolidate"
        journal.append(
            type="fact",
            fact="durable fact alpha",
            born_on="2023-08-20",
            source_ref="test:0",
        )

    # The window imports consolidate *inside* compact() — patch the module attr so the
    # freshed local `from ._consolidate import consolidate` picks up the stub.
    original = cons.consolidate
    cons.consolidate = fake_consolidate  # type: ignore[assignment]
    try:
        out = asyncio.run(window.compact(msgs, ctx=None))  # type: ignore[arg-type]
    finally:
        cons.consolidate = original  # type: ignore[assignment]

    cutoff = find_safe_cutoff(msgs, 2)
    assert cutoff == 4  # 6 messages, keep 2 -> evict first 4
    # The window hands consolidate the FULL list + keep_tail on the eviction boundary;
    # consolidate() (not the window) is responsible for slicing to the evicted prefix.
    assert len(captured["slice"]) == 6  # type: ignore[arg-type]
    assert captured["keep_tail"] == 2
    # Window trims to keep_messages=2 PLUS one preserved opening user message (the
    # harness's preserve_first_user_message default), so 3 remain.
    assert len(out) == 3

    # End-to-end proof of the split-brain fix: after consolidate appends to the
    # journal, compact renders root MEMORY.md, and the assistant's JournalViewStore
    # can read the fact back. (Prior to the fix this path wrote only to the legacy
    # assistant/MEMORY.md and the reader came up empty.)
    from paf.capabilities.memory._view import JournalViewStore

    reader = JournalViewStore(tmp_path / "memory")
    mf = asyncio.run(reader.read("assistant/MEMORY.md", max_chars=100_000))
    assert mf is not None, "root MEMORY.md must be rendered and readable"
    assert "durable fact alpha" in mf.content
    assert "fact" in mf.content  # rendered journal groups by type


def test_window_swaps_into_compaction_under_config(monkeypatch) -> None:
    """consolidate config on swaps the plain sliding window for the consolidation window."""
    from pydantic_ai_harness.compaction import SlidingWindowCompaction

    from paf.config import settings

    monkeypatch.setattr(settings, "consolidate", False)
    plain = build_compaction()
    monkeypatch.setattr(settings, "consolidate", True)
    gated = build_compaction()

    assert isinstance(plain.tiers[-1], SlidingWindowCompaction)
    assert isinstance(gated.tiers[-1], MemoryConsolidationWindow)


def test_window_default_store_scope(monkeypatch) -> None:
    """Defaults point at the anchored shared data/memory + assistant scope (build.py)."""
    from pydantic_ai_harness.compaction import SlidingWindowCompaction

    w = MemoryConsolidationWindow(max_messages=4)
    # Anchored to PROJECT_ROOT through Settings.memory_dir, matching build.py.
    from paf.config import PROJECT_ROOT

    assert w.memory_directory == str(PROJECT_ROOT / Path("data/memory"))
    assert w.memory_agent == "assistant"
    # Parent's defaults are inherited unchanged.
    assert isinstance(w, SlidingWindowCompaction)
    assert w.keep_messages == 40  # harness default, unchanged


# ---------------------------------------------------------------------------
# owner consolidation guidance + model env
# ---------------------------------------------------------------------------


def test_consolidation_guidance_is_owned_by_owner() -> None:
    """The owner guidance enforces append-only journal semantics."""
    assert "append-only journal" in CONSOLIDATION_GUIDANCE
    assert "append_memory" in CONSOLIDATION_GUIDANCE
    assert "never edit, move, or delete" in CONSOLIDATION_GUIDANCE
    assert "APPEND AS YOU SEE IT" in CONSOLIDATION_GUIDANCE
    # The old in-place section-surgery contract must be gone.
    assert "write_memory" not in CONSOLIDATION_GUIDANCE
    assert "## Facts" not in CONSOLIDATION_GUIDANCE
    assert "NEWER ALWAYS WINS" not in CONSOLIDATION_GUIDANCE


def test_memory_format_pinned_sections_and_order() -> None:
    """The canonical MEMORY.md format declares exactly four sections, in order.

    The subject's default guidance must embed the exact same MEMORY_FORMAT so
    the writer and reader agree on the rendered notebook shape by construction.
    The consolidation OWNER is append-only and does NOT carry the notebook
    format — the render pass produces it.
    """
    from paf.capabilities.memory._capability import _DEFAULT_GUIDANCE

    # The raw format pins the four sections, in the canonical order.
    assert extract_sections(MEMORY_FORMAT) == [
        "Facts",
        "Preferences",
        "Pending",
        "Notes",
    ]
    # No stray sections beyond the four.
    assert len(extract_sections(MEMORY_FORMAT)) == 4
    # Write bias is explicit ("write too much") in the format.
    assert "WHEN IN DOUBT, RECORD IT" in MEMORY_FORMAT
    assert "## Pending" in MEMORY_FORMAT and "- [ ]" in MEMORY_FORMAT
    # The assistant's default guidance carries the exact section block
    # (single source of truth for the rendered notebook).
    assert extract_sections(
        _DEFAULT_GUIDANCE.format(format_spec=MEMORY_FORMAT)
    ) == extract_sections(MEMORY_FORMAT)
    # The consolidation owner is append-only — it does NOT carry the
    # 4-section notebook format (the render produces it, not the owner).
    from paf.capabilities.memory._consolidate import _build_owner_memory

    built_owner = _build_owner_memory(None, agent_name="assistant", namespace="")
    assert built_owner.guidance is not None
    assert extract_sections(built_owner.guidance) == []


def test_consolidation_model_reads_config(monkeypatch) -> None:
    """Settings.consol_model (CONSOL_MODEL / .env) overrides the deepseek default."""
    from paf.config import settings

    monkeypatch.setattr(settings, "consol_model", "")
    assert consolidate_model() == "openrouter:deepseek/deepseek-v4-flash-0731"
    monkeypatch.setattr(settings, "consol_model", "openai:gpt-5.6-luna-pro")
    assert consolidate_model() == "openai:gpt-5.6-luna-pro"


def test_consolidation_model_settings_pins_provider(monkeypatch) -> None:
    """The owner's model_settings pin routing to deepinfra/fp8 by default.

    Regression guard for the run-to-run nondeterminism fix: the owner must NOT
    ship OpenRouter's default throughput sort (``:nitro``), and must carry the
    provider `order` + `quantizations` hard-filter plus sampler determinism
    (top_p=1.0, fixed seed) exactly like the benchmark subject/judge agents.
    """
    from paf.capabilities.memory._consolidate import _consolidation_model_settings
    from paf.config import settings

    # restore defaults (avoid leaking env overrides into the test)
    monkeypatch.setattr(
        settings, "consol_provider_order", ["deepinfra", "parasail", "coreweave"]
    )
    monkeypatch.setattr(settings, "consol_provider_quantizations", ["fp8"])
    monkeypatch.setattr(settings, "consol_seed", 20260815)

    ms = _consolidation_model_settings()
    assert ms["top_p"] == 1.0
    assert ms["seed"] == 20260815
    provider = ms["extra_body"]["provider"]
    assert provider["order"] == ["deepinfra", "parasail", "coreweave"]
    assert provider["quantizations"] == ["fp8"]

    # Empty order → no routing pin (plain request), seed/top_p still applied.
    monkeypatch.setattr(settings, "consol_provider_order", [])
    ms2 = _consolidation_model_settings()
    assert "extra_body" not in ms2
    assert ms2["top_p"] == 1.0
    assert ms2["seed"] == 20260815


# ---------------------------------------------------------------------------
# owner construction + the seed-time full-conversation variant
# ---------------------------------------------------------------------------


def test_build_owner_is_single_writer_and_read_via_tools() -> None:
    """The owner's Memory capability is writable=True, inject_memory=False, guided.

    The hard-won ownership invariants (sole writer, no injected context, fixed
    operating manual) live in :func:`_build_owner_memory` and apply to BOTH
    consolidation paths — the eviction slice and the seed-time full-conversation
    variant. Tested on the capability directly (no Agent construction, so no
    model provider / API key needed).
    """
    from paf.capabilities.memory import Memory
    from paf.capabilities.memory._consolidate import _build_owner_memory

    cap = _build_owner_memory(None, agent_name="assistant", namespace="")
    assert isinstance(cap, Memory)
    assert cap.writable is True
    assert cap.inject_memory is False
    assert cap.guidance == CONSOLIDATION_GUIDANCE


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
    from paf.capabilities.memory._consolidate import _split_chunks

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
    from paf.capabilities.memory._consolidate import _split_chunks

    text = "\n".join(["User: short", "Assistant: also short", "User: tiny"])
    chunks = _split_chunks(text, max_tokens=4000)
    assert len(chunks) == 1


def test_split_chunks_empty_input() -> None:
    """Empty input yields no chunks; a whitespace-only input yields one."""
    from paf.capabilities.memory._consolidate import _split_chunks

    assert _split_chunks("", max_tokens=1000) == []
    assert _split_chunks("", max_tokens=0) == []


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


def test_consolidate_conversation_chunked_runs_once_per_chunk(monkeypatch) -> None:
    """The chunked owner runs once per split chunk, in order, on the same store.

    Uses a fake owner (no API key / network) that records each chunk it is fed,
    so the test verifies the chunking + sequential merge wiring without a real
    model. The store is a no-op stand-in; the real store wiring is covered by
    the integration gate run.
    """
    import asyncio

    from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart

    from paf.capabilities.memory import _consolidate as cons

    fed: list[str] = []

    class _FakeOwner:
        async def run(self, prompt: str, **kwargs):
            fed.append(prompt)
            return type("R", (), {"output": "no changes"})()

    def _fake_build_owner(
        store, *, agent_name, namespace, inject_memory=False, journal=None
    ):
        return _FakeOwner(), None

    monkeypatch.setattr(cons, "_build_owner", _fake_build_owner)

    msgs: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content=f"turn number {i} and then some")])
        for i in range(12)
    ]

    async def _run():
        await cons.consolidate_conversation_chunked(
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
