"""Tests for the lean degeneration guard (`src/paf/degeneration_guard.py`).

Covers the real failure mode (the DeepSeek v4.1 "thinking forever" narration
loop), the single-phrase false-positive regression (e.g. "hmm let me get the
exact line first then decide"), near-duplicate variation, normal-thinking
negative controls, the streaming (delta) path used by the runtime hook — and the
`ToolCallGuard` repeat-tool-call detector (the same tool re-issued with the same
UI-identically-rendered args).
"""

from __future__ import annotations

import asyncio

from pydantic_ai.messages import PartDeltaEvent, ThinkingPartDelta

from paf.degeneration_guard import (
    DegenerationError,
    DegenerationGuard,
    ToolCallGuard,
    _tool_key,
    run_guarded,
)

# A real narration loop seen from deepseek-v4.1-flash at medium reasoning effort:
# a block of SEVERAL DISTINCT phrases cycling, with blank lines rendered between
# phrases. Each distinct phrase recurs `CYCLES` times; blank lines between.
CYCLES = 4
_LOOP_BLOCK = [
    "OK, I'll run it now.",
    "Let me do it.",
    "Let me run the command.",
    "Let me go.",
]
KNOWN_REPEAT = "\n\n".join("\n\n".join(_LOOP_BLOCK) for _ in range(CYCLES))


def feed_lines(guard: DegenerationGuard, text: str):
    """Feed a multi-line text through `add_line`; return first detection (or None)."""
    out = None
    for line in text.splitlines():
        result = guard.add_line(line)
        if result is not None:
            return result
    return out


def test_known_repeat_detected_with_defaults():
    """A several-phrase loop cycling 4x trips the DEFAULT guard config."""
    guard = DegenerationGuard()  # window=32, max_repeats=4, min_distinct_lines=3
    result = feed_lines(guard, KNOWN_REPEAT)
    assert result is not None
    assert result.detected
    assert result.reason == "thinking_loop"
    assert result.repeat_count >= 4


def test_single_common_phrase_does_not_trip():
    """Regression: 'hmm let me get the exact line first then decide' repeating a
    few times while thinking is NORMAL and must NOT trip the default config."""
    phrase = "hmm let me get the exact line first then decide"
    # The phrase cycles several times, but it is ONE family amid varied lines.
    text = "\n".join(
        [
            phrase,
            "ok now i need to pass the right args",
            phrase,
            "let me check the signature again",
            phrase,
            "then call it with the parsed values",
            phrase,
            "wrap it in try except to be safe",
            phrase,  # 5 total occurrences of ONE distinct phrase
        ]
    )
    guard = DegenerationGuard()
    assert feed_lines(guard, text) is None


def test_blank_lines_do_not_pollute_or_false_positive():
    """Empty lines are skipped, so a turn with natural paragraph gaps is safe."""
    guard = DegenerationGuard()
    result = feed_lines(
        guard,
        "Let me look at the build output.\n\n\nIt failed.\n\nLet me run the formatter.\n",
    )
    assert result is None


def test_normal_thinking_no_false_positive():
    """A real, non-looping thinking turn must NOT trip the guard."""
    guard = DegenerationGuard()
    result = feed_lines(
        guard,
        "The formatting error is in specs.py line 592.\n"
        "Let me run ruff format on it.\n"
        "It should reformat cleanly.\n"
        "Then I will run the tests.\n"
        "The tests should pass after the fix.\n"
        "Now let me verify the diff.\n",
    )
    assert result is None


def test_two_common_phrases_not_enough():
    """Two frequently-repeated phrases is still normal; needs 3 distinct lines."""
    guard = DegenerationGuard()  # min_distinct_lines defaults to 3
    result = feed_lines(
        guard,
        # two distinct phrases each repeating 4x, other lines vary -> only 2 lines
        "\n".join(
            [
                "let me check the result",
                "let me verify the output",
                "let me check the result",
                "let me verify the output",
                "let me check the result",
                "let me verify the output",
                "let me check the result",
                "let me verify the output",
                "then i will wrap it up",
            ]
        ),
    )
    assert result is None


def test_three_distinct_lines_exact_repeat_trips():
    """Three distinct phrases each recurring 4x *identically* is a loop (the real
    narration pattern repeats whole phrases verbatim, not with fuzzy variation)."""
    guard = DegenerationGuard()  # defaults
    text = "\n".join(
        [
            "Let me run the command",
            "Let me check the result",
            "Let me go",
            "Let me run the command",
            "Let me check the result",
            "Let me go",
            "Let me run the command",
            "Let me check the result",
            "Let me go",
            "Let me run the command",
            "Let me check the result",
            "Let me go",
        ]
    )
    result = feed_lines(guard, text)
    assert result is not None and result.detected
    assert result.reason == "thinking_loop"


def test_stream_thinking_detects_across_delta_boundaries():
    """Runtime feeds chunks that may end mid-line; the guard must still catch the loop."""
    guard = DegenerationGuard()
    text = KNOWN_REPEAT
    result = None
    i = 0
    while i < len(text):
        chunk = text[i : i + 7]  # tiny chunks that split lines arbitrarily
        i += 7
        result = guard.stream_thinking(chunk)
        if result is not None:
            break
    assert result is not None
    assert result.detected


# ----------------------------------------------------------------------
# ToolCallGuard — repeat-tool-call loop detector
# ----------------------------------------------------------------------

SEARCH_ARGS = {
    "pattern": "svg",
    "path": "projects/my-site/src",
    "include_glob": "*.astro",
}


def test_identical_tool_call_loop_detected():
    """The real observed failure: the SAME search_files call re-issued in a row
    trips the guard once it recurs 5 times consecutively."""
    guard = ToolCallGuard()  # max_repeats=5
    result = None
    for n in range(1, 11):  # 10 identical calls in a row
        result = guard.add_tool_call("search_files", SEARCH_ARGS)
        if n == 5:
            # First detection on the 5th consecutive call.
            assert result is not None and result.detected
            assert result.repeat_count == 5
    assert result is not None and result.detected
    assert result.reason == "repeated_tool_call"
    assert result.repeat_count == 10  # still fires each subsequent repeat
    assert result.line == _tool_key("search_files", SEARCH_ARGS)


def test_exactly_five_trips_four_does_not():
    """Threshold boundary: 4 consecutive identical calls must NOT trip; the 5th
    identical call does."""
    guard = ToolCallGuard()
    for _ in range(4):
        assert guard.add_tool_call("search_files", SEARCH_ARGS) is None
    result = guard.add_tool_call("search_files", SEARCH_ARGS)
    assert result is not None and result.detected


def test_count_resets_when_identity_switches():
    """Consecutive-only: an intervening different call resets the run, so a call
    spread across a turn among other calls never accumulates to a false trip."""
    guard = ToolCallGuard()
    # 4 identical, then a different call (resets), then 4 more identical => 8
    # identical total but never 5 in a row -> must NOT trip.
    for _ in range(4):
        guard.add_tool_call("search_files", SEARCH_ARGS)
    guard.add_tool_call("read_file", {"path": "other.md"})
    result = None
    for _ in range(4):
        result = guard.add_tool_call("search_files", SEARCH_ARGS)
    assert result is None  # reset after the switch; max run is 4


def test_arg_key_order_is_canonical():
    """Same args with dict keys in a different order must collapse to ONE key, so
    an arg-order-only drift still trips the loop."""
    guard = ToolCallGuard()
    reordered = {
        "include_glob": "*.astro",
        "path": "projects/my-site/src",
        "pattern": "svg",
    }
    result = None
    for _ in range(5):
        result = guard.add_tool_call("search_files", reordered)
    assert result is not None and result.detected


def test_tool_key_matches_ui_identity():
    """The canonical key is `toolName(humanized_args)` — the string the paf-ui
    ToolBlock trigger renders — so detection matches what the UI would show."""
    key = _tool_key("search_files", SEARCH_ARGS)
    assert key.startswith("search_files(")
    assert "pattern: svg" in key
    assert "path: projects/my-site/src" in key


def test_tool_guard_reset_clears_state():
    guard = ToolCallGuard()
    for _ in range(5):
        guard.add_tool_call("search_files", SEARCH_ARGS)
    guard.reset()
    assert guard.add_tool_call("search_files", SEARCH_ARGS) is None


# ----------------------------------------------------------------------
# run_guarded — shared guarded-run seam for background agent.run call sites
# ----------------------------------------------------------------------


def _fake_agent(thinking_chunks, result_value="ok", include_non_stream_node=False):
    """A minimal stand-in for ``agent.iter`` with the contracts run_guarded uses:
    ``iter(prompt)`` -> async CM yielding a ``run`` with ``.ctx`` and async-iterable
    nodes. The thinking node is a REAL ``pydantic_ai.agent.ModelRequestNode``
    subclass whose ``.stream()`` is stubbed to yield ThinkingPartDelta events
    (so ``isinstance(node, ModelRequestNode)`` holds and the guard streams it).
    ``include_non_stream_node`` prepends a real graph node WITHOUT ``.stream()``
    to exercise the skip path (mirrors UserPromptNode in the real graph)."""

    from pydantic_ai.agent import ModelRequestNode
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    class _FakeModelNode(ModelRequestNode):
        def __init__(self, chunks):
            # ModelRequestNode is a dataclass; supply a minimal request. The
            # real graph machinery is never invoked — only .stream() is used.
            super().__init__(request=ModelRequest(parts=[UserPromptPart(content="go")]))
            self._chunks = list(chunks)

        def stream(self, ctx):  # type: ignore[override]  # stubbed for tests
            async def gen():
                for c in self._chunks:
                    yield PartDeltaEvent(
                        index=0,
                        delta=ThinkingPartDelta(content_delta=c),
                        event_kind="part_delta",
                    )

            return _AsyncGen(gen())

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    # A real graph node WITHOUT .stream() (mirrors UserPromptNode) — the guard
    # must skip it via the ModelRequestNode/CallToolsNode isinstance check. A
    # bare ModelRequest stands in: it has no .stream() and is not a request/tool
    # node, so run_guarded `continue`s over it.
    non_stream = ModelRequest(parts=[UserPromptPart(content="")])

    class _NodeIter:
        """async iterator over the fake nodes."""

        def __init__(self, nodes):
            self._nodes = iter(nodes)

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._nodes)
            except StopIteration:
                raise StopAsyncIteration from None

    class _Run:
        ctx = object()
        result = result_value
        nodes: list = [_FakeModelNode(thinking_chunks)]
        if include_non_stream_node:
            nodes.insert(0, non_stream)

        def __aiter__(self):
            return _NodeIter(self.nodes)

    class _FakeAgent:
        def iter(self, prompt):
            self.prompt = prompt

            class _CM:
                async def __aenter__(self):
                    return _Run()

                async def __aexit__(self, *a):
                    return False

            return _CM()

    return _FakeAgent()


class _AsyncGen:
    """Async iterator + context manager adapter over a sync generator, matching
    pydantic-ai's streamed-response contract (used as ``async with ... as stream``)."""

    def __init__(self, gen):
        self._gen = gen

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return await self._gen.__anext__()
        except StopAsyncIteration:
            raise

    async def aclose(self):
        # Closing the stream mid-iteration (the tool-loop guard raises during the
        # stream) must close the underlying generator, mirroring an async gen's
        # aclose().
        self._gen.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


def test_run_guarded_detects_thinking_loop():
    """A background agent streaming a thinking-narration loop through run_guarded
    raises DegenerationError (the thread-summarizer failure mode). Uses 4 distinct
    lines (the observed loop shape), each recurring 5x -> needs max_repeats=4."""
    block = "\n".join(
        [
            "The user wants me to build the diagram.",
            "Let me look at how existing diagrams are structured.",
            "Let me check the components and existing diagram articles.",
            "Let me search for existing SVG diagrams in the site.",
        ]
    )
    loop = "\n".join([block] * 5) + "\n"  # 4 distinct lines x5 = 20 lines
    agent = _fake_agent([loop])
    try:
        asyncio.run(run_guarded(agent, "build it"))
        raise AssertionError("expected DegenerationError")
    except DegenerationError as e:
        assert e.result.reason == "thinking_loop"


def test_run_guarded_returns_result_when_clean():
    """A non-looping background agent returns its result normally."""
    clean = "Let me build it.\nThen I will verify the diff.\nDone.\n"
    agent = _fake_agent([clean], result_value="the-answer")
    result = asyncio.run(run_guarded(agent, "build it"))
    assert result == "the-answer"


def test_run_guarded_skips_non_streamable_nodes():
    """run_guarded must not call .stream() on graph nodes that lack it (e.g.
    UserPromptNode) — it continues past them and still detects a loop on the
    streamable ModelRequestNode that follows. Regression for the live
    'UserPromptNode' object has no attribute 'stream' crash on the memory
    owner."""
    block = "\n".join(
        [
            "The user wants me to build the diagram.",
            "Let me check the structure.",
            "Let me search for SVG diagrams in the site.",
        ]
    )
    loop = "\n".join([block] * 5) + "\n"
    agent = _fake_agent([loop], include_non_stream_node=True)
    try:
        asyncio.run(run_guarded(agent, "build it"))
        raise AssertionError("expected DegenerationError")
    except DegenerationError:
        pass


def test_run_guarded_tool_loop_raises_during_stream():
    """Regression: the tool-loop guard must raise DURING the tool node's stream
    (at the first event), not after it — so pydantic-ai's CallToolsNode commits
    the partial tool returns as 'interrupted', making the history consistent
    instead of leaving an unprocessed tool call that breaks the next user prompt
    with 'Cannot provide a new user prompt when the message history contains
    unprocessed tool calls'."""

    from pydantic_ai.agent import CallToolsNode
    from pydantic_ai.messages import ModelRequest, ToolCallPart

    started = []
    completed = []

    class _FakeToolNode(CallToolsNode):
        def __init__(self, parts):
            super().__init__(model_response=parts)

        def stream(self, ctx):  # type: ignore[override]  # stubbed for tests
            async def gen():
                started.append(1)  # the tool node's stream starts
                yield None
                completed.append(1)  # only reached if the stream runs to completion

            return _AsyncGen(gen())

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    # 5 identical tool calls in a row -> guard fires on the 5th.
    parts = [
        ModelRequest(parts=[ToolCallPart(tool_name="search_files", args=SEARCH_ARGS)])
        for _ in range(5)
    ]

    class _NodeIter:
        def __init__(self, nodes):
            self._nodes = iter(nodes)

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._nodes)
            except StopIteration:
                raise StopAsyncIteration from None

    class _Run:
        ctx = object()
        result = "ok"
        nodes = [_FakeToolNode(p) for p in parts]

        def __aiter__(self):
            return _NodeIter(self.nodes)

    class _FakeAgent:
        def iter(self, prompt):
            class _CM:
                async def __aenter__(self):
                    return _Run()

                async def __aexit__(self, *a):
                    return False

            return _CM()

    try:
        asyncio.run(run_guarded(_FakeAgent(), "go", tool_guard=ToolCallGuard()))
        raise AssertionError("expected DegenerationError")
    except DegenerationError as e:
        assert e.result.reason == "repeated_tool_call"
    # The 5th (triggering) tool node's stream STARTED (the raise is during the
    # stream) but did NOT complete — pydantic-ai commits the partial returns as
    # 'interrupted' so the history stays consistent.
    assert len(started) == 5
    assert len(completed) == 4


def test_run_guarded_thinking_chunks_across_deltas():
    """Loops detected even when thinking arrives chunked (stream_thinking buffers
    across PartDeltaEvents)."""
    block = "\n".join(["Search the tree.", "Read the index.", "Verify the diff."])
    loop = "\n".join([block] * 5) + "\n"
    agent = _fake_agent([loop])
    try:
        asyncio.run(run_guarded(agent, "go"))
        raise AssertionError("expected DegenerationError")
    except DegenerationError:
        pass
