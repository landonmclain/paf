"""`TurnIndexCapability`: a one-line-per-turn index in context + gated `read_turn`.

This is M2's first feature (see the vault: `Turn-Summary-Index.md`), structured as an
installable pydantic-ai capability. It is the *episodic turn index* arm -- distinct from
compaction (context-fit) and `Memory` (durable facts).

It separates TWO concerns (2026-08-08):

1. **Write path -- collect + summarize, per completed turn.** `after_model_request`
   fires the moment the model answers each turn: it slices *this turn* (last user
   anchor -> the completed response) straight from the live `ctx.messages`, summarizes
   it, and appends one line to `index_store`, keyed by a STABLE `turn_id`. This runs
   synchronously today and is a pure per-turn write that moves to a background worker
   unchanged. It never touches context and is independent of compaction.

2. **Read path -- inject + recall.** `before_model_request` injects a BOUNDED tail of
   the index into the current request (recognition beats recall) and `read_turn`
   resolves a ref back into pydantic-ai persistence through the `HistorySource` seam.
   This is the only place the index touches context.

Architecture (two persistence surfaces, intentionally decoupled):

1. **Its own append-only persistence** (`index_store`) holds the *derived* one-line
   summaries, keyed by `(conversation_id, run_id, turn_id)`. Flat JSONL, flat-file-first.
   The index NEVER stores full turns.
2. **Reads pydantic-ai persistence directly** (`source`) -- it holds a `HistorySource`
   (e.g. `SnapshotHistorySource(store)` over the same store a `StepPersistence` capability
   writes) and resolves those keys back to the original messages on demand via `read_turn`.

It works with ANY pydantic-ai persistence backend because it consumes the `HistorySource`
protocol seam (satisfied by InMemory/File/Sqlite/Mongo step stores via `SnapshotHistorySource`),
not a store of its own choosing.

```python
from pydantic_ai import Agent
from pydantic_ai_harness.conversation_search import SnapshotHistorySource
from pydantic_ai_harness.step_persistence import FileStepStore, StepPersistence
from turn_index import TurnIndexCapability

store = FileStepStore('.step-persistence')
agent = Agent(
    'openrouter:deepseek/deepseek-v4-flash-0731:nitro',
    capabilities=[
        StepPersistence(store=store),                        # pydantic-ai owns writing full history
        TurnIndexCapability(
            source=SnapshotHistorySource(store),             # reads pydantic-ai persistence
            index_store=TurnIndexStore('.turn-index'),   # owns the derived one-line index
        ),
    ],
)
```

The `before_model_request` injection is the "recognition beats recall" keystone: a bounded
one-line-per-turn menu is injected into the current request (Memory-style, marker-stripped so
it never accumulates), letting the model *recognize* a relevant turn and `read_turn` it, instead
of forcing a blind `search` first.

The one-liner is a real LLM summarizer by default (`_summarizer.py`: a pydantic-ai
structured-output `TurnSummary` call with char-budget retry), injected through the
`summarizer` seam and falling back to a deterministic stub when unset. The write
path is deliberately background-ready.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from pydantic_ai.agent.abstract import AgentInstructions
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    TextContent,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AgentToolset

from pydantic_ai_harness.conversation_search._source import HistorySource
from pydantic_ai_harness.step_persistence import FileStepStore

from ._generate import Turn, summarize_turn_append, this_turn_id, this_turn_slice
from ._extract import build_turn_context
from ._store import TurnIndexStore
from ._summarizer import TurnSummarizer
from ._toolset import TurnIndexToolset

_MARKER = 'turn-index.v1'

_INSTRUCTIONS = (
    'A one-line summary of every past turn is kept in context (the turn index). '
    'Each line shows a ref. When a turn looks relevant, call `read_turn` with that ref to pull '
    'the full original turn back into context. When you need the wider picture beyond a single '
    'turn, use `search_conversation_history`.'
)


@dataclass
class TurnIndexCapability(AbstractCapability[AgentDepsT]):
    """Episodic turn index: one-line summaries in context + gated `read_turn` into pyai persistence."""

    source: HistorySource
    """Read seam into pydantic-ai persistence. Use `SnapshotHistorySource` over the store a
    `StepPersistence` capability writes, so it works with any backend."""

    index_store: TurnIndexStore
    """Append-only JSONL store of one-line summaries (the capability's OWN persistence). Never
    stores full turns -- only the (conversation, run, turn_id) keys into `source`."""

    store: FileStepStore | None = None
    """Optional shared pydantic-ai persistence store this capability derives its index from.

    **Deprecated for the write path** (see `after_model_request`): the per-turn
    index is now built straight from the live `ctx.messages` in the request
    hook, so it no longer needs to re-read a persisted run via `self.store`.
    Kept only for back-compat / read_turn resolution when a store is available;
    `None` (e.g. a pure `from_spec` YAML reconstruction that only needs
    read_turn/search) is fully supported for the writer.
    """

    max_lines: int = 40
    """Maximum number of index lines injected into the current request."""

    summarizer: TurnSummarizer | None = None
    """Optional real-LLM turn summarizer for the WRITE path.

    When set, `after_model_request` runs `summarizer.summarize(...)` (a real
    pydantic-ai structured-output call with char-budget retry) to produce the
    one-liner `sentence` plus structured `key_terms` / `outcome`, persisted into
    the index and matched by `search_index`. When `None` (default), the
    deterministic stub one-liner is used -- the pipeline works without a model
    or API key. Defaults to a cheap secondary model, decoupled from the chat
    model, so the index write stays cheap.
    """

    inject_index: bool = True
    """Inject the bounded one-line index into the current request when true; else only static
    guidance (tool availability)."""

    add_instructions: bool = True
    """Emit an instruction telling the model `read_turn` / the index exist."""

    tool_id: str = 'turn-index'
    """Toolset id for `read_turn` / `search_index`."""

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """Provide the `read_turn` + `search_index` toolset over the source + index."""
        return TurnIndexToolset(
            self.source,
            self.index_store,
            tool_id=self.tool_id,
        )

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Tell the model the index + `read_turn` exist, unless `add_instructions` is false."""
        if not self.add_instructions:
            return None
        return _INSTRUCTIONS

    def _render_index(self, conversation_id: str | None) -> str:
        # Scope the injected one-line index to THIS conversation so live `cli-primary`
        # turns show only their own summaries -- never a global sweep of every run.
        entries = self.index_store.read_all(conversation_id=conversation_id)
        if not entries:
            return ''
        entries = entries[-self.max_lines :] if self.max_lines else entries
        lines = []
        for e in entries:
            lines.append(f'- [{e.ref}] {e.summary}')
        return '\n'.join(lines)

    async def after_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        """Index the just-completed user turn, synchronously, on the write path.

        Fires the moment the model finishes answering each turn — the earliest
        point the turn's content is complete and stable. By pydantic-ai's graph
        ordering (`_agent_graph._finish_handling`), `after_model_request` runs
        BEFORE the returned `response` is appended to `ctx.state.message_history`
        (line ~1616 vs ~1625), so the live `ctx.messages` does NOT yet hold the
        final answer. We therefore synthesize the completed turn as
        `this_turn_slice(ctx.messages) + [response]`: the slice from the turn's
        user anchor (user prompt + tool traffic) plus the just-returned response
        that closes it. Summarizes it, and appends one line to `index_store`,
        keyed by a STABLE `turn_id` (the ordinal of the anchor among the run's
        user prompts) so the ref never rots as the run grows.
        """
        try:
            messages = list(ctx.messages)
            turn_id = this_turn_id(messages)
            if turn_id >= 0:
                turn = Turn(
                    run_id=ctx.run_id,
                    conversation_id=ctx.conversation_id,
                    turn_id=turn_id,
                    messages=[*this_turn_slice(messages), response],
                )
                if self.summarizer is not None:
                    rendered = build_turn_context(turn.messages)
                    summary = await self.summarizer.summarize(rendered)
                    await summarize_turn_append(turn=turn, index_store=self.index_store, summary=summary)
                else:
                    await summarize_turn_append(turn=turn, index_store=self.index_store)
        except Exception:
            # Index build must never break the request (matches the harness's
            # `injection_errors='ignore'` / snapshot-fallback posture).
            pass
        return response

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Inject a bounded one-line-per-turn index into only the current user request.

        Uses a scope marker so it strips its own prior injection (`_remove_previous`)
        and never accumulates stale index blocks or clobbers sibling capabilities'
        injections. Logic mirrors `pydantic_ai_harness.memory.Memory.before_model_request`.
        """
        self._remove_previous(request_context.messages, _MARKER)
        if not self.inject_index:
            return request_context

        try:
            rendered = self._render_index(ctx.conversation_id)
        except Exception:
            # Index read failure must not break the request (matches Memory's
            # `injection_errors='ignore'` default posture).
            return request_context
        if not rendered:
            return request_context

        block = f'<turn-index>\n{rendered}\n</turn-index>'
        part = UserPromptPart([TextContent(block, metadata=_MARKER)])
        latest = request_context.messages[-1]
        if not isinstance(latest, ModelRequest):
            raise RuntimeError('model request history must end with a ModelRequest')
        request_context.messages[-1] = replace(latest, parts=[*latest.parts, part])
        return request_context

    def _remove_previous(self, messages: list[ModelMessage], marker: str) -> None:
        for index, message in enumerate(messages):
            if not isinstance(message, ModelRequest):
                continue
            parts: list[ModelRequestPart] = []
            changed = False
            for part in message.parts:
                if not isinstance(part, UserPromptPart) or isinstance(part.content, str):
                    parts.append(part)
                    continue
                content = [
                    item
                    for item in part.content
                    if not (isinstance(item, TextContent) and item.metadata == marker)
                ]
                if len(content) == len(part.content):
                    parts.append(part)
                else:
                    changed = True
                    if content:
                        parts.append(replace(part, content=content))
            if changed:
                messages[index] = replace(message, parts=parts)

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Return the name used by custom capability specs."""
        return 'TurnIndex'

    @classmethod
    def from_spec(
        cls,
        index_dir: str | Path,
        *,
        max_lines: int = 40,
        inject_index: bool = True,
        add_instructions: bool = True,
        tool_id: str = 'turn-index',
    ) -> 'TurnIndexCapability[Any]':
        """Reconstruct the capability from a spec-declarable store directory.

        `source` and `index_store` are not YAML-serializable (a `HistorySource` is
        an object), so `from_spec` rebuilds BOTH from a single root directory:

            - When `index_dir` points at the same directory a `FileStepStore`
              (backing `StepPersistence`) writes, the reconstructed
              `SnapshotHistorySource` resolves `read_turn` refs against exactly
              that persisted history -- the shared-store invariant, reachable
              declaratively.
            - The append-only turn index lives under the same root as its own
              `data/index.jsonl`, so the whole capability is self-owning from a
              spec. Use `TurnIndexCapability(index_dir=...)` in YAML:

                  capabilities:
                    - TurnIndex: {index_dir: data/paf}

        For programmatic assembly that must share a live store instance (as
        `paf.capabilities.persistence.build_persistence` does), construct
        `TurnIndexCapability(source=..., index_store=...)` directly instead.
        """
        from pydantic_ai_harness.conversation_search import SnapshotHistorySource
        from pydantic_ai_harness.step_persistence import FileStepStore

        directory = Path(index_dir)
        directory.mkdir(parents=True, exist_ok=True)
        store = FileStepStore(directory=directory)
        return cls(
            source=SnapshotHistorySource(store),
            index_store=TurnIndexStore(directory),
            store=store,
            max_lines=max_lines,
            inject_index=inject_index,
            add_instructions=add_instructions,
            tool_id=tool_id,
        )