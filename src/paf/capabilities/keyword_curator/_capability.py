"""KeywordCurator capability: per-turn relevance hints from the curated vocabulary.

The read side of the keyword curator. Once per real user turn it matches what the
user just said against the curator's entity/alias vocabulary and injects a short
``<relevance-hints>`` block pointing at the skills and notes most likely to help —
deterministic and zero-LLM per turn. The model that *built* the vocabulary runs
offline (the background curator); this capability only reads its store.

Injection mirrors ``ThreadSummarizer``/``Memory``: strip any prior block (matched by
a scope-qualified ``TextContent`` metadata marker) and append a fresh one to the
latest user prompt only on a real user turn — never mid tool-loop — so a block from
the turn's start stays put instead of churning on every model request. With no
store (nothing curated yet) it injects nothing and costs two ``stat`` calls.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelRequest, TextContent, UserPromptPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tools import AgentDepsT, RunContext

from paf.capabilities._markers import KEYWORD_HINTS_MARKER_BASE
from paf.capabilities._turns import _is_user_turn, _user_prompt_text

from ._config import keyword_dir_for
from ._match import (
    DEFAULT_MIN_SCORE,
    DEFAULT_TOP_K,
    IndexCache,
)
from ._render import DEFAULT_MAX_CHARS, render_hints

log = logging.getLogger(__name__)

DEFAULT_QUERY_TURNS = 2
"""How many of the latest user prompts the match reads (current + previous)."""

_DATE_STAMP = re.compile(r"^\[date: [^\]]*\]\s*")


def recent_user_text(messages: list[Any], turns: int = DEFAULT_QUERY_TURNS) -> str:
    """The latest ``turns`` user prompts' own words, newest first.

    Injected blocks (``<memory>``, ``<thread-summary>``, earlier hints) are
    stripped by the shared marker rule, and the per-message ``[date: …]`` stamp is
    removed, so only what the user actually said is matched.
    """
    texts: list[str] = []
    for msg in reversed(messages):
        if not isinstance(msg, ModelRequest):
            continue
        for part in msg.parts:
            if isinstance(part, UserPromptPart):
                text = _user_prompt_text(part)
                if text and text.strip():
                    texts.append(_DATE_STAMP.sub("", text.strip()))
        if len(texts) >= turns:
            break
    return "\n".join(texts[:turns])


@dataclass(init=False)
class KeywordCurator(AbstractCapability[AgentDepsT]):
    """Injects ``<relevance-hints>`` from the curated keyword store.

    ``@dataclass(init=False)`` is required (not cosmetic): pydantic-ai's spec
    loader validates that custom capabilities carry ``__dataclass_fields__``.
    """

    def __init__(
        self,
        *,
        agent_name: str | None = None,
        directory: str | Path | None = None,
        inject_hints: bool = True,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = DEFAULT_MIN_SCORE,
        query_turns: int = DEFAULT_QUERY_TURNS,
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> None:
        super().__init__()
        self._directory = (
            Path(directory) if directory is not None else keyword_dir_for(agent_name)
        )
        self.agent_name = agent_name
        self.inject_hints = inject_hints
        self.top_k = top_k
        self.min_score = min_score
        self.query_turns = query_turns
        self.max_chars = max_chars
        self._cache = IndexCache(self._directory)

    @property
    def directory(self) -> Path:
        return self._directory

    def _marker(self) -> str:
        scope = hashlib.sha256(str(self._directory).encode()).hexdigest()[:16]
        return f"{KEYWORD_HINTS_MARKER_BASE}:{scope}"

    def _strip_previous(self, messages: list[Any], marker: str) -> None:
        """Drop any prior hints block carrying this capability's marker."""
        for index, message in enumerate(messages):
            if not isinstance(message, ModelRequest):
                continue
            parts: list[Any] = []
            changed = False
            for part in message.parts:
                if not isinstance(part, UserPromptPart) or isinstance(
                    part.content, str
                ):
                    parts.append(part)
                    continue
                content = [
                    item
                    for item in part.content
                    if not (
                        isinstance(item, TextContent)
                        and str(item.metadata or "").startswith(
                            KEYWORD_HINTS_MARKER_BASE
                        )
                    )
                ]
                if len(content) == len(part.content):
                    parts.append(part)
                else:
                    changed = True
                    if content:
                        parts.append(replace(part, content=content))
            if changed:
                messages[index] = replace(message, parts=parts)

    def _hints_block(self, messages: list[Any]) -> tuple[str, int, float]:
        """``(rendered block, hint count, best score)``; ``("", 0, 0.0)`` if none."""
        index = self._cache.get()
        if index.size == 0:
            return "", 0, 0.0
        query = recent_user_text(messages, self.query_turns)
        hints = index.match(query, top_k=self.top_k, min_score=self.min_score)
        if not hints:
            return "", 0, 0.0
        return render_hints(hints, max_chars=self.max_chars), len(hints), hints[0].score

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """On a real user-turn start, replace the hints block with a fresh one."""
        if not self.inject_hints:
            return request_context
        messages = request_context.messages
        if not messages or not _is_user_turn(messages[-1]):
            return request_context  # mid tool-loop: leave the turn's block alone
        marker = self._marker()
        self._strip_previous(messages, marker)
        with ctx.tracer.start_as_current_span(
            "keyword_hints.inject",
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                block, count, best = self._hints_block(messages)
            except Exception:  # a hint glitch must never break the turn
                log.exception("keyword hints failed; continuing without")
                return request_context
            if span.is_recording():
                span.set_attributes(
                    {
                        "keyword_hints.count": count,
                        "keyword_hints.best_score": best,
                        "keyword_hints.sources": self._cache.size,
                        "keyword_hints.injected_chars": len(block),
                    }
                )
            if not block:
                return request_context
            latest = messages[-1]
            if not isinstance(
                latest, ModelRequest
            ):  # pragma: no cover - guaranteed by the agent graph
                return request_context
            part = UserPromptPart([TextContent(block, metadata=marker)])
            request_context.messages[-1] = replace(latest, parts=[*latest.parts, part])
        return request_context

    @classmethod
    def from_spec(cls, **kwargs: object) -> KeywordCurator[AgentDepsT]:
        """Construct from serializable options (same keys as ``__init__``)."""
        agent_name = kwargs.get("agent_name")
        directory = kwargs.get("directory")

        def _num(key: str, default: float, kind: type) -> Any:
            value = kwargs.get(key, default)
            return kind(value) if isinstance(value, int | float | str) else default

        return cls(
            agent_name=str(agent_name) if agent_name is not None else None,
            directory=str(directory) if directory is not None else None,
            inject_hints=bool(kwargs.get("inject_hints", True)),
            top_k=_num("top_k", DEFAULT_TOP_K, int),
            min_score=_num("min_score", DEFAULT_MIN_SCORE, float),
            query_turns=_num("query_turns", DEFAULT_QUERY_TURNS, int),
            max_chars=_num("max_chars", DEFAULT_MAX_CHARS, int),
        )

    @classmethod
    def get_serialization_name(cls) -> str:
        """The name used by custom capability specs (the allowlist key)."""
        return "KeywordCurator"
