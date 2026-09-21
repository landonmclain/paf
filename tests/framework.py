from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import cast

import httpx
from pydantic_ai import Agent, AgentRunResultEvent, ModelSettings, UsageLimits
from pydantic_ai.exceptions import (
    ModelHTTPError,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
)
from pydantic_ai.messages import FunctionToolCallEvent, FunctionToolResultEvent
from pydantic_ai_harness.step_persistence import FileStepStore, StepPersistence

from paf.capabilities import AgentProfile
from paf.capabilities.search import ConversationSearch, SnapshotHistorySource
from paf.observability.instrumentation import (
    _json_dumps,
    root_span,
)
from paf.specs import load_agent_spec

# ---------------------------------------------------------------------------
# Test profile
# ---------------------------------------------------------------------------

# The shared profile for the offline test suite. Tests construct
# Session("test", profile=TEST_PROFILE) instead of a production profile, so
# they never depend on the production allowlist (assistant/demo/contextbench),
# which changes as capabilities are promoted. Minimal + deterministic: the
# store, the background machinery the integration tests exercise, and Shell.
# No Browser/Visual/Subagents/Web — nothing that needs external infra
# (Chromium, Docker, API keys) at construction.
TEST_PROFILE = AgentProfile(
    name="test",
    capability_names=("StepPersistence", "Memory", "ThreadSummarizer", "Shell"),
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SUBJECT_MODEL = os.environ.get(
    "BEAM_LITE_SUBJECT_MODEL",
    "openrouter:deepseek/deepseek-v4-flash-0731",
)

# Transient network-error retry knobs (bender-adjacent resilience). A single
# OpenRouter/provider 5xx/429/408 blip should NOT kill a long graded run
# (2026-08-15: a 502 "Network connection lost" on deepseek-v4-flash aborted a
# 472-Q run at q328/472, discarding 327 completed questions — no JSON written).
# `_run_turn` retries retryable ModelHTTPError statuses up to
# LME_RETRY_MAX_ATTEMPTS total attempts, sleeping LME_RETRY_BASE_BACKOFF seconds
# between attempts (default constant; see LME_RETRY_BACKOFF_EXPONENTIAL), and
# honoring the provider's `Retry-After` header when present. Auth/validation
# 4xx (400/401/403/404/422) and 5xx-class codes NOT in the retryable set stay
# fatal. Tunable per-run. Defaults (Landon 2026-08-15: "30s sleep with 5
# retries") = 6 total attempts (initial + 5 retries) at a constant 30s.
LME_RETRY_MAX_ATTEMPTS = int(os.environ.get("LME_RETRY_MAX_ATTEMPTS", "6"))
LME_RETRY_BASE_BACKOFF = float(os.environ.get("LME_RETRY_BASE_BACKOFF", "30.0"))
# Constant (default) vs exponential backoff. Exponential doubles the base each
# retry, capped at LME_RETRY_MAX_BACKOFF; constant just repeats the base.
LME_RETRY_BACKOFF_EXPONENTIAL = (
    os.environ.get("LME_RETRY_BACKOFF_EXPONENTIAL", "0") == "1"
)
LME_RETRY_MAX_BACKOFF = float(os.environ.get("LME_RETRY_MAX_BACKOFF", "60.0"))
# Retryable=device error HTTP statuses. 5xx = server/upstream blips (502 is what
# OpenRouter surfaces as "Network connection lost"); 429 = rate-limited; 408 =
# provider-side request timeout. Everything else is left for the caller to see.
LME_RETRYABLE_STATUSES = frozenset(
    int(x)
    for x in os.environ.get("LME_RETRYABLE_STATUSES", "408,429,500,502,503,504").split(
        ","
    )
    if x.strip()
)

# ---------------------------------------------------------------------------
# OpenRouter provider-pinning + sampler determinism (2026-08-15).
# The same `deepseek-v4-flash:nitro` slug scores differently run-to-run
# (0.9043 vs 0.9468) because `:nitro` = throughput routing that silently
# switches upstream providers, and providers serve the SAME checkpoint at
# different quantization levels (fp8 vs fp4) -> real output diff at temp 0.
# Fix (vault OpenRouter-Provider-Quantization-Nondeterminism.md):
#   * pin the provider order so there's zero routing bounce,
#   * hard-filter `quantizations: ["fp8"]` so an unexpected failure can't hit
#     an fp4 box,
#   * `top_p: 1.0` + a fixed `seed` for sampler determinism.
# We ride the OpenRouter `provider` object through pydantic-ai's `extra_body`
# (openai_chat passes `extra_body=model_settings.get('extra_body')` verbatim to
# the HTTP request body — verified in pydantic-ai 2.22 source).
# Env-overridable per-run; defaults are the FINAL chosen set for the DeepSeek
# subject (DeepInfra primary, Parasail/CoreWeave fp8 fallbacks, Novita EXCLUDED).
# ---------------------------------------------------------------------------
# Judge chain (gpt-4o-mini): OpenAI first-party is the canonical host. Pin keeps
# the judge from flapping while we measure the subject. Empty disables.
JUDGE_PROVIDER_ORDER = [
    p for p in os.environ.get("LME_JUDGE_PROVIDER_ORDER", "openai").split(",") if p
]
JUDGE_PROVIDER_QUANTIZATIONS = [
    q for q in os.environ.get("LME_JUDGE_PROVIDER_QUANTIZATIONS", "").split(",") if q
]
# Fixed sampler determinism seed (use a stable constant unless overridden).
LME_SAMPLER_SEED = int(os.environ.get("LME_SAMPLER_SEED", "20260815"))


def _openrouter_model_settings(
    provider_order: list[str] | None,
    quantizations: list[str],
    seed: int | None = None,
    extra: dict | None = None,
    base: dict | None = None,
) -> ModelSettings:
    """Build a `model_settings` dict that pins OpenRouter routing + sampler.

    Returns a dict combining sampling determinism (`top_p: 1.0` + fixed seed)
    with the OpenRouter `provider` object passed via `extra_body`. When
    `provider_order` is empty/None, no routing pin is applied (plain request).
    `base` supplies default sampling params (temperature/max_tokens etc.);
    `extra` are extra plain settings added verbatim.
    """
    out: dict = dict(base or {"temperature": 0})
    out["top_p"] = 1.0
    if seed is not None:
        out["seed"] = seed
    if extra:
        out.update(extra)
    pinned_quantizations = [q for q in quantizations if q]
    if provider_order:
        provider: dict = {"order": list(provider_order)}
        if pinned_quantizations:
            provider["quantizations"] = pinned_quantizations
        # allow_fallbacks defaults True on OpenRouter — keep, so a pinned
        # primary dropping still fails over to the listed fallbacks (never to
        # an out-of-quantization fp4 box thanks to the hard filter above).
        extra_body: dict = dict(out.get("extra_body") or {})
        extra_body["provider"] = provider
        out["extra_body"] = extra_body
    return cast(ModelSettings, out)


# Memory presence knob: MEMORY=1 wires the durable Memory capability into the
# harness (inject_memory=True, subject READ-ONLY / writable=False per the M2
# owner model) and uses the turn-bounded, trim-only TurnBoundSlidingWindow tier
# (no inline consolidation — the owner runs at seed time via run_lmem).
# DEFAULT (unset) = ON: SEARCH + MEMORY (no turn index) is the intended standard
# arm. Set MEMORY=0 for a no-memory ablation run (`MEMORY=0 ...`).
MEMORY = os.environ.get("MEMORY", "1") == "1"

# Per-question memory store for the running question (run_lmem only). The
# subject agent is built once but must read the SAME per-question notebook the
# consolidation owner writes. run_lmem sets this to `<memory_dir> /
# <question_id>` before each recall; tiny_lmem (single conversation) leaves it
# None so the subject defaults to `mem_dir`.
_CURRENT_MEMORY_DIR: Path | None = None

def _build_agent(
    store: FileStepStore,
    *,
    memory_dir: Path | None = None,
) -> Agent:
    """Wire the full PAF pipeline (retrieval ladder + durable memory).

    The retrieval ladder is ConversationSearch + read_turn (paf-owned, no turn
    index): in-context -> `search_conversation_history` -> `read_turn`, plus
    the VERIFY-SPECIFICS rung — all steered by the single system prompt (the
    spec's `instructions:`; capability instruction blocks are off).

    The durable `Memory` capability is included by default (`MEMORY=1`),
    targeting ``memory_dir`` (default ``memory_dir``; callers pass a run-scoped
    dir) so a run never touches repo `data/memory`. Set `MEMORY=0` for a
    no-memory ablation.

    Default: a single `system_prompt=` (the spec's `instructions:`) owns ALL
    static guidance — identity, the CONFLICT/INFERENCE answer rules, and the
    retrieval ladder. Every capability instruction block is OFF: ConversationSearch has
    add_instructions=False and Memory uses guidance="" (only the dynamic
    MEMORY.md body still injects), so the prompt IS the one source of recall
    steering. `search_memory` is deliberately not mentioned in the prompt (the
    notebook is always injected in-context).

    With MEMORY=1: the durable `Memory` capability is loaded (a plain capability,
    subject READ-ONLY / writable=False per the M2 owner model, so the subject
    can't contaminate durable memory). The consolidation owner is run separately
    by the runner (``run_lmem``) at seed time over the full haystack.
    """
    spec = load_agent_spec("longmemeval")
    source = SnapshotHistorySource(store)
    caps = [
        StepPersistence(store=store),
        # ConversationSearch: BM25 wide fallback (add_instructions=False — the
        # single system prompt owns the retrieval ladder; this tool is just
        # registered and named by the prompt).
        ConversationSearch(
            source=source,
            scope="conversation",
            add_instructions=False,
        ),
    ]
    if MEMORY:
        # Memory arm: durable notebook injected every request. Per the M2 owner
        # model (_consolidate.py, design note §2c), the subject is READ-ONLY
        # (writable=False) — it can never contaminate durable memory. The
        # consolidation owner (writable=True) is the sole writer, folding each
        # about-to-evict slice into the store at the eviction boundary.
        from paf.capabilities.memory import Memory

        mem_dir = memory_dir or Path("memory")
        # Per-question isolation (run_lmem): the subject's memory store must
        # track the SAME dir the consolidation owner writes. run_lmem sets
        # `_CURRENT_MEMORY_DIR` to `memory/<question_id>` before each recall
        # turn; a store_resolver (resolved per-run, unlike the static store)
        # lets one agent serve every question's notebook without a rebuild.
        # For a single-conversation runner `_CURRENT_MEMORY_DIR` stays
        # None so the store defaults to `mem_dir` as before.
        from paf.capabilities.memory._view import JournalViewStore as _JournalViewStore

        def _mem_store(_ctx):
            root = _CURRENT_MEMORY_DIR or mem_dir
            return _JournalViewStore(root)

        caps.append(
            Memory(
                store=_JournalViewStore(mem_dir),
                store_resolver=_mem_store,
                agent_name="assistant",
                inject_memory=True,
                writable=False,
                # The recall/assistant carries NO memory tools: the journal-
                # rendered notebook is injected in-context, and the recall
                # ladder is search_conversation_history/read_turn (Conversation-
                # Search), not memory tools. Inject-only memory.
                expose_tools=False,
                # The journal view is a finite, curated render of the journal —
                # never a growing legacy notebook. Inject it WHOLE: the
                # journal-rendered MEMORY.md is capped per-tier by the index
                # render (t1/t2/t3), not by line/token budgets.
                # guidance="" disables the Memory instruction block entirely —
                # the single system prompt owns all static guidance. Only the
                # dynamic MEMORY.md body still injects (before_model_request).
                guidance="",
            )
        )
    return Agent(
        spec.model,
        name="locomo-subject",
        # Single system prompt owns all retrieval guidance (ladder + answer
        # rules + read-only memory contract). Capability instructions are off.
        system_prompt=spec.instructions,
        capabilities=caps,
        model_settings=spec.model_settings,
    )


def _run_turn(agent, prompt: str, cid: str, name: str, session: str) -> dict:
    """Run one turn; return {text, tools, tool_outputs, usage} over a root span.

    Graceful tool-cut (bender guard): the whole recall loop is wrapped in
    `capture_run_messages()`. If the subject runs away calling retrieval tools
    and hits `UsageLimitExceeded`, instead of crashing the run we hand the
    captured history (up to the breach) to a ZERO-tool fallback agent that must
    produce a final answer. `RECALL_REQUEST_LIMIT` (default 50) lets a test
    force the cap low to trip the bender cheaply; a normal run never exceeds
    50 requests so the fallback is a dead path unless the model loops.
    """
    tools: list[str] = []
    tool_outputs: list[str] = []
    text: str = ""
    usage: dict = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "total_tokens": 0,
        "requests": 0,
    }
    bailed: bool = False

    from pydantic_ai import capture_run_messages

    # Zero-tool fallback: plain Agent with NO capabilities = no tools, so it
    # is physically incapable of calling any retrieval tool and must answer
    # from whatever the subject retrieved before the cut.
    cutoff_agent = Agent(
        os.environ.get("RECALL_CUTOFF_MODEL", SUBJECT_MODEL),
        system_prompt=(
            "The previous run was cut off because it called too many tools. "
            "You are finishing its answer. Use the conversation history above "
            "ONLY — do not call any tools. Give the single most likely answer."
        ),
    )
    request_limit = int(os.environ.get("RECALL_REQUEST_LIMIT", "50"))
    # Default ON: cap tool calls per recall turn at 25 so a bender can't run
    # forever (or hit the 50-request wall and crash the whole run). 25 is
    # enough for these benchmarks' retrieval needs; on a genuine runaway the
    # UsageLimitExceeded path hands the accumulated context to a zero-tool
    # fallback for a graceful answer instead of crashing. Override via
    # RECALL_TOOL_CALLS_LIMIT (e.g. 0/'' to disable).
    tool_calls_limit = os.environ.get("RECALL_TOOL_CALLS_LIMIT", "25")
    _ul: dict = {"request_limit": request_limit}
    if tool_calls_limit:
        _ul["tool_calls_limit"] = int(tool_calls_limit)

    async def run():
        nonlocal text, bailed

        # A single agent pass over the prompt: stream events, collect the
        # tool names/outputs and usage, set `text`. Returns after the run
        # completes or a bender guard fires. Raises on fatal errors.
        async def _attempt() -> None:
            nonlocal text
            with capture_run_messages() as history:
                try:
                    async with agent.run_stream_events(
                        prompt,
                        conversation_id=cid,
                        usage_limits=UsageLimits(**_ul),
                    ) as es:
                        async for event in es:
                            if isinstance(event, FunctionToolCallEvent):
                                tools.append(str(event.part.tool_name or ""))
                            elif isinstance(event, FunctionToolResultEvent):
                                tpart = getattr(event, "part", None)
                                c = getattr(tpart, "content", None)
                                if c is None:
                                    c = getattr(event, "content", None)
                                if isinstance(c, str) and c.strip():
                                    tool_outputs.append(c)
                            elif isinstance(event, AgentRunResultEvent):
                                u = event.result.usage
                                usage["input_tokens"] = (
                                    getattr(u, "input_tokens", 0) or 0
                                )
                                usage["output_tokens"] = (
                                    getattr(u, "output_tokens", 0) or 0
                                )
                                usage["cache_read_tokens"] = (
                                    getattr(u, "cache_read_tokens", 0) or 0
                                )
                                usage["cache_write_tokens"] = (
                                    getattr(u, "cache_write_tokens", 0) or 0
                                )
                                usage["total_tokens"] = (
                                    getattr(u, "total_tokens", 0) or 0
                                )
                                usage["requests"] = getattr(u, "requests", 0) or 0
                                text = event.result.output
                except UsageLimitExceeded:
                    # Bender: cut the model off AND force a graceful answer.
                    # The captured history ends mid-tool-call (the cap fired
                    # before a tool ran), so repair the dangling tool calls
                    # (synthesized 'interrupted' returns) so the fallback can
                    # be handed a valid history + a fresh user prompt without
                    # triggering "Cannot provide a new user prompt when the
                    # message history contains unprocessed tool calls."
                    bailed = True
                    from pydantic_ai._agent_graph import _repair_dangling_tool_calls

                    repaired = _repair_dangling_tool_calls(
                        history, repair_last_response=True
                    )
                    res = await cutoff_agent.run(
                        "Conclude your answer now, based only on the context above.",
                        message_history=repaired,
                    )
                    text = res.output
                    usage["requests"] = len(history) if history else 1

                except UnexpectedModelBehavior as umb:
                    # Bender, output-token variant: the subject hit its
                    # max_tokens ceiling before producing a response (the
                    # "Model token limit ... exceeded" crash that killed full
                    # runs at a pathological question). Like UsageLimitExceeded,
                    # do NOT let one question kill the whole pool — hand the
                    # captured context to the zero-tool fallback for a graceful
                    # best-effort answer. Repair dangling calls first so the
                    # fallback can be fed a valid history.
                    bailed = True  # noqa: F841  # read via `nonlocal bailed` -> returned in `out`
                    try:
                        from pydantic_ai._agent_graph import _repair_dangling_tool_calls

                        repaired = _repair_dangling_tool_calls(
                            history, repair_last_response=True
                        )
                    except Exception:
                        repaired = history
                    res = await cutoff_agent.run(
                        "Conclude your answer now, based only on the context above.",
                        message_history=repaired,
                    )
                    text = res.output
                    usage["requests"] = len(history) if history else 1
                    usage["token_limit"] = getattr(umb, "model", None)
                    # note: fallback sets usage.requests but may not set total_tokens

        with root_span(
            name,
            **{
                "langfuse.observation.input": _json_dumps(
                    {"prompt": prompt, "conversation_id": cid}
                ),
                "langfuse.session.id": session,
            },
        ) as _span:
            # Retry transient provider/network failures (bender-adjacent
            # resilience). A single 5xx/429/408 blip should NOT kill a long
            # graded run — retry the WHOLE turn on the same prompt, sleeping
            # LME_RETRY_BASE_BACKOFF (constant) between attempts and honoring
            # the provider's `Retry-After` header when present. Auth/validation
            # 4xx and any other status stay fatal and propagate to the caller.
            attempt = 0
            while True:
                attempt += 1
                # Clear residue from a prior failed attempt so we don't
                # double-count tool calls / usage if this retry succeeds.
                tools.clear()
                tool_outputs.clear()
                for _k in (
                    "input_tokens",
                    "output_tokens",
                    "cache_read_tokens",
                    "cache_write_tokens",
                    "total_tokens",
                    "requests",
                ):
                    usage[_k] = 0
                text = ""

                def _delay(
                    exc: ModelHTTPError | None, _attempt: int = attempt
                ) -> float:
                    """Compute the sleep before the next retry.

                    The loop counter is captured as a default arg so ruff's B023
                    (loop-variable-in-closure) is satisfied: _delay is called
                    synchronously in this loop, so the value is read at call time.
                    """
                    if exc is not None and exc.retry_after is not None:
                        return exc.retry_after
                    if LME_RETRY_BACKOFF_EXPONENTIAL:
                        return min(
                            LME_RETRY_BASE_BACKOFF * (2 ** (_attempt - 1)),
                            LME_RETRY_MAX_BACKOFF,
                        )
                    return LME_RETRY_BASE_BACKOFF

                try:
                    await _attempt()
                    return  # succeeded (or bender guard produced an answer)
                except ModelHTTPError as exc:
                    retryable = exc.status_code in LME_RETRYABLE_STATUSES
                    if not retryable or attempt >= LME_RETRY_MAX_ATTEMPTS:
                        raise  # fatal status or out of attempts
                    delay = _delay(exc)
                    print(
                        f"[_run_turn] retryable ModelHTTPError {exc.status_code} "
                        f"(attempt {attempt}/{LME_RETRY_MAX_ATTEMPTS}) on {name!r} — "
                        f"waiting {delay:.1f}s before retrying",
                        flush=True,
                    )
                    await asyncio.sleep(delay)
                except httpx.TransportError as exc:
                    # Raw transport/network failure (no HTTP status code): a
                    # mid-stream read/connect drop, timeout, or protocol error
                    # (e.g. `httpx.ReadError` "Server disconnected without sending
                    # a response"). pydantic-ai surfaces these raw, NOT wrapped in
                    # ModelHTTPError (which requires a status_code), so the
                    # ModelHTTPError handler never saw them and a single blip
                    # killed whole 2-hour runs (2026-08-15 502 & 2026-08-18
                    # httpx.ReadError at q328/472). These are always transient —
                    # retry the whole turn with the same backoff budget.
                    if attempt >= LME_RETRY_MAX_ATTEMPTS:
                        raise  # out of retries — let it propagate fatally
                    delay = _delay(None)
                    print(
                        f"[_run_turn] retryable httpx transport error "
                        f"{type(exc).__name__} (attempt {attempt}/"
                        f"{LME_RETRY_MAX_ATTEMPTS}) on {name!r} — "
                        f"waiting {delay:.1f}s before retrying",
                        flush=True,
                    )
                    await asyncio.sleep(delay)

    asyncio.run(run())
    out = {
        "text": text,
        "tools": tools,
        "tool_outputs": tool_outputs,
        "usage": usage,
        "bailed": bailed,
    }
    return out
