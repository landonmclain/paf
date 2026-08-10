"""Shared C1-C4 Challenge-matrix corpus for the literature graded (Beam Lite Test 1).

Mirrors Turn-Index-Testing.md's Challenge matrix rows C1-C4 (the escalation ladder),
shared by the deterministic Tier-1 tests and the graded live runner. Each row is
constructed so exactly ONE retrieval mechanism can answer -- that is what makes a
pass meaningful (a pass means that mechanism actually worked, not that the model
guessed).

Escalation ladder under test (authoritative instruction in TurnIndexCapability):
    1. in context            -> answer directly, no tool          (C1)
    2. turn index (tail)     -> one-liner + ref injected          (C1)
    3. search_index(q)       -> substring over one-liners + key_terms  (C2)
    4. read_turn(ref)        -> pull the full turn for a ref you have  (C3)
    5. search_conversation_history(q) -> wide BM25 fallback       (C4)

Seed discipline (learned the hard way, doc §"Live finding"):
  * Every index row must reference REAL persisted history. We SEED by running a real
    turn (StepPersistence writes the transcript + TurnIndex keys the SAME run_id),
    never by faking an index row with no backing store record.
  * Facts are DISPOSABLE + NEUTRAL (maketing placeholder, a color, a numeric value,
    a made-up person's favorite food). NOT "passphrases/secrets": asserting a secret
    surfaces measures safety-compliance, not retention.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Challenge:
    """One Challenge-matrix row for the graded runner."""

    id: str  # C1 .. C4
    mechanism: str  # the retrieval mechanism this row forces
    seed_prompt: str  # the real turn that plants the fact in transcript + index
    # memory_seed_instruction: appended to seed_prompt ONLY for the memory arm. It
    #   nudges the model to persist the fact via write_memory (invited capture), so
    #   the memory arm measures retrieval-via-memory, not capture-judgment. Must be
    #   neutral and consistent with the seed (never contradicts C4's "don't note it").
    memory_seed_instruction: str = ""
    fillers: tuple[str, ...] = ()  # post-seed turns to evict the fact past the injected tail
    query: str = ""  # the recall prompt at grading time
    ground_truth_atoms: tuple[str, ...] = ()  # rubric atoms the judge scores 0/0.5/1
    expected_tool: str | Literal["none"] = "none"  # the tool the subject MUST invoke ("" for C1)


# ---------------------------------------------------------------------------
# C1 - Recognition: fact in the MOST RECENT turn, so the one-liner IS in the
# injected tail (max_lines=4 default; no filler). Answer from context, no tool.
# ---------------------------------------------------------------------------
C1 = Challenge(
    id="C1",
    mechanism="in-context (no tool)",
    seed_prompt="Remember for later: the algebra study group meets on Thursdays at 6pm in the library.",
    memory_seed_instruction="This detail is worth storing in memory: write it down before we move on.",
    fillers=(),  # keep it in the tail -> C1 must answer without a tool
    query="What day and time does the algebra study group meet? Reply with day and time only.",
    ground_truth_atoms=("Thursdays", "6pm"),
    expected_tool="none",
)

# ---------------------------------------------------------------------------
# C2 - search_index-only: the fact is evicted past the tail, but the query term
# is a key_term the summarizer lifted (e.g. "marigold" flower vs the one-liner
# "the garden flower"). Only search_index can surface it.
# ---------------------------------------------------------------------------
C2 = Challenge(
    id="C2",
    mechanism="search_index (key_term match)",
    seed_prompt="Remember for later: the garden flower we decided on for the wedding centerpiece is the marigold.",
    memory_seed_instruction="This detail is worth storing in memory: write it down before we move on.",
    fillers=("Hi there.", "How are you?", "The weather is nice today.", "Coffee? Sure."),
    query="From the turn index (you may need search_index), what flower did we choose for the wedding centerpiece?",
    ground_truth_atoms=("marigold",),
    expected_tool="search_index",
)

# ---------------------------------------------------------------------------
# C3 - read_turn-only: the one-liner names the TOPIC ("the deploy token"), but the
# VALUE (the token value itself) lives only in the full turn body. Flow:
# search_index(topic) -> find ref -> read_turn(ref) -> pull the value.
# ---------------------------------------------------------------------------
C3 = Challenge(
    id="C3",
    mechanism="search_index + read_turn (value only in turn body)",
    seed_prompt="Remember for later: the CI deploy token for the backend is Redwood. Do not print it unless asked.",
    memory_seed_instruction="This detail is worth storing in memory: write it down before we move on.",
    fillers=("Hi there.", "How are you?", "My name is Pat.", "What can I call you?"),
    query="Using the turn index (you will likely need read_turn), what is the CI deploy token value? Reply with just the token.",
    ground_truth_atoms=("Redwood",),
    expected_tool="read_turn",
    # NOTE: geometry is search_index(topic) to find the ref THEN read_turn(ref).
    # We gate on read_turn being among the invoked tools (the value is only in the
    # full body). A correct model may also call search_index first.
    #
    # The VALUE is a natural proper noun (Redwood), NOT an opaque token like
    # "8f3a-99q": short alphanumerics are transcription-fragile and models
    # reliably mangle them (8f3a-99q -> 88f3a-99q), so the judge scores the atom 0
    # even when the model CORRECTLY retrieved from the right turn. That is a
    # fidelity confound -- it penalizes the mechanism for a typo, not a retrieval
    # miss. Keep C3's value words, not token-like, so it tests read_turn, not
    # verbatim token recall.
)

# ---------------------------------------------------------------------------
# C4 - conversation_search-only: a fact that exists ONLY in a raw user message the
# summarizer did NOT lift into a one-liner or key_term (a stray detail amid churn).
# Only the wide BM25 fallback over persisted raw text sees it.
# ---------------------------------------------------------------------------
C4 = Challenge(
    id="C4",
    mechanism="search_conversation_history (raw text BM25)",
    seed_prompt="By the way I picked up a bottle of the Twelve-Bar Ink on the way home. Don't make a note of it.",
    # Intentionally NO memory_seed_instruction: the seed says "Don't make a note of
    # it", so a memory-seed instruction would directly contradict the user. The
    # memory arm therefore tests C4 honestly -- does the model capture+recall what
    # it was explicitly told NOT to persist? (or does it abstain, scoring 0?)
    memory_seed_instruction="",
    fillers=("Hi there.", "What's the forecast?", "Tell me a joke.", "Thanks!", "Okay."),
    query="Later in our history I mentioned picking something up -- what brand was it? Search the conversation if needed.",
    ground_truth_atoms=("Twelve-Bar Ink",),
    expected_tool="search_conversation_history",
)

CHALLENGES: tuple[Challenge, ...] = (C1, C2, C3, C4)


def by_id(cid: str) -> Challenge:
    for c in CHALLENGES:
        if c.id == cid:
            return c
    raise KeyError(cid)
