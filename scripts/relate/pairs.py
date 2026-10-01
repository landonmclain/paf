"""Shared fixtures for the fact-relation experiments: smoke pairs + label wordings.

The four labels describe how a newer fact B relates to an older fact A:
same / update / related / unrelated. Two wordings exist because the first one
was ambiguous to tev1 (it read "lives in Tulsa" -> "lives in Bentonville" as
`related`); DEFS_V2 sharpens `update` and `related`. DeepSeek, gemma3 and
qwen3.5 were only ever run with DEFS_V1.
"""

from __future__ import annotations

LABELS = ("same", "update", "related", "unrelated")

DEFS_V1 = {
    "same": "states the same claim, possibly reworded",
    "update": "same claim with a changed or corrected value",
    "related": "same subject, different claim",
    "unrelated": "different subject",
}

DEFS_V2 = {
    **DEFS_V1,
    "update": (
        "same attribute of the same subject with a different value "
        "(a new city, number, date, or choice); B replaces A"
    ),
    "related": "same subject, but about a different attribute; both can be true at once",
}

DEFS = {"v1": DEFS_V1, "v2": DEFS_V2}

# (fact A, fact B, gold label) — a hand-picked smoke set, NOT an evaluation.
PAIRS = [
    ("The user lives in Tulsa.", "The user lives in Bentonville.", "update"),
    (
        "The user prefers dark roast coffee.",
        "The user likes dark-roast coffee best.",
        "same",
    ),
    (
        "The user's site is an Astro project.",
        "The user's site has a /map dependency page.",
        "related",
    ),
    (
        "The user runs Ollama on RX 580 GPUs via Vulkan.",
        "The user keeps git commits unpushed by convention.",
        "unrelated",
    ),
    (
        "The paf test suite takes about 50 seconds.",
        "The paf full unit test suite runs in roughly 50s (~769 tests).",
        "same",
    ),
    (
        "The user's drawdown ceiling for the TQQQ strategy is 35%.",
        "The user's drawdown ceiling for the TQQQ strategy is 30%.",
        "update",
    ),
]


def system_prompt(defs: dict[str, str]) -> str:
    """Instruction text for chat-model backends (decision models take `defs` as options)."""
    lines = "\n".join(f"{k} = {v}" for k, v in defs.items())
    return f"Classify how fact B relates to fact A. Answer with one label:\n{lines}"
