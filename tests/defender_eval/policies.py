"""Evaluate whole injection POLICIES, not just the classifier's verdicts.

`eval.py` answers "does the classifier flag this?". This answers "what does the
model actually get under a given `guardrails.injection` policy, and what does it
cost?", by running each fixture through a real ``InjectionGuard`` (one fresh run
per fixture) and classifying what came out:

- ``passed``: delivered unchanged. For an injection that is a miss.
- ``flagged``: delivered whole, wrapped as untrusted with a warning.
- ``redacted``: delivered with the flagged segments replaced by a marker.
- ``withheld``: replaced with the withheld message (``block``, or a ``redact``
  fallback, whose reason is recorded).

Plus: whether the run was *tightened* (a hit under a tainting action, after
which `on_flag.require_approval` tools are refused), how much of a benign file
survived, and, for injection fixtures that carry an ``injected_marker`` (a
verbatim phrase from the injected text), whether that text actually reached the
model. That last one is ground truth, not the scorer's opinion of itself.

Policies to compare are named presets (``block``, ``redact``, ``flag``,
``observe``, ``off``) or ``spec:<agent>`` (the agent spec's own
``guardrails.injection`` block). Presets use ``mark_untrusted: flagged`` and
tighten the file-writing tools (``PRESET_REQUIRE_APPROVAL``) after a hit, so the
cost of a false alarm shows up as a tightened turn. "Tightened" counts only when
the policy actually lists tools to tighten.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from pydantic_ai.messages import ToolCallPart, ToolReturn
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.usage import RunUsage

from paf.capabilities.injection import InjectionGuard, parse_injection_policy
from paf.capabilities.injection._capability import METADATA_KEY, REDACTION_MARKER

PRESET_REQUIRE_APPROVAL = ["write_file", "edit_file", "create_directory"]
PRESETS: dict[str, dict[str, Any]] = {
    name: {
        "default_action": name,
        "on_flag": {"require_approval": PRESET_REQUIRE_APPROVAL},
    }
    for name in ("off", "observe", "flag", "redact", "block")
}


def resolve(name: str) -> dict[str, Any]:
    """A preset name or ``spec:<agent>`` -> a ``guardrails.injection`` config."""
    if name.startswith("spec:"):
        from paf.specs import _spec_injection_config, get_spec_path

        return _spec_injection_config(get_spec_path(name[5:])) or {}
    if name not in PRESETS:
        raise ValueError(
            f"unknown policy {name!r}: use {sorted(PRESETS)} or spec:<agent>"
        )
    return PRESETS[name]


@dataclass
class Outcome:
    id: str
    label: str
    category: str
    outcome: str
    fallback: str = ""
    kept: float = 1.0
    """Share of the original text that reached the model (1.0 intact, 0.0 none)."""
    tightened: bool = False
    injected_reached: bool | None = None
    """For fixtures with an ``injected_marker``: did that text reach the model?"""


def _strip(delivered: str, tag_prefix: str = "<untrusted-content-") -> str:
    """The original content inside a wrapper, without paf's own annotations."""
    text = delivered
    if tag_prefix in text:
        # The wrapper adds exactly one newline after the opening tag and one
        # before the closing tag; remove those and nothing of the content.
        start = text.index(">", text.index(tag_prefix)) + 2
        end = text.rindex("</untrusted-content-") - 1
        text = text[start:end]
    return text.replace(REDACTION_MARKER, "")


def _classify(row: dict[str, Any], out: Any, guard: InjectionGuard) -> Outcome:
    original = row["text"]
    tightened = guard.tainted and bool(guard.policy.require_approval)
    base = dict(
        id=row["id"],
        label=row["label"],
        category=row["category"],
        tightened=tightened,
    )
    if isinstance(out, ToolReturn):
        meta = (out.metadata or {}).get(METADATA_KEY, {})
        result = Outcome(
            **base, outcome="withheld", fallback=meta.get("fallback", ""), kept=0.0
        )
        delivered = ""
    elif out == original:
        result = Outcome(**base, outcome="passed")
        delivered = original
    else:
        delivered = _strip(str(out))
        outcome = "redacted" if REDACTION_MARKER in str(out) else "flagged"
        result = Outcome(
            **base,
            outcome=outcome,
            kept=round(len(delivered) / max(len(original), 1), 3),
        )
    marker = row.get("injected_marker")
    if marker:
        result.injected_reached = marker in delivered
    return result


def evaluate_policy(
    rows: list[dict[str, Any]], config: dict[str, Any], scorer=None
) -> list[Outcome]:
    """Run every fixture through a fresh guard built from ``config``."""
    policy = parse_injection_policy(config)
    template = (
        InjectionGuard(policy=policy, scorer=scorer)
        if scorer
        else InjectionGuard(policy=policy)
    )
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())

    async def go() -> list[Outcome]:
        out: list[Outcome] = []
        for row in rows:
            guard = await template.for_run(ctx)
            call = ToolCallPart(row["tool"], {}, tool_call_id="eval")
            result = await guard.after_tool_execute(
                ctx,
                call=call,
                tool_def=ToolDefinition(name=row["tool"]),
                args={},
                result=row["text"],
            )
            out.append(_classify(row, result, guard))
        return out

    return asyncio.run(go())


def summarize(outcomes: list[Outcome]) -> dict[str, Any]:
    benign = [o for o in outcomes if o.label == "benign"]
    inj = [o for o in outcomes if o.label == "injection"]
    marked = [o for o in inj if o.injected_reached is not None]
    count = lambda xs, k: sum(o.outcome == k for o in xs)  # noqa: E731
    return {
        "benign": len(benign),
        "benign_intact": sum(o.outcome in ("passed", "flagged") for o in benign),
        "benign_flagged": count(benign, "flagged"),
        "benign_partial": count(benign, "redacted"),
        "benign_lost": count(benign, "withheld"),
        "benign_kept_share": round(
            sum(o.kept for o in benign) / max(len(benign), 1), 3
        ),
        "benign_tightened": sum(o.tightened for o in benign),
        "injection": len(inj),
        "inj_unmarked": count(inj, "passed"),
        "inj_flagged": count(inj, "flagged"),
        "inj_redacted": count(inj, "redacted"),
        "inj_withheld": count(inj, "withheld"),
        "inj_tightened": sum(o.tightened for o in inj),
        "embedded": len(marked),
        "embedded_reached": sum(bool(o.injected_reached) for o in marked),
    }


_ROWS = (
    ("benign", "Benign results"),
    ("benign_intact", "  delivered whole"),
    ("benign_flagged", "    of which wrapped + warned"),
    ("benign_partial", "  delivered partly (redacted)"),
    ("benign_lost", "  lost (withheld)"),
    ("benign_kept_share", "  share of benign text kept"),
    ("benign_tightened", "  turns tightened by a false alarm"),
    ("injection", "Injection results"),
    ("inj_unmarked", "  reached the model UNMARKED (miss)"),
    ("inj_flagged", "  reached the model wrapped + warned"),
    ("inj_redacted", "  redacted"),
    ("inj_withheld", "  withheld"),
    ("inj_tightened", "  turns tightened"),
    ("embedded", "Embedded injections (ground truth)"),
    ("embedded_reached", "  injected text reached the model"),
)


def render_comparison(results: dict[str, list[Outcome]], *, title: str = "") -> str:
    names = list(results)
    sums = {n: summarize(results[n]) for n in names}
    width = max(12, *(len(n) for n in names)) + 2
    lines = []
    if title:
        lines.append(title)
    lines.append(f"{'':40}" + "".join(f"{n:>{width}}" for n in names))
    for key, label in _ROWS:
        cells = "".join(
            f"{(f'{sums[n][key]:.0%}' if isinstance(sums[n][key], float) else sums[n][key]):>{width}}"
            for n in names
        )
        lines.append(f"{label:40}{cells}")
    fallbacks = {
        n: sorted({o.fallback for o in results[n] if o.fallback}) for n in names
    }
    if any(fallbacks.values()):
        lines.append("")
        for n, reasons in fallbacks.items():
            if reasons:
                counts = {r: sum(o.fallback == r for o in results[n]) for r in reasons}
                lines.append(f"  {n}: redact fell back to withholding: {counts}")
    return "\n".join(lines)


def is_code(outcome: Outcome) -> bool:
    return outcome.category.startswith(("code_", "injection_code"))
