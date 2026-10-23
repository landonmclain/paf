"""Policy-level eval (tests/defender_eval/policies.py): what each injection
policy actually delivers, and the relationships between policies that must hold.

Exact counts drift between machines near the classifier's cutoff (see LIVE_ONLY in
test_defender_eval.py), so these gates assert RELATIONSHIPS that are true by
construction on any machine, e.g. "redact never lets more injected text through
than block", rather than absolute numbers.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("onnxruntime")

# A calibration eval over a labeled corpus, not a unit test: opt-in with `-m eval`.
pytestmark = pytest.mark.eval

from tests.defender_eval.eval import build_defense, load_fixtures  # noqa: E402
from tests.defender_eval.policies import (  # noqa: E402
    PRESETS,
    evaluate_policy,
    render_comparison,
    resolve,
    summarize,
)

ROWS = load_fixtures()


@pytest.fixture(scope="module")
def results():
    scorer = build_defense()
    return {
        name: evaluate_policy(ROWS, PRESETS[name], scorer)
        for name in ("observe", "flag", "redact", "block")
    }


@pytest.fixture(scope="module")
def sums(results):
    return {name: summarize(r) for name, r in results.items()}


def test_presets_and_spec_policies_resolve():
    assert resolve("block")["default_action"] == "block"
    assert resolve("spec:secure_assistant") == {} or isinstance(
        resolve("spec:secure_assistant"), dict
    )
    with pytest.raises(ValueError, match="unknown policy"):
        resolve("quarantine")


def test_embedded_injections_carry_a_verbatim_marker():
    tagged = [r for r in ROWS if r.get("injected_marker")]
    assert len(tagged) >= 10
    for r in tagged:
        assert r["label"] == "injection" and r["injected_marker"] in r["text"], r["id"]


def test_observe_changes_nothing(sums):
    s = sums["observe"]
    assert s["benign_intact"] == s["benign"] and s["benign_tightened"] == 0
    assert s["inj_unmarked"] == s["injection"] and s["inj_tightened"] == 0


def test_flag_never_loses_benign_content(sums):
    s = sums["flag"]
    assert s["benign_lost"] == 0 and s["benign_partial"] == 0
    assert s["benign_kept_share"] == 1.0


def test_every_hit_is_handled_the_same_way_across_acting_policies(sums):
    """flag, redact and block share one scorer: the same injections are caught
    (none reach the model unmarked that another policy would have caught)."""
    misses = {n: sums[n]["inj_unmarked"] for n in ("flag", "redact", "block")}
    assert len(set(misses.values())) == 1, misses


def test_redact_never_lets_more_injected_text_through_than_block(sums):
    assert sums["redact"]["embedded_reached"] <= sums["block"]["embedded_reached"]


def test_redact_keeps_at_least_as_much_benign_content_as_block(sums):
    assert sums["redact"]["benign_kept_share"] >= sums["block"]["benign_kept_share"]
    assert sums["redact"]["benign_lost"] <= sums["block"]["benign_lost"]


def test_flag_shows_the_model_injections_but_tightens_those_turns(sums):
    """The flag-mode trade: injected text reaches the model (wrapped), so the
    turn must be tightened every time it does."""
    s = sums["flag"]
    assert s["inj_flagged"] == s["inj_tightened"]
    assert s["embedded_reached"] == s["embedded"]


def test_tightening_counts_only_when_the_policy_lists_tools(results):
    rows = [r for r in ROWS if r["label"] == "injection"][:5]
    scorer = build_defense()
    without = evaluate_policy(rows, {"default_action": "block"}, scorer)
    assert not any(o.tightened for o in without)


def test_the_comparison_renders(results):
    text = render_comparison(results, title="t")
    for name in results:
        assert name in text
    assert "injected text reached the model" in text


def test_redacted_outcomes_record_what_survived(results):
    redacted = [o for o in results["redact"] if o.outcome == "redacted"]
    assert redacted and all(0.0 <= o.kept < 1.0 for o in redacted)
    withheld = [o for o in results["redact"] if o.outcome == "withheld"]
    assert all(o.fallback for o in withheld)  # every fallback says why


def test_outcomes_serialize_for_reports(results):
    from dataclasses import asdict

    json.dumps([asdict(o) for o in results["redact"]])
