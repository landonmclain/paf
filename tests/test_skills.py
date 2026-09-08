"""Tests for the paf Skills capability — list/load + deterministic pointer trigger.

Covers the unified capability (Landon 2026-08-19): the load mechanism
(list_skills / load_skill, folding in the harness Skills / pydantic-ai
deferred-catalog idea) paired with the skill-note-hint pointer engine
(point_to_skill, zero-LLM BM25 over frontmatter vocab). Uses the checked-in
`tests/fixtures/skills/` corpus as the deterministic test library.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic_ai import ModelRetry

from paf.capabilities.build import PROJECT_ROOT, _build_skills
from paf.capabilities.skills import Skills, SkillToolset
from paf.capabilities.skills._corpus import HintCorpus, parse_frontmatter
from paf.specs import _spec_skills_config, get_spec_path

# Deterministic, checked-in test corpus (NOT the live, gitignored data/skills/).
# `data/skills/` is the real running library and is free to grow; the tests pin
# behavior against a fixture that never changes under them.
FIXTURE_SKILLS = str(Path(__file__).resolve().parent / "fixtures" / "skills")


def _toolset(top_k: int = 3) -> SkillToolset:
    return Skills(libraries=(FIXTURE_SKILLS,), top_k=top_k).get_toolset()


# ---------------------------------------------------------------------------
# Capability construction / toolset registration
# ---------------------------------------------------------------------------


def test_skill_toolset_registers_three_tools() -> None:
    """The skill toolset exposes exactly list_skills / load_skill / point_to_skill.

    Same pattern as the FileSystem toolset: methods are registered via
    `add_function`, so they appear in `.tools` and reach the agent.
    """
    ts = _toolset()
    assert set(ts.tools) == {"list_skills", "load_skill", "point_to_skill"}


def test_capability_validates_top_k() -> None:
    """top_k must be a positive integer (config-driven validation)."""
    with pytest.raises(ValueError):
        Skills(libraries=(FIXTURE_SKILLS,), top_k=0)
    with pytest.raises(ValueError):
        Skills(libraries=(FIXTURE_SKILLS,), top_k=-1)


def test_skills_cataloged_and_wired() -> None:
    """Skills is a cataloged paf icon-capability and resolves on the demo profile."""
    from paf.capabilities.registry import CAPABILITY_TYPES

    assert CAPABILITY_TYPES["Skills"] is Skills

    from paf.capabilities.build import build_runtime_capabilities
    from paf.capabilities.profiles import DEMO_PROFILE

    built = build_runtime_capabilities(DEMO_PROFILE)
    caps = [type(c).__name__ for c in built]
    assert "Skills" in caps
    sk = next(c for c in built if type(c).__name__ == "Skills")
    assert isinstance(sk, Skills)
    assert set(sk.get_toolset().tools) == {"list_skills", "load_skill", "point_to_skill"}


# ---------------------------------------------------------------------------
# Corpus ingestion
# ---------------------------------------------------------------------------


def test_corpus_indexes_demo_skill_library() -> None:
    """The fixture yields exactly the two SKILL.md skills (README is a note)."""
    corpus = HintCorpus(roots=FIXTURE_SKILLS)
    # Exactly two SKILL.md skills in the fixture (deterministic), regardless of
    # how many plain .md notes (e.g. this README) may also be scanned.
    skills = [d for d in corpus.docs if d.kind == "skill"]
    assert sorted(d.name for d in skills) == ["code-review", "vectorized-backtest"]
    assert all(d.kind == "skill" for d in skills)
    # ref points at the on-disk SKILL.md so load_skill can read the body.
    assert all(d.ref and d.ref.endswith("SKILL.md") for d in skills)


def test_parse_frontmatter_flat_dict() -> None:
    """frontmatter parsing folds inline lists and returns a flat dict."""
    from pathlib import Path

    fm = parse_frontmatter(Path(FIXTURE_SKILLS) / "code-review" / "SKILL.md")
    assert fm["name"] == "code-review"
    assert "code" in fm["tags"] and "review" in fm["tags"]


# ---------------------------------------------------------------------------
# list_skills
# ---------------------------------------------------------------------------


def test_list_skills_lists_name_and_description() -> None:
    out = _toolset().list_skills()
    assert "code-review" in out
    assert "vectorized-backtest" in out
    # description appears (first line), not just the name
    assert "Review a code change" in out
    assert "Review a code change" in out.split("code-review")[1]


# ---------------------------------------------------------------------------
# load_skill
# ---------------------------------------------------------------------------


def test_load_skill_returns_body_on_demand() -> None:
    body = _toolset().load_skill("code-review")
    assert "# Skill: code-review" in body or "code-review" in body
    assert "Review a code change" in body


def test_load_skill_unknown_raises_model_retry() -> None:
    with pytest.raises(ModelRetry) as excinfo:
        _toolset().load_skill("no-such-skill")
    assert "no-such-skill" in str(excinfo.value)
    assert "Available skills" in str(excinfo.value)


# ---------------------------------------------------------------------------
# point_to_skill (the deterministic trigger)
# ---------------------------------------------------------------------------


def test_point_to_skill_returns_pointer_hints() -> None:
    out = _toolset().point_to_skill("can you review this pull request")
    assert "consider: code-review" in out
    # a pointer, never content
    assert "# Skill" not in out


def test_point_to_skill_ranks_matching_skill() -> None:
    hits = HintCorpus(roots=FIXTURE_SKILLS).rank("backtest my trading strategy", k=3)
    assert hits
    top_name = hits[0][0].name
    assert top_name == "vectorized-backtest"


def test_point_to_skill_respects_top_k() -> None:
    out = _toolset(top_k=1).point_to_skill("code review pull request")
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert len(lines) == 1


def test_point_to_skill_no_match_reports_cleanly() -> None:
    # terms absent from every indexed doc's vocab -> zero BM25 overlap
    out = _toolset().point_to_skill("elephant astronomy")
    assert "No skills" in out


# ---------------------------------------------------------------------------
# Spec-driven library config (option B — the vault pattern)
# ---------------------------------------------------------------------------


def test_build_skills_defaults_to_data_skills() -> None:
    """No config -> the paf-owned data/skills/ default with top_k=3.

    Checks the resolved *default path*, not the mutable on-disk library — the
    spec'd path is PROJECT_ROOT/data/skills (gitignored, free to grow/shrink).
    """
    sk = _build_skills(store=None)
    assert set(sk.get_toolset().tools) == {"list_skills", "load_skill", "point_to_skill"}
    assert sk.libraries == (PROJECT_ROOT / "data" / "skills",)
    assert sk.top_k == 3


def test_build_skills_respects_config_libraries_and_top_k() -> None:
    """A config dict drives the custom skill-library directories + top_k."""
    # Point at the deterministic fixture so the indexed set is stable.
    sk = _build_skills(
        store=None,
        skills_config={
            "libraries": [FIXTURE_SKILLS],
            "top_k": 5,
        },
    )
    ts = sk.get_toolset()
    assert ts.top_k == 5
    shown = ts.list_skills()
    assert "code-review" in shown and "vectorized-backtest" in shown


def test_spec_skills_config_parses_block(tmp_path: Path) -> None:
    """The skills: block resolves into a config dict with full-path libraries."""
    spec = tmp_path / "agent.yaml"
    spec.write_text(
        "name: agent\n"
        "model: openai:gpt-5.4-nano\n"
        "skills:\n"
        "  libraries:\n"
        "    - ~/.hermes/skills\n"
        "    - data/skills\n"
        "  top_k: 5\n"
    )
    cfg = _spec_skills_config(spec)
    assert cfg is not None
    # `~` expands to an absolute path; plain paths pass through.
    assert cfg["libraries"][0] == str(Path.home() / ".hermes" / "skills")
    assert cfg["libraries"][1] == "data/skills"
    assert cfg["top_k"] == 5


def test_spec_skills_config_absent_returns_none(tmp_path: Path) -> None:
    """No skills: block -> None (builder falls back to the demo default)."""
    spec = tmp_path / "agent.yaml"
    spec.write_text("name: agent\nmodel: openai:gpt-5.4-nano\n")
    assert _spec_skills_config(spec) is None


def test_demo_spec_resolves_skills_block() -> None:
    """The shipped demo_assistant.yaml declares a skills: block that resolves."""
    cfg = _spec_skills_config(get_spec_path("demo_assistant"))
    assert cfg is not None
    assert "libraries" in cfg and cfg["libraries"]
    assert cfg["top_k"] == 3
