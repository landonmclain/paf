"""Keyword curator: source readers, the curator call, the worker pass.

No live model: the worker is driven with a fake ``curate`` callable, and the
agent itself with a ``FunctionModel``. Covers digesting, batching, the receipt
validator, drift-prevention plumbing (known entities reach later batches),
incremental passes, partial failure, vanished sources, and the review report.
"""

from __future__ import annotations

import asyncio
import functools
import json
import os
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from paf.capabilities.keyword_curator import (
    CurateBatch,
    CurateResult,
    Entity,
    KeywordJournalStore,
    KeywordWatermark,
    SourceItem,
    curate_batch,
    find_issues,
    make_batches,
    make_record,
    project_seeds,
    read_entity_map,
    run_keyword_pass,
    scan_notes,
    scan_skills,
    scan_sources,
)
from paf.capabilities.keyword_curator._background import known_entities
from paf.capabilities.keyword_curator._curate import (
    CuratorPrompt,
    check_batch,
    format_known,
)
from paf.capabilities.keyword_curator._sources import digest, snapshots


def _sync(fn):
    """Run an async test body to completion (the repo has no async pytest plugin)."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


# --- fixtures ------------------------------------------------------------------


def _skill(
    root: Path, category: str, name: str, desc: str, body: str = "Body."
) -> Path:
    d = root / category / name if category else root / name
    d.mkdir(parents=True, exist_ok=True)
    f = d / "SKILL.md"
    f.write_text(f"---\nname: {name}\ndescription: {desc}\ntags: [a, b]\n---\n{body}\n")
    return f


def _note(root: Path, rel: str, text: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


@pytest.fixture
def skills(tmp_path: Path) -> Path:
    root = tmp_path / "skills"
    _skill(root, "trading", "quantconnect-api", "QuantConnect cloud API usage")
    _skill(root, "", "gmail-cleanup", "Clean up a Gmail inbox")
    return root


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    root = tmp_path / "vault"
    _note(
        root, "Designs/Memory-Rework.md", "# Memory Rework\n\n## Goals\nFix recall.\n"
    )
    _note(root, "Daily/2026-10-08.md", "---\ntags: [daily]\n---\nToday I did things.\n")
    _note(root, ".trash/old.md", "# gone\n")
    return root


def _item(ref: str, text: str = "x", snap=(1.0, 1)) -> SourceItem:
    st = ref.split(":", 1)[0]
    return SourceItem(ref, st, ref.split(":", 1)[1], text, snap)


def _fake_curate(calls: list):
    """A curator that names one entity per source after its label."""

    async def curate(batch, known):
        calls.append(([it.ref for it in batch], [k["canonical"] for k in known]))
        return [
            CurateResult(
                source_ref=it.ref,
                keywords=[it.label],
                entities=[
                    Entity(
                        canonical=it.label.split("/")[-1].removesuffix(".md").lower(),
                        aliases=["al"],
                    )
                ],
            )
            for it in batch
        ]

    return curate


# --- digest / sources ----------------------------------------------------------


def test_digest_keeps_frontmatter_headings_and_excerpt() -> None:
    raw = (
        "---\ntitle: Thing\ntags: [x, y]\ndescription: A thing\nsecret: no\n---\n"
        "# Top\n\nintro paragraph\n\n```\n# not a heading\n```\n\n## Sub\nmore\n"
    )
    out = digest(raw)
    assert (
        "title: Thing" in out and "tags: x, y" in out and "description: A thing" in out
    )
    assert "secret" not in out
    assert "# Top" in out and "## Sub" in out
    assert (
        "not a heading" not in out.split("excerpt:")[0]
    )  # fenced line is not a heading
    assert "intro paragraph" in out


def test_digest_is_bounded_and_keeps_headings_over_excerpt() -> None:
    raw = "# H1\n" + ("word " * 5000)
    out = digest(raw, max_chars=400)
    assert len(out) <= 400
    assert "# H1" in out


def test_scan_skills_refs_and_snapshots(skills: Path) -> None:
    items = scan_skills([skills])
    assert set(items) == {"skill:trading/quantconnect-api", "skill:gmail-cleanup"}
    it = items["skill:trading/quantconnect-api"]
    assert it.source_type == "skill" and "QuantConnect cloud API" in it.text
    st = (skills / "trading/quantconnect-api/SKILL.md").stat()
    assert it.snapshot == (st.st_mtime, st.st_size)


def test_scan_skills_empty_and_missing() -> None:
    assert scan_skills([]) == {}
    assert scan_skills(["/definitely/not/here"]) == {}


def test_scan_notes_scope_and_titles(vault: Path) -> None:
    items = scan_notes(vault, deny=[".trash/**"])
    assert set(items) == {"note:Designs/Memory-Rework.md", "note:Daily/2026-10-08.md"}
    # a note without a title/name gets its filename stem as a title
    assert "title: 2026-10-08" in items["note:Daily/2026-10-08.md"].text
    only = scan_notes(vault, allow=["Designs/**"], deny=[".trash/**"])
    assert set(only) == {"note:Designs/Memory-Rework.md"}


def test_scan_notes_missing_root(tmp_path: Path) -> None:
    assert scan_notes(tmp_path / "nope") == {}


def test_scan_sources_merges_and_sorts(skills: Path, vault: Path) -> None:
    items = scan_sources(
        skill_libraries=[skills], vault_root=vault, vault_deny=[".trash/**"]
    )
    assert list(items) == sorted(items)
    assert {r.split(":")[0] for r in items} == {"skill", "note"}
    assert scan_sources(skill_libraries=[skills]).keys() == scan_skills([skills]).keys()


# --- batching ------------------------------------------------------------------


def test_make_batches_by_chars_and_count() -> None:
    items = [_item(f"skill:s{i}", "x" * 100) for i in range(7)]
    by_chars = make_batches(items, max_chars=250, max_items=99)
    assert [len(b) for b in by_chars] == [2, 2, 2, 1]
    by_count = make_batches(items, max_chars=10**6, max_items=3)
    assert [len(b) for b in by_count] == [3, 3, 1]
    assert [it.ref for b in by_chars for it in b] == [it.ref for it in items]


def test_make_batches_oversized_item_gets_own_batch() -> None:
    items = [
        _item("skill:a", "x" * 50),
        _item("skill:big", "x" * 5000),
        _item("skill:b"),
    ]
    assert [[i.ref for i in b] for b in make_batches(items, max_chars=100)] == [
        ["skill:a"],
        ["skill:big"],
        ["skill:b"],
    ]
    assert make_batches([]) == []


# --- receipt validation --------------------------------------------------------


def _res(ref: str, *entities: str) -> CurateResult:
    return CurateResult(
        source_ref=ref, entities=[Entity(canonical=e) for e in entities]
    )


def test_check_batch_valid() -> None:
    batch = CurateBatch(results=[_res("a", "x"), _res("b", "y")])
    assert check_batch(batch, ["a", "b"]) == []


def test_check_batch_flags_missing_unknown_duplicate_and_empty() -> None:
    batch = CurateBatch(
        results=[_res("a", "x"), _res("a", "x"), _res("zzz", "y"), _res("c")]
    )
    problems = " | ".join(check_batch(batch, ["a", "b", "c"]))
    assert "answered more than once" in problems
    assert "unknown source_ref 'zzz'" in problems
    assert "missing results for: ['b']" in problems
    assert "'c' has no entities" in problems


def test_models_trim_keywords_entities_and_clean_aliases() -> None:
    r = CurateResult(
        source_ref="a",
        keywords=[f"k{i}" for i in range(10)] + ["  "],
        entities=[Entity(canonical=f"e{i}", aliases=["ok", "  "]) for i in range(8)],
    )
    assert len(r.keywords) == 6 and len(r.entities) == 5
    assert r.entities[0].aliases == ["ok"]
    with pytest.raises(ValueError):
        Entity(canonical="   ")


def test_prompt_carries_known_entities_and_sources() -> None:
    known = [{"canonical": "paf", "aliases": ["the framework"]}]
    prompt = CuratorPrompt([_item("skill:a", "BODYTEXT")], known).render()
    assert "- paf (aliases: the framework)" in prompt
    assert "[skill:a] (skill) a" in prompt and "BODYTEXT" in prompt
    assert "(none yet)" in CuratorPrompt([_item("skill:a")], []).render()
    assert format_known([{"canonical": "x", "aliases": []}]) == "- x"


# --- the agent (FunctionModel) -------------------------------------------------


def _model_returning(*payloads: dict):
    """A FunctionModel that returns each payload as the structured output in turn.

    ``run_guarded`` streams the agent, so the model needs a ``stream_function``
    too; both pick the next payload the same way.
    """
    seq = list(payloads)
    seen: list[str] = []

    def _next(messages) -> dict:
        seen.append(str(messages[-1]))
        return seq.pop(0) if len(seq) > 1 else seq[0]

    def call(messages, info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart(tool_name=info.output_tools[0].name, args=_next(messages))
            ]
        )

    async def stream(messages, info: AgentInfo):
        yield {
            0: DeltaToolCall(
                name=info.output_tools[0].name, json_args=json.dumps(_next(messages))
            )
        }

    return FunctionModel(call, stream_function=stream), seen


@_sync
async def test_curate_batch_returns_valid_results() -> None:
    good = {
        "results": [
            {
                "source_ref": "skill:a",
                "keywords": ["lean-cli"],
                "entities": [{"canonical": "quantconnect", "aliases": ["qc"]}],
            }
        ]
    }
    model, _ = _model_returning(good)
    out = await curate_batch([_item("skill:a")], [], model=model)
    assert [r.source_ref for r in out] == ["skill:a"]
    assert out[0].entities[0].aliases == ["qc"]


@_sync
async def test_curate_batch_retries_until_every_source_answered() -> None:
    bad = {"results": []}  # answers nothing -> validator forces a retry
    good = {
        "results": [
            {"source_ref": "skill:a", "keywords": [], "entities": [{"canonical": "x"}]}
        ]
    }
    model, seen = _model_returning(bad, good)
    out = await curate_batch([_item("skill:a")], [], model=model)
    assert [r.source_ref for r in out] == ["skill:a"]
    assert len(seen) == 2  # the first answer was rejected, the model was re-asked
    assert "missing results" in seen[1]


# --- worker pass ---------------------------------------------------------------


def _sources(skills: Path, vault: Path):
    return scan_sources(
        skill_libraries=[skills], vault_root=vault, vault_deny=[".trash/**"]
    )


@_sync
async def test_first_pass_keys_everything_and_builds_map(
    tmp_path: Path, skills: Path, vault: Path
) -> None:
    store = tmp_path / "kw"
    calls: list = []
    res = await run_keyword_pass(
        store, _sources(skills, vault), curate=_fake_curate(calls)
    )
    assert res.ok and len(res.keyed) == 4 and res.entities == 4
    assert KeywordJournalStore(store).keyed_refs() == set(_sources(skills, vault))
    m = read_entity_map(store)
    assert set(m) == {
        "quantconnect-api",
        "gmail-cleanup",
        "memory-rework",
        "2026-10-08",
    }
    assert m["gmail-cleanup"]["sources"] == ["skill:gmail-cleanup"]
    wm = KeywordWatermark.load(store)
    assert set(wm.files) == set(_sources(skills, vault))


@_sync
async def test_second_pass_is_a_noop(tmp_path: Path, skills: Path, vault: Path) -> None:
    store = tmp_path / "kw"
    src = _sources(skills, vault)
    await run_keyword_pass(store, src, curate=_fake_curate([]))
    calls: list = []
    res = await run_keyword_pass(store, src, curate=_fake_curate(calls))
    assert calls == [] and res.keyed == [] and res.batches == 0
    assert res.entities == 4


@_sync
async def test_only_changed_sources_are_recurated(
    tmp_path: Path, skills: Path, vault: Path
) -> None:
    store = tmp_path / "kw"
    await run_keyword_pass(store, _sources(skills, vault), curate=_fake_curate([]))
    f = skills / "gmail-cleanup" / "SKILL.md"
    f.write_text(f.read_text() + "\nMore text so the size changes.\n")
    calls: list = []
    res = await run_keyword_pass(
        store, _sources(skills, vault), curate=_fake_curate(calls)
    )
    assert res.keyed == ["skill:gmail-cleanup"]
    assert [c[0] for c in calls] == [["skill:gmail-cleanup"]]
    # same curated content -> journal append was idempotent (no duplicate line)
    j = KeywordJournalStore(store).read_all()
    assert sum(1 for r in j if r["source_ref"] == "skill:gmail-cleanup") == 1


@_sync
async def test_vanished_source_leaves_the_map(
    tmp_path: Path, skills: Path, vault: Path
) -> None:
    store = tmp_path / "kw"
    await run_keyword_pass(store, _sources(skills, vault), curate=_fake_curate([]))
    (skills / "gmail-cleanup" / "SKILL.md").unlink()
    res = await run_keyword_pass(
        store, _sources(skills, vault), curate=_fake_curate([])
    )
    assert res.removed == ["skill:gmail-cleanup"] and res.keyed == []
    assert "gmail-cleanup" not in read_entity_map(store)
    assert "skill:gmail-cleanup" not in KeywordWatermark.load(store).files
    # the journal is lossless: the line is still there, just not live
    assert "skill:gmail-cleanup" in KeywordJournalStore(store).keyed_refs()


@_sync
async def test_later_batches_see_earlier_entities(tmp_path: Path) -> None:
    items = {f"skill:s{i}": _item(f"skill:s{i}", "x" * 100) for i in range(4)}
    calls: list = []
    await run_keyword_pass(
        tmp_path / "kw", items, curate=_fake_curate(calls), max_batch_items=2
    )
    assert len(calls) == 2
    assert calls[0][1] == []  # nothing known yet
    assert set(calls[1][1]) == {"s0", "s1"}  # drift prevention input


@_sync
async def test_failed_batch_is_not_advanced_and_retries(tmp_path: Path) -> None:
    items = {f"skill:s{i}": _item(f"skill:s{i}", "x" * 100) for i in range(4)}
    store = tmp_path / "kw"
    ok = _fake_curate([])
    state = {"n": 0}

    async def flaky(batch, known):
        state["n"] += 1
        if state["n"] == 2:
            raise RuntimeError("model fell over")
        return await ok(batch, known)

    res = await run_keyword_pass(store, items, curate=flaky, max_batch_items=2)
    assert not res.ok
    assert res.keyed == ["skill:s0", "skill:s1"]
    assert set(res.failed) == {"skill:s2", "skill:s3"}
    assert "RuntimeError" in res.failed["skill:s2"]
    assert set(KeywordWatermark.load(store).files) == {"skill:s0", "skill:s1"}
    # next pass retries only the failures
    calls: list = []
    res2 = await run_keyword_pass(
        store, items, curate=_fake_curate(calls), max_batch_items=2
    )
    assert res2.ok and sorted(res2.keyed) == ["skill:s2", "skill:s3"]
    assert [c[0] for c in calls] == [["skill:s2", "skill:s3"]]


@_sync
async def test_missing_answer_for_a_source_is_a_failure(tmp_path: Path) -> None:
    items = {"skill:a": _item("skill:a"), "skill:b": _item("skill:b")}

    async def drops_b(batch, known):
        return [_res("skill:a", "a")]

    res = await run_keyword_pass(tmp_path / "kw", items, curate=drops_b)
    assert res.keyed == ["skill:a"]
    assert "no result" in res.failed["skill:b"]


@_sync
async def test_limit_caps_a_trial_pass(tmp_path: Path) -> None:
    items = {f"skill:s{i}": _item(f"skill:s{i}") for i in range(5)}
    res = await run_keyword_pass(
        tmp_path / "kw", items, curate=_fake_curate([]), limit=2
    )
    assert res.keyed == ["skill:s0", "skill:s1"]
    # the rest are still pending
    wm = KeywordWatermark.load(tmp_path / "kw")
    assert wm.changed(snapshots(items)) == ["skill:s2", "skill:s3", "skill:s4"]


@_sync
async def test_map_is_rebuilt_when_missing(tmp_path: Path) -> None:
    items = {"skill:a": _item("skill:a")}
    store = tmp_path / "kw"
    await run_keyword_pass(store, items, curate=_fake_curate([]))
    os.unlink(store / "entity-map.json")
    res = await run_keyword_pass(store, items, curate=_fake_curate([]))
    assert res.keyed == [] and (store / "entity-map.json").exists()
    assert json.loads((store / "entity-map.json").read_text())


# --- report --------------------------------------------------------------------


def _emap(**ents: dict) -> dict:
    return {
        k: {
            "canonical": k,
            "aliases": v.get("aliases", []),
            "sources": v.get("sources", ["s"]),
        }
        for k, v in ents.items()
    }


def test_report_flags_ambiguous_aliases() -> None:
    m = _emap(
        memory={"aliases": ["the notebook"]}, journal={"aliases": ["The Notebook"]}
    )
    assert find_issues(m)["ambiguous_terms"] == {"the notebook": ["journal", "memory"]}


def test_report_flags_alias_equal_to_other_canonical() -> None:
    m = _emap(paf={"aliases": ["framework"]}, framework={})
    assert "framework" in find_issues(m)["ambiguous_terms"]


def test_report_flags_similar_and_containing_names() -> None:
    m = _emap(
        **{
            "quantconnect": {},
            "quant-connect": {},
            "memory": {},
            "memory-extractor": {},
        }
    )
    pairs = {(a, b) for a, b, _ in find_issues(m)["similar_names"]}
    assert ("quant-connect", "quantconnect") in pairs or (
        "quantconnect",
        "quant-connect",
    ) in pairs
    assert ("memory", "memory-extractor") in pairs


def test_report_counts_singletons_and_alias_free() -> None:
    m = _emap(a={"aliases": ["x"], "sources": ["1", "2"]}, b={})
    out = find_issues(m)
    assert out["single_source"] == ["b"] and out["no_aliases"] == ["b"]
    assert out["entities"] == 2


def test_report_flags_split_entities_that_share_sources() -> None:
    m = _emap(
        gws={"sources": ["skill:gmail", "skill:google-workspace"]},
        **{"google-workspace": {"sources": ["skill:google-workspace"]}},
        quantconnect={"sources": [f"skill:q{i}" for i in range(8)]},
        composer={"sources": ["skill:q0"]},
    )
    pairs = {frozenset((a, b)) for a, b, _ in find_issues(m)["overlapping_sources"]}
    assert frozenset(("gws", "google-workspace")) in pairs
    # a small entity inside a big one is normal, not a duplicate
    assert frozenset(("composer", "quantconnect")) not in pairs


# --- project seeds -------------------------------------------------------------


class _FakeProjects:
    def __init__(self, rows):
        self._rows = rows

    def list(self):
        return self._rows


ROWS = [
    {"slug": "local-llm", "name": "local llm", "description": "GPU rig notes."},
    {"slug": "lmem", "name": "LongMemEval", "aliases": ["lme"], "description": ""},
    {"slug": "paf", "name": "paf", "description": "The framework."},
]


def test_project_seeds_shape() -> None:
    seeds = {s["canonical"]: s for s in project_seeds(_FakeProjects(ROWS))}
    assert set(seeds) == {"local-llm", "lmem", "paf"}
    assert seeds["local-llm"]["aliases"] == []  # name == slug once normalized
    assert seeds["lmem"]["aliases"] == ["LongMemEval", "lme"]
    assert seeds["paf"]["description"] == "The framework."
    assert all(s["project"] for s in seeds.values())


def test_format_known_marks_projects_with_description() -> None:
    out = format_known(project_seeds(_FakeProjects(ROWS)))
    assert "- local-llm [project]: GPU rig notes." in out
    assert "- lmem [project] (aliases: LongMemEval, lme)" in out
    assert "[project]" not in format_known([{"canonical": "x", "aliases": []}])


def test_known_entities_seeds_first_and_fold_matching_derived(tmp_path: Path) -> None:
    store = KeywordJournalStore(tmp_path)
    for rec in [
        make_record(
            "skill:a", "skill", [], [{"canonical": "paf", "aliases": ["the framework"]}]
        ),
        make_record(
            "skill:b", "skill", [], [{"canonical": "todoist", "aliases": ["td"]}]
        ),
    ]:
        store.append(rec)
    out = known_entities(
        store, {"skill:a", "skill:b"}, seeds=project_seeds(_FakeProjects(ROWS))
    )
    names = [e["canonical"] for e in out]
    assert names[:3] == ["local-llm", "lmem", "paf"]  # seeds first, in registry order
    assert names[3:] == ["todoist"]  # derived non-project entities after
    paf = next(e for e in out if e["canonical"] == "paf")
    assert paf["description"] == "The framework." and paf.get("project")
    assert "the framework" in paf["aliases"] and paf["sources"] == ["skill:a"]


def test_known_entities_cap_never_drops_seeds(tmp_path: Path) -> None:
    store = KeywordJournalStore(tmp_path)
    for i in range(5):
        store.append(make_record(f"skill:{i}", "skill", [], [{"canonical": f"e{i}"}]))
    live = {f"skill:{i}" for i in range(5)}
    out = known_entities(
        store, live, max_known=2, seeds=project_seeds(_FakeProjects(ROWS))
    )
    assert [e["canonical"] for e in out] == ["local-llm", "lmem", "paf"]


@_sync
async def test_seeds_reach_the_first_batch(tmp_path: Path) -> None:
    calls: list = []
    await run_keyword_pass(
        tmp_path / "kw",
        {"skill:a": _item("skill:a")},
        curate=_fake_curate(calls),
        seeds=project_seeds(_FakeProjects(ROWS)),
    )
    assert calls[0][1] == ["local-llm", "lmem", "paf"]
    # without seeds the first batch sees nothing (existing behaviour)
    calls2: list = []
    await run_keyword_pass(
        tmp_path / "kw2", {"skill:a": _item("skill:a")}, curate=_fake_curate(calls2)
    )
    assert calls2[0][1] == []
