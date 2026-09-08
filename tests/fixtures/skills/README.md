# Test fixture: skills

A **deterministic, checked-in** skill library used by `tests/test_skills.py` as
the corpus for the paf `Skills` capability.

## Why this exists

`data/skills/` is the **real running skill library** for the demo agent — it's
gitignored (see `.gitignore` `data` rule) and free to grow/shrink on any machine.
Tests must not index it:

- On a fresh clone `data/skills/` won't exist (gitignored), so a test pinned to
  it would fail before any code runs.
- On a developer machine it can hold any number of skills, so tests asserting an
  exact count would be brittle.

These two SKILL.md files (`code-review`, `vectorized-backtest`) are the original
demo pair, frozen here so the corpus/stoolset tests are stable and hermetic.

## Layout

- `code-review/SKILL.md`          — a skill with a `tags:` list in frontmatter.
- `vectorized-backtest/SKILL.md`  — a second skill (used for ranking checks).

Both carry `name` + `description` + inline `tags: [a, b]` frontmatter — the exact
shape `HintCorpus` / `parse_frontmatter` exercise.
