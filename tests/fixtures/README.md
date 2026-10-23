# Test fixtures

Small, committed corpora the tests run against. Everything here is checked in
and deterministic, so the tests that use it run the same locally and in CI.

- **`sample_repo/`** — a tiny Python + TypeScript codebase for the codebase-map
  tests (`tests/test_codebase_map_*.py`). Each file exists to exercise one
  analyzer behavior; its README lists them. When you change it, update the
  tests' expectations to match.
- **`skills/`** — a skill library for `tests/test_skills.py` (see its README).
