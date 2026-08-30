# code_search_eval fixtures — real-code gold corpora

Two real-code corpora feed the codebase-map harness. Both are **gitignored** and
re-materialized on demand (the extraction / graph / map tests **skip** when a
fixture is absent, so the suite stays green on a clean checkout).

---

## 1. chatbot fixture — real TS/TSX/JS corpus

Real-world TypeScript/React corpus from [vercel/chatbot](https://github.com/vercel/chatbot),
used as the non-Python extraction gate fixture (TS/TSX/JS pass).

- **Pinned commit:** `c2f8235e1f3ea903ad8b7f61447c4f74164b5c58`
- **Code files:** 155 (93 `.tsx`, 60 `.ts`, 1 `.js`, 1 `.mjs`)
- **Repo URL:** `https://github.com/vercel/chatbot.git`

Gitignored (`tests/fixtures/chatbot/`). To re-materialize after a clean checkout:

```bash
mkdir -p tests/fixtures
git clone --depth 1 https://github.com/vercel/chatbot.git tests/fixtures/chatbot
git -C tests/fixtures/chatbot checkout c2f8235e1f3ea903ad8b7f61447c4f74164b5c58
```

---

## 2. paf-src fixture — real Python corpus (this repo's own `src/paf`)

A snapshot of this repo's own `src/paf/` used as the **Python graph/map gold
corpus** — the capability maps its host code, so the ground truth is code Landon
knows. Covers an honest cross-section of real cross-module references: toolset
surface managers (`_toolset.py`), runtime/bootstrap, storage seams, memory
journal, config — the kinds of files a call graph must resolve.

- **Current shape:** 80 `.py` files, ~984K, 66 in-scope scan files / 906 symbols
  (as of 2026-08-29 scan of `src/paf/`; only `src/paf/` sub-tree counts — top-level
  `tests/`, `specs/`, `ui/` are excluded by the fixture copy).
- **Gitignored** (`tests/fixtures/paf-src/`). Re-materialize (snapshot, excludes
  generated artifacts):

```bash
mkdir -p tests/fixtures/paf-src
rsync -a --exclude='__pycache__' --exclude='.agent-codeindex' --exclude='*.pyc' \
  src/paf/ tests/fixtures/paf-src/
```

The copy intentionally excludes `__pycache__` / `.agent-codeindex` so it is pure
source. Snapshot it whenever the gold corpus needs to track a refactor; the graph
and repo-map tests assert against this frozen copy, not the live `src/`.

