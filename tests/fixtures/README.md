# chatbot fixture (code_search_eval — real TS/TSX corpus)

Real-world TypeScript/React corpus from [vercel/chatbot](https://github.com/vercel/chatbot),
used as the non-Python extraction gate fixture (TS/TSX/JS pass).

- **Pinned commit:** `c2f8235e1f3ea903ad8b7f61447c4f74164b5c58`
- **Code files:** 155 (93 `.tsx`, 60 `.ts`, 1 `.js`, 1 `.mjs`)
- **Repo URL:** `https://github.com/vercel/chatbot.git`

The clone itself is **gitignored** (`tests/fixtures/chatbot/` in repo root `.gitignore`).
To re-materialize after a clean checkout:

```bash
mkdir -p tests/fixtures
git clone --depth 1 https://github.com/vercel/chatbot.git tests/fixtures/chatbot
git -C tests/fixtures/chatbot checkout c2f8235e1f3ea903ad8b7f61447c4f74164b5c58
```

The gate should **skip** (like the python gate skips on missing gold) if the fixture
is absent, so the suite stays green on a clean checkout.
