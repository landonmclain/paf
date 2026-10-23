# sample_repo

A small, purpose-built codebase that the codebase-map tests scan. Each file
exists to exercise one analyzer behavior; the tests name which. Keep it small,
and update the tests' expectations when you change it.

- `shop/config.py` is the cross-package hub: called from every package.
- `shop/*/_store.py` share one basename on purpose. Only `billing/_store.py`
  defines a module variable named `_store`, imported by its sibling `ledger.py`;
  no other file may resolve `_store` as an imported symbol.
- `shop/reports/` is coupled only to itself, and uses `OrderStore` in type hints
  only (a reference edge, never a call).
- `web/` is TypeScript: module-level `export const` tables, a typed config
  object, and a component whose imports must not leak path segments.
