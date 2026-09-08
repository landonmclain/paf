---
name: code-review
description: Review a code change for correctness, quality, and repository conventions before commit.
tags: [review, code, quality, pr, conventions]
---

# Skill: code-review

Perform a review of a code change before it is committed.

1. Read `git diff` for the change (use the git capability if present).
2. Check correctness: logic errors, edge cases, off-by-one, error handling.
3. Check conventions: naming, formatting, import order, dead code.
4. Report findings as a prioritized list: blockers first, then suggestions.

When in doubt, prefer a conservative review — flag anything you cannot verify.
