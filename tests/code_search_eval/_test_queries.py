"""Verify _queries.py recovery using REAL spans from process().structure."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tree_sitter_language_pack import ProcessConfig, process

from paf.capabilities.codebase_map._queries import QueryRecovery
from paf.capabilities.codebase_map._schema import Span

CASES = {
    "python": 'def add(a: int, b: int) -> int:\n    """Add two ints."""\n    return a + b\n',
    "typescript": "/** ts doc */\nexport function greet(u: User): string { return 'x'; }\n",
    "tsx": "/** button doc */\nfunction Button({label}: Props): JSX { return <div>{label}</div>; }\n",
    "javascript": "function greet(user) { /** Hi */ return user; }\n",
    "php": "<?php\n/** php doc */\nfunction add(int $a, int $b): int {\n    return $a + $b;\n}\n",
    "go": "package main\n// go doc\nfunc Add(a, b int) int { return a + b }\n",
    "rust": "/// rust doc\npub fn add(a: i32, b: i32) -> i32 { a + b }\n",
    "java": "public class Main {\n    /** java doc */\n    public int add(int a, int b) { return a + b; }\n}\n",
    "c": "/* c doc */\nint add(int a, int b) { return a + b; }\n",
}


def find_first_fn(res):
    """Return the first function-ish structure item with real byte spans."""

    def walk(items):
        for it in items:
            kind = getattr(it.kind, "value", str(it.kind))
            if kind in ("Function", "Method") and it.name:
                return it
            r = walk(it.children)
            if r:
                return r
        return None

    return walk(res.structure)


for lang, src in CASES.items():
    res = process(src, ProcessConfig(language=lang, structure=True, symbols=True))
    fn = find_first_fn(res)
    print(f"\n{lang}:")
    if fn is None:
        print(
            "  no function symbol found in structure; structure=",
            [(s.name, s.kind) for s in res.structure],
        )
        continue
    span = Span(
        start_byte=fn.span.start_byte,
        end_byte=fn.span.end_byte,
        start_line=fn.span.start_line,
        end_line=fn.span.end_line,
    )
    body = None
    if fn.body_span is not None:
        body = Span(
            start_byte=fn.body_span.start_byte,
            end_byte=fn.body_span.end_byte,
            start_line=fn.body_span.start_line,
            end_line=fn.body_span.end_line,
        )
    r = QueryRecovery(language=lang, source=src)
    try:
        print(f"  process signature/visibility: {fn.signature!r} / {fn.visibility!r}")
        print(f"  recovered sig: {r.signature_for(span, body)!r}")
        print(f"  recovered doc: {r.docstring_for(span, body)!r}")
    except Exception as e:  # noqa: BLE001
        print(f"  ERROR {type(e).__name__}: {e}")
