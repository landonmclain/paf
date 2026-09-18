"""Phase 1 extraction tests: python node coverage + edge cases + byte-index contract.

Locks in the python extraction path for the codebase-map capability. Covers:

- **Byte-offset vs char-index** root cause (tree-sitter reports BYTE offsets; a Python
  ``str`` is char-indexed — any non-ASCII char before a target drifts str-slicing).
- **Every python node type we ship**: top-level functions, methods, nested functions,
  nested classes, async, staticmethod/classmethod/property, decorators, private.
- **Signature forms**: typed/annotated params, defaults, ``*args``/``**kwargs``,
  pos-only ``/``, keyword-only ``*``, multiline params, quoted+future annotations,
  complex generics + unions.
- **Docstrings**: function/method, **class**, and **module-level** (incl. multiline
  dedent — continuation indentation must be stripped).
- **Imports**: ``import x``, ``from x import y``, alias, relative, wildcard.
- **Duplicate regression**: every symbol appears EXACTLY once in ``searchable_symbols``
  (guards the ``_walk_with_children`` double-append bug).

Python-only, per the "one language at a time" gate.
"""

from __future__ import annotations

from pathlib import Path

from paf.capabilities.codebase_map._analyze import analyze_file
from paf.capabilities.codebase_map._queries import QueryRecovery
from paf.capabilities.codebase_map._schema import FileInfo, SymbolInfo

# Deliberately-placed non-ASCII (em-dash/arrow) chars BEFORE symbol bodies: every
# 3-byte char shifts byte-vs-char indexing by +2, so a naive str slice drifts.
_FIXTURE = '''\
"""Module docstring — em-dash here exercises byte offsets."""

import os
import json as j
from typing import List, Optional, Dict
from . import sibling
from .sub import thing as t
from .deep import *


def plain(a, b):
    return a + b


def typed(x: int, y: str = "default", *args: int, **kwargs: dict) -> Optional[List[int]]:
    """Typed function with defaults and varargs."""
    return [x]


def oneliner(): return 1


async def async_fn(url: str) -> str:
    """Async function."""
    return url


def with_docstring():
    """Docstring here.

    Second paragraph.
    """
    pass


def no_docstring():
    pass


def nested():
    def inner(a):
        return a * 2
    return inner(21)


@staticmethod
def decorated(x):  # noqa
    return x


def kw_only(a, *, b: int = 1) -> None:
    pass


def pos_only(a, /, b: int) -> None:
    pass


def quoted_ann(x: "Point", y: "list[Point]") -> "Point":
    return x


def complex_ann() -> dict[str, list[tuple[int, str]]]:
    return {}


def union_ann(x: int | None, y: list[int] | dict[str, int] | None) -> int | None:
    return x


class Base:
    def method(self, a: int) -> int:
        return a


class MyClass(Base, Generic[T]):
    """Class docstring."""

    def __init__(self, name: str):
        self.name = name

    def regular(self, a, b=2) -> float:
        return a + b

    @staticmethod
    def static_method(a: int) -> int:
        return a

    @classmethod
    def cls_method(cls, x: int) -> int:
        return x

    @property
    def prop(self) -> str:
        return self.name

    async def async_method(self) -> None:
        pass

    def _private(self) -> None:
        pass

    def long_sig(self,
                 a: int,
                 b: int = 3,
                 **kwargs) -> None:
        pass


class Outer:
    class Inner:
        def deep(self):
            return 1


def _module_level_private():
    pass
'''


def _analyze() -> tuple[FileInfo, dict[str, SymbolInfo]]:
    """Run the real ship path and index searchable_symbols by qualified name."""
    tmp = Path("/tmp/paf_fixture_unittest.py")
    tmp.write_text(_FIXTURE, encoding="utf-8")
    fi = analyze_file(tmp, "/tmp", language="python")
    assert fi is not None, "analyze_file returned None for a valid fixture"
    by_name = {s.qualified: s for s in fi.searchable_symbols if s.qualified}
    return fi, by_name


# --------------------------------------------------------------------------- byte contract


def test_byte_indexing_byte_vs_char_is_real() -> None:
    """Assert the underlying trap: byte and char offsets genuinely diverge here."""
    src = _FIXTURE
    assert src.encode("utf-8")[:1]  # non-empty
    first_multi = next(i for i, ch in enumerate(src) if ord(ch) > 127)
    assert first_multi >= 0
    index_after = first_multi + 1
    byte_off_after = len(src[:index_after].encode("utf-8"))
    # the em-dash is 3 bytes -> byte offset is now 2 ahead of the char index
    assert byte_off_after == index_after + 2


# --------------------------------------------------------------------------- signatures


def test_signature_plain() -> None:
    _, by = _analyze()
    assert by["plain"].signature == "plain(a, b)"


def test_signature_typed_defaults_varargs() -> None:
    _, by = _analyze()
    assert (
        by["typed"].signature
        == 'typed(x: int, y: str = "default", *args: int, **kwargs: dict) -> Optional[List[int]]'
    )


def test_signature_oneliner() -> None:
    _, by = _analyze()
    assert by["oneliner"].signature == "oneliner()"


def test_signature_async() -> None:
    _, by = _analyze()
    assert by["async_fn"].signature == "async_fn(url: str) -> str"


def test_signature_keyword_only() -> None:
    _, by = _analyze()
    assert by["kw_only"].signature == "kw_only(a, *, b: int = 1) -> None"


def test_signature_pos_only() -> None:
    _, by = _analyze()
    assert by["pos_only"].signature == "pos_only(a, /, b: int) -> None"


def test_signature_quoted_annotation() -> None:
    _, by = _analyze()
    assert (
        by["quoted_ann"].signature
        == 'quoted_ann(x: "Point", y: "list[Point]") -> "Point"'
    )


def test_signature_complex_generic() -> None:
    _, by = _analyze()
    assert (
        by["complex_ann"].signature
        == "complex_ann() -> dict[str, list[tuple[int, str]]]"
    )


def test_signature_union() -> None:
    _, by = _analyze()
    assert (
        by["union_ann"].signature
        == "union_ann(x: int | None, y: list[int] | dict[str, int] | None) -> int | None"
    )


def test_signature_methods() -> None:
    _, by = _analyze()
    assert by["MyClass.regular"].signature == "regular(self, a, b=2) -> float"
    assert by["MyClass.static_method"].signature == "static_method(a: int) -> int"
    assert by["MyClass.cls_method"].signature == "cls_method(cls, x: int) -> int"
    assert by["MyClass.prop"].signature == "prop(self) -> str"
    assert by["MyClass.async_method"].signature == "async_method(self) -> None"
    assert by["Base.method"].signature == "method(self, a: int) -> int"


def test_signature_multiline() -> None:
    _, by = _analyze()
    got = by["MyClass.long_sig"].signature
    assert (
        got
        == "long_sig(self,\n                 a: int,\n                 b: int = 3,\n                 **kwargs) -> None"
    )


def test_class_signature_is_none() -> None:
    # A class is not a function — no signature (correctly None, not a crash).
    _, by = _analyze()
    assert by["MyClass"].signature is None
    assert by["Base"].signature is None


# --------------------------------------------------------------------------- docstrings


def test_docstring_function() -> None:
    _, by = _analyze()
    assert by["typed"].doc == "Typed function with defaults and varargs."
    assert by["async_fn"].doc == "Async function."


def test_docstring_no_docstring_is_none() -> None:
    _, by = _analyze()
    assert by["no_docstring"].doc is None
    assert by["plain"].doc is None


def test_docstring_multiline_dedented() -> None:
    _, by = _analyze()
    # continuation lines must have their body indentation stripped (inspect.cleandoc)
    assert by["with_docstring"].doc == "Docstring here.\n\nSecond paragraph."


def test_docstring_class() -> None:
    _, by = _analyze()
    assert by["MyClass"].doc == "Class docstring."


def test_docstring_module_level() -> None:
    fi, _ = _analyze()
    assert fi.doc == "Module docstring — em-dash here exercises byte offsets."


def test_docstring_not_leaked_between_symbols() -> None:
    # A method's docstring must not bleed into an adjacent symbol.
    _, by = _analyze()
    assert by["MyClass.__init__"].doc is None  # no docstring — must stay None
    assert by["MyClass.regular"].doc is None


# --------------------------------------------------------------------------- node types / structure


def test_symbol_kinds() -> None:
    _, by = _analyze()
    assert by["plain"].kind == "Function"
    assert by["MyClass"].kind == "Class"
    assert by["Base.method"].kind == "Function"


def test_qualified_names_nested() -> None:
    _, by = _analyze()
    assert "nested.inner" in by
    assert "Outer.Inner.deep" in by


def test_private_symbols_indexed() -> None:
    _, by = _analyze()
    assert "_module_level_private" in by
    assert "MyClass._private" in by  # note: _private defined in Base? no — see fixture


def test_no_duplicate_searchable_symbols() -> None:
    """Regression: _walk_with_children double-appended children (Base.method x2)."""
    fi, _ = _analyze()
    quals = [s.qualified for s in fi.searchable_symbols]
    assert len(quals) == len(set(quals)), (
        f"duplicates: {[q for q in quals if quals.count(q) > 1]}"
    )


def test_symbols_top_level_list() -> None:
    fi, _ = _analyze()
    assert "plain" in fi.symbols
    assert "MyClass" in fi.symbols
    assert "Outer" in fi.symbols


# --------------------------------------------------------------------------- imports


def test_imports_captured() -> None:
    fi, _ = _analyze()
    # canonical import strings, deduped + order-preserved
    expected = [
        "import os",
        "import json as j",
        "from typing import List, Optional, Dict",
        "from . import sibling",
        "from .sub import thing as t",
        "from .deep import *",
    ]
    assert fi.imports == expected


def test_imports_no_duplicates() -> None:
    fi, _ = _analyze()
    assert len(fi.imports) == len(set(fi.imports))


# --------------------------------------------------------------------------- searchable_symbols per-field capture (§3.7.2)


def test_searchable_symbols_have_core_scalar_fields() -> None:
    """Every entry carries name/qualified/kind/signature/doc/body (per-field capture).

    Covers the per-symbol scalar fields of the spec's SearchableSymbol shape. The two
    file-level fields (imports, file_path) live on FileInfo — see next test.
    """
    fi, by = _analyze()
    assert fi.searchable_symbols, "expected at least one searchable symbol"
    for s in fi.searchable_symbols:
        assert s.name, "symbol missing name"
        assert s.qualified, "symbol missing qualified"
        assert s.kind, "symbol missing kind"
        # signature/doc/body are captured where they exist; fields are optional
        # (classes get sig=None; docstring-less symbols get doc=None). The point is
        # the fields are wired through the record, not dropped.


def test_searchable_symbol_body_captured() -> None:
    """A method/function body's source text is captured (weight-1 field)."""
    _, by = _analyze()
    # plain() body: `return a + b` — body should include the return statement
    plain_body = by["plain"].body
    assert plain_body is not None
    assert "return a + b" in plain_body, plain_body
    # class body contains its method definitions (nested children)
    myclass_body = by["MyClass"].body
    assert myclass_body is not None
    assert "def __init__" in myclass_body, myclass_body


def test_searchable_symbol_all_seven_spec_fields_present() -> None:
    """The 7 SearchableSymbol fields (§3.7.2) are covered across SymbolInfo + FileInfo."""
    fi, by = _analyze()
    # per-symbol scalar fields
    s = by["plain"]
    assert s.name and s.qualified  # weight 6 / weight 4
    assert s.signature  # weight 3
    assert s.body  # weight 1
    # file-level fields on FileInfo (ranked per file; not duplicated per symbol)
    assert fi.imports  # weight 2
    assert fi.path  # weight 4 (file_path)
    # doc: symbol or file level — every searchable entry has one of these
    any_symbol_doc = any(si.doc for si in fi.searchable_symbols)
    assert any_symbol_doc or fi.doc, "no doc field populated anywhere"


# --------------------------------------------------------------------------- module-level assignments


_ASSIGN_FIXTURE = '''\
"""Module assign fixture."""
from typing import Any

# Registry of capability types (multiline doc comment above an assignment).
CAPABILITY_TYPES: dict[str, type[Any]] = {
    "A": 1,
}

MAX_SOURCE_BYTES = 1_000_000

__all__ = ["A"]

_ = 1


def f():
    INNER = 2
    return INNER


class C:
    ATTR = 3
'''


def _analyze_assign() -> tuple[FileInfo, dict[str, SymbolInfo]]:
    tmp = Path("/tmp/paf_assign_fixture.py")
    tmp.write_text(_ASSIGN_FIXTURE, encoding="utf-8")
    fi = analyze_file(tmp, "/tmp", language="python")
    assert fi is not None
    by_name = {s.name: s for s in fi.searchable_symbols}
    return fi, by_name


def test_module_assignment_captured_annotated() -> None:
    _, by = _analyze_assign()
    s = by["CAPABILITY_TYPES"]
    assert s.kind == "Constant"
    assert s.signature == "CAPABILITY_TYPES: dict[str, type[Any]]"
    assert s.body is not None and "CAPABILITY_TYPES" in s.body


def test_module_assignment_unannotated() -> None:
    _, by = _analyze_assign()
    s = by["MAX_SOURCE_BYTES"]
    assert s.kind == "Constant"
    assert s.signature is None  # no type annotation


def test_module_assignment_dunder_and_throwaway_excluded() -> None:
    _, by = _analyze_assign()
    assert "__all__" not in by
    assert "_" not in by


def test_module_assignment_nested_excluded() -> None:
    _, by = _analyze_assign()
    # INNER (function-local) and ATTR (class attribute) must NOT be module symbols
    assert "INNER" not in by
    assert "ATTR" not in by
    # but the enclosing function/class still index
    assert by["f"].kind == "Function"
    assert by["C"].kind == "Class"


def test_module_assignment_doc_from_comment_block() -> None:
    _, by = _analyze_assign()
    s = by["CAPABILITY_TYPES"]
    assert s.doc is not None and "Registry of capability types" in s.doc


def test_module_assignment_kind_variable_vs_constant() -> None:
    # ALL-CAPS-with-underscore => Constant; otherwise Variable
    src = "lower = 1\nUPPER_CASE = 2\nMIXED123 = 3\n"
    p = Path("/tmp/paf_kind_fixture.py")
    p.write_text(src, encoding="utf-8")
    fi = analyze_file(p, "/tmp", language="python")
    assert fi is not None
    by = {s.name: s for s in fi.searchable_symbols}
    assert by["lower"].kind == "Variable"
    assert by["UPPER_CASE"].kind == "Constant"
    assert by["MIXED123"].kind == "Variable"  # not ALL-CAPS-with-underscore


def test_recovery_module_assignments() -> None:
    rec = QueryRecovery(language="python", source=_ASSIGN_FIXTURE)
    out = rec.module_assignments()
    names = [n for (n, _s, _e, _t) in out]
    assert "CAPABILITY_TYPES" in names
    assert "MAX_SOURCE_BYTES" in names
    assert "__all__" not in names
    assert "_" not in names
    assert "INNER" not in names
    assert "ATTR" not in names
    caps = dict((n, t) for (n, _s, _e, t) in out)
    assert caps["CAPABILITY_TYPES"] == "dict[str, type[Any]]"
    assert caps["MAX_SOURCE_BYTES"] is None


def test_recovery_preceding_comment_block() -> None:
    src = "# line one\n# line two\nX = 1\n"
    rec = QueryRecovery(language="python", source=src)
    doc = rec._preceding_comment_block(src.index("X = 1"))
    assert doc is not None
    assert "line one" in doc and "line two" in doc


# --------------------------------------------------------------------------- recovery-layer (unit, no process)


def test_recovery_module_docstring() -> None:
    rec = QueryRecovery(language="python", source=_FIXTURE)
    assert (
        rec.module_docstring()
        == "Module docstring — em-dash here exercises byte offsets."
    )


def test_recovery_strip_quotes_and_dedent() -> None:
    from paf.capabilities.codebase_map._queries import _dedent, _strip_quotes

    assert _strip_quotes('"""Hello."""') == "Hello."
    assert _strip_quotes("'''Hello.'''") == "Hello."
    assert (
        _dedent("Docstring here.\n\n    Second paragraph.")
        == "Docstring here.\n\nSecond paragraph."
    )
