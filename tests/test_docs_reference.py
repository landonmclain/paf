"""The docs name every spec key the code reads.

The spec reference and the security pages are written by hand, so they drift
when a knob is added without them. These tests read the key names straight
from the parsers (``specs.py`` and the policy modules) and fail when the
matching page doesn't mention one, as `` `key` ``.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from paf import specs
from paf.capabilities import PROFILES
from paf.capabilities.compaction import CompactionConfig
from paf.capabilities.guardrails import gating_policy
from paf.capabilities.injection import _policy as injection_policy
from paf.capabilities.shell import _command_policy as command_policy

DOCS = Path(__file__).resolve().parents[1] / "docs"


def _doc(name: str) -> str:
    return (DOCS / name).read_text(encoding="utf-8")


def _missing(keys, doc: str) -> list[str]:
    return sorted(k for k in set(keys) if f"`{k}`" not in doc)


def _read_keys(source: str) -> set[str]:
    """Literal keys a parser reads: ``x.get("key")`` and the block helpers."""
    keys = set(
        re.findall(r'\b(?:data|raw|block|executor|cfg)\.get\(\s*"(\w+)"', source)
    )
    keys |= set(
        re.findall(
            r"_spec_(?:mapping_block|top_level_str|rooted_config)\("
            r'(?:[^()]|\([^()]*\))*?,\s*"(\w+)"',
            source,
        )
    )
    keys |= set(re.findall(r'_spec_rooted_config\([^,]+,\s*"\w+",\s*"(\w+)"', source))
    return keys


def test_spec_reference_covers_every_key_specs_py_reads():
    keys = _read_keys(inspect.getsource(specs))
    assert len(keys) > 40, keys  # the regexes still find the parsers
    assert _missing(keys, _doc("spec-reference.md")) == []


def test_spec_reference_covers_executor_keys():
    source = inspect.getsource(specs._spec_executor_block)
    keys = set(re.findall(r'"(\w+)"', source))
    assert {"backend", "max_writable_size", "writable"} <= keys
    assert _missing(keys, _doc("spec-reference.md")) == []


def test_spec_reference_covers_compaction_and_profiles():
    doc = _doc("spec-reference.md")
    assert _missing(CompactionConfig.__dataclass_fields__, doc) == []
    assert _missing(PROFILES, doc) == []


@pytest.mark.parametrize(
    ("page", "keys"),
    [
        (
            "security/shell-command-policy.md",
            command_policy._KEYS
            | command_policy._RULE_KEYS
            | set(command_policy.EFFECTS),
        ),
        (
            "security/prompt-injection.md",
            injection_policy._KEYS | set(injection_policy.ACTIONS),
        ),
        (
            "security/tool-gating.md",
            set(
                re.findall(
                    r'(?:raw|config)\.get\("(\w+)"', inspect.getsource(gating_policy)
                )
            ),
        ),
    ],
)
def test_security_pages_cover_their_policy_keys(page, keys):
    assert keys
    assert _missing(keys, _doc(page)) == []


def test_prompt_injection_page_covers_on_flag_keys():
    doc = _doc("security/prompt-injection.md")
    assert _missing({f"on_flag.{k}" for k in injection_policy._ON_FLAG_KEYS}, doc) == []


def test_relative_links_resolve():
    for page in DOCS.rglob("*.md"):
        text = page.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", text):
            if "://" in target:
                continue
            assert (page.parent / target).exists(), (
                f"{page.relative_to(DOCS)} → {target}"
            )
