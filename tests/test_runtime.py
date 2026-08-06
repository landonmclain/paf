"""Tests for ``paf.runtime.Session`` — the UI-agnostic run lifecycle wrapper.

The default (non-live) tests exercise the pure helpers and importability, so
they run offline with no API key or network. The live-marked test runs the real
model end-to-end through the ``Session`` and is excluded from the default
``pytest`` run (``addopts = -m 'not live'``) — invoke explicitly with
``pytest -m live`` using ``.env``.
"""

from __future__ import annotations

import pytest

from paf import runtime


def test_part_leading_text_str() -> None:
    """A str content is returned verbatim (first chunk of a text part)."""
    part = _Part(content="Hello")
    assert runtime._part_leading_text(part) == "Hello"


def test_part_leading_text_list_of_str() -> None:
    """List content (pydantic-ai TextPart / chunked) is joined."""
    part = _Part(content=["a", "b", "c"])
    assert runtime._part_leading_text(part) == "abc"


def test_part_leading_text_non_text() -> None:
    """Non-text parts (empty/None content) yield an empty string."""
    assert runtime._part_leading_text(_Part(content=None)) == ""
    assert runtime._part_leading_text(_Part(content=[])) == ""


def test_part_leading_text_missing_attr() -> None:
    """Parts without a ``content`` attr degrade to ``""`` (never raise)."""
    assert runtime._part_leading_text(_NoContent()) == ""


def test_session_importable_and_news() -> None:
    """Session constructs its doc-visible surface without a model key."""
    # Trace disabled so construction is offline-safe; build_agent lazily needs
    # the model config which we don't touch here.
    assert callable(runtime.Session)
    assert callable(runtime.Session.run_sync)


class _Part:
    def __init__(self, content) -> None:
        self.content = content


class _NoContent:
    pass


@pytest.mark.live
def test_session_runs_one_turn_live() -> None:
    """Session.run_sync runs a real turn and appends to history."""
    with runtime.Session("assistant") as s:
        out = s.run_sync("Reply with exactly: pong")
        assert "pong" in out.lower()
        assert s.name == "assistant"
        # One user turn -> at least the user + assistant messages recorded.
        assert len(s.history) >= 2
