"""Unit tests for the transient ModelHTTPError retry logic in `_run_turn`.

`_run_turn` (tests/framework.py) had a single real gap: an OpenRouter/provider
`ModelHTTPError` (5xx / 429 / 408) propagated straight out, aborting an entire
long graded run at the first network blip (2026-08-15: a 502 killed a 472-Q run
at q328). These tests prove the wait-and-retry guard:

  - a retryable status (e.g. 502) is retried with backoff and recovers,
  - a fatal status (e.g. 401) is NOT retried and propagates,
  - retrying is exhausted -> the error still propagates after max attempts.

The agent's `run_stream_events` is faked (async context manager yielding
`AgentRunResultEvent`), so no real model call / tokens are used.
"""

from __future__ import annotations

import types

import httpx
import pytest
from pydantic_ai import AgentRunResultEvent
from pydantic_ai.exceptions import ModelHTTPError

from tests.framework import (
    LME_RETRY_BASE_BACKOFF,
    LME_RETRY_MAX_ATTEMPTS,
    _run_turn,
)


class _FakeAgent:
    """Minimal Agent stand-in whose `run_stream_events` models provider errors.

    The fail counter lives HERE (instance-level) so it persists ACROSS retries:
    `_run_turn` calls `agent.run_stream_events(...)` once per retry, and each
    call returns a fresh stream — but the counter must keep counting so the
    N-th overall attempt (not N-th within one stream) is the one that succeeds.

    `fail_count` = number of total attempts to fail before succeeding.
    `fail_permanently` = when True, never succeed (for exhaustion tests).
    """

    def __init__(self, exc, fail_count: int = 1, fail_permanently: bool = False):
        self._exc = exc
        self._attempts = 0
        self._fail_count = fail_count
        self._fail_permanently = fail_permanently
        self._used_stream = False  # ensure only one result event per stream

    def run_stream_events(self, *args, **kwargs):
        return _FakeStream(self)

    # -- cross-attempt bookkeeping used by _FakeStream ----------------------
    def _is_next_attempt_a_failure(self) -> bool:
        if self._fail_permanently:
            return True
        self._attempts += 1
        return self._attempts <= self._fail_count


class _FakeStream:
    """Async context-manager + async-iterator yielding one result event."""

    def __init__(self, agent: _FakeAgent):
        self._agent = agent
        self._yielded = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._agent._is_next_attempt_a_failure():
            raise self._agent._exc
        if self._yielded:
            raise StopAsyncIteration
        self._yielded = True
        result = types.SimpleNamespace(
            output="final answer",
            usage=types.SimpleNamespace(
                input_tokens=1,
                output_tokens=2,
                cache_read_tokens=3,
                cache_write_tokens=4,
                total_tokens=5,
                requests=1,
            ),
        )
        return AgentRunResultEvent(result=result)


def _fake_sleep(*_a, **_k):
    """No-op asyncio.sleep so retry tests run instantly.

    MUST be async — `_run_turn` does `await asyncio.sleep(delay)`, and
    `_fake_sleep` replaces the `tests.framework.asyncio.sleep` attribute, so it
    must itself be awaitable.
    """
    return _AsyncNoop()


class _AsyncNoop:
    def __await__(self):
        if False:
            yield None  # pragma: no cover
        return None


@pytest.mark.live
def test_retryable_502_recovers(monkeypatch, capsys) -> None:
    """A 502 'Network connection lost' on attempt 1 is retried and succeeds."""
    monkeypatch.setattr("tests.framework.asyncio.sleep", _fake_sleep)
    exc = ModelHTTPError(
        status_code=502, model_name="x", body="Network connection lost."
    )
    agent = _FakeAgent(exc, fail_count=1)
    row = _run_turn(agent, "question", "cid", "lmem.recall.1", "sess")
    assert row["text"] == "final answer"
    assert row["bailed"] is False
    # A retry was actually attempted and logged.
    assert "retryable ModelHTTPError 502" in capsys.readouterr().out


@pytest.mark.live
def test_fatal_401_not_retried(monkeypatch) -> None:
    """A 401 never gets retried — it propagates immediately (fatal status)."""
    slept = {"n": 0}

    async def _noop_sleep(*_a, **_k):
        slept["n"] += 1

    monkeypatch.setattr("tests.framework.asyncio.sleep", _noop_sleep)
    exc = ModelHTTPError(status_code=401, model_name="x", body="Unauthorized")
    agent = _FakeAgent(exc, fail_permanently=True)
    with pytest.raises(ModelHTTPError):
        _run_turn(agent, "q", "cid", "n", "s")
    assert slept["n"] == 0  # never slept => never retried a fatal status


@pytest.mark.live
def test_retry_exhausted_still_raises(monkeypatch) -> None:
    """Retryable but never-resolving -> raises after max attempts, constant backoff."""
    slept: list[float] = []

    async def _spy_sleep(d):
        slept.append(d)

    monkeypatch.setattr("tests.framework.asyncio.sleep", _spy_sleep)
    exc = ModelHTTPError(status_code=502, model_name="x", body="down")
    agent = _FakeAgent(exc, fail_permanently=True)
    with pytest.raises(ModelHTTPError):
        _run_turn(agent, "q", "cid", "n", "s")
    # One sleep per retry that FAILED and was retried, before the final raise:
    # total attempts = LME_RETRY_MAX_ATTEMPTS; sleeps happen on attempts 1..N-1.
    assert len(slept) == LME_RETRY_MAX_ATTEMPTS - 1
    # Constant backoff (Landon default 30s between 5 retries).
    assert all(d == LME_RETRY_BASE_BACKOFF for d in slept)


# ---- Raw transport errors (httpx.ReadError and friends) ----
# A mid-stream disconnect surfaces as a raw `httpx.TransportError` (no HTTP
# status code), which pydantic-ai does NOT wrap into ModelHTTPError. Before the
# httpx.TransportError handling was added to `_run_turn`, a single such blip
# killed a whole 472-Q run at q329 (2026-08-18). These prove the transport path
# retries with the same backoff and recovers.


@pytest.mark.live
def test_transport_error_recovers(monkeypatch, capsys) -> None:
    """A raw httpx.ReadError on attempt 1 is retried and succeeds."""
    monkeypatch.setattr("tests.framework.asyncio.sleep", _fake_sleep)
    agent = _FakeAgent(httpx.ReadError("disconnected"), fail_count=1)
    row = _run_turn(agent, "question", "cid", "lmem.recall.1", "sess")
    assert row["text"] == "final answer"
    assert row["bailed"] is False
    out = capsys.readouterr().out
    assert "retryable httpx transport error" in out
    assert "ReadError" in out


@pytest.mark.live
def test_transport_error_exhausted_still_raises(monkeypatch) -> None:
    """A transport error that never resolves -> raises after max attempts."""
    slept: list[float] = []

    async def _spy_sleep(d):
        slept.append(d)

    monkeypatch.setattr("tests.framework.asyncio.sleep", _spy_sleep)
    agent = _FakeAgent(httpx.ReadTimeout("timed out"), fail_permanently=True)
    with pytest.raises(httpx.ReadTimeout):
        _run_turn(agent, "q", "cid", "n", "s")
    assert len(slept) == LME_RETRY_MAX_ATTEMPTS - 1
    assert all(d == LME_RETRY_BASE_BACKOFF for d in slept)
