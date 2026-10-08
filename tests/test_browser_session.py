"""BrowserSession driver lifecycle (paf.capabilities.browser._toolset)."""

from __future__ import annotations

import asyncio

import pytest

from paf.capabilities.browser import _toolset


class _SlowDriver:
    """An async_playwright stand-in whose start never finishes."""

    instances: list[_SlowDriver] = []

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.exited = False
        _SlowDriver.instances.append(self)

    async def __aenter__(self) -> object:
        self.started.set()
        await asyncio.Event().wait()  # Chromium driver still starting
        raise AssertionError("unreachable")

    async def __aexit__(self, *_: object) -> None:
        self.exited = True


def test_launch_cancelled_mid_start_stops_the_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run cancelled while the driver starts must not leak its connection."""
    _SlowDriver.instances.clear()
    monkeypatch.setattr(_toolset, "async_playwright", _SlowDriver)

    async def scenario() -> None:
        session = _toolset.BrowserSession()
        async with session:
            launch = asyncio.create_task(session._launch())
            await _SlowDriver.instances[0].started.wait()
            launch.cancel()
            with pytest.raises(asyncio.CancelledError):
                await launch

    asyncio.run(scenario())
    assert _SlowDriver.instances[0].exited, "half-started driver was left running"
