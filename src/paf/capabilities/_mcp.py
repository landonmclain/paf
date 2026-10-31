"""An MCP toolset that degrades instead of failing the run.

pydantic-ai starts each MCP server when a run begins. If one cannot start (a
missing command, a slow first download, a server that is down), the exception
ends the whole run. ``TolerantToolset`` logs the failure and offers no tools from
that server, so the agent carries on without it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Self

from pydantic_ai import RunContext
from pydantic_ai.toolsets import AbstractToolset, WrapperToolset
from pydantic_ai.toolsets.abstract import ToolsetTool

logger = logging.getLogger(__name__)


@dataclass
class TolerantToolset(WrapperToolset[Any]):
    """Wrap a toolset so a failure to start it removes its tools, not the run.

    ``down`` is a field so the per-run copies pydantic-ai makes (``for_run``) keep
    it.
    """

    down: bool = False

    def _server_name(self) -> str:
        return str(getattr(self.wrapped, "prefix", None) or self.wrapped.label)

    async def __aenter__(self) -> Self:
        if self.down:
            return self
        try:
            await self.wrapped.__aenter__()
        except Exception as exc:  # noqa: BLE001 - any start-up failure should degrade
            self.down = True
            logger.warning(
                "MCP server %s unavailable: %s: %s",
                self._server_name(),
                type(exc).__name__,
                exc,
            )
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        if self.down:
            return None
        return await self.wrapped.__aexit__(*args)

    async def get_tools(self, ctx: RunContext[Any]) -> dict[str, ToolsetTool[Any]]:
        if self.down:
            return {}
        try:
            return await self.wrapped.get_tools(ctx)
        except Exception as exc:  # noqa: BLE001 - a server that dies after start
            self.down = True
            logger.warning(
                "MCP server %s unavailable: %s: %s",
                self._server_name(),
                type(exc).__name__,
                exc,
            )
            return {}


def tolerant(toolset: AbstractToolset[Any]) -> TolerantToolset:
    return TolerantToolset(wrapped=toolset)
