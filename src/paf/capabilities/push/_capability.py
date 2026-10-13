"""Push capability: the agent-facing ``send_push`` tool, when delivery is wired."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT

from paf.push import PushResult

from ._toolset import PushToolset

# (title, body, urgent) -> PushResult; supplied by the Session (it knows the
# agent, the conversation and the running turn).
Notifier = Callable[[str, str, bool], Awaitable[PushResult]]


@dataclass
class Push(AbstractCapability[AgentDepsT]):
    """Lets an agent send the user a push notification.

    ``notifier`` is wired by the embedding layer (paf-sse). With none (the CLI,
    tests) the capability contributes no toolset at all, so the model is never
    offered a tool that can't work.
    """

    notifier: Notifier | None = None

    def get_toolset(self) -> PushToolset[AgentDepsT] | None:
        if self.notifier is None:
            return None
        return PushToolset[AgentDepsT](self.notifier)

    @classmethod
    def get_serialization_name(cls) -> str:
        """The name used by custom capability specs (the allowlist key)."""
        return "Push"
