"""Python capability: sandboxed code execution for agents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT

from paf.capabilities._sandbox import LocalProcessExecutor, SandboxedExecutor

from ._toolset import PythonToolset


@dataclass
class Python(AbstractCapability[AgentDepsT]):
    """Run model-written Python in the sandbox configured for this agent.

    Unlike ``Shell`` there is no unsandboxed default: model-written Python on
    the host is arbitrary code execution, so construction fails unless a real
    sandbox executor is supplied or ``allow_unsandboxed`` is set deliberately.
    """

    cwd: str | Path = "."
    """Workspace the program runs in (mounted at ``/workspace`` under Docker)."""

    executor: SandboxedExecutor | None = None
    """The execution backend. Required; must not be the local-process backend
    unless ``allow_unsandboxed`` is set."""

    allow_unsandboxed: bool = False
    """Permit the host-process backend. Off by default; for development only."""

    default_timeout: float = 30.0
    max_timeout: float = 120.0
    max_output_chars: int = 20_000
    max_code_chars: int = 60_000

    def __post_init__(self) -> None:
        if self.executor is None:
            raise ValueError("Python needs an executor; none was configured.")
        if (
            isinstance(self.executor, LocalProcessExecutor)
            and not self.allow_unsandboxed
        ):
            raise ValueError(
                "Python refuses the local (unsandboxed) executor. Configure "
                "`python.executor.backend: docker`, or set "
                "`python.allow_unsandboxed: true` to accept running on the host."
            )

    def get_toolset(self) -> PythonToolset[AgentDepsT]:
        assert self.executor is not None
        return PythonToolset[AgentDepsT](
            cwd=Path(self.cwd),
            executor=self.executor,
            default_timeout=self.default_timeout,
            max_timeout=self.max_timeout,
            max_output_chars=self.max_output_chars,
            max_code_chars=self.max_code_chars,
        )
