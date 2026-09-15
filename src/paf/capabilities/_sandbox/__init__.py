"""The sandbox seam: a swappable execution boundary for code-running tools.

Mirrors ``capabilities/_storage``: a protocol seam plus swappable backends that
many capabilities import. Shell, ``run_tests``, ``install_package``, git, and
codebase_map route their command execution through :class:`SandboxedExecutor`
instead of spawning processes directly, so the sandbox is one seam instead of a
per-tool concern.
"""

from paf.capabilities._sandbox._docker import DockerExecutor
from paf.capabilities._sandbox._local import LocalProcessExecutor
from paf.capabilities._sandbox._protocol import (
    BackgroundHandle,
    BackgroundStatus,
    ExecResult,
    SandboxedExecutor,
)

__all__ = [
    'BackgroundHandle',
    'BackgroundStatus',
    'DockerExecutor',
    'ExecResult',
    'LocalProcessExecutor',
    'SandboxedExecutor',
]
