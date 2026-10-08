"""Python capability: sandboxed ``run_python`` for agents.

Runs model-written Python through the shared ``SandboxedExecutor`` seam, so it
inherits whatever boundary the spec configures (Docker: no network, read-only
root and workspace, non-root, resource limits). It refuses to build on the
unsandboxed local backend unless the spec explicitly opts in.
"""

from paf.capabilities.python._capability import Python
from paf.capabilities.python._toolset import PythonToolset

__all__ = ["Python", "PythonToolset"]
