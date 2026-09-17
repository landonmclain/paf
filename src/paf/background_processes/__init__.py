"""Per-process background logic, one module per process.

The coordinator (``paf.background``) stays a thin host over a registry of
self-describing process definitions; each process's worker and its
conversation-scoped storage live here. Adding a process means adding a spec
plus a worker module — never touching the coordinator.
"""
