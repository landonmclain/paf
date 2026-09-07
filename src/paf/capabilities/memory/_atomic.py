"""Atomic-write helpers — re-export shim (canonical home: ``_storage._atomic``).

These helpers used to be inlined here (verbatim copies across memory, vault,
codebase_map, and the archived turn_index). The canonical implementation now
lives in ``paf.capabilities._storage._atomic``; this module re-exports it so
existing importers keep working. New consumers should import from
``paf.capabilities._storage`` directly.
"""

from paf.capabilities._storage._atomic import read_json, write_atomic, write_json_atomic

__all__ = ["read_json", "write_atomic", "write_json_atomic"]
