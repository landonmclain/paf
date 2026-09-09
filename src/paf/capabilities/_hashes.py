"""Content-stable message hashing shared by the persistence + memory read paths.

The harness's ``message_hash`` hashes the raw serialized :class:`ModelMessage`
bytes, which include pydantic's per-instantiation ``timestamp`` on parts. Two
byte-identical conversation messages that get re-instantiated across runs
(replay / ``continue_run`` / steer / snapshot round-trip) therefore hash
DIFFERENTLY, which silently defeats every overlap-reconciliation that keys on
message identity — both within a run's snapshot union and across cumulative
runs — re-collecting already-consumed history (the resume/many-run duplication
bug seen in real conversations).

:func:`_content_hash` hashes durable content (part kind + text) in order, so
identical content is recognizable across re-instantiation while repeated
identical content at distinct positions still differs (position preserved).
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage


def _content_hash(message: "ModelMessage") -> str:
    """Content-stable hash of a message's durable parts, in order."""
    toks: list[str] = []
    for part in getattr(message, "parts", ()):
        text = getattr(part, "content", None)
        toks.append(
            f"{type(part).__name__}:{text}"
            if isinstance(text, str)
            else type(part).__name__
        )
    return hashlib.sha256("\n".join(toks).encode()).hexdigest()
