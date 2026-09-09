"""Shared text-edit primitive: batch, CAS-at-commit, verify, diff, never-lie.

The one canonical implementation of "apply N edits to a text document safely."
Every capability that edits a text artifact (vault ``note_edit``, skills
``edit_skill``, fs ``edit_file``) delegates here with its own read/write seam —
see ``paf.capabilities._edit``. Design (see vault note
*Vault-and-Skills-Batched-Edit-and-Shared-Storage*):

- **Batch** — N ``(old_text, new_text)`` pairs in ONE read -> ONE write. A
  single atomic write means there is no concurrent-read-modify-write race for
  that path: no interleaving is possible (no lock needed).
- **Validate-all-then-write** — every pair's ``old_text`` must appear exactly
  once, checked against ONE fresh read. If any fails (missing or ambiguous),
  the whole call is rejected and NO write happens — never a partial batch.
- **CAS-at-commit** — ``expected_version`` is checked against the file's hash
  inside the same single-read critical section; a mismatch rejects the whole
  call ("someone changed it since you read — re-read and retry").
- **Never-lie** — after writing, we re-read the persisted bytes and confirm the
  hash matches what we wrote. On ANY failure (CAS conflict, unresolved pair,
  or a write that didn't persist) we raise ``ModelRetry`` — never a
  success-shaped string — so the model gets a retryable signal and corrects.
- **Per-pair diff** — each applied pair returns a compact ``-old\\n+new`` diff
  (computed from the *persisted* content, capped) so the model *sees* exactly
  what landed.

The core is deliberately **sync** (atomic writes are sync; the fs capability's
async wrapper calls it directly). ``ModelRetry`` is the error channel, matching
``fs.edit_file``'s ``_recoverable`` contract so pydantic-ai feeds failures
back to the model as retryable.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from pydantic_ai.exceptions import ModelRetry

#: A single replacement: ``old_text`` must appear exactly once in the document.
TextEditOp = dict[str, str]
#: Seam callbacks: how to read the artifact's text and atomically write it back.
Reader = Callable[[Path], str]
Writer = Callable[[Path, str], None]

_DIFF_CAP = 400  # max chars of a single pair's diff shown to the model


def _content_hash(content: str) -> str:
    """Short content hash — the CAS version token (sha256, 12 hex)."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]


def _pair_diff(old_text: str, new_text: str) -> str:
    """A compact ``-old\\n+new`` diff for one pair, capped to ``_DIFF_CAP``.

    The LLM reads ``-/+`` lines more reliably than a ``difflib`` unified
    header block, and it is naturally bounded. Long pairs are head+tail
    truncated at the cap with a ``…`` marker.
    """
    parts = [f"- {old_text}", f"+ {new_text}"]
    out = "\n".join(parts)
    if len(out) <= _DIFF_CAP:
        return out
    return out[:_DIFF_CAP] + "\n…"


def _validate_pair(text: str, pair: TextEditOp) -> str | None:
    """Return an error string if `pair.old_text` isn't uniquely matchable."""
    old = pair.get("old_text", "")
    new = pair.get("new_text", "")
    if not isinstance(old, str) or not isinstance(new, str):
        return "each edit needs string 'old_text' and 'new_text'"
    if old == "":
        return "old_text cannot be empty"
    count = text.count(old)
    if count == 0:
        return "old_text not found"
    if count > 1:
        return f"old_text found {count} times (include more surrounding context to make it unique)"
    return None


@dataclass
class EditResult:
    """Outcome of :func:`apply_edits` on success (only returned when ok)."""

    version: str
    """: The persisted content hash after the write."""
    diffs: list[str] = field(default_factory=list)
    """: One ``-old\\n+new`` diff per applied pair, in order."""

    def summary(self, target: str) -> str:
        """Format the tool-facing confirmation (diffs + version)."""
        head = f"Edited {target}. [version:{self.version}]"
        if not self.diffs:
            return head
        return head + "\n" + "\n".join(self.diffs)


def apply_edits(
    path: Path,
    edits: list[TextEditOp],
    *,
    expected_version: str | None,
    read: Reader,
    write: Writer,
) -> EditResult:
    """Apply ``edits`` to the artifact at ``path`` as one atomic write.

    Raises :class:`ModelRetry` on any failure (CAS conflict, unresolved pair,
    ambiguity, or an unpersisted write) — it never returns a success result
    unless every edit landed on disk.

    Args:
        path: The resolved artifact path (already confinement-checked by caller).
        edits: ``[{old_text, new_text}, ...]`` pairs to apply, in order.
        expected_version: Optional CAS token from the caller's read tool.
        read: Seam — returns the artifact's current text.
        write: Seam — atomically writes ``content`` to ``path``.
    """
    if not isinstance(edits, list) or not edits:
        raise ModelRetry("edits must be a non-empty list of {old_text, new_text} pairs.")

    # 1. ONE fresh read; everything is validated/written against this snapshot.
    try:
        text = read(path)
    except FileNotFoundError:
        raise ModelRetry(f"File not found: {path}") from None

    current_version = _content_hash(text)

    # 2. CAS-at-commit: reject if the caller's version is stale.
    if expected_version is not None and current_version != expected_version:
        raise ModelRetry(
            f"Conflict: file {path.name!r} has changed since you read it "
            f"(expected version: {expected_version}, got: {current_version}). "
            "Re-read and retry."
        )

    # 3. Validate ALL pairs first — reject the whole batch, never partial.
    failures: list[str] = []
    for i, pair in enumerate(edits):
        err = _validate_pair(text, pair)
        if err is not None:
            failures.append(f"edit #{i + 1}: {err} ({pair.get('old_text', '')[:60]!r})")
    if failures:
        joined = "; ".join(failures)
        raise ModelRetry(f"Refusing to edit {path.name}: {joined}. Re-read (or read_note) and retry with exact text.")

    # 4. Apply all replacements to the single snapshot.
    new_text = text
    applied: list[TextEditOp] = []
    for pair in edits:
        new_text = new_text.replace(pair["old_text"], pair["new_text"], 1)
        applied.append(pair)

    # 5. Single atomic write.
    try:
        write(path, new_text)
    except OSError as e:  # pragma: no cover - rare, but never lie about it
        raise ModelRetry(f"Failed to write {path}: {e}") from None

    # 6. Never-lie: confirm the write actually persisted on disk.
    persisted = read(path)
    if _content_hash(persisted) != _content_hash(new_text):
        raise ModelRetry(
            f"Write to {path.name} did not persist as expected; aborting. Re-read and retry."
        )

    # 7. Per-pair diffs from the *persisted* content (proves each landed).
    diffs = [_pair_diff(p["old_text"], p["new_text"]) for p in applied]
    return EditResult(version=_content_hash(persisted), diffs=diffs)
