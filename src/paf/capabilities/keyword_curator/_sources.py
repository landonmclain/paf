"""Read seams: what the curator reads, as bounded text digests.

Two source families, both file-backed:

- **skills** — every ``SKILL.md`` under the configured libraries, ref
  ``skill:<skill_id>`` (the canonical ``category/name`` id the Skills capability
  uses).
- **notes** — every in-scope ``*.md`` under the vault root, ref
  ``note:<vault-relative-path>``.

Each source becomes a :class:`SourceItem`: its ref, a *digest* (the identifying
parts of the file, not the whole thing — frontmatter, headings, the opening of
the body), and an ``(mtime, size)`` snapshot for change detection. The curator
only needs to know what a source is *about*, so a bounded digest keeps every
batch small. Nothing here writes to a source or calls a model.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._watermark import Snapshot

DEFAULT_DIGEST_CHARS = 3000
"""Per-source cap on the digest the curator reads."""

# Frontmatter keys worth showing the curator, in display order. Obsidian's
# `aliases` are included: they are human-declared names for the note.
_FM_KEYS = ("name", "title", "description", "tags", "aliases", "type", "status")
_HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_MAX_HEADINGS = 14
_HAS_TITLE = re.compile(r"^(title|name): ", re.MULTILINE)


# --- secret redaction ------------------------------------------------------------
#
# The digest is what leaves the machine for the model provider, and notes are free
# text: a credential pasted into one must not travel. Redaction is deterministic
# and applies only to the digest (never to the source file). It errs toward
# redacting — a mangled digest costs a slightly worse keyword, a leaked secret
# costs a rotation.

REDACTED = "[REDACTED]"

_KEY_NAMES = (
    r"pass(?:word|wd|phrase)?|secret|token|api[_-]?key|apikey|auth(?:orization)?"
    r"|bearer|credential|private[_-]?key|access[_-]?key|client[_-]?secret"
)
# `password: hunter2`, `wp_password = "abc def"`, `Authorization: Bearer xyz`
_KEYED = re.compile(
    rf"(?i)(\b[\w.-]*(?:{_KEY_NAMES})[\w.-]*\b\s*[:=]\s*)"
    r"(?:bearer\s+)?(?:\"[^\"\n]*\"|'[^'\n]*'|[^\s,;]+(?:[ \t][A-Za-z0-9]{3,6}){0,5})"
)
_TOKEN_SHAPES = re.compile(
    r"\bsk-[A-Za-z0-9_-]{16,}"  # OpenAI / OpenRouter style
    r"|\bgh[pousr]_[A-Za-z0-9]{20,}"  # GitHub
    r"|\bAKIA[0-9A-Z]{16}\b"  # AWS access key id
    r"|\bxox[abprs]-[A-Za-z0-9-]{10,}"  # Slack
    r"|\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"  # JWT
)
_PEM = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)",
    re.DOTALL,
)
# WordPress-style application password: six space-separated groups of four.
_GROUPED = re.compile(r"\b(?:[A-Za-z0-9]{4} ){5}[A-Za-z0-9]{4}\b")


def _looks_random(group: re.Match[str]) -> bool:
    """Six 4-char groups are only a password if they look random, not prose."""
    s = group.group(0)
    return (
        sum(c.isupper() for c in s) >= 2
        and sum(c.isdigit() or c.isupper() for c in s) >= 4
    )


def redact_secrets(text: str) -> str:
    """``text`` with credentials replaced by ``[REDACTED]`` (digest-only)."""
    text = _PEM.sub(REDACTED, text)
    text = _KEYED.sub(lambda m: f"{m.group(1)}{REDACTED}", text)
    text = _TOKEN_SHAPES.sub(REDACTED, text)
    return _GROUPED.sub(lambda m: REDACTED if _looks_random(m) else m.group(0), text)


@dataclass(frozen=True)
class SourceItem:
    """One source the curator can read."""

    ref: str
    """Stable id: ``skill:<skill_id>`` or ``note:<vault-relative-path>``."""
    source_type: str
    """``"skill"`` or ``"note"``."""
    label: str
    """Human label (the skill id / vault-relative path)."""
    text: str
    """The bounded digest the curator reads."""
    snapshot: Snapshot
    """``(mtime, size)`` of the source file, for change detection."""


def _fm_value(value: Any) -> str:
    if isinstance(value, list | tuple | set):
        return ", ".join(str(v) for v in value)
    return str(value)


def digest(raw: str, *, max_chars: int = DEFAULT_DIGEST_CHARS) -> str:
    """The curator-facing digest of one markdown file.

    Frontmatter fields of interest, then headings (outside code fences), then
    the opening of the body. Frontmatter and headings are always kept; the body
    excerpt takes whatever budget remains.
    """
    from paf.capabilities.vault._markdown import parse_frontmatter

    meta, body = parse_frontmatter(raw)
    parts: list[str] = []
    for key in _FM_KEYS:
        if meta.get(key) not in (None, "", [], {}):
            parts.append(f"{key}: {_fm_value(meta[key])}")

    headings: list[str] = []
    in_fence = False
    for line in body.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _HEADING.match(line)
        if m and len(headings) < _MAX_HEADINGS:
            headings.append(f"{'#' * len(m.group(1))} {m.group(2)}")
    if headings:
        parts.append("headings:\n" + "\n".join(headings))

    head = "\n".join(parts)
    budget = max_chars - len(head) - len("\nexcerpt:\n")
    excerpt = ""
    if budget > 80:
        flat = re.sub(r"\n{3,}", "\n\n", body).strip()
        excerpt = flat[:budget].rstrip()
    out = head + (f"\nexcerpt:\n{excerpt}" if excerpt else "")
    return redact_secrets(out[:max_chars]).strip()


def _snapshot(path: Path) -> Snapshot | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_mtime, st.st_size)


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def scan_skills(
    libraries: Iterable[str | Path], *, max_chars: int = DEFAULT_DIGEST_CHARS
) -> dict[str, SourceItem]:
    """Every skill under ``libraries``, keyed by ref. Pure in-memory scan.

    Skill ids and categories come from the Skills capability's own corpus, so
    refs match what ``load_skill`` accepts. When two libraries define the same
    id the first wins.
    """
    from paf.capabilities.skills._corpus import HintCorpus

    out: dict[str, SourceItem] = {}
    roots = [str(p) for p in libraries]
    if not roots:
        return out
    for doc in HintCorpus(roots=roots).docs:
        ref = f"skill:{doc.skill_id}"
        if ref in out:
            continue
        snap = _snapshot(doc.skill_file)
        raw = _read(doc.skill_file)
        if snap is None or raw is None:
            continue
        out[ref] = SourceItem(
            ref=ref,
            source_type="skill",
            label=doc.skill_id,
            text=digest(raw, max_chars=max_chars),
            snapshot=snap,
        )
    return out


def scan_notes(
    root: str | Path,
    *,
    allow: Iterable[str] = (),
    deny: Iterable[str] = (),
    max_chars: int = DEFAULT_DIGEST_CHARS,
) -> dict[str, SourceItem]:
    """Every in-scope note under the vault ``root``, keyed by ref.

    ``allow``/``deny`` are the vault capability's glob patterns (deny wins), so
    the curator reads exactly the notes the agent's vault can read.
    """
    from paf.capabilities.vault._store import ScanScope

    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        return {}
    scope = ScanScope(root=base, allow=list(allow), deny=list(deny))
    out: dict[str, SourceItem] = {}
    for path in scope.iter_notes():
        rel = path.relative_to(base).as_posix()
        snap = _snapshot(path)
        raw = _read(path)
        if snap is None or raw is None:
            continue
        text = digest(raw, max_chars=max_chars)
        if not _HAS_TITLE.search(text):
            text = f"title: {path.stem}\n{text}".strip()
        out[f"note:{rel}"] = SourceItem(
            ref=f"note:{rel}",
            source_type="note",
            label=rel,
            text=text,
            snapshot=snap,
        )
    return out


def scan_sources(
    *,
    skill_libraries: Iterable[str | Path] = (),
    vault_root: str | Path | None = None,
    vault_allow: Iterable[str] = (),
    vault_deny: Iterable[str] = (),
    max_chars: int = DEFAULT_DIGEST_CHARS,
) -> dict[str, SourceItem]:
    """All skill and note sources, keyed by ref (sorted)."""
    items = scan_skills(skill_libraries, max_chars=max_chars)
    if vault_root is not None:
        items.update(
            scan_notes(
                vault_root, allow=vault_allow, deny=vault_deny, max_chars=max_chars
            )
        )
    return dict(sorted(items.items()))


def snapshots(items: dict[str, SourceItem]) -> dict[str, Snapshot]:
    """``{ref: (mtime, size)}`` for the watermark's ``changed``/``vanished``."""
    return {ref: item.snapshot for ref, item in items.items()}
