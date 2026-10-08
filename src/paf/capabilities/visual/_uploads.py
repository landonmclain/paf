"""User image uploads: saved to disk, described once by the vision model.

The flow (paf-ui -> ``ui/sse.py``):

1. ``POST /api/uploads`` hands the raw bytes to :meth:`UploadStore.save`, which
   validates the image, names it by content hash (``img_<16 hex>``, so a
   re-upload dedupes) and writes ``<id>.<ext>`` plus a ``<id>.json`` sidecar
   under ``data/uploads/``.
2. :class:`Describer` starts the description right away in the background, so
   it is usually done before the message is sent. The result is cached in the
   sidecar (keyed by model, so changing the spec's ``visual.model`` re-describes).
3. On send, the chat endpoint waits for each description and appends an
   :func:`image_block` per image to the user's text. The main model (text-only
   deepseek) reads the description; the saved history holds text, never pixels.
4. Follow-up questions go through ``describe_screenshot(ref="img_...")``, which
   re-reads the file from disk (:meth:`UploadStore.model_image`).

The vision model is the spec's ``visual.model`` -- any pydantic-ai model string,
remote (``openrouter:openai/gpt-6-luna``) or local (``ollama:...``).
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import io
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from paf.config import DATA_DIR

log = logging.getLogger(__name__)

UPLOAD_ROOT = DATA_DIR / "uploads"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
# Long side of the copy sent to the vision model: enough for dense screenshots,
# small enough to keep image tokens (and local-model VRAM) in check.
MODEL_MAX_SIDE = 2048

# PIL format name -> (media type, file extension).
_FORMATS = {
    "PNG": ("image/png", "png"),
    "JPEG": ("image/jpeg", "jpg"),
    "WEBP": ("image/webp", "webp"),
    "GIF": ("image/gif", "gif"),
}
_ID_RE = re.compile(r"^img_[0-9a-f]{16}$")

DESCRIBE_PROMPT = """\
Describe this image for an assistant that cannot see it.

- Start with one line saying what it is (screenshot of an app, photo, diagram, \
document, chart, ...).
- Then the details that matter: layout, subjects, state, anything notable.
- Transcribe the meaningful text faithfully. Keep code, numbers, identifiers \
and file names exact; render tables as markdown tables.
- Describe only what is visible; do not guess beyond it.

Be thorough but concise."""


class UploadError(ValueError):
    """An upload was rejected (not an image, unsupported type, too large)."""


def is_upload_id(value: object) -> bool:
    """True when `value` is a well-formed upload id (safe to use in a path)."""
    return isinstance(value, str) and bool(_ID_RE.match(value))


@dataclass
class UploadMeta:
    """Sidecar record for one upload (``<id>.json``)."""

    id: str
    media_type: str
    ext: str
    name: str
    width: int
    height: int
    size: int
    created_at: float = field(default_factory=time.time)
    description: str | None = None
    description_model: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class UploadStore:
    """Content-addressed image store under ``root`` (``data/uploads``)."""

    def __init__(self, root: Path = UPLOAD_ROOT) -> None:
        self.root = root

    def _meta_path(self, upload_id: str) -> Path:
        return self.root / f"{upload_id}.json"

    def save(self, data: bytes, name: str | None = None) -> UploadMeta:
        """Validate and store `data`; return its (possibly existing) record."""
        from PIL import Image, UnidentifiedImageError

        if not data:
            raise UploadError("empty upload")
        if len(data) > MAX_UPLOAD_BYTES:
            raise UploadError(
                f"image is {len(data) / 1e6:.1f} MB; the limit is "
                f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB"
            )
        try:
            with Image.open(io.BytesIO(data)) as im:
                fmt, width, height = im.format, im.width, im.height
                im.verify()
        except (UnidentifiedImageError, OSError, SyntaxError) as e:
            raise UploadError("not a readable image") from e
        if fmt not in _FORMATS:
            raise UploadError(
                f"unsupported image type {fmt}; use PNG, JPEG, WebP or GIF"
            )
        media_type, ext = _FORMATS[fmt]

        upload_id = "img_" + hashlib.sha256(data).hexdigest()[:16]
        existing = self.meta(upload_id)
        if existing is not None:
            return existing
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / f"{upload_id}.{ext}").write_bytes(data)
        meta = UploadMeta(
            id=upload_id,
            media_type=media_type,
            ext=ext,
            name=_clean_name(name) or f"{upload_id}.{ext}",
            width=width,
            height=height,
            size=len(data),
        )
        self._write_meta(meta)
        return meta

    def meta(self, upload_id: str) -> UploadMeta | None:
        if not is_upload_id(upload_id):
            return None
        try:
            raw = json.loads(self._meta_path(upload_id).read_text(encoding="utf-8"))
            return UploadMeta(**raw)
        except (OSError, ValueError, TypeError):
            return None

    def path(self, upload_id: str) -> Path | None:
        """The stored original, or None for an unknown/malformed id."""
        meta = self.meta(upload_id)
        if meta is None:
            return None
        p = self.root / f"{upload_id}.{meta.ext}"
        return p if p.is_file() else None

    def model_image(self, upload_id: str) -> tuple[bytes, str] | None:
        """``(bytes, media_type)`` for the vision model: the original when it is
        small enough, else a copy shrunk to ``MODEL_MAX_SIDE`` (GIFs always
        become a PNG of the first frame)."""
        from PIL import Image

        meta, path = self.meta(upload_id), self.path(upload_id)
        if meta is None or path is None:
            return None
        data = path.read_bytes()
        if meta.ext != "gif" and max(meta.width, meta.height) <= MODEL_MAX_SIDE:
            return data, meta.media_type
        with Image.open(io.BytesIO(data)) as im:
            im.thumbnail((MODEL_MAX_SIDE, MODEL_MAX_SIDE))
            out = io.BytesIO()
            if meta.ext == "jpg":
                im.convert("RGB").save(out, "JPEG", quality=90)
                return out.getvalue(), "image/jpeg"
            im.save(out, "PNG")
            return out.getvalue(), "image/png"

    def set_description(
        self, upload_id: str, text: str, model: str
    ) -> UploadMeta | None:
        return self._update(
            upload_id, description=text, description_model=model, error=None
        )

    def set_error(
        self, upload_id: str, error: str, model: str | None
    ) -> UploadMeta | None:
        return self._update(
            upload_id, description=None, description_model=model, error=error
        )

    def _update(self, upload_id: str, **changes: Any) -> UploadMeta | None:
        meta = self.meta(upload_id)
        if meta is None:
            return None
        for key, value in changes.items():
            setattr(meta, key, value)
        self._write_meta(meta)
        return meta

    def _write_meta(self, meta: UploadMeta) -> None:
        # Write-then-rename so a concurrent reader never sees a half-written file.
        tmp = self._meta_path(meta.id).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta.to_dict(), indent=1), encoding="utf-8")
        tmp.replace(self._meta_path(meta.id))


def _clean_name(name: str | None) -> str:
    """A display-only file name: basename, printable, bounded."""
    if not name:
        return ""
    base = name.replace("\\", "/").rsplit("/", 1)[-1]
    return "".join(c for c in base if c.isprintable())[:120].strip()


class Describer:
    """One description per (upload, model), shared by every waiter.

    :meth:`start` kicks the vision call off in the background (upload time);
    :meth:`wait` returns the record once it is described (send time). A cached
    description for the same model is reused; a failure is recorded on the
    sidecar (``error``) and retried on the next ``start``.
    """

    def __init__(self, store: UploadStore) -> None:
        self.store = store
        self._tasks: dict[tuple[str, str], asyncio.Task[UploadMeta | None]] = {}

    def _done(self, meta: UploadMeta | None, model: str) -> bool:
        return (
            meta is not None
            and meta.description is not None
            and meta.description_model == model
        )

    def pending(self, upload_id: str, model: str) -> bool:
        """True while a description of `upload_id` by `model` is in flight."""
        task = self._tasks.get((upload_id, model))
        return task is not None and not task.done()

    def start(
        self,
        upload_id: str,
        model: str,
        model_settings: dict[str, Any] | None = None,
    ) -> asyncio.Task[UploadMeta | None] | None:
        """Begin describing `upload_id` unless it already is (or was) for `model`."""
        if self._done(self.store.meta(upload_id), model):
            return None
        key = (upload_id, model)
        task = self._tasks.get(key)
        if task is None or task.done():
            task = asyncio.create_task(self._describe(upload_id, model, model_settings))
            self._tasks[key] = task
            task.add_done_callback(lambda _t, k=key: self._tasks.pop(k, None))
        return task

    async def wait(
        self,
        upload_id: str,
        model: str,
        model_settings: dict[str, Any] | None = None,
        timeout: float = 180.0,
    ) -> UploadMeta | None:
        """The record for `upload_id`, described by `model` when that succeeds."""
        task = self.start(upload_id, model, model_settings)
        if task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout)
            except TimeoutError:
                log.warning("image description timed out for %s", upload_id)
        return self.store.meta(upload_id)

    async def _describe(
        self,
        upload_id: str,
        model: str,
        model_settings: dict[str, Any] | None,
    ) -> UploadMeta | None:
        from ._vision import _run_vision

        image = self.store.model_image(upload_id)
        if image is None:
            return None
        data, media_type = image
        try:
            text = await _run_vision(
                model, [data], DESCRIBE_PROMPT, media_type, model_settings
            )
        except Exception as e:  # noqa: BLE001 - provider errors vary; record, don't crash
            log.warning("describing %s with %s failed: %r", upload_id, model, e)
            return self.store.set_error(upload_id, f"{type(e).__name__}: {e}", model)
        return self.store.set_description(upload_id, text.strip(), model)


# --- the text the main model sees -----------------------------------------------

_BLOCK_RE = re.compile(
    r'\n*<image id="(img_[0-9a-f]{16})" name="([^"]*)" size="(\d+)x(\d+)">\n.*?\n</image>',
    re.DOTALL,
)


def image_block(meta: UploadMeta) -> str:
    """The text that stands in for an uploaded image in the user's message."""
    if meta.description:
        body = meta.description
    elif meta.error:
        body = f"(The image could not be described: {meta.error})"
    else:
        body = "(No description available: no vision model is configured.)"
    follow_up = (
        f'For details not covered here, call describe_screenshot(ref="{meta.id}", '
        "prompt=<your question>)."
    )
    return (
        f'<image id="{meta.id}" name="{html.escape(meta.name, quote=True)}" '
        f'size="{meta.width}x{meta.height}">\n{body}\n\n{follow_up}\n</image>'
    )


def with_image_blocks(text: str, metas: list[UploadMeta]) -> str:
    """`text` with one :func:`image_block` per image appended."""
    blocks = "\n\n".join(image_block(m) for m in metas)
    if not blocks:
        return text
    return f"{text}\n\n{blocks}" if text.strip() else blocks


def split_image_blocks(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Inverse of :func:`with_image_blocks` for display: the user's own text and
    the images (``{id, name, width, height}``) that were attached."""
    images = [
        {
            "id": m.group(1),
            "name": html.unescape(m.group(2)),
            "width": int(m.group(3)),
            "height": int(m.group(4)),
        }
        for m in _BLOCK_RE.finditer(text)
    ]
    if not images:
        return text, []
    return _BLOCK_RE.sub("", text).strip(), images
