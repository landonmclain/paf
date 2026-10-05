"""Image uploads: disk store, background description, <image> prompt blocks,
and the ui/sse.py endpoints that tie them together (vision model faked)."""

from __future__ import annotations

import asyncio
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic_ai.messages import ModelRequest, UserPromptPart

from paf.capabilities.visual import _store as shot_store
from paf.capabilities.visual import _toolset, _uploads, _vision
from paf.capabilities.visual._uploads import (
    MODEL_MAX_SIDE,
    Describer,
    UploadError,
    UploadStore,
    is_upload_id,
    split_image_blocks,
    with_image_blocks,
)
from ui import sse


def _png(w: int = 40, h: int = 30, color: str = "red") -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (w, h), color).save(out, "PNG")
    return out.getvalue()


@pytest.fixture
def store(tmp_path) -> UploadStore:
    return UploadStore(tmp_path / "uploads")


@pytest.fixture
def fake_vision(monkeypatch) -> list[tuple[str, str, int]]:
    """Replace the vision call; records (model, media_type, bytes) per call."""
    calls: list[tuple[str, str, int]] = []

    async def fake(model, images, prompt, media_type="image/png", model_settings=None):
        calls.append((model, media_type, len(images[0])))
        settings_seen.append(model_settings)
        return f"A red rectangle (described by {model})."

    monkeypatch.setattr(_vision, "_run_vision", fake)
    monkeypatch.setattr(_toolset, "_run_vision", fake)
    return calls


settings_seen: list[dict | None] = []


# --- store -----------------------------------------------------------------------


def test_save_names_by_content_and_dedupes(store) -> None:
    data = _png()
    meta = store.save(data, "../../etc/shot.png")
    assert is_upload_id(meta.id)
    assert (meta.media_type, meta.ext, meta.width, meta.height) == (
        "image/png",
        "png",
        40,
        30,
    )
    assert meta.name == "shot.png"  # basename only
    assert store.path(meta.id).read_bytes() == data
    again = store.save(data, "other.png")
    assert again.id == meta.id and again.name == "shot.png"  # existing record wins


def test_save_rejects_non_images_and_oversize(store, monkeypatch) -> None:
    with pytest.raises(UploadError):
        store.save(b"not an image")
    with pytest.raises(UploadError):
        store.save(b"")
    monkeypatch.setattr(_uploads, "MAX_UPLOAD_BYTES", 10)
    with pytest.raises(UploadError, match="limit"):
        store.save(_png())


def test_unknown_or_malformed_ids_resolve_to_nothing(store) -> None:
    assert store.meta("img_0000000000000000") is None
    assert store.path("../secret") is None
    assert store.model_image("img_../../x") is None


def test_model_image_shrinks_large_and_keeps_small(store) -> None:
    small = store.save(_png(), "s.png")
    data, media = store.model_image(small.id)
    assert media == "image/png" and data == store.path(small.id).read_bytes()

    big = store.save(_png(3000, 1000, "blue"), "b.png")
    data, media = store.model_image(big.id)
    with Image.open(io.BytesIO(data)) as im:
        assert max(im.size) == MODEL_MAX_SIDE
    assert media == "image/png"


# --- prompt blocks ---------------------------------------------------------------


def test_blocks_round_trip_for_display(store) -> None:
    meta = store.save(_png(), 'a "quoted" name.png')
    store.set_description(meta.id, "A red rectangle.\n\n| a | b |", "m")
    meta = store.meta(meta.id)

    text = with_image_blocks("what is this?", [meta])
    assert "A red rectangle." in text and f'ref="{meta.id}"' in text
    clean, images = split_image_blocks(text)
    assert clean == "what is this?"
    assert images == [
        {"id": meta.id, "name": 'a "quoted" name.png', "width": 40, "height": 30}
    ]

    only_image = with_image_blocks("", [meta])
    assert split_image_blocks(only_image) == ("", images)
    assert split_image_blocks("plain text") == ("plain text", [])


# --- describer -------------------------------------------------------------------


def test_describer_describes_once_and_caches(store, fake_vision) -> None:
    meta = store.save(_png(), "x.png")
    describer = Describer(store)

    async def run():
        describer.start(meta.id, "m1")
        a, b = await asyncio.gather(
            describer.wait(meta.id, "m1"), describer.wait(meta.id, "m1")
        )
        return a, b, await describer.wait(meta.id, "m1")

    a, b, c = asyncio.run(run())
    assert len(fake_vision) == 1  # concurrent waiters + later wait share one call
    assert a.description == c.description == "A red rectangle (described by m1)."
    assert c.description_model == "m1"

    asyncio.run(describer.wait(meta.id, "m2"))  # a new model re-describes
    assert [call[0] for call in fake_vision] == ["m1", "m2"]


def test_describer_and_tool_pass_model_settings(store, fake_vision) -> None:
    settings = {"openai_reasoning_effort": "none"}
    meta = store.save(_png(), "x.png")
    settings_seen.clear()
    asyncio.run(Describer(store).wait(meta.id, "ollama:x", settings))
    toolset = _toolset.VisualToolset(
        model="ollama:x", uploads=store, model_settings=settings
    )
    asyncio.run(toolset.describe_screenshot(meta.id, "?"))
    assert settings_seen == [settings, settings]


def test_describer_records_failure(store, monkeypatch) -> None:
    async def boom(*_a, **_k):
        raise RuntimeError("provider down")

    monkeypatch.setattr(_vision, "_run_vision", boom)
    meta = store.save(_png(), "x.png")
    out = asyncio.run(Describer(store).wait(meta.id, "m"))
    assert out.description is None and "provider down" in out.error
    assert "could not be described" in with_image_blocks("", [out])


# --- endpoints + chat ------------------------------------------------------------


@pytest.fixture
def client(store, fake_vision, monkeypatch):
    monkeypatch.setattr(sse, "_uploads", store)
    monkeypatch.setattr(sse, "_describer", Describer(store))
    monkeypatch.setattr(
        sse, "spec_visual_config", lambda agent: {"model": "fake:vision"}
    )
    with TestClient(sse.app) as c:
        yield c


def test_upload_serve_and_describe(client, store, fake_vision) -> None:
    data = _png()
    resp = client.post("/api/uploads?name=pic.png", content=data)
    assert resp.status_code == 200
    up = resp.json()
    assert (
        up["status"] in {"describing", "ready"}
        and up["url"] == f"/api/uploads/{up['id']}"
    )

    img = client.get(up["url"])
    assert img.status_code == 200 and img.content == data
    assert img.headers["content-type"] == "image/png"

    # The chat path waits for the description and appends it to the prompt.
    prompt = asyncio.run(sse._prompt_with_images("look", [up["id"]], "assistant"))
    assert prompt.startswith("look\n\n<image ") and "described by fake:vision" in prompt
    assert client.get(f"/api/uploads/{up['id']}/meta").json()["status"] == "ready"


def test_upload_and_chat_reject_bad_input(client) -> None:
    assert client.post("/api/uploads", content=b"nope").status_code == 400
    assert client.get("/api/uploads/img_0000000000000000").status_code == 404
    assert client.get("/api/uploads/not-an-id").status_code == 404
    bad = client.post("/api/chat", json={"conversation_id": "c", "images": ["../x"]})
    assert bad.status_code == 422
    empty = client.post("/api/chat", json={"conversation_id": "c"})
    assert empty.status_code == 422


def test_history_splits_images_out_of_user_text(store) -> None:
    meta = store.save(_png(), "pic.png")
    store.set_description(meta.id, "A red rectangle.", "m")
    content = with_image_blocks("what is this?", [store.meta(meta.id)])
    out = sse._serialize_history(
        [ModelRequest(parts=[UserPromptPart(content=content)])]
    )
    assert out == [
        {
            "role": "user",
            "text": "what is this?",
            "images": [{"id": meta.id, "name": "pic.png", "width": 40, "height": 30}],
        }
    ]


# --- follow-up tool + screenshot refs --------------------------------------------


def test_describe_screenshot_reads_uploads_from_disk(store, fake_vision) -> None:
    meta = store.save(_png(), "pic.png")
    toolset = _toolset.VisualToolset(model="fake:vision", uploads=store)
    answer = asyncio.run(toolset.describe_screenshot(meta.id, "what color?"))
    assert "described by fake:vision" in answer
    assert fake_vision[-1][:2] == ("fake:vision", "image/png")
    missing = asyncio.run(toolset.describe_screenshot("img_0000000000000000", "?"))
    assert "no uploaded image" in missing


def test_screenshot_refs_are_stable_across_restashes() -> None:
    s = shot_store.ScreenshotStore(max_images=2)
    a, b = _png(color="red"), _png(color="blue")
    ref_a = s.save(a)
    assert s.save(a) == ref_a  # the hook re-stashes history every request
    s.save(b)
    s.save(_png(color="green"))  # evicts the least recently stashed (a)
    assert s.get(ref_a) is None and len(s) == 2
