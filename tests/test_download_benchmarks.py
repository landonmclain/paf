"""Tests for scripts/download_benchmarks.py.

Offline: the fetch is stubbed, so no network is needed. The one end-to-end
check (the pinned checksum against a dataset already on disk) is skipped when
the dataset has not been downloaded yet.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

_SCRIPT = REPO_ROOT / "scripts" / "download_benchmarks.py"
_spec = importlib.util.spec_from_file_location("download_benchmarks", _SCRIPT)
assert _spec and _spec.loader
dbs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dbs)


def _question(qid: str) -> dict:
    return {
        "question_id": qid,
        "question_type": "single-session-user",
        "question": "What did I say?",
        "answer": "Something.",
        "question_date": "2024-01-01",
        "haystack_dates": ["2023-12-01"],
        "haystack_session_ids": ["s1"],
        "haystack_sessions": [[{"role": "user", "content": "hi", "has_answer": True}]],
        "answer_session_ids": ["s1"],
    }


def _good_payload(n: int = 3) -> list[dict]:
    return [_question(f"q{i}") for i in range(n)]


def test_validate_accepts_a_well_formed_payload():
    summary = dbs.validate(_good_payload(), {"question_count": 3})
    assert "3 questions" in summary
    assert "1.00 sessions/question" in summary


def test_validate_rejects_a_wrong_question_count():
    with pytest.raises(ValueError, match="expected 3 questions, got 2"):
        dbs.validate(_good_payload(2), {"question_count": 3})


def test_validate_rejects_missing_keys():
    payload = _good_payload(1)
    del payload[0]["haystack_dates"]
    with pytest.raises(ValueError, match="missing"):
        dbs.validate(payload, {"question_count": 1})


def test_validate_rejects_answer_sessions_absent_from_the_haystack():
    payload = _good_payload(1)
    payload[0]["answer_session_ids"] = ["not-in-haystack"]
    with pytest.raises(ValueError, match="not in the haystack"):
        dbs.validate(payload, {"question_count": 1})


def test_validate_rejects_a_non_list():
    with pytest.raises(ValueError, match="top-level JSON array"):
        dbs.validate({"questions": []}, {"question_count": 0})


class _FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc) -> bool:
        return False


def _stub(monkeypatch, body: bytes):
    def urlopen(url, timeout=None):  # noqa: ARG002 - signature mirrors urllib
        return _FakeResponse(body)

    monkeypatch.setattr(dbs.urllib.request, "urlopen", urlopen)


def test_download_writes_and_verifies(tmp_path, monkeypatch, capsys):
    body = json.dumps(_good_payload(2)).encode()
    spec = {
        "url": "https://example.invalid/dataset.json",
        "dest": tmp_path / "dataset.json",
        "sha256": hashlib.sha256(body).hexdigest(),
        "question_count": 2,
    }
    _stub(monkeypatch, body)
    assert dbs.download("test-dataset", spec, force=False) is True
    assert (tmp_path / "dataset.json").read_bytes() == body
    assert "wrote" in capsys.readouterr().out


def test_download_is_idempotent(tmp_path, monkeypatch, capsys):
    body = json.dumps(_good_payload(2)).encode()
    dest = tmp_path / "dataset.json"
    dest.write_bytes(body)
    spec = {
        "url": "https://example.invalid/dataset.json",
        "dest": dest,
        "sha256": hashlib.sha256(body).hexdigest(),
        "question_count": 2,
    }

    def urlopen(url, timeout=None):  # noqa: ARG002 - must not be reached
        raise AssertionError("should not re-download a verified file")

    monkeypatch.setattr(dbs.urllib.request, "urlopen", urlopen)
    assert dbs.download("test-dataset", spec, force=False) is True
    assert "already present" in capsys.readouterr().out


def test_download_refuses_a_checksum_mismatch(tmp_path, monkeypatch):
    body = json.dumps(_good_payload(2)).encode()
    spec = {
        "url": "https://example.invalid/dataset.json",
        "dest": tmp_path / "dataset.json",
        "sha256": "0" * 64,
        "question_count": 2,
    }
    _stub(monkeypatch, body)
    assert dbs.download("test-dataset", spec, force=False) is False
    assert not (tmp_path / "dataset.json").exists()


def test_download_refuses_a_bad_schema(tmp_path, monkeypatch):
    body = json.dumps([{"question_id": "q0"}]).encode()
    spec = {
        "url": "https://example.invalid/dataset.json",
        "dest": tmp_path / "dataset.json",
        "sha256": hashlib.sha256(body).hexdigest(),
        "question_count": 1,
    }
    _stub(monkeypatch, body)
    assert dbs.download("test-dataset", spec, force=False) is False
    assert not (tmp_path / "dataset.json").exists()


def test_pinned_checksum_matches_the_dataset_on_disk():
    """If the benchmark data is present, it must be the pinned revision."""
    for name, spec in dbs.DATASETS.items():
        dest = Path(spec["dest"])
        if not dest.exists():
            pytest.skip(f"{name} not downloaded; run scripts/download_benchmarks.py")
        assert dbs.sha256(dest.read_bytes()) == spec["sha256"], (
            f"{dest} is not the pinned revision of {name}"
        )
