"""API checks that load no model and call no Claude."""
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import api

client = TestClient(api.app)


def jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64)).save(buf, "JPEG")
    return buf.getvalue()


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_unknown_fill_estimator_is_rejected():
    r = client.post("/identify?fill=bogus", files={"image": ("a.jpg", jpeg())})
    assert r.status_code == 400 and "unknown fill estimator" in r.json()["detail"]


@pytest.mark.parametrize("query", ["pipeline=claude&fill=none", "pipeline=local&fill=geometry:opus-grid"])
def test_claude_steps_need_a_key(monkeypatch, query):
    monkeypatch.setattr(api, "has_api_key", lambda: False)
    assert client.post(f"/identify?{query}", files={"image": ("a.jpg", jpeg())}).status_code == 503


def test_not_an_image():
    r = client.post("/identify?fill=none&ocr=false", files={"image": ("notes.txt", b"hello")})
    assert r.status_code == 400 and r.json()["detail"] == "notes.txt is not an image"


def test_preview_reports_the_photo_size():
    r = client.post("/preview", files={"image": ("a.jpg", jpeg())})
    assert r.status_code == 200 and r.headers["x-width"] == "64"
