import io
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def _fake_image_bytes() -> bytes:
    img = Image.new("RGB", (224, 224), color=(50, 140, 60))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client():
    mock_model = MagicMock()
    mock_model.predict.return_value = [np.array([0.1, 0.7, 0.1, 0.1])]
    with patch("api.main._load_model", return_value=mock_model):
        from api.main import app

        yield TestClient(app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_schema(client):
    response = client.post(
        "/predict",
        files={"file": ("leaf.png", _fake_image_bytes(), "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["disease_class"] == "Blast"
    assert body["confidence"] == 0.7
    assert body["grad_cam_available"] is True


def test_predict_invalid_type(client):
    response = client.post(
        "/predict",
        files={"file": ("leaf.txt", b"not-an-image", "text/plain")},
    )
    assert response.status_code == 422
