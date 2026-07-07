"""FastAPI for rice leaf disease prediction."""

from __future__ import annotations

import io
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.inference import model_path, predict_image

app = FastAPI(title="Rice Leaf Disease API", version="1.0.0")
_model = None


def _load_model():
    global _model
    if _model is None:
        path = model_path(ROOT)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing model: {path}. Run python src/train.py")
        _model = keras.models.load_model(path, compile=False)
    return _model


@app.get("/health")
def health() -> dict:
    path = model_path(ROOT)
    return {"status": "ok", "model_loaded": os.path.exists(path)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=422, detail="Upload a JPEG or PNG image")
    try:
        image = Image.open(io.BytesIO(await file.read()))
        model = _load_model()
        result = predict_image(model, image)
        result["grad_cam_available"] = True
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc
