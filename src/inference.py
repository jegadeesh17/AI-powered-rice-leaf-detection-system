"""Shared inference helpers for Streamlit and FastAPI."""

from __future__ import annotations

import os

import numpy as np
from PIL import Image

CLASSES = ["Bacterialblight", "Blast", "Brownspot", "Tungro"]
DISPLAY_NAMES = {
    "Bacterialblight": "Bacterial Leaf Blight",
    "Blast": "Rice Blast",
    "Brownspot": "Brown Spot",
    "Tungro": "Rice Tungro Disease",
}
IMG_SIZE = 224
MODEL_FILENAME = "ai_system_rice_leaf_final.keras"


def model_path(root: str | None = None) -> str:
    base = root or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(base, "models", MODEL_FILENAME)


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.expand_dims(np.array(image), axis=0)


def predict_image(model, image: Image.Image) -> dict:
    tensor = preprocess_image(image)
    probs = model.predict(tensor, verbose=0)[0]
    idx = int(np.argmax(probs))
    label = CLASSES[idx]
    return {
        "disease_class": label,
        "display_name": DISPLAY_NAMES[label],
        "confidence": float(probs[idx]),
        "probabilities": {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))},
    }
