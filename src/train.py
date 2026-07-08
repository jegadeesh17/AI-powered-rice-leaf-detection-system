"""One-command training script for rice leaf disease classification."""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")

import matplotlib

matplotlib.use("Agg")

import keras
import numpy as np
from keras.applications import EfficientNetB0
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
from PIL import Image
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.evaluation import plot_confusion_matrix
from src.model_builder import build_transfer_model

SPLIT_ROOT = os.path.join(ROOT, "data", "processed", "rice_leaf_split")
MODEL_DIR = os.path.join(ROOT, "models")
FINAL_MODEL = os.path.join(MODEL_DIR, "ai_system_rice_leaf_final.keras")
METRICS_JSON = os.path.join(ROOT, "reports", "metrics.json")
CM_PATH = os.path.join(ROOT, "visualizations", "confusion_matrix.png")
IMG_SIZE = 224


def _count_split_images() -> int:
    train_path = os.path.join(SPLIT_ROOT, "train")
    if not os.path.isdir(train_path):
        return 0
    total = 0
    for root, _dirs, files in os.walk(train_path):
        total += sum(1 for f in files if f.lower().endswith((".png", ".jpg", ".jpeg")))
    return total


def _looks_like_demo_data() -> bool:
    """Detect seed_demo_data.py artifacts (Class_split_N.png naming / tiny set)."""
    if _count_split_images() == 0:
        return False
    for root, _dirs, files in os.walk(SPLIT_ROOT):
        for name in files:
            if name.endswith("_train_0.png") or name.endswith("_val_0.png"):
                return True
    return _count_split_images() < 200


def _ensure_data(*, allow_demo: bool = False) -> None:
    image_count = _count_split_images()
    is_demo = _looks_like_demo_data() if image_count > 0 else False

    if image_count > 0:
        if is_demo and not allow_demo:
            raise SystemExit(
                "Found demo/fake images under data/processed/rice_leaf_split/.\n"
                "Download the real Mendeley dataset first:\n"
                "  python scripts/download_and_split_dataset.py --replace\n"
                "Or pass --demo only if you intentionally want the toy seed."
            )
        return

    if allow_demo:
        from scripts.seed_demo_data import main as seed_main

        seed_main()
        return

    raise SystemExit(
        "No training images found under data/processed/rice_leaf_split/.\n"
        "Download + split the real dataset:\n"
        "  pip install py7zr\n"
        "  python scripts/download_and_split_dataset.py --replace\n"
        "Or for a quick synthetic smoke test: python src/train.py --demo"
    )


def _load_split(split: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    root = os.path.join(SPLIT_ROOT, split)
    class_names = sorted(name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name)))
    images, labels = [], []
    for class_name in class_names:
        class_dir = os.path.join(root, class_name)
        label = class_names.index(class_name)
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                path = os.path.join(class_dir, fname)
                img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
                images.append(np.array(img, dtype=np.float32))
                labels.append(label)
    if not images:
        raise SystemExit(f"No images found in {root}. Run scripts/download_and_split_dataset.py first.")
    return np.stack(images), to_categorical(labels, num_classes=len(class_names)), class_names


def train(epochs: int = 5, batch_size: int = 8, fine_tune_epochs: int = 3, *, allow_demo: bool = False) -> dict:
    _ensure_data(allow_demo=allow_demo)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(CM_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(METRICS_JSON), exist_ok=True)

    x_train, y_train, class_names = _load_split("train")
    x_val, y_val, _ = _load_split("val")
    x_test, y_test, _ = _load_split("test")
    num_classes = len(class_names)

    model = build_transfer_model(EfficientNetB0, num_classes)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])

    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True, monitor="val_accuracy"),
        ReduceLROnPlateau(patience=2, factor=0.5),
    ]
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks)

    base = None
    for layer in model.layers:
        if hasattr(layer, "layers") and len(getattr(layer, "layers", [])) > 10:
            base = layer
            break
    if base is not None and fine_tune_epochs > 0:
        base.trainable = True
        for layer in base.layers[:-20]:
            layer.trainable = False
        model.compile(optimizer=keras.optimizers.Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy"])
        model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=fine_tune_epochs, batch_size=batch_size, callbacks=callbacks)

    preds = model.predict(x_test, verbose=0)
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(preds, axis=1)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    accuracy = float(report["accuracy"])
    plot_confusion_matrix(y_true, y_pred, class_names, save_path=CM_PATH)

    model.save(FINAL_MODEL)
    metrics = {
        "test_accuracy": round(accuracy, 4),
        "per_class": {
            name: {
                "precision": round(float(report[name]["precision"]), 4),
                "recall": round(float(report[name]["recall"]), 4),
                "f1": round(float(report[name]["f1-score"]), 4),
            }
            for name in class_names
        },
        "epochs": epochs,
        "fine_tune_epochs": fine_tune_epochs,
        "batch_size": batch_size,
        "classes": class_names,
    }
    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    from scripts.export_evaluation import main as export_main

    export_main(metrics)
    print(f"Saved model to {FINAL_MODEL}")
    print(f"Test accuracy: {accuracy:.4f}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train rice leaf disease classifier")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--fine-tune-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--demo", action="store_true", help="Fast demo run on seeded minimal data")
    args = parser.parse_args()
    if args.demo:
        args.epochs = 2
        args.fine_tune_epochs = 1
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        fine_tune_epochs=args.fine_tune_epochs,
        allow_demo=args.demo,
    )
