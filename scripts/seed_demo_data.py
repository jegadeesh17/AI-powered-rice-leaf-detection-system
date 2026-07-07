"""Create a minimal labeled image split when the Mendeley dataset is unavailable."""

from __future__ import annotations

import os
import random

from PIL import Image, ImageDraw

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPLIT_ROOT = os.path.join(ROOT, "data", "processed", "rice_leaf_split")
CLASSES = ["Bacterialblight", "Blast", "Brownspot", "Tungro"]
PALETTES = {
    "Bacterialblight": (50, 140, 60),
    "Blast": (45, 120, 55),
    "Brownspot": (55, 130, 50),
    "Tungro": (190, 160, 40),
}


def _make_leaf(class_name: str, seed: int) -> Image.Image:
    random.seed(seed)
    base = PALETTES[class_name]
    img = Image.new("RGB", (224, 224), color=base)
    draw = ImageDraw.Draw(img)
    draw.line([(112, 0), (112, 224)], fill=(30, 90, 40), width=4)
    if class_name == "Bacterialblight":
        draw.rectangle([(0, 0), (40, 224)], fill=(210, 210, 150))
    elif class_name == "Blast":
        for x, y in [(60, 50), (150, 130)]:
            draw.ellipse([(x - 18, y - 10), (x + 18, y + 10)], fill=(130, 80, 60))
    elif class_name == "Brownspot":
        for x, y in [(40, 30), (170, 45), (100, 85)]:
            draw.ellipse([(x - 8, y - 8), (x + 8, y + 8)], fill=(90, 45, 25))
    return img


def main(images_per_split: int = 12) -> str:
    for split in ("train", "val", "test"):
        for class_name in CLASSES:
            out_dir = os.path.join(SPLIT_ROOT, split, class_name)
            os.makedirs(out_dir, exist_ok=True)
            for i in range(images_per_split):
                path = os.path.join(out_dir, f"{class_name}_{split}_{i}.png")
                if not os.path.exists(path):
                    _make_leaf(class_name, hash((split, class_name, i)) % 10_000).save(path)
    print(f"Demo dataset ready at {SPLIT_ROOT}")
    return SPLIT_ROOT


if __name__ == "__main__":
    main()
