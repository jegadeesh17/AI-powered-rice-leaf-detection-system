import os

import keras
import numpy as np
from keras import layers, Sequential
from PIL import Image


def get_augmentation_layer():
    """Returns a sequential layer for image augmentation."""
    return Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
            layers.RandomContrast(0.2),
            layers.RandomBrightness(0.2),
        ],
        name="data_augmentation",
    )


class _DirectoryDataset:
    """Lightweight image directory loader without TensorFlow dependency."""

    def __init__(self, root: str, image_size: int = 224, batch_size: int = 8, shuffle: bool = True):
        self.image_size = image_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.class_names = sorted(
            name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))
        )
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.samples: list[tuple[str, int]] = []
        for class_name in self.class_names:
            class_dir = os.path.join(root, class_name)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(class_dir, fname), self.class_to_idx[class_name]))

    def __len__(self) -> int:
        return max(1, int(np.ceil(len(self.samples) / self.batch_size)))

    def _batch_indices(self):
        indices = np.arange(len(self.samples))
        if self.shuffle:
            np.random.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            yield indices[start : start + self.batch_size]

    def __iter__(self):
        for batch_idx in self._batch_indices():
            images, labels = [], []
            for idx in batch_idx:
                path, label = self.samples[idx]
                img = Image.open(path).convert("RGB").resize((self.image_size, self.image_size))
                images.append(np.array(img, dtype=np.float32))
                one_hot = np.zeros(len(self.class_names), dtype=np.float32)
                one_hot[label] = 1.0
                labels.append(one_hot)
            yield np.stack(images), np.stack(labels)


def load_datasets(train_path, val_path, test_path, img_size=224, batch_size=8):
    """Loads training, validation, and test datasets from class folders."""
    train_ds = _DirectoryDataset(train_path, img_size, batch_size, shuffle=True)
    val_ds = _DirectoryDataset(val_path, img_size, batch_size, shuffle=False)
    test_ds = _DirectoryDataset(test_path, img_size, batch_size, shuffle=False)
    return train_ds, val_ds, test_ds
