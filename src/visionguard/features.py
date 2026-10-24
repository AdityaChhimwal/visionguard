"""Pretrained visual feature extraction for the baseline detector."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .images import load_image, prepare_batch


class FeatureExtractorError(RuntimeError):
    """Raised when the configured pretrained feature extractor cannot be used."""


class VGG16FeatureExtractor:
    """VGG16 block-5 features with global average pooling (512 values/image)."""

    identifier = "vgg16_block5_global_average"

    def __init__(self, image_size: tuple[int, int] = (224, 224), weights: str = "imagenet") -> None:
        if image_size != (224, 224):
            raise FeatureExtractorError("The VGG16 baseline requires image_size [224, 224]")
        if weights != "imagenet":
            raise FeatureExtractorError("The baseline currently supports only VGG16 ImageNet weights")
        self.image_size = image_size
        self.weights = weights
        self._model = self._build_model()

    @staticmethod
    def _build_model():
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise FeatureExtractorError("TensorFlow is required for the VGG16 baseline") from exc
        try:
            base = tf.keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        except Exception as exc:  # TensorFlow errors differ across platforms and caches.
            raise FeatureExtractorError(
                "Unable to load VGG16 ImageNet weights. Download them once during training, then retain the local cache."
            ) from exc
        return tf.keras.Sequential(
            [base, tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")],
            name="vgg16_block5_global_average",
        )

    def extract_images(self, images: Sequence) -> np.ndarray:
        """Extract a 2-D float32 embedding matrix from decoded PIL images."""
        batch = prepare_batch(list(images), self.image_size)
        embeddings = self._model.predict(batch, verbose=0)
        return np.asarray(embeddings, dtype=np.float32)

    def extract_paths(self, paths: Sequence[Path], batch_size: int = 16) -> np.ndarray:
        """Extract embeddings from paths in bounded batches."""
        if not paths:
            raise FeatureExtractorError("Cannot extract features from an empty path collection")
        if batch_size <= 0:
            raise FeatureExtractorError("batch_size must be positive")
        batches: list[np.ndarray] = []
        for start in range(0, len(paths), batch_size):
            images = [load_image(path) for path in paths[start : start + batch_size]]
            batches.append(self.extract_images(images))
        return np.concatenate(batches, axis=0)


def create_feature_extractor(
    identifier: str, image_size: tuple[int, int] = (224, 224), weights: str = "imagenet"
) -> VGG16FeatureExtractor:
    """Build a feature extractor recorded in an artifact's metadata."""
    if identifier != VGG16FeatureExtractor.identifier:
        raise FeatureExtractorError(f"Unsupported feature extractor: {identifier}")
    return VGG16FeatureExtractor(image_size=image_size, weights=weights)
