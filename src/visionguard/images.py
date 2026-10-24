"""Safe, shared image decoding and model preprocessing."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError

SUPPORTED_UPLOAD_FORMATS = frozenset({"PNG", "JPEG", "BMP"})
ImageFile.LOAD_TRUNCATED_IMAGES = False


class ImageInputError(ValueError):
    """Raised for an image that cannot safely enter the inference pipeline."""


def decode_image_bytes(
    image_bytes: bytes,
    *,
    max_pixels: int = 40_000_000,
    max_dimension: int = 10_000,
) -> Image.Image:
    """Validate bytes as a supported image and return an owned RGB image."""
    if not image_bytes:
        raise ImageInputError("Image upload is empty")
    try:
        with Image.open(BytesIO(image_bytes)) as source:
            source.load()
            if source.format not in SUPPORTED_UPLOAD_FORMATS:
                raise ImageInputError(
                    f"Unsupported image format '{source.format}'. Supported formats: PNG, JPEG, BMP"
                )
            if source.width <= 0 or source.height <= 0:
                raise ImageInputError("Image has invalid dimensions")
            if source.width > max_dimension or source.height > max_dimension:
                raise ImageInputError(f"Image dimensions exceed the {max_dimension}px limit")
            if source.width * source.height > max_pixels:
                raise ImageInputError("Image exceeds the pixel safety limit")
            return source.convert("RGB").copy()
    except ImageInputError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageInputError("Upload is not a valid readable image") from exc


def load_image(path: str | Path) -> Image.Image:
    """Load an on-disk image through the same validation path as API uploads."""
    image_path = Path(path)
    try:
        return decode_image_bytes(image_path.read_bytes())
    except FileNotFoundError as exc:
        raise ImageInputError(f"Image does not exist: {image_path}") from exc


def prepare_batch(images: list[Image.Image], image_size: tuple[int, int]) -> np.ndarray:
    """Resize RGB images deterministically and apply VGG16 preprocessing."""
    if not images:
        raise ImageInputError("Cannot preprocess an empty image batch")
    resized = [image.resize(image_size, Image.Resampling.LANCZOS) for image in images]
    array = np.asarray([np.asarray(image, dtype=np.float32) for image in resized], dtype=np.float32)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ImageInputError("Preprocessing expected RGB images with three channels")
    # Equivalent to keras.applications.vgg16.preprocess_input: RGB -> BGR and
    # ImageNet channel means are subtracted. Keeping it here avoids duplicated
    # training/inference preprocessing paths.
    array = array[..., ::-1]
    array[..., 0] -= 103.939
    array[..., 1] -= 116.779
    array[..., 2] -= 123.68
    return array
