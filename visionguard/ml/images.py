"""The same orientation, resize and normalization are used at every stage."""

from io import BytesIO

import numpy as np
from PIL import Image, ImageOps

FORMATS = ("JPEG", "PNG", "WEBP")
MAX_PIXELS = 12_000_000
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
PREPROCESSING = {
    "version": 1,
    "orientation": "EXIF transpose",
    "color": "RGB",
    "resize": "whole frame, bilinear, square",
    "mean": MEAN.tolist(),
    "std": STD.tolist(),
}


class ImageRejected(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def decode_image(data: bytes, max_pixels: int = MAX_PIXELS) -> Image.Image:
    try:
        with Image.open(BytesIO(data), formats=FORMATS) as opened:
            if opened.width * opened.height > max_pixels:
                raise ImageRejected("pixel_limit", "The image has too many pixels.")
            if getattr(opened, "n_frames", 1) != 1:
                raise ImageRejected("invalid_image", "Use a single still image.")
            opened.verify()
        with Image.open(BytesIO(data), formats=FORMATS) as opened:
            return ImageOps.exif_transpose(opened).convert("RGB")
    except ImageRejected:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageRejected("pixel_limit", "The image has too many pixels.") from exc
    except (OSError, ValueError, SyntaxError, EOFError, IndexError) as exc:
        raise ImageRejected("invalid_image", "Choose a valid JPEG, PNG, or WebP image.") from exc


def preprocess(image: Image.Image, size: int) -> np.ndarray:
    resized = image.resize((size, size), Image.Resampling.BILINEAR)
    rgb = np.asarray(resized, dtype=np.float32) / np.float32(255)
    return np.ascontiguousarray(((rgb - MEAN) / STD).transpose(2, 0, 1)[None])


def align_map(anomaly_map: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(
        Image.fromarray(anomaly_map.astype(np.float32, copy=False)).resize(
            size, Image.Resampling.BILINEAR
        )
    )
