"""Produce bounded previews with a calibration-fixed overlay."""

import base64
from io import BytesIO

import numpy as np
from PIL import Image

from visionguard.ml.images import align_map

PREVIEW_EDGE = 800


def render_comparison(
    image: Image.Image, anomaly_map: np.ndarray, heatmap_high: float
) -> tuple[Image.Image, Image.Image]:
    original = image.copy()
    original.thumbnail((PREVIEW_EDGE, PREVIEW_EDGE), Image.Resampling.LANCZOS)
    scaled = np.clip(align_map(anomaly_map, original.size) / heatmap_high, 0, 1)[..., None]
    # Low distances are transparent; warm colors indicate larger distances.
    amber = np.array([255, 193, 68], dtype=np.float32)
    red = np.array([249, 54, 35], dtype=np.float32)
    color = amber + (red - amber) * scaled
    alpha = 0.78 * scaled ** 2
    rgb = np.asarray(original, dtype=np.float32)
    overlay = Image.fromarray(np.rint(rgb * (1 - alpha) + color * alpha).astype(np.uint8))
    return original, overlay


def png_url(image: Image.Image) -> str:
    with BytesIO() as stream:
        image.save(stream, format="PNG", compress_level=3)
        return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def jpeg_url(image: Image.Image) -> str:
    with BytesIO() as stream:
        image.save(stream, format="JPEG", quality=88, subsampling=0)
        return "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode("ascii")
