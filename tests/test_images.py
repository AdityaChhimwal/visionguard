from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from visionguard.ml.images import MEAN, STD, ImageRejected, decode_image, preprocess


def encoded(image, fmt="PNG", **kwargs):
    output = BytesIO()
    image.save(output, fmt, **kwargs)
    return output.getvalue()


def test_exif_orientation_is_applied_before_model_resize():
    exif = Image.Exif()
    exif[274] = 6
    image = decode_image(encoded(Image.new("RGB", (23, 12), "white"), "JPEG", exif=exif))
    assert image.size == (12, 23)
    tensor = preprocess(image, 32)
    assert tensor.shape == (1, 3, 32, 32) and tensor.dtype == np.float32
    np.testing.assert_allclose(tensor[0, :, 16, 16], (1 - MEAN) / STD)


def test_grayscale_becomes_rgb_and_png_limits_are_enforced():
    payload = encoded(Image.new("L", (31, 18), 42))
    assert decode_image(payload).getpixel((1, 1)) == (42, 42, 42)
    with pytest.raises(ImageRejected) as exc:
        decode_image(payload, max_pixels=500)
    assert exc.value.code == "pixel_limit"


@pytest.mark.parametrize("payload", [
    b"plain text",
    encoded(Image.new("RGB", (10, 10)), "GIF"),
    encoded(Image.new("RGB", (10, 10)))[:-20],
])
def test_unsupported_malformed_and_truncated_uploads(payload):
    with pytest.raises(ImageRejected) as exc:
        decode_image(payload)
    assert exc.value.code == "invalid_image"
