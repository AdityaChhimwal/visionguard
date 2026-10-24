import numpy as np
import pytest

from visionguard.images import ImageInputError, decode_image_bytes, prepare_batch


def test_decode_image_bytes_converts_to_rgb(image_bytes: bytes) -> None:
    image = decode_image_bytes(image_bytes)

    assert image.mode == "RGB"
    assert image.size == (12, 8)


def test_decode_image_bytes_rejects_corrupted_input() -> None:
    with pytest.raises(ImageInputError, match="valid readable"):
        decode_image_bytes(b"not an image")


def test_prepare_batch_has_expected_shape_and_vgg_channel_order(image_bytes: bytes) -> None:
    image = decode_image_bytes(image_bytes)
    batch = prepare_batch([image], (4, 3))

    assert batch.shape == (1, 3, 4, 3)
    assert batch.dtype == np.float32
    assert np.isclose(batch[0, 0, 0, 0], 60 - 103.939)
