import numpy as np
from PIL import Image

from visionguard.ml.visuals import render_comparison


def test_overlay_preserves_landmark_on_a_nonsquare_frame():
    image = Image.new("RGB", (1200, 600), (60, 60, 60))
    signal = np.zeros((32, 32), dtype=np.float32)
    signal[4:9, 23:27] = 2
    original, overlay = render_comparison(image, signal, 2)
    assert original.size == overlay.size == (800, 400)
    assert overlay.getpixel((625, 80))[0] > 185
    assert overlay.getpixel((80, 290)) == original.getpixel((80, 290))
    _, weaker = render_comparison(image, signal * 0.25, 2)
    assert weaker.getpixel((625, 80))[0] < overlay.getpixel((625, 80))[0] - 80
