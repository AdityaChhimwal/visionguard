from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image


def png_bytes(color: tuple[int, int, int] = (20, 40, 60), size: tuple[int, int] = (12, 8)) -> bytes:
    image = Image.new("RGB", size, color=color)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def write_png(path: Path, color: tuple[int, int, int] = (20, 40, 60)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png_bytes(color))
    return path


@pytest.fixture
def image_bytes() -> bytes:
    return png_bytes()


@pytest.fixture
def mvtec_root(tmp_path: Path) -> Path:
    root = tmp_path / "mvtec"
    write_png(root / "widget" / "train" / "good" / "000.png")
    write_png(root / "widget" / "train" / "good" / "001.png", (80, 90, 100))
    write_png(root / "widget" / "test" / "good" / "000.png")
    write_png(root / "widget" / "test" / "crack" / "000.png")
    write_png(root / "widget" / "ground_truth" / "crack" / "000_mask.png")
    return root
