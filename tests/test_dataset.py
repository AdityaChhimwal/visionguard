from pathlib import Path

import pytest

from scripts.dataset import normal_split, test_pairs as collect_test_pairs


def test_split_stable_and_excludes_other_folders(tmp_path: Path):
    train = tmp_path / "train/good"
    train.mkdir(parents=True)
    for i in reversed(range(10)):
        (train / f"{i}.png").write_bytes(bytes([i]))
    (tmp_path / "test/good").mkdir(parents=True)
    (tmp_path / "test/good/0.png").write_bytes(b"test")
    one = normal_split(tmp_path, 42, 0.2)
    assert one == normal_split(tmp_path, 42, 0.2)
    assert one["calibration"] != normal_split(tmp_path, 17, 0.2)["calibration"]
    assert len(one["fit"]) == 8 and len(one["calibration"]) == 2
    assert not set(one["fit"]) & set(one["calibration"])
    assert set(one["fit"]) | set(one["calibration"]) == set(one["sha256"])


def test_duplicate_normal_images_fail_before_splitting(tmp_path: Path):
    train = tmp_path / "train/good"
    train.mkdir(parents=True)
    for i in range(5):
        (train / f"{i}.png").write_bytes(b"duplicate")
    with pytest.raises(ValueError, match="Duplicate"):
        normal_split(tmp_path, 42, 0.4)


def test_defects_require_corresponding_masks(tmp_path: Path):
    for name in ("test/good/000.png", "test/crack/000.png"):
        path = tmp_path / name
        path.parent.mkdir(parents=True)
        path.touch()
    with pytest.raises(ValueError, match="Missing ground-truth mask"):
        collect_test_pairs(tmp_path)
    mask = tmp_path / "ground_truth/crack/000_mask.png"
    mask.parent.mkdir(parents=True)
    mask.touch()
    assert dict(collect_test_pairs(tmp_path))[tmp_path / "test/crack/000.png"] == mask
