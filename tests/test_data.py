from pathlib import Path

import pytest

from visionguard.data import DatasetValidationError, image_files, test_records as collect_test_records, training_images, validate_dataset


def test_validate_dataset_reports_expected_mvtec_structure(mvtec_root: Path) -> None:
    summary = validate_dataset(mvtec_root, ["widget"])

    assert summary.valid
    assert summary.categories["widget"].train_good_images == 2
    assert summary.categories["widget"].test_groups == {"crack": 1, "good": 1}
    records = collect_test_records(mvtec_root, "widget")
    assert [(record.defect_type, record.label) for record in records] == [("crack", 1), ("good", 0)]
    assert records[0].mask_path is not None


def test_validate_dataset_rejects_missing_required_split(mvtec_root: Path) -> None:
    (mvtec_root / "widget" / "ground_truth").rename(mvtec_root / "widget" / "masks")

    with pytest.raises(DatasetValidationError, match="missing ground-truth directory"):
        validate_dataset(mvtec_root, ["widget"])


def test_validate_dataset_rejects_missing_defect_mask(mvtec_root: Path) -> None:
    (mvtec_root / "widget" / "ground_truth" / "crack" / "000_mask.png").unlink()

    with pytest.raises(DatasetValidationError, match="missing ground-truth mask"):
        validate_dataset(mvtec_root, ["widget"])


def test_image_files_only_returns_supported_direct_children(tmp_path: Path) -> None:
    valid = tmp_path / "valid.png"
    valid.write_bytes(b"x")
    (tmp_path / "note.txt").write_text("not an image", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "inside.jpg").write_bytes(b"x")

    assert image_files(tmp_path) == [valid]


def test_training_images_requires_two_normal_examples(mvtec_root: Path) -> None:
    (mvtec_root / "widget" / "train" / "good" / "001.png").unlink()

    with pytest.raises(DatasetValidationError, match="at least two"):
        training_images(mvtec_root, "widget")
