"""MVTec AD ingestion, validation, and benchmark record discovery."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError

SUPPORTED_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp"})


class DatasetValidationError(ValueError):
    """Raised when a dataset cannot safely be used for a VisionGuard run."""


@dataclass(frozen=True)
class ImageRecord:
    """One MVTec test image and its evaluation-only label information."""

    path: Path
    label: int
    defect_type: str
    mask_path: Path | None = None


@dataclass
class CategorySummary:
    """Validation findings for one MVTec category."""

    category: str
    train_good_images: int = 0
    test_images: int = 0
    test_groups: dict[str, int] = field(default_factory=dict)
    unreadable_images: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class DatasetSummary:
    """Structured dataset validation result."""

    root: Path
    categories: dict[str, CategorySummary] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "valid": self.valid,
            "errors": self.errors,
            "categories": {name: value.as_dict() for name, value in self.categories.items()},
        }


def image_files(directory: Path) -> list[Path]:
    """Return supported direct child images in stable order."""
    if not directory.is_dir():
        return []
    return sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS),
        key=lambda path: path.name,
    )


def discover_categories(dataset_root: str | Path) -> list[str]:
    """Find candidate categories beneath an MVTec dataset root."""
    root = Path(dataset_root)
    if not root.is_dir():
        raise DatasetValidationError(f"Dataset root does not exist or is not a directory: {root}")
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def _verify_image(path: Path) -> str | None:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.width <= 0 or image.height <= 0:
                return "image has non-positive dimensions"
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return str(exc)
    return None


def validate_dataset(
    dataset_root: str | Path,
    categories: Iterable[str] | None = None,
    *,
    verify_images: bool = True,
    strict: bool = True,
) -> DatasetSummary:
    """Validate the MVTec split structure without modifying dataset contents.

    The expected protocol is normal-only ``train/good`` images and a ``test``
    directory containing ``good`` plus one or more defect groups. Ground-truth
    masks are required for defect groups but never used as model inputs.
    """
    root = Path(dataset_root)
    available = discover_categories(root)
    requested = list(categories) if categories is not None else available
    summary = DatasetSummary(root=root)

    if not requested:
        summary.errors.append(f"No category directories found under dataset root: {root}")

    for category in requested:
        category_path = root / category
        if category not in available:
            summary.errors.append(f"Category '{category}' does not exist under {root}")
            continue

        train_good = category_path / "train" / "good"
        test_root = category_path / "test"
        ground_truth = category_path / "ground_truth"
        category_summary = CategorySummary(category=category)
        summary.categories[category] = category_summary

        if not train_good.is_dir():
            summary.errors.append(f"{category}: missing normal training directory: {train_good}")
            continue
        if not test_root.is_dir():
            summary.errors.append(f"{category}: missing test directory: {test_root}")
            continue
        if not ground_truth.is_dir():
            summary.errors.append(f"{category}: missing ground-truth directory: {ground_truth}")
            continue

        train_images = image_files(train_good)
        category_summary.train_good_images = len(train_images)
        if not train_images:
            summary.errors.append(f"{category}: train/good contains no supported images")

        test_groups = sorted(path for path in test_root.iterdir() if path.is_dir())
        if not test_groups:
            summary.errors.append(f"{category}: test contains no split directories")
            continue
        if not (test_root / "good").is_dir():
            summary.errors.append(f"{category}: test/good is missing")

        paths_to_verify = list(train_images)
        for group_path in test_groups:
            group_images = image_files(group_path)
            category_summary.test_groups[group_path.name] = len(group_images)
            category_summary.test_images += len(group_images)
            paths_to_verify.extend(group_images)
            if not group_images:
                summary.errors.append(f"{category}: test/{group_path.name} contains no supported images")
            if group_path.name != "good" and not (ground_truth / group_path.name).is_dir():
                summary.errors.append(
                    f"{category}: missing ground-truth directory for defect group '{group_path.name}'"
                )
            if group_path.name != "good" and (ground_truth / group_path.name).is_dir():
                for image_path in group_images:
                    expected_mask = ground_truth / group_path.name / f"{image_path.stem}_mask.png"
                    if not expected_mask.is_file():
                        summary.errors.append(
                            f"{category}: missing ground-truth mask for {group_path.name}/{image_path.name}: {expected_mask}"
                        )
                    elif verify_images:
                        paths_to_verify.append(expected_mask)

        if verify_images:
            for image_path in paths_to_verify:
                error = _verify_image(image_path)
                if error:
                    category_summary.unreadable_images.append(str(image_path))
                    summary.errors.append(f"{category}: unreadable image {image_path}: {error}")

    if strict and summary.errors:
        raise DatasetValidationError("Dataset validation failed:\n- " + "\n- ".join(summary.errors))
    return summary


def training_images(dataset_root: str | Path, category: str) -> list[Path]:
    """Return the normal-only MVTec training images for *category*."""
    images = image_files(Path(dataset_root) / category / "train" / "good")
    if len(images) < 2:
        raise DatasetValidationError(
            f"{category}: at least two normal training images are required for nearest-neighbor scoring"
        )
    return images


def test_records(dataset_root: str | Path, category: str) -> list[ImageRecord]:
    """Return labeled test records for evaluation only.

    Labels and mask paths intentionally live in this evaluation helper instead
    of the prediction pipeline, preventing ground truth from influencing
    inference decisions.
    """
    category_path = Path(dataset_root) / category
    records: list[ImageRecord] = []
    for group_path in sorted((category_path / "test").iterdir(), key=lambda path: path.name):
        if not group_path.is_dir():
            continue
        is_good = group_path.name == "good"
        for image_path in image_files(group_path):
            mask_path = None
            if not is_good:
                candidate = category_path / "ground_truth" / group_path.name / f"{image_path.stem}_mask.png"
                if candidate.is_file():
                    mask_path = candidate
            records.append(
                ImageRecord(path=image_path, label=0 if is_good else 1, defect_type=group_path.name, mask_path=mask_path)
            )
    if not records:
        raise DatasetValidationError(f"{category}: no supported test images found")
    return records


def count_by_extension(paths: Iterable[Path]) -> dict[str, int]:
    """Small utility useful in validation reports and tests."""
    return dict(sorted(Counter(path.suffix.lower() for path in paths).items()))
