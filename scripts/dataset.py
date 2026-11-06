"""Inventory MVTec AD and create a normal-only manifest."""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def normal_split(root: Path, seed: int, fraction: float) -> dict:
    paths = sorted(p.relative_to(root).as_posix() for p in root.glob("train/good/*.png"))
    if not 0 < fraction < 1 or len(paths) < 4:
        raise ValueError("Need at least four train/good PNGs and a fraction between zero and one.")
    digests = {name: sha256(root / name) for name in paths}
    if len(set(digests.values())) != len(paths):
        raise ValueError("Duplicate normal image content would compromise the split.")
    random.Random(seed).shuffle(paths)
    size = math.ceil(len(paths) * fraction)
    if size < 2 or len(paths) - size < 2:
        raise ValueError("Need at least two images on both sides of the split.")
    return {
        "seed": seed,
        "calibration_fraction": fraction,
        "fit": sorted(paths[size:]),
        "calibration": sorted(paths[:size]),
        "sha256": digests,
    }


def test_pairs(root: Path) -> list[tuple[Path, Path | None]]:
    pairs = []
    for image in sorted(root.glob("test/*/*.png")):
        defect = image.parent.name
        mask = root / "ground_truth" / defect / (image.stem + "_mask.png")
        if defect != "good" and not mask.is_file():
            raise ValueError(f"Missing ground-truth mask for {image.relative_to(root)}")
        pairs.append((image, None if defect == "good" else mask))
    if {mask is not None for _, mask in pairs} != {False, True}:
        raise ValueError("Test set must include both good and defective images.")
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Path to category directory")
    args = parser.parse_args()
    manifest = normal_split(args.dataset, 42, 0.2)
    pairs = test_pairs(args.dataset)
    print(json.dumps({
        "fit": len(manifest["fit"]),
        "calibration": len(manifest["calibration"]),
        "test_normal": sum(mask is None for _, mask in pairs),
        "test_defective": sum(mask is not None for _, mask in pairs),
    }, indent=2))


if __name__ == "__main__":
    main()
