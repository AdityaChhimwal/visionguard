"""Evaluate the frozen serving graph on the untouched category test directory."""

import argparse
import json
import platform
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageOps
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

from scripts.dataset import sha256, test_pairs
from visionguard.ml.artifacts import SPLIT_FILE, Artifact
from visionguard.ml.calibration import classify
from visionguard.ml.images import align_map, decode_image


def mask_at_original_size(mask: Path | None, size: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.zeros((size[1], size[0]), dtype=bool)
    with Image.open(mask) as opened:
        oriented = ImageOps.exif_transpose(opened).convert("L")
    if oriented.size != size:
        raise ValueError(f"Ground truth dimensions differ from image: {mask.name}")
    values = np.asarray(oriented)
    if not np.isin(values, (0, 255)).all() or not values.any():
        raise ValueError(f"Expected a nonempty binary mask: {mask.name}")
    return values > 0


def summarize(labels: list[int], scores: list[float], threshold: float) -> dict:
    predicted = [int(classify(score, threshold) == "ANOMALY") for score in scores]
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel().tolist()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, average="binary", zero_division=0
    )
    return {
        "image_auroc": float(roc_auc_score(labels, scores)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


class PixelAurocHistogram:
    """Count aligned positive and negative pixels in fixed monotonic score bins."""

    def __init__(self, bins: int, scale: float):
        if bins < 2:
            raise ValueError("Pixel AUROC needs at least two bins.")
        self.positive = np.zeros(bins, dtype=np.int64)
        self.negative = np.zeros(bins, dtype=np.int64)
        self.scale = scale

    def add(self, mask: np.ndarray, scores: np.ndarray) -> None:
        # A monotonic transform maps distances in [0, infinity) to [0, 1).
        # The calibration scale is frozen and requires no test-set range fit.
        scores = np.maximum(scores.ravel(), 0)
        bins = self.positive.size
        index = np.minimum((scores / (scores + self.scale) * bins).astype(np.int32), bins - 1)
        positive = np.bincount(index[mask.ravel()], minlength=bins)
        self.positive += positive
        self.negative += np.bincount(index, minlength=bins) - positive

    def auroc(self) -> tuple[float, int]:
        positive_count = int(self.positive.sum())
        negative_count = int(self.negative.sum())
        if not positive_count or not negative_count:
            raise ValueError("Pixel AUROC requires positive and negative pixels.")
        # A positive beats every negative in a lower bin and ties half in the same bin.
        ranks = self.negative.cumsum() - 0.5 * self.negative
        area = float(self.positive @ ranks) / (positive_count * negative_count)
        return area, positive_count + negative_count


def evaluate(
    dataset: Path, bundle: Path, category: str, version: str, threads: int,
    *, pixel_bins: int | None = None, command: list[str] | None = None,
) -> dict:
    started = perf_counter()
    artifact = Artifact(bundle, category, version, threads)
    training_hashes = set(
        json.loads((bundle / SPLIT_FILE).read_text(encoding="utf-8"))["sha256"].values()
    )
    labels, scores, pixels, maps, rows = [], [], [], [], []
    histogram = (
        PixelAurocHistogram(pixel_bins, artifact.heatmap_high) if pixel_bins is not None else None
    )
    seen = set()
    for path, mask in test_pairs(dataset):
        digest = sha256(path)
        if digest in training_hashes or digest in seen:
            raise ValueError(f"Duplicate image content at {path.relative_to(dataset)}")
        seen.add(digest)
        image = decode_image(path.read_bytes())
        score, anomaly_map = artifact.predict(image)
        target = int(mask is not None)
        labels.append(target)
        scores.append(score)
        pixel_mask = mask_at_original_size(mask, image.size)
        aligned = align_map(anomaly_map, image.size)
        if histogram is not None:
            histogram.add(pixel_mask, aligned)
        else:
            pixels.append(pixel_mask.ravel())
            maps.append(aligned.ravel())
        rows.append({
            "image": path.relative_to(dataset).as_posix(),
            "target": "ANOMALY" if target else "NORMAL",
            "score": score,
            "label": classify(score, artifact.threshold),
        })
    if histogram is not None:
        pixel_auroc, pixel_count = histogram.auroc()
        pixel_protocol = (
            "Pooled original-resolution masks and bilinear aligned raw maps. "
            f"Approximate AUROC from {pixel_bins} fixed bins of score / (score + heatmap_high)."
        )
    else:
        pixel_targets = np.concatenate(pixels)
        pixel_scores = np.concatenate(maps)
        del pixels, maps
        pixel_auroc = float(roc_auc_score(pixel_targets, pixel_scores))
        pixel_count = len(pixel_targets)
        pixel_protocol = "Pooled original-resolution masks; bilinear aligned raw maps"
    return {
        "category": category,
        "version": version,
        "dataset_scope": "All PNGs in the supplied category test directory",
        "count": len(rows),
        "normal": labels.count(0),
        "defective": labels.count(1),
        "threshold": artifact.threshold,
        **summarize(labels, scores, artifact.threshold),
        "pixel_auroc": pixel_auroc,
        "pixel_count": pixel_count,
        "pixel_protocol": pixel_protocol,
        "artifact": artifact.metadata,
        "runtime": {
            "os": platform.platform(),
            "cpu": platform.processor(),
            "threads": threads,
            "seconds": round(perf_counter() - started, 3),
            "command": command or ["python", "-m", "scripts.evaluate", *sys.argv[1:]],
        },
        "predictions": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "artifact", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--pixel-bins", type=int, help="Use this many fixed AUROC bins")
    args = parser.parse_args()
    report = evaluate(
        args.dataset, args.artifact, args.category, args.version, args.threads,
        pixel_bins=args.pixel_bins,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"predictions", "artifact"}}, indent=2))


if __name__ == "__main__":
    main()
