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


def evaluate(dataset: Path, bundle: Path, category: str, version: str, threads: int) -> dict:
    started = perf_counter()
    artifact = Artifact(bundle, category, version, threads)
    training_hashes = set(
        json.loads((bundle / SPLIT_FILE).read_text(encoding="utf-8"))["sha256"].values()
    )
    labels, scores, pixels, maps, rows = [], [], [], [], []
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
        pixels.append(mask_at_original_size(mask, image.size).ravel())
        maps.append(align_map(anomaly_map, image.size).ravel())
        rows.append({
            "image": path.relative_to(dataset).as_posix(),
            "target": "ANOMALY" if target else "NORMAL",
            "score": score,
            "label": classify(score, artifact.threshold),
        })
    pixel_targets = np.concatenate(pixels)
    pixel_scores = np.concatenate(maps)
    del pixels, maps
    return {
        "category": category,
        "version": version,
        "dataset_scope": "All PNGs in the supplied category test directory",
        "count": len(rows),
        "normal": labels.count(0),
        "defective": labels.count(1),
        "threshold": artifact.threshold,
        **summarize(labels, scores, artifact.threshold),
        "pixel_auroc": float(roc_auc_score(pixel_targets, pixel_scores)),
        "pixel_count": len(pixel_targets),
        "pixel_protocol": "Pooled original-resolution masks; bilinear aligned raw maps",
        "artifact": artifact.metadata,
        "runtime": {
            "os": platform.platform(),
            "cpu": platform.processor(),
            "threads": threads,
            "seconds": round(perf_counter() - started, 3),
            "command": ["python", "-m", "scripts.evaluate", *sys.argv[1:]],
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
    args = parser.parse_args()
    report = evaluate(args.dataset, args.artifact, args.category, args.version, args.threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"predictions", "artifact"}}, indent=2))


if __name__ == "__main__":
    main()
