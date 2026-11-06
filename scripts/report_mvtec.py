"""Summarize the complete set of measured MVTec AD evaluation reports."""

import argparse
import json
import math
from pathlib import Path
from statistics import fmean

from scripts.fetch_mvtec import CATEGORY_COUNTS

METRICS = ("image_auroc", "pixel_auroc", "precision", "recall", "f1")
DECISIONS = ("tn", "fp", "fn", "tp")


def summarize(reports: Path) -> str:
    lines = [
        "| Category | Fit / calibration normals | Test normal / defective | Image AUROC | "
        "Pixel AUROC (binned) | Precision | Recall | F1 | TN / FP / FN / TP |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    values = {name: [] for name in METRICS}
    totals = {name: 0 for name in DECISIONS}
    fit_total, calibration_total, normal_total, defective_total, pixel_total = 0, 0, 0, 0, 0
    for category, (training, normal, defective) in CATEGORY_COUNTS.items():
        path = reports / f"{category}-evaluation.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if (report["category"], report["count"], report["normal"], report["defective"]) != (
            category, normal + defective, normal, defective,
        ):
            raise ValueError(f"{path} does not cover the complete test split for {category}.")
        calibration_settings = report["artifact"]["calibration"]
        calibration = calibration_settings["count"]
        model = report["artifact"]["model"]
        if (model["calibration_fraction"], calibration_settings["target"]) != (0.2, 0.05):
            raise ValueError(f"{path} uses a different calibration protocol.")
        if calibration != math.ceil(training * model["calibration_fraction"]):
            raise ValueError(f"{path} has an inconsistent calibration split.")
        fit = training - calibration
        matrix = report["confusion_matrix"]
        if matrix["tn"] + matrix["fp"] != normal or matrix["fn"] + matrix["tp"] != defective:
            raise ValueError(f"{path} has an inconsistent confusion matrix.")
        if "65536 fixed bins" not in report["pixel_protocol"] or report["pixel_count"] <= 0:
            raise ValueError(f"{path} uses a different pixel evaluation protocol.")

        metrics = [float(report[name]) for name in METRICS]
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in metrics):
            raise ValueError(f"{path} contains an invalid metric.")
        for name, value in zip(METRICS, metrics, strict=True):
            values[name].append(value)
        for name in DECISIONS:
            totals[name] += matrix[name]
        fit_total += fit
        calibration_total += calibration
        normal_total += normal
        defective_total += defective
        pixel_total += report["pixel_count"]

        results = " | ".join(f"{value:.4f}" for value in metrics)
        decisions = " / ".join(str(matrix[name]) for name in DECISIONS)
        lines.append(
            f"| {category.replace('_', ' ').title()} | {fit} / {calibration} | "
            f"{normal} / {defective} | {results} | {decisions} |"
        )

    means = " | ".join(f"{fmean(values[name]):.4f}" for name in METRICS)
    decisions = " / ".join(f"{totals[name]:,}" for name in DECISIONS)
    lines.append(
        f"| **Macro mean / totals** | {fit_total:,} / {calibration_total:,} | "
        f"{normal_total:,} / {defective_total:,} | {means} | {decisions} |"
    )
    lines.extend([
        "",
        f"The reports cover all {normal_total + defective_total:,} test images and "
        f"{pixel_total:,} original-resolution pixels. Macro means weight each category equally. "
        "The last column sums decisions at the individual category thresholds.",
        "",
        "Each model uses disjoint normal images for fitting and calibration. "
        "The frozen threshold targets a 5% false-positive rate on its held-out normal split. "
        "This rate can differ on the test split. Precision, recall and F1 treat defects as positive.",
        "",
        "Pixel AUROC compares original-resolution masks with bilinearly aligned anomaly maps. "
        "It approximates score ranking with 65,536 fixed bins, using one image at a time. "
        "Scores in the same bin are treated as ties. These results apply to the MVTec AD "
        "benchmark and its imaging conditions.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize(args.reports)
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
