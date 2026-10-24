"""Image-level evaluation and concise error analysis for MVTec test data."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .data import ImageRecord
from .detector import is_anomaly


@dataclass(frozen=True)
class ScoredRecord:
    """Evaluation-only pairing of an MVTec image record and detector score."""

    record: ImageRecord
    score: float


def _safe_round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


def evaluate_image_level(scored_records: Iterable[ScoredRecord], threshold: float) -> dict[str, object]:
    """Compute image-level metrics; no test labels are used to change threshold."""
    rows = list(scored_records)
    if not rows:
        raise ValueError("Cannot evaluate an empty collection")
    labels = np.asarray([row.record.label for row in rows], dtype=int)
    scores = np.asarray([row.score for row in rows], dtype=float)
    predictions = np.asarray([int(is_anomaly(score, threshold)) for score in scores], dtype=int)

    true_positive = int(np.sum((labels == 1) & (predictions == 1)))
    false_positive = int(np.sum((labels == 0) & (predictions == 1)))
    true_negative = int(np.sum((labels == 0) & (predictions == 0)))
    false_negative = int(np.sum((labels == 1) & (predictions == 0)))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    roc_auc: float | None = None
    average_precision: float | None = None
    if len(np.unique(labels)) == 2:
        from sklearn.metrics import average_precision_score, roc_auc_score

        roc_auc = float(roc_auc_score(labels, scores))
        average_precision = float(average_precision_score(labels, scores))

    by_defect: dict[str, list[float]] = {}
    for row in rows:
        by_defect.setdefault(row.record.defect_type, []).append(row.score)
    score_summaries = {
        name: {"count": len(values), "mean": _safe_round(float(np.mean(values))), "median": _safe_round(float(np.median(values)))}
        for name, values in sorted(by_defect.items())
    }
    false_positives = sorted(
        (row for row in rows if row.record.label == 0 and is_anomaly(row.score, threshold)),
        key=lambda row: row.score,
        reverse=True,
    )[:10]
    false_negatives = sorted(
        (row for row in rows if row.record.label == 1 and not is_anomaly(row.score, threshold)),
        key=lambda row: row.score,
    )[:10]

    def samples(values: Iterable[ScoredRecord]) -> list[dict[str, object]]:
        return [
            {"path": str(item.record.path), "defect_type": item.record.defect_type, "score": _safe_round(item.score)}
            for item in values
        ]

    return {
        "protocol": "Image-level MVTec test evaluation. Test labels are used only for these metrics, not fitting or threshold selection.",
        "threshold": _safe_round(threshold),
        "sample_count": len(rows),
        "normal_count": int(np.sum(labels == 0)),
        "anomaly_count": int(np.sum(labels == 1)),
        "auroc": _safe_round(roc_auc),
        "average_precision": _safe_round(average_precision),
        "precision_at_threshold": _safe_round(precision),
        "recall_at_threshold": _safe_round(recall),
        "f1_at_threshold": _safe_round(f1),
        "confusion_matrix": {"true_negative": true_negative, "false_positive": false_positive, "false_negative": false_negative, "true_positive": true_positive},
        "score_summary_by_test_group": score_summaries,
        "error_analysis": {"highest_scoring_false_positives": samples(false_positives), "lowest_scoring_false_negatives": samples(false_negatives)},
    }


def save_evaluation(report: dict[str, object], output_dir: str | Path, category: str) -> Path:
    """Persist evaluation/error-analysis JSON in an ignored output directory."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{category}_image_level_evaluation.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
