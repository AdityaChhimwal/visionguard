from pathlib import Path

from visionguard.data import ImageRecord
from visionguard.evaluation import ScoredRecord, evaluate_image_level


def test_evaluation_computes_image_metrics_without_tuning_threshold() -> None:
    rows = [
        ScoredRecord(ImageRecord(Path("normal.png"), 0, "good"), 0.1),
        ScoredRecord(ImageRecord(Path("defect.png"), 1, "crack"), 0.9),
    ]

    report = evaluate_image_level(rows, threshold=0.5)

    assert report["auroc"] == 1.0
    assert report["confusion_matrix"] == {
        "true_negative": 1,
        "false_positive": 0,
        "false_negative": 0,
        "true_positive": 1,
    }
