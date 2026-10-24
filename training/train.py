"""Train the normal-only VGG16 nearest-neighbor VisionGuard baseline."""

from __future__ import annotations

import argparse
import logging
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

from visionguard.artifacts import save_artifact
from visionguard.config import TrainingConfig, load_config
from visionguard.data import test_records, training_images, validate_dataset
from visionguard.detector import NearestNeighborDetector, select_normal_threshold
from visionguard.evaluation import ScoredRecord, evaluate_image_level, save_evaluation
from visionguard.features import create_feature_extractor

LOGGER = logging.getLogger("visionguard.training")


def train(config: TrainingConfig, *, evaluate: bool = True) -> tuple[Path, Path | None]:
    """Fit, persist, and optionally evaluate a detector without test-set fitting."""
    validation = validate_dataset(config.dataset_root, [config.category], verify_images=True, strict=True)
    category_summary = validation.categories[config.category]
    LOGGER.info(
        "Validated category=%s normal_training_images=%s test_images=%s",
        config.category,
        category_summary.train_good_images,
        category_summary.test_images,
    )
    normal_paths = training_images(config.dataset_root, config.category)
    extractor = create_feature_extractor(config.feature_extractor, config.image_size, config.weights)
    LOGGER.info("Extracting %s normal training embeddings", len(normal_paths))
    normal_embeddings = extractor.extract_paths(normal_paths, batch_size=config.batch_size)
    detector = NearestNeighborDetector().fit(normal_embeddings)
    calibration_scores = detector.leave_one_out_scores()
    threshold = select_normal_threshold(calibration_scores, config.threshold_percentile)
    metadata = {
        "model_version": config.model_version,
        "category": config.category,
        "feature_extractor": config.feature_extractor,
        "feature_layer": "VGG16 block5_pool followed by global average pooling",
        "embedding_normalization": "L2 normalized per image before Euclidean nearest-neighbor scoring",
        "image_size": list(config.image_size),
        "weights": config.weights,
        "threshold_policy": {
            "method": "percentile of leave-one-out normal training scores",
            "percentile": config.threshold_percentile,
            "decision_rule": "anomaly when score >= threshold",
            "calibration_sample_count": len(calibration_scores),
            "note": "No test image, test label, or ground-truth mask was used for fitting or threshold selection.",
        },
        "training_configuration": config.as_metadata(),
        "runtime": {"python": sys.version.split()[0], "platform": platform.platform()},
    }
    artifact_path = save_artifact(config.artifact_dir, detector, threshold, metadata)
    LOGGER.info("Saved artifact to %s with threshold=%.6f", artifact_path, threshold)

    evaluation_path: Path | None = None
    if evaluate:
        records = test_records(config.dataset_root, config.category)
        LOGGER.info("Evaluating %s held-out MVTec test images", len(records))
        embeddings = extractor.extract_paths([record.path for record in records], batch_size=config.batch_size)
        scores = detector.score_embeddings(embeddings)
        report = evaluate_image_level(
            [ScoredRecord(record=record, score=float(score)) for record, score in zip(records, scores, strict=True)], threshold
        )
        evaluation_path = save_evaluation(report, config.output_dir, config.category)
        LOGGER.info("Saved image-level evaluation and error analysis to %s", evaluation_path)
        LOGGER.info("AUROC=%s AP=%s", report["auroc"], report["average_precision"])
    return artifact_path, evaluation_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json", help="Path to JSON training configuration")
    parser.add_argument("--skip-evaluation", action="store_true", help="Fit and save the model without test-set metrics")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    config = load_config(args.config)
    artifact_path, evaluation_path = train(config, evaluate=not args.skip_evaluation)
    print(f"Training complete. Artifact: {artifact_path}")
    if evaluation_path:
        print(f"Evaluation report: {evaluation_path}")


if __name__ == "__main__":
    main()
