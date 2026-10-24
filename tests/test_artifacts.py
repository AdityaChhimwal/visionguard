from pathlib import Path

import numpy as np
import pytest

from visionguard.artifacts import ArtifactError, load_artifact, save_artifact
from visionguard.detector import NearestNeighborDetector


def metadata() -> dict:
    return {
        "model_version": "test-v1",
        "category": "widget",
        "feature_extractor": "vgg16_block5_global_average",
        "image_size": [224, 224],
        "weights": "imagenet",
    }


def test_artifact_round_trip_preserves_detector_and_metadata(tmp_path: Path) -> None:
    detector = NearestNeighborDetector(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    save_artifact(tmp_path / "artifact", detector, 0.5, metadata())

    loaded = load_artifact(tmp_path / "artifact")

    assert loaded.threshold == 0.5
    assert loaded.metadata["model_version"] == "test-v1"
    assert loaded.detector.score_embedding(np.array([1.0, 0.0])) == pytest.approx(0.0)


def test_missing_artifact_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="incomplete"):
        load_artifact(tmp_path / "does-not-exist")
