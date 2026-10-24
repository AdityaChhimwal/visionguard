"""Load persisted VisionGuard artifacts and score new images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .artifacts import LoadedArtifact, load_artifact
from .detector import is_anomaly
from .features import create_feature_extractor
from .images import decode_image_bytes


@dataclass(frozen=True)
class Prediction:
    """Stable API-facing output of one inference request."""

    prediction: str
    anomaly_score: float
    threshold: float
    model_version: str
    category: str


class VisionGuardPredictor:
    """Owns one loaded artifact and one reusable pretrained feature extractor."""

    def __init__(self, artifact: LoadedArtifact) -> None:
        self.artifact = artifact
        metadata = artifact.metadata
        image_size = tuple(int(value) for value in metadata["image_size"])
        self.extractor = create_feature_extractor(
            identifier=str(metadata["feature_extractor"]), image_size=image_size, weights=str(metadata["weights"])
        )

    @classmethod
    def from_artifact(cls, artifact_dir: str | Path) -> "VisionGuardPredictor":
        return cls(load_artifact(artifact_dir))

    @property
    def metadata(self) -> dict:
        return self.artifact.metadata

    def predict_bytes(self, image_bytes: bytes) -> Prediction:
        image = decode_image_bytes(image_bytes)
        embedding = self.extractor.extract_images([image])[0]
        score = self.artifact.detector.score_embedding(embedding)
        anomalous = is_anomaly(score, self.artifact.threshold)
        return Prediction(
            prediction="anomaly" if anomalous else "normal",
            anomaly_score=score,
            threshold=self.artifact.threshold,
            model_version=str(self.artifact.metadata["model_version"]),
            category=str(self.artifact.metadata["category"]),
        )
