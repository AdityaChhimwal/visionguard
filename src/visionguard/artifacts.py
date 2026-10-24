"""Persist and load the minimum state needed for repeatable inference."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np

from .detector import NearestNeighborDetector

ARTIFACT_SCHEMA_VERSION = 1
EMBEDDINGS_FILE = "reference_embeddings.npy"
METADATA_FILE = "metadata.json"


class ArtifactError(RuntimeError):
    """Raised for absent, incompatible, or corrupted model artifacts."""


@dataclass(frozen=True)
class LoadedArtifact:
    """Inference-ready detector and traceability metadata."""

    detector: NearestNeighborDetector
    threshold: float
    metadata: dict[str, Any]


def _atomic_write_bytes(target: Path, data: bytes) -> None:
    with NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
        temporary.write(data)
        temporary_path = Path(temporary.name)
    try:
        os.replace(temporary_path, target)
    finally:
        temporary_path.unlink(missing_ok=True)


def save_artifact(
    artifact_dir: str | Path,
    detector: NearestNeighborDetector,
    threshold: float,
    metadata: dict[str, Any],
) -> Path:
    """Save a compact inference artifact and return its directory.

    A VGG16 network is not serialized: its documented architecture and
    ImageNet weights are reconstructed from metadata when inference starts.
    """
    references = detector._require_fitted()
    if not np.isfinite(threshold):
        raise ArtifactError("Artifact threshold must be finite")
    directory = Path(artifact_dir)
    directory.mkdir(parents=True, exist_ok=True)
    artifact_metadata = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "threshold": float(threshold),
        "reference_embedding_count": int(references.shape[0]),
        "embedding_dimension": int(references.shape[1]),
        "scoring_method": detector.scoring_method,
        **metadata,
    }
    required = {"model_version", "category", "feature_extractor", "image_size", "weights"}
    missing = sorted(required - artifact_metadata.keys())
    if missing:
        raise ArtifactError(f"Metadata missing required fields: {', '.join(missing)}")

    with NamedTemporaryFile(dir=directory, suffix=".npy", delete=False) as temporary:
        np.save(temporary, references)
        temporary_path = Path(temporary.name)
    try:
        os.replace(temporary_path, directory / EMBEDDINGS_FILE)
    finally:
        temporary_path.unlink(missing_ok=True)
    _atomic_write_bytes(
        directory / METADATA_FILE,
        json.dumps(artifact_metadata, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )
    return directory


def load_artifact(artifact_dir: str | Path) -> LoadedArtifact:
    """Load and validate persisted detector state without loading TensorFlow."""
    directory = Path(artifact_dir)
    metadata_path = directory / METADATA_FILE
    embeddings_path = directory / EMBEDDINGS_FILE
    if not metadata_path.is_file() or not embeddings_path.is_file():
        raise ArtifactError(
            f"Model artifact is incomplete at {directory}. Expected {METADATA_FILE} and {EMBEDDINGS_FILE}."
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        references = np.load(embeddings_path, allow_pickle=False)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Unable to read model artifact at {directory}") from exc
    if metadata.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactError("Artifact schema version is unsupported")
    required = {"threshold", "model_version", "category", "feature_extractor", "image_size", "weights"}
    missing = sorted(required - metadata.keys())
    if missing:
        raise ArtifactError(f"Artifact metadata missing required fields: {', '.join(missing)}")
    try:
        threshold = float(metadata["threshold"])
        detector = NearestNeighborDetector(references)
    except (TypeError, ValueError) as exc:
        raise ArtifactError("Artifact contains invalid reference embeddings or threshold") from exc
    if not np.isfinite(threshold):
        raise ArtifactError("Artifact threshold is not finite")
    if int(metadata.get("embedding_dimension", references.shape[1])) != references.shape[1]:
        raise ArtifactError("Artifact embedding dimension does not match reference embeddings")
    return LoadedArtifact(detector=detector, threshold=threshold, metadata=metadata)
