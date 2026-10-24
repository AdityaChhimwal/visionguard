"""Normal-only nearest-neighbor anomaly detector."""

from __future__ import annotations

import numpy as np


class DetectorError(ValueError):
    """Raised when embeddings are not usable by the nearest-neighbor detector."""


def l2_normalize(rows: np.ndarray) -> np.ndarray:
    """Normalize a batch of row vectors while rejecting zero embeddings."""
    values = np.asarray(rows, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise DetectorError("Embeddings must be a non-empty 2-D array")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise DetectorError("Embeddings must not contain zero vectors")
    return values / norms


class NearestNeighborDetector:
    """Scores an image by distance to its nearest normal reference embedding."""

    scoring_method = "minimum_euclidean_distance_between_l2_normalized_embeddings"

    def __init__(self, reference_embeddings: np.ndarray | None = None) -> None:
        self.reference_embeddings: np.ndarray | None = None
        if reference_embeddings is not None:
            self.fit(reference_embeddings)

    def fit(self, embeddings: np.ndarray) -> "NearestNeighborDetector":
        normalized = l2_normalize(embeddings)
        if len(normalized) < 2:
            raise DetectorError("At least two normal reference embeddings are required")
        self.reference_embeddings = normalized
        return self

    def _require_fitted(self) -> np.ndarray:
        if self.reference_embeddings is None:
            raise DetectorError("Detector has not been fitted")
        return self.reference_embeddings

    def score_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Return one minimum Euclidean distance per query embedding."""
        references = self._require_fitted()
        queries = l2_normalize(embeddings)
        if queries.shape[1] != references.shape[1]:
            raise DetectorError(
                f"Embedding dimension mismatch: got {queries.shape[1]}, expected {references.shape[1]}"
            )
        # For unit vectors, ||a-b||^2 = 2 - 2(a.b); this avoids a large
        # (n_query, n_reference, dimension) temporary tensor.
        squared_distances = np.maximum(2.0 - 2.0 * (queries @ references.T), 0.0)
        return np.sqrt(np.min(squared_distances, axis=1)).astype(np.float32)

    def score_embedding(self, embedding: np.ndarray) -> float:
        values = np.asarray(embedding, dtype=np.float32)
        if values.ndim != 1:
            raise DetectorError("A single embedding must be one-dimensional")
        return float(self.score_embeddings(values[None, :])[0])

    def leave_one_out_scores(self) -> np.ndarray:
        """Score each reference against all *other* references, excluding self."""
        references = self._require_fitted()
        if len(references) < 2:
            raise DetectorError("Leave-one-out scoring requires at least two references")
        squared_distances = np.maximum(2.0 - 2.0 * (references @ references.T), 0.0)
        np.fill_diagonal(squared_distances, np.inf)
        return np.sqrt(np.min(squared_distances, axis=1)).astype(np.float32)


def select_normal_threshold(scores: np.ndarray, percentile: float) -> float:
    """Choose a normal-only percentile threshold from leakage-free scores."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise DetectorError("Threshold scores must be a non-empty finite 1-D array")
    if not 0 < percentile < 100:
        raise DetectorError("percentile must be strictly between 0 and 100")
    return float(np.percentile(values, percentile))


def is_anomaly(score: float, threshold: float) -> bool:
    """Apply the documented inclusive decision boundary."""
    if not np.isfinite(score) or not np.isfinite(threshold):
        raise DetectorError("score and threshold must be finite")
    return score >= threshold
