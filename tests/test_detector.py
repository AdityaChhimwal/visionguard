import numpy as np
import pytest

from visionguard.detector import DetectorError, NearestNeighborDetector, is_anomaly, select_normal_threshold


def test_nearest_neighbor_scores_controlled_embeddings() -> None:
    detector = NearestNeighborDetector(np.array([[3.0, 0.0], [0.0, 2.0]], dtype=np.float32))

    assert detector.score_embedding(np.array([6.0, 0.0], dtype=np.float32)) == pytest.approx(0.0)
    assert detector.score_embedding(np.array([-1.0, 0.0], dtype=np.float32)) == pytest.approx(np.sqrt(2.0))


def test_leave_one_out_excludes_self_neighbor() -> None:
    detector = NearestNeighborDetector(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))

    assert np.allclose(detector.leave_one_out_scores(), np.sqrt(2.0))


def test_threshold_is_normal_only_percentile_and_boundary_is_inclusive() -> None:
    threshold = select_normal_threshold(np.array([0.1, 0.2, 0.3]), 50.0)

    assert threshold == pytest.approx(0.2)
    assert not is_anomaly(0.199, threshold)
    assert is_anomaly(0.2, threshold)


def test_zero_embedding_is_rejected() -> None:
    with pytest.raises(DetectorError, match="zero vectors"):
        NearestNeighborDetector(np.array([[0.0, 0.0], [1.0, 0.0]]))
