import pytest

from visionguard.ml.calibration import calibrate, classify


def test_threshold_is_frozen_and_ties_are_normal():
    calibration = calibrate(list(range(1, 43)), list(range(101, 143)), 0.05)
    assert calibration["threshold"] == 40
    assert calibration["heatmap_high"] == 140
    assert calibration["observed_fraction_above"] == pytest.approx(2 / 42)
    assert classify(40, calibration["threshold"]) == "NORMAL"
    assert classify(40.001, calibration["threshold"]) == "ANOMALY"


def test_nonfinite_calibration_fails_closed():
    with pytest.raises(ValueError, match="non-finite"):
        calibrate([1.0, float("nan")], [1.0, 2.0], 0.05)
