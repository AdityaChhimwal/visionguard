"""Fit decisions and display scale from held-out normal outputs only."""

import numpy as np


def calibrate(scores: list[float], pixel_peaks: list[float], target: float) -> dict:
    values = np.asarray(scores, dtype=np.float32)
    peaks = np.asarray(pixel_peaks, dtype=np.float32)
    if not (0 < target < 1 and len(values) == len(peaks) >= 2):
        raise ValueError("Calibration needs paired scores and map peaks from at least two images.")
    if not (np.isfinite(values).all() and np.isfinite(peaks).all()):
        raise ValueError("Calibration produced non-finite outputs.")
    threshold = float(np.quantile(values, 1 - target, method="higher"))
    high = float(np.quantile(peaks, 1 - target, method="higher"))
    if threshold <= 0 or high <= 0:
        raise ValueError("Calibration produced a nonpositive distance or display scale.")
    return {
        "threshold": threshold,
        "heatmap_high": high,
        "target": target,
        "count": len(values),
        "observed_fraction_above": float(np.mean(values > threshold)),
        "quantile_method": "higher",
    }


def classify(score: float, threshold: float) -> str:
    return "ANOMALY" if score > threshold else "NORMAL"
