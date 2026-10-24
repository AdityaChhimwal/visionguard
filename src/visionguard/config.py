"""Configuration loading for reproducible VisionGuard runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class ConfigurationError(ValueError):
    """Raised when a training configuration is malformed."""


@dataclass(frozen=True)
class TrainingConfig:
    """Settings that affect a saved anomaly detector."""

    dataset_root: Path
    artifact_dir: Path
    output_dir: Path
    category: str
    image_size: tuple[int, int] = (224, 224)
    batch_size: int = 16
    threshold_percentile: float = 95.0
    seed: int = 42
    feature_extractor: str = "vgg16_block5_global_average"
    weights: str = "imagenet"
    model_version: str = "visionguard-v0.1.0"

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "TrainingConfig":
        required = {"dataset_root", "artifact_dir", "output_dir", "category"}
        missing = sorted(required - values.keys())
        if missing:
            raise ConfigurationError(f"Missing required configuration fields: {', '.join(missing)}")

        image_size = values.get("image_size", (224, 224))
        if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
            raise ConfigurationError("image_size must contain exactly two positive integers")
        try:
            normalized_size = (int(image_size[0]), int(image_size[1]))
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("image_size must contain exactly two positive integers") from exc
        if min(normalized_size) <= 0:
            raise ConfigurationError("image_size values must be positive")

        percentile = float(values.get("threshold_percentile", 95.0))
        if not 0 < percentile < 100:
            raise ConfigurationError("threshold_percentile must be strictly between 0 and 100")

        batch_size = int(values.get("batch_size", 16))
        if batch_size <= 0:
            raise ConfigurationError("batch_size must be positive")

        return cls(
            dataset_root=Path(values["dataset_root"]),
            artifact_dir=Path(values["artifact_dir"]),
            output_dir=Path(values["output_dir"]),
            category=str(values["category"]),
            image_size=normalized_size,
            batch_size=batch_size,
            threshold_percentile=percentile,
            seed=int(values.get("seed", 42)),
            feature_extractor=str(values.get("feature_extractor", "vgg16_block5_global_average")),
            weights=str(values.get("weights", "imagenet")),
            model_version=str(values.get("model_version", "visionguard-v0.1.0")),
        )

    def as_metadata(self) -> dict[str, Any]:
        """Return JSON-safe configuration values for a model artifact."""
        values = asdict(self)
        for key in ("dataset_root", "artifact_dir", "output_dir"):
            values[key] = str(values[key])
        values["image_size"] = list(self.image_size)
        return values


def load_config(path: str | Path) -> TrainingConfig:
    """Load a JSON training configuration from *path*."""
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"Configuration file does not exist: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"Configuration file is not valid JSON: {config_path}: {exc.msg}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError("Configuration root must be a JSON object")
    return TrainingConfig.from_mapping(raw)
